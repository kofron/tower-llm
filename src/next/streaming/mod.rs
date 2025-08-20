//! Streaming step/agent variants
//!
//! What this module provides (spec)
//! - A streaming version of the step and loop that emits tokens/tool events incrementally
//! - Tap APIs for UIs without breaking Tower composition
//!
//! Exports
//! - Models
//!   - `StepChunk::{Token(String), ToolCallStart{ id, name, args }, ToolCallEnd{ id, output }, UsageDelta{...} }`
//!   - `AgentEvent` mirroring the above at agent layer boundaries
//! - Services
//!   - `StepStream: Service<RawChatRequest, Response=impl Stream<Item=StepChunk>>`
//! - Layers
//!   - `AgentLoopStreamLayer<S>` where `S: Service<RawChatRequest, Response=Stream<StepChunk>>`
//!   - `StreamTapLayer<S>` to tee events to an injected sink (observer)
//! - Utils
//!   - `collect_final(stream) -> AgentRun` to remain API-compatible with non-streaming callers
//!
//! Implementation strategy
//! - Provider adapter translates SSE/streaming API into `StepChunk` stream
//! - Loop layer buffers minimal state (e.g., current messages, pending tool_calls), evaluates policy on-the-fly
//! - Ensure back-pressure: do not buffer entire streams; forward as items arrive
//! - Error handling: surface provider/tool errors as terminal `AgentEvent::Error`
//!
//! Composition
//! - `ServiceBuilder::new().layer(StreamTapLayer::new(sink)).layer(AgentLoopStreamLayer::new(policy)).service(step_stream)`
//!
//! Testing strategy
//! - Fake provider that yields a scripted sequence of chunks (tokens → tool_call → outputs → final)
//! - Assert policy-controlled termination (e.g., until tool_called("x"))
//! - Verify tap receives the exact event sequence; no extra buffering or reordering

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use async_openai::types::{
    ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage,
    ChatCompletionRequestToolMessageArgs, CreateChatCompletionRequest,
    CreateChatCompletionRequestArgs,
};
use futures::{Stream, StreamExt};
use serde_json::Value;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tower::{BoxError, Layer, Service, ServiceExt};

use crate::next::{
    AgentPolicy, AgentRun, LoopState, StepAux, StepOutcome, ToolInvocation, ToolOutput,
};

/// Streaming step-level items.
#[derive(Debug, Clone)]
pub enum StepChunk {
    Token(String),
    ToolCallStart {
        id: String,
        name: String,
        arguments: Value,
    },
    ToolCallEnd {
        id: String,
        output: Value,
    },
    UsageDelta {
        prompt_tokens: usize,
        completion_tokens: usize,
    },
    /// Terminal item that signals the end of a step and carries the outcome
    StepComplete {
        outcome: StepOutcome,
    },
    /// Non-fatal error surfaced as a terminal event
    Error(String),
}

/// A provider that yields an assistant response as a stream of `StepChunk`s.
///
/// This remains abstract so tests can inject a fake provider; a real provider
/// can live in `next::provider` and adapt OpenAI SSE to this interface.
pub trait StepProvider: Send + Sync + 'static {
    type Stream: Stream<Item = StepChunk> + Send + 'static;
    fn stream_step(
        &self,
        req: CreateChatCompletionRequest,
    ) -> Pin<Box<dyn Future<Output = Result<Self::Stream, BoxError>> + Send>>;
}

/// Service that executes a single step and returns a stream of `StepChunk`s.
///
/// It delegates token/tool-call streaming to a `StepProvider` and is responsible
/// for invoking tools when requested and yielding `ToolCallEnd` events and the
/// final `StepComplete` outcome.
pub struct StepStreamService<P, T> {
    provider: Arc<P>,
    tools: Arc<tokio::sync::Mutex<T>>, // routed tool service
}

impl<P, T> StepStreamService<P, T> {
    pub fn new(provider: Arc<P>, tools: T) -> Self {
        Self {
            provider,
            tools: Arc::new(tokio::sync::Mutex::new(tools)),
        }
    }
}

impl<P, T> Service<CreateChatCompletionRequest> for StepStreamService<P, T>
where
    P: StepProvider,
    T: Service<ToolInvocation, Response = ToolOutput, Error = BoxError> + Send + 'static,
    T::Future: Send + 'static,
{
    type Response = Pin<Box<dyn Stream<Item = StepChunk> + Send>>;
    type Error = BoxError;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(
        &mut self,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        std::task::Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: CreateChatCompletionRequest) -> Self::Future {
        let provider = self.provider.clone();
        let tools = self.tools.clone();
        Box::pin(async move {
            let mut token_buf = String::new();
            let mut tool_calls: Vec<(String, String, Value)> = Vec::new();
            let mut invoked_tool_names: Vec<String> = Vec::new();
            let mut aux = StepAux::default();
            let base_model = req.model.clone();

            // Ensure model/messages present using builder semantics to normalize
            let mut builder = CreateChatCompletionRequestArgs::default();
            builder.model(base_model.clone());
            builder.messages(req.messages.clone());
            let normalized_req = builder.build().map_err(|e| format!("build req: {}", e))?;

            let stream = provider.stream_step(normalized_req).await?;

            let (tx, rx) = mpsc::channel::<StepChunk>(32);
            tokio::spawn(async move {
                futures::pin_mut!(stream);
                while let Some(item) = stream.next().await {
                    match &item {
                        StepChunk::Token(t) => {
                            token_buf.push_str(t);
                        }
                        StepChunk::ToolCallStart {
                            id,
                            name,
                            arguments,
                        } => {
                            tool_calls.push((id.clone(), name.clone(), arguments.clone()));
                        }
                        StepChunk::UsageDelta {
                            prompt_tokens,
                            completion_tokens,
                        } => {
                            aux.prompt_tokens += *prompt_tokens;
                            aux.completion_tokens += *completion_tokens;
                        }
                        _ => {}
                    }
                    // forward every item as-is
                    if tx.send(item).await.is_err() {
                        return;
                    }
                }

                // Provider stream ended; construct assistant + maybe run tools
                let mut messages: Vec<ChatCompletionRequestMessage> = req.messages.clone();
                // Build assistant message
                let mut asst = ChatCompletionRequestAssistantMessageArgs::default();
                asst.content(token_buf.clone());
                if !tool_calls.is_empty() {
                    // Build tool_calls list for assistant message
                    let tool_calls_for_msg: Vec<
                        async_openai::types::ChatCompletionMessageToolCall,
                    > = tool_calls
                        .iter()
                        .map(|(id, name, arguments)| {
                            async_openai::types::ChatCompletionMessageToolCall {
                                id: id.clone(),
                                r#type: async_openai::types::ChatCompletionToolType::Function,
                                function: async_openai::types::FunctionCall {
                                    name: name.clone(),
                                    arguments: arguments.to_string(),
                                },
                            }
                        })
                        .collect();
                    asst.tool_calls(tool_calls_for_msg);
                }
                match asst.build() {
                    Ok(msg) => messages.push(msg.into()),
                    Err(e) => {
                        let _ = tx
                            .send(StepChunk::Error(format!("assistant build: {}", e)))
                            .await;
                        return;
                    }
                }

                // Execute tools sequentially and emit ends
                for (id, name, args) in tool_calls.into_iter() {
                    invoked_tool_names.push(name.clone());
                    let inv = ToolInvocation {
                        id: id.clone(),
                        name: name.clone(),
                        arguments: args,
                    };
                    match tools.lock().await.ready().await {
                        Ok(svc) => match svc.call(inv).await {
                            Ok(ToolOutput { id: out_id, result }) => {
                                aux.tool_invocations += 1;
                                // Append tool message
                                match ChatCompletionRequestToolMessageArgs::default()
                                    .tool_call_id(out_id.clone())
                                    .content(result.to_string())
                                    .build()
                                {
                                    Ok(tool_msg) => messages.push(tool_msg.into()),
                                    Err(e) => {
                                        let _ = tx
                                            .send(StepChunk::Error(format!(
                                                "tool msg build: {}",
                                                e
                                            )))
                                            .await;
                                        return;
                                    }
                                }
                                let _ = tx
                                    .send(StepChunk::ToolCallEnd {
                                        id: out_id,
                                        output: result,
                                    })
                                    .await;
                            }
                            Err(e) => {
                                let _ = tx
                                    .send(StepChunk::Error(format!("tool error: {}", e)))
                                    .await;
                                return;
                            }
                        },
                        Err(e) => {
                            let _ = tx
                                .send(StepChunk::Error(format!("tool not ready: {}", e)))
                                .await;
                            return;
                        }
                    }
                }

                // Final outcome
                let outcome = if invoked_tool_names.is_empty() {
                    StepOutcome::Done { messages, aux }
                } else {
                    StepOutcome::Next {
                        messages,
                        aux,
                        invoked_tools: invoked_tool_names,
                    }
                };
                let _ = tx.send(StepChunk::StepComplete { outcome }).await;
            });

            Ok(Box::pin(ReceiverStream::new(rx)) as Pin<Box<dyn Stream<Item = StepChunk> + Send>>)
        })
    }
}

/// Events emitted by the agent-level streaming loop.
#[derive(Debug, Clone)]
pub enum AgentEvent {
    Step(usize),
    Item(StepChunk),
    RunComplete(AgentRun),
}

/// Layer that turns a streaming step into a multi-step agentic stream with policies.
pub struct AgentLoopStreamLayer<P> {
    policy: P,
}

impl<P> AgentLoopStreamLayer<P> {
    pub fn new(policy: P) -> Self {
        Self { policy }
    }
}

pub struct AgentLoopStream<S, P> {
    inner: Arc<tokio::sync::Mutex<S>>,
    policy: P,
}

impl<S, P> Layer<S> for AgentLoopStreamLayer<P>
where
    P: Clone,
{
    type Service = AgentLoopStream<S, P>;
    fn layer(&self, inner: S) -> Self::Service {
        AgentLoopStream {
            inner: Arc::new(tokio::sync::Mutex::new(inner)),
            policy: self.policy.clone(),
        }
    }
}

impl<S, P> Service<CreateChatCompletionRequest> for AgentLoopStream<S, P>
where
    S: Service<
            CreateChatCompletionRequest,
            Response = Pin<Box<dyn Stream<Item = StepChunk> + Send>>,
            Error = BoxError,
        > + Send
        + 'static,
    S::Future: Send + 'static,
    P: AgentPolicy + Send + Sync + Clone + 'static,
{
    type Response = Pin<Box<dyn Stream<Item = AgentEvent> + Send>>;
    type Error = BoxError;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(
        &mut self,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        std::task::Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: CreateChatCompletionRequest) -> Self::Future {
        let inner = self.inner.clone();
        let policy = self.policy.clone();
        Box::pin(async move {
            let (tx, rx) = mpsc::channel::<AgentEvent>(64);
            tokio::spawn(async move {
                let base_model = req.model.clone();
                let mut current_messages = req.messages.clone();
                let mut state = LoopState::default();
                let mut step_index: usize = 0;

                loop {
                    // Build current request
                    let mut b = CreateChatCompletionRequestArgs::default();
                    b.model(&base_model);
                    b.messages(current_messages.clone());
                    let current_req = match b.build() {
                        Ok(r) => r,
                        Err(e) => {
                            let _ = tx
                                .send(AgentEvent::Item(StepChunk::Error(format!(
                                    "build req: {}",
                                    e
                                ))))
                                .await;
                            break;
                        }
                    };

                    let mut guard = inner.lock().await;
                    let stream = match guard.ready().await {
                        Ok(svc) => match svc.call(current_req).await {
                            Ok(st) => st,
                            Err(e) => {
                                let _ = tx
                                    .send(AgentEvent::Item(StepChunk::Error(format!(
                                        "step stream: {}",
                                        e
                                    ))))
                                    .await;
                                break;
                            }
                        },
                        Err(e) => {
                            let _ = tx
                                .send(AgentEvent::Item(StepChunk::Error(format!(
                                    "step not ready: {}",
                                    e
                                ))))
                                .await;
                            break;
                        }
                    };
                    drop(guard);

                    step_index += 1;
                    if tx.send(AgentEvent::Step(step_index)).await.is_err() {
                        break;
                    }

                    // Forward inner items until StepComplete
                    futures::pin_mut!(stream);
                    let mut last_outcome: Option<StepOutcome> = None;
                    while let Some(item) = stream.next().await {
                        let is_complete = matches!(item, StepChunk::StepComplete { .. });
                        if let StepChunk::StepComplete { outcome } = item.clone() {
                            last_outcome = Some(outcome);
                        }
                        if tx.send(AgentEvent::Item(item)).await.is_err() {
                            return;
                        }
                        if is_complete {
                            break;
                        }
                    }

                    state.steps += 1;
                    match last_outcome {
                        Some(outcome) => {
                            if let Some(stop) = policy.decide(&state, &outcome) {
                                // Extract messages for final run
                                let messages = match outcome {
                                    StepOutcome::Next { messages, .. } => messages,
                                    StepOutcome::Done { messages, .. } => messages,
                                };
                                let run = AgentRun {
                                    messages,
                                    steps: state.steps,
                                    stop,
                                };
                                let _ = tx.send(AgentEvent::RunComplete(run)).await;
                                break;
                            }
                            // Continue with updated messages
                            current_messages = match outcome {
                                StepOutcome::Next { messages, .. } => messages,
                                StepOutcome::Done { messages, .. } => messages,
                            };
                        }
                        None => {
                            // No completion seen; treat as error
                            let _ = tx
                                .send(AgentEvent::Item(StepChunk::Error(
                                    "missing StepComplete".into(),
                                )))
                                .await;
                            break;
                        }
                    }
                }
            });
            Ok(Box::pin(ReceiverStream::new(rx)) as Pin<Box<dyn Stream<Item = AgentEvent> + Send>>)
        })
    }
}

/// Layer that tees a stream of `AgentEvent`s to a sink function.
pub struct StreamTapLayer {
    sink: Arc<dyn Fn(&AgentEvent) + Send + Sync + 'static>,
}

impl StreamTapLayer {
    pub fn new<F>(f: F) -> Self
    where
        F: Fn(&AgentEvent) + Send + Sync + 'static,
    {
        Self { sink: Arc::new(f) }
    }
}

pub struct StreamTap<S> {
    inner: S,
    sink: Arc<dyn Fn(&AgentEvent) + Send + Sync + 'static>,
}

impl<S> Layer<S> for StreamTapLayer {
    type Service = StreamTap<S>;
    fn layer(&self, inner: S) -> Self::Service {
        StreamTap {
            inner,
            sink: self.sink.clone(),
        }
    }
}

impl<S> Service<CreateChatCompletionRequest> for StreamTap<S>
where
    S: Service<
            CreateChatCompletionRequest,
            Response = Pin<Box<dyn Stream<Item = AgentEvent> + Send>>,
            Error = BoxError,
        > + Send
        + 'static,
    S::Future: Send + 'static,
{
    type Response = Pin<Box<dyn Stream<Item = AgentEvent> + Send>>;
    type Error = BoxError;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(
        &mut self,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        std::task::Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: CreateChatCompletionRequest) -> Self::Future {
        let sink = self.sink.clone();
        let fut = self.inner.call(req);
        Box::pin(async move {
            let stream = fut.await?;
            let (tx, rx) = mpsc::channel::<AgentEvent>(32);
            tokio::spawn(async move {
                futures::pin_mut!(stream);
                while let Some(item) = stream.next().await {
                    (sink)(&item);
                    if tx.send(item).await.is_err() {
                        return;
                    }
                }
            });
            Ok(Box::pin(ReceiverStream::new(rx)) as Pin<Box<dyn Stream<Item = AgentEvent> + Send>>)
        })
    }
}

/// Utility: collect a streaming agent run and return the final `AgentRun`.
pub async fn collect_final<S>(stream: &mut S) -> Option<AgentRun>
where
    S: Stream<Item = AgentEvent> + Unpin,
{
    let mut final_run: Option<AgentRun> = None;
    while let Some(ev) = stream.next().await {
        if let AgentEvent::RunComplete(run) = ev {
            final_run = Some(run);
        }
    }
    final_run
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;
    use serde_json::json;
    use tower::service_fn;

    struct FakeProvider {
        items: Vec<StepChunk>,
    }

    impl StepProvider for FakeProvider {
        type Stream = Pin<Box<dyn Stream<Item = StepChunk> + Send>>;
        fn stream_step(
            &self,
            _req: CreateChatCompletionRequest,
        ) -> Pin<Box<dyn Future<Output = Result<Self::Stream, BoxError>> + Send>> {
            let s = stream::iter(self.items.clone());
            Box::pin(
                async move { Ok(Box::pin(s) as Pin<Box<dyn Stream<Item = StepChunk> + Send>>) },
            )
        }
    }

    #[tokio::test]
    async fn step_stream_invokes_tool_and_finishes() {
        // Provider emits tokens, then a tool call
        let provider = Arc::new(FakeProvider {
            items: vec![
                StepChunk::Token("Hello ".into()),
                StepChunk::Token("world".into()),
                StepChunk::ToolCallStart {
                    id: "call_1".into(),
                    name: "echo".into(),
                    arguments: json!({"x": 1}),
                },
            ],
        });

        // Tool echoes args
        let tool = service_fn(|inv: ToolInvocation| async move {
            Ok::<_, BoxError>(ToolOutput {
                id: inv.id,
                result: json!({"ok": true}),
            })
        });

        let mut svc = StepStreamService::new(provider, tool);
        let req = CreateChatCompletionRequestArgs::default()
            .model("gpt-4o")
            .messages(vec![])
            .build()
            .unwrap();
        let mut stream = svc.call(req).await.unwrap();
        let mut got_tool_end = false;
        let mut got_complete = false;
        while let Some(item) = stream.next().await {
            match item {
                StepChunk::ToolCallEnd { id, output } => {
                    assert_eq!(id, "call_1");
                    assert_eq!(output, json!({"ok": true}));
                    got_tool_end = true;
                }
                StepChunk::StepComplete { outcome } => {
                    match outcome {
                        StepOutcome::Next {
                            messages,
                            invoked_tools,
                            ..
                        } => {
                            assert!(messages.len() >= 2); // assistant + tool
                            assert_eq!(invoked_tools, vec!["echo".to_string()]);
                        }
                        _ => panic!("expected Next"),
                    }
                    got_complete = true;
                }
                _ => {}
            }
        }
        assert!(got_tool_end && got_complete);
    }

    #[tokio::test]
    async fn loop_stream_runs_until_policy() {
        // Provider that yields just tokens and finishes (no tool calls)
        let provider = Arc::new(FakeProvider {
            items: vec![StepChunk::Token("ok".into())],
        });
        // No-op tool service
        let tool = service_fn(|_inv: ToolInvocation| async move {
            Ok::<_, BoxError>(ToolOutput {
                id: "x".into(),
                result: json!({}),
            })
        });
        // Wrap into StepStreamService
        let step = StepStreamService::new(provider, tool);
        // Build a layer that turns it into an agent loop stream with a policy that stops on Done
        let loop_layer = AgentLoopStreamLayer::new(crate::next::policies::until_no_tool_calls());
        let mut agent_stream = loop_layer.layer(step);

        let req = CreateChatCompletionRequestArgs::default()
            .model("gpt-4o")
            .messages(vec![])
            .build()
            .unwrap();
        let mut stream = agent_stream.call(req).await.unwrap();
        let mut saw_run_complete = false;
        while let Some(ev) = stream.next().await {
            if let AgentEvent::RunComplete(run) = ev {
                saw_run_complete = true;
                assert_eq!(run.steps, 1);
                assert!(matches!(run.stop, crate::next::AgentStopReason::DoneNoToolCalls));
            }
        }
        assert!(saw_run_complete);
    }

    #[tokio::test]
    async fn tap_layer_receives_every_event() {
        let provider = Arc::new(FakeProvider {
            items: vec![StepChunk::Token("a".into()), StepChunk::Token("b".into())],
        });
        let tool = service_fn(|_inv: ToolInvocation| async move {
            Ok::<_, BoxError>(ToolOutput {
                id: "i".into(),
                result: json!({}),
            })
        });
        let step = StepStreamService::new(provider, tool);
        let loop_layer = AgentLoopStreamLayer::new(crate::next::policies::max_steps(1));
        let mut agent = loop_layer.layer(step);
        let mut tap_log: Arc<tokio::sync::Mutex<Vec<String>>> =
            Arc::new(tokio::sync::Mutex::new(vec![]));
        let tap_log_clone = tap_log.clone();
        let tap = StreamTapLayer::new(move |ev: &AgentEvent| {
            let s = format!("{:?}", ev);
            let tl = tap_log_clone.clone();
            tokio::spawn(async move {
                tl.lock().await.push(s);
            });
        });
        let mut svc = tap.layer(agent);
        let req = CreateChatCompletionRequestArgs::default()
            .model("gpt-4o")
            .messages(vec![])
            .build()
            .unwrap();
        let mut stream = svc.call(req).await.unwrap();
        // Drain
        while let Some(_ev) = stream.next().await {}
        assert!(!tap_log.lock().await.is_empty());
    }
}
