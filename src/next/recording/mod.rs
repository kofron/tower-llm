//! Recording and replay I/O surfaces
//!
//! What this module provides (spec)
//! - Pluggable I/O surfaces for capturing and replaying runs
//! - Integrates with `codec` for lossless message/event fidelity
//!
//! Exports
//! - Services
//!   - `TraceWriter: Service<WriteTrace { id, items }, Response=()>`
//!   - `TraceReader: Service<ReadTrace { id }, Response=Trace>`
//!   - `ReplayService: Service<RawChatRequest, Response=StepOutcome>` (reads from a `Trace`)
//! - Layers
//!   - `RecorderLayer<S>` taps `StepOutcome`/`AgentRun` and writes via `TraceWriter`
//! - Utils
//!   - Trace format (ndjson), `TraceVersion`, integrity checks (hashes)
//!
//! Implementation strategy
//! - `RecorderLayer` calls `codec::messages_to_items`/`items_to_messages` as needed
//! - `ReplayService` reads precomputed outcomes and serves them in sequence (for step) or as a final run (for agent)
//! - Writers/readers are constructor-injected services supporting file/db backends
//!
//! Composition
//! - `ServiceBuilder::new().layer(RecorderLayer::new(writer)).service(step)`
//! - Or: `let agent = ReplayService::new(reader, trace_id);`
//!
//! Testing strategy
//! - Roundtrip with fake provider: live run → record → replay; assert same final messages/events
//! - Corruption tests: invalid trace produces explicit error

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use async_openai::types::{CreateChatCompletionRequest, CreateChatCompletionRequestArgs};
use tokio::sync::Mutex;
use tower::{BoxError, Layer, Service, ServiceExt};

use crate::next::codec::{items_to_messages, messages_to_items};
use crate::next::{StepOutcome};
use openai_agents_rs::items::RunItem;

#[derive(Debug, Clone)]
pub struct WriteTrace { pub id: String, pub items: Vec<RunItem> }
#[derive(Debug, Clone)]
pub struct ReadTrace { pub id: String }
#[derive(Debug, Clone, Default)]
pub struct Trace { pub items: Vec<RunItem> }

/// Simple in-memory trace store for tests.
#[derive(Default, Clone)]
pub struct InMemoryTraceStore(Arc<Mutex<HashMap<String, Trace>>>);

impl Service<WriteTrace> for InMemoryTraceStore {
    type Response = ();
    type Error = BoxError;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;
    fn poll_ready(&mut self, _cx: &mut std::task::Context<'_>) -> std::task::Poll<Result<(), Self::Error>> { std::task::Poll::Ready(Ok(())) }
    fn call(&mut self, req: WriteTrace) -> Self::Future {
        let store = self.0.clone();
        Box::pin(async move {
            store.lock().await.insert(req.id, Trace { items: req.items });
            Ok(())
        })
    }
}

impl Service<ReadTrace> for InMemoryTraceStore {
    type Response = Trace;
    type Error = BoxError;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;
    fn poll_ready(&mut self, _cx: &mut std::task::Context<'_>) -> std::task::Poll<Result<(), Self::Error>> { std::task::Poll::Ready(Ok(())) }
    fn call(&mut self, req: ReadTrace) -> Self::Future {
        let store = self.0.clone();
        Box::pin(async move {
            let trace = store.lock().await.get(&req.id).cloned().unwrap_or_default();
            Ok(trace)
        })
    }
}

/// Recorder layer that captures step outcomes into a trace writer.
pub struct RecorderLayer<W> { writer: W, trace_id: String }
impl<W> RecorderLayer<W> { pub fn new(writer: W, trace_id: impl Into<String>) -> Self { Self { writer, trace_id: trace_id.into() } } }

pub struct Recorder<S, W> { inner: S, writer: W, trace_id: String }

impl<S, W> Layer<S> for RecorderLayer<W>
where W: Clone {
    type Service = Recorder<S, W>;
    fn layer(&self, inner: S) -> Self::Service { Recorder { inner, writer: self.writer.clone(), trace_id: self.trace_id.clone() } }
}

impl<S, W> Service<CreateChatCompletionRequest> for Recorder<S, W>
where
    S: Service<CreateChatCompletionRequest, Response = StepOutcome, Error = BoxError> + Send + 'static,
    S::Future: Send + 'static,
    W: Service<WriteTrace, Response = (), Error = BoxError> + Clone + Send + 'static,
    W::Future: Send + 'static,
{
    type Response = StepOutcome;
    type Error = BoxError;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut std::task::Context<'_>) -> std::task::Poll<Result<(), Self::Error>> { self.inner.poll_ready(cx) }

    fn call(&mut self, req: CreateChatCompletionRequest) -> Self::Future {
        let mut writer = self.writer.clone();
        let trace_id = self.trace_id.clone();
        let fut = self.inner.call(req);
        Box::pin(async move {
            let out = fut.await?;
            let messages = match &out { StepOutcome::Next { messages, .. } | StepOutcome::Done { messages, .. } => messages.clone() };
            let items = messages_to_items(&messages).map_err(|e| format!("codec: {}", e))?;
            ServiceExt::ready(&mut writer).await?.call(WriteTrace { id: trace_id, items }).await?;
            Ok(out)
        })
    }
}

/// Service that replays a stored trace as a `StepOutcome::Done` using codec reconstruction.
pub struct ReplayService<R> { reader: R, trace_id: String, model: String }
impl<R> ReplayService<R> { pub fn new(reader: R, trace_id: impl Into<String>, model: impl Into<String>) -> Self { Self { reader, trace_id: trace_id.into(), model: model.into() } } }

impl<R> Service<CreateChatCompletionRequest> for ReplayService<R>
where
    R: Service<ReadTrace, Response = Trace, Error = BoxError> + Send + Clone + 'static,
    R::Future: Send + 'static,
{
    type Response = StepOutcome;
    type Error = BoxError;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, _cx: &mut std::task::Context<'_>) -> std::task::Poll<Result<(), Self::Error>> { std::task::Poll::Ready(Ok(())) }

    fn call(&mut self, _req: CreateChatCompletionRequest) -> Self::Future {
        let mut reader = self.reader.clone();
        let trace_id = self.trace_id.clone();
        let model = self.model.clone();
        Box::pin(async move {
            let trace = Service::call(&mut reader, ReadTrace { id: trace_id }).await?;
            let messages = items_to_messages(&trace.items);
            let _req = CreateChatCompletionRequestArgs::default().model(model).messages(messages.clone()).build()?;
            // Return Done with reconstructed messages
            Ok(StepOutcome::Done { messages, aux: Default::default() })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_openai::types::{ChatCompletionRequestUserMessageArgs};
    use tower::service_fn;

    fn req_with_user(s: &str) -> CreateChatCompletionRequest {
        let msg = ChatCompletionRequestUserMessageArgs::default().content(s).build().unwrap();
        CreateChatCompletionRequestArgs::default().model("gpt-4o").messages(vec![msg.into()]).build().unwrap()
    }

    #[tokio::test]
    async fn records_trace_on_step_done() {
        let writer = InMemoryTraceStore::default();
        let inner = service_fn(|req: CreateChatCompletionRequest| async move { Ok::<_, BoxError>(StepOutcome::Done { messages: req.messages, aux: Default::default() }) });
        let mut svc = RecorderLayer::new(writer.clone(), "t1").layer(inner);
        let _ = ServiceExt::ready(&mut svc).await.unwrap().call(req_with_user("hi")).await.unwrap();
        let trace = ServiceExt::ready(&mut writer.clone()).await.unwrap().call(ReadTrace { id: "t1".into() }).await.unwrap();
        assert!(!trace.items.is_empty());
    }

    #[tokio::test]
    async fn replay_restores_messages() {
        let store = InMemoryTraceStore::default();
        // Write a trace
        let msgs = req_with_user("hi").messages;
        let items = messages_to_items(&msgs).unwrap();
        ServiceExt::ready(&mut store.clone()).await.unwrap().call(WriteTrace { id: "t2".into(), items }).await.unwrap();
        // Replay
        let mut replay = ReplayService::new(store.clone(), "t2", "gpt-4o");
        let out = ServiceExt::ready(&mut replay).await.unwrap().call(req_with_user("ignored")).await.unwrap();
        match out { StepOutcome::Done { messages, .. } => assert!(!messages.is_empty()), _ => panic!("expected done") }
    }
}

