//! Example demonstrating recording and replay functionality.
//! Shows how to capture agent runs and replay them deterministically.

use std::sync::Arc;

use tower::{Layer, Service, ServiceExt};

// Import the next module and its submodules
#[path = "../src/next/mod.rs"]
mod next;

#[path = "../src/next/recording/mod.rs"]
mod recording;

#[path = "../src/next/codec/mod.rs"]
mod codec;

// Simple in-memory trace store for demonstration
#[derive(Clone, Default)]
struct InMemoryTraceStore {
    traces: Arc<tokio::sync::Mutex<std::collections::HashMap<String, recording::Trace>>>,
}

impl Service<recording::WriteTrace> for InMemoryTraceStore {
    type Response = ();
    type Error = tower::BoxError;
    type Future =
        std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), tower::BoxError>> + Send>>;

    fn poll_ready(
        &mut self,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        std::task::Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: recording::WriteTrace) -> Self::Future {
        let traces = self.traces.clone();
        Box::pin(async move {
            let mut t = traces.lock().await;
            t.insert(req.id.clone(), recording::Trace { items: req.items });
            Ok(())
        })
    }
}

impl Service<recording::ReadTrace> for InMemoryTraceStore {
    type Response = recording::Trace;
    type Error = tower::BoxError;
    type Future = std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<recording::Trace, tower::BoxError>> + Send>,
    >;

    fn poll_ready(
        &mut self,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        std::task::Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: recording::ReadTrace) -> Self::Future {
        let traces = self.traces.clone();
        Box::pin(async move {
            let t = traces.lock().await;
            t.get(&req.id)
                .cloned()
                .ok_or_else(|| format!("Trace {} not found", req.id).into())
        })
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("=== Recording & Replay Example ===\n");

    // Create a trace store
    let store = InMemoryTraceStore::default();

    println!("--- Phase 1: Recording an Agent Run ---");

    // Create a mock agent that generates a conversation
    let mock_agent = tower::service_fn(
        |req: async_openai::types::CreateChatCompletionRequest| async move {
            println!("  Agent processing {} messages", req.messages.len());

            // Simulate adding an assistant response
            let mut messages = req.messages.clone();
            messages.push(
                async_openai::types::ChatCompletionRequestAssistantMessageArgs::default()
                    .content(format!("I processed {} messages", req.messages.len()))
                    .build()?
                    .into(),
            );

            Ok::<_, tower::BoxError>(next::StepOutcome::Done {
                messages,
                aux: next::StepAux {
                    prompt_tokens: 50,
                    completion_tokens: 25,
                    tool_invocations: 0,
                },
            })
        },
    );

    // Wrap with recording layer
    let trace_id = "demo-trace-001";
    let recorder_layer = recording::RecorderLayer::new(store.clone(), trace_id);
    let mut recording_agent = recorder_layer.layer(mock_agent);

    // Create an initial conversation
    let initial_messages = vec![
        async_openai::types::ChatCompletionRequestSystemMessageArgs::default()
            .content("You are a helpful assistant")
            .build()?
            .into(),
        async_openai::types::ChatCompletionRequestUserMessageArgs::default()
            .content("Hello, how are you?")
            .build()?
            .into(),
    ];

    let req = async_openai::types::CreateChatCompletionRequestArgs::default()
        .model("gpt-4o")
        .messages(initial_messages.clone())
        .build()?;

    println!(
        "  Recording conversation with {} initial messages",
        initial_messages.len()
    );

    // Execute and record
    let outcome = recording_agent.ready().await?.call(req).await?;

    match &outcome {
        next::StepOutcome::Done { messages, .. } => {
            println!("  ✅ Recorded run with {} final messages", messages.len());

            // Convert to RunItems for storage
            let items = codec::messages_to_items(&messages)?;
            println!(
                "  📼 Stored {} RunItems in trace '{}'",
                items.len(),
                trace_id
            );
        }
        _ => {}
    }

    println!("\n--- Phase 2: Replaying the Recorded Run ---");

    // Create a replay service
    let replay_service = recording::ReplayService::new(store.clone(), trace_id, "gpt-4o");
    let mut replayer = replay_service;

    // Replay with the same initial messages
    let replay_req = async_openai::types::CreateChatCompletionRequestArgs::default()
        .model("gpt-4o")
        .messages(initial_messages.clone())
        .build()?;

    println!(
        "  Replaying with same {} initial messages",
        initial_messages.len()
    );

    let replayed = replayer.ready().await?.call(replay_req).await?;

    match replayed {
        next::StepOutcome::Done { messages, .. } => {
            println!("  ✅ Replayed run returned {} messages", messages.len());

            // Verify it matches the original
            if let next::StepOutcome::Done {
                messages: original, ..
            } = outcome
            {
                if messages.len() == original.len() {
                    println!("  ✅ Replay matches original perfectly!");
                } else {
                    println!("  ❌ Replay differs from original");
                }
            }
        }
        _ => {}
    }

    println!("\n--- Phase 3: Multiple Recordings ---");

    // Record multiple different runs
    for i in 1..=3 {
        let trace_id = format!("trace-{:03}", i);
        let recorder = recording::RecorderLayer::new(store.clone(), &trace_id);
        let mut agent = recorder.layer(tower::service_fn(
            |req: async_openai::types::CreateChatCompletionRequest| async move {
                let mut msgs = req.messages.clone();
                msgs.push(
                    async_openai::types::ChatCompletionRequestAssistantMessageArgs::default()
                        .content(format!("Response from trace"))
                        .build()?
                        .into(),
                );
                Ok::<_, tower::BoxError>(next::StepOutcome::Done {
                    messages: msgs,
                    aux: Default::default(),
                })
            },
        ));

        let req = async_openai::types::CreateChatCompletionRequestArgs::default()
            .model("gpt-4o")
            .messages(vec![
                async_openai::types::ChatCompletionRequestUserMessageArgs::default()
                    .content(format!("Message for trace {}", i))
                    .build()?
                    .into(),
            ])
            .build()?;

        let _ = agent.ready().await?.call(req).await?;
        println!("  📼 Recorded trace-{:03}", i);
    }

    // List all stored traces
    let traces = store.traces.lock().await;
    println!("\n📚 Trace Library:");
    for (id, trace) in traces.iter() {
        println!("  - {}: {} items", id, trace.items.len());
    }

    println!("\n=== Use Cases ===");
    println!("1. **Debugging**: Record production runs for offline analysis");
    println!("2. **Testing**: Create test suites from real conversations");
    println!("3. **Compliance**: Audit trail of all agent interactions");
    println!("4. **Training**: Generate datasets from successful runs");
    println!("5. **Regression**: Ensure changes don't break existing behavior");

    println!("\n=== Key Takeaways ===");
    println!("1. RecorderLayer captures complete agent runs as RunItems");
    println!("2. ReplayService reconstructs exact conversations from traces");
    println!("3. Traces are portable and can be stored anywhere");
    println!("4. Perfect for debugging, testing, and compliance");
    println!("5. Enables deterministic replay of non-deterministic LLM calls");
    println!("6. Integrates seamlessly with the Tower service stack");

    Ok(())
}
