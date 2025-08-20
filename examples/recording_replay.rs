//! Example demonstrating recording and replay functionality.
//! Shows how to capture agent runs and replay them deterministically.

//

use tower::{Layer, Service, ServiceExt};

// Import the next module and its submodules
// Core module is now at root level
// use tower_llm directly

// Use the library-provided in-memory trace store

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("=== Recording & Replay Example ===\n");

    // Create a trace store
    let store = tower_llm::recording::InMemoryTraceStore::default();

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

            Ok::<_, tower::BoxError>(tower_llm::StepOutcome::Done {
                messages,
                aux: tower_llm::StepAux {
                    prompt_tokens: 50,
                    completion_tokens: 25,
                    tool_invocations: 0,
                },
            })
        },
    );

    // Wrap with recording layer
    let trace_id = "demo-trace-001";
    let recorder_layer = tower_llm::recording::RecorderLayer::new(store.clone(), trace_id);
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

    if let tower_llm::StepOutcome::Done { messages, .. } = &outcome {
        println!("  ✅ Recorded run with {} final messages", messages.len());

        // Convert to RunItems for storage
        let items = tower_llm::codec::messages_to_items(messages)?;
        println!(
            "  📼 Stored {} RunItems in trace '{}'",
            items.len(),
            trace_id
        );
    }

    println!("\n--- Phase 2: Replaying the Recorded Run ---");

    // Create a replay service
    let replay_service =
        tower_llm::recording::ReplayService::new(store.clone(), trace_id, "gpt-4o");
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

    if let tower_llm::StepOutcome::Done { messages, .. } = replayed {
        println!("  ✅ Replayed run returned {} messages", messages.len());

        // Verify it matches the original
        if let tower_llm::StepOutcome::Done { messages: original, .. } = outcome {
            if messages.len() == original.len() {
                println!("  ✅ Replay matches original perfectly!");
            } else {
                println!("  ❌ Replay differs from original");
            }
        }
    }

    println!("\n--- Phase 3: Multiple Recordings ---");

    // Record multiple different runs
    for i in 1..=3 {
        let trace_id = format!("trace-{:03}", i);
        let recorder = tower_llm::recording::RecorderLayer::new(store.clone(), &trace_id);
        let mut agent = recorder.layer(tower::service_fn(
            |req: async_openai::types::CreateChatCompletionRequest| async move {
                let mut msgs = req.messages.clone();
                msgs.push(
                    async_openai::types::ChatCompletionRequestAssistantMessageArgs::default()
                        .content("Response from trace")
                        .build()?
                        .into(),
                );
                Ok::<_, tower::BoxError>(tower_llm::StepOutcome::Done {
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

    // List stored traces by reading known IDs
    println!("\n📚 Trace Library:");
    for id in [
        "demo-trace-001".to_string(),
        "trace-001".to_string(),
        "trace-002".to_string(),
        "trace-003".to_string(),
    ]
    .into_iter()
    {
        let trace = tower::Service::call(
            &mut store.clone(),
            tower_llm::recording::ReadTrace { id: id.clone() },
        )
        .await
        .unwrap_or_default();
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
