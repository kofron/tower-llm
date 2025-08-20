//! Example demonstrating the sessions module for conversation memory management.
//! This shows how to persist agent conversations across multiple runs.

use std::sync::Arc;

use async_openai::types::{ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequestArgs};
use tower::Layer;

// Import the next module and its submodules
// Core module is now at root level
// use tower_llm directly

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("=== Sessions Memory Example ===\n");

    // Create an in-memory session store
    let store = tower_llm::sessions::InMemorySessionStore::default();
    let session_id = tower_llm::sessions::SessionId("user_123_session".to_string());

    println!("📦 Created in-memory session store");
    println!("🔑 Session ID: {}\n", session_id.0);

    // Create a mock agent service that echoes messages
    let mock_agent = tower::service_fn(|req: async_openai::types::CreateChatCompletionRequest| {
        async move {
            println!("  Agent received {} messages", req.messages.len());
            for (i, msg) in req.messages.iter().enumerate() {
                let role = match msg {
                    async_openai::types::ChatCompletionRequestMessage::System(_) => "System",
                    async_openai::types::ChatCompletionRequestMessage::User(_) => "User",
                    async_openai::types::ChatCompletionRequestMessage::Assistant(_) => "Assistant",
                    _ => "Other",
                };
                println!("    [{}] {}", i, role);
            }

            // Return a simple response with the updated messages
            let mut response_messages = req.messages.clone();
            response_messages.push(
                async_openai::types::ChatCompletionRequestAssistantMessageArgs::default()
                    .content("I've processed your message.")
                    .build()?
                    .into(),
            );

            Ok::<_, tower::BoxError>(tower_llm::StepOutcome::Done {
                messages: response_messages,
                aux: Default::default(),
            })
        }
    });

    // Wrap the agent with the memory layer
    let memory_layer = tower_llm::sessions::MemoryLayer::new(
        Arc::new(store.clone()),
        Arc::new(store.clone()),
        session_id.clone(),
    );
    let mut agent_with_memory = memory_layer.layer(mock_agent);

    println!("--- First Interaction ---");
    println!("User: 'Hello, my name is Alice'\n");

    // First interaction
    let req1 = CreateChatCompletionRequestArgs::default()
        .model("gpt-4o")
        .messages(vec![ChatCompletionRequestUserMessageArgs::default()
            .content("Hello, my name is Alice")
            .build()?
            .into()])
        .build()?;

    let outcome1 = agent_with_memory.ready().await?.call(req1).await?;

    match outcome1 {
        tower_llm::StepOutcome::Done { messages, .. } => {
            println!(
                "\n✅ First interaction complete. Total messages: {}",
                messages.len()
            );
        }
        _ => println!("Unexpected outcome"),
    }

    println!("\n--- Second Interaction (Same Session) ---");
    println!("User: 'What's my name?'\n");

    // Second interaction - should remember the context
    let req2 = CreateChatCompletionRequestArgs::default()
        .model("gpt-4o")
        .messages(vec![ChatCompletionRequestUserMessageArgs::default()
            .content("What's my name?")
            .build()?
            .into()])
        .build()?;

    let outcome2 = agent_with_memory.ready().await?.call(req2).await?;

    match outcome2 {
        tower_llm::StepOutcome::Done { messages, .. } => {
            println!(
                "\n✅ Second interaction complete. Total messages: {}",
                messages.len()
            );
            println!("   The agent has access to the full conversation history!");
        }
        _ => println!("Unexpected outcome"),
    }

    // Demonstrate session persistence
    println!("\n--- Checking Session Persistence ---");

    // Load the session directly from the store
    use tower::{Service, ServiceExt};
    let mut store_svc = store.clone();
    let stored_history = Service::<tower_llm::sessions::LoadSession>::call(
        &mut store_svc,
        tower_llm::sessions::LoadSession {
            id: session_id.clone(),
        },
    )
    .await?;
    println!("📚 Session contains {} messages:", stored_history.len());

    for (i, msg) in stored_history.iter().enumerate() {
        let role = match msg {
            async_openai::types::ChatCompletionRequestMessage::User(_) => "User",
            async_openai::types::ChatCompletionRequestMessage::Assistant(_) => "Assistant",
            _ => "Other",
        };
        println!("   [{}] {}", i, role);
    }

    println!("\n--- Creating New Session ---");

    // Create a new session for comparison
    let new_session_id = tower_llm::sessions::SessionId("user_456_session".to_string());
    let memory_layer2 = tower_llm::sessions::MemoryLayer::new(
        Arc::new(store.clone()),
        Arc::new(store.clone()),
        new_session_id.clone(),
    );
    let mut new_agent = memory_layer2.layer(tower::service_fn(
        |req: async_openai::types::CreateChatCompletionRequest| async move {
            println!(
                "  New session agent received {} messages",
                req.messages.len()
            );
            Ok::<_, tower::BoxError>(tower_llm::StepOutcome::Done {
                messages: req.messages,
                aux: Default::default(),
            })
        },
    ));

    let req3 = CreateChatCompletionRequestArgs::default()
        .model("gpt-4o")
        .messages(vec![ChatCompletionRequestUserMessageArgs::default()
            .content("Do you know who I am?")
            .build()?
            .into()])
        .build()?;

    let _ = new_agent.ready().await?.call(req3).await?;

    println!("   New session has no prior context - starts fresh!\n");

    println!("=== Key Takeaways ===");
    println!("1. MemoryLayer automatically loads conversation history before each call");
    println!("2. It saves the updated history after each interaction");
    println!("3. Different sessions maintain separate conversation contexts");
    println!("4. The InMemorySessionStore can be replaced with persistent storage");
    println!("   (e.g., SqliteSessionStore, RedisSessionStore, etc.)");
    println!("5. This enables stateful, multi-turn conversations that survive restarts");

    Ok(())
}
