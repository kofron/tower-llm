use async_openai::types::{
    ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageArgs,
    ChatCompletionRequestUserMessageArgs, ChatCompletionResponseMessage,
    CreateChatCompletionRequestArgs,
};
use std::sync::Arc;
use tower::ServiceExt;
use tower_llm::{
    policies,
    provider::{FixedProvider, ProviderResponse},
    Agent, CompositePolicy,
};

#[tokio::main]
#[allow(deprecated)]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Initialize tracing with debug level for tower_llm
    tracing_subscriber::fmt()
        .with_target(false)
        .with_level(true)
        .with_env_filter("tower_llm=debug,trace_model_demo=debug")
        .init();

    println!("=== Model Tracing Demo ===\n");
    println!("This demo shows how model parameters are traced through the system.\n");

    // Create a mock provider that returns a fixed response
    let mock_response = ProviderResponse {
        assistant: ChatCompletionResponseMessage {
            content: Some("4".to_string()),
            tool_calls: None,
            role: async_openai::types::Role::Assistant,
            function_call: None,
            refusal: None,
            audio: None, // Add missing field
        },
        prompt_tokens: 50,
        completion_tokens: 10,
    };
    let mock_provider = FixedProvider::new(mock_response);

    // Create a client (required for builder, but won't be used with mock provider)
    let client = Arc::new(async_openai::Client::new());

    // Build an agent with gpt-5 model
    println!("Building agent with model: gpt-5");

    // Create a request with explicit model override
    let messages: Vec<ChatCompletionRequestMessage> = vec![
        ChatCompletionRequestSystemMessageArgs::default()
            .content("You are a helpful assistant.")
            .build()?
            .into(),
        ChatCompletionRequestUserMessageArgs::default()
            .content("What is 2+2?")
            .build()?
            .into(),
    ];

    println!("\nScenario 1: Request with explicit model override (gpt-4o)");
    println!("==========================================================");
    let req1 = CreateChatCompletionRequestArgs::default()
        .model("gpt-4o") // Override with gpt-4o
        .messages(messages.clone())
        .build()?;

    println!("Request model field: {:?}\n", req1.model);

    let agent1 = Agent::builder(client.clone())
        .model("gpt-5") // Agent configured with gpt-5
        .temperature(0.7)
        .max_tokens(1000)
        .with_provider(mock_provider.clone())
        .policy(CompositePolicy::new(vec![policies::max_steps(1)]))
        .build();
    let run1 = agent1.oneshot(req1).await?;
    println!("✅ Completed with {} steps\n", run1.steps);

    println!("\nScenario 2: Request without model override");
    println!("===========================================");
    // Create request without explicit model (should use agent's default)
    let req2 = CreateChatCompletionRequestArgs::default()
        .model("gpt-5") // Use agent's model
        .messages(messages)
        .build()?;

    println!("Request model field: {:?}\n", req2.model);

    let agent2 = Agent::builder(client)
        .model("gpt-5") // Agent configured with gpt-5
        .temperature(0.7)
        .max_tokens(1000)
        .with_provider(mock_provider)
        .policy(CompositePolicy::new(vec![policies::max_steps(1)]))
        .build();
    let run2 = agent2.oneshot(req2).await?;
    println!("✅ Completed with {} steps\n", run2.steps);

    println!("\n📝 Check the debug logs above to see:");
    println!("   - AgentLoop starting with model");
    println!("   - Step service preparing API request");
    println!("   - OpenAIProvider sending request to API");
    println!("\nThe logs show which model is actually being used at each layer!");

    Ok(())
}
