//! Complete agent example using the next module with real OpenAI API.
//! Demonstrates tools, policies, and the full agent loop.

use std::sync::Arc;

use async_openai::{config::OpenAIConfig, Client};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

// Import the experimental next module
// Core module is now at root level
// use tower_llm directly

#[derive(Debug, Deserialize, JsonSchema)]
struct CalculatorArgs {
    operation: String,
    a: f64,
    b: f64,
}

#[derive(Debug, Serialize)]
struct CalculatorResult {
    result: f64,
    expression: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Initialize tracing for visibility
    tracing_subscriber::fmt::init();

    println!("=== Complete Agent Example with Next Module ===\n");

    // Create the OpenAI client
    let client: Arc<Client<OpenAIConfig>> = Arc::new(Client::new());

    // Define tools using the typed helper
    let calculator = tower_llm::tool_typed(
        "calculator",
        "Perform basic arithmetic operations (add, subtract, multiply, divide)",
        |args: CalculatorArgs| async move {
            let result = match args.operation.as_str() {
                "add" => args.a + args.b,
                "subtract" => args.a - args.b,
                "multiply" => args.a * args.b,
                "divide" => {
                    if args.b == 0.0 {
                        return Err("Division by zero".into());
                    }
                    args.a / args.b
                }
                _ => return Err(format!("Unknown operation: {}", args.operation).into()),
            };

            let expression = format!("{} {} {} = {}", args.a, args.operation, args.b, result);
            println!("  🧮 Calculator: {}", expression);

            Ok::<_, tower::BoxError>(json!(CalculatorResult { result, expression }))
        },
    );

    #[derive(Debug, Deserialize, JsonSchema)]
    struct TimeArgs {
        // OpenAI requires at least empty properties, so we add an optional field
        #[serde(default)]
        timezone: Option<String>,
    }

    let get_time = tower_llm::tool_typed(
        "get_current_time",
        "Get the current time in UTC",
        |_args: TimeArgs| async move {
            let now = chrono::Utc::now();
            let time_str = now.format("%Y-%m-%d %H:%M:%S UTC").to_string();
            println!("  🕐 Time: {}", time_str);
            Ok::<_, tower::BoxError>(json!({ "time": time_str }))
        },
    );

    // Build the agent with tools and policies
    let mut agent = tower_llm::Agent::builder(client.clone())
        .model("gpt-4o-mini") // Using mini for cost efficiency
        .temperature(0.7)
        .tool(calculator)
        .tool(get_time)
        .policy(tower_llm::CompositePolicy::new(vec![
            tower_llm::policies::until_no_tool_calls(),
            tower_llm::policies::max_steps(5),
        ]))
        .build();

    // Example 1: Simple calculation
    println!("--- Example 1: Math Problem ---");
    println!("User: What's 15 * 7 + 23?");

    let result1 = tower_llm::run(
        &mut agent,
        "You are a helpful assistant with calculator and time tools.",
        "What's 15 * 7 + 23? Please calculate step by step.",
    )
    .await?;

    println!("\nAgent completed in {} steps", result1.steps);
    println!("Stop reason: {:?}\n", result1.stop);

    // Example 2: Current time
    println!("--- Example 2: Time Query ---");
    println!("User: What time is it?");

    let result2 = tower_llm::run(
        &mut agent,
        "You are a helpful assistant with calculator and time tools.",
        "What time is it right now?",
    )
    .await?;

    println!("\nAgent completed in {} steps", result2.steps);
    println!("Stop reason: {:?}\n", result2.stop);

    // Example 3: Complex multi-step
    println!("--- Example 3: Multi-step Problem ---");
    println!("User: Calculate (100 / 4) * 3 - 10");

    let result3 = tower_llm::run(
        &mut agent,
        "You are a helpful assistant. Break down complex calculations into steps.",
        "Calculate (100 / 4) * 3 - 10. Show your work step by step.",
    )
    .await?;

    println!("\nAgent completed in {} steps", result3.steps);
    println!("Stop reason: {:?}\n", result3.stop);

    // Extract the final assistant message
    if let Some(last_msg) = result3.messages.last() {
        if let async_openai::types::ChatCompletionRequestMessage::Assistant(asst) = last_msg {
            println!("Final Answer: {:?}", asst.content);
        }
    }

    println!("\n=== Summary ===");
    println!("✅ Agent successfully used tools to solve problems");
    println!("✅ Policy correctly stopped execution when appropriate");
    println!("✅ The next module provides a clean Tower-based architecture");
    println!("\nKey features demonstrated:");
    println!("- Typed tool definitions with automatic schema generation");
    println!("- Composable policies for controlling agent loops");
    println!("- Clean separation between single steps and multi-turn loops");
    println!("- Static dependency injection throughout");

    Ok(())
}
