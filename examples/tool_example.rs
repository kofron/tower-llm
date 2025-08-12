//! Example showing tool usage
//!
//! Run with: cargo run --example tool_example

use openai_agents_rs::{runner::RunConfig, Agent, FunctionTool, Runner};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a simple tool
    let weather_tool = Arc::new(FunctionTool::simple(
        "get_weather",
        "Get the current weather for a city",
        |city: String| {
            // In a real app, this would call a weather API
            format!("The weather in {} is sunny and 72°F", city)
        },
    ));

    // Create an agent with the tool
    let agent = Agent::simple(
        "WeatherBot",
        "You are a helpful weather assistant. Use the get_weather tool to answer questions about weather."
    )
    .with_tool(weather_tool);

    // Run the agent
    let result = Runner::run(
        agent,
        "What's the weather like in San Francisco?",
        RunConfig::default(),
    )
    .await?;

    // Print the result
    if result.is_success() {
        println!("Response: {}", result.final_output);

        // Show the execution trace
        println!("\nExecution trace:");
        for item in &result.items {
            println!("  - {:?}", item);
        }
    }

    Ok(())
}
