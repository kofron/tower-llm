//! # Example: Using Tools with an Agent
//!
//! This example demonstrates how to create a simple tool and provide it to an
//! agent. The agent will then use this tool to answer a user's question.
//!
//! To run this example, you first need to set your `OPENAI_API_KEY` environment
//! variable.
//!
//! ```bash
//! export OPENAI_API_KEY="your-api-key"
//! cargo run --example tool_example
//! ```

use openai_agents_rs::{runner::RunConfig, Agent, FunctionTool, Runner};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Running weather agent with a tool...");

    // 1. Create a tool from a simple function
    //
    // Tools are the primary way for agents to interact with the outside world.
    // The `FunctionTool::simple` constructor is a convenient way to create a
    // tool from a function that takes a `String` and returns a `String`.
    // The three arguments are the tool's name, a description of what it does,
    // and the function to execute.
    let weather_tool = Arc::new(FunctionTool::simple(
        "get_weather",
        "Get the current weather for a city",
        |city: String| {
            // In a real application, this function would call an external API.
            // For this example, we'll just return a hardcoded response.
            format!("The weather in {} is sunny and 72°F.", city)
        },
    ));

    // 2. Create an agent and provide it with the tool
    //
    // We create an agent as usual, but we also use the `with_tool` builder
    // method to give the agent access to our weather tool. The agent's
    // instructions should encourage it to use the tool when appropriate.
    let agent = Agent::simple(
        "WeatherBot",
        "You are a helpful weather assistant. Use the get_weather tool to answer questions about the weather."
    )
    .with_tool(weather_tool);

    // 3. Run the agent
    //
    // We run the agent with a prompt that should trigger the use of the tool.
    let result = Runner::run(
        agent,
        "What's the weather like in San Francisco?",
        RunConfig::default(),
    )
    .await?;

    // 4. Print the result and the execution trace
    //
    // The final output will be a natural language response that incorporates
    // the information from the tool. The `RunResult` also contains a list of
    // `items`, which provides a detailed trace of the execution, including
    // the tool call and its output.
    if result.is_success() {
        println!("\nFinal Response:\n{}", result.final_output);

        // Show the detailed execution trace
        println!("\nExecution Trace:");
        for item in &result.items {
            match item {
                openai_agents_rs::items::RunItem::ToolCall(tc) => {
                    println!("  - Tool Call: {}({})", tc.tool_name, tc.arguments);
                }
                openai_agents_rs::items::RunItem::ToolOutput(to) => {
                    println!("  - Tool Output: {}", to.output);
                }
                _ => {}
            }
        }
    } else {
        println!("\nError: {:?}", result.error());
    }

    Ok(())
}
