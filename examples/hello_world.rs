//! Simple hello world example demonstrating the Agents SDK
//!
//! Run with: cargo run --example hello_world

use openai_agents_rs::{runner::RunConfig, Agent, Runner};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a simple agent
    let agent = Agent::simple(
        "Assistant",
        "You are a helpful assistant that writes haikus about programming concepts.",
    );

    // Run the agent
    let result = Runner::run(
        agent,
        "Write a haiku about recursion in programming",
        RunConfig::default(),
    )
    .await?;

    // Print the result
    if result.is_success() {
        println!("Agent response:");
        println!("{}", result.final_output);
        println!("\nTokens used: {}", result.usage.total.total_tokens);
    } else {
        println!("Error: {:?}", result.error());
    }

    Ok(())
}
