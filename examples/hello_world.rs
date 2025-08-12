//! # Simple "Hello, World" Example
//!
//! This example demonstrates the most basic usage of the OpenAI Agents SDK.
//! It creates a simple agent, runs it with a prompt, and prints the result.
//!
//! To run this example, you first need to set your `OPENAI_API_KEY` environment
//! variable.
//!
//! ```bash
//! export OPENAI_API_KEY="your-api-key"
//! cargo run --example hello_world
//! ```

use openai_agents_rs::{runner::RunConfig, Agent, Runner};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Running simple haiku agent...");

    // 1. Create a simple agent
    //
    // An agent is the core of the SDK. It is defined by a name and a set of
    // instructions that guide its behavior. The `Agent::simple` constructor
    // provides an easy way to create a basic agent.
    let agent = Agent::simple(
        "HaikuBot",
        "You are a helpful assistant that writes haikus about programming concepts.",
    );

    // 2. Run the agent
    //
    // The `Runner` is responsible for executing the agent. The `run` method
    // takes the agent, a user prompt, and a `RunConfig`. The `RunConfig`
    // can be used to customize the execution, but `default()` is sufficient
    // for this simple case.
    let result = Runner::run(
        agent,
        "Write a haiku about recursion in programming",
        RunConfig::default(),
    )
    .await?;

    // 3. Process the result
    //
    // The `run` method returns a `RunResult`, which contains the final output
    // from the agent, as well as other useful information like usage statistics.
    if result.is_success() {
        println!("\nAgent response:");
        println!("{}", result.final_output);
        println!("\nTokens used: {}", result.usage.total.total_tokens);
    } else {
        println!("\nError: {:?}", result.error());
    }

    Ok(())
}
