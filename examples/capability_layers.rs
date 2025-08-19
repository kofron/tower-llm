//! Example demonstrating capability-based environment.
//!
//! This example shows how to create an environment with capabilities
//! that can be accessed by layers and tools.

use openai_agents_rs::{
    env::{EnvBuilder, InMemoryMetrics, LoggingCapability},
    layers, Agent, FunctionTool,
};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    // Create an environment with logging and metrics capabilities
    let _env = EnvBuilder::new()
        .with_capability(Arc::new(LoggingCapability))
        .with_capability(Arc::new(InMemoryMetrics::default()))
        .build();

    // Create tools with standard layers
    let calculator = FunctionTool::simple("add", "Adds two numbers", |input: String| {
        // Parse input as "a + b"
        let parts: Vec<&str> = input.split('+').collect();
        if parts.len() == 2 {
            let a: f64 = parts[0].trim().parse().unwrap_or(0.0);
            let b: f64 = parts[1].trim().parse().unwrap_or(0.0);
            format!("{}", a + b)
        } else {
            "Invalid input".to_string()
        }
    })
    .layer(layers::boxed_retry_times(2))
    .layer(layers::boxed_timeout_secs(5));

    let weather = FunctionTool::simple("weather", "Gets weather", |city: String| {
        format!("The weather in {} is sunny", city)
    })
    .layer(layers::boxed_timeout_secs(5));

    // Create an agent with the tools
    let _agent = Agent::simple("Assistant", "A helpful assistant")
        .with_tool(Arc::new(calculator))
        .with_tool(Arc::new(weather));

    println!("Created agent with capability-aware environment!");
    println!("The environment provides:");
    println!("  - Logging capability");
    println!("  - Metrics capability");
    println!();
    println!("Tools have layers applied:");
    println!("  - Calculator: retry (2 times) + timeout (5s)");
    println!("  - Weather: timeout (5s)");
    println!();
    println!("In a real application, you would run the agent with:");
    println!("  let runner = Runner::new(env);");
    println!("  let result = runner.run(agent, \"What's 5 + 3?\", RunConfig::default()).await?;");

    Ok(())
}
