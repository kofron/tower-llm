//! # Example: Tool-scope Timeout Layer
//!
//! This example demonstrates attaching a timeout policy at the tool scope.
//! The tool manages its own timeout layer, showing how tools can be self-contained
//! and manage their own cross-cutting concerns.
//!
//! To run this example, set `OPENAI_API_KEY` and run:
//!
//! ```bash
//! export OPENAI_API_KEY="your-api-key"
//! cargo run --example tool_scope_timeout
//! ```

use openai_agents_rs::{layers, runner::RunConfig, Agent, FunctionTool, Runner};
use std::sync::Arc;

fn make_slow_tool_with_timeout() -> Arc<dyn openai_agents_rs::Tool> {
    // Create the base tool
    let tool = FunctionTool::new(
        "slow".to_string(),
        "Sleeps for a while".to_string(),
        serde_json::json!({
            "type": "object",
            "properties": {},
        }),
        |_args| {
            std::thread::sleep(std::time::Duration::from_millis(250));
            Ok(serde_json::json!("done"))
        },
    );
    
    // Add a timeout layer to the tool itself
    // The tool now manages its own timeout policy
    let layered = tool.layer(layers::boxed_timeout_secs(2));
    Arc::new(layered)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Running tool-scope timeout example...\n");
    println!("Note: The tool now manages its own timeout layer!");

    let agent = Agent::simple(
        "TimeoutAgent",
        "You have a slow tool named 'slow'. If appropriate, call it to complete the task.",
    )
    .with_tool(make_slow_tool_with_timeout());

    let cfg = RunConfig::default().with_parallel_tools(true);
    let result = Runner::run(agent, "Please run the slow tool and finish", cfg).await?;

    if result.is_success() {
        println!("\nFinal Response:\n{}", result.final_output);
        println!("\nTool events:");
        for item in &result.items {
            if let openai_agents_rs::items::RunItem::ToolOutput(o) = item {
                println!(
                    "- tool_call_id={} output={} error={:?}",
                    o.tool_call_id, o.output, o.error
                );
            }
        }
    } else {
        println!("\nError: {:?}", result.error());
    }

    Ok(())
}
