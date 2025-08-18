//! # Example: Contextual Tool Output Handling
//!
//! This example shows how to attach a contextual handler that can observe and
//! rewrite tool outputs, or even finalize the run early.
//!
//! To run this example, set your `OPENAI_API_KEY` and run:
//!
//! ```bash
//! export OPENAI_API_KEY="your-api-key"
//! cargo run --example contextual
//! ```
//!
//! Expected: The tool should be called twice, and the final output should contain 'HELLO WORLD'.  Because AI, it might not be exactly that, but it should at the very least contain it.

use openai_agents_rs::{runner::RunConfig, Agent, ContextStep, FunctionTool, Runner, ToolContext};
use serde_json::Value;
use std::sync::Arc;

#[derive(Clone, Default)]
struct MyContext {
    tool_calls_seen: usize,
}

struct MyHandler;

impl ToolContext<MyContext> for MyHandler {
    fn on_tool_output(
        &self,
        mut ctx: MyContext,
        tool_name: &str,
        _arguments: &Value,
        result: Result<Value, String>,
    ) -> openai_agents_rs::Result<ContextStep<MyContext>> {
        ctx.tool_calls_seen += 1;

        match result {
            Ok(v) => {
                // Wrap successful outputs in an envelope the model will see
                let rewritten = serde_json::json!({
                    "tool": tool_name,
                    "payload": v,
                    "meta": {"calls": ctx.tool_calls_seen}
                });
                Ok(ContextStep::rewrite(ctx, rewritten))
            }
            Err(msg) => {
                // Recover from errors by providing a structured fallback
                let rewritten = serde_json::json!({
                    "tool": tool_name,
                    "error": msg,
                    "meta": {"recovered": true, "calls": ctx.tool_calls_seen}
                });
                Ok(ContextStep::rewrite(ctx, rewritten))
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Running agent with contextual tool handling...");

    // A simple uppercase tool
    let uppercase = Arc::new(FunctionTool::simple(
        "uppercase",
        "Converts input to uppercase",
        |s: String| s.to_uppercase(),
    ));

    // Build the agent with the tool and the contextual handler
    let agent = Agent::simple(
        "ContextualBot",
        "You are helpful. Use the uppercase tool when asked to transform text.",
    )
    .with_tool(uppercase)
    .with_context_factory(MyContext::default, MyHandler);

    // Run with a prompt that should encourage a tool call
    let result = Runner::run(
        agent,
        "Please uppercase the phrase: 'hello world' and then summarize it.",
        RunConfig::default(),
    )
    .await?;

    if result.is_success() {
        println!("\nFinal Response:\n{}", result.final_output);

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
