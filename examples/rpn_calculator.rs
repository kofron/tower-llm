//! # Example: RPN Calculator with Contextual Handler
//!
//! This example demonstrates using a contextual handler to maintain an
//! execution stack for a Reverse Polish Notation (RPN) calculator. The
//! tool simply echoes requested operations, while the handler updates a
//! per-run stack and rewrites tool outputs to include the new stack state.
//!
//! To run this example:
//!
//! ```bash
//! export OPENAI_API_KEY="your-api-key"
//! cargo run --example rpn_calculator
//! ```

use openai_agents_rs::{
    runner::RunConfig, Agent, ContextStep, ContextualAgent, FunctionTool, RunResultWithContext,
    Runner, ToolContext,
};
use serde_json::Value;
use std::sync::Arc;

#[derive(Clone, Debug, Default)]
struct StackCtx {
    stack: Vec<f64>,
}

struct RpnHandler;

impl ToolContext<StackCtx> for RpnHandler {
    fn on_tool_output(
        &self,
        mut ctx: StackCtx,
        tool_name: &str,
        _arguments: &Value,
        result: Result<Value, String>,
    ) -> openai_agents_rs::Result<ContextStep<StackCtx>> {
        let _ = tool_name; // single tool in this example

        if let Ok(Value::Object(map)) = result {
            let op = map.get("op").and_then(|v| v.as_str()).unwrap_or("");

            match op {
                "push" => {
                    if let Some(v) = map.get("value").and_then(|v| v.as_f64()) {
                        ctx.stack.push(v);
                    }
                }
                "add" => {
                    if let (Some(b), Some(a)) = (ctx.stack.pop(), ctx.stack.pop()) {
                        ctx.stack.push(a + b);
                    }
                }
                "sub" => {
                    if let (Some(b), Some(a)) = (ctx.stack.pop(), ctx.stack.pop()) {
                        ctx.stack.push(a - b);
                    }
                }
                "mul" => {
                    if let (Some(b), Some(a)) = (ctx.stack.pop(), ctx.stack.pop()) {
                        ctx.stack.push(a * b);
                    }
                }
                "div" => {
                    if let (Some(b), Some(a)) = (ctx.stack.pop(), ctx.stack.pop()) {
                        if b != 0.0 {
                            ctx.stack.push(a / b);
                        } else {
                            // leave stack unchanged on divide-by-zero
                        }
                    }
                }
                _ => {}
            }
        }

        let rewritten = serde_json::json!({
            "stack": ctx.stack,
            "top": ctx.stack.last().cloned(),
        });
        Ok(ContextStep::rewrite(ctx, rewritten))
    }
}

fn rpn_tool() -> Arc<FunctionTool> {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "op": { "type": "string", "enum": ["push", "add", "sub", "mul", "div"], "description": "RPN operation" },
            "value": { "type": ["number", "null"], "description": "Value to push when op == 'push'" }
        },
        "required": ["op"]
    });

    // The tool echoes the requested operation. The handler maintains the stack.
    let func = |args: Value| -> openai_agents_rs::Result<Value> { Ok(args) };
    Arc::new(FunctionTool::new(
        "rpn".to_string(),
        "Apply an RPN operation. For push, include 'value'.".to_string(),
        schema,
        func,
    ))
}

fn build_agent() -> ContextualAgent<StackCtx> {
    let instructions = r#"
You are an RPN calculator assistant. Use the rpn tool to manipulate a stack:
- To push a number: {"op": "push", "value": <number>}
- To add/sub/mul/div: {"op": "add"} or {"op": "sub"} etc.
After finishing the requested computation, provide a concise explanation of the result.
"#;

    Agent::simple("RPNCalc", instructions)
        .with_tool(rpn_tool())
        .with_context_factory_typed(|| StackCtx::default(), RpnHandler)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Running RPN calculator with contextual handler...\n");

    let agent = build_agent();
    let RunResultWithContext { result, context } = Runner::run_with_context(
        agent,
        // Example: (3 5 + 2 *) -> (3 + 5) * 2 = 16
        "Compute the RPN expression: 3 5 + 2 *",
        RunConfig::default(),
    )
    .await?;

    if result.is_success() {
        println!("Final Response:\n{}\n", result.final_output);
        println!("Final Stack: {:?}", context.stack);
    } else {
        println!("Error: {:?}", result.error());
    }

    Ok(())
}
