//! # Example: TypedFunctionTool
//!
//! Demonstrates using `TypedFunctionTool` to define a strongly-typed tool with an
//! explicit JSON schema for inputs and automatic serde serialization of outputs.
//!
//! ```bash
//! export OPENAI_API_KEY="your-api-key"
//! cargo run --example typed_tool
//! ```

use openai_agents_rs::{runner::RunConfig, Agent, Runner, TypedFunctionTool};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct AddArgs {
    a: i64,
    b: i64,
}

#[derive(Serialize)]
struct AddOut {
    sum: i64,
}

fn make_add_tool() -> TypedFunctionTool<
    AddArgs,
    AddOut,
    impl Fn(AddArgs) -> openai_agents_rs::Result<AddOut> + Send + Sync + 'static,
> {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        "required": ["a", "b"]
    });
    TypedFunctionTool::new("add", "Adds two integers", schema, |args: AddArgs| {
        Ok(AddOut {
            sum: args.a + args.b,
        })
    })
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let agent = Agent::simple(
        "TypedAgent",
        "You have a tool named 'add' that adds two integers. Use it when asked to add numbers.",
    )
    .with_tool(std::sync::Arc::new(make_add_tool()));

    let cfg = RunConfig::default();
    let result = Runner::run(agent, "What is 41 + 1?", cfg).await?;

    if result.is_success() {
        println!("Final: {}", result.final_output);
    } else {
        println!("Error: {:?}", result.error());
    }
    Ok(())
}
