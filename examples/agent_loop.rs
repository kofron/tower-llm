use std::sync::Arc;

use async_openai::{config::OpenAIConfig, Client};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;
// no Tower imports needed thanks to DX sugar

// Import the experimental module without modifying lib.rs
// Core module is now at root level
// use openai_agents_rs directly

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    tracing_subscriber::fmt::init();

    // Tool: add two numbers (typed args)
    #[derive(Deserialize, JsonSchema)]
    struct AddArgs {
        a: f64,
        b: f64,
    }

    let add_tool = openai_agents_rs::tool_typed("calc_add", "Add two numbers", |args: AddArgs| async move {
        Ok::<_, tower::BoxError>(json!({ "sum": args.a + args.b }))
    });

    // Build an agent via the sugar builder
    let client: Arc<Client<OpenAIConfig>> = Arc::new(Client::new());
    let policy = openai_agents_rs::Policy::new()
        .until_no_tool_calls()
        .or_max_steps(4)
        .build();

    let mut agent = openai_agents_rs::Agent::builder(client)
        .model("gpt-4o")
        .temperature(0.0)
        .max_tokens(256)
        .tool(add_tool)
        .policy(policy)
        .build();

    let run = openai_agents_rs::run(
        &mut agent,
        "You are a careful assistant. For arithmetic, always call the appropriate tool.",
        "Compute ((2 + 3) + (10 + 5)).",
    )
    .await?;
    println!("Agent stopped after {} steps: {:?}", run.steps, run.stop);
    println!(
        "Final messages: {}",
        serde_json::to_string_pretty(&run.messages)?
    );

    Ok(())
}
