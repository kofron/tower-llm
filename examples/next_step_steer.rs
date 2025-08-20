use std::sync::Arc;

use async_openai::{config::OpenAIConfig, Client};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;
// no Tower imports needed thanks to DX sugar

// Import the experimental module without modifying lib.rs
#[path = "../src/next/mod.rs"]
mod next;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    tracing_subscriber::fmt::init();

    // Define a simple add tool (typed)
    #[derive(Deserialize, JsonSchema)]
    struct AddArgs {
        a: f64,
        b: f64,
    }
    let add_tool = next::tool_typed("calc_add", "Add two numbers", |args: AddArgs| async move {
        Ok::<_, tower::BoxError>(json!({ "sum": args.a + args.b }))
    });

    // Build an agent with a max-1-step policy to mimic a single step
    let client: Arc<Client<OpenAIConfig>> = Arc::new(Client::new());
    let policy = next::Policy::new().or_max_steps(1).build();
    let mut agent = next::Agent::builder(client)
        .model("gpt-4o")
        .temperature(0.0)
        .max_tokens(256)
        .tool(add_tool)
        .policy(policy)
        .build();

    let run = next::run(
        &mut agent,
        "You are a careful assistant. For arithmetic, always call the appropriate tool.",
        "Add 2 and 3.",
    )
    .await?;
    println!("Stopped after {} steps: {:?}", run.steps, run.stop);
    println!("Messages: {}", serde_json::to_string_pretty(&run.messages)?);

    Ok(())
}
