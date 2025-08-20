use serde_json::json;
use futures::FutureExt;
use tower::{Service, ServiceExt};

// Core module is now at root level
// use openai_agents_rs directly

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Define a simple echo tool
    let echo = openai_agents_rs::ToolDef::from_handler(
        "echo",
        "Echo back the input",
        json!({
            "type": "object",
            "properties": { "text": {"type": "string"} },
            "required": ["text"],
        }),
        std::sync::Arc::new(|args: serde_json::Value| {
            futures::future::ready(Ok::<_, tower::BoxError>(args)).boxed()
        }),
    );

    let (mut router, _specs) = openai_agents_rs::ToolRouter::new(vec![echo]);
    let out = router
        .ready()
        .await?
        .call(openai_agents_rs::ToolInvocation { id: "1".into(), name: "echo".into(), arguments: json!({"text":"hi"}) })
        .await?;
    println!("tool out={}", out.result);
    Ok(())
}


