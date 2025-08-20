use serde_json::json;
use futures::FutureExt;
use tower::{Service, ServiceExt};

#[path = "../src/next/mod.rs"]
mod next;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Define a simple echo tool
    let echo = next::ToolDef::from_handler(
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

    let (mut router, _specs) = next::ToolRouter::new(vec![echo]);
    let out = router
        .ready()
        .await?
        .call(next::ToolInvocation { id: "1".into(), name: "echo".into(), arguments: json!({"text":"hi"}) })
        .await?;
    println!("tool out={}", out.result);
    Ok(())
}


