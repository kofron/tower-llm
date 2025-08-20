use futures::FutureExt;
use serde_json::json;
use tower::{Service, ServiceExt};

// Core module is now at root level
// use tower_llm directly

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Define a simple echo tool
    let echo = tower_llm::ToolDef::from_handler(
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

    let (mut router, _specs) = tower_llm::ToolRouter::new(vec![echo]);
    let out = router
        .ready()
        .await?
        .call(tower_llm::ToolInvocation {
            id: "1".into(),
            name: "echo".into(),
            arguments: json!({"text":"hi"}),
        })
        .await?;
    println!("tool out={}", out.result);
    Ok(())
}
