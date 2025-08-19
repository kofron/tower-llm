use openai_agents_rs::env::{EnvBuilder, Approval, ApprovalCapability};
use openai_agents_rs::service::{ApprovalLayer, InputSchemaLayer, RetryLayer, ToolRequest, ToolResponse};
use openai_agents_rs::{tool::FunctionTool, tool_service::IntoToolService, Tool};
use std::sync::Arc;
use tower::{Layer, ServiceExt};

// Custom approval capability that only approves safe tools
#[derive(Default)]
struct SafeToolApproval;

impl Approval for SafeToolApproval {
    fn request_approval(&self, operation: &str, details: &str) -> bool {
        println!("Approval request: {}", operation);
        println!("Details: {}", details);
        
        // Only approve operations for tools that are not "danger"
        if operation.contains("tool:danger") {
            println!("❌ DENIED - dangerous tool");
            false
        } else {
            println!("✅ APPROVED - safe tool");
            true
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let safe = Arc::new(FunctionTool::simple("safe", "ok", |s: String| s));

    // Create environment with approval capability using the general wrapper
    let env = EnvBuilder::new()
        .with_capability(Arc::new(ApprovalCapability::new(SafeToolApproval)))
        .build();

    // Compose a typed stack manually (advanced) using service-based tools
    // Approval → Retry → InputSchema → ServiceTool
    let base = <FunctionTool as Clone>::clone(&safe).into_service();
    let stack = ApprovalLayer.layer(
        RetryLayer::times(2).layer(InputSchemaLayer::lenient(safe.parameters_schema()).layer(base)),
    );

    // Call with env that has approval capability
    let req = ToolRequest {
        env: env.clone(),
        run_id: "r".into(),
        agent: "A".into(),
        tool_call_id: "id1".into(),
        tool_name: "safe".into(),
        arguments: serde_json::json!({"input":"ok"}),
    };

    let resp: ToolResponse = stack.clone().oneshot(req).await?;
    println!("response: error={:?} output={}", resp.error, resp.output);
    
    // Also demonstrate denial for dangerous tool
    let danger_req = ToolRequest {
        env: env.clone(),
        run_id: "r2".into(),
        agent: "A".into(),
        tool_call_id: "id2".into(),
        tool_name: "danger".into(),
        arguments: serde_json::json!({"input":"evil"}),
    };
    
    let danger_resp: ToolResponse = stack.oneshot(danger_req).await?;
    println!("danger response: error={:?} output={}", danger_resp.error, danger_resp.output);
    
    Ok(())
}
