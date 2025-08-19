use openai_agents_rs::env::Env;
use openai_agents_rs::service::{HasApproval, ApprovalLayer, InputSchemaLayer, RetryLayer, ToolRequest, ToolResponse};
use openai_agents_rs::{tool::FunctionTool, tool_service::IntoToolService, Tool};
use std::sync::Arc;
use tower::{Layer, ServiceExt};

// Advanced: typed Env implementing HasApproval used by ApprovalLayer
#[derive(Clone, Default)]
struct EnvAllowSafe;

impl Env for EnvAllowSafe {
    fn capability<T: std::any::Any + Send + Sync>(&self) -> Option<Arc<T>> {
        None
    }
}

impl HasApproval for EnvAllowSafe {
    fn approve(&self, _agent: &str, tool: &str, _args: &serde_json::Value) -> bool {
        tool != "danger"
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let safe = Arc::new(FunctionTool::simple("safe", "ok", |s: String| s));

    // Compose a typed stack manually (advanced) using service-based tools
    // Approval → Retry → InputSchema → ServiceTool
    let base = <FunctionTool as Clone>::clone(&safe).into_service::<EnvAllowSafe>();
    let stack = ApprovalLayer.layer(
        RetryLayer::times(2).layer(InputSchemaLayer::lenient(safe.parameters_schema()).layer(base)),
    );

    // Call with typed Env implementing HasApproval
    let req = ToolRequest::<EnvAllowSafe> {
        env: EnvAllowSafe,
        run_id: "r".into(),
        agent: "A".into(),
        tool_call_id: "id1".into(),
        tool_name: "safe".into(),
        arguments: serde_json::json!({"input":"ok"}),
    };

    let resp: ToolResponse = stack.oneshot(req).await?;
    println!("response: error={:?} output={}", resp.error, resp.output);
    Ok(())
}
