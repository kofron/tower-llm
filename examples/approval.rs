use openai_agents_rs::{
    service::BoxedApprovalLayer,
    runner::RunConfig, 
    Agent, 
    FunctionTool, 
    Runner
};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let safe = Arc::new(FunctionTool::simple("safe", "ok", |s: String| s));
    let danger = Arc::new(FunctionTool::simple("danger", "", |s: String| s));

    let agent = Agent::simple("Approval", "Use tools")
        .with_tool(safe)
        .with_tool(danger);

    // TODO: Convert to typed layer API once predicate-based approval layer is available
    // or migrate to capability-based approval when Runner supports custom Env
    let cfg = RunConfig::default().with_run_layers(vec![
        Arc::new(BoxedApprovalLayer::new(|_agent, tool, _args| tool != "danger"))
    ]);

    let _ = Runner::run(agent, "Try some tools", cfg).await?;
    Ok(())
}
