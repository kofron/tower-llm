use openai_agents_rs::{layers, runner::RunConfig, Agent, FunctionTool, Runner};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let safe = Arc::new(FunctionTool::simple("safe", "ok", |s: String| s));
    let danger = Arc::new(FunctionTool::simple("danger", "", |s: String| s));

    let agent = Agent::simple("Approval", "Use tools")
        .with_tool(safe)
        .with_tool(danger);

    // Deny the tool named "danger"
    let cfg = RunConfig::default().with_run_layers(vec![layers::boxed_approval_with(
        |_agent, tool, _args| tool != "danger",
    )]);

    let _ = Runner::run(agent, "Try some tools", cfg).await?;
    Ok(())
}
