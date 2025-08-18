//! # Example: Mixed Run-scoped and Per-agent Context
//!
//! This example demonstrates a run-scoped counter applied across the run and a
//! per-agent handler that wraps the tool output for a specific agent.
//!
//! Expected: The output should show that the per-agent wrapping occurred, and
//! the run-scoped counter incremented.

use openai_agents_rs::{
    group::AgentGroupBuilder, runner::RunConfig, Agent, ContextStep, Runner, ToolContext,
};
use serde_json::Value;

#[derive(Clone, Default)]
struct RunCtx {
    total: usize,
}
struct RunCount;
impl ToolContext<RunCtx> for RunCount {
    fn on_tool_output(
        &self,
        mut ctx: RunCtx,
        _tool: &str,
        _args: &Value,
        result: Result<Value, String>,
    ) -> openai_agents_rs::Result<ContextStep<RunCtx>> {
        ctx.total += 1;
        Ok(ContextStep::rewrite(ctx, result.unwrap_or(Value::Null)))
    }
}

#[derive(Clone, Default)]
struct AgentCtx {
    wrapped: bool,
}
struct AgentWrap;
impl ToolContext<AgentCtx> for AgentWrap {
    fn on_tool_output(
        &self,
        mut ctx: AgentCtx,
        _tool: &str,
        _args: &Value,
        result: Result<Value, String>,
    ) -> openai_agents_rs::Result<ContextStep<AgentCtx>> {
        ctx.wrapped = true;
        let v = result.unwrap_or(Value::Null);
        Ok(ContextStep::rewrite(
            ctx,
            serde_json::json!({"agent_wrapped": v}),
        ))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let specialist =
        Agent::simple("Specialist", "Use tools").with_context_typed(AgentCtx::default(), AgentWrap);
    let coordinator = Agent::simple("Coordinator", "Delegate to Specialist");
    let group = AgentGroupBuilder::new(coordinator)
        .with_handoff(specialist.into_inner(), "Do work")
        .build();

    let config = RunConfig::default().with_run_context(RunCtx::default, RunCount);
    let out = Runner::run_with_run_context(
        group.into_agent(),
        "Perform a task using tools",
        config,
        RunCtx::default,
        RunCount,
    )
    .await?;

    println!("Response:\n{}", out.result.final_output);
    println!("Run counter: {}", out.context.total);
    Ok(())
}
