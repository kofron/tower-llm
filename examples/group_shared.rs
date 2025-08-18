//! # Example: Agent Group with Run-scoped Shared Context
//!
//! This example composes agents and uses a run-scoped context to count how many
//! tool outputs occur across the entire run (including handoffs).
//!
//! Expected: The response should be produced after delegations, and the
//! run-scoped context counter should be incremented at least once.

use openai_agents_rs::{
    group::AgentGroupBuilder, runner::RunConfig, Agent, ContextStep, Runner, ToolContext,
};
use serde_json::Value;

#[derive(Clone, Default)]
struct Ctx {
    n: usize,
}

struct Count;
impl ToolContext<Ctx> for Count {
    fn on_tool_output(
        &self,
        mut ctx: Ctx,
        _tool: &str,
        _args: &Value,
        result: Result<Value, String>,
    ) -> openai_agents_rs::Result<ContextStep<Ctx>> {
        ctx.n += 1;
        Ok(ContextStep::rewrite(ctx, result.unwrap_or(Value::Null)))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_target(true)
        .compact()
        .init();

    let researcher = Agent::simple("Researcher", "Use tools to find facts when asked.");
    let analyst = Agent::simple("Analyst", "Summarize findings when asked.");

    let coordinator = Agent::simple(
        "Coordinator",
        "Delegate: ask 'Researcher' to find facts, then 'Analyst' to summarize.",
    );

    let group = AgentGroupBuilder::new(coordinator)
        .with_handoff(researcher, "Search knowledge")
        .with_handoff(analyst, "Analyze info")
        .build();

    let config = RunConfig::default().with_run_context(Ctx::default, Count);

    let out = Runner::run_with_run_context(
        group.into_agent(),
        "Find information about Rust (the programming language) and summarize it",
        config,
        Ctx::default,
        Count,
    )
    .await?;
    println!("Messages:\n{:?}", out.result.messages);
    println!("Response:\n{}", out.result.final_output);
    println!("Run-scoped counter: {}", out.context.n);
    Ok(())
}
