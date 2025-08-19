//! # Example: Agent Group (No Shared Context)
//!
//! This example composes multiple agents into a single top-level agent via
//! handoffs. There is no shared run-scoped context; each agent operates with
//! its own local state.
//!
//! Expected: The coordinator should hand off to a specialist and return a
//! response. Logs should show agent handoffs. There is no shared context state.

use openai_agents_rs::{group::AgentGroupBuilder, runner::RunConfig, Agent, Runner};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    let researcher = Agent::simple(
        "Researcher",
        "Use the 'search' tool to find facts when asked.",
    );
    let analyst = Agent::simple(
        "Analyst",
        "When asked to summarize, provide a concise summary.",
    );

    let coordinator = Agent::simple(
        "Coordinator",
        "Delegate tasks: ask 'Researcher' to find facts, then 'Analyst' to summarize the findings.",
    );

    let group = AgentGroupBuilder::new(coordinator)
        .with_handoff(researcher, "Search knowledge")
        .with_handoff(analyst, "Analyze info")
        .build();

    let result = Runner::run(
        group.into_agent(),
        "Find information about Rust and summarize it",
        RunConfig::default(),
    )
    .await?;

    println!("Response:\n{}", result.final_output);
    Ok(())
}
