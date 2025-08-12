//! # OpenAI Agents SDK for Rust
//!
//! A lightweight yet powerful framework for building multi-agent workflows,
//! wrapping the `async-openai` crate for LLM interactions. This SDK provides
//! the core components to define agents, orchestrate their execution, and
//! manage state.
//!
//! ## Core Concepts
//!
//! - **[`Agent`]**: The fundamental building block, representing an entity that
//!   can process input and generate a response. Agents are defined by their
//!   configuration, including their identity, instructions, and tools.
//! - **[`Runner`]**: The engine that executes an agent's logic. It manages the
//!   interaction loop with the LLM, handles tool calls, and orchestrates the
//!   overall workflow.
//! - **[`Tool`]**: A function or capability that an agent can use to interact
//!   with the outside world, such as calling an API or accessing a database.
//! - **[`Session`]**: Manages the state of an interaction, including the history
//!   of messages. A [`SqliteSession`] is provided for persistent state.
//!
//! ## Getting Started
//!
//! To get started, you'll need to have your OpenAI API key set in the
//! `OPENAI_API_KEY` environment variable.
//!
//! Here's a simple example of how to create an agent and run it:
//!
//! ```rust,no_run
//! use openai_agents_rs::{runner::RunConfig, Agent, Runner};
//!
//! # async fn run_agent() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a simple agent that writes haikus.
//! let agent = Agent::simple(
//!     "HaikuBot",
//!     "You are a helpful assistant that writes haikus about programming concepts.",
//! );
//!
//! // Run the agent with a specific prompt.
//! let result = Runner::run(
//!     agent,
//!     "Write a haiku about recursion in programming",
//!     RunConfig::default(),
//! )
//! .await?;
//!
//! // Print the result.
//! if result.is_success() {
//!     println!("Agent response: {}", result.final_output);
//!     assert!(!result.final_output.is_null());
//! } else {
//!     panic!("Agent run failed: {:?}", result.error());
//! }
//! # Ok(())
//! # }
//! ```
//!
//! [`Agent`]: crate::agent::Agent
//! [`Runner`]: crate::runner::Runner
//! [`Tool`]: crate::tool::Tool
//! [`Session`]: crate::memory::Session
//! [`SqliteSession`]: crate::sqlite_session::SqliteSession

pub mod agent;
pub mod config;
pub mod error;
pub mod guardrail;
pub mod handoff;
pub mod items;
pub mod memory;
pub mod model;
pub mod result;
pub mod retry;
pub mod runner;
pub mod sqlite_session;
pub mod tool;
pub mod tracing;
pub mod usage;

// Re-export core types for convenience
pub use agent::{Agent, AgentConfig};
pub use error::{AgentsError, Result};
pub use guardrail::{InputGuardrail, OutputGuardrail};
pub use handoff::Handoff;
pub use memory::Session;
pub use result::{RunResult, StreamingRunResult};
pub use runner::Runner;
pub use sqlite_session::SqliteSession;
pub use tool::{FunctionTool, Tool};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_imports() {
        // Verify that all modules compile
        let _ = std::mem::size_of::<AgentsError>();
    }
}
