//! OpenAI Agents SDK for Rust
//!
//! A lightweight yet powerful framework for building multi-agent workflows,
//! wrapping the async-openai crate for LLM interactions.

pub mod agent;
pub mod error;
pub mod guardrail;
pub mod handoff;
pub mod items;
pub mod memory;
pub mod model;
pub mod result;
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
