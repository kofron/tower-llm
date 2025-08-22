//! # Tower LLM
//!
//! A powerful Tower-based framework for building multi-agent workflows with OpenAI LLMs.
//! This SDK provides composable, type-safe components using Tower's service architecture
//! for maximum flexibility and performance.
//!
//! ## Core Concepts
//!
//! - **Agent**: A Tower service that processes chat requests through an LLM with tools
//! - **Tools**: Type-safe functions that agents can call, with automatic schema generation
//! - **Layers**: Tower middleware for cross-cutting concerns (retry, timeout, observability)
//! - **Static DI**: All dependencies are injected at construction time - no runtime lookups
//!
//! ## Getting Started
//!
//! Set your OpenAI API key in the `OPENAI_API_KEY` environment variable.
//!
//! ```rust,no_run
//! use tower_llm::{Agent, tool_typed, policies, CompositePolicy, run_user};
//! use async_openai::{config::OpenAIConfig, Client};
//! use schemars::JsonSchema;
//! use serde::Deserialize;
//! use std::sync::Arc;
//! use tower::{Service, ServiceExt};
//!
//! #[derive(Debug, Deserialize, JsonSchema)]
//! struct AddArgs {
//!     a: f64,
//!     b: f64,
//! }
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
//! // Create OpenAI client
//! let client = Arc::new(Client::<OpenAIConfig>::new());
//!
//! // Define a tool
//! let calculator = tool_typed(
//!     "add",
//!     "Add two numbers",
//!     |args: AddArgs| async move {
//!         Ok(serde_json::json!({ "sum": args.a + args.b }))
//!     },
//! );
//!
//! // Build an agent with per-agent instructions
//! let agent = Agent::builder(client)
//!     .model("gpt-4o")
//!     .instructions("You are a helpful math assistant")
//!     .tool(calculator)
//!     .policy(CompositePolicy::new(vec![policies::until_no_tool_calls()]))
//!     .build();
//!
//! // Use the agent with Tower's Service trait
//! let mut agent = agent;
//! // Send only a user message; the agent injects its instructions as a system message
//! let response = run_user(&mut agent, "What is 2 + 2?").await?;
//!
//! println!("Agent: {:?}", response);
//! # Ok(())
//! # }
//! ```

pub mod approvals;
pub mod auto_compaction;
pub mod budgets;
pub mod codec;
pub mod concurrency;
pub mod error;
pub mod groups;
pub mod items;
pub mod memory;
pub mod observability;
pub mod provider;
pub mod recording;
pub mod resilience;
pub mod result;
pub mod sessions;
pub mod sqlite_session;
pub mod streaming;
pub mod tap;
pub mod validation;

// Core module with main implementation
mod core;

// Re-export core types
pub use core::{
    policies, run, run_user, simple_chat_request, simple_user_request, tool_typed, Agent,
    AgentBuilder, AgentLoop, AgentLoopLayer,
    AgentPolicy, AgentRun, AgentStopReason, AgentSvc, CompositePolicy, LoopState, Policy, PolicyFn,
    Step, StepAux, StepLayer, StepOutcome, ToolDef, ToolInvocation, ToolOutput, ToolRouter,
    ToolSvc,
};

// Re-export join policy for tool execution configuration
pub use core::ToolJoinPolicy;

// Public re-exports for convenience
pub use error::{AgentsError, Result};
pub use memory::Session;
pub use result::{RunResult, RunResultWithContext, StreamingRunResult};
pub use sqlite_session::SqliteSession;

// Re-export async-openai types that users need
pub use async_openai::{
    config::OpenAIConfig,
    types::{
        ChatCompletionRequestMessage, CreateChatCompletionRequest, CreateChatCompletionRequestArgs,
        ReasoningEffort,
    },
    Client,
};

// Re-export Tower traits that users need
pub use tower::{Layer, Service, ServiceExt};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_imports() {
        // Verify that all modules compile
        let _ = std::mem::size_of::<AgentsError>();
    }
}
