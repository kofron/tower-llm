//! # Agent Handoff System
//!
//! The handoff system is a powerful feature that enables the creation of
//! sophisticated, multi-agent workflows. It allows one agent to transfer
//! control of a conversation to another, more specialized agent. This is
//! particularly useful for tasks that require different areas of expertise.
//!
//! ## The Handoff Mechanism
//!
//! A handoff is essentially a specialized tool that, when called, switches the
//! active agent in the conversation. The [`Handoff`] struct represents a potential
//! handoff target. It includes the target agent and a description of its
//! capabilities, which helps the primary agent decide when to delegate.
//!
//! When a handoff occurs, [`HandoffData`] is used to pass contextual information,
//! such as the reason for the handoff, from the originating agent to the
//! target agent. The [`HandoffDecision`] enum is used to represent the outcome
//! of a handoff evaluation.
//!
//! ### Example: A Multi-Agent Help Desk
//!
//! Imagine a help desk system with a general-purpose "Triage" agent that
//! initially handles all user requests. If a request is about a technical
//! issue, the Triage agent can hand it off to a "TechnicalSupport" agent.
//!
//! ```rust
//! use openai_agents_rs::{Agent, Handoff};
//!
//! // The specialist agent.
//! let tech_support_agent = Agent::simple(
//!     "TechnicalSupport",
//!     "You are a technical support specialist.",
//! );
//!
//! // The handoff configuration.
//! let tech_support_handoff = Handoff::new(
//!     tech_support_agent,
//!     "Handles technical support questions about our product.",
//! );
//!
//! // The primary agent, configured with the handoff.
//! let triage_agent = Agent::simple(
//!     "TriageBot",
//!     "You are a first-line support agent. You route questions to specialists.",
//! )
//! .with_handoff(tech_support_handoff);
//!
//! assert_eq!(triage_agent.handoffs().len(), 1);
//! assert_eq!(triage_agent.handoffs()[0].name, "TechnicalSupport");
//! ```
//!
//! Handoffs allow agents to delegate tasks to specialized agents.

use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::agent::Agent;
use crate::error::Result;
use crate::tool::{Tool, ToolResult};
use async_trait::async_trait;
use serde_json::Value;

/// Represents a potential handoff target, allowing one agent to transfer
/// control to another.
///
/// A `Handoff` is treated like a tool by the agent, but instead of executing a
/// function, it switches the active agent in the conversation.
#[derive(Clone)]
pub struct Handoff {
    /// The name of the target agent. This is used to identify the handoff
    /// and is what the primary agent will call as a "tool".
    pub name: String,

    /// A description of the target agent's capabilities. This is provided to
    /// the primary agent to help it decide when a handoff is appropriate.
    pub description: String,

    /// An `Arc` pointing to the actual `Agent` instance to hand off to.
    pub agent: Arc<Agent>,
}

impl Handoff {
    /// Creates a new `Handoff` configuration.
    ///
    /// The `name` of the handoff is automatically taken from the target agent's
    /// name.
    pub fn new(agent: Agent, description: impl Into<String>) -> Self {
        let name = agent.name().to_string();
        Self {
            name,
            description: description.into(),
            agent: Arc::new(agent),
        }
    }

    /// Creates a new `Handoff` with a custom name.
    ///
    /// This allows you to define a different name for the handoff "tool" than
    /// the target agent's actual name.
    pub fn with_name(
        agent: Agent,
        name: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            agent: Arc::new(agent),
        }
    }

    /// Returns a reference to the target `Agent`.
    pub fn agent(&self) -> &Agent {
        &self.agent
    }
}

impl std::fmt::Debug for Handoff {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Handoff")
            .field("name", &self.name)
            .field("description", &self.description)
            .finish()
    }
}

/// Adapter to expose a handoff as a Tool to the model provider.
#[derive(Clone, Debug)]
pub struct HandoffTool {
    handoff: Handoff,
}

impl From<Handoff> for HandoffTool {
    fn from(h: Handoff) -> Self {
        Self { handoff: h }
    }
}

#[async_trait]
impl Tool for HandoffTool {
    fn name(&self) -> &str {
        &self.handoff.name
    }

    fn description(&self) -> &str {
        &self.handoff.description
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "reason": {"type": "string", "description": "Reason for handoff"}
            }
        })
    }

    async fn execute(&self, _arguments: Value) -> Result<ToolResult> {
        // This tool should never be executed directly; the runner intercepts handoffs.
        Ok(ToolResult::success(serde_json::json!({"handoff": true})))
    }
}

/// Contains the data passed between agents during a handoff.
///
/// This struct provides contextual information to the target agent, such as
/// which agent initiated the handoff and for what reason.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandoffData {
    /// The name of the agent initiating the handoff.
    pub from_agent: String,

    /// The name of the agent that will take over the conversation.
    pub to_agent: String,

    /// An optional explanation for why the handoff is occurring.
    pub reason: Option<String>,

    /// Any additional context or data to be passed to the target agent, as a
    /// `serde_json::Value`.
    pub context: Option<serde_json::Value>,
}

impl HandoffData {
    /// Creates a new `HandoffData` instance.
    pub fn new(from: impl Into<String>, to: impl Into<String>) -> Self {
        Self {
            from_agent: from.into(),
            to_agent: to.into(),
            reason: None,
            context: None,
        }
    }

    /// Sets the reason for the handoff.
    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = Some(reason.into());
        self
    }

    /// Sets the context data for the handoff.
    pub fn with_context(mut self, context: serde_json::Value) -> Self {
        self.context = Some(context);
        self
    }
}

/// Represents the possible outcomes of a handoff evaluation.
///
/// This enum is used to signal whether the conversation should continue with
/// the current agent, be handed off to another, or be completed.
#[derive(Debug, Clone)]
pub enum HandoffDecision {
    /// The conversation should continue with the current agent.
    Continue,

    /// The conversation should be handed off to the specified agent.
    HandOff(Handoff, HandoffData),

    /// The task is complete, and the run should end with the provided message.
    Complete(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::Agent;

    #[test]
    fn test_handoff_creation() {
        let agent = Agent::simple("Specialist", "I handle special cases");
        let handoff = Handoff::new(agent.clone(), "Handles complex queries");

        assert_eq!(handoff.name, "Specialist");
        assert_eq!(handoff.description, "Handles complex queries");
        assert_eq!(handoff.agent().name(), "Specialist");
    }

    #[test]
    fn test_handoff_with_custom_name() {
        let agent = Agent::simple("Agent1", "Instructions");
        let handoff = Handoff::with_name(agent, "CustomName", "Custom description");

        assert_eq!(handoff.name, "CustomName");
        assert_eq!(handoff.description, "Custom description");
    }

    #[test]
    fn test_handoff_data() {
        let data = HandoffData::new("AgentA", "AgentB")
            .with_reason("User needs specialized help")
            .with_context(serde_json::json!({"priority": "high"}));

        assert_eq!(data.from_agent, "AgentA");
        assert_eq!(data.to_agent, "AgentB");
        assert_eq!(data.reason, Some("User needs specialized help".to_string()));
        assert!(data.context.is_some());
    }

    #[test]
    fn test_handoff_data_serialization() {
        let data = HandoffData::new("Source", "Target").with_reason("Testing");

        let serialized = serde_json::to_string(&data).unwrap();
        let deserialized: HandoffData = serde_json::from_str(&serialized).unwrap();

        assert_eq!(data.from_agent, deserialized.from_agent);
        assert_eq!(data.to_agent, deserialized.to_agent);
        assert_eq!(data.reason, deserialized.reason);
    }

    #[test]
    fn test_handoff_decision() {
        // Test Continue
        let decision = HandoffDecision::Continue;
        assert!(matches!(decision, HandoffDecision::Continue));

        // Test Complete
        let complete = HandoffDecision::Complete("Task done".to_string());
        if let HandoffDecision::Complete(msg) = complete {
            assert_eq!(msg, "Task done");
        } else {
            panic!("Expected Complete variant");
        }

        // Test HandOff
        let agent = Agent::simple("Helper", "Helps");
        let handoff = Handoff::new(agent, "Description");
        let data = HandoffData::new("Main", "Helper");
        let handoff_decision = HandoffDecision::HandOff(handoff.clone(), data.clone());

        if let HandoffDecision::HandOff(h, d) = handoff_decision {
            assert_eq!(h.name, "Helper");
            assert_eq!(d.from_agent, "Main");
        } else {
            panic!("Expected HandOff variant");
        }
    }

    #[test]
    fn test_handoff_debug_format() {
        let agent = Agent::simple("Debug", "Debug agent");
        let handoff = Handoff::new(agent, "Debug description");

        let debug_str = format!("{:?}", handoff);
        assert!(debug_str.contains("Debug"));
        assert!(debug_str.contains("Debug description"));
        assert!(!debug_str.contains("agent:")); // Should not show the agent field
    }
}
