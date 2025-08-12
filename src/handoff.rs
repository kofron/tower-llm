//! Handoff system for transferring control between agents
//!
//! Handoffs allow agents to delegate tasks to specialized agents.

use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::agent::Agent;

/// A handoff represents the ability to transfer control to another agent
#[derive(Clone)]
pub struct Handoff {
    /// Name of the target agent
    pub name: String,

    /// Description of what this agent handles
    pub description: String,

    /// The actual agent to hand off to
    pub agent: Arc<Agent>,
}

impl Handoff {
    /// Create a new handoff
    pub fn new(agent: Agent, description: impl Into<String>) -> Self {
        let name = agent.name().to_string();
        Self {
            name,
            description: description.into(),
            agent: Arc::new(agent),
        }
    }

    /// Create a handoff with a custom name
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

    /// Get the target agent
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

/// Data passed during a handoff
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandoffData {
    /// The agent initiating the handoff
    pub from_agent: String,

    /// The target agent
    pub to_agent: String,

    /// Reason for the handoff
    pub reason: Option<String>,

    /// Any context to pass along
    pub context: Option<serde_json::Value>,
}

impl HandoffData {
    /// Create new handoff data
    pub fn new(from: impl Into<String>, to: impl Into<String>) -> Self {
        Self {
            from_agent: from.into(),
            to_agent: to.into(),
            reason: None,
            context: None,
        }
    }

    /// Set the reason for handoff
    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = Some(reason.into());
        self
    }

    /// Set context data
    pub fn with_context(mut self, context: serde_json::Value) -> Self {
        self.context = Some(context);
        self
    }
}

/// Result of a handoff decision
#[derive(Debug, Clone)]
pub enum HandoffDecision {
    /// Continue with current agent
    Continue,

    /// Hand off to another agent
    HandOff(Handoff, HandoffData),

    /// Complete the task
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
