//! # Agent Group
//!
//! A thin builder to compose multiple agents into a single top-level agent via
//! handoffs. This allows a group of agents to "act like" one from the runner's
//! perspective.

use crate::{agent::Agent, handoff::Handoff};

/// A group of agents represented as a single top-level agent with configured handoffs.
#[derive(Clone)]
pub struct AgentGroup {
    root: Agent,
}

impl AgentGroup {
    /// Returns the composed top-level `Agent`.
    pub fn into_agent(self) -> Agent {
        self.root
    }
}

/// Builder for composing a group of agents.
pub struct AgentGroupBuilder {
    root: Agent,
    handoffs: Vec<Handoff>,
}

impl AgentGroupBuilder {
    /// Create a new builder with the specified root agent (the coordinator).
    pub fn new(root: Agent) -> Self {
        Self {
            root,
            handoffs: Vec::new(),
        }
    }

    /// Add a handoff from the root to the specified target agent.
    pub fn with_handoff(mut self, target: Agent, description: impl Into<String>) -> Self {
        self.handoffs.push(Handoff::new(target, description));
        self
    }

    /// Add multiple handoffs at once.
    pub fn with_handoffs(mut self, handoffs: Vec<Handoff>) -> Self {
        self.handoffs.extend(handoffs);
        self
    }

    /// Build the composed group into a top-level `Agent`.
    pub fn build(self) -> AgentGroup {
        let composed = self.root.with_handoffs(self.handoffs);
        AgentGroup { root: composed }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_group_builder_adds_handoffs() {
        let root = Agent::simple("Root", "Coordinates");
        let a = Agent::simple("A", "Specialist A");
        let b = Agent::simple("B", "Specialist B");

        let group = AgentGroupBuilder::new(root)
            .with_handoff(a, "A desc")
            .with_handoff(b, "B desc")
            .build();

        let agent = group.clone().into_agent();
        assert_eq!(agent.handoffs().len(), 2);
        assert_eq!(agent.handoffs()[0].name, "A");
        assert_eq!(agent.handoffs()[1].name, "B");
    }
}
