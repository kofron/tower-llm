//! Agent definition and configuration
//!
//! Agents are the core abstraction - LLMs configured with instructions, tools, and handoffs.

use std::sync::Arc;

use crate::guardrail::{InputGuardrail, OutputGuardrail};
use crate::handoff::Handoff;
use crate::items::Message;
use crate::tool::Tool;

/// Configuration for an agent
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// Name of the agent
    pub name: String,

    /// System instructions for the agent
    pub instructions: String,

    /// Description used when this agent is a handoff target
    pub handoff_description: Option<String>,

    /// Tools available to this agent
    pub tools: Vec<Arc<dyn Tool>>,

    /// Other agents this agent can hand off to
    pub handoffs: Vec<Handoff>,

    /// Input guardrails to validate user input
    pub input_guardrails: Vec<Arc<dyn InputGuardrail>>,

    /// Output guardrails to validate agent output
    pub output_guardrails: Vec<Arc<dyn OutputGuardrail>>,

    /// Model to use (e.g., "gpt-4", "gpt-3.5-turbo")
    pub model: String,

    /// Maximum number of turns before stopping
    pub max_turns: Option<usize>,

    /// Temperature for generation (0.0 to 2.0)
    pub temperature: Option<f32>,

    /// Maximum tokens to generate
    pub max_tokens: Option<u32>,

    /// Output schema if structured output is needed
    pub output_schema: Option<serde_json::Value>,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            name: "Assistant".to_string(),
            instructions: "You are a helpful assistant.".to_string(),
            handoff_description: None,
            tools: vec![],
            handoffs: vec![],
            input_guardrails: vec![],
            output_guardrails: vec![],
            model: "gpt-4".to_string(),
            max_turns: Some(10),
            temperature: Some(0.7),
            max_tokens: None,
            output_schema: None,
        }
    }
}

/// An agent that can process messages and use tools
#[derive(Clone)]
pub struct Agent {
    pub config: AgentConfig,
}

impl Agent {
    /// Create a new agent with the given configuration
    pub fn new(config: AgentConfig) -> Self {
        Self { config }
    }

    /// Create a simple agent with just a name and instructions
    pub fn simple(name: impl Into<String>, instructions: impl Into<String>) -> Self {
        Self::new(AgentConfig {
            name: name.into(),
            instructions: instructions.into(),
            ..Default::default()
        })
    }

    /// Builder pattern: set the model
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.config.model = model.into();
        self
    }

    /// Builder pattern: add a tool
    pub fn with_tool(mut self, tool: Arc<dyn Tool>) -> Self {
        self.config.tools.push(tool);
        self
    }

    /// Builder pattern: add multiple tools
    pub fn with_tools(mut self, tools: Vec<Arc<dyn Tool>>) -> Self {
        self.config.tools.extend(tools);
        self
    }

    /// Builder pattern: add a handoff
    pub fn with_handoff(mut self, handoff: Handoff) -> Self {
        self.config.handoffs.push(handoff);
        self
    }

    /// Builder pattern: add multiple handoffs
    pub fn with_handoffs(mut self, handoffs: Vec<Handoff>) -> Self {
        self.config.handoffs.extend(handoffs);
        self
    }

    /// Builder pattern: set temperature
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.config.temperature = Some(temperature);
        self
    }

    /// Builder pattern: set max turns
    pub fn with_max_turns(mut self, max_turns: usize) -> Self {
        self.config.max_turns = Some(max_turns);
        self
    }

    /// Builder pattern: set max tokens
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.config.max_tokens = Some(max_tokens);
        self
    }

    /// Builder pattern: add input guardrail
    pub fn with_input_guardrail(mut self, guardrail: Arc<dyn InputGuardrail>) -> Self {
        self.config.input_guardrails.push(guardrail);
        self
    }

    /// Builder pattern: add output guardrail
    pub fn with_output_guardrail(mut self, guardrail: Arc<dyn OutputGuardrail>) -> Self {
        self.config.output_guardrails.push(guardrail);
        self
    }

    /// Builder pattern: set output schema for structured outputs
    pub fn with_output_schema(mut self, schema: serde_json::Value) -> Self {
        self.config.output_schema = Some(schema);
        self
    }

    /// Get the agent's name
    pub fn name(&self) -> &str {
        &self.config.name
    }

    /// Get the agent's instructions
    pub fn instructions(&self) -> &str {
        &self.config.instructions
    }

    /// Get available tools
    pub fn tools(&self) -> &[Arc<dyn Tool>] {
        &self.config.tools
    }

    /// Get available handoffs
    pub fn handoffs(&self) -> &[Handoff] {
        &self.config.handoffs
    }

    /// Check if the agent has any tools
    pub fn has_tools(&self) -> bool {
        !self.config.tools.is_empty()
    }

    /// Check if the agent has any handoffs
    pub fn has_handoffs(&self) -> bool {
        !self.config.handoffs.is_empty()
    }

    /// Build the system message for this agent
    pub fn build_system_message(&self) -> Message {
        let mut content = self.config.instructions.clone();

        // Add tool descriptions if any
        if !self.config.tools.is_empty() {
            content.push_str("\n\nYou have access to the following tools:\n");
            for tool in &self.config.tools {
                content.push_str(&format!("- {}: {}\n", tool.name(), tool.description()));
            }
        }

        // Add handoff descriptions if any
        if !self.config.handoffs.is_empty() {
            content.push_str("\n\nYou can hand off to the following agents:\n");
            for handoff in &self.config.handoffs {
                content.push_str(&format!("- {}: {}\n", handoff.name, handoff.description));
            }
        }

        Message::system(content)
    }
}

impl std::fmt::Debug for Agent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Agent")
            .field("name", &self.config.name)
            .field("model", &self.config.model)
            .field("tools_count", &self.config.tools.len())
            .field("handoffs_count", &self.config.handoffs.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::FunctionTool;

    #[test]
    fn test_agent_creation() {
        let agent = Agent::simple("TestAgent", "You are a test agent");
        assert_eq!(agent.name(), "TestAgent");
        assert_eq!(agent.instructions(), "You are a test agent");
        assert_eq!(agent.config.model, "gpt-4");
    }

    #[test]
    fn test_agent_builder() {
        let tool = Arc::new(FunctionTool::simple(
            "test_tool",
            "A test tool",
            |s: String| s.to_uppercase(),
        ));

        let agent = Agent::simple("Builder", "Test instructions")
            .with_model("gpt-3.5-turbo")
            .with_temperature(0.5)
            .with_max_turns(5)
            .with_max_tokens(1000)
            .with_tool(tool.clone());

        assert_eq!(agent.config.model, "gpt-3.5-turbo");
        assert_eq!(agent.config.temperature, Some(0.5));
        assert_eq!(agent.config.max_turns, Some(5));
        assert_eq!(agent.config.max_tokens, Some(1000));
        assert_eq!(agent.tools().len(), 1);
        assert!(agent.has_tools());
    }

    #[test]
    fn test_agent_with_handoffs() {
        let spanish_agent = Agent::simple("Spanish", "Speaks Spanish");
        let english_agent = Agent::simple("English", "Speaks English");

        let handoff1 = Handoff::new(spanish_agent, "Handles Spanish requests");
        let handoff2 = Handoff::new(english_agent, "Handles English requests");

        let triage_agent =
            Agent::simple("Triage", "Routes requests").with_handoffs(vec![handoff1, handoff2]);

        assert_eq!(triage_agent.handoffs().len(), 2);
        assert!(triage_agent.has_handoffs());
    }

    #[test]
    fn test_system_message_generation() {
        let tool = Arc::new(FunctionTool::simple(
            "weather",
            "Get weather information",
            |s: String| format!("Weather for {}", s),
        ));

        let helper_agent = Agent::simple("Helper", "I help with tasks");
        let handoff = Handoff::new(helper_agent, "Handles complex tasks");

        let agent = Agent::simple("Main", "I am the main agent")
            .with_tool(tool)
            .with_handoff(handoff);

        let sys_msg = agent.build_system_message();
        assert_eq!(sys_msg.role, crate::items::Role::System);
        assert!(sys_msg.content.contains("I am the main agent"));
        assert!(sys_msg.content.contains("weather"));
        assert!(sys_msg.content.contains("Helper"));
    }

    #[test]
    fn test_agent_with_output_schema() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "confidence": {"type": "number"}
            },
            "required": ["answer"]
        });

        let agent = Agent::simple("Structured", "Provides structured output")
            .with_output_schema(schema.clone());

        assert_eq!(agent.config.output_schema, Some(schema));
    }

    #[test]
    fn test_agent_default_config() {
        let config = AgentConfig::default();
        assert_eq!(config.name, "Assistant");
        assert_eq!(config.model, "gpt-4");
        assert_eq!(config.max_turns, Some(10));
        assert_eq!(config.temperature, Some(0.7));
        assert!(config.tools.is_empty());
        assert!(config.handoffs.is_empty());
    }

    #[test]
    fn test_agent_clone() {
        let agent = Agent::simple("Original", "Original instructions");
        let cloned = agent.clone();

        assert_eq!(cloned.name(), "Original");
        assert_eq!(cloned.instructions(), "Original instructions");
    }

    #[test]
    fn test_agent_debug_format() {
        let agent = Agent::simple("Debug", "Debug agent");
        let debug_str = format!("{:?}", agent);

        assert!(debug_str.contains("Debug"));
        assert!(debug_str.contains("gpt-4"));
        assert!(debug_str.contains("tools_count"));
    }
}
