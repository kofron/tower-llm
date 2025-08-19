//! # Agent (orientation)
//!
//! An `Agent` represents a configured participant in a workflow: a name,
//! instructions, tools, and optional handoffs. Agents are composable and can be
//! grouped; policy layers and context shape tool execution at run/agent/tool
//! scopes. This module defines the `Agent` API and its configuration surface.

use std::sync::Arc;

use crate::guardrail::{InputGuardrail, OutputGuardrail};
use crate::handoff::Handoff;
use crate::items::Message;
use crate::service::ErasedToolLayer;
use crate::tool::Tool;

/// Defines the complete configuration for an [`Agent`].
///
/// Holds identity, instructions, tools, handoffs, model settings, and optional
/// context/policy hooks. Built for composition and reuse.
#[derive(Clone)]
pub struct AgentConfig {
    /// The name of the agent, used for identification and in logs.
    pub name: String,

    /// The system instructions that guide the agent's behavior. These are
    /// typically used to set the context and define the agent's persona.
    pub instructions: String,

    /// A description of the agent's capabilities, used when this agent is a
    /// potential handoff target for another agent.
    pub handoff_description: Option<String>,

    /// A list of tools that the agent can use to perform actions. Tools are
    /// functions that can be called to interact with external systems.
    pub tools: Vec<Arc<dyn Tool>>,

    /// A list of other agents that this agent can hand off control to. This
    /// enables the creation of multi-agent workflows.
    pub handoffs: Vec<Handoff>,

    /// A set of guardrails to validate user input before it is processed by
    /// the agent. Guardrails can enforce constraints or prevent malicious input.
    pub input_guardrails: Vec<Arc<dyn InputGuardrail>>,

    /// A set of guardrails to validate the agent's output before it is sent
    /// to the user. This can be used for content moderation or to ensure
    /// the output follows a specific format.
    pub output_guardrails: Vec<Arc<dyn OutputGuardrail>>,

    /// The name of the LLM model to use for generating responses (e.g., "gpt-4",
    /// "gpt-3.5-turbo"). The availability of models depends on the LLM provider.
    pub model: String,

    /// The maximum number of turns (user messages and agent responses) in a
    /// conversation before the run is stopped. This prevents infinite loops.
    pub max_turns: Option<usize>,

    /// The temperature for the LLM's response generation, controlling the
    /// randomness of the output. Higher values (e.g., 0.8) make the output
    /// more random, while lower values (e.g., 0.2) make it more deterministic.
    pub temperature: Option<f32>,

    /// The maximum number of tokens to generate in a single response from the
    /// LLM. If `None`, the model's default limit will be used.
    pub max_tokens: Option<u32>,

    /// An optional JSON schema to enforce structured output from the agent.
    /// When provided, the agent will attempt to generate a response that
    /// conforms to this schema.
    pub output_schema: Option<serde_json::Value>,

    /// Optional dynamic agent-scope policy layers applied around the tool stack for this agent.
    pub agent_layers: Vec<Arc<dyn ErasedToolLayer>>,
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
            model: "gpt-5".to_string(),
            max_turns: Some(10),
            temperature: Some(1.0),
            max_tokens: None,
            output_schema: None,
            agent_layers: Vec::new(),
        }
    }
}

/// Represents an AI agent that can process messages, use tools, and interact
/// in a multi-agent workflow.
///
/// The `Agent` struct is the central component of the SDK. It encapsulates the
/// agent's configuration and provides a builder-style interface for easy
/// construction. Agents are designed to be cloned and shared across tasks.
///
/// ## Example: Using the Builder Pattern
///
/// ```rust
/// use openai_agents_rs::{Agent, tool::FunctionTool};
/// use std::sync::Arc;
///
/// // A simple function to be used as a tool.
/// fn get_weather(location: String) -> String {
///     if location.to_lowercase().contains("san francisco") {
///         "The weather in San Francisco is 70°F and sunny.".to_string()
///     } else {
///         format!("I don't have the weather for {}.", location)
///     }
/// }
///
/// // Create a tool from the function.
/// let weather_tool = Arc::new(FunctionTool::simple(
///     "get_weather",
///     "Gets the current weather for a specified location.",
///     get_weather,
/// ));
///
/// // Build an agent with specific configurations.
/// let weather_agent = Agent::simple("WeatherBot", "I provide weather updates.")
///     .with_model("gpt-3.5-turbo")
///     .with_tool(weather_tool)
///     .with_temperature(0.5);
///
/// assert_eq!(weather_agent.config.model, "gpt-3.5-turbo");
/// assert_eq!(weather_agent.config.temperature, Some(0.5));
/// assert_eq!(weather_agent.tools().len(), 1);
/// ```
#[derive(Clone)]
pub struct Agent {
    /// The configuration that defines the agent's behavior and capabilities.
    pub config: AgentConfig,
}

impl Agent {
    /// Creates a new agent with the given configuration.
    ///
    /// This is the primary constructor for creating an `Agent`. It takes an
    /// [`AgentConfig`] struct that specifies all the necessary parameters.
    pub fn new(config: AgentConfig) -> Self {
        Self { config }
    }

    /// Creates a simple agent with just a name and instructions.
    ///
    /// This is a convenience method for creating a basic agent without needing to
    /// configure all the options in [`AgentConfig`]. The other settings will use
    /// their default values.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the agent.
    /// * `instructions` - The system instructions for the agent.
    pub fn simple(name: impl Into<String>, instructions: impl Into<String>) -> Self {
        Self::new(AgentConfig {
            name: name.into(),
            instructions: instructions.into(),
            ..Default::default()
        })
    }

    /// Sets the model for the agent.
    ///
    /// This method is part of the builder pattern and allows you to chain calls
    /// to configure the agent.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.config.model = model.into();
        self
    }

    /// Adds a tool to the agent.
    ///
    /// This method is part of the builder pattern. Tools are capabilities that
    /// the agent can use to interact with the outside world.
    pub fn with_tool(mut self, tool: Arc<dyn Tool>) -> Self {
        self.config.tools.push(tool);
        self
    }

    /// Attach dynamic agent-scope policy layers.
    /// 
    /// **Deprecated**: Use `.layer()` instead for type-safe composition.
    /// 
    /// # Migration
    /// ```rust,ignore
    /// // Old: 
    /// agent.with_agent_layers(vec![layers::boxed_timeout_secs(30)])
    /// 
    /// // New:
    /// agent.layer(TimeoutLayer::secs(30))
    /// ```
    #[deprecated(since = "0.2.0", note = "Use `.layer()` for typed composition instead")]
    pub fn with_agent_layers(mut self, layers: Vec<Arc<dyn ErasedToolLayer>>) -> Self {
        self.config.agent_layers = layers;
        self
    }



    /// Adds multiple tools to the agent.
    ///
    /// This method is part of the builder pattern and is a convenient way to
    /// add a list of tools at once.
    pub fn with_tools(mut self, tools: Vec<Arc<dyn Tool>>) -> Self {
        self.config.tools.extend(tools);
        self
    }

    /// Adds a handoff target to the agent.
    ///
    /// This method is part of the builder pattern. Handoffs allow the agent to
    /// delegate tasks to other specialized agents.
    pub fn with_handoff(mut self, handoff: Handoff) -> Self {
        self.config.handoffs.push(handoff);
        self
    }

    /// Adds multiple handoff targets to the agent.
    ///
    /// This method is part of the builder pattern, allowing for the addition of
    /// a list of handoff targets.
    pub fn with_handoffs(mut self, handoffs: Vec<Handoff>) -> Self {
        self.config.handoffs.extend(handoffs);
        self
    }

    /// Sets the temperature for the agent's LLM.
    ///
    /// This method is part of the builder pattern. The temperature controls the
    /// randomness of the generated responses.
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.config.temperature = Some(temperature);
        self
    }

    /// Sets the maximum number of turns for a conversation.
    ///
    /// This method is part of the builder pattern and helps prevent excessively
    /// long or looping conversations.
    pub fn with_max_turns(mut self, max_turns: usize) -> Self {
        self.config.max_turns = Some(max_turns);
        self
    }

    /// Sets the maximum number of tokens for a single response.
    ///
    /// This method is part of the builder pattern. It limits the length of the
    /// generated text to control costs and response time.
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.config.max_tokens = Some(max_tokens);
        self
    }

    /// Adds an input guardrail to the agent.
    ///
    /// This method is part of the builder pattern. Input guardrails validate
    /// user input before it is processed.
    pub fn with_input_guardrail(mut self, guardrail: Arc<dyn InputGuardrail>) -> Self {
        self.config.input_guardrails.push(guardrail);
        self
    }

    /// Adds an output guardrail to the agent.
    ///
    /// This method is part of the builder pattern. Output guardrails validate
    /// the agent's response before it is sent to the user.
    pub fn with_output_guardrail(mut self, guardrail: Arc<dyn OutputGuardrail>) -> Self {
        self.config.output_guardrails.push(guardrail);
        self
    }

    /// Sets the output schema for the agent to enforce structured output.
    ///
    /// This method is part of the builder pattern. When a schema is provided,
    /// the agent will try to produce a JSON object that conforms to it.
    pub fn with_output_schema(mut self, schema: serde_json::Value) -> Self {
        self.config.output_schema = Some(schema);
        self
    }



    /// Returns the agent's name.
    pub fn name(&self) -> &str {
        &self.config.name
    }

    /// Returns the agent's instructions.
    pub fn instructions(&self) -> &str {
        &self.config.instructions
    }

    /// Returns a slice of the tools available to the agent.
    pub fn tools(&self) -> &[Arc<dyn Tool>] {
        &self.config.tools
    }

    /// Returns a slice of the handoff targets available to the agent.
    pub fn handoffs(&self) -> &[Handoff] {
        &self.config.handoffs
    }

    /// Checks if the agent has any tools.
    pub fn has_tools(&self) -> bool {
        !self.config.tools.is_empty()
    }

    /// Checks if the agent has any handoff targets.
    pub fn has_handoffs(&self) -> bool {
        !self.config.handoffs.is_empty()
    }

    /// Constructs the system message for the agent based on its configuration.
    ///
    /// The system message includes the agent's instructions, as well as the
    /// descriptions of its available tools and handoff targets. This message
    /// is used to prime the LLM with the agent's context and capabilities.
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
    /// Apply a typed layer to this agent, returning a typed wrapper.
    /// 
    /// This is the preferred API over `with_agent_layers()` as it provides
    /// compile-time type safety and follows Tower's fluent composition pattern.
    ///
    /// # Example
    /// ```rust,no_run
    /// use openai_agents_rs::{Agent, service::{TimeoutLayer, RetryLayer}};
    /// use std::time::Duration;
    /// 
    /// let agent = Agent::simple("Assistant", "Be helpful")
    ///     .layer(TimeoutLayer::from_duration(Duration::from_secs(30)))
    ///     .layer(RetryLayer::times(3));
    /// ```
    pub fn layer<L>(self, layer: L) -> LayeredAgent<L, Self> {
        LayeredAgent::new(layer, self)
    }
}

/// A typed wrapper for an agent with layers applied.
/// 
/// This follows Tower's `Layered` pattern, providing compile-time type safety
/// for layer composition while maintaining the agent interface.
#[derive(Clone)]
pub struct LayeredAgent<L, A> {
    layer: L,
    inner: A,
}

impl<L, A> LayeredAgent<L, A> {
    /// Create a new layered agent.
    pub fn new(layer: L, inner: A) -> Self {
        Self { layer, inner }
    }
    
    /// Apply another layer, returning a new typed wrapper.
    pub fn layer<L2>(self, layer: L2) -> LayeredAgent<L2, Self> {
        LayeredAgent::new(layer, self)
    }
}

impl<L, A> LayeredAgent<L, A>
where
    A: AgentLike + Clone,
{
    /// Get the underlying agent configuration.
    /// This allows the layered agent to be used anywhere an Agent is expected.
    pub fn inner_agent(&self) -> A {
        self.inner.clone()
    }
}

/// Trait to allow both Agent and LayeredAgent to be used interchangeably
pub trait AgentLike {
    fn config(&self) -> &AgentConfig;
    fn name(&self) -> &str;
    fn instructions(&self) -> &str;
    fn tools(&self) -> &[Arc<dyn Tool>];
    fn handoffs(&self) -> &[Handoff];
}

impl AgentLike for Agent {
    fn config(&self) -> &AgentConfig {
        &self.config
    }
    
    fn name(&self) -> &str {
        &self.config.name
    }
    
    fn instructions(&self) -> &str {
        &self.config.instructions
    }
    
    fn tools(&self) -> &[Arc<dyn Tool>] {
        &self.config.tools
    }
    
    fn handoffs(&self) -> &[Handoff] {
        &self.config.handoffs
    }
}

impl<L, A: AgentLike> AgentLike for LayeredAgent<L, A> {
    fn config(&self) -> &AgentConfig {
        self.inner.config()
    }
    
    fn name(&self) -> &str {
        self.inner.name()
    }
    
    fn instructions(&self) -> &str {
        self.inner.instructions()
    }
    
    fn tools(&self) -> &[Arc<dyn Tool>] {
        self.inner.tools()
    }
    
    fn handoffs(&self) -> &[Handoff] {
        self.inner.handoffs()
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
        assert_eq!(agent.config.model, "gpt-5");
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
        assert_eq!(config.model, "gpt-5");
        assert_eq!(config.max_turns, Some(10));
        assert_eq!(config.temperature, Some(1.0));
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
        assert!(debug_str.contains("gpt-5"));
        assert!(debug_str.contains("tools_count"));
    }

    #[test]
    fn test_agent_layer_chaining_compiles() {
        use crate::service::{TimeoutLayer, RetryLayer};
        use std::time::Duration;

        // Test that agent layer chaining compiles and can be chained
        let agent = Agent::simple("LayeredAgent", "Test layering")
            .layer(TimeoutLayer::from_duration(Duration::from_secs(30)))
            .layer(RetryLayer::times(3));
            
        // Verify the underlying agent is accessible
        assert_eq!(agent.name(), "LayeredAgent");
        assert_eq!(agent.instructions(), "Test layering");
        
        // Test double layering compiles
        let double_layered = agent
            .layer(TimeoutLayer::from_duration(Duration::from_secs(60)));
            
        assert_eq!(double_layered.name(), "LayeredAgent");
    }
}
