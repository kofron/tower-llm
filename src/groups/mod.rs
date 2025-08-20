//! Multi-agent orchestration and handoffs
//!
//! This module provides two key abstractions for agent coordination:
//!
//! ## AgentPicker - "WHO starts the conversation?"
//! - Routes initial messages to appropriate agents
//! - One-time decision at conversation start
//! - Based on message content, user context, etc.
//!
//! ## HandoffPolicy - "HOW do agents collaborate?"  
//! - Defines handoff tools and triggers
//! - Runtime decisions during conversation
//! - Supports explicit tools, sequential workflows, conditional logic
//!
//! ## Key Distinction
//! - **Picker**: Choose starting agent based on conversation context
//! - **Policy**: Define collaboration patterns between agents
//!
//! These work together but serve different purposes:
//! ```rust
//! let group = GroupBuilder::new()
//!     .picker(route_by_topic())           // WHO: Route by message topic
//!     .handoff_policy(explicit_handoffs()) // HOW: Agents use handoff tools
//!     .build();
//! ```
//!
//! What this module provides (spec)
//! - A Tower-native router between multiple agent services with explicit handoff events
//!
//! Exports
//! - Models
//!   - `AgentName` newtype
//!   - `PickRequest { messages, last_stop: AgentStopReason }`
//!   - `HandoffRequest` - request for agent handoff
//!   - `HandoffResponse` - result of handoff attempt
//! - Services
//!   - `GroupRouter: Service<RawChatRequest, Response=AgentRun>`
//!   - `AgentPicker: Service<PickRequest, Response=AgentName>`
//! - Layers
//!   - `HandoffLayer` that annotates runs with AgentStart/AgentEnd/Handoff events
//! - Traits
//!   - `HandoffPolicy` - defines handoff tools and runtime behavior
//! - Utils
//!   - `GroupBuilder` to assemble named `AgentSvc`s and a picker strategy
//!
//! Implementation strategy
//! - Use `tower::steer` or a small name→index map, routing to boxed `AgentSvc`s
//! - `AgentPicker` decides next agent based on the current transcript and stop reason
//! - `HandoffLayer` wraps the router to emit handoff events into the run
//!
//! Composition
//! - `GroupBuilder::new().agent("triage", a).agent("specialist", b).picker(p).build()`
//! - Can be wrapped by resilience/observability layers as needed
//!
//! Testing strategy
//! - Build two fake agents that return deterministic responses
//! - A picker that selects based on a message predicate
//! - Assert the handoff events sequence and final run aggregation

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};

use async_openai::types::{ChatCompletionTool, CreateChatCompletionRequest};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tower::{BoxError, Layer, Service, ServiceExt};

use crate::core::{AgentRun, AgentStopReason, AgentSvc, LoopState, StepOutcome, ToolInvocation, ToolOutput};

pub type AgentName = String;

#[derive(Debug, Clone)]
pub struct PickRequest {
    pub messages: Vec<async_openai::types::ChatCompletionRequestMessage>,
    pub last_stop: AgentStopReason,
}

pub trait AgentPicker: Service<PickRequest, Response = AgentName, Error = BoxError> {}
impl<T> AgentPicker for T where T: Service<PickRequest, Response = AgentName, Error = BoxError> {}

// ================================================================================================
// Handoff Types
// ================================================================================================

/// Request to handoff conversation to another agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandoffRequest {
    /// Target agent to handoff to
    pub target_agent: String,
    /// Optional context data to pass to target agent
    pub context: Option<Value>,
    /// Optional reason for the handoff
    pub reason: Option<String>,
}

/// Response from a handoff attempt.
#[derive(Debug, Clone)]
pub struct HandoffResponse {
    /// Whether the handoff was successful
    pub success: bool,
    /// The target agent (for confirmation)
    pub target_agent: String,
    /// Any context returned from the handoff
    pub context: Option<Value>,
}

/// Outcome of group coordination - either continue, handoff, or finish.
#[derive(Debug, Clone)]
pub enum GroupOutcome {
    /// Continue with current agent
    Continue(AgentRun),
    /// Handoff to another agent
    Handoff(HandoffRequest),
    /// Conversation is complete
    Done(AgentRun),
}

/// Trait defining handoff policies - how agents collaborate during execution.
///
/// This trait separates handoff behavior from initial agent routing:
/// - `AgentPicker`: WHO starts the conversation (initial routing)
/// - `HandoffPolicy`: HOW agents collaborate during conversation (runtime handoffs)
pub trait HandoffPolicy: Send + Sync + 'static {
    /// Generate handoff tools that the LLM can call.
    /// These tools will be injected into the agent's available tools.
    fn handoff_tools(&self) -> Vec<ChatCompletionTool>;
    
    /// Handle a handoff tool call by converting it to a HandoffRequest.
    /// This is called when the LLM invokes one of the handoff tools.
    fn handle_handoff_tool(&self, invocation: &ToolInvocation) -> Result<HandoffRequest, BoxError>;
    
    /// Make runtime handoff decisions based on agent state and step outcome.
    /// This allows for automatic handoffs based on conditions (e.g., no tools called).
    fn should_handoff(&self, state: &LoopState, outcome: &StepOutcome) -> Option<HandoffRequest>;
    
    /// Check if a tool call is a handoff tool managed by this policy.
    fn is_handoff_tool(&self, tool_name: &str) -> bool;
}

pub struct GroupBuilder<P> {
    agents: HashMap<AgentName, AgentSvc>,
    picker: Option<P>,
}

impl<P> GroupBuilder<P> {
    pub fn new() -> Self {
        Self {
            agents: HashMap::new(),
            picker: None,
        }
    }
    pub fn agent(mut self, name: impl Into<String>, svc: AgentSvc) -> Self {
        self.agents.insert(name.into(), svc);
        self
    }
    pub fn picker(mut self, p: P) -> Self {
        self.picker = Some(p);
        self
    }
    pub fn build(self) -> GroupRouter<P> {
        GroupRouter {
            agents: std::sync::Arc::new(tokio::sync::Mutex::new(self.agents)),
            picker: self.picker.expect("picker"),
        }
    }
}

pub struct GroupRouter<P> {
    agents: std::sync::Arc<tokio::sync::Mutex<HashMap<AgentName, AgentSvc>>>,
    picker: P,
}

impl<P> Service<CreateChatCompletionRequest> for GroupRouter<P>
where
    P: AgentPicker + Clone + Send + 'static,
    P::Future: Send + 'static,
{
    type Response = AgentRun;
    type Error = BoxError;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(
        &mut self,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        std::task::Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: CreateChatCompletionRequest) -> Self::Future {
        let mut picker = self.picker.clone();
        let agents = self.agents.clone();
        Box::pin(async move {
            let pick = ServiceExt::ready(&mut picker)
                .await?
                .call(PickRequest {
                    messages: req.messages.clone(),
                    last_stop: AgentStopReason::DoneNoToolCalls,
                })
                .await?;
            let mut guard = agents.lock().await;
            let agent = guard
                .get_mut(&pick)
                .ok_or_else(|| format!("unknown agent: {}", pick))?;
            let run = ServiceExt::ready(agent).await?.call(req).await?;
            Ok(run)
        })
    }
}

// ================================================================================================
// Handoff Policy Implementations
// ================================================================================================

/// Explicit handoff policy - generates a handoff tool for specific target agent.
/// The LLM can call this tool to explicitly handoff to the target agent.
#[derive(Debug, Clone)]
pub struct ExplicitHandoffPolicy {
    target_agent: String,
    tool_name: Option<String>,
    description: Option<String>,
}

impl ExplicitHandoffPolicy {
    /// Create new explicit handoff policy for target agent.
    pub fn new(target_agent: impl Into<String>) -> Self {
        Self {
            target_agent: target_agent.into(),
            tool_name: None,
            description: None,
        }
    }
    
    /// Set custom tool name (default: "handoff_to_{target}")
    pub fn with_tool_name(mut self, name: impl Into<String>) -> Self {
        self.tool_name = Some(name.into());
        self
    }
    
    /// Set custom tool description
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }
    
    fn tool_name(&self) -> String {
        self.tool_name.as_ref()
            .map(|n| n.clone())
            .unwrap_or_else(|| format!("handoff_to_{}", self.target_agent))
    }
}

impl HandoffPolicy for ExplicitHandoffPolicy {
    fn handoff_tools(&self) -> Vec<ChatCompletionTool> {
        let tool_name = self.tool_name();
        let description = self.description.as_ref()
            .map(|d| d.clone())
            .unwrap_or_else(|| format!("Hand off the conversation to {}", self.target_agent));
        
        vec![ChatCompletionTool {
            r#type: async_openai::types::ChatCompletionToolType::Function,
            function: async_openai::types::FunctionObject {
                name: tool_name,
                description: Some(description),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "reason": {
                            "type": "string",
                            "description": "Reason for the handoff"
                        },
                        "context": {
                            "type": "object", 
                            "description": "Optional context to pass to the target agent"
                        }
                    }
                })),
                ..Default::default()
            }
        }]
    }
    
    fn handle_handoff_tool(&self, invocation: &ToolInvocation) -> Result<HandoffRequest, BoxError> {
        if !self.is_handoff_tool(&invocation.name) {
            return Err(format!("Not a handoff tool: {}", invocation.name).into());
        }
        
        let reason = invocation.arguments.get("reason")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
            
        let context = invocation.arguments.get("context").cloned();
        
        Ok(HandoffRequest {
            target_agent: self.target_agent.clone(),
            context,
            reason,
        })
    }
    
    fn should_handoff(&self, _state: &LoopState, _outcome: &StepOutcome) -> Option<HandoffRequest> {
        // Explicit handoffs only trigger via tool calls, not automatically
        None
    }
    
    fn is_handoff_tool(&self, tool_name: &str) -> bool {
        tool_name == self.tool_name()
    }
}

/// Sequential handoff policy - automatically hands off to the next agent in sequence
/// when the current agent completes without tool calls.
#[derive(Debug, Clone)]
pub struct SequentialHandoffPolicy {
    agents: Vec<String>,
    current_index: Arc<AtomicUsize>,
}

impl SequentialHandoffPolicy {
    /// Create new sequential handoff policy with agent sequence.
    pub fn new(agents: Vec<String>) -> Self {
        Self {
            agents,
            current_index: Arc::new(AtomicUsize::new(0)),
        }
    }
    
    fn next_agent(&self) -> Option<String> {
        let current = self.current_index.fetch_add(1, Ordering::SeqCst);
        if current + 1 < self.agents.len() {
            Some(self.agents[current + 1].clone())
        } else {
            None
        }
    }
}

impl HandoffPolicy for SequentialHandoffPolicy {
    fn handoff_tools(&self) -> Vec<ChatCompletionTool> {
        // Sequential handoffs are automatic, no tools needed
        vec![]
    }
    
    fn handle_handoff_tool(&self, invocation: &ToolInvocation) -> Result<HandoffRequest, BoxError> {
        Err(format!("Sequential policy has no handoff tools: {}", invocation.name).into())
    }
    
    fn should_handoff(&self, _state: &LoopState, outcome: &StepOutcome) -> Option<HandoffRequest> {
        match outcome {
            StepOutcome::Done { .. } => {
                // When agent completes without tool calls, move to next agent
                self.next_agent().map(|target| HandoffRequest {
                    target_agent: target,
                    context: None,
                    reason: Some("Sequential workflow step complete".to_string()),
                })
            }
            _ => None,
        }
    }
    
    fn is_handoff_tool(&self, _tool_name: &str) -> bool {
        false // No handoff tools for sequential policy
    }
}

/// Enum for composing different handoff policies.
#[derive(Debug, Clone)]
pub enum AnyHandoffPolicy {
    Explicit(ExplicitHandoffPolicy),
    Sequential(SequentialHandoffPolicy),
    Composite(CompositeHandoffPolicy),
}

impl From<ExplicitHandoffPolicy> for AnyHandoffPolicy {
    fn from(policy: ExplicitHandoffPolicy) -> Self {
        AnyHandoffPolicy::Explicit(policy)
    }
}

impl From<SequentialHandoffPolicy> for AnyHandoffPolicy {
    fn from(policy: SequentialHandoffPolicy) -> Self {
        AnyHandoffPolicy::Sequential(policy)
    }
}

impl From<CompositeHandoffPolicy> for AnyHandoffPolicy {
    fn from(policy: CompositeHandoffPolicy) -> Self {
        AnyHandoffPolicy::Composite(policy)
    }
}

impl HandoffPolicy for AnyHandoffPolicy {
    fn handoff_tools(&self) -> Vec<ChatCompletionTool> {
        match self {
            AnyHandoffPolicy::Explicit(p) => p.handoff_tools(),
            AnyHandoffPolicy::Sequential(p) => p.handoff_tools(),
            AnyHandoffPolicy::Composite(p) => p.handoff_tools(),
        }
    }
    
    fn handle_handoff_tool(&self, invocation: &ToolInvocation) -> Result<HandoffRequest, BoxError> {
        match self {
            AnyHandoffPolicy::Explicit(p) => p.handle_handoff_tool(invocation),
            AnyHandoffPolicy::Sequential(p) => p.handle_handoff_tool(invocation),
            AnyHandoffPolicy::Composite(p) => p.handle_handoff_tool(invocation),
        }
    }
    
    fn should_handoff(&self, state: &LoopState, outcome: &StepOutcome) -> Option<HandoffRequest> {
        match self {
            AnyHandoffPolicy::Explicit(p) => p.should_handoff(state, outcome),
            AnyHandoffPolicy::Sequential(p) => p.should_handoff(state, outcome),
            AnyHandoffPolicy::Composite(p) => p.should_handoff(state, outcome),
        }
    }
    
    fn is_handoff_tool(&self, tool_name: &str) -> bool {
        match self {
            AnyHandoffPolicy::Explicit(p) => p.is_handoff_tool(tool_name),
            AnyHandoffPolicy::Sequential(p) => p.is_handoff_tool(tool_name),
            AnyHandoffPolicy::Composite(p) => p.is_handoff_tool(tool_name),
        }
    }
}

/// Composite handoff policy - combines multiple handoff policies.
#[derive(Debug, Clone)]
pub struct CompositeHandoffPolicy {
    policies: Vec<AnyHandoffPolicy>,
}

impl CompositeHandoffPolicy {
    /// Create new composite policy from a list of policies.
    pub fn new(policies: Vec<AnyHandoffPolicy>) -> Self {
        Self { policies }
    }
}

impl HandoffPolicy for CompositeHandoffPolicy {
    fn handoff_tools(&self) -> Vec<ChatCompletionTool> {
        self.policies
            .iter()
            .flat_map(|p| p.handoff_tools())
            .collect()
    }
    
    fn handle_handoff_tool(&self, invocation: &ToolInvocation) -> Result<HandoffRequest, BoxError> {
        for policy in &self.policies {
            if policy.is_handoff_tool(&invocation.name) {
                return policy.handle_handoff_tool(invocation);
            }
        }
        Err(format!("No policy handles handoff tool: {}", invocation.name).into())
    }
    
    fn should_handoff(&self, state: &LoopState, outcome: &StepOutcome) -> Option<HandoffRequest> {
        // Return first handoff decision from any policy
        for policy in &self.policies {
            if let Some(handoff) = policy.should_handoff(state, outcome) {
                return Some(handoff);
            }
        }
        None
    }
    
    fn is_handoff_tool(&self, tool_name: &str) -> bool {
        self.policies.iter().any(|p| p.is_handoff_tool(tool_name))
    }
}

// ================================================================================================
// Convenience Constructors
// ================================================================================================

/// Create an explicit handoff policy for a target agent.
/// 
/// Example:
/// ```rust
/// let policy = explicit_handoff_to("specialist")
///     .with_description("Escalate complex issues to specialist");
/// ```
pub fn explicit_handoff_to(target: impl Into<String>) -> ExplicitHandoffPolicy {
    ExplicitHandoffPolicy::new(target)
}

/// Create a sequential handoff policy with agent sequence.
/// 
/// Example:
/// ```rust  
/// let policy = sequential_handoff(vec!["researcher", "writer", "reviewer"]);
/// ```
pub fn sequential_handoff(agents: Vec<String>) -> SequentialHandoffPolicy {
    SequentialHandoffPolicy::new(agents)
}

/// Create a composite handoff policy combining multiple policies.
/// 
/// Example:
/// ```rust
/// let policy = composite_handoff(vec![
///     AnyHandoffPolicy::Explicit(explicit_handoff_to("specialist")),
///     AnyHandoffPolicy::Sequential(sequential_handoff(vec!["a".to_string(), "b".to_string()])),
/// ]);
/// ```
pub fn composite_handoff(policies: Vec<AnyHandoffPolicy>) -> CompositeHandoffPolicy {
    CompositeHandoffPolicy::new(policies)
}

// ================================================================================================
// Handoff Layer - Tower Integration
// ================================================================================================

/// Enhanced ToolOutput that can signal handoff requests.
#[derive(Debug, Clone)]
pub enum ToolOutputResult {
    /// Regular tool output
    Tool(ToolOutput),
    /// Handoff request from a handoff tool
    Handoff(HandoffRequest),
}

impl From<ToolOutput> for ToolOutputResult {
    fn from(output: ToolOutput) -> Self {
        ToolOutputResult::Tool(output)
    }
}

/// Layer that adds handoff capabilities to tool services.
/// 
/// This layer:
/// 1. Wraps an existing tool service
/// 2. Adds handoff tools from the policy to the available tools
/// 3. Intercepts handoff tool calls and converts them to HandoffRequest
/// 4. Passes through regular tool calls unchanged
#[derive(Debug, Clone)]
pub struct HandoffLayer<P> {
    handoff_policy: P,
}

impl<P> HandoffLayer<P>
where
    P: HandoffPolicy,
{
    /// Create a new HandoffLayer with the given handoff policy.
    pub fn new(policy: P) -> Self {
        Self {
            handoff_policy: policy,
        }
    }
    
    /// Get the handoff tools that this layer will inject.
    pub fn handoff_tools(&self) -> Vec<ChatCompletionTool> {
        self.handoff_policy.handoff_tools()
    }
}

impl<S, P> Layer<S> for HandoffLayer<P>
where
    P: HandoffPolicy + Clone,
{
    type Service = HandoffService<S, P>;
    
    fn layer(&self, inner: S) -> Self::Service {
        HandoffService::new(inner, self.handoff_policy.clone())
    }
}

/// Service that wraps a tool service and adds handoff capabilities.
/// 
/// This service handles both regular tool calls and handoff tool calls:
/// - Regular tools: Pass through to inner service, return ToolOutputResult::Tool
/// - Handoff tools: Process with policy, return ToolOutputResult::Handoff
#[derive(Debug, Clone)]  
pub struct HandoffService<S, P> {
    inner: S,
    handoff_policy: P,
}

impl<S, P> HandoffService<S, P>
where
    P: HandoffPolicy,
{
    /// Create a new HandoffService wrapping the inner service.
    pub fn new(inner: S, policy: P) -> Self {
        Self {
            inner,
            handoff_policy: policy,
        }
    }
}

impl<S, P> Service<ToolInvocation> for HandoffService<S, P>
where
    S: Service<ToolInvocation, Response = ToolOutput, Error = BoxError> + Send + 'static,
    S::Future: Send + 'static,
    P: HandoffPolicy + Clone,
{
    type Response = ToolOutputResult;
    type Error = BoxError;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(
        &mut self,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: ToolInvocation) -> Self::Future {
        // Check if this is a handoff tool call
        if self.handoff_policy.is_handoff_tool(&req.name) {
            // Handle handoff tool call
            let policy = self.handoff_policy.clone();
            Box::pin(async move {
                let handoff_request = policy.handle_handoff_tool(&req)?;
                Ok(ToolOutputResult::Handoff(handoff_request))
            })
        } else {
            // Regular tool call - pass through to inner service
            let future = self.inner.call(req);
            Box::pin(async move {
                let output = future.await?;
                Ok(ToolOutputResult::Tool(output))
            })
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use async_openai::types::{
        CreateChatCompletionRequest, CreateChatCompletionRequestArgs,
    };
    use tower::util::BoxService;

    // ================================================================================================
    // Handoff Policy Tests
    // ================================================================================================
    
    mod handoff_policy_tests {
        use super::*;
        
        #[test]
        fn explicit_handoff_policy_generates_correct_tools() {
            let policy = explicit_handoff_to("specialist")
                .with_description("Escalate to specialist");
            
            let tools = policy.handoff_tools();
            assert_eq!(tools.len(), 1);
            
            let tool = &tools[0];
            assert_eq!(tool.function.name, "handoff_to_specialist");
            assert_eq!(tool.function.description, Some("Escalate to specialist".to_string()));
            
            // Verify tool parameters schema
            let params = tool.function.parameters.as_ref().unwrap();
            assert!(params.get("properties").is_some());
            assert!(params["properties"]["reason"].is_object());
        }
        
        #[test] 
        fn explicit_handoff_policy_handles_tool_calls() {
            let policy = explicit_handoff_to("specialist");
            
            let invocation = ToolInvocation {
                id: "test_id".to_string(),
                name: "handoff_to_specialist".to_string(),
                arguments: serde_json::json!({
                    "reason": "Complex technical issue",
                    "context": {"priority": "high"}
                }),
            };
            
            let result = policy.handle_handoff_tool(&invocation).unwrap();
            assert_eq!(result.target_agent, "specialist");
            assert_eq!(result.reason, Some("Complex technical issue".to_string()));
            assert!(result.context.is_some());
        }
        
        #[test]
        fn explicit_handoff_policy_rejects_wrong_tools() {
            let policy = explicit_handoff_to("specialist");
            
            let invocation = ToolInvocation {
                id: "test_id".to_string(),
                name: "some_other_tool".to_string(),
                arguments: serde_json::json!({}),
            };
            
            let result = policy.handle_handoff_tool(&invocation);
            assert!(result.is_err());
            assert!(result.unwrap_err().to_string().contains("Not a handoff tool"));
        }
        
        #[test]
        fn explicit_handoff_policy_no_automatic_handoffs() {
            let policy = explicit_handoff_to("specialist");
            let state = LoopState { steps: 1 };
            let outcome = StepOutcome::Done {
                messages: vec![],
                aux: crate::core::StepAux::default(),
            };
            
            // Explicit policies don't trigger automatic handoffs
            assert!(policy.should_handoff(&state, &outcome).is_none());
        }
        
        #[test]
        fn sequential_handoff_policy_advances_correctly() {
            let agents = vec!["a".to_string(), "b".to_string(), "c".to_string()];
            let policy = sequential_handoff(agents.clone());
            
            let state = LoopState { steps: 1 };
            let outcome = StepOutcome::Done {
                messages: vec![],
                aux: crate::core::StepAux::default(),
            };
            
            // First handoff: a -> b
            let handoff1 = policy.should_handoff(&state, &outcome).unwrap();
            assert_eq!(handoff1.target_agent, "b");
            assert!(handoff1.reason.is_some());
            
            // Second handoff: b -> c  
            let handoff2 = policy.should_handoff(&state, &outcome).unwrap();
            assert_eq!(handoff2.target_agent, "c");
            
            // Third call: no more agents
            let handoff3 = policy.should_handoff(&state, &outcome);
            assert!(handoff3.is_none());
        }
        
        #[test]
        fn sequential_handoff_policy_no_tools() {
            let policy = sequential_handoff(vec!["a".to_string(), "b".to_string()]);
            
            // Sequential policies don't generate tools
            assert!(policy.handoff_tools().is_empty());
            assert!(!policy.is_handoff_tool("any_tool"));
        }
        
        #[test]
        fn composite_handoff_policy_combines_tools() {
            let explicit1 = explicit_handoff_to("specialist");
            let explicit2 = explicit_handoff_to("supervisor");
            let sequential = sequential_handoff(vec!["a".to_string(), "b".to_string()]);
            
            let composite = composite_handoff(vec![
                AnyHandoffPolicy::Explicit(explicit1),
                AnyHandoffPolicy::Explicit(explicit2),
                AnyHandoffPolicy::Sequential(sequential),
            ]);
            
            let tools = composite.handoff_tools();
            // Should have 2 tools (from explicit policies), sequential has none
            assert_eq!(tools.len(), 2);
            
            let tool_names: Vec<&str> = tools.iter()
                .map(|t| t.function.name.as_str())
                .collect();
            assert!(tool_names.contains(&"handoff_to_specialist"));
            assert!(tool_names.contains(&"handoff_to_supervisor"));
        }
        
        #[test]
        fn composite_handoff_policy_routes_to_correct_handler() {
            let explicit = explicit_handoff_to("specialist");
            let sequential = sequential_handoff(vec!["a".to_string()]);
            
            let composite = composite_handoff(vec![
                AnyHandoffPolicy::Explicit(explicit),
                AnyHandoffPolicy::Sequential(sequential),
            ]);
            
            let invocation = ToolInvocation {
                id: "test_id".to_string(),
                name: "handoff_to_specialist".to_string(),
                arguments: serde_json::json!({"reason": "test"}),
            };
            
            let result = composite.handle_handoff_tool(&invocation).unwrap();
            assert_eq!(result.target_agent, "specialist");
        }
        
        #[test]
        fn composite_handoff_policy_first_match_wins() {
            let explicit = explicit_handoff_to("specialist");
            let sequential = sequential_handoff(vec!["a".to_string(), "b".to_string()]);
            
            let composite = composite_handoff(vec![
                AnyHandoffPolicy::Explicit(explicit),
                AnyHandoffPolicy::Sequential(sequential),
            ]);
            
            let state = LoopState { steps: 1 };
            let outcome = StepOutcome::Done {
                messages: vec![],
                aux: crate::core::StepAux::default(),
            };
            
            // Explicit policy returns None, sequential returns Some
            // Should get sequential result since explicit is first but returns None
            let result = composite.should_handoff(&state, &outcome).unwrap();
            assert_eq!(result.target_agent, "b"); // First handoff in sequential
        }
        
        #[test]
        fn any_handoff_policy_conversions_work() {
            let explicit = explicit_handoff_to("specialist");
            let sequential = sequential_handoff(vec!["a".to_string()]);
            
            // Test From impls
            let _any1: AnyHandoffPolicy = explicit.into();
            let _any2: AnyHandoffPolicy = sequential.into();
            
            // Should compile and work
            assert!(true);
        }
        
        #[test]
        fn handoff_request_serialization() {
            let request = HandoffRequest {
                target_agent: "specialist".to_string(),
                context: Some(serde_json::json!({"priority": "high"})),
                reason: Some("Complex issue".to_string()),
            };
            
            // Should serialize/deserialize correctly
            let json = serde_json::to_string(&request).unwrap();
            let deserialized: HandoffRequest = serde_json::from_str(&json).unwrap();
            
            assert_eq!(deserialized.target_agent, request.target_agent);
            assert_eq!(deserialized.reason, request.reason);
            assert_eq!(deserialized.context, request.context);
        }
    }

    // ================================================================================================
    // Handoff Layer Integration Tests  
    // ================================================================================================
    
    mod handoff_layer_tests {
        use super::*;
        use tower::{service_fn, ServiceExt};
        
        // Mock tool service for testing
        fn mock_tool_service() -> tower::util::BoxService<ToolInvocation, ToolOutput, BoxError> {
            tower::util::BoxService::new(service_fn(|req: ToolInvocation| async move {
                Ok(ToolOutput {
                    id: req.id,
                    result: serde_json::json!({"status": "success", "tool": req.name}),
                })
            }))
        }
        
        #[tokio::test]
        async fn handoff_layer_passes_through_regular_tools() {
            let policy = explicit_handoff_to("specialist");
            let layer = HandoffLayer::new(policy);
            let mut service = layer.layer(mock_tool_service());
            
            let invocation = ToolInvocation {
                id: "test_id".to_string(),
                name: "regular_tool".to_string(),
                arguments: serde_json::json!({"param": "value"}),
            };
            
            let result = ServiceExt::ready(&mut service).await.unwrap()
                .call(invocation).await.unwrap();
            
            match result {
                ToolOutputResult::Tool(output) => {
                    assert_eq!(output.id, "test_id");
                    assert_eq!(output.result["tool"], "regular_tool");
                }
                ToolOutputResult::Handoff(_) => panic!("Expected tool output, got handoff"),
            }
        }
        
        #[tokio::test]
        async fn handoff_layer_intercepts_handoff_tools() {
            let policy = explicit_handoff_to("specialist");
            let layer = HandoffLayer::new(policy);
            let mut service = layer.layer(mock_tool_service());
            
            let invocation = ToolInvocation {
                id: "test_id".to_string(),
                name: "handoff_to_specialist".to_string(),
                arguments: serde_json::json!({"reason": "Complex issue"}),
            };
            
            let result = ServiceExt::ready(&mut service).await.unwrap()
                .call(invocation).await.unwrap();
            
            match result {
                ToolOutputResult::Handoff(handoff) => {
                    assert_eq!(handoff.target_agent, "specialist");
                    assert_eq!(handoff.reason, Some("Complex issue".to_string()));
                }
                ToolOutputResult::Tool(_) => panic!("Expected handoff, got tool output"),
            }
        }
        
        #[tokio::test]
        async fn handoff_layer_exposes_policy_tools() {
            let policy = explicit_handoff_to("specialist")
                .with_description("Escalate to specialist");
            let layer = HandoffLayer::new(policy);
            
            let tools = layer.handoff_tools();
            assert_eq!(tools.len(), 1);
            assert_eq!(tools[0].function.name, "handoff_to_specialist");
            assert_eq!(tools[0].function.description, Some("Escalate to specialist".to_string()));
        }
        
        #[tokio::test]
        async fn handoff_layer_with_composite_policy() {
            let composite = composite_handoff(vec![
                AnyHandoffPolicy::Explicit(explicit_handoff_to("specialist")),
                AnyHandoffPolicy::Explicit(explicit_handoff_to("supervisor")),
            ]);
            
            let layer = HandoffLayer::new(composite.clone());
            let mut service = layer.layer(mock_tool_service());
            
            // Test first handoff tool
            let invocation1 = ToolInvocation {
                id: "test_id1".to_string(),
                name: "handoff_to_specialist".to_string(),
                arguments: serde_json::json!({"reason": "Technical issue"}),
            };
            
            let result1 = ServiceExt::ready(&mut service).await.unwrap()
                .call(invocation1).await.unwrap();
            
            match result1 {
                ToolOutputResult::Handoff(handoff) => {
                    assert_eq!(handoff.target_agent, "specialist");
                }
                _ => panic!("Expected handoff"),
            }
            
            // Test second handoff tool
            let invocation2 = ToolInvocation {
                id: "test_id2".to_string(), 
                name: "handoff_to_supervisor".to_string(),
                arguments: serde_json::json!({"reason": "Escalation needed"}),
            };
            
            let result2 = ServiceExt::ready(&mut service).await.unwrap()
                .call(invocation2).await.unwrap();
            
            match result2 {
                ToolOutputResult::Handoff(handoff) => {
                    assert_eq!(handoff.target_agent, "supervisor");
                }
                _ => panic!("Expected handoff"),
            }
            
            // Verify layer exposes both tools
            let tools = layer.handoff_tools();
            assert_eq!(tools.len(), 2);
            let tool_names: Vec<&str> = tools.iter().map(|t| t.function.name.as_str()).collect();
            assert!(tool_names.contains(&"handoff_to_specialist"));
            assert!(tool_names.contains(&"handoff_to_supervisor"));
        }
        
        #[tokio::test] 
        async fn handoff_layer_error_handling() {
            let policy = explicit_handoff_to("specialist");
            let layer = HandoffLayer::new(policy);
            let mut service = layer.layer(mock_tool_service());
            
            // Try to call handoff tool with invalid arguments
            let invocation = ToolInvocation {
                id: "test_id".to_string(),
                name: "handoff_to_specialist".to_string(),
                arguments: serde_json::json!({}), // Missing required fields - should still work
            };
            
            // Should succeed even with minimal arguments
            let result = ServiceExt::ready(&mut service).await.unwrap()
                .call(invocation).await.unwrap();
            
            match result {
                ToolOutputResult::Handoff(handoff) => {
                    assert_eq!(handoff.target_agent, "specialist");
                    assert_eq!(handoff.reason, None); // No reason provided
                }
                _ => panic!("Expected handoff"),
            }
        }
    }

    // ================================================================================================
    // Original Group Router Tests
    // ================================================================================================

    #[tokio::test]
    async fn routes_to_named_agent() {
        let a: AgentSvc = BoxService::new(tower::service_fn(
            |_r: CreateChatCompletionRequest| async move {
                Ok::<_, BoxError>(AgentRun {
                    messages: vec![],
                    steps: 1,
                    stop: AgentStopReason::DoneNoToolCalls,
                })
            },
        ));
        let picker =
            tower::service_fn(|_pr: PickRequest| async move { Ok::<_, BoxError>("a".to_string()) });
        let router = GroupBuilder::new().agent("a", a).picker(picker).build();
        let mut svc = router;
        let req = CreateChatCompletionRequestArgs::default()
            .model("gpt-4o")
            .messages(vec![])
            .build()
            .unwrap();
        let run = ServiceExt::ready(&mut svc)
            .await
            .unwrap()
            .call(req)
            .await
            .unwrap();
        assert_eq!(run.steps, 1);
    }
}
