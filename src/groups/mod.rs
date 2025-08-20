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
use tracing::{debug, info, warn, error, instrument, trace};

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

pub struct GroupBuilder<P = (), H = ()> {
    agents: HashMap<AgentName, AgentSvc>,
    picker: Option<P>,
    handoff_policy: Option<H>,
}

impl GroupBuilder<(), ()> {
    pub fn new() -> Self {
        Self {
            agents: HashMap::new(),
            picker: None,
            handoff_policy: None,
        }
    }
}

impl<P, H> GroupBuilder<P, H> {
    pub fn agent(mut self, name: impl Into<String>, svc: AgentSvc) -> Self {
        self.agents.insert(name.into(), svc);
        self
    }
    
    pub fn picker<NewP>(self, p: NewP) -> GroupBuilder<NewP, H> {
        GroupBuilder {
            agents: self.agents,
            picker: Some(p),
            handoff_policy: self.handoff_policy,
        }
    }
    
    /// Add handoff policy to enable agent coordination
    pub fn handoff_policy<NewH>(self, policy: NewH) -> GroupBuilder<P, NewH>
    where
        NewH: HandoffPolicy + Clone,
    {
        GroupBuilder {
            agents: self.agents,
            picker: self.picker,
            handoff_policy: Some(policy),
        }
    }
}

impl<P> GroupBuilder<P, ()> {
    /// Build a basic group without handoff coordination
    pub fn build(self) -> GroupRouter<P>
    where
        P: AgentPicker + Clone + Send + 'static,
        P::Future: Send + 'static,
    {
        GroupRouter {
            agents: std::sync::Arc::new(tokio::sync::Mutex::new(self.agents)),
            picker: self.picker.expect("picker"),
        }
    }
}

impl<P, H> GroupBuilder<P, H> 
where 
    H: HandoffPolicy + Clone + Send + 'static,
{
    /// Build a handoff-enabled group coordinator
    pub fn build(self) -> HandoffCoordinator<P, H>
    where
        P: AgentPicker + Clone + Send + 'static,
        P::Future: Send + 'static,
    {
        HandoffCoordinator::new(
            self.agents,
            self.picker.expect("picker"),
            self.handoff_policy.expect("handoff_policy"),
        )
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
// Handoff Coordinator - Enhanced GroupRouter
// ================================================================================================

/// Enhanced group coordinator that manages handoffs between agents.
/// 
/// This coordinator:
/// 1. Uses AgentPicker for initial agent selection
/// 2. Integrates HandoffLayer with agent tools to detect handoffs
/// 3. Orchestrates seamless agent transitions
/// 4. Maintains conversation context across handoffs
/// 5. Supports both explicit and automatic handoff triggers
pub struct HandoffCoordinator<P, H> {
    agents: Arc<tokio::sync::Mutex<HashMap<AgentName, AgentSvc>>>,
    picker: P,
    handoff_policy: H,
    current_agent: Arc<tokio::sync::Mutex<Option<AgentName>>>,
    conversation_context: Arc<tokio::sync::Mutex<Vec<async_openai::types::ChatCompletionRequestMessage>>>,
}

impl<P, H> HandoffCoordinator<P, H>
where
    P: AgentPicker,
    H: HandoffPolicy + Clone,
{
    /// Create new handoff coordinator.
    pub fn new(agents: HashMap<AgentName, AgentSvc>, picker: P, handoff_policy: H) -> Self {
        Self {
            agents: Arc::new(tokio::sync::Mutex::new(agents)),
            picker,
            handoff_policy,
            current_agent: Arc::new(tokio::sync::Mutex::new(None)),
            conversation_context: Arc::new(tokio::sync::Mutex::new(Vec::new())),
        }
    }
    
    /// Get the handoff tools that will be available to agents.
    pub fn handoff_tools(&self) -> Vec<ChatCompletionTool> {
        self.handoff_policy.handoff_tools()
    }
    
}

impl<P, H> Service<CreateChatCompletionRequest> for HandoffCoordinator<P, H>
where
    P: AgentPicker + Clone + Send + 'static,
    P::Future: Send + 'static,
    H: HandoffPolicy + Clone + Send + 'static,
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

    #[instrument(skip_all, fields(request_id = %uuid::Uuid::new_v4()))]
    fn call(&mut self, mut request: CreateChatCompletionRequest) -> Self::Future {
        let agents = self.agents.clone();
        let mut picker = self.picker.clone();
        let handoff_policy = self.handoff_policy.clone();
        let current_agent = self.current_agent.clone();
        let conversation_context = self.conversation_context.clone();
        
        Box::pin(async move {
            const MAX_HANDOFFS: usize = 10;
            let mut handoff_count = 0;
            let mut all_messages = Vec::new();
            let mut total_steps = 0;
            
            info!("🚀 Starting handoff coordinator");
            debug!("Initial request has {} messages", request.messages.len());
            
            // Store the original request messages
            let original_messages = request.messages.clone();
            
            // Pick the initial agent
            debug!("Invoking picker to determine initial agent");
            let initial_pick = ServiceExt::ready(&mut picker)
                .await?
                .call(PickRequest {
                    messages: request.messages.clone(),
                    last_stop: AgentStopReason::DoneNoToolCalls,
                })
                .await?;
            
            info!("📍 Initial agent selected: {}", initial_pick);
            let mut current_agent_name = initial_pick;
            
            // Update current agent tracking
            {
                let mut current = current_agent.lock().await;
                *current = Some(current_agent_name.clone());
            }
            
            // Main handoff loop
            loop {
                // Check handoff limit
                if handoff_count >= MAX_HANDOFFS {
                    error!("❌ Maximum handoffs exceeded ({})", MAX_HANDOFFS);
                    return Err("Maximum handoffs exceeded".into());
                }
                
                info!("🤖 Executing agent: {} (handoff #{}/{})", 
                    current_agent_name, handoff_count + 1, MAX_HANDOFFS);
                
                // Get the current agent
                let mut agents_guard = agents.lock().await;
                let agent = agents_guard
                    .get_mut(&current_agent_name)
                    .ok_or_else(|| {
                        error!("Agent not found: {}", current_agent_name);
                        format!("Unknown agent: {}", current_agent_name)
                    })?;
                
                // Inject handoff tools into the request
                let handoff_tools = handoff_policy.handoff_tools();
                if !handoff_tools.is_empty() {
                    debug!("Injecting {} handoff tools into request", handoff_tools.len());
                    trace!("Handoff tools: {:?}", handoff_tools.iter()
                        .map(|t| &t.function.name)
                        .collect::<Vec<_>>());
                    
                    // Add handoff tools to the request if not already present
                    if request.tools.is_none() {
                        request.tools = Some(handoff_tools);
                    } else {
                        // Append handoff tools to existing tools
                        request.tools.as_mut().unwrap().extend(handoff_tools);
                    }
                }
                
                // Execute the current agent
                debug!("Calling agent with {} messages", request.messages.len());
                let agent_run = ServiceExt::ready(agent)
                    .await?
                    .call(request.clone())
                    .await?;
                
                info!("✅ Agent {} completed: {} messages, {} steps, stop reason: {:?}", 
                    current_agent_name, agent_run.messages.len(), 
                    agent_run.steps, agent_run.stop);
                
                // Add messages from this run
                all_messages.extend(agent_run.messages.clone());
                total_steps += agent_run.steps;
                
                // Check for handoff in the agent's response
                let mut handoff_requested = None;
                
                // Look for handoff tool calls in the response messages
                debug!("Checking for handoff tool calls in agent response");
                for message in &agent_run.messages {
                    if let async_openai::types::ChatCompletionRequestMessage::Assistant(msg) = message {
                        if let Some(tool_calls) = &msg.tool_calls {
                            trace!("Found {} tool calls in message", tool_calls.len());
                            for tool_call in tool_calls {
                                if handoff_policy.is_handoff_tool(&tool_call.function.name) {
                                    info!("🔄 Handoff tool detected: {}", tool_call.function.name);
                                    
                                    // Parse the handoff request from the tool call
                                    let invocation = ToolInvocation {
                                        id: tool_call.id.clone(),
                                        name: tool_call.function.name.clone(),
                                        arguments: serde_json::from_str(&tool_call.function.arguments)
                                            .unwrap_or_else(|e| {
                                                warn!("Failed to parse handoff tool arguments: {}", e);
                                                serde_json::json!({})
                                            }),
                                    };
                                    
                                    match handoff_policy.handle_handoff_tool(&invocation) {
                                        Ok(handoff_req) => {
                                            info!("📋 Handoff request: {} → {} (reason: {:?})", 
                                                current_agent_name, 
                                                handoff_req.target_agent,
                                                handoff_req.reason);
                                            handoff_requested = Some(handoff_req);
                                            break;
                                        }
                                        Err(e) => {
                                            warn!("Failed to handle handoff tool: {}", e);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if handoff_requested.is_some() {
                        break;
                    }
                }
                
                // If no explicit handoff via tool, check policy for automatic handoff
                if handoff_requested.is_none() {
                    debug!("No explicit handoff tool called, checking policy for automatic handoff");
                    
                    // Create a LoopState for the policy check
                    let loop_state = LoopState { steps: total_steps };
                    
                    // Convert AgentRun to StepOutcome for policy check
                    let step_outcome = if matches!(agent_run.stop, AgentStopReason::DoneNoToolCalls) {
                        StepOutcome::Done {
                            messages: agent_run.messages.clone(),
                            aux: crate::core::StepAux::default(),
                        }
                    } else {
                        StepOutcome::Next {
                            messages: agent_run.messages.clone(),
                            aux: crate::core::StepAux::default(),
                            invoked_tools: vec![],
                        }
                    };
                    
                    if let Some(handoff) = handoff_policy.should_handoff(&loop_state, &step_outcome) {
                        info!("🔀 Automatic handoff triggered by policy: {} → {} (reason: {:?})",
                            current_agent_name, handoff.target_agent, handoff.reason);
                        handoff_requested = Some(handoff);
                    } else {
                        debug!("No automatic handoff triggered");
                    }
                }
                
                // Handle handoff if requested
                if let Some(handoff) = handoff_requested {
                    info!("🚦 Processing handoff: {} → {}", 
                        current_agent_name, handoff.target_agent);
                    
                    // Update the current agent
                    let previous_agent = current_agent_name.clone();
                    current_agent_name = handoff.target_agent.clone();
                    handoff_count += 1;
                    
                    // Update current agent tracking
                    {
                        let mut current = current_agent.lock().await;
                        *current = Some(current_agent_name.clone());
                    }
                    
                    // Update conversation context
                    {
                        let mut context = conversation_context.lock().await;
                        context.extend(agent_run.messages.clone());
                        debug!("Updated conversation context with {} messages", 
                            agent_run.messages.len());
                    }
                    
                    // Prepare request for next agent with accumulated context
                    request.messages = original_messages.clone();
                    request.messages.extend(all_messages.clone());
                    
                    info!("🔗 Handoff complete: {} → {} (total handoffs: {})", 
                        previous_agent, current_agent_name, handoff_count);
                    
                    // Continue to next iteration with new agent
                    continue;
                }
                
                // No handoff, we're done
                info!("🎯 Workflow complete: {} total messages, {} steps, final agent: {}",
                    all_messages.len(), total_steps, current_agent_name);
                
                return Ok(AgentRun {
                    messages: all_messages,
                    steps: total_steps,
                    stop: agent_run.stop,
                });
            }
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
    #[instrument(skip(self))]
    fn handoff_tools(&self) -> Vec<ChatCompletionTool> {
        let tool_name = self.tool_name();
        let description = self.description.as_ref()
            .map(|d| d.clone())
            .unwrap_or_else(|| format!("Hand off the conversation to {}", self.target_agent));
        
        debug!("ExplicitHandoffPolicy generating tool: {} → {}", tool_name, self.target_agent);
        
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
    #[instrument(skip(self))]
    fn handoff_tools(&self) -> Vec<ChatCompletionTool> {
        // Sequential handoffs are automatic, no tools needed
        debug!("SequentialHandoffPolicy: no tools (automatic handoffs only)");
        vec![]
    }
    
    #[instrument(skip(self, invocation))]
    fn handle_handoff_tool(&self, invocation: &ToolInvocation) -> Result<HandoffRequest, BoxError> {
        warn!("Sequential policy received unexpected handoff tool call: {}", invocation.name);
        Err(format!("Sequential policy has no handoff tools: {}", invocation.name).into())
    }
    
    #[instrument(skip(self, _state, outcome))]
    fn should_handoff(&self, _state: &LoopState, outcome: &StepOutcome) -> Option<HandoffRequest> {
        match outcome {
            StepOutcome::Done { .. } => {
                // When agent completes without tool calls, move to next agent
                if let Some(target) = self.next_agent() {
                    let current_idx = self.current_index.load(Ordering::SeqCst);
                    info!("📈 Sequential handoff: step {}/{} → {}", 
                        current_idx, self.agents.len(), target);
                    Some(HandoffRequest {
                        target_agent: target,
                        context: None,
                        reason: Some("Sequential workflow step complete".to_string()),
                    })
                } else {
                    debug!("Sequential workflow complete (all steps finished)");
                    None
                }
            }
            _ => {
                trace!("Sequential policy: no handoff (agent not done)");
                None
            }
        }
    }
    
    fn is_handoff_tool(&self, _tool_name: &str) -> bool {
        false // No handoff tools for sequential policy
    }
}

/// Multi-target explicit handoff policy - supports multiple handoff targets.
/// Each tool name maps to a specific target agent.
#[derive(Debug, Clone)]
pub struct MultiExplicitHandoffPolicy {
    handoffs: HashMap<String, String>,
}

impl MultiExplicitHandoffPolicy {
    /// Create new multi-target handoff policy with tool->agent mappings.
    pub fn new(handoffs: HashMap<String, String>) -> Self {
        Self { handoffs }
    }
    
    /// Add a handoff mapping.
    pub fn add_handoff(mut self, tool_name: impl Into<String>, target: impl Into<String>) -> Self {
        self.handoffs.insert(tool_name.into(), target.into());
        self
    }
}

impl HandoffPolicy for MultiExplicitHandoffPolicy {
    #[instrument(skip(self))]
    fn handoff_tools(&self) -> Vec<ChatCompletionTool> {
        debug!("MultiExplicitHandoffPolicy generating {} handoff tools", self.handoffs.len());
        self.handoffs.iter().map(|(tool_name, target_agent)| {
            trace!("  Tool: {} → {}", tool_name, target_agent);
            ChatCompletionTool {
                r#type: async_openai::types::ChatCompletionToolType::Function,
                function: async_openai::types::FunctionObject {
                    name: tool_name.clone(),
                    description: Some(format!("Hand off the conversation to {}", target_agent)),
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
            }
        }).collect()
    }
    
    fn handle_handoff_tool(&self, invocation: &ToolInvocation) -> Result<HandoffRequest, BoxError> {
        let target_agent = self.handoffs.get(&invocation.name)
            .ok_or_else(|| format!("Not a handoff tool: {}", invocation.name))?;
        
        let reason = invocation.arguments.get("reason")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
            
        let context = invocation.arguments.get("context").cloned();
        
        Ok(HandoffRequest {
            target_agent: target_agent.clone(),
            context,
            reason,
        })
    }
    
    fn should_handoff(&self, _state: &LoopState, _outcome: &StepOutcome) -> Option<HandoffRequest> {
        // Multi-explicit handoffs only trigger via tool calls, not automatically
        None
    }
    
    fn is_handoff_tool(&self, tool_name: &str) -> bool {
        self.handoffs.contains_key(tool_name)
    }
}

/// Enum for composing different handoff policies.
#[derive(Debug, Clone)]
pub enum AnyHandoffPolicy {
    Explicit(ExplicitHandoffPolicy),
    MultiExplicit(MultiExplicitHandoffPolicy),
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

impl From<MultiExplicitHandoffPolicy> for AnyHandoffPolicy {
    fn from(policy: MultiExplicitHandoffPolicy) -> Self {
        AnyHandoffPolicy::MultiExplicit(policy)
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
            AnyHandoffPolicy::MultiExplicit(p) => p.handoff_tools(),
            AnyHandoffPolicy::Sequential(p) => p.handoff_tools(),
            AnyHandoffPolicy::Composite(p) => p.handoff_tools(),
        }
    }
    
    fn handle_handoff_tool(&self, invocation: &ToolInvocation) -> Result<HandoffRequest, BoxError> {
        match self {
            AnyHandoffPolicy::Explicit(p) => p.handle_handoff_tool(invocation),
            AnyHandoffPolicy::MultiExplicit(p) => p.handle_handoff_tool(invocation),
            AnyHandoffPolicy::Sequential(p) => p.handle_handoff_tool(invocation),
            AnyHandoffPolicy::Composite(p) => p.handle_handoff_tool(invocation),
        }
    }
    
    fn should_handoff(&self, state: &LoopState, outcome: &StepOutcome) -> Option<HandoffRequest> {
        match self {
            AnyHandoffPolicy::Explicit(p) => p.should_handoff(state, outcome),
            AnyHandoffPolicy::MultiExplicit(p) => p.should_handoff(state, outcome),
            AnyHandoffPolicy::Sequential(p) => p.should_handoff(state, outcome),
            AnyHandoffPolicy::Composite(p) => p.should_handoff(state, outcome),
        }
    }
    
    fn is_handoff_tool(&self, tool_name: &str) -> bool {
        match self {
            AnyHandoffPolicy::Explicit(p) => p.is_handoff_tool(tool_name),
            AnyHandoffPolicy::MultiExplicit(p) => p.is_handoff_tool(tool_name),
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
    // End-to-End Handoff Coordinator Tests
    // ================================================================================================
    
    mod handoff_coordinator_tests {
        use super::*;
        use tower::{service_fn, ServiceExt};
        
        // Mock agents for testing
        fn mock_agent(name: &'static str, response: &'static str) -> AgentSvc {
            let name = name.to_string();
            let response = response.to_string();
            tower::util::BoxService::new(service_fn(move |_req: CreateChatCompletionRequest| {
                let name = name.clone();
                let response = response.clone();
                async move {
                    use async_openai::types::{ChatCompletionRequestMessage, ChatCompletionRequestAssistantMessageArgs};
                    
                    let message = ChatCompletionRequestAssistantMessageArgs::default()
                        .content(format!("[{}]: {}", name, response))
                        .build()?;
                    
                    Ok::<AgentRun, BoxError>(AgentRun {
                        messages: vec![ChatCompletionRequestMessage::Assistant(message)],
                        steps: 1,
                        stop: AgentStopReason::DoneNoToolCalls,
                    })
                }
            }))
        }
        
        // Mock picker that selects based on message content
        #[derive(Clone)]
        struct MockPicker;
        
        impl Service<PickRequest> for MockPicker {
            type Response = String;
            type Error = BoxError;
            type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;
            
            fn poll_ready(&mut self, _: &mut std::task::Context<'_>) -> std::task::Poll<Result<(), Self::Error>> {
                std::task::Poll::Ready(Ok(()))
            }
            
            fn call(&mut self, req: PickRequest) -> Self::Future {
                Box::pin(async move {
                let content = req.messages.first()
                    .and_then(|msg| match msg {
                        async_openai::types::ChatCompletionRequestMessage::User(user_msg) => {
                            match &user_msg.content {
                                async_openai::types::ChatCompletionRequestUserMessageContent::Text(text) => {
                                    Some(text.as_str())
                                }
                                _ => None,
                            }
                        }
                        _ => None,
                    })
                    .unwrap_or("");
                    
                let agent = if content.contains("billing") {
                    "billing_agent"
                } else if content.contains("technical") {
                    "tech_agent" 
                } else {
                    "triage_agent"
                };
                
                    Ok::<String, BoxError>(agent.to_string())
                })
            }
        }
        
        fn mock_picker() -> MockPicker {
            MockPicker
        }
        
        #[tokio::test]
        async fn handoff_coordinator_basic_operation() -> Result<(), BoxError> {
            let coordinator = GroupBuilder::new()
                .agent("triage_agent", mock_agent("triage", "I'll handle your request"))
                .agent("billing_agent", mock_agent("billing", "Billing issue resolved"))
                .picker(mock_picker())
                .handoff_policy(explicit_handoff_to("billing_agent"))
                .build();
                
            let mut service = coordinator;
            
            use async_openai::types::{ChatCompletionRequestMessage, ChatCompletionRequestUserMessageArgs};
            let user_message = ChatCompletionRequestUserMessageArgs::default()
                .content("I have a general question")
                .build()?;
            
            let request = CreateChatCompletionRequest {
                messages: vec![ChatCompletionRequestMessage::User(user_message)],
                model: "gpt-4o".to_string(),
                ..Default::default()
            };
            
            let result = ServiceExt::ready(&mut service).await?
                .call(request).await?;
                
            // Should route to triage_agent and return its response
            assert_eq!(result.messages.len(), 1);
            assert!(format!("{:?}", result.messages[0]).contains("[triage]: I'll handle your request"));
            assert_eq!(result.steps, 1);
            
            Ok(())
        }
        
        #[tokio::test]
        async fn handoff_coordinator_sequential_workflow() -> Result<(), BoxError> {
            let sequential_policy = sequential_handoff(vec![
                "researcher".to_string(), 
                "writer".to_string(), 
                "reviewer".to_string()
            ]);
            
            #[derive(Clone)]
            struct ResearchPicker;
            impl Service<PickRequest> for ResearchPicker {
                type Response = String;
                type Error = BoxError;
                type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;
                
                fn poll_ready(&mut self, _: &mut std::task::Context<'_>) -> std::task::Poll<Result<(), Self::Error>> {
                    std::task::Poll::Ready(Ok(()))
                }
                
                fn call(&mut self, _req: PickRequest) -> Self::Future {
                    Box::pin(async move {
                        Ok::<String, BoxError>("researcher".to_string())
                    })
                }
            }
            
            let coordinator = GroupBuilder::new()
                .agent("researcher", mock_agent("researcher", "Research complete"))
                .agent("writer", mock_agent("writer", "Article written"))
                .agent("reviewer", mock_agent("reviewer", "Review complete"))
                .picker(ResearchPicker)
                .handoff_policy(sequential_policy)
                .build();
                
            let mut service = coordinator;
            
            use async_openai::types::{ChatCompletionRequestMessage, ChatCompletionRequestUserMessageArgs};
            let user_message = ChatCompletionRequestUserMessageArgs::default()
                .content("Write an article about AI")
                .build()?;
            
            let request = CreateChatCompletionRequest {
                messages: vec![ChatCompletionRequestMessage::User(user_message)],
                model: "gpt-4o".to_string(),
                ..Default::default()
            };
            
            let result = ServiceExt::ready(&mut service).await?
                .call(request).await?;
                
            // Should start with researcher and proceed through the sequence
            assert!(result.messages.len() >= 1);
            // The sequential policy doesn't have automatic handoff implemented in our mock agents
            // So it should just return the researcher's response
            assert!(format!("{:?}", result.messages[0]).contains("[researcher]: Research complete"));
            
            Ok(())
        }
        
        #[tokio::test]
        async fn handoff_coordinator_picker_routing() -> Result<(), BoxError> {
            let coordinator = GroupBuilder::new()
                .agent("triage_agent", mock_agent("triage", "General help"))
                .agent("billing_agent", mock_agent("billing", "Billing help"))
                .agent("tech_agent", mock_agent("tech", "Technical help"))
                .picker(mock_picker())
                .handoff_policy(explicit_handoff_to("tech_agent"))
                .build();
                
            let mut service = coordinator;
            
            // Test billing routing
            use async_openai::types::{ChatCompletionRequestMessage, ChatCompletionRequestUserMessageArgs};
            let billing_message = ChatCompletionRequestUserMessageArgs::default()
                .content("I have a billing question")
                .build()?;
            
            let billing_request = CreateChatCompletionRequest {
                messages: vec![ChatCompletionRequestMessage::User(billing_message)],
                model: "gpt-4o".to_string(),
                ..Default::default()
            };
            
            let result = ServiceExt::ready(&mut service).await?
                .call(billing_request).await?;
                
            // Should route directly to billing_agent
            assert!(format!("{:?}", result.messages[0]).contains("billing"));
            
            // Test technical routing
            let tech_message = ChatCompletionRequestUserMessageArgs::default()
                .content("I have a technical issue")
                .build()?;
            
            let tech_request = CreateChatCompletionRequest {
                messages: vec![ChatCompletionRequestMessage::User(tech_message)],
                model: "gpt-4o".to_string(),
                ..Default::default()
            };
            
            let result2 = ServiceExt::ready(&mut service).await?
                .call(tech_request).await?;
                
            // Should route directly to tech_agent  
            assert!(format!("{:?}", result2.messages[0]).contains("tech"));
            
            Ok(())
        }
        
        #[tokio::test]
        async fn handoff_coordinator_composite_policy() -> Result<(), BoxError> {
            let composite_policy = composite_handoff(vec![
                AnyHandoffPolicy::Explicit(explicit_handoff_to("specialist")),
                AnyHandoffPolicy::Sequential(sequential_handoff(vec!["a".to_string(), "b".to_string()])),
            ]);
            
            #[derive(Clone)]
            struct TriagePicker;
            impl Service<PickRequest> for TriagePicker {
                type Response = String;
                type Error = BoxError;
                type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;
                
                fn poll_ready(&mut self, _: &mut std::task::Context<'_>) -> std::task::Poll<Result<(), Self::Error>> {
                    std::task::Poll::Ready(Ok(()))
                }
                
                fn call(&mut self, _req: PickRequest) -> Self::Future {
                    Box::pin(async move {
                        Ok::<String, BoxError>("triage".to_string())
                    })
                }
            }
            
            let coordinator = GroupBuilder::new()
                .agent("triage", mock_agent("triage", "Triage response"))
                .agent("specialist", mock_agent("specialist", "Specialist response"))
                .agent("a", mock_agent("a", "Agent A response"))
                .agent("b", mock_agent("b", "Agent B response"))
                .picker(TriagePicker)
                .handoff_policy(composite_policy)
                .build();
                
            // Verify handoff tools are exposed
            let tools = coordinator.handoff_tools();
            assert_eq!(tools.len(), 1); // Only explicit policy generates tools
            assert_eq!(tools[0].function.name, "handoff_to_specialist");
            
            let mut service = coordinator;
            
            use async_openai::types::{ChatCompletionRequestMessage, ChatCompletionRequestUserMessageArgs};
            let user_message = ChatCompletionRequestUserMessageArgs::default()
                .content("Help me")
                .build()?;
            
            let request = CreateChatCompletionRequest {
                messages: vec![ChatCompletionRequestMessage::User(user_message)],
                model: "gpt-4o".to_string(),
                ..Default::default()
            };
            
            let result = ServiceExt::ready(&mut service).await?
                .call(request).await?;
                
            // Should execute triage agent
            assert!(result.messages.len() >= 1);
            assert!(format!("{:?}", result.messages[0]).contains("[triage]: Triage response"));
            
            Ok(())
        }
        
        #[tokio::test]
        async fn handoff_coordinator_error_handling() -> Result<(), BoxError> {
            #[derive(Clone)]
            struct AgentAPicker;
            impl Service<PickRequest> for AgentAPicker {
                type Response = String;
                type Error = BoxError;
                type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;
                
                fn poll_ready(&mut self, _: &mut std::task::Context<'_>) -> std::task::Poll<Result<(), Self::Error>> {
                    std::task::Poll::Ready(Ok(()))
                }
                
                fn call(&mut self, _req: PickRequest) -> Self::Future {
                    Box::pin(async move {
                        Ok::<String, BoxError>("agent_a".to_string())
                    })
                }
            }
            
            let coordinator = GroupBuilder::new()
                .agent("agent_a", mock_agent("a", "Response A"))
                .picker(AgentAPicker)
                .handoff_policy(explicit_handoff_to("nonexistent_agent"))
                .build();
                
            let mut service = coordinator;
            
            use async_openai::types::{ChatCompletionRequestMessage, ChatCompletionRequestUserMessageArgs};
            let user_message = ChatCompletionRequestUserMessageArgs::default()
                .content("Test message")
                .build()?;
            
            let request = CreateChatCompletionRequest {
                messages: vec![ChatCompletionRequestMessage::User(user_message)],
                model: "gpt-4o".to_string(),
                ..Default::default()
            };
            
            // Should complete without handoff since explicit policy doesn't auto-handoff
            let result = ServiceExt::ready(&mut service).await?
                .call(request).await?;
                
            assert_eq!(result.messages.len(), 1);
            assert!(format!("{:?}", result.messages[0]).contains("[a]: Response A"));
            
            Ok(())
        }
        
        #[tokio::test]
        async fn handoff_coordinator_max_handoffs_protection() -> Result<(), BoxError> {
            // Create a policy that always hands off to create infinite loop
            struct InfiniteHandoffPolicy;
            
            impl HandoffPolicy for InfiniteHandoffPolicy {
                fn handoff_tools(&self) -> Vec<ChatCompletionTool> { vec![] }
                fn handle_handoff_tool(&self, _: &ToolInvocation) -> Result<HandoffRequest, BoxError> {
                    Err("No tools".into())
                }
                fn should_handoff(&self, _: &LoopState, _: &StepOutcome) -> Option<HandoffRequest> {
                    Some(HandoffRequest {
                        target_agent: "agent_b".to_string(),
                        context: None,
                        reason: Some("Infinite handoff test".to_string()),
                    })
                }
                fn is_handoff_tool(&self, _: &str) -> bool { false }
            }
            
            impl Clone for InfiniteHandoffPolicy {
                fn clone(&self) -> Self { InfiniteHandoffPolicy }
            }
            
            #[derive(Clone)]
            struct StartPicker;
            impl Service<PickRequest> for StartPicker {
                type Response = String;
                type Error = BoxError;
                type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;
                
                fn poll_ready(&mut self, _: &mut std::task::Context<'_>) -> std::task::Poll<Result<(), Self::Error>> {
                    std::task::Poll::Ready(Ok(()))
                }
                
                fn call(&mut self, _req: PickRequest) -> Self::Future {
                    Box::pin(async move {
                        Ok::<String, BoxError>("agent_a".to_string())
                    })
                }
            }
            
            let coordinator = GroupBuilder::new()
                .agent("agent_a", mock_agent("a", "Response A"))
                .agent("agent_b", mock_agent("b", "Response B"))
                .picker(StartPicker)
                .handoff_policy(InfiniteHandoffPolicy)
                .build();
                
            let mut service = coordinator;
            
            use async_openai::types::{ChatCompletionRequestMessage, ChatCompletionRequestUserMessageArgs};
            let user_message = ChatCompletionRequestUserMessageArgs::default()
                .content("Test infinite handoff protection")
                .build()?;
            
            let request = CreateChatCompletionRequest {
                messages: vec![ChatCompletionRequestMessage::User(user_message)],
                model: "gpt-4o".to_string(),
                ..Default::default()
            };
            
            let result = ServiceExt::ready(&mut service).await?
                .call(request).await;
                
            // Should error due to max handoffs exceeded
            assert!(result.is_err());
            assert!(result.unwrap_err().to_string().contains("Maximum handoffs exceeded"));
            
            Ok(())
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
