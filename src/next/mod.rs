//! Experimental next-gen Tower composition: static DI, raw OpenAI I/O,
//! step service + steer-based tool routing. This module is intentionally
//! independent of the existing runtime. It may borrow ideas but should not
//! touch existing code paths.

pub mod approvals;
pub mod budgets;
pub mod codec;
pub mod concurrency;
pub mod groups;
pub mod layers;
pub mod observability;
pub mod provider;
pub mod recording;
pub mod resilience;
pub mod services;
pub mod sessions;
pub mod streaming;
pub mod utils;

use std::{future::Future, pin::Pin, sync::Arc};

use async_openai::{
    config::OpenAIConfig,
    types::{
        ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage,
        ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestToolMessageArgs,
        ChatCompletionRequestUserMessageArgs, ChatCompletionTool, ChatCompletionToolArgs,
        ChatCompletionToolType, CreateChatCompletionRequest, CreateChatCompletionRequestArgs,
        FunctionObjectArgs,
    },
    Client,
};
use futures::future::BoxFuture;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde_json::Value;
use tower::{util::BoxService, BoxError, Layer, Service, ServiceExt};

// =============================
// Tool service modeling
// =============================

/// Uniform tool invocation passed to routed tool services.
#[derive(Debug, Clone)]
pub struct ToolInvocation {
    pub id: String,   // tool_call_id
    pub name: String, // function.name
    pub arguments: Value,
}

/// Uniform tool output produced by tool services.
#[derive(Debug, Clone)]
pub struct ToolOutput {
    pub id: String, // same as invocation.id
    pub result: Value,
}

/// Boxed tool service type alias.
pub type ToolSvc = BoxService<ToolInvocation, ToolOutput, BoxError>;

/// Definition of a tool: function spec (for OpenAI) + service implementation.
pub struct ToolDef {
    pub name: &'static str,
    pub description: &'static str,
    pub parameters_schema: Value,
    pub service: ToolSvc,
}

impl ToolDef {
    /// Create a tool definition from a handler function that takes JSON args and returns JSON.
    pub fn from_handler(
        name: &'static str,
        description: &'static str,
        parameters_schema: Value,
        handler: std::sync::Arc<
            dyn Fn(Value) -> BoxFuture<'static, Result<Value, BoxError>> + Send + Sync + 'static,
        >,
    ) -> Self {
        let handler_arc = handler.clone();
        let svc = tower::service_fn(move |inv: ToolInvocation| {
            let handler = handler_arc.clone();
            async move {
                if inv.name != name {
                    return Err::<ToolOutput, BoxError>(
                        format!("routed to wrong tool: expected={}, got={}", name, inv.name).into(),
                    );
                }
                let out = (handler)(inv.arguments).await?;
                Ok(ToolOutput {
                    id: inv.id,
                    result: out,
                })
            }
        });
        Self {
            name,
            description,
            parameters_schema,
            service: BoxService::new(svc),
        }
    }

    /// Convert this tool's function signature into an OpenAI ChatCompletionTool spec.
    pub fn to_openai_tool(&self) -> ChatCompletionTool {
        let func = FunctionObjectArgs::default()
            .name(self.name)
            .description(self.description)
            .parameters(self.parameters_schema.clone())
            .build()
            .expect("valid function object");
        ChatCompletionToolArgs::default()
            .r#type(ChatCompletionToolType::Function)
            .function(func)
            .build()
            .expect("valid chat tool")
    }
}

/// DX sugar: create a tool from a typed handler.
/// - `A` is the input args struct (Deserialize + JsonSchema)
/// - `R` is the output type (Serialize)
pub fn tool_typed<A, H, Fut, R>(
    name: &'static str,
    description: &'static str,
    handler: H,
) -> ToolDef
where
    A: DeserializeOwned + JsonSchema + Send + 'static,
    R: serde::Serialize + Send + 'static,
    H: Fn(A) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<R, BoxError>> + Send + 'static,
{
    let schema = schemars::schema_for!(A);
    let params_value = serde_json::to_value(schema.schema).expect("schema to value");
    let handler_arc_inner = Arc::new(handler);
    let handler_arc: Arc<
        dyn Fn(Value) -> BoxFuture<'static, Result<Value, BoxError>> + Send + Sync,
    > = Arc::new(move |raw: Value| {
        let h = handler_arc_inner.clone();
        Box::pin(async move {
            let args: A = serde_json::from_value(raw)?;
            let out: R = (h.as_ref())(args).await?;
            let val = serde_json::to_value(out)?;
            Ok(val)
        })
    });
    ToolDef::from_handler(name, description, params_value, handler_arc)
}

/// Simple router service over tools using a name → index table.
pub struct ToolRouter {
    name_to_index: std::collections::HashMap<&'static str, usize>,
    services: Vec<ToolSvc>, // index 0 is the unknown-tool fallback
}

impl ToolRouter {
    pub fn new(tools: Vec<ToolDef>) -> (Self, Vec<ChatCompletionTool>) {
        use std::collections::HashMap;

        let unknown = BoxService::new(tower::service_fn(|inv: ToolInvocation| async move {
            Err::<ToolOutput, BoxError>(format!("unknown tool: {}", inv.name).into())
        }));

        let mut services: Vec<ToolSvc> = vec![unknown];
        let mut specs: Vec<ChatCompletionTool> = Vec::with_capacity(tools.len());
        let mut name_to_index: HashMap<&'static str, usize> = HashMap::new();

        for (i, td) in tools.into_iter().enumerate() {
            name_to_index.insert(td.name, i + 1);
            specs.push(td.to_openai_tool());
            services.push(td.service);
        }

        (
            Self {
                name_to_index,
                services,
            },
            specs,
        )
    }
}

impl Service<ToolInvocation> for ToolRouter {
    type Response = ToolOutput;
    type Error = BoxError;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(
        &mut self,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        // We check readiness per selected service inside `call`.
        std::task::Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: ToolInvocation) -> Self::Future {
        let idx = self
            .name_to_index
            .get(req.name.as_str())
            .copied()
            .unwrap_or(0);

        // Safe: index 0 is always present (unknown fallback)
        let svc: &mut ToolSvc = &mut self.services[idx];
        // Call selected service and forward its future
        let fut = svc.call(req);
        Box::pin(async move { fut.await })
    }
}

// =============================
// Step service and layer
// =============================

/// Auxiliary accounting captured per step.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct StepAux {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub tool_invocations: usize,
}

/// Outcome of a single agent step.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum StepOutcome {
    Next {
        messages: Vec<ChatCompletionRequestMessage>,
        aux: StepAux,
        invoked_tools: Vec<String>,
    },
    Done {
        messages: Vec<ChatCompletionRequestMessage>,
        aux: StepAux,
    },
}

/// One-step agent service parameterized by a routed tool service `S`.
pub struct Step<S> {
    client: Arc<Client<OpenAIConfig>>,
    model: String,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    tools: Arc<tokio::sync::Mutex<S>>,
    tool_specs: Arc<Vec<ChatCompletionTool>>, // supplied to requests if missing
}

impl<S> Step<S> {
    pub fn new(
        client: Arc<Client<OpenAIConfig>>,
        model: impl Into<String>,
        tools: S,
        tool_specs: Vec<ChatCompletionTool>,
    ) -> Self {
        Self {
            client,
            model: model.into(),
            temperature: None,
            max_tokens: None,
            tools: Arc::new(tokio::sync::Mutex::new(tools)),
            tool_specs: Arc::new(tool_specs),
        }
    }

    pub fn temperature(mut self, t: f32) -> Self {
        self.temperature = Some(t);
        self
    }

    pub fn max_tokens(mut self, mt: u32) -> Self {
        self.max_tokens = Some(mt);
        self
    }
}

/// Layer that lifts a routed tool service `S` into a `Step<S>` service.
pub struct StepLayer {
    client: Arc<Client<OpenAIConfig>>,
    model: String,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    tool_specs: Arc<Vec<ChatCompletionTool>>,
}

impl StepLayer {
    pub fn new(
        client: Arc<Client<OpenAIConfig>>,
        model: impl Into<String>,
        tool_specs: Vec<ChatCompletionTool>,
    ) -> Self {
        Self {
            client,
            model: model.into(),
            temperature: None,
            max_tokens: None,
            tool_specs: Arc::new(tool_specs),
        }
    }

    pub fn temperature(mut self, t: f32) -> Self {
        self.temperature = Some(t);
        self
    }

    pub fn max_tokens(mut self, mt: u32) -> Self {
        self.max_tokens = Some(mt);
        self
    }
}

impl<S> Layer<S> for StepLayer {
    type Service = Step<S>;

    fn layer(&self, tools: S) -> Self::Service {
        let mut s = Step::new(
            self.client.clone(),
            self.model.clone(),
            tools,
            (*self.tool_specs).clone(),
        );
        s.temperature = self.temperature;
        s.max_tokens = self.max_tokens;
        s
    }
}

impl<S> Service<CreateChatCompletionRequest> for Step<S>
where
    S: Service<ToolInvocation, Response = ToolOutput, Error = BoxError> + Send + 'static,
    S::Future: Send + 'static,
{
    type Response = StepOutcome;
    type Error = BoxError;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(
        &mut self,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        let _ = cx; // Always ready; we await tools readiness inside `call`
        std::task::Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: CreateChatCompletionRequest) -> Self::Future {
        let client = self.client.clone();
        let model = self.model.clone();
        let temperature = self.temperature;
        let max_tokens = self.max_tokens;
        let tools = self.tools.clone();
        let tool_specs = self.tool_specs.clone();

        Box::pin(async move {
            // Rebuild request using builder to avoid deprecated field access
            let effective_model: Option<String> = req.model.clone().into();

            let mut builder = CreateChatCompletionRequestArgs::default();
            builder.messages(req.messages.clone());
            if let Some(m) = effective_model.as_ref() {
                builder.model(m);
            } else {
                builder.model(&model);
            }
            if let Some(t) = req.temperature.or(temperature) {
                builder.temperature(t);
            }
            if let Some(mt) = max_tokens {
                builder.max_tokens(mt);
            }
            if let Some(ts) = req.tools.clone() {
                builder.tools(ts);
            } else if !tool_specs.is_empty() {
                builder.tools((*tool_specs).clone());
            }

            let rebuilt_req = builder
                .build()
                .map_err(|e| format!("request build error: {}", e))?;

            let mut messages = rebuilt_req.messages.clone();

            // Single OpenAI call
            let resp = client.chat().create(rebuilt_req).await?;
            let usage = resp.usage.clone().unwrap_or_default();
            let mut aux = StepAux {
                prompt_tokens: usage.prompt_tokens as usize,
                completion_tokens: usage.completion_tokens as usize,
                tool_invocations: 0,
            };

            let choice = resp
                .choices
                .first()
                .ok_or_else(|| "no choices".to_string())?;
            let assistant = choice.message.clone();

            // Append assistant message by constructing request-side equivalent
            let mut asst_builder = ChatCompletionRequestAssistantMessageArgs::default();
            if let Some(content) = assistant.content.clone() {
                asst_builder.content(content);
            } else {
                asst_builder.content("");
            }
            if let Some(tool_calls) = assistant.tool_calls.clone() {
                asst_builder.tool_calls(tool_calls);
            }
            let asst_req = asst_builder
                .build()
                .map_err(|e| format!("assistant msg build error: {}", e))?;
            messages.push(ChatCompletionRequestMessage::from(asst_req));

            // Execute tool calls if present
            let tool_calls = assistant.tool_calls.unwrap_or_default();
            if tool_calls.is_empty() {
                return Ok(StepOutcome::Done { messages, aux });
            }

            let mut invoked_names: Vec<String> = Vec::with_capacity(tool_calls.len());
            for tc in tool_calls {
                let name = tc.function.name;
                let args: Value = serde_json::from_str(&tc.function.arguments)?;
                let inv = ToolInvocation {
                    id: tc.id.clone(),
                    name: name.clone(),
                    arguments: args,
                };
                invoked_names.push(name);
                let mut guard = tools.lock().await;
                let ToolOutput { id, result } = guard.ready().await?.call(inv).await?;
                aux.tool_invocations += 1;

                // Append tool role message
                let tool_msg = ChatCompletionRequestToolMessageArgs::default()
                    .content(result.to_string())
                    .tool_call_id(id)
                    .build()?;
                messages.push(tool_msg.into());
            }

            Ok(StepOutcome::Next {
                messages,
                aux,
                invoked_tools: invoked_names,
            })
        })
    }
}

// =============================
// Convenience helpers for examples/tests
// =============================

/// Build a simple chat request from plain strings.
pub fn simple_chat_request(system: &str, user: &str) -> CreateChatCompletionRequest {
    let sys = ChatCompletionRequestSystemMessageArgs::default()
        .content(system)
        .build()
        .expect("system msg");
    let usr = ChatCompletionRequestUserMessageArgs::default()
        .content(user)
        .build()
        .expect("user msg");
    CreateChatCompletionRequestArgs::default()
        .model("gpt-4o")
        .messages(vec![sys.into(), usr.into()])
        .build()
        .expect("chat req")
}

// =============================
// Agent loop: composable policies and layer
// =============================

/// Stop reasons reported by the agent loop.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum AgentStopReason {
    DoneNoToolCalls,
    MaxSteps,
    ToolCalled(String),
    TokensBudgetExceeded,
    ToolBudgetExceeded,
    TimeBudgetExceeded,
}

// =============================
// DX sugar: Policy builder, Agent builder, run helpers
// =============================

/// Chainable policy builder.
#[derive(Default, Clone)]
pub struct Policy {
    inner: CompositePolicy,
}

#[allow(dead_code)]
impl Policy {
    pub fn new() -> Self {
        Self {
            inner: CompositePolicy::default(),
        }
    }
    pub fn until_no_tool_calls(mut self) -> Self {
        self.inner.policies.push(policies::until_no_tool_calls());
        self
    }
    pub fn or_tool(mut self, name: impl Into<String>) -> Self {
        self.inner.policies.push(policies::until_tool_called(name));
        self
    }
    pub fn or_max_steps(mut self, max: usize) -> Self {
        self.inner.policies.push(policies::max_steps(max));
        self
    }
    pub fn build(self) -> CompositePolicy {
        self.inner
    }
}

/// Boxed agent service type for ergonomic returns.
pub type AgentSvc = BoxService<CreateChatCompletionRequest, AgentRun, BoxError>;

/// Thin facade to build an agent stack from tools, model, and policy.
pub struct Agent;

pub struct AgentBuilder {
    client: Arc<Client<OpenAIConfig>>,
    model: String,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    tools: Vec<ToolDef>,
    policy: CompositePolicy,
}

impl Agent {
    pub fn builder(client: Arc<Client<OpenAIConfig>>) -> AgentBuilder {
        AgentBuilder {
            client,
            model: "gpt-4o".to_string(),
            temperature: None,
            max_tokens: None,
            tools: Vec::new(),
            policy: CompositePolicy::default(),
        }
    }
}

impl AgentBuilder {
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }
    pub fn temperature(mut self, t: f32) -> Self {
        self.temperature = Some(t);
        self
    }
    pub fn max_tokens(mut self, mt: u32) -> Self {
        self.max_tokens = Some(mt);
        self
    }
    pub fn tool(mut self, tool: ToolDef) -> Self {
        self.tools.push(tool);
        self
    }
    pub fn tools(mut self, tools: Vec<ToolDef>) -> Self {
        self.tools.extend(tools);
        self
    }
    pub fn policy(mut self, policy: CompositePolicy) -> Self {
        self.policy = policy;
        self
    }

    pub fn build(self) -> AgentSvc {
        let (router, specs) = ToolRouter::new(self.tools);
        let step = StepLayer::new(self.client, self.model, specs)
            .temperature(self.temperature.unwrap_or(0.0))
            .max_tokens(self.max_tokens.unwrap_or(512))
            .layer(router);
        let agent = AgentLoopLayer::new(self.policy).layer(step);
        BoxService::new(agent)
    }
}

/// Convenience: run a prompt through an agent service.
pub async fn run(agent: &mut AgentSvc, system: &str, user: &str) -> Result<AgentRun, BoxError> {
    let req = simple_chat_request(system, user);
    let resp = ServiceExt::ready(agent).await?.call(req).await?;
    Ok(resp)
}

/// Loop state visible to policies.
#[derive(Debug, Clone, Default)]
pub struct LoopState {
    pub steps: usize,
}

/// Policy interface controlling loop termination.
pub trait AgentPolicy: Send + Sync {
    fn decide(&self, state: &LoopState, last: &StepOutcome) -> Option<AgentStopReason>;
}

/// Function-backed policy for ergonomic composition.
#[derive(Clone)]
pub struct PolicyFn(
    pub Arc<dyn Fn(&LoopState, &StepOutcome) -> Option<AgentStopReason> + Send + Sync + 'static>,
);

impl AgentPolicy for PolicyFn {
    fn decide(&self, state: &LoopState, last: &StepOutcome) -> Option<AgentStopReason> {
        (self.0)(state, last)
    }
}

/// Composite policy: stop when any sub-policy returns a stop reason.
#[derive(Clone, Default)]
pub struct CompositePolicy {
    policies: Vec<PolicyFn>,
}

#[allow(dead_code)]
impl CompositePolicy {
    pub fn new(policies: Vec<PolicyFn>) -> Self {
        Self { policies }
    }
    pub fn push(&mut self, p: PolicyFn) {
        self.policies.push(p);
    }
}

impl AgentPolicy for CompositePolicy {
    fn decide(&self, state: &LoopState, last: &StepOutcome) -> Option<AgentStopReason> {
        for p in &self.policies {
            if let Some(r) = p.decide(state, last) {
                return Some(r);
            }
        }
        None
    }
}

/// Built-in policies
#[allow(dead_code)]
pub mod policies {
    use super::*;

    pub fn until_no_tool_calls() -> PolicyFn {
        PolicyFn(Arc::new(|_s, last| match last {
            StepOutcome::Done { .. } => Some(AgentStopReason::DoneNoToolCalls),
            _ => None,
        }))
    }

    pub fn until_tool_called(tool_name: impl Into<String>) -> PolicyFn {
        let target = tool_name.into();
        PolicyFn(Arc::new(move |_s, last| match last {
            StepOutcome::Next { invoked_tools, .. } => {
                if invoked_tools.iter().any(|n| n == &target) {
                    Some(AgentStopReason::ToolCalled(target.clone()))
                } else {
                    None
                }
            }
            _ => None,
        }))
    }

    pub fn max_steps(max: usize) -> PolicyFn {
        PolicyFn(Arc::new(move |s, _| {
            if s.steps >= max {
                Some(AgentStopReason::MaxSteps)
            } else {
                None
            }
        }))
    }
}

/// Final run summary from the agent loop.
#[derive(Debug, Clone)]
pub struct AgentRun {
    pub messages: Vec<ChatCompletionRequestMessage>,
    pub steps: usize,
    pub stop: AgentStopReason,
}

/// Layer to wrap a step service with an agent loop controlled by a policy.
pub struct AgentLoopLayer<P> {
    policy: P,
}

impl<P> AgentLoopLayer<P> {
    pub fn new(policy: P) -> Self {
        Self { policy }
    }
}

pub struct AgentLoop<S, P> {
    inner: Arc<tokio::sync::Mutex<S>>,
    policy: P,
}

impl<S, P> Layer<S> for AgentLoopLayer<P>
where
    P: Clone,
{
    type Service = AgentLoop<S, P>;
    fn layer(&self, inner: S) -> Self::Service {
        AgentLoop {
            inner: Arc::new(tokio::sync::Mutex::new(inner)),
            policy: self.policy.clone(),
        }
    }
}

impl<S, P> Service<CreateChatCompletionRequest> for AgentLoop<S, P>
where
    S: Service<CreateChatCompletionRequest, Response = StepOutcome, Error = BoxError>
        + Send
        + 'static,
    S::Future: Send + 'static,
    P: AgentPolicy + Send + Sync + Clone + 'static,
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
        let inner = self.inner.clone();
        let policy = self.policy.clone();
        Box::pin(async move {
            let mut state = LoopState::default();
            let base_model = req.model.clone();
            let mut current_messages = req.messages.clone();
            loop {
                // Rebuild request for this iteration
                let mut builder = CreateChatCompletionRequestArgs::default();
                builder.model(&base_model);
                builder.messages(current_messages.clone());
                let current_req = builder
                    .build()
                    .map_err(|e| format!("build req error: {}", e))?;

                let mut guard = inner.lock().await;
                let outcome = guard.ready().await?.call(current_req).await?;
                drop(guard);

                state.steps += 1;

                if let Some(stop) = policy.decide(&state, &outcome) {
                    let messages = match outcome {
                        StepOutcome::Next { messages, .. } => messages,
                        StepOutcome::Done { messages, .. } => messages,
                    };
                    return Ok(AgentRun {
                        messages,
                        steps: state.steps,
                        stop,
                    });
                }

                match outcome {
                    StepOutcome::Next { messages, .. } => {
                        current_messages = messages;
                    }
                    StepOutcome::Done { messages, .. } => {
                        return Ok(AgentRun {
                            messages,
                            steps: state.steps,
                            stop: AgentStopReason::DoneNoToolCalls,
                        });
                    }
                }
            }
        })
    }
}
