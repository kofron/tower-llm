//! Core agent implementation using Tower services and static dependency injection.

use std::{future::Future, pin::Pin, sync::Arc};

use crate::groups::HandoffPolicy;
use crate::provider::{ModelService, OpenAIProvider, ProviderResponse};
use async_openai::{
    config::OpenAIConfig,
    types::{
        ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage,
        ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestToolMessageArgs,
        ChatCompletionRequestUserMessageArgs, ChatCompletionTool, ChatCompletionToolArgs,
        ChatCompletionToolType, CreateChatCompletionRequest, CreateChatCompletionRequestArgs,
        FunctionObjectArgs, ReasoningEffort,
    },
    Client,
};
use futures::future::BoxFuture;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde_json::Value;
use tokio::sync::Semaphore;
use tower::{
    util::{BoxCloneService, BoxService},
    BoxError, Layer, Service, ServiceExt,
};
use tracing::{debug, trace};

/// Join policy for parallel tool execution
#[derive(Debug, Clone, Copy, Default)]
pub enum ToolJoinPolicy {
    /// Return error on the first failing tool; pending tools are cancelled
    #[default]
    FailFast,
    /// Run all tools to completion; if any fail, surface an aggregated error at the end
    JoinAll,
}

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
pub type ToolSvc = BoxCloneService<ToolInvocation, ToolOutput, BoxError>;

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
            service: BoxCloneService::new(svc),
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
#[derive(Clone)]
pub struct ToolRouter {
    name_to_index: std::collections::HashMap<&'static str, usize>,
    services: Vec<ToolSvc>, // index 0 is the unknown-tool fallback
}

impl ToolRouter {
    pub fn new(tools: Vec<ToolDef>) -> (Self, Vec<ChatCompletionTool>) {
        use std::collections::HashMap;

        let unknown = BoxCloneService::new(tower::service_fn(|inv: ToolInvocation| async move {
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
        Box::pin(fut)
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
pub struct Step<S, P> {
    provider: Arc<tokio::sync::Mutex<P>>,
    model: String,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    reasoning_effort: Option<ReasoningEffort>,
    instructions: Option<String>,
    tools: S,
    tool_specs: Arc<Vec<ChatCompletionTool>>, // supplied to requests if missing
    parallel_tools: bool,
    tool_concurrency_limit: Option<usize>,
    join_policy: ToolJoinPolicy,
}

impl<S, P> Step<S, P> {
    pub fn new(
        provider: P,
        model: impl Into<String>,
        tools: S,
        tool_specs: Vec<ChatCompletionTool>,
    ) -> Self {
        Self {
            provider: Arc::new(tokio::sync::Mutex::new(provider)),
            model: model.into(),
            temperature: None,
            max_tokens: None,
            reasoning_effort: None,
            instructions: None,
            tools,
            tool_specs: Arc::new(tool_specs),
            parallel_tools: false,
            tool_concurrency_limit: None,
            join_policy: ToolJoinPolicy::FailFast,
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

    pub fn enable_parallel_tools(mut self, enabled: bool) -> Self {
        self.parallel_tools = enabled;
        self
    }

    pub fn tool_concurrency_limit(mut self, limit: usize) -> Self {
        self.tool_concurrency_limit = Some(limit);
        self
    }

    pub fn tool_join_policy(mut self, policy: ToolJoinPolicy) -> Self {
        self.join_policy = policy;
        self
    }

    pub fn reasoning_effort(mut self, effort: ReasoningEffort) -> Self {
        self.reasoning_effort = Some(effort);
        self
    }
}

/// Layer that lifts a routed tool service `S` into a `Step<S>` service.
pub struct StepLayer<P> {
    provider: P,
    model: String,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    reasoning_effort: Option<ReasoningEffort>,
    instructions: Option<String>,
    tool_specs: Arc<Vec<ChatCompletionTool>>,
    parallel_tools: bool,
    tool_concurrency_limit: Option<usize>,
    join_policy: ToolJoinPolicy,
}

impl<P> StepLayer<P> {
    pub fn new(provider: P, model: impl Into<String>, tool_specs: Vec<ChatCompletionTool>) -> Self {
        Self {
            provider,
            model: model.into(),
            temperature: None,
            max_tokens: None,
            reasoning_effort: None,
            instructions: None,
            tool_specs: Arc::new(tool_specs),
            parallel_tools: false,
            tool_concurrency_limit: None,
            join_policy: ToolJoinPolicy::FailFast,
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

    pub fn parallel_tools(mut self, enabled: bool) -> Self {
        self.parallel_tools = enabled;
        self
    }

    pub fn tool_concurrency_limit(mut self, limit: usize) -> Self {
        self.tool_concurrency_limit = Some(limit);
        self
    }

    pub fn tool_join_policy(mut self, policy: ToolJoinPolicy) -> Self {
        self.join_policy = policy;
        self
    }

    pub fn reasoning_effort(mut self, effort: ReasoningEffort) -> Self {
        self.reasoning_effort = Some(effort);
        self
    }

    pub fn instructions(mut self, text: impl Into<String>) -> Self {
        self.instructions = Some(text.into());
        self
    }
}

impl<S, P> Layer<S> for StepLayer<P>
where
    P: Clone,
{
    type Service = Step<S, P>;

    fn layer(&self, tools: S) -> Self::Service {
        let mut s = Step::new(
            self.provider.clone(),
            self.model.clone(),
            tools,
            (*self.tool_specs).clone(),
        );
        s.temperature = self.temperature;
        s.max_tokens = self.max_tokens;
        s.reasoning_effort = self.reasoning_effort.clone();
        s.instructions = self.instructions.clone();
        // propagate instructions if StepLayer has it
        // Note: StepLayer currently doesn't store instructions; this will be set via AgentBuilder mapping below
        s.parallel_tools = self.parallel_tools;
        s.tool_concurrency_limit = self.tool_concurrency_limit;
        s.join_policy = self.join_policy;
        s
    }
}

impl<S, P> Service<CreateChatCompletionRequest> for Step<S, P>
where
    S: Service<ToolInvocation, Response = ToolOutput, Error = BoxError> + Clone + Send + 'static,
    S::Future: Send + 'static,
    P: ModelService + Send + 'static,
    P::Future: Send + 'static,
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
        let provider = self.provider.clone();
        let model = self.model.clone();
        let temperature = self.temperature;
        let max_tokens = self.max_tokens;
        let reasoning_effort = self.reasoning_effort.clone();
        let tools = self.tools.clone();
        let tool_specs = self.tool_specs.clone();
        let parallel_tools = self.parallel_tools;
        let _tool_concurrency_limit = self.tool_concurrency_limit;
        let join_policy = self.join_policy;
        let instructions = self.instructions.clone();

        Box::pin(async move {
            // Rebuild request using builder to avoid deprecated field access
            let effective_model: Option<String> = req.model.clone().into();

            // Determine which model will be used
            let model_to_use = if let Some(m) = effective_model.as_ref() {
                m.clone()
            } else {
                model.clone()
            };

            // Log model parameters
            debug!(
                model = %model_to_use,
                temperature = ?req.temperature.or(temperature),
                max_tokens = ?max_tokens,
                tools_count = if req.tools.is_some() {
                    req.tools.as_ref().map(|t| t.len())
                } else {
                    Some(tool_specs.len())
                },
                "Step service preparing API request"
            );

            // Prepare messages with optional agent-level instructions injection
            let mut injected_messages = req.messages.clone();
            if let Some(instr) = instructions {
                // Build a system message for the instructions
                let sys_msg = ChatCompletionRequestSystemMessageArgs::default()
                    .content(instr)
                    .build()
                    .map(ChatCompletionRequestMessage::from)
                    .map_err(|e| format!("system msg build error: {}", e))?;
                // Ensure exactly one system message at the front
                if let Some(pos) = injected_messages
                    .iter()
                    .position(|m| matches!(m, ChatCompletionRequestMessage::System(_)))
                {
                    injected_messages.remove(pos);
                }
                injected_messages.insert(0, sys_msg);
            }

            let mut builder = CreateChatCompletionRequestArgs::default();
            builder.messages(injected_messages);
            if let Some(m) = effective_model.as_ref() {
                builder.model(m);
            } else {
                builder.model(&model);
            }
            if let Some(t) = req.temperature.or(temperature) {
                builder.temperature(t);
            }
            // Use request's max_tokens if set, otherwise use layer's max_tokens if set
            #[allow(deprecated)]
            if let Some(mt) = req.max_tokens.or(max_tokens) {
                builder.max_tokens(mt);
            }
            if let Some(effort) = reasoning_effort {
                builder.reasoning_effort(effort);
            }
            if let Some(ts) = req.tools.clone() {
                builder.tools(ts);
            } else if !tool_specs.is_empty() {
                builder.tools((*tool_specs).clone());
            }

            let rebuilt_req = builder
                .build()
                .map_err(|e| format!("request build error: {}", e))?;

            // Trace the final request model
            trace!(
                final_model = ?rebuilt_req.model,
                messages_count = rebuilt_req.messages.len(),
                "Step service final request built"
            );

            let mut messages = rebuilt_req.messages.clone();

            // Single OpenAI call
            // Provider call
            let mut p = provider.lock().await;
            let ProviderResponse {
                assistant,
                prompt_tokens,
                completion_tokens,
            } = ServiceExt::ready(&mut *p).await?.call(rebuilt_req).await?;
            let mut aux = StepAux {
                prompt_tokens,
                completion_tokens,
                tool_invocations: 0,
            };

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
            let invocations: Vec<ToolInvocation> = tool_calls
                .into_iter()
                .map(|tc| {
                    let name = tc.function.name;
                    invoked_names.push(name.clone());
                    let args: Value =
                        serde_json::from_str(&tc.function.arguments).unwrap_or(Value::Null);
                    ToolInvocation {
                        id: tc.id,
                        name,
                        arguments: args,
                    }
                })
                .collect();

            if invocations.len() > 1 && parallel_tools {
                // Fire in parallel, preserve order
                let sem = _tool_concurrency_limit.map(|n| Arc::new(Semaphore::new(n)));
                match join_policy {
                    ToolJoinPolicy::FailFast => {
                        let futures: Vec<_> = invocations
                            .into_iter()
                            .map(|inv| {
                                let mut svc = tools.clone();
                                let sem_cl = sem.clone();
                                async move {
                                    let _permit = match &sem_cl {
                                        Some(s) => Some(
                                            s.clone().acquire_owned().await.expect("semaphore"),
                                        ),
                                        None => None,
                                    };
                                    let ToolOutput { id, result } =
                                        ServiceExt::ready(&mut svc).await?.call(inv).await?;
                                    Ok::<(String, Value), BoxError>((id, result))
                                }
                            })
                            .collect();
                        let outputs: Vec<(String, Value)> =
                            futures::future::try_join_all(futures).await?;
                        for (id, result) in outputs {
                            aux.tool_invocations += 1;
                            let tool_msg = ChatCompletionRequestToolMessageArgs::default()
                                .content(result.to_string())
                                .tool_call_id(id)
                                .build()?;
                            messages.push(tool_msg.into());
                        }
                    }
                    ToolJoinPolicy::JoinAll => {
                        let futures: Vec<_> =
                            invocations
                                .into_iter()
                                .enumerate()
                                .map(|(idx, inv)| {
                                    let mut svc = tools.clone();
                                    let sem_cl = sem.clone();
                                    async move {
                                        let _permit = match &sem_cl {
                                            Some(s) => Some(
                                                s.clone().acquire_owned().await.expect("semaphore"),
                                            ),
                                            None => None,
                                        };
                                        let res =
                                            ServiceExt::ready(&mut svc).await?.call(inv).await;
                                        match res {
                                            Ok(ToolOutput { id, result }) => Ok::<
                                                Result<(usize, String, Value), BoxError>,
                                                BoxError,
                                            >(
                                                Ok((idx, id, result)),
                                            ),
                                            Err(e) => Ok(Err(e)),
                                        }
                                    }
                                })
                                .collect();
                        let results = futures::future::join_all(futures).await;
                        let mut successes: Vec<(usize, String, Value)> = Vec::new();
                        let mut errors: Vec<String> = Vec::new();
                        for item in results.into_iter() {
                            match item {
                                Ok(Ok((idx, id, result))) => successes.push((idx, id, result)),
                                Ok(Err(e)) => errors.push(format!("{}", e)),
                                Err(e) => errors.push(format!("{}", e)),
                            }
                        }
                        successes.sort_by_key(|(idx, _, _)| *idx);
                        for (_idx, id, result) in successes.into_iter() {
                            aux.tool_invocations += 1;
                            let tool_msg = ChatCompletionRequestToolMessageArgs::default()
                                .content(result.to_string())
                                .tool_call_id(id)
                                .build()?;
                            messages.push(tool_msg.into());
                        }
                        if !errors.is_empty() {
                            return Err(
                                format!("one or more tools failed: {}", errors.join("; ")).into()
                            );
                        }
                    }
                }
            } else {
                // Sequential
                for inv in invocations {
                    let mut svc = tools.clone();
                    let ToolOutput { id, result } =
                        ServiceExt::ready(&mut svc).await?.call(inv).await?;
                    aux.tool_invocations += 1;
                    let tool_msg = ChatCompletionRequestToolMessageArgs::default()
                        .content(result.to_string())
                        .tool_call_id(id)
                        .build()?;
                    messages.push(tool_msg.into());
                }
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

/// Build a simple chat request with only a user message.
#[allow(dead_code)]
pub fn simple_user_request(user: &str) -> CreateChatCompletionRequest {
    let usr = ChatCompletionRequestUserMessageArgs::default()
        .content(user)
        .build()
        .expect("user msg");
    CreateChatCompletionRequestArgs::default()
        .model("gpt-4o")
        .messages(vec![usr.into()])
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
    reasoning_effort: Option<ReasoningEffort>,
    instructions: Option<String>,
    tools: Vec<ToolDef>,
    policy: CompositePolicy,
    handoff: Option<crate::groups::AnyHandoffPolicy>,
    provider: Option<
        tower::util::BoxCloneService<
            CreateChatCompletionRequest,
            crate::provider::ProviderResponse,
            BoxError,
        >,
    >,
    enable_parallel_tools: bool,
    tool_concurrency_limit: Option<usize>,
    tool_join_policy: ToolJoinPolicy,
    agent_service_map: Option<Arc<dyn Fn(AgentSvc) -> AgentSvc + Send + Sync + 'static>>, // optional final wrapper
    auto_compaction: Option<crate::auto_compaction::CompactionPolicy>,
}

impl Agent {
    pub fn builder(client: Arc<Client<OpenAIConfig>>) -> AgentBuilder {
        AgentBuilder {
            client,
            model: "gpt-4o".to_string(),
            temperature: None,
            max_tokens: None,
            reasoning_effort: None,
            instructions: None,
            tools: Vec::new(),
            policy: CompositePolicy::default(),
            handoff: None,
            provider: None,
            enable_parallel_tools: false,
            tool_concurrency_limit: None,
            tool_join_policy: ToolJoinPolicy::FailFast,
            agent_service_map: None,
            auto_compaction: None,
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
    pub fn reasoning_effort(mut self, effort: ReasoningEffort) -> Self {
        self.reasoning_effort = Some(effort);
        self
    }

    /// Set agent-level instructions (system prompt). These will be injected on each step.
    pub fn instructions(mut self, text: impl Into<String>) -> Self {
        self.instructions = Some(text.into());
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

    /// Enable handoff-aware tool interception and advertise handoff tools
    pub fn handoff_policy(mut self, policy: crate::groups::AnyHandoffPolicy) -> Self {
        self.handoff = Some(policy);
        self
    }

    /// Override the non-streaming provider (useful for testing with a fixed/mocked model)
    pub fn with_provider<P>(mut self, provider: P) -> Self
    where
        P: crate::provider::ModelService + Clone + Send + 'static,
        P::Future: Send + 'static,
    {
        self.provider = Some(tower::util::BoxCloneService::new(provider));
        self
    }

    /// Enable or disable parallel tool execution within a step
    pub fn parallel_tools(mut self, enabled: bool) -> Self {
        self.enable_parallel_tools = enabled;
        self
    }

    /// Set an optional concurrency limit for parallel tool execution
    pub fn tool_concurrency_limit(mut self, limit: usize) -> Self {
        self.tool_concurrency_limit = Some(limit);
        self
    }

    /// Configure how parallel tool errors are handled (fail fast or join all)
    pub fn tool_join_policy(mut self, policy: ToolJoinPolicy) -> Self {
        self.tool_join_policy = policy;
        self
    }

    /// Optional: wrap the final built agent service with a custom function.
    /// This enables applying Tower layers at the agent boundary.
    pub fn map_agent_service<F>(mut self, f: F) -> Self
    where
        F: Fn(AgentSvc) -> AgentSvc + Send + Sync + 'static,
    {
        self.agent_service_map = Some(Arc::new(f));
        self
    }

    /// Enable auto-compaction with the specified policy
    pub fn auto_compaction(mut self, policy: crate::auto_compaction::CompactionPolicy) -> Self {
        self.auto_compaction = Some(policy);
        self
    }

    pub fn build(self) -> AgentSvc {
        let (router, mut specs) = ToolRouter::new(self.tools);
        // If handoff policy provided, wrap router and extend tool specs
        let routed: ToolSvc = if let Some(policy) = &self.handoff {
            let hand_spec = policy.handoff_tools();
            if !hand_spec.is_empty() {
                specs.extend(hand_spec);
            }
            crate::groups::layer_tool_router_with_handoff(router, policy.clone())
        } else {
            // No handoff layer; clonable box of the router
            BoxCloneService::new(router)
        };

        let base_provider: tower::util::BoxCloneService<
            CreateChatCompletionRequest,
            crate::provider::ProviderResponse,
            BoxError,
        > = if let Some(p) = self.provider {
            p
        } else {
            tower::util::BoxCloneService::new(OpenAIProvider::new(self.client))
        };
        let mut step_layer = StepLayer::new(base_provider.clone(), self.model, specs)
            .parallel_tools(self.enable_parallel_tools)
            .tool_join_policy(self.tool_join_policy);
        if let Some(instr) = &self.instructions {
            step_layer = step_layer.instructions(instr.clone());
        }
        // Only set temperature if explicitly provided
        if let Some(t) = self.temperature {
            step_layer = step_layer.temperature(t);
        }
        // Only set max_tokens if explicitly provided
        if let Some(mt) = self.max_tokens {
            step_layer = step_layer.max_tokens(mt);
        }
        if let Some(effort) = self.reasoning_effort {
            step_layer = step_layer.reasoning_effort(effort);
        }
        if let Some(lim) = self.tool_concurrency_limit {
            step_layer = step_layer.tool_concurrency_limit(lim);
        }
        let step = step_layer.layer(routed);

        // Apply auto-compaction if configured
        let step_with_compaction: BoxService<CreateChatCompletionRequest, StepOutcome, BoxError> =
            if let Some(compaction_policy) = self.auto_compaction {
                // Use the provider for compaction
                let compaction_provider = base_provider;
                let token_counter = crate::auto_compaction::SimpleTokenCounter::new();
                let compaction_layer = crate::auto_compaction::AutoCompactionLayer::new(
                    compaction_policy,
                    compaction_provider,
                    token_counter,
                );
                BoxService::new(compaction_layer.layer(step))
            } else {
                BoxService::new(step)
            };

        let agent = AgentLoopLayer::new(self.policy).layer(step_with_compaction);
        let boxed = BoxService::new(agent);
        match &self.agent_service_map {
            Some(map) => (map)(boxed),
            None => boxed,
        }
    }

    /// Build an agent service wrapped with session memory persistence
    pub fn build_with_session<Ls, Ss>(
        self,
        load: Arc<Ls>,
        save: Arc<Ss>,
        session_id: crate::sessions::SessionId,
    ) -> AgentSvc
    where
        Ls: Service<
                crate::sessions::LoadSession,
                Response = crate::sessions::History,
                Error = BoxError,
            > + Send
            + Sync
            + Clone
            + 'static,
        Ls::Future: Send + 'static,
        Ss: Service<crate::sessions::SaveSession, Response = (), Error = BoxError>
            + Send
            + Sync
            + Clone
            + 'static,
        Ss::Future: Send + 'static,
    {
        let (router, mut specs) = ToolRouter::new(self.tools);
        let routed: ToolSvc = if let Some(policy) = &self.handoff {
            let hand_spec = policy.handoff_tools();
            if !hand_spec.is_empty() {
                specs.extend(hand_spec);
            }
            crate::groups::layer_tool_router_with_handoff(router, policy.clone())
        } else {
            BoxCloneService::new(router)
        };

        let base_provider: tower::util::BoxCloneService<
            CreateChatCompletionRequest,
            crate::provider::ProviderResponse,
            BoxError,
        > = if let Some(p) = self.provider {
            p
        } else {
            tower::util::BoxCloneService::new(OpenAIProvider::new(self.client))
        };
        let mut step_layer = StepLayer::new(base_provider.clone(), self.model, specs)
            .parallel_tools(self.enable_parallel_tools)
            .tool_join_policy(self.tool_join_policy);
        if let Some(instr) = &self.instructions {
            step_layer = step_layer.instructions(instr.clone());
        }
        // Only set temperature if explicitly provided
        if let Some(t) = self.temperature {
            step_layer = step_layer.temperature(t);
        }
        // Only set max_tokens if explicitly provided
        if let Some(mt) = self.max_tokens {
            step_layer = step_layer.max_tokens(mt);
        }
        if let Some(effort) = self.reasoning_effort {
            step_layer = step_layer.reasoning_effort(effort);
        }
        if let Some(lim) = self.tool_concurrency_limit {
            step_layer = step_layer.tool_concurrency_limit(lim);
        }
        let step = step_layer.layer(routed);

        // Apply auto-compaction if configured (before memory layer)
        let step_with_compaction: BoxService<CreateChatCompletionRequest, StepOutcome, BoxError> =
            if let Some(compaction_policy) = self.auto_compaction {
                // Use the provider for compaction
                let compaction_provider = base_provider;
                let token_counter = crate::auto_compaction::SimpleTokenCounter::new();
                let compaction_layer = crate::auto_compaction::AutoCompactionLayer::new(
                    compaction_policy,
                    compaction_provider,
                    token_counter,
                );
                BoxService::new(compaction_layer.layer(step))
            } else {
                BoxService::new(step)
            };

        // Attach memory layer
        let mem_layer = crate::sessions::MemoryLayer::new(load, save, session_id);
        let step_with_mem = mem_layer.layer(step_with_compaction);
        let agent = AgentLoopLayer::new(self.policy).layer(step_with_mem);
        let boxed = BoxService::new(agent);
        match &self.agent_service_map {
            Some(map) => (map)(boxed),
            None => boxed,
        }
    }
}

/// Convenience: run a prompt through an agent service.
pub async fn run(agent: &mut AgentSvc, system: &str, user: &str) -> Result<AgentRun, BoxError> {
    let req = simple_chat_request(system, user);
    let resp = ServiceExt::ready(agent).await?.call(req).await?;
    Ok(resp)
}

/// Convenience: run a user message through an agent service. System instructions come from the agent.
#[allow(dead_code)]
pub async fn run_user(agent: &mut AgentSvc, user: &str) -> Result<AgentRun, BoxError> {
    let req = simple_user_request(user);
    let resp = ServiceExt::ready(agent).await?.call(req).await?;
    Ok(resp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_openai::types::ChatCompletionRequestUserMessageArgs;

    #[tokio::test]
    async fn step_injects_instructions_prepend_or_replace() {
        // Provider that echoes back with minimal tokens
        #[allow(deprecated)]
        let assistant = async_openai::types::ChatCompletionResponseMessage {
            content: Some("ok".into()),
            role: async_openai::types::Role::Assistant,
            tool_calls: None,
            function_call: None,
            refusal: None,
            audio: None,
        };
        let provider = crate::provider::FixedProvider::new(crate::provider::ProviderResponse {
            assistant,
            prompt_tokens: 1,
            completion_tokens: 1,
        });

        // No tools
        let (router, specs) = ToolRouter::new(vec![]);
        let step = StepLayer::new(provider, "gpt-4o", specs)
            .instructions("AGENT INSTR")
            .layer(router);
        let mut svc = tower::ServiceExt::boxed(step);

        // Build request with only user message
        let user = ChatCompletionRequestUserMessageArgs::default()
            .content("hello")
            .build()
            .unwrap();
        let req = CreateChatCompletionRequestArgs::default()
            .model("gpt-4o")
            .messages(vec![user.into()])
            .build()
            .unwrap();

        let out = tower::ServiceExt::ready(&mut svc)
            .await
            .unwrap()
            .call(req)
            .await
            .unwrap();
        let msgs = match out {
            StepOutcome::Next { messages, .. } => messages,
            StepOutcome::Done { messages, .. } => messages,
        };
        // First message must be system with injected content
        match &msgs[0] {
            ChatCompletionRequestMessage::System(s) => {
                if let async_openai::types::ChatCompletionRequestSystemMessageContent::Text(t) =
                    &s.content
                {
                    assert_eq!(t, "AGENT INSTR");
                } else {
                    panic!("expected text content in system message");
                }
            }
            _ => panic!("expected first message to be system"),
        }
    }

    #[test]
    fn builds_user_request() {
        let _ = simple_user_request("hi");
    }

    #[tokio::test]
    async fn run_user_executes_with_instructions() {
        #[allow(deprecated)]
        let assistant = async_openai::types::ChatCompletionResponseMessage {
            content: Some("ok".into()),
            role: async_openai::types::Role::Assistant,
            tool_calls: None,
            function_call: None,
            refusal: None,
            audio: None,
        };
        let provider = crate::provider::FixedProvider::new(crate::provider::ProviderResponse {
            assistant,
            prompt_tokens: 1,
            completion_tokens: 1,
        });
        let client =
            std::sync::Arc::new(async_openai::Client::<async_openai::config::OpenAIConfig>::new());
        let mut agent = Agent::builder(client)
            .with_provider(provider)
            .model("gpt-4o")
            .instructions("INSTR")
            .policy(CompositePolicy::new(vec![policies::max_steps(1)]))
            .build();
        let run = run_user(&mut agent, "hello").await.unwrap();
        assert!(!run.messages.is_empty());
        match &run.messages[0] {
            ChatCompletionRequestMessage::System(s) => match &s.content {
                async_openai::types::ChatCompletionRequestSystemMessageContent::Text(t) => {
                    assert_eq!(t, "INSTR");
                }
                _ => panic!("expected text content"),
            },
            _ => panic!("expected first message to be system"),
        }
    }

    #[tokio::test]
    async fn sessions_preserve_agent_instructions_in_merged_request() {
        use crate::sessions::{InMemorySessionStore, SessionId};
        // Capturing provider
        #[derive(Clone)]
        struct CapturingProvider {
            captured: std::sync::Arc<tokio::sync::Mutex<Option<CreateChatCompletionRequest>>>,
        }
        impl tower::Service<CreateChatCompletionRequest> for CapturingProvider {
            type Response = crate::provider::ProviderResponse;
            type Error = BoxError;
            type Future = std::pin::Pin<
                Box<dyn std::future::Future<Output = Result<Self::Response, Self::Error>> + Send>,
            >;
            fn poll_ready(
                &mut self,
                _cx: &mut std::task::Context<'_>,
            ) -> std::task::Poll<Result<(), Self::Error>> {
                std::task::Poll::Ready(Ok(()))
            }
            fn call(&mut self, req: CreateChatCompletionRequest) -> Self::Future {
                let captured = self.captured.clone();
                Box::pin(async move {
                    *captured.lock().await = Some(req);
                    #[allow(deprecated)]
                    let assistant = async_openai::types::ChatCompletionResponseMessage {
                        content: Some("ok".into()),
                        role: async_openai::types::Role::Assistant,
                        tool_calls: None,
                        function_call: None,
                        refusal: None,
                        audio: None,
                    };
                    Ok(crate::provider::ProviderResponse {
                        assistant,
                        prompt_tokens: 1,
                        completion_tokens: 1,
                    })
                })
            }
        }

        let captured = std::sync::Arc::new(tokio::sync::Mutex::new(None));
        let provider = CapturingProvider {
            captured: captured.clone(),
        };

        let client =
            std::sync::Arc::new(async_openai::Client::<async_openai::config::OpenAIConfig>::new());
        let load = std::sync::Arc::new(InMemorySessionStore::default());
        let save = load.clone();
        let mut agent = Agent::builder(client)
            .with_provider(provider)
            .model("gpt-4o")
            .instructions("INSTR")
            .policy(CompositePolicy::new(vec![policies::max_steps(1)]))
            .build_with_session(load, save, SessionId("s1".into()));

        let req = CreateChatCompletionRequestArgs::default()
            .model("gpt-4o")
            .messages(vec![ChatCompletionRequestUserMessageArgs::default()
                .content("hi")
                .build()
                .unwrap()
                .into()])
            .build()
            .unwrap();
        let _ = tower::ServiceExt::ready(&mut agent)
            .await
            .unwrap()
            .call(req)
            .await
            .unwrap();

        let got = captured.lock().await.clone().expect("captured");
        match &got.messages[0] {
            ChatCompletionRequestMessage::System(s) => match &s.content {
                async_openai::types::ChatCompletionRequestSystemMessageContent::Text(t) => {
                    assert_eq!(t, "INSTR");
                }
                _ => panic!("expected text content"),
            },
            _ => panic!("expected first message to be system"),
        }
    }

    #[tokio::test]
    async fn auto_compaction_preserves_instructions() {
        use crate::auto_compaction::{CompactionPolicy, CompactionStrategy, ProactiveThreshold};
        // Capturing provider
        #[derive(Clone)]
        struct CapturingProvider {
            captured: std::sync::Arc<tokio::sync::Mutex<Option<CreateChatCompletionRequest>>>,
        }
        impl tower::Service<CreateChatCompletionRequest> for CapturingProvider {
            type Response = crate::provider::ProviderResponse;
            type Error = BoxError;
            type Future = std::pin::Pin<
                Box<dyn std::future::Future<Output = Result<Self::Response, Self::Error>> + Send>,
            >;
            fn poll_ready(
                &mut self,
                _cx: &mut std::task::Context<'_>,
            ) -> std::task::Poll<Result<(), Self::Error>> {
                std::task::Poll::Ready(Ok(()))
            }
            fn call(&mut self, req: CreateChatCompletionRequest) -> Self::Future {
                let captured = self.captured.clone();
                Box::pin(async move {
                    *captured.lock().await = Some(req);
                    #[allow(deprecated)]
                    let assistant = async_openai::types::ChatCompletionResponseMessage {
                        content: Some("ok".into()),
                        role: async_openai::types::Role::Assistant,
                        tool_calls: None,
                        function_call: None,
                        refusal: None,
                        audio: None,
                    };
                    Ok(crate::provider::ProviderResponse {
                        assistant,
                        prompt_tokens: 1,
                        completion_tokens: 1,
                    })
                })
            }
        }

        let captured = std::sync::Arc::new(tokio::sync::Mutex::new(None));
        let provider = CapturingProvider {
            captured: captured.clone(),
        };
        let client =
            std::sync::Arc::new(async_openai::Client::<async_openai::config::OpenAIConfig>::new());

        let policy = CompactionPolicy {
            compaction_model: "gpt-4o-mini".to_string(),
            proactive_threshold: Some(ProactiveThreshold {
                token_threshold: 1,
                percentage_threshold: None,
            }),
            compaction_strategy: CompactionStrategy::PreserveSystemAndRecent { recent_count: 1 },
            ..Default::default()
        };

        let mut agent = Agent::builder(client)
            .with_provider(provider)
            .model("gpt-4o")
            .instructions("INSTR")
            .auto_compaction(policy)
            .policy(CompositePolicy::new(vec![policies::max_steps(1)]))
            .build();

        let mut long_user = String::new();
        for _ in 0..200 {
            long_user.push('x');
        }
        let req = CreateChatCompletionRequestArgs::default()
            .model("gpt-4o")
            .messages(vec![ChatCompletionRequestUserMessageArgs::default()
                .content(long_user)
                .build()
                .unwrap()
                .into()])
            .build()
            .unwrap();
        let _ = tower::ServiceExt::ready(&mut agent)
            .await
            .unwrap()
            .call(req)
            .await
            .unwrap();

        let got = captured.lock().await.clone().expect("captured");
        match &got.messages[0] {
            ChatCompletionRequestMessage::System(s) => match &s.content {
                async_openai::types::ChatCompletionRequestSystemMessageContent::Text(t) => {
                    assert_eq!(t, "INSTR");
                }
                _ => panic!("expected text content"),
            },
            _ => panic!("expected first message to be system"),
        }
    }
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
#[allow(clippy::type_complexity)]
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
            // Preserve all original request parameters
            let base_temperature = req.temperature;
            #[allow(deprecated)]
            let base_max_tokens = req.max_tokens;
            let base_max_completion_tokens = req.max_completion_tokens;
            let base_tools = req.tools.clone();

            // Log the initial model for the agent loop
            debug!(
                model = ?base_model,
                initial_messages = current_messages.len(),
                "AgentLoop starting with model"
            );

            loop {
                // Rebuild request for this iteration, preserving original parameters
                let mut builder = CreateChatCompletionRequestArgs::default();
                builder.model(&base_model);
                builder.messages(current_messages.clone());
                if let Some(t) = base_temperature {
                    builder.temperature(t);
                }
                if let Some(mt) = base_max_tokens {
                    builder.max_tokens(mt);
                }
                if let Some(mct) = base_max_completion_tokens {
                    builder.max_completion_tokens(mct);
                }
                if let Some(tools) = base_tools.clone() {
                    builder.tools(tools);
                }
                let current_req = builder
                    .build()
                    .map_err(|e| format!("build req error: {}", e))?;

                trace!(
                    step = state.steps + 1,
                    model = ?current_req.model,
                    messages = current_messages.len(),
                    "AgentLoop iteration"
                );

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
