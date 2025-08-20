//! # Runner (orientation)
//!
//! The `Runner` coordinates an agent run: it interacts with the model, routes
//! tool-calls through the Tower stack (Run → Agent → Tool → BaseTool), and
//! maintains ordering and state across turns and handoffs. Policy layers and
//! tool execution are composed in `service.rs`; this module focuses on orchestration.

use std::any::Any;
use std::sync::{Arc, Mutex};
use tracing::{debug, info};

use crate::agent::Agent;

use crate::error::{AgentsError, Result};
use crate::guardrail::GuardrailRunner;
use crate::items::{
    HandoffItem, Message, MessageItem, Role, RunItem, ToolCallItem, ToolOutputItem,
};
use crate::memory::Session;
use crate::model::ModelProvider;
use crate::result::{RunResult, StreamEvent, StreamingRunResult};
use crate::service::{DefaultEnv, Effect, ToolRequest};
use crate::tracing::{AgentSpan, GenerationSpan, ToolSpan, TracingContext};
use crate::usage::UsageStats;
use futures::future::join_all;
use tower::ServiceExt; // for oneshot

fn truncate_for_log(s: &str, max: usize) -> String {
    if s.len() > max {
        let mut out = s[..max].to_string();
        out.push('…');
        out
    } else {
        s.to_string()
    }
}

fn format_messages_for_log(messages: &[Message]) -> String {
    let mut lines = Vec::new();
    for (idx, m) in messages.iter().enumerate() {
        match m.role {
            Role::User => {
                lines.push(format!(
                    "{:02} USER     | {}",
                    idx,
                    truncate_for_log(&m.content, 160)
                ));
            }
            Role::System => {
                lines.push(format!(
                    "{:02} SYSTEM   | {}",
                    idx,
                    truncate_for_log(&m.content, 160)
                ));
            }
            Role::Assistant => {
                if let Some(tool_calls) = &m.tool_calls {
                    let calls: Vec<String> = tool_calls
                        .iter()
                        .map(|tc| format!("id={}, name={}", tc.id, tc.name))
                        .collect();
                    lines.push(format!(
                        "{:02} ASSIST   | tool_calls=[{}] content=\"{}\"",
                        idx,
                        calls.join(", "),
                        truncate_for_log(&m.content, 120)
                    ));
                } else {
                    lines.push(format!(
                        "{:02} ASSIST   | {}",
                        idx,
                        truncate_for_log(&m.content, 160)
                    ));
                }
            }
            Role::Tool => {
                let tcid = m
                    .tool_call_id
                    .as_deref()
                    .unwrap_or("<missing tool_call_id>");
                lines.push(format!(
                    "{:02} TOOL     | tool_call_id={} payload={}",
                    idx,
                    tcid,
                    truncate_for_log(&m.content, 120)
                ));
            }
        }
    }
    lines.join("\n")
}

/// Configuration for an agent run.
///
/// `RunConfig` provides the necessary settings to control the execution of an
/// agent. It allows you to specify the maximum number of turns, enable or
/// disable streaming, and provide a session for persistent conversation history.
///
/// ## Example
///
/// ```rust
/// use openai_agents_rs::runner::RunConfig;
/// use openai_agents_rs::sqlite_session::SqliteSession;
/// use std::sync::Arc;
///
/// # async fn config() -> Result<(), Box<dyn std::error::Error>> {
/// // Create a session to store conversation history.
/// let session = Arc::new(SqliteSession::new("test_session", "test_session.db").await?);
///
/// // Configure the run to use the session and limit the conversation to 5 turns.
/// let config = RunConfig {
///     max_turns: Some(5),
///     session: Some(session),
///     ..Default::default()
/// };
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct RunConfig {
    /// The maximum number of turns (LLM calls) to execute before stopping the
    /// run. This is a safeguard against infinite loops and excessive token usage.
    /// If not set, the agent's default `max_turns` will be used, which defaults
    /// to 10.
    pub max_turns: Option<usize>,

    /// A boolean flag to determine whether to stream events during the run.
    /// When `true`, the `run_stream` method should be used to receive real-time
    /// updates as the agent executes.
    pub stream: bool,

    /// An optional session to manage the conversation history. The session is
    /// responsible for loading previous messages and saving new ones. If not
    /// provided, the conversation will be stateless.
    pub session: Option<Arc<dyn Session>>,

    /// The model provider to use for generating responses. If not provided,
    /// a default `OpenAIProvider` will be used in production, or a `MockProvider`
    /// in test environments.
    pub model_provider: Option<Arc<dyn ModelProvider>>,

    /// Whether to execute tool calls in parallel within a single turn.
    /// Defaults to true.
    pub parallel_tools: bool,

    /// Optional maximum number of concurrent tool calls when `parallel_tools` is true.
    /// If `None`, no explicit limit is enforced.
    pub max_concurrency: Option<usize>,

    /// Deprecated: Use typed `.layer()` API instead.
    /// Step 8: ErasedToolLayer removed - this field kept for backward compatibility.
    #[deprecated(note = "Use typed .layer() API instead")]
    pub run_layers: Vec<()>, // Empty vector to maintain API
}

impl std::fmt::Debug for RunConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunConfig")
            .field("max_turns", &self.max_turns)
            .field("stream", &self.stream)
            .field("session", &self.session.is_some())
            .field("model_provider", &self.model_provider.is_some())
            .finish()
    }
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            max_turns: Some(10),
            stream: false,
            session: None,
            model_provider: None,

            parallel_tools: true,
            max_concurrency: None,
            run_layers: Vec::new(),
        }
    }
}

impl RunConfig {
    // Context handlers have been removed in favor of Tower layers
    // Use layers attached to tools, agents, or runs instead

    /// Convenience: set a model provider.
    pub fn with_model_provider(mut self, provider: Option<Arc<dyn ModelProvider>>) -> Self {
        self.model_provider = provider;
        self
    }

    /// Toggle parallel execution of tool calls within a single turn.
    pub fn with_parallel_tools(mut self, enabled: bool) -> Self {
        self.parallel_tools = enabled;
        self
    }

    /// Set maximum number of concurrent tool calls when running in parallel.
    pub fn with_max_concurrency(mut self, limit: usize) -> Self {
        self.max_concurrency = Some(limit.max(1));
        self
    }

    /// Attach dynamic run-scope policy layers (scope-agnostic, applied at run scope).
    ///
    /// **Deprecated**: Use `.layer()` instead for type-safe composition.
    ///
    /// # Migration
    /// ```text
    /// // Old:
    /// RunConfig::default().with_run_layers(vec![layers::boxed_timeout_secs(30)])
    ///
    /// // New:
    /// RunConfig::default().layer(TimeoutLayer::secs(30))
    /// ```
    #[deprecated(since = "0.2.0", note = "Use `.layer()` for typed composition instead")]
    pub fn with_run_layers(
        mut self,
        _layers: Vec<()>, // No-op: ErasedToolLayer removed in Step 8
    ) -> Self {
        // No-op: ErasedToolLayer removed in Step 8
        self
    }

    /// Apply a typed layer to this run config, returning a typed wrapper.
    ///
    /// This is the preferred API over `with_run_layers()` as it provides
    /// compile-time type safety and follows Tower's fluent composition pattern.
    ///
    /// # Example
    /// ```rust,ignore
    /// use openai_agents_rs::{runner::RunConfig, service::{TimeoutLayer, RetryLayer}};
    /// use std::time::Duration;
    ///
    /// let config = RunConfig::default()
    ///     .layer(TimeoutLayer::from_duration(Duration::from_secs(30)))
    ///     .layer(RetryLayer::times(3));
    /// ```
    pub fn layer<L>(self, layer: L) -> LayeredRunConfig<L, Self> {
        LayeredRunConfig::new(layer, self)
    }
}

/// A typed wrapper for a run config with layers applied.
///
/// This follows Tower's `Layered` pattern, providing compile-time type safety
/// for layer composition while maintaining the RunConfig interface.
#[derive(Clone)]
pub struct LayeredRunConfig<L, R> {
    layer: L,
    inner: R,
}

impl<L, R> LayeredRunConfig<L, R> {
    /// Create a new layered run config.
    pub fn new(layer: L, inner: R) -> Self {
        Self { layer, inner }
    }

    /// Apply another layer, returning a new typed wrapper.
    pub fn layer<L2>(self, layer: L2) -> LayeredRunConfig<L2, Self> {
        LayeredRunConfig::new(layer, self)
    }
}

impl<L, R> LayeredRunConfig<L, R>
where
    R: RunConfigLike + Clone,
{
    /// Get the underlying run config.
    /// This allows the layered config to be used anywhere a RunConfig is expected.
    pub fn inner_config(&self) -> R {
        self.inner.clone()
    }
}

/// Trait to allow both RunConfig and LayeredRunConfig to be used interchangeably
pub trait RunConfigLike {
    fn max_turns(&self) -> Option<usize>;
    fn stream(&self) -> bool;
    fn session(&self) -> &Option<Arc<dyn crate::memory::Session>>;
    fn model_provider(&self) -> &Option<Arc<dyn crate::model::ModelProvider>>;
    fn parallel_tools(&self) -> bool;
    fn max_concurrency(&self) -> Option<usize>;
    fn run_layers(&self) -> &Vec<()>; // ErasedToolLayer removed in Step 8
}

impl RunConfigLike for RunConfig {
    fn max_turns(&self) -> Option<usize> {
        self.max_turns
    }

    fn stream(&self) -> bool {
        self.stream
    }

    fn session(&self) -> &Option<Arc<dyn crate::memory::Session>> {
        &self.session
    }

    fn model_provider(&self) -> &Option<Arc<dyn crate::model::ModelProvider>> {
        &self.model_provider
    }

    fn parallel_tools(&self) -> bool {
        self.parallel_tools
    }

    fn max_concurrency(&self) -> Option<usize> {
        self.max_concurrency
    }

    fn run_layers(&self) -> &Vec<()> {
        // ErasedToolLayer removed in Step 8
        &self.run_layers
    }
}

impl<L, R: RunConfigLike> RunConfigLike for LayeredRunConfig<L, R> {
    fn max_turns(&self) -> Option<usize> {
        self.inner.max_turns()
    }

    fn stream(&self) -> bool {
        self.inner.stream()
    }

    fn session(&self) -> &Option<Arc<dyn crate::memory::Session>> {
        self.inner.session()
    }

    fn model_provider(&self) -> &Option<Arc<dyn crate::model::ModelProvider>> {
        self.inner.model_provider()
    }

    fn parallel_tools(&self) -> bool {
        self.inner.parallel_tools()
    }

    fn max_concurrency(&self) -> Option<usize> {
        self.inner.max_concurrency()
    }

    fn run_layers(&self) -> &Vec<()> {
        // ErasedToolLayer removed in Step 8
        self.inner.run_layers()
    }
}

/// The main runner for executing agents.
///
/// `Runner` provides a set of static methods to run an agent with different
/// execution strategies:
///
/// - **[`run`]**: Executes the agent asynchronously and returns the final result
///   once the run is complete.
/// - **[`run_sync`]**: A blocking version of `run` that executes the agent
///   synchronously and blocks until the run is finished.
/// - **[`run_stream`]**: Executes the agent and returns a stream of events
///   as the run progresses.
///
/// The runner is stateless and can be used to execute multiple agents concurrently.
///
/// ## Example: Running an Agent
///
/// ```text
/// use openai_agents_rs::{Agent, Runner, runner::RunConfig};
///
/// # async fn run_agent() -> Result<(), Box<dyn std::error::Error>> {
/// let agent = Agent::simple(
///     "EchoAgent",
///     "You are an agent that echoes the user's input."
/// );
///
/// let result = Runner::run(
///     agent,
///     "Hello, world!",
///     RunConfig::default()
/// ).await?;
///
/// if result.is_success() {
///     assert!(result.final_output.as_str().unwrap().contains("Hello, world!"));
/// }
/// # Ok(())
/// # }
/// ```
///
/// [`run`]: Self::run
/// [`run_sync`]: Self::run_sync
/// [`run_stream`]: Self::run_stream
pub struct Runner;

impl Runner {
    /// Executes an agent asynchronously and returns the result.
    ///
    /// This is the primary method for running an agent. It orchestrates the
    /// entire agent loop, including handling tool calls, applying guardrails,
    /// and managing state. The run will continue until the agent produces a
    /// final response or an error occurs.
    ///
    /// # Arguments
    ///
    /// * `agent` - The [`Agent`] to be executed.
    /// * `input` - The user's input to start the conversation.
    /// * `config` - The [`RunConfig`] to control the execution.
    pub async fn run(
        agent: Agent,
        input: impl Into<String>,
        config: RunConfig,
    ) -> Result<RunResult> {
        Self::run_with_env(agent, input, config, DefaultEnv).await
    }

    /// Executes an agent with a custom environment and returns the result.
    ///
    /// This method allows you to provide a custom environment with capabilities
    /// that can be accessed by layers (e.g., ApprovalLayer requiring approval
    /// capability). This enables the full Tower-based architecture with
    /// capability-driven policies.
    ///
    /// # Arguments
    ///
    /// * `agent` - The [`Agent`] to be executed.
    /// * `input` - The user's input to start the conversation.
    /// * `config` - The [`RunConfig`] to control the execution.
    /// * `env` - The custom environment providing capabilities.
    ///
    /// # Example
    ///
    /// ```text
    /// use openai_agents_rs::{Agent, Runner, runner::RunConfig, env::EnvBuilder};
    /// use std::sync::Arc;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let env = EnvBuilder::new()
    ///     .with_capability(Arc::new(MyApprovalCapability))
    ///     .build();
    ///
    /// let agent = Agent::simple("Bot", "I need approval")
    ///     .layer(openai_agents_rs::layers::ApprovalLayer);
    ///
    /// let result = Runner::run_with_env(
    ///     agent,
    ///     "Do something that needs approval",
    ///     RunConfig::default(),
    ///     env
    /// ).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn run_with_env<E: crate::env::Env>(
        agent: Agent,
        input: impl Into<String>,
        config: RunConfig,
        env: E,
    ) -> Result<RunResult> {
        let input = input.into();
        info!(agent = %agent.name(), "Starting agent run");

        let context = Arc::new(Mutex::new(TracingContext::new()));
        let _trace_id = context.lock().unwrap().trace_id().to_string();

        // Run input guardrails
        if !agent.config.input_guardrails.is_empty() {
            GuardrailRunner::check_input(&agent.config.input_guardrails, &input).await?;
        }

        // Initialize conversation with session history if available
        let mut messages = vec![agent.build_system_message()];

        if let Some(session) = &config.session {
            let history = session.get_messages(None).await?;
            messages.extend(history);
        }

        messages.push(Message::user(input));

        // Run the agent loop with custom environment
        let (result, _state, _run_state) =
            Self::run_loop_with_env(agent, messages, config.clone(), context.clone(), env).await?;

        // Save to session if configured
        if let Some(session) = &config.session {
            session.add_items(result.items.clone()).await?;
        }

        Ok(result)
    }

    // Context-based run methods have been removed in favor of Tower layers

    // Context-based run methods have been removed in favor of Tower layers
    // Use layers attached to tools, agents, or runs instead

    /// Executes an agent synchronously and blocks until the result is available.
    ///
    /// This method is a convenience wrapper around the `run` method, providing
    /// a blocking alternative for use cases where an async runtime is not
    /// readily available. It creates a new Tokio runtime to execute the agent.
    pub fn run_sync(
        agent: Agent,
        input: impl Into<String>,
        config: RunConfig,
    ) -> Result<RunResult> {
        let runtime = tokio::runtime::Runtime::new()?;
        runtime.block_on(Self::run(agent, input, config))
    }

    /// Executes an agent and returns a stream of events.
    ///
    /// This method is designed for real-time applications where you want to
    /// receive updates as the agent executes. It spawns a background task to
    /// run the agent and returns a [`StreamingRunResult`] that provides a
    /// stream of [`StreamEvent`]s.
    pub async fn run_stream(
        agent: Agent,
        input: impl Into<String>,
        config: RunConfig,
    ) -> Result<StreamingRunResult> {
        Self::run_stream_with_env(agent, input, config, DefaultEnv).await
    }

    /// Executes an agent with a custom environment and returns a stream of events.
    ///
    /// Like `run_stream`, but allows you to provide a custom environment with
    /// capabilities for use by layers.
    ///
    /// # Arguments
    ///
    /// * `agent` - The [`Agent`] to be executed.
    /// * `input` - The user's input to start the conversation.
    /// * `config` - The [`RunConfig`] to control the execution.
    /// * `env` - The custom environment providing capabilities.
    pub async fn run_stream_with_env<E: crate::env::Env + Clone>(
        agent: Agent,
        input: impl Into<String>,
        config: RunConfig,
        env: E,
    ) -> Result<StreamingRunResult> {
        let mut stream_config = config;
        stream_config.stream = true;

        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let trace_id = crate::tracing::gen_trace_id();

        // Spawn the run in a background task
        let input = input.into();
        tokio::spawn(async move {
            let result = Self::run_with_env(agent, input, stream_config, env).await;

            match result {
                Ok(res) => {
                    let _ = tx.send(StreamEvent::RunCompleted { result: res });
                }
                Err(e) => {
                    let _ = tx.send(StreamEvent::Error {
                        error: e.to_string(),
                    });
                }
            }
        });

        let stream = Box::pin(tokio_stream::wrappers::UnboundedReceiverStream::new(rx));
        Ok(StreamingRunResult::new(stream, trace_id))
    }

    /// The core logic for the agent's execution loop.
    ///
    /// This private method manages the turn-by-turn interaction with the LLM,
    /// handling tool calls, handoffs, and final responses. It is called by the
    /// public `run` methods.
    async fn run_loop(
        mut agent: Agent,
        mut messages: Vec<Message>,
        config: RunConfig,
        context: Arc<Mutex<TracingContext>>,
    ) -> Result<(
        RunResult,
        Option<Box<dyn Any + Send>>, // per-agent contextual state
        Option<Box<dyn Any + Send>>, // run-scoped state
    )> {
        Self::run_loop_with_env(agent, messages, config, context, DefaultEnv).await
    }

    /// The core logic for the agent's execution loop with custom environment.
    ///
    /// This private method manages the turn-by-turn interaction with the LLM,
    /// handling tool calls, handoffs, and final responses. It propagates the
    /// custom environment through tool requests for capability-based layers.
    async fn run_loop_with_env<E: crate::env::Env>(
        mut agent: Agent,
        mut messages: Vec<Message>,
        config: RunConfig,
        context: Arc<Mutex<TracingContext>>,
        env: E,
    ) -> Result<(
        RunResult,
        Option<Box<dyn Any + Send>>, // per-agent contextual state
        Option<Box<dyn Any + Send>>, // run-scoped state
    )> {
        let mut items = Vec::new();
        let mut usage_stats = UsageStats::new();
        let mut turn_count = 0;
        let max_turns = config
            .max_turns
            .unwrap_or(agent.config.max_turns.unwrap_or(10));

        let model_provider = config.model_provider.unwrap_or_else(|| {
            #[cfg(not(test))]
            {
                Arc::new(crate::model::OpenAIProvider::new(&agent.config.model))
            }
            #[cfg(test)]
            {
                Arc::new(crate::model::MockProvider::new(&agent.config.model))
            }
        });

        // Tower layers will be applied per tool

        loop {
            turn_count += 1;

            if turn_count > max_turns {
                return Err(AgentsError::MaxTurnsExceeded { max_turns });
            }

            debug!(turn = turn_count, agent = %agent.name(), "Starting turn");

            // Start agent span
            let agent_span = AgentSpan::new(
                context.clone(),
                agent.name().to_string(),
                agent.instructions().to_string(),
            );

            // Get LLM response
            let gen_span = GenerationSpan::new(context.clone(), agent.config.model.clone());

            // Advertise both regular tools and handoffs (as tools) to the provider
            let mut advertised_tools: Vec<Arc<dyn crate::tool::Tool>> = agent.config.tools.clone();
            for h in agent.handoffs() {
                let ht = crate::handoff::HandoffTool::from(h.clone());
                advertised_tools.push(Arc::new(ht));
            }

            debug!(
                target: "runner::messages",
                "\n=== Sending to provider (model: {}) ===\n{}\n=== end ===",
                agent.config.model,
                format_messages_for_log(&messages)
            );

            let (response, usage) = model_provider
                .complete(
                    messages.clone(),
                    advertised_tools,
                    agent.config.temperature,
                    agent.config.max_tokens,
                )
                .await?;

            gen_span.complete_with_usage(usage.clone());
            usage_stats.record(&agent.config.model, agent.name(), usage);

            // Process response content
            if let Some(content) = &response.content {
                if !content.is_empty() {
                    // Run output guardrails
                    let final_content = if !agent.config.output_guardrails.is_empty() {
                        GuardrailRunner::check_output(&agent.config.output_guardrails, content)
                            .await?
                    } else {
                        content.clone()
                    };

                    messages.push(Message::assistant(&final_content));
                    items.push(RunItem::Message(MessageItem {
                        id: uuid::Uuid::new_v4().to_string(),
                        role: Role::Assistant,
                        content: final_content.clone(),
                        created_at: chrono::Utc::now(),
                    }));

                    // If no tool calls, this is the final output
                    if response.tool_calls.is_empty() {
                        agent_span.complete();

                        let trace_id = context.lock().unwrap().trace_id().to_string();
                        // Context state removed - use layers instead
                        let contextual_state: Option<Box<dyn Any + Send>> = None;
                        let run_scoped_state: Option<Box<dyn Any + Send>> = None;
                        return Ok((
                            RunResult::success(
                                serde_json::Value::String(final_content),
                                items,
                                agent.name().to_string(),
                                usage_stats,
                                trace_id,
                            ),
                            contextual_state,
                            run_scoped_state,
                        ));
                    }
                }
            }

            // If there are tool calls, we need to add the assistant message with tool calls first
            if !response.tool_calls.is_empty() {
                // Convert response tool calls to message tool calls
                let message_tool_calls: Vec<crate::items::ToolCall> = response
                    .tool_calls
                    .iter()
                    .map(|tc| crate::items::ToolCall {
                        id: tc.id.clone(),
                        name: tc.name.clone(),
                        arguments: tc.arguments.clone(),
                    })
                    .collect();

                // Add assistant message with tool calls
                messages.push(Message::assistant_with_tool_calls(
                    response.content.clone().unwrap_or_default(),
                    message_tool_calls.clone(),
                ));

                // Also save as an assistant message item for session history
                items.push(RunItem::Message(MessageItem {
                    id: uuid::Uuid::new_v4().to_string(),
                    role: Role::Assistant,
                    content: response.content.clone().unwrap_or_default(),
                    created_at: chrono::Utc::now(),
                }));

                debug!(
                    target: "runner::messages",
                    "\n↳ Appended assistant tool_calls message\n{}\n---",
                    format_messages_for_log(&messages)
                );
            }

            // Process tool calls
            if !response.tool_calls.is_empty() {
                // Handoff short-circuit: if any call is a handoff, process the first and start a new turn
                if let Some(handoff_call) = response
                    .tool_calls
                    .iter()
                    .find(|tc| agent.handoffs().iter().any(|h| h.name == tc.name))
                {
                    // Emit ToolCall item for the handoff
                    items.push(RunItem::ToolCall(ToolCallItem {
                        id: handoff_call.id.clone(),
                        tool_name: handoff_call.name.clone(),
                        arguments: handoff_call.arguments.clone(),
                        created_at: chrono::Utc::now(),
                    }));

                    if let Some(handoff) = agent
                        .handoffs()
                        .iter()
                        .find(|h| h.name == handoff_call.name)
                    {
                        info!(from = %agent.name(), to = %handoff.name, "Handoff detected");
                        items.push(RunItem::Handoff(HandoffItem {
                            id: uuid::Uuid::new_v4().to_string(),
                            from_agent: agent.name().to_string(),
                            to_agent: handoff.name.clone(),
                            reason: None,
                            created_at: chrono::Utc::now(),
                        }));
                        let handoff_ack =
                            serde_json::json!({ "handoff": handoff.name, "ack": true });
                        messages.push(Message::tool(handoff_ack.to_string(), &handoff_call.id));
                        items.push(RunItem::ToolOutput(ToolOutputItem {
                            id: uuid::Uuid::new_v4().to_string(),
                            tool_call_id: handoff_call.id.clone(),
                            output: handoff_ack,
                            error: None,
                            created_at: chrono::Utc::now(),
                        }));
                        debug!(
                            target: "runner::messages",
                            "\n↳ Appended handoff TOOL reply (tool_call_id={})\n{}\n---",
                            handoff_call.id,
                            format_messages_for_log(&messages)
                        );
                        agent = handoff.agent().clone();

                        continue; // next turn
                    }
                }

                // Emit all ToolCall items first (preserve order)
                for tool_call in &response.tool_calls {
                    items.push(RunItem::ToolCall(ToolCallItem {
                        id: tool_call.id.clone(),
                        tool_name: tool_call.name.clone(),
                        arguments: tool_call.arguments.clone(),
                        created_at: chrono::Utc::now(),
                    }));
                }

                // Execute tool calls based on config
                let trace_id = context.lock().unwrap().trace_id().to_string();
                let mut finalize_with: Option<serde_json::Value> = None;
                if !config.parallel_tools {
                    // Sequential execution
                    for tool_call in &response.tool_calls {
                        let tool_opt = agent
                            .tools()
                            .iter()
                            .find(|t| t.name() == tool_call.name)
                            .cloned();
                        let name = tool_call.name.clone();
                        let args = tool_call.arguments.clone();
                        let id = tool_call.id.clone();
                        if let Some(tool) = tool_opt {
                            let span = ToolSpan::new(context.clone(), name.clone(), args.clone());

                            // LayeredTool removed in Step 8: Tools now use uniform Tower service composition
                            // No per-tool layers extracted; all layering happens via .into_service().layer(...)
                            let tool_layers: Vec<()> = Vec::new();

                            let mut stack = {
                                // Use service-based tool path for Arc<dyn Tool>
                                use crate::env::DefaultEnv;
                                use crate::service::{InputSchemaLayer, ToolRequest, ToolResponse};
                                use crate::tool::ToolResult;
                                use std::future::Future;
                                use std::pin::Pin;
                                use std::sync::Arc;
                                use tower::BoxError;
                                use tower::{util::BoxService, Layer};

                                let schema = tool.parameters_schema();
                                let tool_arc = tool.clone();

                                // Create tool service directly for Arc<dyn Tool>
                                #[derive(Clone)]
                                struct ToolService {
                                    tool: Arc<dyn crate::tool::Tool>,
                                }

                                impl tower::Service<ToolRequest<DefaultEnv>> for ToolService {
                                    type Response = ToolResponse;
                                    type Error = BoxError;
                                    type Future = Pin<
                                        Box<
                                            dyn Future<
                                                    Output = std::result::Result<
                                                        Self::Response,
                                                        Self::Error,
                                                    >,
                                                > + Send,
                                        >,
                                    >;

                                    fn poll_ready(
                                        &mut self,
                                        _cx: &mut std::task::Context<'_>,
                                    ) -> std::task::Poll<std::result::Result<(), Self::Error>>
                                    {
                                        std::task::Poll::Ready(Ok(()))
                                    }

                                    fn call(
                                        &mut self,
                                        req: ToolRequest<DefaultEnv>,
                                    ) -> Self::Future {
                                        let tool = self.tool.clone();
                                        Box::pin(async move {
                                            match tool.execute(req.arguments).await {
                                                Ok(ToolResult {
                                                    output,
                                                    is_final,
                                                    error,
                                                }) => {
                                                    if let Some(err) = error {
                                                        Ok(ToolResponse {
                                                            output: serde_json::Value::Null,
                                                            error: Some(err),
                                                            effect:
                                                                crate::service::Effect::Continue,
                                                        })
                                                    } else {
                                                        let effect = if is_final {
                                                            crate::service::Effect::Final(
                                                                output.clone(),
                                                            )
                                                        } else {
                                                            crate::service::Effect::Continue
                                                        };
                                                        Ok(ToolResponse {
                                                            output,
                                                            error: None,
                                                            effect,
                                                        })
                                                    }
                                                }
                                                Err(e) => Ok(ToolResponse {
                                                    output: serde_json::Value::Null,
                                                    error: Some(e.to_string()),
                                                    effect: crate::service::Effect::Continue,
                                                }),
                                            }
                                        })
                                    }
                                }

                                let tool_service = ToolService { tool: tool_arc };

                                // Add default schema validation (to be moved to tool constructors in Step 6)
                                let with_schema =
                                    InputSchemaLayer::lenient(schema).layer(tool_service);
                                BoxService::new(with_schema)
                            };
                            // Step 8: LayeredTool and ErasedToolLayer removed - no dynamic layer application needed
                            // All layering now happens via typed .layer() composition at tool/agent/run creation time
                            // Apply layers Tool → Agent → Run (inner-to-outer) for runtime order Run → Agent → Tool → Base
                            // Tool layers applied first (innermost, closest to base) - no longer applied dynamically
                            // Agent layers wrap tool layers - no longer applied dynamically
                            // Run layers wrap everything (outermost) - no longer applied dynamically
                            let req = ToolRequest::<DefaultEnv> {
                                env: DefaultEnv,
                                run_id: trace_id.clone(),
                                agent: agent.name().to_string(),
                                tool_call_id: id.clone(),
                                tool_name: name.clone(),
                                arguments: args.clone(),
                            };
                            let res = stack.oneshot(req).await;
                            match res {
                                Ok(resp) => {
                                    if let Some(err) = resp.error.clone() {
                                        messages
                                            .push(Message::tool(format!("Error: {}", err), &id));
                                        items.push(RunItem::ToolOutput(ToolOutputItem {
                                            id: uuid::Uuid::new_v4().to_string(),
                                            tool_call_id: id.clone(),
                                            output: serde_json::Value::Null,
                                            error: Some(err),
                                            created_at: chrono::Utc::now(),
                                        }));
                                    } else {
                                        let content = serde_json::to_string(&resp.output)
                                            .unwrap_or_else(|_| "null".to_string());
                                        messages.push(Message::tool(content, &id));
                                        items.push(RunItem::ToolOutput(ToolOutputItem {
                                            id: uuid::Uuid::new_v4().to_string(),
                                            tool_call_id: id.clone(),
                                            output: resp.output.clone(),
                                            error: None,
                                            created_at: chrono::Utc::now(),
                                        }));
                                    }
                                    span.success();
                                    if matches!(resp.effect, Effect::Final(_)) {
                                        let v = if let Effect::Final(v) = resp.effect {
                                            v
                                        } else {
                                            serde_json::Value::Null
                                        };
                                        let final_val = if v.is_null() { resp.output } else { v };
                                        agent_span.complete();
                                        let trace_id =
                                            context.lock().unwrap().trace_id().to_string();
                                        // Context state removed - use layers instead
                                        let contextual_state: Option<Box<dyn Any + Send>> = None;
                                        let run_scoped_state: Option<Box<dyn Any + Send>> = None;
                                        return Ok((
                                            RunResult::success(
                                                final_val,
                                                items,
                                                agent.name().to_string(),
                                                usage_stats,
                                                trace_id,
                                            ),
                                            contextual_state,
                                            run_scoped_state,
                                        ));
                                    }
                                }
                                Err(e) => {
                                    span.error(e.to_string());
                                    messages.push(Message::tool(format!("Error: {}", e), &id));
                                    items.push(RunItem::ToolOutput(ToolOutputItem {
                                        id: uuid::Uuid::new_v4().to_string(),
                                        tool_call_id: id.clone(),
                                        output: serde_json::Value::Null,
                                        error: Some(e.to_string()),
                                        created_at: chrono::Utc::now(),
                                    }));
                                }
                            }
                        } else {
                            messages.push(Message::tool(
                                format!("Error: Unknown tool '{}'", name),
                                &id,
                            ));
                            items.push(RunItem::ToolOutput(ToolOutputItem {
                                id: uuid::Uuid::new_v4().to_string(),
                                tool_call_id: id.clone(),
                                output: serde_json::Value::Null,
                                error: Some(format!("Unknown tool '{}'", name)),
                                created_at: chrono::Utc::now(),
                            }));
                        }
                    }
                } else {
                    // Parallel execution (optionally limit concurrency)
                    use tokio::sync::Semaphore;
                    let semaphore = config
                        .max_concurrency
                        .map(|n| std::sync::Arc::new(Semaphore::new(n)));
                    let run_layers_clone = config.run_layers.clone();
                    let agent_layers_clone_outer = agent.config.agent_layers.clone();
                    let env_clone_outer = env.clone();
                    let futures_vec = response
                        .tool_calls
                        .iter()
                        .map(|tool_call| {
                            let tool_opt = agent
                                .tools()
                                .iter()
                                .find(|t| t.name() == tool_call.name)
                                .cloned();
                            // Layers are now attached to tools directly
                            let name = tool_call.name.clone();
                            let args = tool_call.arguments.clone();
                            let id = tool_call.id.clone();
                            let context = context.clone();
                            let agent_name = agent.name().to_string();
                            let run_id_value = trace_id.clone();
                            let semaphore = semaphore.clone();
                            let run_layers_clone = run_layers_clone.clone();
                            let agent_layers_clone = agent_layers_clone_outer.clone();
                            let env_clone = env_clone_outer.clone();
                            async move {
                                let _permit = if let Some(sem) = semaphore {
                                    Some(sem.acquire_owned().await.expect("semaphore"))
                                } else {
                                    None
                                };
                                if let Some(tool) = tool_opt {
                                    let span =
                                        ToolSpan::new(context.clone(), name.clone(), args.clone());

                                    // LayeredTool removed in Step 8: Tools now use uniform Tower service composition
                                    // No per-tool layers extracted; all layering happens via .into_service().layer(...)
                                    let tool_layers: Vec<()> = Vec::new();

                                    let mut stack =
                                        Self::create_tool_service_stack::<E>(&tool, &env_clone);
                                    // Step 8: LayeredTool and ErasedToolLayer removed - no dynamic layer application needed
                                    // All layering now happens via typed .layer() composition at tool/agent/run creation time
                                    // Apply layers Tool → Agent → Run (inner-to-outer) for runtime order Run → Agent → Tool → Base
                                    // Tool layers applied first (innermost, closest to base) - no longer applied dynamically
                                    // Agent layers wrap tool layers - no longer applied dynamically
                                    // Run layers wrap everything (outermost) - no longer applied dynamically
                                    let req = ToolRequest::<E> {
                                        env: env_clone.clone(),
                                        run_id: run_id_value.clone(),
                                        agent: agent_name,
                                        tool_call_id: id.clone(),
                                        tool_name: name.clone(),
                                        arguments: args.clone(),
                                    };
                                    let out: std::result::Result<
                                        crate::service::ToolResponse,
                                        tower::BoxError,
                                    > = stack.oneshot(req).await;
                                    match &out {
                                        Ok(_) => span.success(),
                                        Err(e) => span.error(e.to_string()),
                                    }
                                    (id, name, args, out)
                                } else {
                                    (
                                        id.clone(),
                                        name.clone(),
                                        args.clone(),
                                        Err(crate::error::AgentsError::Other(format!(
                                            "Unknown tool '{}'",
                                            name
                                        ))
                                        .into()),
                                    )
                                }
                            }
                        })
                        .collect::<Vec<_>>();

                    let results = join_all(futures_vec).await;
                    for (tcid, _name, _args, res) in results {
                        match res {
                            Ok(resp) => {
                                if let Some(err) = resp.error.clone() {
                                    messages.push(Message::tool(format!("Error: {}", err), &tcid));
                                    items.push(RunItem::ToolOutput(ToolOutputItem {
                                        id: uuid::Uuid::new_v4().to_string(),
                                        tool_call_id: tcid.clone(),
                                        output: serde_json::Value::Null,
                                        error: Some(err),
                                        created_at: chrono::Utc::now(),
                                    }));
                                } else {
                                    let content = serde_json::to_string(&resp.output)
                                        .unwrap_or_else(|_| "null".to_string());
                                    messages.push(Message::tool(content, &tcid));
                                    items.push(RunItem::ToolOutput(ToolOutputItem {
                                        id: uuid::Uuid::new_v4().to_string(),
                                        tool_call_id: tcid.clone(),
                                        output: resp.output.clone(),
                                        error: None,
                                        created_at: chrono::Utc::now(),
                                    }));
                                }
                                if matches!(resp.effect, Effect::Final(_))
                                    && finalize_with.is_none()
                                {
                                    let v = if let Effect::Final(v) = resp.effect {
                                        v
                                    } else {
                                        serde_json::Value::Null
                                    };
                                    finalize_with = Some(if v.is_null() { resp.output } else { v });
                                }
                            }
                            Err(e) => {
                                messages.push(Message::tool(format!("Error: {}", e), &tcid));
                                items.push(RunItem::ToolOutput(ToolOutputItem {
                                    id: uuid::Uuid::new_v4().to_string(),
                                    tool_call_id: tcid.clone(),
                                    output: serde_json::Value::Null,
                                    error: Some(e.to_string()),
                                    created_at: chrono::Utc::now(),
                                }));
                            }
                        }
                    }
                }

                debug!(
                    target: "runner::messages",
                    "\n↳ Appended TOOL replies (batched)\n{}\n---",
                    format_messages_for_log(&messages)
                );

                if let Some(final_val) = finalize_with {
                    agent_span.complete();
                    let trace_id = context.lock().unwrap().trace_id().to_string();
                    // Context state removed - use layers instead
                    let contextual_state: Option<Box<dyn Any + Send>> = None;
                    let run_scoped_state: Option<Box<dyn Any + Send>> = None;
                    return Ok((
                        RunResult::success(
                            final_val,
                            items,
                            agent.name().to_string(),
                            usage_stats,
                            trace_id,
                        ),
                        contextual_state,
                        run_scoped_state,
                    ));
                }
            }

            // Complete the agent span if not already completed
            agent_span.complete();
        }
    }

    /// Helper method to create a tool service stack generic over environment type.
    ///
    /// This creates the service stack that was previously in build_tool_stack,
    /// but now it's environment-generic and applies layers based on environment capabilities.
    fn create_tool_service_stack<E: crate::env::Env>(
        tool: &Arc<dyn crate::tool::Tool>,
        env: &E, // Environment used to determine which layers to apply
    ) -> tower::util::BoxService<
        crate::service::ToolRequest<E>,
        crate::service::ToolResponse,
        tower::BoxError,
    > {
        use crate::service::{ApprovalLayer, InputSchemaLayer, ToolRequest, ToolResponse};
        use crate::tool::ToolResult;
        use std::future::Future;
        use std::pin::Pin;
        use std::sync::Arc;
        use tower::BoxError;
        use tower::{util::BoxService, Layer};

        let schema = tool.parameters_schema();
        let tool_arc = tool.clone();

        // Create tool service directly for Arc<dyn Tool>, generic over environment
        #[derive(Clone)]
        struct ToolService<E> {
            tool: Arc<dyn crate::tool::Tool>,
            _phantom: std::marker::PhantomData<E>,
        }

        impl<E: crate::env::Env> tower::Service<ToolRequest<E>> for ToolService<E> {
            type Response = ToolResponse;
            type Error = BoxError;
            type Future = Pin<
                Box<dyn Future<Output = std::result::Result<Self::Response, Self::Error>> + Send>,
            >;

            fn poll_ready(
                &mut self,
                _cx: &mut std::task::Context<'_>,
            ) -> std::task::Poll<std::result::Result<(), Self::Error>> {
                std::task::Poll::Ready(Ok(()))
            }

            fn call(&mut self, req: ToolRequest<E>) -> Self::Future {
                let tool = self.tool.clone();
                Box::pin(async move {
                    match tool.execute(req.arguments).await {
                        Ok(ToolResult {
                            output,
                            is_final,
                            error,
                        }) => {
                            if let Some(err) = error {
                                Ok(ToolResponse {
                                    output: serde_json::Value::Null,
                                    error: Some(err),
                                    effect: crate::service::Effect::Continue,
                                })
                            } else {
                                let effect = if is_final {
                                    crate::service::Effect::Final(output.clone())
                                } else {
                                    crate::service::Effect::Continue
                                };
                                Ok(ToolResponse {
                                    output,
                                    error: None,
                                    effect,
                                })
                            }
                        }
                        Err(e) => Ok(ToolResponse {
                            output: serde_json::Value::Null,
                            error: Some(e.to_string()),
                            effect: crate::service::Effect::Continue,
                        }),
                    }
                })
            }
        }

        let tool_service = ToolService::<E> {
            tool: tool_arc,
            _phantom: std::marker::PhantomData,
        };

        // Apply layers based on environment capabilities - this is the Tower way!
        // Start with schema validation (to be moved to tool constructors in Step 6)
        let with_schema = InputSchemaLayer::lenient(schema).layer(tool_service);

        // Auto-apply ApprovalLayer if environment has approval capability
        // This demonstrates capability-driven layer application
        if env.capability::<crate::env::ApprovalCapability>().is_some()
            || env.capability::<crate::env::AutoApprove>().is_some()
            || env.capability::<crate::env::ManualApproval>().is_some()
        {
            let with_approval = ApprovalLayer.layer(with_schema);
            BoxService::new(with_approval)
        } else {
            BoxService::new(with_schema)
        }
    }
}

// TODO: Rewrite tests to use the new Tower-based architecture
// The old tests have been removed as they relied on the deleted context system

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::MockProvider;
    use crate::tool::FunctionTool;

    #[tokio::test]
    async fn test_simple_run() {
        let agent = Agent::simple("TestAgent", "You are a test agent");

        struct MockP;
        #[async_trait::async_trait]
        impl crate::model::ModelProvider for MockP {
            async fn complete(
                &self,
                _messages: Vec<crate::items::Message>,
                _tools: Vec<std::sync::Arc<dyn crate::tool::Tool>>,
                _temperature: Option<f32>,
                _max_tokens: Option<u32>,
            ) -> crate::error::Result<(crate::items::ModelResponse, crate::usage::Usage)>
            {
                Ok((
                    crate::items::ModelResponse::new_message("Hello! How can I help you?"),
                    crate::usage::Usage::new(0, 0),
                ))
            }
            fn model_name(&self) -> &str {
                "test-model"
            }
        }
        let provider = Arc::new(MockP);

        let config = RunConfig {
            model_provider: Some(provider),
            ..Default::default()
        };

        let result = Runner::run(agent, "Hi", config).await.unwrap();

        assert!(result.is_success());
        assert_eq!(result.final_agent, "TestAgent");
        assert!(result.final_output.is_string());
    }

    #[test]
    fn test_run_sync() {
        let agent = Agent::simple("SyncAgent", "Sync test agent");

        let provider = Arc::new(MockProvider::new("test-model").with_message("Sync response"));

        let config = RunConfig {
            model_provider: Some(provider),
            ..Default::default()
        };

        let result = Runner::run_sync(agent, "Test", config).unwrap();

        assert!(result.is_success());
        assert_eq!(result.final_output, serde_json::json!("Sync response"));
    }

    #[tokio::test]
    async fn test_run_with_tools() {
        let tool = Arc::new(FunctionTool::simple(
            "uppercase",
            "Converts to uppercase",
            |s: String| s.to_uppercase(),
        ));

        let agent = Agent::simple("ToolAgent", "Use tools when needed").with_tool(tool);

        let provider = Arc::new(
            MockProvider::new("test-model")
                .with_tool_call("uppercase", serde_json::json!({"input": "hello"}))
                .with_message("The result is: HELLO"),
        );

        let config = RunConfig {
            model_provider: Some(provider),
            ..Default::default()
        };

        let result = Runner::run(agent, "Make 'hello' uppercase", config)
            .await
            .unwrap();

        assert!(result.is_success());
        // Should have tool call and output items
        assert!(result
            .items
            .iter()
            .any(|item| matches!(item, RunItem::ToolCall(_))));
        assert!(result
            .items
            .iter()
            .any(|item| matches!(item, RunItem::ToolOutput(_))));
    }

    #[test]
    fn test_runconfig_layer_chaining_compiles() {
        use crate::service::{RetryLayer, TimeoutLayer};
        use std::time::Duration;

        // Test that RunConfig layer chaining compiles and can be chained
        let config = RunConfig::default()
            .layer(TimeoutLayer::from_duration(Duration::from_secs(30)))
            .layer(RetryLayer::times(3));

        // Test double layering compiles
        let double_layered = config.layer(TimeoutLayer::from_duration(Duration::from_secs(60)));

        // Verify it's a layered config (type check passes)
        let _: LayeredRunConfig<_, _> = double_layered;
    }

    #[tokio::test]
    async fn test_runner_uses_service_tools() {
        use crate::tool::FunctionTool;

        // Create a simple tool
        let tool = Arc::new(FunctionTool::simple("echo", "Echoes input", |s: String| {
            s.to_uppercase()
        }));

        let agent = Agent::simple("ServiceAgent", "I use service-based tools").with_tool(tool);

        // Mock provider that generates one tool call then finishes
        struct MockP {
            call_count: std::sync::atomic::AtomicUsize,
        }
        impl MockP {
            fn new() -> Self {
                Self {
                    call_count: std::sync::atomic::AtomicUsize::new(0),
                }
            }
        }
        #[async_trait::async_trait]
        impl crate::model::ModelProvider for MockP {
            async fn complete(
                &self,
                _messages: Vec<crate::items::Message>,
                _tools: Vec<std::sync::Arc<dyn crate::tool::Tool>>,
                _temperature: Option<f32>,
                _max_tokens: Option<u32>,
            ) -> crate::error::Result<(crate::items::ModelResponse, crate::usage::Usage)>
            {
                let count = self
                    .call_count
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);

                if count == 0 {
                    // First call: return a tool call to test service integration
                    let tool_call = crate::items::ToolCall {
                        id: "test_1".to_string(),
                        name: "echo".to_string(),
                        arguments: serde_json::json!({"input": "hello"}),
                    };
                    Ok((
                        crate::items::ModelResponse::new_tool_calls(vec![tool_call]),
                        crate::usage::Usage::new(10, 20),
                    ))
                } else {
                    // Subsequent calls: return a final message
                    Ok((
                        crate::items::ModelResponse::new_message(
                            "Tool execution completed successfully!",
                        ),
                        crate::usage::Usage::new(5, 10),
                    ))
                }
            }
            fn model_name(&self) -> &str {
                "test-service-model"
            }
        }

        let provider = Arc::new(MockP::new());
        let config = RunConfig {
            model_provider: Some(provider),
            ..Default::default()
        };

        // This test verifies that the runner successfully uses service-based tools
        // without any BaseToolService adapter
        let result = Runner::run(agent, "test service tools", config).await;

        // The test passes if execution completes successfully using the service path
        if let Err(e) = &result {
            println!("Runner error: {:?}", e);
        }
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_success());

        // Should have tool call and tool output (proving service path worked)
        assert!(result
            .items
            .iter()
            .any(|item| matches!(item, RunItem::ToolCall(_))));
        assert!(result
            .items
            .iter()
            .any(|item| matches!(item, RunItem::ToolOutput(_))));
    }

    #[tokio::test]
    async fn test_runner_with_custom_env_approval_denied() {
        use crate::env::{Approval, ApprovalCapability, EnvBuilder};
        use crate::tool::FunctionTool;

        // Create a test approval that always denies
        #[derive(Default)]
        struct DenyAllApproval;
        impl Approval for DenyAllApproval {
            fn request_approval(&self, _operation: &str, _details: &str) -> bool {
                false // Always deny
            }
        }

        let env = EnvBuilder::new()
            .with_capability(Arc::new(ApprovalCapability::new(DenyAllApproval)))
            .build();

        let tool = Arc::new(FunctionTool::simple("test", "Test tool", |s: String| s));
        let agent = Agent::simple("TestAgent", "I test approval").with_tool(tool);

        struct MockP {
            call_count: std::sync::atomic::AtomicUsize,
        }
        impl MockP {
            fn new() -> Self {
                Self {
                    call_count: std::sync::atomic::AtomicUsize::new(0),
                }
            }
        }
        #[async_trait::async_trait]
        impl crate::model::ModelProvider for MockP {
            async fn complete(
                &self,
                _messages: Vec<crate::items::Message>,
                _tools: Vec<std::sync::Arc<dyn crate::tool::Tool>>,
                _temperature: Option<f32>,
                _max_tokens: Option<u32>,
            ) -> crate::error::Result<(crate::items::ModelResponse, crate::usage::Usage)>
            {
                let count = self
                    .call_count
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);

                if count == 0 {
                    // First call: return a tool call that should be denied
                    let tool_call = crate::items::ToolCall {
                        id: "test_1".to_string(),
                        name: "test".to_string(),
                        arguments: serde_json::json!({"input": "test"}),
                    };
                    Ok((
                        crate::items::ModelResponse::new_tool_calls(vec![tool_call]),
                        crate::usage::Usage::new(10, 20),
                    ))
                } else {
                    // Subsequent calls: return final message after seeing the denial
                    Ok((
                        crate::items::ModelResponse::new_message(
                            "I see the tool was denied access.",
                        ),
                        crate::usage::Usage::new(5, 10),
                    ))
                }
            }
            fn model_name(&self) -> &str {
                "test-model"
            }
        }

        let config = RunConfig {
            model_provider: Some(Arc::new(MockP::new())),
            ..Default::default()
        };

        let result = Runner::run_with_env(agent, "test approval", config, env)
            .await
            .unwrap();

        // Should have a tool call that was denied
        assert!(result
            .items
            .iter()
            .any(|item| matches!(item, RunItem::ToolCall(_))));
        assert!(result.items.iter().any(|item| {
            matches!(item, RunItem::ToolOutput(output) if output.error.as_deref() == Some("approval denied"))
        }));
    }

    #[tokio::test]
    async fn test_runner_with_custom_env_approval_allowed() {
        use crate::env::{Approval, ApprovalCapability, EnvBuilder};
        use crate::tool::FunctionTool;

        // Create a test approval that always approves
        #[derive(Default)]
        struct ApproveAllApproval;
        impl Approval for ApproveAllApproval {
            fn request_approval(&self, _operation: &str, _details: &str) -> bool {
                true // Always approve
            }
        }

        let env = EnvBuilder::new()
            .with_capability(Arc::new(ApprovalCapability::new(ApproveAllApproval)))
            .build();

        let tool = Arc::new(FunctionTool::simple("test", "Test tool", |s: String| {
            format!("processed: {}", s)
        }));
        let agent = Agent::simple("TestAgent", "I test approval").with_tool(tool);

        struct MockP {
            call_count: std::sync::atomic::AtomicUsize,
        }
        impl MockP {
            fn new() -> Self {
                Self {
                    call_count: std::sync::atomic::AtomicUsize::new(0),
                }
            }
        }
        #[async_trait::async_trait]
        impl crate::model::ModelProvider for MockP {
            async fn complete(
                &self,
                _messages: Vec<crate::items::Message>,
                _tools: Vec<std::sync::Arc<dyn crate::tool::Tool>>,
                _temperature: Option<f32>,
                _max_tokens: Option<u32>,
            ) -> crate::error::Result<(crate::items::ModelResponse, crate::usage::Usage)>
            {
                let count = self
                    .call_count
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);

                if count == 0 {
                    // First call: return a tool call that should be approved
                    let tool_call = crate::items::ToolCall {
                        id: "test_1".to_string(),
                        name: "test".to_string(),
                        arguments: serde_json::json!({"input": "hello"}),
                    };
                    Ok((
                        crate::items::ModelResponse::new_tool_calls(vec![tool_call]),
                        crate::usage::Usage::new(10, 20),
                    ))
                } else {
                    // Subsequent calls: return a final message
                    Ok((
                        crate::items::ModelResponse::new_message("Success!"),
                        crate::usage::Usage::new(5, 10),
                    ))
                }
            }
            fn model_name(&self) -> &str {
                "test-model"
            }
        }

        let config = RunConfig {
            model_provider: Some(Arc::new(MockP::new())),
            ..Default::default()
        };

        let result = Runner::run_with_env(agent, "test approval", config, env)
            .await
            .unwrap();
        assert!(result.is_success());

        // Should have successful tool execution (no error)
        assert!(result
            .items
            .iter()
            .any(|item| matches!(item, RunItem::ToolCall(_))));
        assert!(result.items.iter().any(|item| {
            matches!(item, RunItem::ToolOutput(output) if output.error.is_none() && output.output.as_str() == Some("processed: hello"))
        }));
    }

    #[tokio::test]
    async fn test_runner_without_approval_capability_auto_applies_layer() {
        use crate::env::DefaultEnv;
        use crate::tool::FunctionTool;

        // Test that DefaultEnv (no approval capability) results in no ApprovalLayer applied
        let tool = Arc::new(FunctionTool::simple("test", "Test tool", |s: String| s));
        let agent = Agent::simple("TestAgent", "I test no approval").with_tool(tool);

        struct MockP {
            call_count: std::sync::atomic::AtomicUsize,
        }
        impl MockP {
            fn new() -> Self {
                Self {
                    call_count: std::sync::atomic::AtomicUsize::new(0),
                }
            }
        }
        #[async_trait::async_trait]
        impl crate::model::ModelProvider for MockP {
            async fn complete(
                &self,
                _messages: Vec<crate::items::Message>,
                _tools: Vec<std::sync::Arc<dyn crate::tool::Tool>>,
                _temperature: Option<f32>,
                _max_tokens: Option<u32>,
            ) -> crate::error::Result<(crate::items::ModelResponse, crate::usage::Usage)>
            {
                let count = self
                    .call_count
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);

                if count == 0 {
                    // First call: return a tool call
                    let tool_call = crate::items::ToolCall {
                        id: "test_1".to_string(),
                        name: "test".to_string(),
                        arguments: serde_json::json!({"input": "test"}),
                    };
                    Ok((
                        crate::items::ModelResponse::new_tool_calls(vec![tool_call]),
                        crate::usage::Usage::new(10, 20),
                    ))
                } else {
                    // Subsequent calls: return final message
                    Ok((
                        crate::items::ModelResponse::new_message(
                            "Tool executed successfully without approval layer.",
                        ),
                        crate::usage::Usage::new(5, 10),
                    ))
                }
            }
            fn model_name(&self) -> &str {
                "test-model"
            }
        }

        let config = RunConfig {
            model_provider: Some(Arc::new(MockP::new())),
            ..Default::default()
        };

        // Use DefaultEnv (no approval capability) - should work without ApprovalLayer being applied
        let result = Runner::run_with_env(agent, "test no approval", config, DefaultEnv)
            .await
            .unwrap();

        // Should execute successfully since DefaultEnv doesn't trigger ApprovalLayer application
        assert!(result
            .items
            .iter()
            .any(|item| matches!(item, RunItem::ToolCall(_))));
        assert!(result
            .items
            .iter()
            .any(|item| { matches!(item, RunItem::ToolOutput(output) if output.error.is_none()) }));
    }
}
