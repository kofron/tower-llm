//! # Runner (orientation)
//!
//! The `Runner` coordinates an agent run: it interacts with the model, routes
//! tool-calls through the Tower stack (Agent → Run → Tool → BaseTool), and
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

    /// Optional dynamic run-scope policy layers applied inside the agent scope.
    /// These are applied around the tool stack after the agent/run context layers are composed.
    pub run_layers: Vec<Arc<dyn crate::service::ErasedToolLayer>>,
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
    /// Attach a run-scoped context handler that applies across all agents in this run.
    ///
    /// The handler is invoked after each tool execution (success or error).
    /// It can forward, rewrite, or finalize the run via a [`ContextDecision`].
    /// The context value is constructed once per run using the provided factory
    /// and evolves across turns and handoffs.
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
    pub fn with_run_layers(
        mut self,
        layers: Vec<Arc<dyn crate::service::ErasedToolLayer>>,
    ) -> Self {
        self.run_layers = layers;
        self
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
/// ```rust,no_run
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

        // Run the agent loop
        let (result, _state, _run_state) =
            Self::run_loop(agent, messages, config.clone(), context.clone()).await?;

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
        let mut stream_config = config;
        stream_config.stream = true;

        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let trace_id = crate::tracing::gen_trace_id();

        // Spawn the run in a background task
        let input = input.into();
        tokio::spawn(async move {
            let result = Self::run(agent, input, stream_config).await;

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

                            // Check if this is a LayeredTool and extract its layers
                            let tool_layers = if let Some(layered) =
                                tool.as_any().downcast_ref::<crate::tool::LayeredTool>()
                            {
                                layered.layers().to_vec()
                            } else {
                                Vec::new()
                            };

                            let mut stack = crate::service::build_tool_stack::<DefaultEnv>(tool);
                            // Apply dynamic layers: run-scope then agent-scope (Agent wraps Run)
                            for l in &config.run_layers {
                                stack = l.layer_boxed(stack);
                            }
                            let agent_layers = agent.config.agent_layers.clone();
                            for l in &agent_layers {
                                stack = l.layer_boxed(stack);
                            }
                            // Apply tool's own layers (from LayeredTool)
                            for l in &tool_layers {
                                stack = l.layer_boxed(stack);
                            }
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
                            async move {
                                let _permit = if let Some(sem) = semaphore {
                                    Some(sem.acquire_owned().await.expect("semaphore"))
                                } else {
                                    None
                                };
                                if let Some(tool) = tool_opt {
                                    let span =
                                        ToolSpan::new(context.clone(), name.clone(), args.clone());

                                    // Check if this is a LayeredTool and extract its layers
                                    let tool_layers = if let Some(layered) =
                                        tool.as_any().downcast_ref::<crate::tool::LayeredTool>()
                                    {
                                        layered.layers().to_vec()
                                    } else {
                                        Vec::new()
                                    };

                                    let mut stack =
                                        crate::service::build_tool_stack::<DefaultEnv>(tool);
                                    for l in &run_layers_clone {
                                        stack = l.layer_boxed(stack);
                                    }
                                    for l in &agent_layers_clone {
                                        stack = l.layer_boxed(stack);
                                    }
                                    // Apply tool's own layers (from LayeredTool)
                                    for l in &tool_layers {
                                        stack = l.layer_boxed(stack);
                                    }
                                    let req = ToolRequest::<DefaultEnv> {
                                        env: DefaultEnv,
                                        run_id: run_id_value.clone(),
                                        agent: agent_name,
                                        tool_call_id: id.clone(),
                                        tool_name: name.clone(),
                                        arguments: args.clone(),
                                    };
                                    let out = stack.oneshot(req).await;
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
}

#[cfg(test)]
mod tests {
    use super::*;
    // Context imports removed - using Tower layers instead
    use crate::items::ToolCall as MsgToolCall;
    use crate::layers;
    use crate::model::MockProvider;
    use crate::tool::FunctionTool;
    use std::sync::Mutex;

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

    #[derive(Clone, Default)]
    struct Ctx {
        count: usize,
    }

    struct ForwardHandler;
    impl ToolContext<Ctx> for ForwardHandler {
        fn on_tool_output(
            &self,
            mut ctx: Ctx,
            _tool_name: &str,
            _arguments: &serde_json::Value,
            _result: std::result::Result<serde_json::Value, String>,
        ) -> Result<ContextStep<Ctx>> {
            ctx.count += 1;
            Ok(ContextStep::forward(ctx))
        }
    }

    struct RewriteHandler;
    impl ToolContext<Ctx> for RewriteHandler {
        fn on_tool_output(
            &self,
            mut ctx: Ctx,
            _tool_name: &str,
            _arguments: &serde_json::Value,
            _result: std::result::Result<serde_json::Value, String>,
        ) -> Result<ContextStep<Ctx>> {
            ctx.count += 1;
            Ok(ContextStep::rewrite(ctx, serde_json::json!("OVERRIDDEN")))
        }
    }

    struct FinalHandler;
    impl ToolContext<Ctx> for FinalHandler {
        fn on_tool_output(
            &self,
            ctx: Ctx,
            _tool_name: &str,
            _arguments: &serde_json::Value,
            _result: std::result::Result<serde_json::Value, String>,
        ) -> Result<ContextStep<Ctx>> {
            Ok(ContextStep::final_output(ctx, serde_json::json!("done")))
        }
    }

    struct RewriteOnErrorHandler;
    impl ToolContext<Ctx> for RewriteOnErrorHandler {
        fn on_tool_output(
            &self,
            ctx: Ctx,
            _tool_name: &str,
            _arguments: &serde_json::Value,
            result: std::result::Result<serde_json::Value, String>,
        ) -> Result<ContextStep<Ctx>> {
            match result {
                Ok(v) => Ok(ContextStep::rewrite(ctx, v)),
                Err(_) => Ok(ContextStep::rewrite(ctx, serde_json::json!("RECOVERED"))),
            }
        }
    }

    #[tokio::test]
    async fn test_context_forward_noop() {
        let tool = Arc::new(FunctionTool::simple(
            "uppercase",
            "Converts to uppercase",
            |s: String| s.to_uppercase(),
        ));

        let agent = Agent::simple("CtxAgent", "Use tools")
            .with_tool(tool)
            .with_context_factory(Ctx::default, ForwardHandler);

        let provider = Arc::new(
            MockProvider::new("test-model")
                .with_tool_call("uppercase", serde_json::json!({"input": "hello"}))
                .with_message("The result is: HELLO"),
        );

        let config = RunConfig {
            model_provider: Some(provider),
            ..Default::default()
        };
        let result = Runner::run(agent, "Run", config).await.unwrap();

        // Should contain tool output with original uppercase output
        assert!(result.items.iter().any(|item| match item {
            RunItem::ToolOutput(o) => o.output == serde_json::json!("HELLO"),
            _ => false,
        }));
    }

    #[tokio::test]
    async fn test_context_rewrite() {
        let tool = Arc::new(FunctionTool::simple(
            "uppercase",
            "Converts to uppercase",
            |s: String| s.to_uppercase(),
        ));

        let agent = Agent::simple("CtxRewrite", "Use tools")
            .with_tool(tool)
            .with_context_factory(Ctx::default, RewriteHandler);

        let provider = Arc::new(
            MockProvider::new("test-model")
                .with_tool_call("uppercase", serde_json::json!({"input": "hello"}))
                .with_message("The result is: HELLO"),
        );

        let config = RunConfig {
            model_provider: Some(provider),
            ..Default::default()
        };
        let result = Runner::run(agent, "Run", config).await.unwrap();

        // Tool output should be rewritten
        assert!(result.items.iter().any(|item| match item {
            RunItem::ToolOutput(o) => o.output == serde_json::json!("OVERRIDDEN"),
            _ => false,
        }));
    }

    #[tokio::test]
    async fn test_context_final() {
        let tool = Arc::new(FunctionTool::simple(
            "uppercase",
            "Converts to uppercase",
            |s: String| s.to_uppercase(),
        ));

        let agent = Agent::simple("CtxFinal", "Use tools")
            .with_tool(tool)
            .with_context_factory(Ctx::default, FinalHandler);

        let provider = Arc::new(
            MockProvider::new("test-model")
                .with_tool_call("uppercase", serde_json::json!({"input": "hello"})),
        );

        let config = RunConfig {
            model_provider: Some(provider),
            ..Default::default()
        };
        let result = Runner::run(agent, "Run", config).await.unwrap();

        assert!(result.is_success());
        assert_eq!(result.final_output, serde_json::json!("done"));
    }

    #[tokio::test]
    async fn test_context_error_rewrite() {
        let tool = Arc::new(crate::tool::FunctionTool::new(
            "failing".to_string(),
            "Always fails".to_string(),
            serde_json::json!({}),
            |_args| {
                Err(crate::error::AgentsError::ToolExecutionError {
                    message: "Intentional failure".to_string(),
                })
            },
        ));

        let agent = Agent::simple("CtxErr", "Use tools")
            .with_tool(tool)
            .with_context_factory(Ctx::default, RewriteOnErrorHandler);

        let provider = Arc::new(
            MockProvider::new("test-model").with_tool_call("failing", serde_json::json!({})),
        );

        let config = RunConfig {
            model_provider: Some(provider),
            ..Default::default()
        };
        let result = Runner::run(agent, "Run", config).await.unwrap();

        // Tool error should be rewritten to a successful output
        assert!(result.items.iter().any(|item| match item {
            RunItem::ToolOutput(o) =>
                o.output == serde_json::json!("RECOVERED") && o.error.is_none(),
            _ => false,
        }));
    }

    struct CountingHandler;
    impl ToolContext<Ctx> for CountingHandler {
        fn on_tool_output(
            &self,
            mut ctx: Ctx,
            _tool_name: &str,
            _arguments: &serde_json::Value,
            result: std::result::Result<serde_json::Value, String>,
        ) -> Result<ContextStep<Ctx>> {
            ctx.count += 1;
            let payload = match result {
                Ok(v) => serde_json::json!({"value": v, "count": ctx.count}),
                Err(e) => serde_json::json!({"error": e, "count": ctx.count}),
            };
            Ok(ContextStep::rewrite(ctx, payload))
        }
    }

    struct ErrOnceCountingHandler {
        fail_first: Mutex<bool>,
    }

    impl ErrOnceCountingHandler {
        fn new() -> Self {
            Self {
                fail_first: Mutex::new(true),
            }
        }
    }

    impl ToolContext<Ctx> for ErrOnceCountingHandler {
        fn on_tool_output(
            &self,
            mut ctx: Ctx,
            _tool_name: &str,
            _arguments: &serde_json::Value,
            result: std::result::Result<serde_json::Value, String>,
        ) -> Result<ContextStep<Ctx>> {
            let mut flag = self.fail_first.lock().unwrap();
            if *flag {
                *flag = false;
                return Err(crate::error::AgentsError::Other(
                    "handler failure".to_string(),
                ));
            }
            ctx.count += 1;
            let payload = match result {
                Ok(v) => serde_json::json!({"value": v, "count": ctx.count}),
                Err(e) => serde_json::json!({"error": e, "count": ctx.count}),
            };
            Ok(ContextStep::rewrite(ctx, payload))
        }
    }

    #[tokio::test]
    async fn test_context_handler_failure_fallback_and_reset() {
        let tool = Arc::new(FunctionTool::simple(
            "uppercase",
            "Converts to uppercase",
            |s: String| s.to_uppercase(),
        ));

        let agent = Agent::simple("CtxFail", "Use tools")
            .with_tool(tool)
            .with_context_factory(Ctx::default, ErrOnceCountingHandler::new());

        // Two consecutive tool calls, then a final message
        let tc1 = MsgToolCall {
            id: uuid::Uuid::new_v4().to_string(),
            name: "uppercase".to_string(),
            arguments: serde_json::json!({"input":"a"}),
        };
        let tc2 = MsgToolCall {
            id: uuid::Uuid::new_v4().to_string(),
            name: "uppercase".to_string(),
            arguments: serde_json::json!({"input":"b"}),
        };

        let provider = Arc::new(
            MockProvider::new("test-model")
                .with_response(crate::items::ModelResponse::new_tool_calls(vec![tc1]))
                .with_response(crate::items::ModelResponse::new_tool_calls(vec![tc2]))
                .with_message("done"),
        );

        let config = RunConfig {
            model_provider: Some(provider),
            ..Default::default()
        };
        let result = Runner::run(agent, "Run", config).await.unwrap();

        // First output forwarded (no rewrite) due to handler failure; second rewritten
        let outputs: Vec<_> = result
            .items
            .iter()
            .filter_map(|item| match item {
                RunItem::ToolOutput(o) => Some(o.output.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0], serde_json::json!("A"));
        assert_eq!(outputs[1]["count"], serde_json::json!(1));
    }

    #[tokio::test]
    async fn test_context_multiple_tool_calls_single_turn_accumulates() {
        let tool = Arc::new(FunctionTool::simple(
            "uppercase",
            "Converts to uppercase",
            |s: String| s.to_uppercase(),
        ));

        let agent = Agent::simple("CtxMulti", "Use tools")
            .with_tool(tool)
            .with_context_factory(Ctx::default, CountingHandler);

        let tc1 = MsgToolCall {
            id: uuid::Uuid::new_v4().to_string(),
            name: "uppercase".to_string(),
            arguments: serde_json::json!({"input":"a"}),
        };
        let tc2 = MsgToolCall {
            id: uuid::Uuid::new_v4().to_string(),
            name: "uppercase".to_string(),
            arguments: serde_json::json!({"input":"b"}),
        };

        let provider = Arc::new(
            MockProvider::new("test-model")
                .with_response(crate::items::ModelResponse::new_tool_calls(vec![tc1, tc2]))
                .with_message("ok"),
        );

        let config = RunConfig {
            model_provider: Some(provider),
            ..Default::default()
        };
        let result = Runner::run(agent, "Run", config).await.unwrap();

        let outputs: Vec<_> = result
            .items
            .iter()
            .filter_map(|item| match item {
                RunItem::ToolOutput(o) => Some(o.output.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0]["count"], serde_json::json!(1));
        assert_eq!(outputs[1]["count"], serde_json::json!(2));
        assert_eq!(outputs[0]["value"], serde_json::json!("A"));
        assert_eq!(outputs[1]["value"], serde_json::json!("B"));
    }

    #[tokio::test]
    async fn test_parallel_tool_calls_preserve_order() {
        let tool = Arc::new(FunctionTool::simple(
            "uppercase",
            "Converts to uppercase",
            |s: String| s.to_uppercase(),
        ));

        let agent = Agent::simple("OrderPreserve", "Use tools").with_tool(tool);

        // Two tool calls in a single turn
        let tc1 = MsgToolCall {
            id: uuid::Uuid::new_v4().to_string(),
            name: "uppercase".to_string(),
            arguments: serde_json::json!({"input":"a"}),
        };
        let tc2 = MsgToolCall {
            id: uuid::Uuid::new_v4().to_string(),
            name: "uppercase".to_string(),
            arguments: serde_json::json!({"input":"b"}),
        };

        let provider = Arc::new(
            MockProvider::new("test-model")
                .with_response(crate::items::ModelResponse::new_tool_calls(vec![tc1, tc2]))
                .with_message("ok"),
        );

        let config = RunConfig {
            model_provider: Some(provider),
            ..Default::default()
        };
        let result = Runner::run(agent, "Run", config).await.unwrap();

        // Extract tool outputs in the order they were appended to items
        let outputs: Vec<_> = result
            .items
            .iter()
            .filter_map(|item| match item {
                RunItem::ToolOutput(o) => Some(o.output.clone()),
                _ => None,
            })
            .collect();

        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0], serde_json::json!("A"));
        assert_eq!(outputs[1], serde_json::json!("B"));
    }

    #[tokio::test]
    async fn test_parallel_mixed_known_unknown_preserve_order_and_errors() {
        let tool = Arc::new(FunctionTool::simple(
            "uppercase",
            "Converts to uppercase",
            |s: String| s.to_uppercase(),
        ));

        let agent = Agent::simple("Mixed", "Use tools").with_tool(tool);

        // Known then unknown tool in single turn
        let tc1 = MsgToolCall {
            id: uuid::Uuid::new_v4().to_string(),
            name: "uppercase".to_string(),
            arguments: serde_json::json!({"input":"a"}),
        };
        let tc2 = MsgToolCall {
            id: uuid::Uuid::new_v4().to_string(),
            name: "nonexistent".to_string(),
            arguments: serde_json::json!({}),
        };

        let provider = Arc::new(
            MockProvider::new("test-model")
                .with_response(crate::items::ModelResponse::new_tool_calls(vec![tc1, tc2]))
                .with_message("ok"),
        );

        let config = RunConfig::default().with_model_provider(Some(provider));
        let result = Runner::run(agent, "Run", config).await.unwrap();

        // Collect outputs, ensure first is success A, second is error
        let outputs: Vec<_> = result
            .items
            .iter()
            .filter_map(|i| match i {
                RunItem::ToolOutput(o) => Some((o.output.clone(), o.error.clone())),
                _ => None,
            })
            .collect();

        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].0, serde_json::json!("A"));
        assert!(outputs[0].1.is_none());
        assert!(outputs[1].0.is_null());
        assert!(outputs[1].1.is_some());
    }

    #[tokio::test]
    async fn test_parallel_with_run_scoped_rewrite() {
        #[derive(Clone, Default)]
        struct RC {
            n: usize,
        }
        struct RH;
        impl ToolContext<RC> for RH {
            fn on_tool_output(
                &self,
                mut ctx: RC,
                _tool: &str,
                _args: &serde_json::Value,
                result: std::result::Result<serde_json::Value, String>,
            ) -> Result<ContextStep<RC>> {
                let new_n = ctx.n + 1;
                ctx.n = new_n;
                let v = result.unwrap_or(serde_json::json!(null));
                Ok(ContextStep::rewrite(
                    ctx,
                    serde_json::json!({"run": new_n, "value": v}),
                ))
            }
        }

        let tool = Arc::new(FunctionTool::simple(
            "uppercase",
            "Converts to uppercase",
            |s: String| s.to_uppercase(),
        ));

        let agent = Agent::simple("ParallelCtx", "Use tools").with_tool(tool);

        let tc1 = MsgToolCall {
            id: uuid::Uuid::new_v4().to_string(),
            name: "uppercase".to_string(),
            arguments: serde_json::json!({"input":"a"}),
        };
        let tc2 = MsgToolCall {
            id: uuid::Uuid::new_v4().to_string(),
            name: "uppercase".to_string(),
            arguments: serde_json::json!({"input":"b"}),
        };

        let provider = Arc::new(
            MockProvider::new("test-model")
                .with_response(crate::items::ModelResponse::new_tool_calls(vec![tc1, tc2]))
                .with_message("ok"),
        );

        let config = RunConfig::default()
            .with_model_provider(Some(provider))
            .with_run_context(RC::default, RH);
        let result = Runner::run(agent, "Run", config).await.unwrap();

        let outputs: Vec<_> = result
            .items
            .iter()
            .filter_map(|i| match i {
                RunItem::ToolOutput(o) => Some(o.output.clone()),
                _ => None,
            })
            .collect();

        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0]["run"], serde_json::json!(1));
        assert_eq!(outputs[0]["value"], serde_json::json!("A"));
        assert_eq!(outputs[1]["run"], serde_json::json!(2));
        assert_eq!(outputs[1]["value"], serde_json::json!("B"));
    }

    #[tokio::test]
    async fn test_parallel_error_rewrite_with_run_scoped_handler() {
        #[derive(Clone, Default)]
        struct RC;
        struct RH;
        impl ToolContext<RC> for RH {
            fn on_tool_output(
                &self,
                ctx: RC,
                tool: &str,
                _args: &serde_json::Value,
                result: std::result::Result<serde_json::Value, String>,
            ) -> Result<ContextStep<RC>> {
                match result {
                    Ok(v) => Ok(ContextStep::rewrite(ctx, v)),
                    Err(_e) => Ok(ContextStep::rewrite(
                        ctx,
                        serde_json::json!(format!("RECOVERED:{}", tool)),
                    )),
                }
            }
        }

        let tool_ok = Arc::new(FunctionTool::simple("ok", "ok", |s: String| {
            s.to_uppercase()
        }));
        let tool_fail = Arc::new(crate::tool::FunctionTool::new(
            "fail".to_string(),
            "Always fails".to_string(),
            serde_json::json!({"type":"object"}),
            |_args| {
                Err(crate::error::AgentsError::ToolExecutionError {
                    message: "boom".into(),
                })
            },
        ));

        let agent = Agent::simple("ErrRewrite", "Use tools")
            .with_tool(tool_ok)
            .with_tool(tool_fail);

        let tc1 = MsgToolCall {
            id: uuid::Uuid::new_v4().to_string(),
            name: "fail".to_string(),
            arguments: serde_json::json!({}),
        };
        let tc2 = MsgToolCall {
            id: uuid::Uuid::new_v4().to_string(),
            name: "ok".to_string(),
            arguments: serde_json::json!({"input":"a"}),
        };
        let provider = Arc::new(
            MockProvider::new("test-model")
                .with_response(crate::items::ModelResponse::new_tool_calls(vec![tc1, tc2]))
                .with_message("ok"),
        );
        let config = RunConfig::default()
            .with_model_provider(Some(provider))
            .with_run_context(RC::default, RH);
        let result = Runner::run(agent, "Run", config).await.unwrap();

        // Expect first output recovered string, second uppercase A
        let outputs: Vec<_> = result
            .items
            .iter()
            .filter_map(|i| match i {
                RunItem::ToolOutput(o) => Some(o.output.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0], serde_json::json!("RECOVERED:fail"));
        assert_eq!(outputs[1], serde_json::json!("A"));
    }

    #[tokio::test]
    async fn test_streaming_with_context_rewrite_collects() {
        let tool = Arc::new(FunctionTool::simple(
            "uppercase",
            "Converts to uppercase",
            |s: String| s.to_uppercase(),
        ));

        let agent = Agent::simple("CtxStream", "Use tools")
            .with_tool(tool)
            .with_context_factory(Ctx::default, CountingHandler);

        let provider = Arc::new(
            MockProvider::new("test-model")
                .with_tool_call("uppercase", serde_json::json!({"input":"a"}))
                .with_message("done"),
        );

        let config = RunConfig {
            model_provider: Some(provider),
            ..Default::default()
        };
        let streaming = Runner::run_stream(agent, "Run", config).await.unwrap();
        let result = streaming.collect().await.unwrap();

        let outputs: Vec<_> = result
            .items
            .iter()
            .filter_map(|item| match item {
                RunItem::ToolOutput(o) => Some(o.output.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0]["count"], serde_json::json!(1));
        assert_eq!(outputs[0]["value"], serde_json::json!("A"));
    }

    #[tokio::test]
    async fn test_streaming_run_scoped_rewrite_collects() {
        #[derive(Clone, Default)]
        struct RC {
            n: usize,
        }
        struct RH;
        impl ToolContext<RC> for RH {
            fn on_tool_output(
                &self,
                mut ctx: RC,
                _tool: &str,
                _args: &serde_json::Value,
                result: std::result::Result<serde_json::Value, String>,
            ) -> Result<ContextStep<RC>> {
                ctx.n += 1;
                Ok(ContextStep::rewrite(
                    ctx,
                    result.unwrap_or(serde_json::json!(null)),
                ))
            }
        }

        let tool = Arc::new(FunctionTool::simple(
            "uppercase",
            "Converts to uppercase",
            |s: String| s.to_uppercase(),
        ));
        let agent = Agent::simple("RSStream", "Use tools").with_tool(tool);

        let provider = Arc::new(
            MockProvider::new("test-model")
                .with_tool_call("uppercase", serde_json::json!({"input":"abc"}))
                .with_message("ok"),
        );

        let config = RunConfig::default()
            .with_model_provider(Some(provider))
            .with_run_context(RC::default, RH);
        let streaming = Runner::run_stream(agent, "Run", config).await.unwrap();
        let result = streaming.collect().await.unwrap();

        let outputs: Vec<_> = result
            .items
            .iter()
            .filter_map(|i| match i {
                RunItem::ToolOutput(o) => Some(o.output.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(outputs.len(), 1);
        // We don't have the RC value here, but rewrite applied should reflect uppercase
        assert_eq!(outputs[0], serde_json::json!("ABC"));
    }

    #[tokio::test]
    async fn test_run_scoped_run_with_run_context_typed_api() {
        #[derive(Clone, Default)]
        struct RC {
            n: usize,
        }

        struct Inc;
        impl ToolContext<RC> for Inc {
            fn on_tool_output(
                &self,
                mut ctx: RC,
                _tool: &str,
                _args: &serde_json::Value,
                result: std::result::Result<serde_json::Value, String>,
            ) -> Result<ContextStep<RC>> {
                ctx.n += 1;
                Ok(ContextStep::rewrite(
                    ctx,
                    result.unwrap_or(serde_json::json!(null)),
                ))
            }
        }

        let tool = Arc::new(FunctionTool::simple(
            "uppercase",
            "Converts to uppercase",
            |s: String| s.to_uppercase(),
        ));
        let agent = Agent::simple("TypedRun", "Use tools").with_tool(tool);

        let provider = Arc::new(
            MockProvider::new("test-model")
                .with_tool_call("uppercase", serde_json::json!({"input":"abc"}))
                .with_message("ok"),
        );

        let config = RunConfig::default().with_model_provider(Some(provider));
        let out = Runner::run_with_run_context(agent, "Run", config, RC::default, Inc)
            .await
            .unwrap();

        assert!(out.result.is_success());
        assert_eq!(out.context.n, 1);
    }

    struct RunScopedErrOnce {
        fail_first: Mutex<bool>,
    }
    impl RunScopedErrOnce {
        fn new() -> Self {
            Self {
                fail_first: Mutex::new(true),
            }
        }
    }
    impl ToolContext<RunCtx> for RunScopedErrOnce {
        fn on_tool_output(
            &self,
            mut ctx: RunCtx,
            _tool: &str,
            _args: &serde_json::Value,
            result: std::result::Result<serde_json::Value, String>,
        ) -> Result<ContextStep<RunCtx>> {
            let mut flag = self.fail_first.lock().unwrap();
            if *flag {
                *flag = false;
                return Err(crate::error::AgentsError::Other(
                    "run handler fail".to_string(),
                ));
            }
            ctx.n += 1;
            Ok(ContextStep::rewrite(
                ctx,
                result.unwrap_or(serde_json::json!(null)),
            ))
        }
    }

    #[tokio::test]
    async fn test_run_scoped_handler_failure_fallback_and_reset() {
        let tool = Arc::new(FunctionTool::simple(
            "uppercase",
            "Converts to uppercase",
            |s: String| s.to_uppercase(),
        ));
        let agent = Agent::simple("RunFail", "Use tools").with_tool(tool);

        let tc1 = MsgToolCall {
            id: uuid::Uuid::new_v4().to_string(),
            name: "uppercase".to_string(),
            arguments: serde_json::json!({"input":"a"}),
        };
        let tc2 = MsgToolCall {
            id: uuid::Uuid::new_v4().to_string(),
            name: "uppercase".to_string(),
            arguments: serde_json::json!({"input":"b"}),
        };
        let provider = Arc::new(
            MockProvider::new("test-model")
                .with_response(crate::items::ModelResponse::new_tool_calls(vec![tc1]))
                .with_response(crate::items::ModelResponse::new_tool_calls(vec![tc2]))
                .with_message("done"),
        );

        let config = RunConfig::default()
            .with_run_context(RunCtx::default, RunScopedErrOnce::new())
            .with_model_provider(Some(provider));
        let result = Runner::run(agent, "Run", config).await.unwrap();

        let outputs: Vec<_> = result
            .items
            .iter()
            .filter_map(|i| match i {
                RunItem::ToolOutput(o) => Some(o.output.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(outputs.len(), 2);
        // First forwarded (no rewrite) == "A"; second rewritten payload has no enforced shape here; ensure not Null
        assert_eq!(outputs[0], serde_json::json!("A"));
        assert!(outputs[1] != serde_json::Value::Null);
    }

    struct RunScopedFinal;
    impl ToolContext<RunCtx> for RunScopedFinal {
        fn on_tool_output(
            &self,
            ctx: RunCtx,
            _tool: &str,
            _args: &serde_json::Value,
            _result: std::result::Result<serde_json::Value, String>,
        ) -> Result<ContextStep<RunCtx>> {
            Ok(ContextStep::final_output(ctx, serde_json::json!("stop")))
        }
    }

    #[tokio::test]
    async fn test_run_scoped_finalization_precedence() {
        let tool = Arc::new(FunctionTool::simple(
            "uppercase",
            "Converts to uppercase",
            |s: String| s.to_uppercase(),
        ));
        let agent = Agent::simple("RunFinal", "Use tools").with_tool(tool);
        let provider = Arc::new(
            MockProvider::new("test-model")
                .with_tool_call("uppercase", serde_json::json!({"input":"x"})),
        );
        let config = RunConfig::default()
            .with_run_context(RunCtx::default, RunScopedFinal)
            .with_model_provider(Some(provider));
        let result = Runner::run(agent, "Run", config).await.unwrap();
        assert!(result.is_success());
        assert_eq!(result.final_output, serde_json::json!("stop"));
    }

    #[tokio::test]
    async fn test_agent_group_handoff_executes() {
        use crate::group::AgentGroupBuilder;
        let specialist = Agent::simple("Specialist", "I handle special tasks");
        let root = Agent::simple("Coordinator", "Delegate to Specialist");
        let group = AgentGroupBuilder::new(root)
            .with_handoff(specialist, "Do work")
            .build();

        let provider = Arc::new(
            MockProvider::new("test-model")
                .with_tool_call("Specialist", serde_json::json!({}))
                .with_message("Task complete"),
        );

        let config = RunConfig::default().with_model_provider(Some(provider));
        let result = Runner::run(group.into_agent(), "Run", config)
            .await
            .unwrap();

        assert!(result.is_success());
        assert_eq!(result.final_agent, "Specialist");
        assert!(result
            .items
            .iter()
            .any(|i| matches!(i, RunItem::Handoff(_))));
    }

    #[tokio::test]
    async fn test_agent_group_with_run_scoped_context() {
        use crate::group::AgentGroupBuilder;

        #[derive(Clone, Default)]
        struct RC {
            n: usize,
        }
        struct RH;
        impl ToolContext<RC> for RH {
            fn on_tool_output(
                &self,
                mut ctx: RC,
                _tool: &str,
                _args: &serde_json::Value,
                result: std::result::Result<serde_json::Value, String>,
            ) -> Result<ContextStep<RC>> {
                ctx.n += 1;
                Ok(ContextStep::rewrite(
                    ctx,
                    result.unwrap_or(serde_json::json!(null)),
                ))
            }
        }

        let tool = Arc::new(FunctionTool::simple(
            "uppercase",
            "Converts to uppercase",
            |s: String| s.to_uppercase(),
        ));

        let specialist = Agent::simple("Specialist", "Use tools").with_tool(tool);
        let root = Agent::simple("Coordinator", "Delegate to Specialist");
        let group = AgentGroupBuilder::new(root)
            .with_handoff(specialist, "Do work")
            .build();

        let provider = Arc::new(
            MockProvider::new("test-model")
                .with_tool_call("Specialist", serde_json::json!({}))
                .with_tool_call("uppercase", serde_json::json!({"input":"x"}))
                .with_message("done"),
        );

        let config = RunConfig::default()
            .with_model_provider(Some(provider))
            .with_run_context(RC::default, RH);
        let result = Runner::run(group.into_agent(), "Run", config)
            .await
            .unwrap();

        // Ensure run-scoped rewrite applied to the tool output after handoff
        let outputs: Vec<_> = result
            .items
            .iter()
            .filter_map(|i| match i {
                RunItem::ToolOutput(o) => Some(o.output.clone()),
                _ => None,
            })
            .collect();
        assert!(!outputs.is_empty());
        assert_eq!(outputs.last().unwrap(), &serde_json::json!("X"));
    }

    #[tokio::test]
    async fn test_handoffs_are_exposed_as_tools_to_provider() {
        use async_trait::async_trait;

        // Spy provider that records tool names passed into `complete`
        struct ProviderSpy {
            model: String,
            recorded: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
        }

        #[async_trait]
        impl crate::model::ModelProvider for ProviderSpy {
            async fn complete(
                &self,
                _messages: Vec<crate::items::Message>,
                tools: Vec<std::sync::Arc<dyn crate::tool::Tool>>,
                _temperature: Option<f32>,
                _max_tokens: Option<u32>,
            ) -> crate::error::Result<(crate::items::ModelResponse, crate::usage::Usage)>
            {
                let mut names: Vec<String> = tools.iter().map(|t| t.name().to_string()).collect();
                names.sort();
                self.recorded.lock().unwrap().extend(names);
                Ok((
                    crate::items::ModelResponse::new_message("ok"),
                    crate::usage::Usage::new(0, 0),
                ))
            }

            fn model_name(&self) -> &str {
                &self.model
            }
        }

        let recorded = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let spy = std::sync::Arc::new(ProviderSpy {
            model: "spy".to_string(),
            recorded: recorded.clone(),
        });

        let specialist = Agent::simple("Specialist", "I handle special tasks");
        let handoff = crate::handoff::Handoff::new(specialist, "Do work");
        let coordinator =
            Agent::simple("Coordinator", "Delegate to Specialist").with_handoff(handoff);

        let config = RunConfig::default().with_model_provider(Some(spy));
        let _ = Runner::run(coordinator, "Hi", config).await.unwrap();

        let tools_seen = recorded.lock().unwrap().clone();
        assert!(
            tools_seen.iter().any(|n| n == "Specialist"),
            "handoff name should be exposed as a tool"
        );
    }

    #[tokio::test]
    async fn test_handoff_generates_tool_message_reply() {
        use async_trait::async_trait;

        // Spy provider that returns a handoff tool_call first, then records messages on second call
        struct Spy {
            model: String,
            first_id: std::sync::Arc<std::sync::Mutex<Option<String>>>,
            saw_tool_reply: std::sync::Arc<std::sync::Mutex<bool>>,
        }

        #[async_trait]
        impl crate::model::ModelProvider for Spy {
            async fn complete(
                &self,
                messages: Vec<crate::items::Message>,
                _tools: Vec<std::sync::Arc<dyn crate::tool::Tool>>,
                _temperature: Option<f32>,
                _max_tokens: Option<u32>,
            ) -> crate::error::Result<(crate::items::ModelResponse, crate::usage::Usage)>
            {
                let mut guard = self.first_id.lock().unwrap();
                if guard.is_none() {
                    // First call: return a tool_call for a handoff named "Specialist"
                    let tc = crate::items::ToolCall {
                        id: "call_1".to_string(),
                        name: "Specialist".to_string(),
                        arguments: serde_json::json!({}),
                    };
                    *guard = Some(tc.id.clone());
                    Ok((
                        crate::items::ModelResponse::new_tool_calls(vec![tc]),
                        crate::usage::Usage::new(0, 0),
                    ))
                } else {
                    // Second call: ensure there is a tool message responding to the tool_call_id
                    let id = guard.clone().unwrap();
                    let has_tool_reply = messages.iter().any(|m| {
                        m.role == crate::items::Role::Tool && m.tool_call_id.as_deref() == Some(&id)
                    });
                    *self.saw_tool_reply.lock().unwrap() = has_tool_reply;
                    Ok((
                        crate::items::ModelResponse::new_message("ok"),
                        crate::usage::Usage::new(0, 0),
                    ))
                }
            }

            fn model_name(&self) -> &str {
                &self.model
            }
        }

        let spy = std::sync::Arc::new(Spy {
            model: "spy".to_string(),
            first_id: std::sync::Arc::new(std::sync::Mutex::new(None)),
            saw_tool_reply: std::sync::Arc::new(std::sync::Mutex::new(false)),
        });

        let specialist = Agent::simple("Specialist", "I handle special tasks");
        let coordinator = Agent::simple("Coordinator", "Delegate to Specialist")
            .with_handoff(crate::handoff::Handoff::new(specialist, "Do work"));
        let config = RunConfig::default().with_model_provider(Some(spy.clone()));
        let _ = Runner::run(coordinator, "Hi", config).await.unwrap();

        assert!(
            *spy.saw_tool_reply.lock().unwrap(),
            "Expected a tool message replying to the handoff tool_call_id"
        );
    }
    #[derive(Clone, Default)]
    struct RunCtx {
        n: usize,
    }

    struct RunScopedRewrite;
    impl ToolContext<RunCtx> for RunScopedRewrite {
        fn on_tool_output(
            &self,
            mut ctx: RunCtx,
            _tool: &str,
            _args: &serde_json::Value,
            result: std::result::Result<serde_json::Value, String>,
        ) -> Result<ContextStep<RunCtx>> {
            ctx.n += 1;
            let payload = match result {
                Ok(v) => serde_json::json!({"run": ctx.n, "value": v}),
                Err(e) => serde_json::json!({"run": ctx.n, "error": e}),
            };
            Ok(ContextStep::rewrite(ctx, payload))
        }
    }

    #[tokio::test]
    async fn test_run_scoped_rewrite_across_handoff() {
        // Agent A hands off to B; run-scoped context should rewrite both outputs
        let tool = Arc::new(FunctionTool::simple(
            "uppercase",
            "Converts to uppercase",
            |s: String| s.to_uppercase(),
        ));

        let agent_a = Agent::simple("A", "Use tools").with_tool(tool.clone());
        let agent_b = Agent::simple("B", "Use tools").with_tool(tool);
        let handoff = crate::handoff::Handoff::new(agent_b, "B does follow-up");
        let agent = agent_a.with_handoff(handoff);

        // First turn: A calls tool; second turn: after handoff, B calls tool
        let tc1 = MsgToolCall {
            id: uuid::Uuid::new_v4().to_string(),
            name: "uppercase".to_string(),
            arguments: serde_json::json!({"input":"a"}),
        };
        let tc2 = MsgToolCall {
            id: uuid::Uuid::new_v4().to_string(),
            name: "uppercase".to_string(),
            arguments: serde_json::json!({"input":"b"}),
        };

        let provider = Arc::new(
            MockProvider::new("test-model")
                .with_response(crate::items::ModelResponse::new_tool_calls(vec![tc1]))
                .with_response(crate::items::ModelResponse::new_tool_calls(vec![tc2]))
                .with_message("done"),
        );

        let config = RunConfig::default()
            .with_run_context(RunCtx::default, RunScopedRewrite)
            .with_model_provider(Some(provider));

        let result = Runner::run(agent, "Run", config).await.unwrap();

        let outputs: Vec<_> = result
            .items
            .iter()
            .filter_map(|i| match i {
                RunItem::ToolOutput(o) => Some(o.output.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0]["run"], serde_json::json!(1));
        assert_eq!(outputs[0]["value"], serde_json::json!("A"));
        assert_eq!(outputs[1]["run"], serde_json::json!(2));
        assert_eq!(outputs[1]["value"], serde_json::json!("B"));
    }

    struct AgentRewrite;
    impl ToolContext<Ctx> for AgentRewrite {
        fn on_tool_output(
            &self,
            mut ctx: Ctx,
            _tool_name: &str,
            _args: &serde_json::Value,
            result: std::result::Result<serde_json::Value, String>,
        ) -> Result<ContextStep<Ctx>> {
            let v = result.unwrap_or(serde_json::json!(null));
            let wrapped = serde_json::json!({"agent": true, "inner": v});
            ctx.count += 1;
            Ok(ContextStep::rewrite(ctx, wrapped))
        }
    }

    #[tokio::test]
    async fn test_run_scoped_then_per_agent_ordering() {
        let tool = Arc::new(FunctionTool::simple(
            "uppercase",
            "Converts to uppercase",
            |s: String| s.to_uppercase(),
        ));

        let agent = Agent::simple("Order", "Use tools")
            .with_tool(tool)
            .with_context_factory(Ctx::default, AgentRewrite);

        let provider = Arc::new(
            MockProvider::new("test-model")
                .with_tool_call("uppercase", serde_json::json!({"input":"x"}))
                .with_message("ok"),
        );

        let config = RunConfig::default()
            .with_run_context(RunCtx::default, RunScopedRewrite)
            .with_model_provider(Some(provider));
        let result = Runner::run(agent, "Run", config).await.unwrap();

        let outputs: Vec<_> = result
            .items
            .iter()
            .filter_map(|i| match i {
                RunItem::ToolOutput(o) => Some(o.output.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(outputs.len(), 1);
        // Outer layer from per-agent, inner value from run-scoped rewrite
        assert_eq!(outputs[0]["agent"], serde_json::json!(true));
        assert_eq!(outputs[0]["inner"]["run"], serde_json::json!(1));
        assert_eq!(outputs[0]["inner"]["value"], serde_json::json!("X"));
    }

    #[tokio::test]
    async fn test_tool_scope_timeout_layer_times_out() {
        // Slow tool that sleeps 50ms
        let slow = FunctionTool::new(
            "slow".to_string(),
            "Sleeps".to_string(),
            serde_json::json!({"type":"object"}),
            |_args| {
                std::thread::sleep(std::time::Duration::from_millis(50));
                Ok(serde_json::json!("ok"))
            },
        )
        .layer(layers::boxed_timeout_secs(0));

        let agent = Agent::simple("TimeoutAgent", "Use tools").with_tool(Arc::new(slow));

        let tc = MsgToolCall {
            id: uuid::Uuid::new_v4().to_string(),
            name: "slow".to_string(),
            arguments: serde_json::json!({}),
        };
        let provider = Arc::new(
            MockProvider::new("test-model")
                .with_response(crate::items::ModelResponse::new_tool_calls(vec![tc]))
                .with_message("done"),
        );

        let config = RunConfig::default().with_model_provider(Some(provider));
        let result = Runner::run(agent, "Run", config).await.unwrap();
        let mut saw_timeout = false;
        for item in &result.items {
            if let RunItem::ToolOutput(o) = item {
                if let Some(err) = &o.error {
                    saw_timeout |= err.contains("timeout")
                }
            }
        }
        assert!(saw_timeout, "expected timeout error from tool-scope layer");
    }

    #[tokio::test]
    async fn test_max_concurrency_limits_parallel_execution() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        // Tracking concurrent executions
        static ACTIVE: once_cell::sync::Lazy<AtomicUsize> =
            once_cell::sync::Lazy::new(|| AtomicUsize::new(0));
        static MAX_OBSERVED: once_cell::sync::Lazy<AtomicUsize> =
            once_cell::sync::Lazy::new(|| AtomicUsize::new(0));

        let tool = Arc::new(FunctionTool::new(
            "block".to_string(),
            "Blocks briefly".to_string(),
            serde_json::json!({"type":"object"}),
            |_args| {
                let current = ACTIVE.fetch_add(1, Ordering::SeqCst) + 1;
                // update max observed
                loop {
                    let max = MAX_OBSERVED.load(Ordering::SeqCst);
                    if current <= max
                        || MAX_OBSERVED
                            .compare_exchange(max, current, Ordering::SeqCst, Ordering::SeqCst)
                            .is_ok()
                    {
                        break;
                    }
                }
                std::thread::sleep(std::time::Duration::from_millis(30));
                ACTIVE.fetch_sub(1, Ordering::SeqCst);
                Ok(serde_json::json!("ok"))
            },
        ));

        let agent = Agent::simple("Limiter", "Use tools").with_tool(tool);
        let tc1 = MsgToolCall {
            id: uuid::Uuid::new_v4().to_string(),
            name: "block".into(),
            arguments: serde_json::json!({}),
        };
        let tc2 = MsgToolCall {
            id: uuid::Uuid::new_v4().to_string(),
            name: "block".into(),
            arguments: serde_json::json!({}),
        };
        let tc3 = MsgToolCall {
            id: uuid::Uuid::new_v4().to_string(),
            name: "block".into(),
            arguments: serde_json::json!({}),
        };
        let tc4 = MsgToolCall {
            id: uuid::Uuid::new_v4().to_string(),
            name: "block".into(),
            arguments: serde_json::json!({}),
        };
        let provider = Arc::new(
            MockProvider::new("test-model")
                .with_response(crate::items::ModelResponse::new_tool_calls(vec![
                    tc1, tc2, tc3, tc4,
                ]))
                .with_message("done"),
        );

        // Reset counters
        ACTIVE.store(0, Ordering::SeqCst);
        MAX_OBSERVED.store(0, Ordering::SeqCst);

        let config = RunConfig::default()
            .with_model_provider(Some(provider))
            .with_parallel_tools(true)
            .with_max_concurrency(2);
        let _ = Runner::run(agent, "Run", config).await.unwrap();

        let max_seen = MAX_OBSERVED.load(Ordering::SeqCst);
        assert!(
            max_seen <= 2,
            "expected max concurrency <= 2, got {}",
            max_seen
        );
    }

    #[tokio::test]
    async fn test_retry_layer_retries_under_parallel_and_preserves_order() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static ATTEMPTS_A: once_cell::sync::Lazy<AtomicUsize> =
            once_cell::sync::Lazy::new(|| AtomicUsize::new(0));
        static ATTEMPTS_B: once_cell::sync::Lazy<AtomicUsize> =
            once_cell::sync::Lazy::new(|| AtomicUsize::new(0));

        // Tool A fails first 1 attempt, then succeeds
        let tool_a = Arc::new(FunctionTool::new(
            "flakyA".to_string(),
            "Flaky tool A".to_string(),
            serde_json::json!({"type":"object"}),
            |_args| {
                let n = ATTEMPTS_A.fetch_add(1, Ordering::SeqCst) + 1;
                if n <= 1 {
                    Err(crate::error::AgentsError::ToolExecutionError {
                        message: "temporary failure A".to_string(),
                    })
                } else {
                    Ok(serde_json::json!("okA"))
                }
            },
        ));

        // Tool B fails first 2 attempts, then succeeds
        let tool_b = Arc::new(FunctionTool::new(
            "flakyB".to_string(),
            "Flaky tool B".to_string(),
            serde_json::json!({"type":"object"}),
            |_args| {
                let n = ATTEMPTS_B.fetch_add(1, Ordering::SeqCst) + 1;
                if n <= 2 {
                    Err(crate::error::AgentsError::ToolExecutionError {
                        message: "temporary failure B".to_string(),
                    })
                } else {
                    Ok(serde_json::json!("okB"))
                }
            },
        ));

        let agent = Agent::simple("RetryPar", "Use tools")
            .with_tool(tool_a)
            .with_tool(tool_b);

        let tc1 = MsgToolCall {
            id: uuid::Uuid::new_v4().to_string(),
            name: "flakyA".into(),
            arguments: serde_json::json!({}),
        };
        let tc2 = MsgToolCall {
            id: uuid::Uuid::new_v4().to_string(),
            name: "flakyB".into(),
            arguments: serde_json::json!({}),
        };
        let provider = Arc::new(
            MockProvider::new("test-model")
                .with_response(crate::items::ModelResponse::new_tool_calls(vec![tc1, tc2]))
                .with_message("done"),
        );

        ATTEMPTS_A.store(0, Ordering::SeqCst);
        ATTEMPTS_B.store(0, Ordering::SeqCst);

        let config = RunConfig::default()
            .with_model_provider(Some(provider))
            .with_parallel_tools(true)
            .with_run_layers(vec![crate::layers::boxed_retry_times(3)]);

        let result = Runner::run(agent, "Run", config).await.unwrap();
        assert!(result.is_success());

        // Verify retries happened as expected
        assert!(ATTEMPTS_A.load(Ordering::SeqCst) >= 2);
        assert!(ATTEMPTS_B.load(Ordering::SeqCst) >= 3);
    }

    #[tokio::test]
    async fn test_parallel_run_scope_timeout_layer_times_out() {
        // Two slow tools; run-scope timeout forces both to time out even when run in parallel.
        let slow1 = Arc::new(FunctionTool::new(
            "slow1".to_string(),
            "Sleeps".to_string(),
            serde_json::json!({"type":"object"}),
            |_args| {
                std::thread::sleep(std::time::Duration::from_millis(50));
                Ok(serde_json::json!("ok1"))
            },
        ));
        let slow2 = Arc::new(FunctionTool::new(
            "slow2".to_string(),
            "Sleeps".to_string(),
            serde_json::json!({"type":"object"}),
            |_args| {
                std::thread::sleep(std::time::Duration::from_millis(50));
                Ok(serde_json::json!("ok2"))
            },
        ));

        let agent = Agent::simple("TimeoutRun", "Use tools")
            .with_tool(slow1)
            .with_tool(slow2);

        let tc1 = MsgToolCall {
            id: uuid::Uuid::new_v4().to_string(),
            name: "slow1".to_string(),
            arguments: serde_json::json!({}),
        };
        let tc2 = MsgToolCall {
            id: uuid::Uuid::new_v4().to_string(),
            name: "slow2".to_string(),
            arguments: serde_json::json!({}),
        };
        let provider = Arc::new(
            MockProvider::new("test-model")
                .with_response(crate::items::ModelResponse::new_tool_calls(vec![tc1, tc2]))
                .with_message("ok"),
        );

        let config = RunConfig::default()
            .with_model_provider(Some(provider))
            .with_parallel_tools(true)
            .with_run_layers(vec![layers::boxed_timeout_secs(0)]);
        let result = Runner::run(agent, "Run", config).await.unwrap();

        // Expect two tool outputs with timeout errors
        let outs: Vec<_> = result
            .items
            .iter()
            .filter_map(|i| match i {
                RunItem::ToolOutput(o) => Some(o.error.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(outs.len(), 2);
        assert!(outs
            .iter()
            .all(|e| e.as_ref().map(|s| s.contains("timeout")).unwrap_or(false)));
    }

    #[tokio::test]
    async fn test_parallel_handoff_short_circuits_other_tool_calls() {
        // Specialist agent (no tools required for this test)
        let specialist = Agent::simple("Specialist", "I handle special tasks");
        let handoff = crate::handoff::Handoff::new(specialist, "Do work");

        // Coordinator with a regular tool plus a handoff
        let up = Arc::new(FunctionTool::simple(
            "uppercase",
            "Converts to uppercase",
            |s: String| s.to_uppercase(),
        ));
        let coordinator = Agent::simple("Coordinator", "Delegate and use tools")
            .with_tool(up)
            .with_handoff(handoff);

        // First provider response includes BOTH a handoff tool call and a regular tool call
        let tc_h = MsgToolCall {
            id: uuid::Uuid::new_v4().to_string(),
            name: "Specialist".to_string(),
            arguments: serde_json::json!({}),
        };
        let tc_u = MsgToolCall {
            id: uuid::Uuid::new_v4().to_string(),
            name: "uppercase".to_string(),
            arguments: serde_json::json!({"input":"x"}),
        };
        let provider = Arc::new(
            MockProvider::new("test-model")
                .with_response(crate::items::ModelResponse::new_tool_calls(vec![
                    tc_h, tc_u,
                ]))
                .with_message("done"),
        );

        let config = RunConfig::default().with_model_provider(Some(provider));
        let result = Runner::run(coordinator, "Run", config).await.unwrap();

        // We expect: Handoff performed, ACK tool message inserted; the non-handoff tool call is ignored in that turn
        let tool_calls: Vec<_> = result
            .items
            .iter()
            .filter_map(|i| match i {
                RunItem::ToolCall(c) => Some(c.tool_name.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(
            tool_calls.len(),
            1,
            "only the handoff ToolCall should be recorded"
        );
        assert_eq!(tool_calls[0], "Specialist");

        // Ensure a Handoff item exists
        assert!(result
            .items
            .iter()
            .any(|i| matches!(i, RunItem::Handoff(_))));

        // Ensure the uppercase tool was NOT executed in that turn (no ToolOutput with uppercase result "X")
        let had_uppercase_output = result.items.iter().any(|i| match i {
            RunItem::ToolOutput(o) => o.output == serde_json::json!("X"),
            _ => false,
        });
        assert!(!had_uppercase_output);
    }

    #[tokio::test]
    async fn test_parallel_with_session_persistence() {
        use tempfile::NamedTempFile;
        let tmp = NamedTempFile::new().unwrap();
        let db_path = tmp.path().to_path_buf();

        // Two simple tools executed in parallel
        let t1 = Arc::new(FunctionTool::simple("t1", "echo", |s: String| s));
        let t2 = Arc::new(FunctionTool::simple("t2", "upper", |s: String| {
            s.to_uppercase()
        }));
        let agent = Agent::simple("PersistPar", "Use tools")
            .with_tool(t1)
            .with_tool(t2);

        let c1 = MsgToolCall {
            id: uuid::Uuid::new_v4().to_string(),
            name: "t1".to_string(),
            arguments: serde_json::json!({"input":"a"}),
        };
        let c2 = MsgToolCall {
            id: uuid::Uuid::new_v4().to_string(),
            name: "t2".to_string(),
            arguments: serde_json::json!({"input":"b"}),
        };
        let provider = Arc::new(
            MockProvider::new("test-model")
                .with_response(crate::items::ModelResponse::new_tool_calls(vec![c1, c2]))
                .with_message("ok"),
        );

        let session = Arc::new(
            crate::sqlite_session::SqliteSession::new("sess", &db_path)
                .await
                .unwrap(),
        );
        let config = RunConfig::default()
            .with_model_provider(Some(provider))
            .with_parallel_tools(true)
            .with_max_concurrency(2);
        let config = RunConfig {
            session: Some(session.clone()),
            ..config
        };

        let result = Runner::run(agent, "Run", config).await.unwrap();
        assert!(result.is_success());

        // Verify persisted items
        let saved = session.get_items(None).await.unwrap();
        assert!(
            !saved.is_empty(),
            "expected items to be persisted to session"
        );
        // Should contain at least one assistant message (tool_calls) and two tool outputs
        let tool_outputs = saved
            .iter()
            .filter(|i| matches!(i, RunItem::ToolOutput(_)))
            .count();
        assert!(tool_outputs >= 2);
    }

    #[tokio::test]
    async fn test_parallel_approval_layer_denies_specific_tool() {
        // Two tools; approval predicate denies one of them.
        let safe = Arc::new(FunctionTool::simple("safe", "ok", |s: String| s));
        let danger = Arc::new(FunctionTool::simple("danger", "", |s: String| s));

        let agent = Agent::simple("ApprovalPar", "Use tools")
            .with_tool(safe)
            .with_tool(danger);

        let c1 = MsgToolCall {
            id: uuid::Uuid::new_v4().to_string(),
            name: "danger".to_string(),
            arguments: serde_json::json!({"input":"a"}),
        };
        let c2 = MsgToolCall {
            id: uuid::Uuid::new_v4().to_string(),
            name: "safe".to_string(),
            arguments: serde_json::json!({"input":"b"}),
        };
        let provider = Arc::new(
            MockProvider::new("test-model")
                .with_response(crate::items::ModelResponse::new_tool_calls(vec![c1, c2]))
                .with_message("ok"),
        );

        let cfg = RunConfig::default()
            .with_model_provider(Some(provider))
            .with_parallel_tools(true)
            .with_run_layers(vec![layers::boxed_approval_with(|_agent, tool, _args| {
                tool != "danger"
            })]);
        let result = Runner::run(agent, "Run", cfg).await.unwrap();

        // Expect one denied with approval error, one success
        let mut saw_denied = false;
        let mut saw_safe = false;
        for item in &result.items {
            if let RunItem::ToolOutput(o) = item {
                if let Some(err) = &o.error {
                    if err.contains("approval") {
                        saw_denied = true;
                    }
                } else if o.output == serde_json::json!("b") {
                    saw_safe = true;
                }
            }
        }
        assert!(
            saw_denied,
            "expected danger tool to be denied by approval layer"
        );
        assert!(saw_safe, "expected safe tool to execute successfully");
    }

    #[tokio::test]
    async fn test_max_turns_exceeded() {
        let agent = Agent::simple("LoopAgent", "Keep asking questions");

        // Provider that always returns tool calls (would loop forever)
        let provider = Arc::new(
            MockProvider::new("test-model")
                .with_tool_call("nonexistent", serde_json::json!({}))
                .with_tool_call("nonexistent", serde_json::json!({}))
                .with_tool_call("nonexistent", serde_json::json!({})),
        );

        let config = RunConfig {
            model_provider: Some(provider),
            max_turns: Some(2),
            ..Default::default()
        };

        let result = Runner::run(agent, "Start", config).await;

        assert!(result.is_err());
        if let Err(AgentsError::MaxTurnsExceeded { max_turns }) = result {
            assert_eq!(max_turns, 2);
        } else {
            panic!("Expected MaxTurnsExceeded error");
        }
    }

    #[tokio::test]
    async fn test_handoff() {
        let specialist = Agent::simple("Specialist", "I handle special tasks");
        let handoff = crate::handoff::Handoff::new(specialist, "Handles special requests");

        let main_agent = Agent::simple("Main", "I delegate to specialists").with_handoff(handoff);

        let provider = Arc::new(
            MockProvider::new("test-model")
                // First response triggers handoff
                .with_tool_call("Specialist", serde_json::json!({}))
                // Second response from specialist
                .with_message("Task completed by specialist"),
        );

        let config = RunConfig {
            model_provider: Some(provider),
            ..Default::default()
        };

        let result = Runner::run(main_agent, "Do something special", config)
            .await
            .unwrap();

        assert!(result.is_success());
        assert_eq!(result.final_agent, "Specialist");
        assert!(result
            .items
            .iter()
            .any(|item| matches!(item, RunItem::Handoff(_))));
    }

    #[tokio::test]
    async fn test_streaming() {
        let agent = Agent::simple("StreamAgent", "Streaming test");

        let provider = Arc::new(MockProvider::new("test-model").with_message("Streamed response"));

        let config = RunConfig {
            model_provider: Some(provider),
            ..Default::default()
        };

        let streaming = Runner::run_stream(agent, "Stream test", config)
            .await
            .unwrap();
        let result = streaming.collect().await.unwrap();

        assert!(result.is_success());
        assert_eq!(result.final_output, serde_json::json!("Streamed response"));
    }
}
