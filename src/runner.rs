//! # Agent Execution Runner
//!
//! The [`Runner`] is the engine that drives the agent's execution. It orchestrates
//! the entire lifecycle of an agent run, from receiving user input to generating
//! a final response. The runner implements the core logic for the agent loop,
//! handling tool calls, handoffs, and guardrails.
//!
//! ## The Agent Loop
//!
//! The runner's primary responsibility is to manage the interaction loop with the
//! Language Model (LLM). This loop consists of the following steps:
//!
//! 1.  **Input Processing**: The user's input is processed, and any configured
//!     [`InputGuardrail`]s are applied to validate the input.
//! 2.  **Message History**: The conversation history is loaded from the
//!     [`Session`], if one is provided.
//! 3.  **LLM Interaction**: The agent's system message, along with the message
//!     history, is sent to the LLM to get a response.
//! 4.  **Response Handling**: The LLM's response is processed. If it contains
//!     tool calls, the corresponding tools are executed. If it's a handoff,
//!     control is transferred to another agent.
//! 5.  **Output Validation**: Any configured [`OutputGuardrail`]s are applied to
//!     validate the agent's final output.
//! 6.  **State Management**: The new messages are saved to the session to
//!     maintain the conversation's state.
//!
//! The loop continues until the agent produces a final response, a tool indicates
//! the run is complete, or the maximum number of turns is reached.
//!
//! [`InputGuardrail`]: crate::guardrail::InputGuardrail
//! [`OutputGuardrail`]: crate::guardrail::OutputGuardrail
//! [`Session`]: crate::memory::Session

use std::any::Any;
use std::sync::{Arc, Mutex};
use tracing::{debug, info, warn};

use crate::agent::Agent;
use crate::context::ContextDecision;
use crate::context::ContextualAgent;
use crate::error::{AgentsError, Result};
use crate::guardrail::GuardrailRunner;
use crate::items::{
    HandoffItem, Message, MessageItem, Role, RunItem, ToolCallItem, ToolOutputItem,
};
use crate::memory::Session;
use crate::model::ModelProvider;
use crate::result::{RunResult, StreamEvent, StreamingRunResult};
use crate::tracing::{AgentSpan, GenerationSpan, ToolSpan, TracingContext};
use crate::usage::UsageStats;

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

    /// Optional run-scoped context handler that applies across all agents in this run.
    ///
    /// When set via [`RunConfig::with_run_context`], the handler will receive every
    /// tool output (or error) across the entire run, including after handoffs.
    /// Its decisions are applied before any per-agent context handler.
    pub run_context: Option<crate::context::RunContextSpec>,
}

impl std::fmt::Debug for RunConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunConfig")
            .field("max_turns", &self.max_turns)
            .field("stream", &self.stream)
            .field("session", &self.session.is_some())
            .field("model_provider", &self.model_provider.is_some())
            .field("run_context", &self.run_context.is_some())
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
            run_context: None,
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
    pub fn with_run_context<C, H, F>(mut self, factory_fn: F, handler: H) -> Self
    where
        C: Send + Sync + 'static,
        H: crate::context::ToolContext<C> + Send + Sync + 'static,
        F: Fn() -> C + Send + Sync + 'static,
    {
        let factory = std::sync::Arc::new(move || Box::new(factory_fn()) as Box<dyn Any + Send>);
        let erased: std::sync::Arc<dyn crate::context::ErasedToolContextHandler> =
            std::sync::Arc::new(crate::context::TypedHandler::<C, H>::new(handler));
        self.run_context = Some(crate::context::RunContextSpec {
            factory,
            handler: erased,
        });
        self
    }

    /// Convenience: set a model provider.
    pub fn with_model_provider(mut self, provider: Option<Arc<dyn ModelProvider>>) -> Self {
        self.model_provider = provider;
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

    /// Executes a typed contextual agent and returns the result along with the final context value.
    pub async fn run_with_context<C>(
        agent: ContextualAgent<C>,
        input: impl Into<String>,
        config: RunConfig,
    ) -> Result<crate::result::RunResultWithContext<C>>
    where
        C: Send + Sync + 'static,
    {
        let input = input.into();
        let agent = agent.into_inner();
        info!(agent = %agent.name(), "Starting agent run (typed context)");

        let context = Arc::new(Mutex::new(TracingContext::new()));
        let _trace_id = context.lock().unwrap().trace_id().to_string();

        if !agent.config.input_guardrails.is_empty() {
            GuardrailRunner::check_input(&agent.config.input_guardrails, &input).await?;
        }

        let mut messages = vec![agent.build_system_message()];
        if let Some(session) = &config.session {
            let history = session.get_messages(None).await?;
            messages.extend(history);
        }
        messages.push(Message::user(input));

        // Run loop and capture final context state
        let (result, state, _run_state) =
            Self::run_loop(agent.clone(), messages, config.clone(), context.clone()).await?;

        if let Some(session) = &config.session {
            session.add_items(result.items.clone()).await?;
        }

        let state =
            state.ok_or_else(|| AgentsError::Other("No contextual state present".to_string()))?;
        let typed = state
            .downcast::<C>()
            .map_err(|_| AgentsError::Other("Context type mismatch".to_string()))?;
        Ok(crate::result::RunResultWithContext {
            result,
            context: *typed,
        })
    }

    /// Executes an agent with a run-scoped context and returns the result along with the final run context value.
    ///
    /// This is a convenience around attaching a run-scoped handler to the
    /// [`RunConfig`] and invoking the regular runner. It ensures the final
    /// typed context is only available once the run completes.
    pub async fn run_with_run_context<C>(
        agent: Agent,
        input: impl Into<String>,
        mut config: RunConfig,
        factory: impl Fn() -> C + Send + Sync + 'static,
        handler: impl crate::context::ToolContext<C> + 'static,
    ) -> Result<crate::result::RunResultWithContext<C>>
    where
        C: Send + Sync + 'static,
    {
        // Attach run-scoped context
        config = config.with_run_context(factory, handler);

        let input = input.into();
        info!(agent = %agent.name(), "Starting agent run (run-scoped context)");

        let context = Arc::new(Mutex::new(TracingContext::new()));
        let _trace_id = context.lock().unwrap().trace_id().to_string();

        if !agent.config.input_guardrails.is_empty() {
            GuardrailRunner::check_input(&agent.config.input_guardrails, &input).await?;
        }

        let mut messages = vec![agent.build_system_message()];
        if let Some(session) = &config.session {
            let history = session.get_messages(None).await?;
            messages.extend(history);
        }
        messages.push(Message::user(input));

        // Run loop and capture final run-scoped context state
        let (result, _state, run_state) =
            Self::run_loop(agent.clone(), messages, config.clone(), context.clone()).await?;

        if let Some(session) = &config.session {
            session.add_items(result.items.clone()).await?;
        }

        let state = run_state
            .ok_or_else(|| AgentsError::Other("No run-scoped context present".to_string()))?;
        let typed = state
            .downcast::<C>()
            .map_err(|_| AgentsError::Other("Run context type mismatch".to_string()))?;
        Ok(crate::result::RunResultWithContext {
            result,
            context: *typed,
        })
    }

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

        // Initialize per-run contextual state if configured (per-agent)
        let mut contextual_state: Option<Box<dyn Any + Send>> = agent
            .config
            .tool_context
            .as_ref()
            .map(|spec| (spec.factory)());

        // Initialize run-scoped context if configured (applies across handoffs)
        let mut run_scoped_state: Option<Box<dyn Any + Send>> =
            config.run_context.as_ref().map(|spec| (spec.factory)());

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
            for tool_call in &response.tool_calls {
                items.push(RunItem::ToolCall(ToolCallItem {
                    id: tool_call.id.clone(),
                    tool_name: tool_call.name.clone(),
                    arguments: tool_call.arguments.clone(),
                    created_at: chrono::Utc::now(),
                }));

                // Check if this is a handoff
                if let Some(handoff) = agent.handoffs().iter().find(|h| h.name == tool_call.name) {
                    info!(from = %agent.name(), to = %handoff.name, "Handoff detected");

                    items.push(RunItem::Handoff(HandoffItem {
                        id: uuid::Uuid::new_v4().to_string(),
                        from_agent: agent.name().to_string(),
                        to_agent: handoff.name.clone(),
                        reason: None,
                        created_at: chrono::Utc::now(),
                    }));

                    // IMPORTANT: respond to the tool_call with a tool message so the provider protocol remains valid
                    let handoff_ack = serde_json::json!({
                        "handoff": handoff.name,
                        "ack": true
                    });
                    messages.push(Message::tool(handoff_ack.to_string(), &tool_call.id));
                    items.push(RunItem::ToolOutput(ToolOutputItem {
                        id: uuid::Uuid::new_v4().to_string(),
                        tool_call_id: tool_call.id.clone(),
                        output: handoff_ack,
                        error: None,
                        created_at: chrono::Utc::now(),
                    }));

                    debug!(
                        target: "runner::messages",
                        "\n↳ Appended handoff TOOL reply (tool_call_id={})\n{}\n---",
                        tool_call.id,
                        format_messages_for_log(&messages)
                    );

                    // Switch to the new agent
                    agent = handoff.agent().clone();

                    break; // Exit the tool call loop to start a new turn
                }

                // Execute regular tool
                if let Some(tool) = agent.tools().iter().find(|t| t.name() == tool_call.name) {
                    let tool_span = ToolSpan::new(
                        context.clone(),
                        tool_call.name.clone(),
                        tool_call.arguments.clone(),
                    );

                    match tool.execute(tool_call.arguments.clone()).await {
                        Ok(result) => {
                            tool_span.success();

                            // Prepare original success output
                            let finalizing = result.is_final;
                            let mut output_value = result.output.clone();
                            let mut error_value: Option<String> = result.error.clone();

                            // First apply run-scoped handler if present
                            if let (Some(spec), Some(state)) =
                                (config.run_context.as_ref(), run_scoped_state.take())
                            {
                                let input_to_handler = if let Some(err_msg) = &error_value {
                                    Err(err_msg.clone())
                                } else {
                                    Ok(output_value.clone())
                                };
                                match spec.handler.on_tool_output(
                                    state,
                                    &tool_call.name,
                                    &tool_call.arguments,
                                    input_to_handler,
                                ) {
                                    Ok((new_state, decision)) => {
                                        run_scoped_state = Some(new_state);
                                        match decision {
                                            ContextDecision::Forward => {}
                                            ContextDecision::Rewrite(v) => {
                                                output_value = v;
                                                error_value = None;
                                            }
                                            ContextDecision::Final(v) => {
                                                agent_span.complete();
                                                let trace_id =
                                                    context.lock().unwrap().trace_id().to_string();
                                                return Ok((
                                                    RunResult::success(
                                                        v,
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
                                    Err(err) => {
                                        warn!(error = %err, "Run-scoped context handler failed on success output");
                                        run_scoped_state = config
                                            .run_context
                                            .as_ref()
                                            .map(|spec| (spec.factory)());
                                    }
                                }
                            }

                            // Apply contextual handler if present
                            if let (Some(spec), Some(state)) =
                                (agent.config.tool_context.as_ref(), contextual_state.take())
                            {
                                let input_to_handler = if let Some(err_msg) = &error_value {
                                    Err(err_msg.clone())
                                } else {
                                    Ok(output_value.clone())
                                };
                                let decision_res = spec.handler.on_tool_output(
                                    state,
                                    &tool_call.name,
                                    &tool_call.arguments,
                                    input_to_handler,
                                );

                                match decision_res {
                                    Ok((new_state, decision)) => {
                                        contextual_state = Some(new_state);
                                        match decision {
                                            ContextDecision::Forward => {}
                                            ContextDecision::Rewrite(v) => {
                                                output_value = v;
                                                if error_value.is_some() {
                                                    error_value = None;
                                                }
                                            }
                                            ContextDecision::Final(v) => {
                                                agent_span.complete();
                                                let trace_id =
                                                    context.lock().unwrap().trace_id().to_string();
                                                return Ok((
                                                    RunResult::success(
                                                        v,
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
                                    Err(err) => {
                                        // If handler fails, log and fall back to forward behavior
                                        warn!(error = %err, "Context handler failed on success output");
                                        contextual_state = agent
                                            .config
                                            .tool_context
                                            .as_ref()
                                            .map(|spec| (spec.factory)());
                                    }
                                }
                            }

                            // Record message and item with possibly rewritten output
                            messages.push(Message::tool(output_value.to_string(), &tool_call.id));
                            items.push(RunItem::ToolOutput(ToolOutputItem {
                                id: uuid::Uuid::new_v4().to_string(),
                                tool_call_id: tool_call.id.clone(),
                                output: output_value.clone(),
                                error: error_value,
                                created_at: chrono::Utc::now(),
                            }));

                            debug!(
                                target: "runner::messages",
                                "\n↳ Appended TOOL reply (tool_call_id={})\n{}\n---",
                                tool_call.id,
                                format_messages_for_log(&messages)
                            );

                            if finalizing {
                                agent_span.complete();

                                let trace_id = context.lock().unwrap().trace_id().to_string();
                                return Ok((
                                    RunResult::success(
                                        output_value,
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
                            tool_span.error(e.to_string());
                            warn!(error = %e, "Tool execution failed");

                            // Run-scoped handler can rewrite or finalize errors first
                            let mut handled_by_run = false;
                            if let (Some(spec), Some(state)) =
                                (config.run_context.as_ref(), run_scoped_state.take())
                            {
                                match spec.handler.on_tool_output(
                                    state,
                                    &tool_call.name,
                                    &tool_call.arguments,
                                    Err(e.to_string()),
                                ) {
                                    Ok((new_state, decision)) => {
                                        run_scoped_state = Some(new_state);
                                        match decision {
                                            ContextDecision::Forward => {}
                                            ContextDecision::Rewrite(v) => {
                                                messages.push(Message::tool(
                                                    v.to_string(),
                                                    &tool_call.id,
                                                ));
                                                items.push(RunItem::ToolOutput(ToolOutputItem {
                                                    id: uuid::Uuid::new_v4().to_string(),
                                                    tool_call_id: tool_call.id.clone(),
                                                    output: v,
                                                    error: None,
                                                    created_at: chrono::Utc::now(),
                                                }));
                                                debug!(
                                                    target: "runner::messages",
                                                    "\n↳ Appended TOOL reply (run-scoped rewrite) id={}\n{}\n---",
                                                    tool_call.id,
                                                    format_messages_for_log(&messages)
                                                );
                                                handled_by_run = true;
                                            }
                                            ContextDecision::Final(v) => {
                                                agent_span.complete();
                                                let trace_id =
                                                    context.lock().unwrap().trace_id().to_string();
                                                return Ok((
                                                    RunResult::success(
                                                        v,
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
                                    Err(err) => {
                                        warn!(error = %err, "Run-scoped context handler failed on error output");
                                        run_scoped_state = config
                                            .run_context
                                            .as_ref()
                                            .map(|spec| (spec.factory)());
                                    }
                                }
                            }

                            // Give the per-agent context handler a chance to rewrite or finalize errors
                            let mut handled = false;
                            if let (Some(spec), Some(state)) =
                                (agent.config.tool_context.as_ref(), contextual_state.take())
                            {
                                let decision_res = spec.handler.on_tool_output(
                                    state,
                                    &tool_call.name,
                                    &tool_call.arguments,
                                    Err(e.to_string()),
                                );
                                match decision_res {
                                    Ok((new_state, decision)) => {
                                        contextual_state = Some(new_state);
                                        match decision {
                                            ContextDecision::Forward => {
                                                // fall through to default error behavior
                                            }
                                            ContextDecision::Rewrite(v) => {
                                                // Treat as success rewrite
                                                messages.push(Message::tool(
                                                    v.to_string(),
                                                    &tool_call.id,
                                                ));
                                                items.push(RunItem::ToolOutput(ToolOutputItem {
                                                    id: uuid::Uuid::new_v4().to_string(),
                                                    tool_call_id: tool_call.id.clone(),
                                                    output: v,
                                                    error: None,
                                                    created_at: chrono::Utc::now(),
                                                }));
                                                debug!(
                                                    target: "runner::messages",
                                                    "\n↳ Appended TOOL reply (agent rewrite) id={}\n{}\n---",
                                                    tool_call.id,
                                                    format_messages_for_log(&messages)
                                                );
                                                handled = true;
                                            }
                                            ContextDecision::Final(v) => {
                                                agent_span.complete();
                                                let trace_id =
                                                    context.lock().unwrap().trace_id().to_string();
                                                return Ok((
                                                    RunResult::success(
                                                        v,
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
                                    Err(err) => {
                                        warn!(error = %err, "Context handler failed on error output");
                                        contextual_state = agent
                                            .config
                                            .tool_context
                                            .as_ref()
                                            .map(|spec| (spec.factory)());
                                    }
                                }
                            }

                            if !handled && !handled_by_run {
                                messages
                                    .push(Message::tool(format!("Error: {}", e), &tool_call.id));

                                items.push(RunItem::ToolOutput(ToolOutputItem {
                                    id: uuid::Uuid::new_v4().to_string(),
                                    tool_call_id: tool_call.id.clone(),
                                    output: serde_json::Value::Null,
                                    error: Some(e.to_string()),
                                    created_at: chrono::Utc::now(),
                                }));
                                debug!(
                                    target: "runner::messages",
                                    "\n↳ Appended TOOL error reply id={}\n{}\n---",
                                    tool_call.id,
                                    format_messages_for_log(&messages)
                                );
                            }
                        }
                    }
                } else {
                    warn!(tool = %tool_call.name, "Unknown tool called");

                    messages.push(Message::tool(
                        format!("Error: Unknown tool '{}'", tool_call.name),
                        &tool_call.id,
                    ));
                    debug!(
                        target: "runner::messages",
                        "\n↳ Appended TOOL unknown reply id={}\n{}\n---",
                        tool_call.id,
                        format_messages_for_log(&messages)
                    );
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
    use crate::context::{ContextStep, ToolContext};
    use crate::items::ToolCall as MsgToolCall;
    use crate::model::MockProvider;
    use crate::tool::FunctionTool;
    use std::sync::Mutex;

    #[tokio::test]
    async fn test_simple_run() {
        let agent = Agent::simple("TestAgent", "You are a test agent");

        let provider =
            Arc::new(MockProvider::new("test-model").with_message("Hello! How can I help you?"));

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
