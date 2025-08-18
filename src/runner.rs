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
use crate::error::{AgentsError, Result};
use crate::guardrail::GuardrailRunner;
use crate::items::{
    HandoffItem, Message, MessageItem, Role, RunItem, ToolCallItem, ToolOutputItem,
};
use crate::memory::Session;
use crate::model::ModelProvider;
use crate::result::{RunResult, StreamEvent, StreamingRunResult};
use crate::context::ContextualAgent;
use crate::tracing::{AgentSpan, GenerationSpan, ToolSpan, TracingContext};
use crate::usage::UsageStats;

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
        }
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
        let (result, _state) = Self::run_loop(agent, messages, config.clone(), context.clone()).await?;

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
        let (result, state) = Self::run_loop(agent.clone(), messages, config.clone(), context.clone()).await?;

        if let Some(session) = &config.session {
            session.add_items(result.items.clone()).await?;
        }

        let state = state.ok_or_else(|| AgentsError::Other("No contextual state present".to_string()))?;
        let typed = state
            .downcast::<C>()
            .map_err(|_| AgentsError::Other("Context type mismatch".to_string()))?;
        Ok(crate::result::RunResultWithContext { result, context: *typed })
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
    ) -> Result<(RunResult, Option<Box<dyn Any + Send>>)> {
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

        // Initialize per-run contextual state if configured
        let mut contextual_state: Option<Box<dyn Any + Send>> = agent
            .config
            .tool_context
            .as_ref()
            .map(|spec| (spec.factory)());

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

            let (response, usage) = model_provider
                .complete(
                    messages.clone(),
                    agent.config.tools.clone(),
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
                                ));
                            }
                        }
                        Err(e) => {
                            tool_span.error(e.to_string());
                            warn!(error = %e, "Tool execution failed");

                            // Give the context handler a chance to rewrite or finalize errors
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

                            if !handled {
                                messages
                                    .push(Message::tool(format!("Error: {}", e), &tool_call.id));

                                items.push(RunItem::ToolOutput(ToolOutputItem {
                                    id: uuid::Uuid::new_v4().to_string(),
                                    tool_call_id: tool_call.id.clone(),
                                    output: serde_json::Value::Null,
                                    error: Some(e.to_string()),
                                    created_at: chrono::Utc::now(),
                                }));
                            }
                        }
                    }
                } else {
                    warn!(tool = %tool_call.name, "Unknown tool called");

                    messages.push(Message::tool(
                        format!("Error: Unknown tool '{}'", tool_call.name),
                        &tool_call.id,
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
            .with_context_factory(|| Ctx::default(), ForwardHandler);

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
            .with_context_factory(|| Ctx::default(), RewriteHandler);

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
            .with_context_factory(|| Ctx::default(), FinalHandler);

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
            .with_context_factory(|| Ctx::default(), RewriteOnErrorHandler);

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
            .with_context_factory(|| Ctx::default(), ErrOnceCountingHandler::new());

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
            .with_context_factory(|| Ctx::default(), CountingHandler);

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
            .with_context_factory(|| Ctx::default(), CountingHandler);

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
