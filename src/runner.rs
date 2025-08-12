//! Runner for executing agents
//!
//! The runner implements the core agent loop, handling tool calls, handoffs, and guardrails.

use std::sync::{Arc, Mutex};
use tracing::{debug, info, warn};

use crate::agent::Agent;
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

/// Configuration for a run
#[derive(Clone)]
pub struct RunConfig {
    /// Maximum number of turns (LLM calls) before stopping
    pub max_turns: Option<usize>,

    /// Whether to stream events
    pub stream: bool,

    /// Session for conversation history
    pub session: Option<Arc<dyn Session>>,

    /// Model provider to use
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

/// The main runner for executing agents
pub struct Runner;

impl Runner {
    /// Run an agent asynchronously
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
        let result = Self::run_loop(agent, messages, config.clone(), context.clone()).await?;

        // Save to session if configured
        if let Some(session) = &config.session {
            session.add_items(result.items.clone()).await?;
        }

        Ok(result)
    }

    /// Run an agent synchronously (blocking)
    pub fn run_sync(
        agent: Agent,
        input: impl Into<String>,
        config: RunConfig,
    ) -> Result<RunResult> {
        let runtime = tokio::runtime::Runtime::new()?;
        runtime.block_on(Self::run(agent, input, config))
    }

    /// Run an agent with streaming
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

    /// The main agent loop
    async fn run_loop(
        mut agent: Agent,
        mut messages: Vec<Message>,
        config: RunConfig,
        context: Arc<Mutex<TracingContext>>,
    ) -> Result<RunResult> {
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
                        return Ok(RunResult::success(
                            serde_json::Value::String(final_content),
                            items,
                            agent.name().to_string(),
                            usage_stats,
                            trace_id,
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

                            messages.push(Message::tool(result.output.to_string(), &tool_call.id));

                            let output = result.output.clone();
                            items.push(RunItem::ToolOutput(ToolOutputItem {
                                id: uuid::Uuid::new_v4().to_string(),
                                tool_call_id: tool_call.id.clone(),
                                output: result.output,
                                error: result.error,
                                created_at: chrono::Utc::now(),
                            }));

                            if result.is_final {
                                agent_span.complete();

                                let trace_id = context.lock().unwrap().trace_id().to_string();
                                return Ok(RunResult::success(
                                    output,
                                    items,
                                    agent.name().to_string(),
                                    usage_stats,
                                    trace_id,
                                ));
                            }
                        }
                        Err(e) => {
                            tool_span.error(e.to_string());
                            warn!(error = %e, "Tool execution failed");

                            messages.push(Message::tool(format!("Error: {}", e), &tool_call.id));

                            items.push(RunItem::ToolOutput(ToolOutputItem {
                                id: uuid::Uuid::new_v4().to_string(),
                                tool_call_id: tool_call.id.clone(),
                                output: serde_json::Value::Null,
                                error: Some(e.to_string()),
                                created_at: chrono::Utc::now(),
                            }));
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
    use crate::model::MockProvider;
    use crate::tool::FunctionTool;

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
