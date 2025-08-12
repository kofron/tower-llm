//! # Agent Execution Results
//!
//! This module defines the data structures that encapsulate the results of an
//! agent's execution. It provides support for both fully completed runs and
//! real-time streaming of events.
//!
//! - [`RunResult`]: A comprehensive summary of a completed agent run, including
//!   the final output, all generated items, and usage statistics.
//! - [`StreamingRunResult`]: A handle for an in-progress agent run that provides
//!   a stream of [`StreamEvent`]s, allowing for real-time processing of the
//!   agent's activities.
//! - [`StreamEvent`]: An enumeration of all possible events that can occur
//!   during an agent's execution, such as content generation, tool calls, and
//!   guardrail checks.
//!
//! ## Working with `RunResult`
//!
//! A `RunResult` is returned by the `Runner::run` and `Runner::run_sync` methods.
//! It contains all the information about the completed run.
//!
//! ```rust
//! use openai_agents_rs::result::RunResult;
//! use openai_agents_rs::items::RunItem;
//! use openai_agents_rs::usage::UsageStats;
//! use serde_json::json;
//!
//! // Create a successful result.
//! let result = RunResult::success(
//!     json!("This is the final output."),
//!     vec![], // A list of all items generated during the run.
//!     "EchoAgent".to_string(),
//!     UsageStats::new(),
//!     "trace_123".to_string(),
//! );
//!
//! assert!(result.is_success());
//! assert_eq!(result.final_agent, "EchoAgent");
//! println!("Summary: {}", result.summary());
//! ```
//!
//! ## Handling Streaming Results
//!
//! For real-time applications, `run_stream` returns a [`StreamingRunResult`],
//! which you can use to `collect` the final result.
//!
//! ```rust,no_run
//! use openai_agents_rs::{Agent, Runner, runner::RunConfig};
//! use futures::StreamExt;
//!
//! # async fn run_agent_stream() -> Result<(), Box<dyn std::error::Error>> {
//! let agent = Agent::simple("Streamer", "You stream responses.");
//! let mut streaming_result = Runner::run_stream(
//!     agent,
//!     "Tell me a story.",
//!     RunConfig::default(),
//! ).await?;
//!
//! // You can process events as they come in.
//! while let Some(event) = streaming_result.stream.next().await {
//!     // Process each event...
//! }
//! # Ok(())
//! # }
//! ```
//! Results from agent execution
//!
//! Provides both synchronous and streaming result types.

use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use std::pin::Pin;

use crate::error::Result;
use crate::items::{Message, RunItem};
use crate::usage::{Usage, UsageStats};

/// Represents the final result of a completed agent run.
///
/// This struct provides a comprehensive summary of the agent's execution,
/// including the final output, a complete history of all generated items,
/// usage statistics, and status information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunResult {
    /// The final output produced by the agent, as a `serde_json::Value`. This
    /// could be a simple string or a more complex JSON object.
    pub final_output: serde_json::Value,

    /// A vector containing all the `RunItem`s that were generated during the
    /// execution, in the order they occurred. This provides a detailed trace
    /// of the agent's actions.
    pub items: Vec<RunItem>,

    /// The sequence of messages that constitute the conversation history,
    /// suitable for being passed to the next run to maintain context.
    pub messages: Vec<Message>,

    /// The name of the agent that produced the final output. This is particularly
    /// relevant in multi-agent workflows where handoffs can occur.
    pub final_agent: String,

    /// A detailed breakdown of the token usage for the entire run, including
    /// both prompt and completion tokens.
    pub usage: UsageStats,

    /// A unique identifier for the run, useful for tracing and debugging.
    pub trace_id: String,

    /// A boolean flag indicating whether the run completed successfully.
    pub success: bool,

    /// If the run failed, this field contains a string describing the error.
    pub error: Option<String>,
}

impl RunResult {
    /// Creates a new `RunResult` for a successful run.
    pub fn success(
        final_output: serde_json::Value,
        items: Vec<RunItem>,
        final_agent: String,
        usage: UsageStats,
        trace_id: String,
    ) -> Self {
        let messages = crate::items::ItemHelpers::to_messages(&items);

        Self {
            final_output,
            items,
            messages,
            final_agent,
            usage,
            trace_id,
            success: true,
            error: None,
        }
    }

    /// Creates a new `RunResult` for a failed run.
    pub fn failure(error: String, trace_id: String) -> Self {
        Self {
            final_output: serde_json::Value::Null,
            items: vec![],
            messages: vec![],
            final_agent: String::new(),
            usage: UsageStats::new(),
            trace_id,
            success: false,
            error: Some(error),
        }
    }

    /// Returns `true` if the run completed successfully.
    pub fn is_success(&self) -> bool {
        self.success
    }

    /// Returns the error message if the run failed.
    pub fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }

    /// Returns the conversation history as a vector of `Message`s.
    pub fn to_input_messages(&self) -> Vec<Message> {
        self.messages.clone()
    }

    /// Provides a human-readable summary of the run.
    pub fn summary(&self) -> String {
        if self.success {
            format!(
                "Run completed successfully\n\
                 Final agent: {}\n\
                 Items generated: {}\n\
                 Total tokens: {}\n\
                 Trace ID: {}",
                self.final_agent,
                self.items.len(),
                self.usage.total.total_tokens,
                self.trace_id
            )
        } else {
            format!(
                "Run failed\n\
                 Error: {}\n\
                 Trace ID: {}",
                self.error.as_ref().unwrap_or(&"Unknown error".to_string()),
                self.trace_id
            )
        }
    }
}

/// An enumeration of all possible events that can be emitted during a
/// streaming agent run.
///
/// These events provide a real-time view into the agent's execution, allowing
/// for dynamic and responsive user interfaces.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum StreamEvent {
    /// Indicates that a new agent has been selected, either at the start of
    /// a run or due to a handoff.
    AgentSelected {
        agent_name: String,
        reason: Option<String>,
    },

    /// Fired when the model begins generating a response.
    GenerationStarted { model: String },

    /// A chunk of content has been generated by the model.
    ContentDelta { delta: String },

    /// The full content of the model's response has been generated.
    ContentComplete { content: String },

    /// A tool call has been initiated by the agent.
    ToolCallStarted {
        tool_name: String,
        tool_call_id: String,
    },

    /// A tool call has finished execution.
    ToolCallCompleted {
        tool_call_id: String,
        result: serde_json::Value,
    },

    /// A guardrail has been checked.
    GuardrailCheck {
        guardrail_name: String,
        passed: bool,
        reason: Option<String>,
    },

    /// A new `RunItem` has been generated and added to the history.
    ItemGenerated { item: RunItem },

    /// An update on the token usage.
    UsageUpdate { usage: Usage },

    /// The entire run has completed successfully. This is the final event.
    RunCompleted { result: RunResult },

    /// An error occurred during the run. This is a terminal event.
    Error { error: String },
}

/// A type alias for a pinned, boxed, dynamic stream of `StreamEvent`s.
pub type EventStream = Pin<Box<dyn Stream<Item = StreamEvent> + Send>>;

/// Represents the result of an agent run that is currently in progress.
///
/// This struct provides access to the `EventStream`, allowing you to process
/// events as they occur. It also contains the `trace_id` for the run.
pub struct StreamingRunResult {
    /// The stream of `StreamEvent`s from the agent's execution.
    pub stream: EventStream,

    /// The unique trace ID for this run.
    pub trace_id: String,
}

impl StreamingRunResult {
    /// Creates a new `StreamingRunResult`.
    pub fn new(stream: EventStream, trace_id: String) -> Self {
        Self { stream, trace_id }
    }

    /// Consumes the stream and collects all events into a final `RunResult`.
    ///
    /// This method is useful when you want to use the streaming API but only
    /// need the final result.
    pub async fn collect(mut self) -> Result<RunResult> {
        use futures::StreamExt;

        let mut items = Vec::new();
        let mut usage = UsageStats::new();
        let mut final_agent = String::new();
        let mut final_output = serde_json::Value::Null;
        let mut error = None;

        while let Some(event) = self.stream.next().await {
            match event {
                StreamEvent::ItemGenerated { item } => {
                    items.push(item);
                }
                StreamEvent::AgentSelected { agent_name, .. } => {
                    final_agent = agent_name;
                }
                StreamEvent::UsageUpdate { usage: u } => {
                    usage.total.add_usage(&u);
                }
                StreamEvent::RunCompleted { result } => {
                    return Ok(result);
                }
                StreamEvent::Error { error: e } => {
                    error = Some(e);
                    break;
                }
                StreamEvent::ContentComplete { content } => {
                    final_output = serde_json::Value::String(content);
                }
                _ => {}
            }
        }

        if let Some(e) = error {
            Ok(RunResult::failure(e, self.trace_id))
        } else {
            Ok(RunResult::success(
                final_output,
                items,
                final_agent,
                usage,
                self.trace_id,
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::items::{MessageItem, Role};
    use chrono::Utc;

    #[test]
    fn test_run_result_success() {
        let items = vec![
            RunItem::Message(MessageItem {
                id: "1".to_string(),
                role: Role::User,
                content: "Hello".to_string(),
                created_at: Utc::now(),
            }),
            RunItem::Message(MessageItem {
                id: "2".to_string(),
                role: Role::Assistant,
                content: "Hi!".to_string(),
                created_at: Utc::now(),
            }),
        ];

        let result = RunResult::success(
            serde_json::json!("Final output"),
            items.clone(),
            "TestAgent".to_string(),
            UsageStats::new(),
            "trace_123".to_string(),
        );

        assert!(result.is_success());
        assert_eq!(result.final_agent, "TestAgent");
        assert_eq!(result.items.len(), 2);
        assert_eq!(result.messages.len(), 2);
        assert!(result.error.is_none());
    }

    #[test]
    fn test_run_result_failure() {
        let result =
            RunResult::failure("Something went wrong".to_string(), "trace_456".to_string());

        assert!(!result.is_success());
        assert_eq!(result.error(), Some("Something went wrong"));
        assert!(result.items.is_empty());
        assert!(result.messages.is_empty());
    }

    #[test]
    fn test_run_result_summary() {
        let result = RunResult::success(
            serde_json::json!("Done"),
            vec![],
            "Agent".to_string(),
            UsageStats::new(),
            "trace_789".to_string(),
        );

        let summary = result.summary();
        assert!(summary.contains("completed successfully"));
        assert!(summary.contains("Agent"));
        assert!(summary.contains("trace_789"));
    }

    #[test]
    fn test_stream_event_serialization() {
        let event = StreamEvent::AgentSelected {
            agent_name: "TestAgent".to_string(),
            reason: Some("User request".to_string()),
        };

        let serialized = serde_json::to_string(&event).unwrap();
        assert!(serialized.contains("\"type\":\"AgentSelected\""));
        assert!(serialized.contains("TestAgent"));

        let deserialized: StreamEvent = serde_json::from_str(&serialized).unwrap();
        if let StreamEvent::AgentSelected { agent_name, .. } = deserialized {
            assert_eq!(agent_name, "TestAgent");
        } else {
            panic!("Wrong event type");
        }
    }

    #[test]
    fn test_various_stream_events() {
        let events = vec![
            StreamEvent::GenerationStarted {
                model: "gpt-4".to_string(),
            },
            StreamEvent::ContentDelta {
                delta: "Hello".to_string(),
            },
            StreamEvent::ContentComplete {
                content: "Hello, world!".to_string(),
            },
            StreamEvent::ToolCallStarted {
                tool_name: "calculator".to_string(),
                tool_call_id: "call_123".to_string(),
            },
            StreamEvent::ToolCallCompleted {
                tool_call_id: "call_123".to_string(),
                result: serde_json::json!({"result": 42}),
            },
            StreamEvent::GuardrailCheck {
                guardrail_name: "ContentFilter".to_string(),
                passed: true,
                reason: None,
            },
            StreamEvent::UsageUpdate {
                usage: Usage::new(100, 50),
            },
            StreamEvent::Error {
                error: "Test error".to_string(),
            },
        ];

        // Test that all events can be serialized and deserialized
        for event in events {
            let serialized = serde_json::to_string(&event).unwrap();
            let _deserialized: StreamEvent = serde_json::from_str(&serialized).unwrap();
        }
    }

    #[tokio::test]
    async fn test_streaming_result_collect() {
        use futures::stream;

        let events = vec![
            StreamEvent::AgentSelected {
                agent_name: "TestAgent".to_string(),
                reason: None,
            },
            StreamEvent::ContentComplete {
                content: "Response".to_string(),
            },
            StreamEvent::RunCompleted {
                result: RunResult::success(
                    serde_json::json!("Final"),
                    vec![],
                    "TestAgent".to_string(),
                    UsageStats::new(),
                    "trace_stream".to_string(),
                ),
            },
        ];

        let stream = Box::pin(stream::iter(events));
        let streaming_result = StreamingRunResult::new(stream, "trace_stream".to_string());

        let result = streaming_result.collect().await.unwrap();
        assert!(result.is_success());
        assert_eq!(result.final_agent, "TestAgent");
    }

    #[tokio::test]
    async fn test_streaming_result_with_error() {
        use futures::stream;

        let events = vec![
            StreamEvent::AgentSelected {
                agent_name: "TestAgent".to_string(),
                reason: None,
            },
            StreamEvent::Error {
                error: "Stream error".to_string(),
            },
        ];

        let stream = Box::pin(stream::iter(events));
        let streaming_result = StreamingRunResult::new(stream, "trace_error".to_string());

        let result = streaming_result.collect().await.unwrap();
        assert!(!result.is_success());
        assert_eq!(result.error(), Some("Stream error"));
    }
}
