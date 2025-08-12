//! Results from agent execution
//!
//! Provides both synchronous and streaming result types.

use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use std::pin::Pin;

use crate::error::Result;
use crate::items::{Message, RunItem};
use crate::usage::{Usage, UsageStats};

/// Result from a completed agent run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunResult {
    /// The final output from the agent
    pub final_output: serde_json::Value,

    /// All items generated during the run
    pub items: Vec<RunItem>,

    /// Messages that can be used for conversation history
    pub messages: Vec<Message>,

    /// The agent that produced the final output
    pub final_agent: String,

    /// Total usage statistics
    pub usage: UsageStats,

    /// Trace ID for debugging
    pub trace_id: String,

    /// Whether the run completed successfully
    pub success: bool,

    /// Error message if the run failed
    pub error: Option<String>,
}

impl RunResult {
    /// Create a successful result
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

    /// Create a failed result
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

    /// Check if the run was successful
    pub fn is_success(&self) -> bool {
        self.success
    }

    /// Get the error message if the run failed
    pub fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }

    /// Convert to input messages for the next run
    pub fn to_input_messages(&self) -> Vec<Message> {
        self.messages.clone()
    }

    /// Get a summary of the run
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

/// Event emitted during streaming execution
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum StreamEvent {
    /// Agent has been selected/changed
    AgentSelected {
        agent_name: String,
        reason: Option<String>,
    },

    /// Model started generating
    GenerationStarted { model: String },

    /// Partial content from the model
    ContentDelta { delta: String },

    /// Complete content from the model
    ContentComplete { content: String },

    /// Tool call initiated
    ToolCallStarted {
        tool_name: String,
        tool_call_id: String,
    },

    /// Tool call completed
    ToolCallCompleted {
        tool_call_id: String,
        result: serde_json::Value,
    },

    /// Guardrail check
    GuardrailCheck {
        guardrail_name: String,
        passed: bool,
        reason: Option<String>,
    },

    /// Run item generated
    ItemGenerated { item: RunItem },

    /// Usage update
    UsageUpdate { usage: Usage },

    /// Run completed
    RunCompleted { result: RunResult },

    /// Error occurred
    Error { error: String },
}

/// Stream of events from a running agent
pub type EventStream = Pin<Box<dyn Stream<Item = StreamEvent> + Send>>;

/// Result from a streaming agent run
pub struct StreamingRunResult {
    /// Stream of events
    pub stream: EventStream,

    /// Trace ID for this run
    pub trace_id: String,
}

impl StreamingRunResult {
    /// Create a new streaming result
    pub fn new(stream: EventStream, trace_id: String) -> Self {
        Self { stream, trace_id }
    }

    /// Collect all events and return the final result
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
