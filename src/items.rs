//! # Core Data Structures for Agent Communication
//!
//! This module defines the fundamental data structures that represent the
//! various components of an agent's conversation and execution trace. These
//! "items" are used to construct the conversation history sent to the LLM,
//! to represent tool calls and their outputs, and to log the agent's actions.
//!
//! ## Key Data Structures
//!
//! - **[`Role`]**: An enum representing the speaker in a conversation (e.g.,
//!   `System`, `User`, `Assistant`).
//! - **[`Message`]**: The primary unit of conversation, containing the content
//!   of a message and the role of the speaker.
//! - **[`ToolCall`]**: Represents a request from the agent to execute a tool,
//!   including the tool's name and arguments.
//! - **[`ModelResponse`]**: Encapsulates the raw response from the LLM, which
//!   can include both text content and tool calls.
//! - **[`RunItem`]**: A comprehensive enum that represents a single step in the
//!   agent's execution trace. It can be a message, a tool call, a tool output,
//!   or a handoff.
//!
//! These structures are designed to be serializable, allowing them to be easily
//! stored for session management and logging.
//!
//! ### Example: Creating Different Message Types
//!
//! ```rust
//! use openai_agents_rs::items::{Message, Role};
//!
//! let system_message = Message::system("You are a helpful assistant.");
//! assert_eq!(system_message.role, Role::System);
//!
//! let user_message = Message::user("What is the weather like today?");
//! assert_eq!(user_message.role, Role::User);
//!
//! let assistant_message = Message::assistant("I can help with that.");
//! assert_eq!(assistant_message.role, Role::Assistant);
//! ```
//! Items representing messages, tool calls, and model responses
//!
//! This module defines the core data structures for agent communication.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

/// Represents the role of a message's author in a conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// The system message, which sets the context and instructions for the agent.
    System,
    /// A message from the end-user.
    User,
    /// A message from the AI assistant.
    Assistant,
    /// A message containing the output of a tool.
    Tool,
}

/// A single message in a conversation, forming the basis of interaction with
/// the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// The role of the message author.
    pub role: Role,
    /// The text content of the message.
    pub content: String,
    /// The name of the author, used in some multi-agent contexts.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// A unique identifier for the tool call this message is a response to.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// A list of tool calls requested by the assistant.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

impl Message {
    /// Creates a new `Message` with the `System` role.
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: content.into(),
            name: None,
            tool_call_id: None,
            tool_calls: None,
        }
    }

    /// Creates a new `Message` with the `User` role.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
            name: None,
            tool_call_id: None,
            tool_calls: None,
        }
    }

    /// Creates a new `Message` with the `Assistant` role.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
            name: None,
            tool_call_id: None,
            tool_calls: None,
        }
    }

    /// Creates a new `Message` from the assistant that includes tool calls.
    pub fn assistant_with_tool_calls(
        content: impl Into<String>,
        tool_calls: Vec<ToolCall>,
    ) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
            name: None,
            tool_call_id: None,
            tool_calls: Some(tool_calls),
        }
    }

    /// Creates a new `Message` with the `Tool` role, representing a tool's output.
    pub fn tool(content: impl Into<String>, tool_call_id: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: content.into(),
            name: None,
            tool_call_id: Some(tool_call_id.into()),
            tool_calls: None,
        }
    }
}

/// Represents a request from the LLM to call a specific tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// A unique identifier for this tool call.
    pub id: String,
    /// The name of the tool to be executed.
    pub name: String,
    /// The arguments to be passed to the tool, as a JSON value.
    pub arguments: Value,
}

/// Encapsulates a response from the LLM, which may include text content and
/// tool calls.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelResponse {
    /// A unique identifier for the response.
    pub id: String,
    /// The text content of the response, if any.
    pub content: Option<String>,
    /// A list of tool calls requested by the model.
    pub tool_calls: Vec<ToolCall>,
    /// The reason why the model stopped generating the response.
    pub finish_reason: Option<String>,
    /// The timestamp of when the response was created.
    pub created_at: DateTime<Utc>,
}

impl ModelResponse {
    /// Creates a new `ModelResponse` that contains only a text message.
    pub fn new_message(content: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            content: Some(content.into()),
            tool_calls: vec![],
            finish_reason: Some("stop".to_string()),
            created_at: Utc::now(),
        }
    }

    /// Creates a new `ModelResponse` that contains one or more tool calls.
    pub fn new_tool_calls(tool_calls: Vec<ToolCall>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            content: None,
            tool_calls,
            finish_reason: Some("tool_calls".to_string()),
            created_at: Utc::now(),
        }
    }

    /// Returns `true` if the response contains any tool calls.
    pub fn has_tool_calls(&self) -> bool {
        !self.tool_calls.is_empty()
    }

    /// Returns `true` if the response has non-empty text content.
    pub fn has_content(&self) -> bool {
        self.content.is_some() && !self.content.as_ref().unwrap().is_empty()
    }
}

/// An enum representing a single, discrete step in an agent's execution trace.
///
/// `RunItem` is used to log the history of a conversation, which can then be
/// stored in a session for maintaining context.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum RunItem {
    /// A message from the user or the assistant.
    Message(MessageItem),
    /// A request from the agent to call a tool.
    ToolCall(ToolCallItem),
    /// The output of a tool execution.
    ToolOutput(ToolOutputItem),
    /// A handoff of control from one agent to another.
    Handoff(HandoffItem),
}

/// A structured representation of a message within a `RunItem`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageItem {
    pub id: String,
    pub role: Role,
    pub content: String,
    pub created_at: DateTime<Utc>,
}

/// A structured representation of a tool call within a `RunItem`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallItem {
    pub id: String,
    pub tool_name: String,
    pub arguments: Value,
    pub created_at: DateTime<Utc>,
}

/// A structured representation of a tool's output within a `RunItem`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolOutputItem {
    pub id: String,
    pub tool_call_id: String,
    pub output: Value,
    pub error: Option<String>,
    pub created_at: DateTime<Utc>,
}

/// A structured representation of a handoff within a `RunItem`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandoffItem {
    pub id: String,
    pub from_agent: String,
    pub to_agent: String,
    pub reason: Option<String>,
    pub created_at: DateTime<Utc>,
}

/// A collection of helper functions for working with `RunItem`s.
pub struct ItemHelpers;

impl ItemHelpers {
    /// Converts a slice of `RunItem`s into a `Vec<Message>` suitable for use as
    /// conversation history.
    pub fn to_messages(items: &[RunItem]) -> Vec<Message> {
        let mut messages = Vec::new();
        let mut pending_tool_calls: Vec<ToolCall> = Vec::new();

        for (i, item) in items.iter().enumerate() {
            match item {
                RunItem::Message(msg) => {
                    // If we have pending tool calls from before this message,
                    // and this is an assistant message, attach them
                    if msg.role == Role::Assistant && !pending_tool_calls.is_empty() {
                        messages.push(Message {
                            role: msg.role,
                            content: msg.content.clone(),
                            name: None,
                            tool_call_id: None,
                            tool_calls: Some(pending_tool_calls.clone()),
                        });
                        pending_tool_calls.clear();
                    } else {
                        messages.push(Message {
                            role: msg.role,
                            content: msg.content.clone(),
                            name: None,
                            tool_call_id: None,
                            tool_calls: None,
                        });
                    }
                }
                RunItem::ToolCall(tool_call) => {
                    // Collect tool calls that should be attached to the previous assistant message
                    pending_tool_calls.push(ToolCall {
                        id: tool_call.id.clone(),
                        name: tool_call.tool_name.clone(),
                        arguments: tool_call.arguments.clone(),
                    });

                    // Check if this is the last tool call before tool outputs
                    // If the next item is a ToolOutput, we should create an assistant message now
                    if i + 1 < items.len() {
                        if let RunItem::ToolOutput(_) = &items[i + 1] {
                            // Create an assistant message with the tool calls
                            if !pending_tool_calls.is_empty() {
                                messages.push(Message {
                                    role: Role::Assistant,
                                    content: String::new(),
                                    name: None,
                                    tool_call_id: None,
                                    tool_calls: Some(pending_tool_calls.clone()),
                                });
                                pending_tool_calls.clear();
                            }
                        }
                    }
                }
                RunItem::ToolOutput(output) => {
                    let content = if let Some(error) = &output.error {
                        format!("Error: {}", error)
                    } else {
                        output.output.to_string()
                    };
                    messages.push(Message::tool(content, &output.tool_call_id));
                }
                _ => {}
            }
        }

        // If we still have pending tool calls at the end, create an assistant message for them
        if !pending_tool_calls.is_empty() {
            messages.push(Message {
                role: Role::Assistant,
                content: String::new(),
                name: None,
                tool_call_id: None,
                tool_calls: Some(pending_tool_calls),
            });
        }

        messages
    }

    /// Filters a slice of `RunItem`s by their type.
    pub fn filter_by_type<T>(
        items: &[RunItem],
        filter_fn: impl Fn(&RunItem) -> Option<&T>,
    ) -> Vec<&T> {
        items.iter().filter_map(filter_fn).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_message_creation() {
        let sys_msg = Message::system("You are a helpful assistant");
        assert_eq!(sys_msg.role, Role::System);
        assert_eq!(sys_msg.content, "You are a helpful assistant");
        assert!(sys_msg.tool_call_id.is_none());

        let user_msg = Message::user("Hello");
        assert_eq!(user_msg.role, Role::User);
        assert_eq!(user_msg.content, "Hello");

        let tool_msg = Message::tool("Result", "call_123");
        assert_eq!(tool_msg.role, Role::Tool);
        assert_eq!(tool_msg.tool_call_id, Some("call_123".to_string()));
    }

    #[test]
    fn test_model_response() {
        let response = ModelResponse::new_message("Hello, how can I help?");
        assert!(response.has_content());
        assert!(!response.has_tool_calls());
        assert_eq!(response.content, Some("Hello, how can I help?".to_string()));

        let tool_call = ToolCall {
            id: "call_1".to_string(),
            name: "get_weather".to_string(),
            arguments: serde_json::json!({"city": "Tokyo"}),
        };

        let tool_response = ModelResponse::new_tool_calls(vec![tool_call]);
        assert!(!tool_response.has_content());
        assert!(tool_response.has_tool_calls());
        assert_eq!(tool_response.tool_calls.len(), 1);
    }

    #[test]
    fn test_run_items() {
        let msg_item = RunItem::Message(MessageItem {
            id: "msg_1".to_string(),
            role: Role::User,
            content: "Hello".to_string(),
            created_at: Utc::now(),
        });

        let tool_item = RunItem::ToolCall(ToolCallItem {
            id: "call_1".to_string(),
            tool_name: "calculator".to_string(),
            arguments: serde_json::json!({"operation": "add", "a": 1, "b": 2}),
            created_at: Utc::now(),
        });

        // Test serialization
        let serialized = serde_json::to_string(&msg_item).unwrap();
        assert!(serialized.contains("\"type\":\"Message\""));

        let serialized_tool = serde_json::to_string(&tool_item).unwrap();
        assert!(serialized_tool.contains("\"type\":\"ToolCall\""));
    }

    #[test]
    fn test_item_helpers_to_messages() {
        let items = vec![
            RunItem::Message(MessageItem {
                id: "1".to_string(),
                role: Role::User,
                content: "What's the weather?".to_string(),
                created_at: Utc::now(),
            }),
            RunItem::ToolCall(ToolCallItem {
                id: "2".to_string(),
                tool_name: "get_weather".to_string(),
                arguments: serde_json::json!({"city": "Paris"}),
                created_at: Utc::now(),
            }),
            RunItem::ToolOutput(ToolOutputItem {
                id: "3".to_string(),
                tool_call_id: "2".to_string(),
                output: serde_json::json!({"temp": 20, "condition": "sunny"}),
                error: None,
                created_at: Utc::now(),
            }),
        ];

        let messages = ItemHelpers::to_messages(&items);
        assert_eq!(messages.len(), 3); // User Message, Assistant with tool_calls, and ToolOutput
        assert_eq!(messages[0].role, Role::User);
        assert_eq!(messages[1].role, Role::Assistant); // Assistant message with tool_calls
        assert!(messages[1].tool_calls.is_some());
        assert_eq!(messages[2].role, Role::Tool);
    }

    #[test]
    fn test_handoff_item() {
        let handoff = HandoffItem {
            id: "handoff_1".to_string(),
            from_agent: "triage".to_string(),
            to_agent: "specialist".to_string(),
            reason: Some("User needs technical support".to_string()),
            created_at: Utc::now(),
        };

        let item = RunItem::Handoff(handoff.clone());
        let serialized = serde_json::to_string(&item).unwrap();
        assert!(serialized.contains("\"type\":\"Handoff\""));
        assert!(serialized.contains("\"from_agent\":\"triage\""));
    }

    #[test]
    fn test_role_serialization() {
        let role = Role::Assistant;
        let serialized = serde_json::to_string(&role).unwrap();
        assert_eq!(serialized, "\"assistant\"");

        let deserialized: Role = serde_json::from_str("\"system\"").unwrap();
        assert_eq!(deserialized, Role::System);
    }
}
