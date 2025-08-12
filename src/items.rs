//! Items representing messages, tool calls, and model responses
//!
//! This module defines the core data structures for agent communication.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

/// Role in a conversation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// A message in the conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: content.into(),
            name: None,
            tool_call_id: None,
            tool_calls: None,
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
            name: None,
            tool_call_id: None,
            tool_calls: None,
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
            name: None,
            tool_call_id: None,
            tool_calls: None,
        }
    }

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

/// A tool call made by the model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

/// Response from the model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelResponse {
    pub id: String,
    pub content: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub finish_reason: Option<String>,
    pub created_at: DateTime<Utc>,
}

impl ModelResponse {
    pub fn new_message(content: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            content: Some(content.into()),
            tool_calls: vec![],
            finish_reason: Some("stop".to_string()),
            created_at: Utc::now(),
        }
    }

    pub fn new_tool_calls(tool_calls: Vec<ToolCall>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            content: None,
            tool_calls,
            finish_reason: Some("tool_calls".to_string()),
            created_at: Utc::now(),
        }
    }

    pub fn has_tool_calls(&self) -> bool {
        !self.tool_calls.is_empty()
    }

    pub fn has_content(&self) -> bool {
        self.content.is_some() && !self.content.as_ref().unwrap().is_empty()
    }
}

/// A run item representing a single step in the agent execution
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum RunItem {
    Message(MessageItem),
    ToolCall(ToolCallItem),
    ToolOutput(ToolOutputItem),
    Handoff(HandoffItem),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageItem {
    pub id: String,
    pub role: Role,
    pub content: String,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallItem {
    pub id: String,
    pub tool_name: String,
    pub arguments: Value,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolOutputItem {
    pub id: String,
    pub tool_call_id: String,
    pub output: Value,
    pub error: Option<String>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandoffItem {
    pub id: String,
    pub from_agent: String,
    pub to_agent: String,
    pub reason: Option<String>,
    pub created_at: DateTime<Utc>,
}

/// Helper functions for working with items
pub struct ItemHelpers;

impl ItemHelpers {
    /// Convert run items to messages for the conversation history
    pub fn to_messages(items: &[RunItem]) -> Vec<Message> {
        let mut messages = Vec::new();

        for item in items {
            match item {
                RunItem::Message(msg) => {
                    messages.push(Message {
                        role: msg.role,
                        content: msg.content.clone(),
                        name: None,
                        tool_call_id: None,
                        tool_calls: None,
                    });
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

        messages
    }

    /// Filter items by type
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
        assert_eq!(messages.len(), 2); // Message and ToolOutput
        assert_eq!(messages[0].role, Role::User);
        assert_eq!(messages[1].role, Role::Tool);
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
