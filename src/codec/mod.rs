//! Bijection codec between RunItem and raw OpenAI chat messages
//!
//! What this module provides (spec)
//! - Deterministic, lossless conversions used by recording/replay and persistence layers
//! - Pure utility functions decoupled from services
//!
//! Exports
//! - Models (reuse from `items.rs` if desired)
//!   - `RunItem::{Message, ToolCall, ToolOutput, Handoff}`
//! - Utils (pure)
//!   - `messages_to_items(messages: &[RawChatMessage]) -> Vec<RunItem>`
//!   - `items_to_messages(items: &[RunItem]) -> Vec<RawChatMessage>`
//!   - `CodecError` for invalid sequences (e.g., tool output without prior tool_call)
//!
//! Implementation strategy
//! - One-pass state machine for `messages_to_items`:
//!   - Accumulate assistant `tool_calls` to attach to the preceding assistant message
//!   - Immediately emit `ToolOutput` for `tool` role messages with resolved `tool_call_id`
//! - The inverse reconstructs messages, ensuring assistant tool_calls coalesce correctly
//! - Avoid allocations by pre-sizing vectors and reusing buffers
//!
//! Composition
//! - Recorder and session layers call these functions at the edge
//! - Does not depend on Tower; purely functional
//!
//! Testing strategy
//! - Property tests (e.g., with `proptest`) asserting round-trip identity
//! - Golden-case unit tests for:
//!   - Multiple tool_calls in a single assistant message
//!   - Interleaved ToolCall/ToolOutput
//!   - Empty content and error outputs
//! - Fuzz invalid sequences to ensure `CodecError` surfaces clearly

use async_openai::types::{
    ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestAssistantMessageContent,
    ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageArgs,
    ChatCompletionRequestSystemMessageContent, ChatCompletionRequestToolMessageArgs,
    ChatCompletionRequestToolMessageContent, ChatCompletionRequestUserMessageArgs,
    ChatCompletionRequestUserMessageContent, ChatCompletionToolType,
};
use serde_json::Value;

use crate::items::{HandoffItem, MessageItem, Role, RunItem, ToolCallItem, ToolOutputItem};

use chrono::Utc;
use uuid::Uuid;

#[derive(thiserror::Error, Debug)]
pub enum CodecError {
    #[error("invalid tool output without tool_call_id")]
    MissingToolCallId,
}

/// Convert raw OpenAI request messages to a sequence of RunItems.
pub fn messages_to_items(
    messages: &[ChatCompletionRequestMessage],
) -> Result<Vec<RunItem>, CodecError> {
    let mut items: Vec<RunItem> = Vec::with_capacity(messages.len());
    for m in messages {
        match m {
            ChatCompletionRequestMessage::System(s) => {
                items.push(RunItem::Message(MessageItem {
                    id: Uuid::new_v4().to_string(),
                    role: Role::System,
                    content: match &s.content {
                        ChatCompletionRequestSystemMessageContent::Text(t) => t.clone(),
                        _ => String::new(),
                    },
                    created_at: Utc::now(),
                }));
            }
            ChatCompletionRequestMessage::User(u) => {
                items.push(RunItem::Message(MessageItem {
                    id: Uuid::new_v4().to_string(),
                    role: Role::User,
                    content: match &u.content {
                        ChatCompletionRequestUserMessageContent::Text(t) => t.clone(),
                        _ => String::new(),
                    },
                    created_at: Utc::now(),
                }));
            }
            ChatCompletionRequestMessage::Assistant(a) => {
                // Always emit assistant message (content may be empty)
                items.push(RunItem::Message(MessageItem {
                    id: Uuid::new_v4().to_string(),
                    role: Role::Assistant,
                    content: if let Some(ChatCompletionRequestAssistantMessageContent::Text(t)) =
                        &a.content
                    {
                        t.clone()
                    } else {
                        String::new()
                    },
                    created_at: Utc::now(),
                }));
                if let Some(tool_calls) = &a.tool_calls {
                    for tc in tool_calls {
                        // tc.type must be function
                        if tc.r#type == ChatCompletionToolType::Function {
                            let args: Value = serde_json::from_str(&tc.function.arguments)
                                .unwrap_or(Value::String(tc.function.arguments.clone()));
                            items.push(RunItem::ToolCall(ToolCallItem {
                                id: tc.id.clone(),
                                tool_name: tc.function.name.clone(),
                                arguments: args,
                                created_at: Utc::now(),
                            }));
                        }
                    }
                }
            }
            ChatCompletionRequestMessage::Tool(t) => {
                let tcid = t.tool_call_id.clone();
                if tcid.is_empty() {
                    return Err(CodecError::MissingToolCallId);
                }
                // Attempt JSON parse, fallback to raw string
                let content_str = match &t.content {
                    ChatCompletionRequestToolMessageContent::Text(s) => s.clone(),
                    _ => String::new(),
                };
                let output = serde_json::from_str::<Value>(&content_str)
                    .unwrap_or(Value::String(content_str));
                items.push(RunItem::ToolOutput(ToolOutputItem {
                    id: Uuid::new_v4().to_string(),
                    tool_call_id: tcid,
                    output,
                    error: None,
                    created_at: Utc::now(),
                }));
            }
            ChatCompletionRequestMessage::Developer(_)
            | ChatCompletionRequestMessage::Function(_) => {
                // Not used in our normal flow; treat as system message string if needed
                // For now, map to a system message placeholder
                items.push(RunItem::Message(MessageItem {
                    id: Uuid::new_v4().to_string(),
                    role: Role::System,
                    content: String::new(),
                    created_at: Utc::now(),
                }));
            }
        }
    }
    Ok(items)
}

/// Convert RunItems back to raw OpenAI request messages, preserving tool_calls semantics.
pub fn items_to_messages(items: &[RunItem]) -> Vec<ChatCompletionRequestMessage> {
    let mut messages: Vec<ChatCompletionRequestMessage> = Vec::new();

    let mut i = 0;
    while i < items.len() {
        match &items[i] {
            RunItem::Message(msg) => {
                match msg.role {
                    Role::System => {
                        let sys = ChatCompletionRequestSystemMessageArgs::default()
                            .content(msg.content.clone())
                            .build()
                            .expect("sys build");
                        messages.push(sys.into());
                    }
                    Role::User => {
                        let usr = ChatCompletionRequestUserMessageArgs::default()
                            .content(msg.content.clone())
                            .build()
                            .expect("user build");
                        messages.push(usr.into());
                    }
                    Role::Assistant => {
                        // Look ahead to see if there are tool calls following this assistant message
                        let mut j = i + 1;
                        let mut tool_calls = Vec::new();
                        while j < items.len() {
                            if let RunItem::ToolCall(tc) = &items[j] {
                                tool_calls.push(
                                    async_openai::types::ChatCompletionMessageToolCall {
                                        id: tc.id.clone(),
                                        r#type: ChatCompletionToolType::Function,
                                        function: async_openai::types::FunctionCall {
                                            name: tc.tool_name.clone(),
                                            arguments: tc.arguments.to_string(),
                                        },
                                    },
                                );
                                j += 1;
                            } else {
                                break;
                            }
                        }

                        // Build assistant message with or without tool calls
                        let mut builder = ChatCompletionRequestAssistantMessageArgs::default();
                        builder.content(msg.content.clone());
                        if !tool_calls.is_empty() {
                            builder.tool_calls(tool_calls);
                            // Skip the tool call items we just processed
                            i = j - 1;
                        }
                        let assistant = builder.build().expect("assistant build");
                        messages.push(assistant.into());
                    }
                    Role::Tool => {
                        // Should not happen as tool outputs are ToolOutput items; ignore
                    }
                }
            }
            RunItem::ToolCall(_tc) => {
                // Tool calls should have been handled when processing the preceding assistant message
                // If we get here, it means there's a tool call without a preceding assistant message
                // This shouldn't happen in well-formed data, but we'll skip it
            }
            RunItem::ToolOutput(out) => {
                let content = out
                    .error
                    .as_ref()
                    .map(|e| format!("Error: {}", e))
                    .unwrap_or_else(|| out.output.to_string());
                let tool_msg = ChatCompletionRequestToolMessageArgs::default()
                    .content(content)
                    .tool_call_id(out.tool_call_id.clone())
                    .build()
                    .expect("tool build");
                messages.push(tool_msg.into());
            }
            RunItem::Handoff(HandoffItem { .. }) => {
                // Agent-only; not part of the bijection to raw messages
            }
        }
        i += 1;
    }

    messages
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_openai::types::ChatCompletionRequestMessage as ReqMsg;
    

    fn assistant_with_calls(name: &str, args: Value, id: &str) -> ReqMsg {
        let tc = async_openai::types::ChatCompletionMessageToolCall {
            id: id.to_string(),
            r#type: ChatCompletionToolType::Function,
            function: async_openai::types::FunctionCall {
                name: name.to_string(),
                arguments: args.to_string(),
            },
        };
        ChatCompletionRequestAssistantMessageArgs::default()
            .content("")
            .tool_calls(vec![tc])
            .build()
            .unwrap()
            .into()
    }

    #[test]
    fn maps_assistant_tool_calls_to_items() {
        let user = ChatCompletionRequestUserMessageArgs::default()
            .content("hi")
            .build()
            .unwrap()
            .into();
        let asst = assistant_with_calls("calc", serde_json::json!({"a":1}), "call_1");
        let tool = ChatCompletionRequestToolMessageArgs::default()
            .content("{\"sum\":2}")
            .tool_call_id("call_1")
            .build()
            .unwrap()
            .into();
        let items = messages_to_items(&[user, asst, tool]).unwrap();
        assert!(matches!(items[0], RunItem::Message(_)));
        assert!(matches!(items[1], RunItem::Message(_))); // assistant message
        assert!(matches!(items[2], RunItem::ToolCall(_)));
        assert!(matches!(items[3], RunItem::ToolOutput(_)));
    }

    #[test]
    fn roundtrip_messages_identity_basic() {
        let sys = ChatCompletionRequestSystemMessageArgs::default()
            .content("sys")
            .build()
            .unwrap()
            .into();
        let usr = ChatCompletionRequestUserMessageArgs::default()
            .content("hello")
            .build()
            .unwrap()
            .into();
        let asst = ChatCompletionRequestAssistantMessageArgs::default()
            .content("ok")
            .build()
            .unwrap()
            .into();
        let orig = vec![sys, usr, asst];
        let items = messages_to_items(&orig).unwrap();
        let back = items_to_messages(&items);
        assert_eq!(back.len(), orig.len());
        // Verify content is preserved
        if let ChatCompletionRequestMessage::System(s2) = &back[0] {
            if let ChatCompletionRequestSystemMessageContent::Text(t) = &s2.content {
                assert_eq!(t, "sys");
            } else {
                panic!("expected system text");
            }
        } else {
            panic!("expected system message");
        }
    }

    #[test]
    fn preserves_assistant_text_content() {
        let asst = ChatCompletionRequestAssistantMessageArgs::default()
            .content("assistant says ok")
            .build()
            .unwrap()
            .into();
        let items = messages_to_items(&[asst]).unwrap();
        // Expect first item is assistant message with same content
        match &items[0] {
            RunItem::Message(m) => {
                assert_eq!(m.role, Role::Assistant);
                assert_eq!(m.content, "assistant says ok");
            }
            _ => panic!("expected message item"),
        }
    }

    #[test]
    fn tool_output_json_roundtrip_value() {
        // Assistant calls a tool, tool returns JSON content
        let asst = assistant_with_calls("calc", serde_json::json!({"a":1}), "call_x");
        let tool = ChatCompletionRequestToolMessageArgs::default()
            .content("{\"sum\": 3, \"ok\": true}")
            .tool_call_id("call_x")
            .build()
            .unwrap()
            .into();

        let items = messages_to_items(&[asst, tool]).unwrap();
        // Last item should be ToolOutput with parsed JSON
        match &items[1] {
            RunItem::ToolCall(_) => {}
            _ => panic!("expected tool call at index 1"),
        }
        match &items[2] {
            RunItem::ToolOutput(out) => {
                let expected = serde_json::json!({"sum":3, "ok": true});
                assert_eq!(out.output, expected);
            }
            _ => panic!("expected tool output at index 2"),
        }

        // Convert back to messages and validate content parses to same value
        let back = items_to_messages(&items);
        // Find the tool message
        let tool_msg = back
            .iter()
            .find(|m| matches!(m, ChatCompletionRequestMessage::Tool(_)))
            .expect("tool msg present");
        if let ChatCompletionRequestMessage::Tool(t) = tool_msg {
            if let ChatCompletionRequestToolMessageContent::Text(txt) = &t.content {
                let val: Value = serde_json::from_str(txt).unwrap();
                assert_eq!(val, serde_json::json!({"sum":3, "ok": true}));
            } else {
                panic!("expected text content for tool message");
            }
        }
    }

    #[test]
    fn tool_output_plain_string_roundtrip_value() {
        // Assistant calls a tool, tool returns plain string content
        let asst = assistant_with_calls("echo", serde_json::json!({"v":1}), "c1");
        let tool = ChatCompletionRequestToolMessageArgs::default()
            .content("hello world")
            .tool_call_id("c1")
            .build()
            .unwrap()
            .into();
        let items = messages_to_items(&[asst, tool]).unwrap();
        match &items[2] {
            RunItem::ToolOutput(out) => {
                assert_eq!(out.output, Value::String("hello world".to_string()));
            }
            _ => panic!("expected tool output at index 2"),
        }

        // Back to messages: content will be JSON-escaped string; ensure it parses to same value
        let back = items_to_messages(&items);
        let tool_msg = back
            .iter()
            .find(|m| matches!(m, ChatCompletionRequestMessage::Tool(_)))
            .expect("tool msg present");
        if let ChatCompletionRequestMessage::Tool(t) = tool_msg {
            if let ChatCompletionRequestToolMessageContent::Text(txt) = &t.content {
                let parsed: Value = serde_json::from_str(txt).unwrap_or(Value::String(txt.clone()));
                assert_eq!(parsed, Value::String("hello world".to_string()));
            } else {
                panic!("expected text content");
            }
        }
    }

    #[test]
    fn missing_tool_call_id_is_error() {
        // Tool message without tool_call_id should error
        let tool = ChatCompletionRequestToolMessageArgs::default()
            .content("{\"x\":1}")
            .tool_call_id("")
            .build()
            .unwrap()
            .into();
        let err = messages_to_items(&[tool]).unwrap_err();
        match err {
            CodecError::MissingToolCallId => {}
        }
    }
}
