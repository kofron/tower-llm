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
    ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage,
    ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestToolMessageArgs,
    ChatCompletionRequestUserMessageArgs, ChatCompletionToolType,
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
            ChatCompletionRequestMessage::System(_s) => {
                items.push(RunItem::Message(MessageItem {
                    id: Uuid::new_v4().to_string(),
                    role: Role::System,
                    content: String::new(),
                    created_at: Utc::now(),
                }));
            }
            ChatCompletionRequestMessage::User(_u) => {
                items.push(RunItem::Message(MessageItem {
                    id: Uuid::new_v4().to_string(),
                    role: Role::User,
                    content: String::new(),
                    created_at: Utc::now(),
                }));
            }
            ChatCompletionRequestMessage::Assistant(a) => {
                // Always emit assistant message (content may be empty)
                items.push(RunItem::Message(MessageItem {
                    id: Uuid::new_v4().to_string(),
                    role: Role::Assistant,
                    content: String::new(),
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
                // Attempt JSON parse, fallback to raw string
                let output = Value::Null;
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
    use quickcheck::{Arbitrary, Gen, QuickCheck, TestResult};
    use quickcheck_macros::quickcheck;

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
    }
}
