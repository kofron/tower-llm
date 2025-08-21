//! Conversation validation utilities (pure, test-focused).
//!
//! What this module provides
//! - `validate_conversation` to detect ordering and tool-call consistency violations
//! - `ValidationPolicy` to configure strictness per test or example
//! - `ViolationCode` and `Violation` to describe issues precisely for assertions
//!
//! Invariants validated (when enabled by policy)
//! - Role sequencing: first non-system is user; no assistant before first user; optional repeated-role rejection
//! - Tool-calls: assistant tool_calls must be followed by tool messages with matching ids, before the next assistant
//! - Tool outputs: unknown, duplicate, and out-of-order tool responses detected
//! - Structure: system-not-first; unsupported message kinds (developer/function) when disallowed
//! - Extras: duplicate/empty tool_call ids; empty tool message ids; tool-before-assistant; contiguity of tool responses; require at least one user
//!
//! Policies
//! - `allow_system_anywhere`, `require_user_first`, `allow_repeated_roles`
//! - `enforce_tool_response_order`, `allow_unknown_tool_response`, `allow_duplicate_tool_response`
//! - `allow_developer_and_function`, `enforce_contiguous_tool_responses`, `require_user_present`, `allow_dangling_tool_calls`
//!
//! Quick start (tests/examples)
//! ```rust
//! use tower_llm::validation::{validate_conversation, ValidationPolicy};
//! use async_openai::types::*;
//!
//! let sys = ChatCompletionRequestSystemMessageArgs::default().content("sys").build().unwrap();
//! let usr = ChatCompletionRequestUserMessageArgs::default().content("hi").build().unwrap();
//! let asst = ChatCompletionRequestAssistantMessageArgs::default().content("ok").build().unwrap();
//! let msgs = vec![sys.into(), usr.into(), asst.into()];
//! assert!(validate_conversation(&msgs, &ValidationPolicy::default()).is_none());
//! ```
//!
//! This module is self-contained and has no side effects. It is intended for
//! tests and examples, but can be used in layers to assert correctness post-transform.

use async_openai::types::ChatCompletionRequestMessage as ReqMsg;

pub mod gen;
pub mod mutate;

/// Configuration controlling which rules are enforced.
#[derive(Debug, Clone)]
pub struct ValidationPolicy {
    /// Allow `system` messages to appear anywhere (not only before first non-system).
    pub allow_system_anywhere: bool,
    /// Require the first non-system message to be `user`.
    pub require_user_first: bool,
    /// Allow repeated adjacent roles (no error on runs like user→user).
    pub allow_repeated_roles: bool,
    /// Enforce that tool responses follow the exact declared order of tool_calls.
    pub enforce_tool_response_order: bool,
    /// Allow tool responses with unknown tool_call_id (no matching assistant tool_call).
    pub allow_unknown_tool_response: bool,
    /// Allow multiple tool responses for the same tool_call_id.
    pub allow_duplicate_tool_response: bool,
    /// Allow Developer/Function message kinds.
    pub allow_developer_and_function: bool,
    /// Enforce that tool responses are immediately contiguous after the assistant.
    pub enforce_contiguous_tool_responses: bool,
    /// Require at least one user message to be present in the conversation.
    pub require_user_present: bool,
    /// Allow assistant tool_calls to be dangling (missing corresponding tool responses) at boundaries/end.
    pub allow_dangling_tool_calls: bool,
}

impl Default for ValidationPolicy {
    fn default() -> Self {
        Self {
            allow_system_anywhere: false,
            require_user_first: true,
            allow_repeated_roles: false,
            enforce_tool_response_order: true,
            allow_unknown_tool_response: false,
            allow_duplicate_tool_response: false,
            allow_developer_and_function: false,
            enforce_contiguous_tool_responses: false,
            require_user_present: false,
            allow_dangling_tool_calls: false,
        }
    }
}

/// Well-defined codes for violations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ViolationCode {
    AssistantBeforeUser {
        assistant_index: usize,
    },
    RepeatedRole {
        role: String,
        start_index: usize,
        count: usize,
    },
    MissingToolResponses {
        assistant_index: usize,
        missing_ids: Vec<String>,
    },
    UnknownToolResponse {
        tool_index: usize,
        tool_call_id: String,
    },
    DuplicateToolResponse {
        tool_call_id: String,
        indices: Vec<usize>,
    },
    ToolResponsesOutOfOrder {
        assistant_index: usize,
        expected: Vec<String>,
        observed: Vec<String>,
    },
    SystemNotFirst {
        system_index: usize,
    },
    UnsupportedMessageType {
        index: usize,
        kind: String,
    },
    DuplicateToolCallIdsInAssistant {
        assistant_index: usize,
        duplicate_ids: Vec<String>,
    },
    EmptyToolCallIdInAssistant {
        assistant_index: usize,
        positions: Vec<usize>,
    },
    EmptyToolMessageId {
        tool_index: usize,
    },
    ToolBeforeAssistant {
        tool_index: usize,
    },
    ToolResponsesNotContiguous {
        assistant_index: usize,
        interrupt_index: usize,
    },
    NoUserMessage,
}

/// A single violation with human-readable message and structured data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Violation {
    pub code: ViolationCode,
    pub message: String,
}

impl Violation {
    pub fn new(code: ViolationCode) -> Self {
        let message = match &code {
            ViolationCode::AssistantBeforeUser { assistant_index } => {
                format!(
                    "assistant appears before first user at index {}",
                    assistant_index
                )
            }
            ViolationCode::RepeatedRole {
                role,
                start_index,
                count,
            } => {
                format!(
                    "role '{}' repeats starting at index {} (count = {})",
                    role, start_index, count
                )
            }
            ViolationCode::MissingToolResponses {
                assistant_index,
                missing_ids,
            } => {
                format!(
                    "assistant with tool_calls at index {} missing responses for ids: {}",
                    assistant_index,
                    missing_ids.join(", ")
                )
            }
            ViolationCode::UnknownToolResponse {
                tool_index,
                tool_call_id,
            } => {
                format!(
                    "tool response at index {} references unknown id '{}'",
                    tool_index, tool_call_id
                )
            }
            ViolationCode::DuplicateToolResponse {
                tool_call_id,
                indices,
            } => {
                format!(
                    "duplicate tool responses for id '{}' at indices {:?}",
                    tool_call_id, indices
                )
            }
            ViolationCode::ToolResponsesOutOfOrder {
                assistant_index,
                expected,
                observed,
            } => {
                format!(
                    "tool responses out of order after assistant index {} (expected {:?}, observed {:?})",
                    assistant_index, expected, observed
                )
            }
            ViolationCode::SystemNotFirst { system_index } => {
                format!(
                    "system message appears after convo start at index {}",
                    system_index
                )
            }
            ViolationCode::UnsupportedMessageType { index, kind } => {
                format!("unsupported message kind '{}' at index {}", kind, index)
            }
            ViolationCode::DuplicateToolCallIdsInAssistant {
                assistant_index,
                duplicate_ids,
            } => {
                format!(
                    "assistant at index {} has duplicate tool_call ids: {:?}",
                    assistant_index, duplicate_ids
                )
            }
            ViolationCode::EmptyToolCallIdInAssistant {
                assistant_index,
                positions,
            } => {
                format!(
                    "assistant at index {} has empty tool_call id(s) at positions {:?}",
                    assistant_index, positions
                )
            }
            ViolationCode::EmptyToolMessageId { tool_index } => {
                format!(
                    "tool message at index {} has empty tool_call_id",
                    tool_index
                )
            }
            ViolationCode::ToolBeforeAssistant { tool_index } => {
                format!(
                    "tool message at index {} occurs before any assistant tool_calls",
                    tool_index
                )
            }
            ViolationCode::ToolResponsesNotContiguous {
                assistant_index,
                interrupt_index,
            } => {
                format!(
                    "non-tool message at index {} interrupted contiguous tool responses after assistant index {}",
                    interrupt_index, assistant_index
                )
            }
            ViolationCode::NoUserMessage => "conversation contains no user message".to_string(),
        };
        Self { code, message }
    }
}

/// Validate a conversation and return violations, if any.
/// Returns None when no violations are found.
pub fn validate_conversation(
    messages: &[ReqMsg],
    policy: &ValidationPolicy,
) -> Option<Vec<Violation>> {
    let mut violations: Vec<Violation> = Vec::new();

    // 1) Baseline role checks
    let mut first_non_system_seen = false;
    let mut first_user_seen = false;
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum R {
        System,
        User,
        Assistant,
        Tool,
        Developer,
        Function,
    }
    let mut last_role: Option<R> = None;
    for (i, m) in messages.iter().enumerate() {
        // System not first
        if !policy.allow_system_anywhere && first_non_system_seen {
            if let ReqMsg::System(_) = m {
                violations.push(Violation::new(ViolationCode::SystemNotFirst {
                    system_index: i,
                }));
            }
        }

        // Detect first_non_system and user gating
        match m {
            ReqMsg::System(_) => {}
            _ => {
                if !first_non_system_seen {
                    first_non_system_seen = true;
                    if policy.require_user_first
                        && !matches!(m, ReqMsg::User(_))
                        && matches!(m, ReqMsg::Assistant(_))
                    {
                        violations.push(Violation::new(ViolationCode::AssistantBeforeUser {
                            assistant_index: i,
                        }));
                    }
                }
            }
        }
        if matches!(m, ReqMsg::User(_)) {
            first_user_seen = true;
        }

        // Repeated roles
        if !policy.allow_repeated_roles {
            let current_role = match m {
                ReqMsg::System(_) => R::System,
                ReqMsg::User(_) => R::User,
                ReqMsg::Assistant(_) => R::Assistant,
                ReqMsg::Tool(_) => R::Tool,
                ReqMsg::Developer(_) => R::Developer,
                ReqMsg::Function(_) => R::Function,
            };
            if let Some(prev) = last_role {
                if prev == current_role {
                    // Count the run length starting at i-1
                    let mut count = 2usize; // includes (i-1) and i
                    let mut j = i + 1;
                    while j < messages.len() {
                        let rj = match &messages[j] {
                            ReqMsg::System(_) => R::System,
                            ReqMsg::User(_) => R::User,
                            ReqMsg::Assistant(_) => R::Assistant,
                            ReqMsg::Tool(_) => R::Tool,
                            ReqMsg::Developer(_) => R::Developer,
                            ReqMsg::Function(_) => R::Function,
                        };
                        if rj == current_role {
                            count += 1;
                            j += 1;
                        } else {
                            break;
                        }
                    }
                    let role_str = match current_role {
                        R::System => "system",
                        R::User => "user",
                        R::Assistant => "assistant",
                        R::Tool => "tool",
                        R::Developer => "developer",
                        R::Function => "function",
                    };
                    violations.push(Violation::new(ViolationCode::RepeatedRole {
                        role: role_str.to_string(),
                        start_index: i - 1,
                        count,
                    }));
                }
            }
            last_role = Some(current_role);
        }
    }

    // Special cases after baseline scan
    if policy.require_user_present && !first_user_seen {
        violations.push(Violation::new(ViolationCode::NoUserMessage));
    }

    // 2) Tool-call invariants, scan with state
    use std::collections::{HashMap, HashSet};
    #[derive(Default)]
    struct Expected {
        assistant_index: usize,
        expected_ids: Vec<String>,
        seen_order: Vec<String>,
        seen_counts: HashMap<String, Vec<usize>>, // id -> indices where seen
        contiguity_broken: bool,
    }
    let mut current: Option<Expected> = None;

    let mut i = 0usize;
    while i < messages.len() {
        match &messages[i] {
            ReqMsg::Assistant(a) => {
                // On assistant boundary, finalize previous expectations and check contiguity if broken by assistant
                if let Some(mut exp) = current.take() {
                    if policy.enforce_contiguous_tool_responses && !exp.contiguity_broken {
                        let pending = exp.expected_ids.len() - exp.seen_counts.keys().count();
                        if pending > 0 {
                            violations.push(Violation::new(
                                ViolationCode::ToolResponsesNotContiguous {
                                    assistant_index: exp.assistant_index,
                                    interrupt_index: i,
                                },
                            ));
                            exp.contiguity_broken = true;
                        }
                    }
                    // Missing ids
                    let expected_set: HashSet<&String> = exp.expected_ids.iter().collect();
                    let observed_set: HashSet<&String> = exp.seen_order.iter().collect();
                    let missing: Vec<String> = expected_set
                        .difference(&observed_set)
                        .cloned()
                        .cloned()
                        .collect();
                    if !missing.is_empty() && !policy.allow_dangling_tool_calls {
                        violations.push(Violation::new(ViolationCode::MissingToolResponses {
                            assistant_index: exp.assistant_index,
                            missing_ids: missing,
                        }));
                    }
                    // Order check
                    if policy.enforce_tool_response_order && !exp.seen_order.is_empty() {
                        // Compare sequence of first occurrences to the prefix of expected_ids of same length
                        let expected_prefix: Vec<String> = exp
                            .expected_ids
                            .iter()
                            .take(exp.seen_order.len())
                            .cloned()
                            .collect();
                        if exp.seen_order != expected_prefix {
                            violations.push(Violation::new(
                                ViolationCode::ToolResponsesOutOfOrder {
                                    assistant_index: exp.assistant_index,
                                    expected: expected_prefix,
                                    observed: exp.seen_order,
                                },
                            ));
                        }
                    }
                }

                // Validate assistant tool_calls for duplicates and empty ids
                if let Some(tool_calls) = &a.tool_calls {
                    if !tool_calls.is_empty() {
                        use std::collections::HashSet;
                        let mut seen: HashSet<&str> = HashSet::new();
                        let mut dups: Vec<String> = Vec::new();
                        let mut empties: Vec<usize> = Vec::new();
                        for (k, tc) in tool_calls.iter().enumerate() {
                            if tc.id.is_empty() {
                                empties.push(k);
                            }
                            if !tc.id.is_empty() && !seen.insert(tc.id.as_str()) {
                                dups.push(tc.id.clone());
                            }
                        }
                        if !dups.is_empty() {
                            violations.push(Violation::new(
                                ViolationCode::DuplicateToolCallIdsInAssistant {
                                    assistant_index: i,
                                    duplicate_ids: dups,
                                },
                            ));
                        }
                        if !empties.is_empty() {
                            violations.push(Violation::new(
                                ViolationCode::EmptyToolCallIdInAssistant {
                                    assistant_index: i,
                                    positions: empties,
                                },
                            ));
                        }
                    }
                }

                // Start new expectations (if any tool_calls)
                if let Some(tool_calls) = &a.tool_calls {
                    if !tool_calls.is_empty() {
                        let expected_ids: Vec<String> =
                            tool_calls.iter().map(|tc| tc.id.clone()).collect();
                        current = Some(Expected {
                            assistant_index: i,
                            expected_ids,
                            seen_order: Vec::new(),
                            seen_counts: HashMap::new(),
                            contiguity_broken: false,
                        });
                    }
                }
            }
            ReqMsg::Tool(t) => {
                let id = t.tool_call_id.clone();
                if id.is_empty() {
                    violations.push(Violation::new(ViolationCode::EmptyToolMessageId {
                        tool_index: i,
                    }));
                }
                if let Some(exp) = current.as_mut() {
                    let known = exp.expected_ids.iter().any(|x| x == &id);
                    if !known {
                        if !policy.allow_unknown_tool_response {
                            violations.push(Violation::new(ViolationCode::UnknownToolResponse {
                                tool_index: i,
                                tool_call_id: id.clone(),
                            }));
                        }
                    } else {
                        let entry = exp.seen_counts.entry(id.clone()).or_default();
                        entry.push(i);
                        if entry.len() == 1 {
                            exp.seen_order.push(id.clone());
                        } else if !policy.allow_duplicate_tool_response {
                            violations.push(Violation::new(ViolationCode::DuplicateToolResponse {
                                tool_call_id: id.clone(),
                                indices: entry.clone(),
                            }));
                        }
                    }
                } else {
                    // No active expectation window
                    violations.push(Violation::new(ViolationCode::ToolBeforeAssistant {
                        tool_index: i,
                    }));
                }
            }
            ReqMsg::System(_) | ReqMsg::User(_) => {
                if let Some(exp) = current.as_mut() {
                    if policy.enforce_contiguous_tool_responses && !exp.contiguity_broken {
                        // pending responses remain?
                        let pending = exp.expected_ids.len() - exp.seen_counts.keys().count();
                        if pending > 0 {
                            violations.push(Violation::new(
                                ViolationCode::ToolResponsesNotContiguous {
                                    assistant_index: exp.assistant_index,
                                    interrupt_index: i,
                                },
                            ));
                            exp.contiguity_broken = true;
                        }
                    }
                }
            }
            ReqMsg::Developer(_) => {
                if !policy.allow_developer_and_function {
                    violations.push(Violation::new(ViolationCode::UnsupportedMessageType {
                        index: i,
                        kind: "developer".into(),
                    }));
                }
                if let Some(exp) = current.as_mut() {
                    if policy.enforce_contiguous_tool_responses && !exp.contiguity_broken {
                        let pending = exp.expected_ids.len() - exp.seen_counts.keys().count();
                        if pending > 0 {
                            violations.push(Violation::new(
                                ViolationCode::ToolResponsesNotContiguous {
                                    assistant_index: exp.assistant_index,
                                    interrupt_index: i,
                                },
                            ));
                            exp.contiguity_broken = true;
                        }
                    }
                }
            }
            ReqMsg::Function(_) => {
                if !policy.allow_developer_and_function {
                    violations.push(Violation::new(ViolationCode::UnsupportedMessageType {
                        index: i,
                        kind: "function".into(),
                    }));
                }
                if let Some(exp) = current.as_mut() {
                    if policy.enforce_contiguous_tool_responses && !exp.contiguity_broken {
                        let pending = exp.expected_ids.len() - exp.seen_counts.keys().count();
                        if pending > 0 {
                            violations.push(Violation::new(
                                ViolationCode::ToolResponsesNotContiguous {
                                    assistant_index: exp.assistant_index,
                                    interrupt_index: i,
                                },
                            ));
                            exp.contiguity_broken = true;
                        }
                    }
                }
            }
        }
        i += 1;
    }

    // Finalize at end-of-list
    if let Some(exp) = current.take() {
        use std::collections::HashSet;
        let expected_set: HashSet<&String> = exp.expected_ids.iter().collect();
        let observed_set: HashSet<&String> = exp.seen_order.iter().collect();
        let missing: Vec<String> = expected_set
            .difference(&observed_set)
            .cloned()
            .cloned()
            .collect();
        if !missing.is_empty() && !policy.allow_dangling_tool_calls {
            violations.push(Violation::new(ViolationCode::MissingToolResponses {
                assistant_index: exp.assistant_index,
                missing_ids: missing,
            }));
        }
        if policy.enforce_tool_response_order && !exp.seen_order.is_empty() {
            let expected_prefix: Vec<String> = exp
                .expected_ids
                .iter()
                .take(exp.seen_order.len())
                .cloned()
                .collect();
            if exp.seen_order != expected_prefix {
                violations.push(Violation::new(ViolationCode::ToolResponsesOutOfOrder {
                    assistant_index: exp.assistant_index,
                    expected: expected_prefix,
                    observed: exp.seen_order,
                }));
            }
        }
    }

    if violations.is_empty() {
        None
    } else {
        Some(violations)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_openai::types::*;

    fn req(messages: Vec<ChatCompletionRequestMessage>) -> Vec<ChatCompletionRequestMessage> {
        messages
    }

    #[test]
    fn valid_simple_conversation() {
        let sys = ChatCompletionRequestSystemMessageArgs::default()
            .content("sys")
            .build()
            .unwrap();
        let usr = ChatCompletionRequestUserMessageArgs::default()
            .content("hi")
            .build()
            .unwrap();
        let asst = ChatCompletionRequestAssistantMessageArgs::default()
            .content("ok")
            .build()
            .unwrap();
        let msgs = req(vec![sys.into(), usr.into(), asst.into()]);
        let out = validate_conversation(&msgs, &ValidationPolicy::default());
        assert!(out.is_none());
    }

    #[test]
    fn assistant_before_user_detected() {
        let asst = ChatCompletionRequestAssistantMessageArgs::default()
            .content("hi")
            .build()
            .unwrap();
        let usr = ChatCompletionRequestUserMessageArgs::default()
            .content("later")
            .build()
            .unwrap();
        let msgs = req(vec![asst.into(), usr.into()]);
        let out = validate_conversation(&msgs, &ValidationPolicy::default()).unwrap();
        assert!(out
            .iter()
            .any(|v| matches!(v.code, ViolationCode::AssistantBeforeUser { .. })));
    }

    #[test]
    fn repeated_role_detected() {
        let usr1 = ChatCompletionRequestUserMessageArgs::default()
            .content("a")
            .build()
            .unwrap();
        let usr2 = ChatCompletionRequestUserMessageArgs::default()
            .content("b")
            .build()
            .unwrap();
        let msgs = req(vec![usr1.into(), usr2.into()]);
        let out = validate_conversation(&msgs, &ValidationPolicy::default()).unwrap();
        assert!(out
            .iter()
            .any(|v| matches!(v.code, ViolationCode::RepeatedRole { .. })));
    }

    #[test]
    fn tool_calls_valid_sequence() {
        let tc = ChatCompletionMessageToolCall {
            id: "c1".to_string(),
            r#type: ChatCompletionToolType::Function,
            function: FunctionCall {
                name: "calc".into(),
                arguments: "{}".into(),
            },
        };
        let asst = ChatCompletionRequestAssistantMessageArgs::default()
            .content("")
            .tool_calls(vec![tc])
            .build()
            .unwrap();
        let tool = ChatCompletionRequestToolMessageArgs::default()
            .tool_call_id("c1")
            .content("{}")
            .build()
            .unwrap();
        let msgs = req(vec![
            ChatCompletionRequestMessage::Assistant(asst),
            ChatCompletionRequestMessage::Tool(tool),
        ]);
        let policy = ValidationPolicy {
            require_user_first: false,
            ..Default::default()
        };
        let out = validate_conversation(&msgs, &policy);
        assert!(out.is_none());
    }

    #[test]
    fn missing_tool_response_detected() {
        let tc = ChatCompletionMessageToolCall {
            id: "c1".to_string(),
            r#type: ChatCompletionToolType::Function,
            function: FunctionCall {
                name: "calc".into(),
                arguments: "{}".into(),
            },
        };
        let asst = ChatCompletionRequestAssistantMessageArgs::default()
            .content("")
            .tool_calls(vec![tc])
            .build()
            .unwrap();
        let msgs = req(vec![ChatCompletionRequestMessage::Assistant(asst)]);
        let out = validate_conversation(&msgs, &ValidationPolicy::default()).unwrap();
        assert!(out
            .iter()
            .any(|v| matches!(v.code, ViolationCode::MissingToolResponses { .. })));
    }

    #[test]
    fn missing_tool_response_allowed_by_policy() {
        let tc = ChatCompletionMessageToolCall {
            id: "c1".to_string(),
            r#type: ChatCompletionToolType::Function,
            function: FunctionCall {
                name: "calc".into(),
                arguments: "{}".into(),
            },
        };
        let asst = ChatCompletionRequestAssistantMessageArgs::default()
            .content("")
            .tool_calls(vec![tc])
            .build()
            .unwrap();
        let msgs = req(vec![ChatCompletionRequestMessage::Assistant(asst)]);
        let policy = ValidationPolicy {
            require_user_first: false,
            allow_dangling_tool_calls: true,
            ..Default::default()
        };
        let out = validate_conversation(&msgs, &policy);
        assert!(out.is_none());
    }

    #[test]
    fn unknown_tool_response_detected() {
        let tool = ChatCompletionRequestToolMessageArgs::default()
            .tool_call_id("unknown")
            .content("{}")
            .build()
            .unwrap();
        let msgs = req(vec![ChatCompletionRequestMessage::Tool(tool)]);
        let out = validate_conversation(&msgs, &ValidationPolicy::default()).unwrap();
        assert!(out.iter().any(|v| matches!(
            v.code,
            ViolationCode::UnknownToolResponse { .. } | ViolationCode::ToolBeforeAssistant { .. }
        )));
    }

    #[test]
    fn duplicate_tool_response_detected() {
        let tc = ChatCompletionMessageToolCall {
            id: "c1".to_string(),
            r#type: ChatCompletionToolType::Function,
            function: FunctionCall {
                name: "calc".into(),
                arguments: "{}".into(),
            },
        };
        let asst = ChatCompletionRequestAssistantMessageArgs::default()
            .content("")
            .tool_calls(vec![tc])
            .build()
            .unwrap();
        let tool1 = ChatCompletionRequestToolMessageArgs::default()
            .tool_call_id("c1")
            .content("{}")
            .build()
            .unwrap();
        let tool2 = ChatCompletionRequestToolMessageArgs::default()
            .tool_call_id("c1")
            .content("{}")
            .build()
            .unwrap();
        let msgs = req(vec![
            ChatCompletionRequestMessage::Assistant(asst),
            ChatCompletionRequestMessage::Tool(tool1),
            ChatCompletionRequestMessage::Tool(tool2),
        ]);
        let out = validate_conversation(&msgs, &ValidationPolicy::default()).unwrap();
        assert!(out
            .iter()
            .any(|v| matches!(v.code, ViolationCode::DuplicateToolResponse { .. })));
    }

    #[test]
    fn out_of_order_tool_responses_detected() {
        let tc1 = ChatCompletionMessageToolCall {
            id: "c1".to_string(),
            r#type: ChatCompletionToolType::Function,
            function: FunctionCall {
                name: "calc".into(),
                arguments: "{}".into(),
            },
        };
        let tc2 = ChatCompletionMessageToolCall {
            id: "c2".to_string(),
            r#type: ChatCompletionToolType::Function,
            function: FunctionCall {
                name: "calc".into(),
                arguments: "{}".into(),
            },
        };
        let asst = ChatCompletionRequestAssistantMessageArgs::default()
            .content("")
            .tool_calls(vec![tc1, tc2])
            .build()
            .unwrap();
        let tool_b = ChatCompletionRequestToolMessageArgs::default()
            .tool_call_id("c2")
            .content("{}")
            .build()
            .unwrap();
        let tool_a = ChatCompletionRequestToolMessageArgs::default()
            .tool_call_id("c1")
            .content("{}")
            .build()
            .unwrap();
        let msgs = req(vec![
            ChatCompletionRequestMessage::Assistant(asst),
            ChatCompletionRequestMessage::Tool(tool_b),
            ChatCompletionRequestMessage::Tool(tool_a),
        ]);
        let out = validate_conversation(&msgs, &ValidationPolicy::default()).unwrap();
        assert!(out
            .iter()
            .any(|v| matches!(v.code, ViolationCode::ToolResponsesOutOfOrder { .. })));
    }

    #[test]
    fn system_not_first_detected() {
        let usr = ChatCompletionRequestUserMessageArgs::default()
            .content("u")
            .build()
            .unwrap();
        let sys = ChatCompletionRequestSystemMessageArgs::default()
            .content("s")
            .build()
            .unwrap();
        let msgs = req(vec![usr.into(), sys.into()]);
        let out = validate_conversation(&msgs, &ValidationPolicy::default()).unwrap();
        assert!(out
            .iter()
            .any(|v| matches!(v.code, ViolationCode::SystemNotFirst { .. })));
    }

    #[test]
    fn duplicate_tool_call_ids_in_assistant_detected() {
        let tc1 = ChatCompletionMessageToolCall {
            id: "dup".to_string(),
            r#type: ChatCompletionToolType::Function,
            function: FunctionCall {
                name: "a".into(),
                arguments: "{}".into(),
            },
        };
        let tc2 = ChatCompletionMessageToolCall {
            id: "dup".to_string(),
            r#type: ChatCompletionToolType::Function,
            function: FunctionCall {
                name: "b".into(),
                arguments: "{}".into(),
            },
        };
        let asst = ChatCompletionRequestAssistantMessageArgs::default()
            .content("")
            .tool_calls(vec![tc1, tc2])
            .build()
            .unwrap();
        let msgs = req(vec![ChatCompletionRequestMessage::Assistant(asst)]);
        let policy = ValidationPolicy {
            require_user_first: false,
            ..Default::default()
        };
        let out = validate_conversation(&msgs, &policy).unwrap();
        assert!(out.iter().any(|v| matches!(
            v.code,
            ViolationCode::DuplicateToolCallIdsInAssistant { .. }
        )));
    }

    #[test]
    fn empty_tool_call_id_in_assistant_detected() {
        let tc1 = ChatCompletionMessageToolCall {
            id: "".to_string(),
            r#type: ChatCompletionToolType::Function,
            function: FunctionCall {
                name: "a".into(),
                arguments: "{}".into(),
            },
        };
        let asst = ChatCompletionRequestAssistantMessageArgs::default()
            .content("")
            .tool_calls(vec![tc1])
            .build()
            .unwrap();
        let msgs = req(vec![ChatCompletionRequestMessage::Assistant(asst)]);
        let policy = ValidationPolicy {
            require_user_first: false,
            ..Default::default()
        };
        let out = validate_conversation(&msgs, &policy).unwrap();
        assert!(out
            .iter()
            .any(|v| matches!(v.code, ViolationCode::EmptyToolCallIdInAssistant { .. })));
    }

    #[test]
    fn empty_tool_message_id_detected() {
        let tool = ChatCompletionRequestToolMessageArgs::default()
            .tool_call_id("")
            .content("{}")
            .build()
            .unwrap();
        let msgs = req(vec![ChatCompletionRequestMessage::Tool(tool)]);
        let out = validate_conversation(&msgs, &ValidationPolicy::default()).unwrap();
        assert!(out
            .iter()
            .any(|v| matches!(v.code, ViolationCode::EmptyToolMessageId { .. })));
    }

    #[test]
    fn tool_before_assistant_detected() {
        let tool = ChatCompletionRequestToolMessageArgs::default()
            .tool_call_id("id1")
            .content("{}")
            .build()
            .unwrap();
        let msgs = req(vec![ChatCompletionRequestMessage::Tool(tool)]);
        let out = validate_conversation(&msgs, &ValidationPolicy::default()).unwrap();
        assert!(out
            .iter()
            .any(|v| matches!(v.code, ViolationCode::ToolBeforeAssistant { .. })));
    }

    #[test]
    fn contiguity_enforced_detects_interruptions() {
        // Assistant with one tool call, then a user before tool response
        let tc = ChatCompletionMessageToolCall {
            id: "c1".into(),
            r#type: ChatCompletionToolType::Function,
            function: FunctionCall {
                name: "x".into(),
                arguments: "{}".into(),
            },
        };
        let asst = ChatCompletionRequestAssistantMessageArgs::default()
            .content("")
            .tool_calls(vec![tc])
            .build()
            .unwrap();
        let user = ChatCompletionRequestUserMessageArgs::default()
            .content("oops")
            .build()
            .unwrap();
        let tool = ChatCompletionRequestToolMessageArgs::default()
            .tool_call_id("c1")
            .content("{}")
            .build()
            .unwrap();
        let msgs = req(vec![
            ChatCompletionRequestMessage::Assistant(asst),
            ChatCompletionRequestMessage::User(user),
            ChatCompletionRequestMessage::Tool(tool),
        ]);
        let policy = ValidationPolicy {
            require_user_first: false,
            enforce_contiguous_tool_responses: true,
            ..Default::default()
        };
        let out = validate_conversation(&msgs, &policy).unwrap();
        assert!(out
            .iter()
            .any(|v| matches!(v.code, ViolationCode::ToolResponsesNotContiguous { .. })));
    }

    #[test]
    fn require_user_present_detects_absence() {
        let asst = ChatCompletionRequestAssistantMessageArgs::default()
            .content("hi")
            .build()
            .unwrap();
        let msgs = req(vec![ChatCompletionRequestMessage::Assistant(asst)]);
        let policy = ValidationPolicy {
            require_user_present: true,
            require_user_first: false,
            ..Default::default()
        };
        let out = validate_conversation(&msgs, &policy).unwrap();
        assert!(out
            .iter()
            .any(|v| matches!(v.code, ViolationCode::NoUserMessage)));
    }
}
