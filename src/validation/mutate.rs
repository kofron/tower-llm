//! Mutators to introduce specific violations into otherwise valid conversations.

use async_openai::types::*;

#[derive(Debug, Clone, Copy)]
pub enum MutationKind {
    AssistantBeforeUser,
    RepeatedUser,
    MissingOneToolResponse,
    UnknownToolResponse,
    ReorderToolResponses,
    DuplicateToolResponse,
    SystemNotFirst,
    DuplicateToolCallIdsInAssistant,
    EmptyToolCallIdInAssistant,
    EmptyToolMessageId,
    ToolBeforeAssistant,
    ToolResponsesNotContiguous,
    RemoveAllUsers,
}

/// Apply a mutation in-place. Returns true if mutation was applied.
pub fn apply_violation(
    messages: &mut Vec<ChatCompletionRequestMessage>,
    kind: MutationKind,
) -> bool {
    match kind {
        MutationKind::AssistantBeforeUser => {
            // Find first user; insert assistant at beginning if user exists.
            if messages
                .iter()
                .any(|m| matches!(m, ChatCompletionRequestMessage::User(_)))
            {
                let asst = ChatCompletionRequestAssistantMessageArgs::default()
                    .content("early")
                    .build()
                    .unwrap();
                messages.insert(0, asst.into());
                return true;
            }
            false
        }
        MutationKind::RepeatedUser => {
            // Duplicate first user consecutively
            if let Some((idx, _)) = messages
                .iter()
                .enumerate()
                .find(|(_, m)| matches!(m, ChatCompletionRequestMessage::User(_)))
            {
                if let ChatCompletionRequestMessage::User(u) = messages[idx].clone() {
                    messages.insert(idx + 1, ChatCompletionRequestMessage::User(u));
                    return true;
                }
            }
            false
        }
        MutationKind::MissingOneToolResponse => {
            // Remove the first tool message if any
            if let Some((idx, _)) = messages
                .iter()
                .enumerate()
                .find(|(_, m)| matches!(m, ChatCompletionRequestMessage::Tool(_)))
            {
                messages.remove(idx);
                return true;
            }
            false
        }
        MutationKind::UnknownToolResponse => {
            let t = ChatCompletionRequestToolMessageArgs::default()
                .tool_call_id("unknown_id")
                .content("{}")
                .build()
                .unwrap();
            // Insert near the end
            messages.push(t.into());
            true
        }
        MutationKind::ReorderToolResponses => {
            // Look for a block of two consecutive tool messages and swap them
            for i in 0..messages.len().saturating_sub(1) {
                if matches!(messages[i], ChatCompletionRequestMessage::Tool(_))
                    && matches!(messages[i + 1], ChatCompletionRequestMessage::Tool(_))
                {
                    messages.swap(i, i + 1);
                    return true;
                }
            }
            false
        }
        MutationKind::DuplicateToolResponse => {
            // Duplicate the first tool message and re-insert after it.
            if let Some((idx, m)) = messages
                .iter()
                .enumerate()
                .find(|(_, m)| matches!(m, ChatCompletionRequestMessage::Tool(_)))
            {
                messages.insert(idx + 1, m.clone());
                return true;
            }
            false
        }
        MutationKind::SystemNotFirst => {
            // Move first system to the middle, if any
            if let Some((idx, m)) = messages
                .iter()
                .enumerate()
                .find(|(_, m)| matches!(m, ChatCompletionRequestMessage::System(_)))
            {
                let sys = m.clone();
                messages.remove(idx);
                let insert_at = (messages.len() / 2).max(1);
                messages.insert(insert_at, sys);
                return true;
            }
            false
        }
        MutationKind::DuplicateToolCallIdsInAssistant => {
            for msg in messages.iter_mut() {
                if let ChatCompletionRequestMessage::Assistant(asst) = msg {
                    if let Some(calls) = asst.tool_calls.as_mut() {
                        if calls.len() >= 2 {
                            let id = calls[0].id.clone();
                            calls[1].id = id;
                            return true;
                        }
                    }
                }
            }
            false
        }
        MutationKind::EmptyToolCallIdInAssistant => {
            for msg in messages.iter_mut() {
                if let ChatCompletionRequestMessage::Assistant(asst) = msg {
                    if let Some(calls) = asst.tool_calls.as_mut() {
                        if !calls.is_empty() {
                            calls[0].id = String::new();
                            return true;
                        }
                    }
                }
            }
            false
        }
        MutationKind::EmptyToolMessageId => {
            for m in messages.iter_mut() {
                if let ChatCompletionRequestMessage::Tool(tmsg) = m {
                    tmsg.tool_call_id = String::new();
                    return true;
                }
            }
            false
        }
        MutationKind::ToolBeforeAssistant => {
            // Insert a tool message at the beginning with a fresh id
            let tool = ChatCompletionRequestToolMessageArgs::default()
                .tool_call_id("pre_tool")
                .content("{}")
                .build()
                .unwrap();
            messages.insert(0, ChatCompletionRequestMessage::Tool(tool));
            true
        }
        MutationKind::ToolResponsesNotContiguous => {
            // Find an assistant with tool_calls and interrupt with a user before the first tool response
            for i in 0..messages.len() {
                if let ChatCompletionRequestMessage::Assistant(asst) = &messages[i] {
                    if asst
                        .tool_calls
                        .as_ref()
                        .map(|v| !v.is_empty())
                        .unwrap_or(false)
                        && i + 1 < messages.len()
                    {
                        // Insert a user immediately after assistant
                        let u = ChatCompletionRequestUserMessageArgs::default()
                            .content("interrupt")
                            .build()
                            .unwrap();
                        messages.insert(i + 1, ChatCompletionRequestMessage::User(u));
                        return true;
                    }
                }
            }
            false
        }
        MutationKind::RemoveAllUsers => {
            let had_users = messages
                .iter()
                .any(|m| matches!(m, ChatCompletionRequestMessage::User(_)));
            messages.retain(|m| !matches!(m, ChatCompletionRequestMessage::User(_)));
            had_users
        }
    }
}
