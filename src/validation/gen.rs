//! Proptest-based generators for valid conversations.

use async_openai::types::*;
use proptest::prelude::*;

#[derive(Debug, Clone)]
pub struct GeneratorConfig {
    pub min_messages: usize,
    pub max_messages: usize,
    pub must_have_system: bool,
    pub must_have_tool_calls: bool,
    pub min_tool_calls: usize,
    pub allow_developer: bool,
    pub enforce_tool_order: bool,
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        Self {
            min_messages: 2,
            max_messages: 12,
            must_have_system: true,
            must_have_tool_calls: false,
            min_tool_calls: 0,
            allow_developer: false,
            enforce_tool_order: true,
        }
    }
}

/// Strategy that yields valid conversations that should pass the default validator.
pub fn valid_conversation(
    cfg: GeneratorConfig,
) -> impl Strategy<Value = Vec<ChatCompletionRequestMessage>> {
    // Keep an intentionally simple generator for MVP: up to 3 turns, optional tools.
    let turn = (any::<bool>(), any::<bool>()); // (with_tools, with_text)
    let turns = proptest::collection::vec(turn, 1..=3);
    let sys_flag = proptest::strategy::Just(cfg.must_have_system);
    (sys_flag, turns).prop_map(move |(must_sys, turns)| {
        let mut msgs: Vec<ChatCompletionRequestMessage> = Vec::new();
        if must_sys {
            let sys = ChatCompletionRequestSystemMessageArgs::default()
                .content("sys")
                .build()
                .unwrap();
            msgs.push(sys.into());
        }
        // Ensure first non-system is user
        let usr = ChatCompletionRequestUserMessageArgs::default()
            .content("hi")
            .build()
            .unwrap();
        msgs.push(usr.into());

        let mut tool_id_counter = 1usize;
        let last_turn = turns.len().saturating_sub(1);
        for (idx, (with_tools, with_text)) in turns.into_iter().enumerate() {
            // Assistant response
            if with_tools || cfg.must_have_tool_calls {
                let min_calls = cfg.min_tool_calls.max(1);
                let num_calls = std::cmp::max(min_calls, 1);
                let mut calls: Vec<ChatCompletionMessageToolCall> = Vec::new();
                for _ in 0..num_calls {
                    let id = format!("c{}", tool_id_counter);
                    tool_id_counter += 1;
                    calls.push(ChatCompletionMessageToolCall {
                        id: id.clone(),
                        r#type: ChatCompletionToolType::Function,
                        function: FunctionCall {
                            name: "tool".into(),
                            arguments: "{}".into(),
                        },
                    });
                }
                let asst = ChatCompletionRequestAssistantMessageArgs::default()
                    .content("")
                    .tool_calls(calls.clone())
                    .build()
                    .unwrap();
                msgs.push(asst.into());
                // Tool responses, in order
                for tc in calls.into_iter() {
                    let t = ChatCompletionRequestToolMessageArgs::default()
                        .tool_call_id(tc.id)
                        .content("{}")
                        .build()
                        .unwrap();
                    msgs.push(t.into());
                }
            } else {
                let content = if with_text { "ok" } else { "" };
                let asst = ChatCompletionRequestAssistantMessageArgs::default()
                    .content(content)
                    .build()
                    .unwrap();
                msgs.push(asst.into());
            }
            // Always add a user between assistant turns, except after the last turn
            if idx != last_turn {
                let u = ChatCompletionRequestUserMessageArgs::default()
                    .content("next")
                    .build()
                    .unwrap();
                msgs.push(u.into());
            }
        }
        msgs
    })
}
