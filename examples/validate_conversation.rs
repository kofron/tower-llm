//! Example: validate a conversation and print violations.

use async_openai::types::*;
use tower_llm::validation::{validate_conversation, ValidationPolicy};

fn main() {
    // Assistant with a tool call, but no corresponding tool response (invalid)
    let tc = ChatCompletionMessageToolCall {
        id: "call_x".to_string(),
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
    let msgs = vec![ChatCompletionRequestMessage::Assistant(asst)];

    let policy = ValidationPolicy::default();
    match validate_conversation(&msgs, &policy) {
        None => println!("No violations."),
        Some(violations) => {
            println!("Violations ({}):", violations.len());
            for v in violations {
                println!("- {}", v.message);
            }
        }
    }
}
