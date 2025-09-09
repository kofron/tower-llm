//! Streaming agent demo that talks to OpenAI directly.
//!
//! This example shows how to stitch together the `streaming` primitives to create a
//! terminal experience where tokens, tool calls, and tool outputs arrive in real time.
//! It also keeps the raw SSE payloads available (guarded by the
//! `STREAMING_AGENT_SHOW_RAW` environment variable) so the same binary can double as a
//! debugging aid.
//!
//! Highlights
//! - Uses two arithmetic tools (`add_numbers`, `multiply_numbers`) routed through
//!   `ToolRouter`, so the model can choose which tool to call mid-stream.
//! - Streams `gpt-4o` responses using `Client::chat().create_stream` and forwards
//!   assistant tokens as they are received.
//! - Collects partial tool-call deltas, executes tools when the response finishes with
//!   `finish_reason = ToolCalls`, and then continues the conversation.
//! - Prints friendly progress messages (with optional raw JSON dumps) – a handy jumping
//!   off point for TUIs or CLIs.

use std::collections::BTreeMap;
use std::io::{self, Write};
use std::sync::Arc;

use async_openai::types::{
    ChatCompletionMessageToolCallChunk, ChatCompletionRequestAssistantMessageArgs,
    ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageArgs,
    ChatCompletionRequestToolMessageArgs, ChatCompletionRequestUserMessageArgs,
    ChatCompletionStreamResponseDelta, CreateChatCompletionRequestArgs,
    CreateChatCompletionStreamResponse, FinishReason,
};
use async_openai::Client;
use futures::StreamExt;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{json, Value};
use tower::{BoxError, Service, ServiceExt};

use tower_llm::{tool_typed, ToolDef, ToolInvocation, ToolOutput, ToolRouter};

#[derive(Debug, Deserialize, JsonSchema)]
struct AddArgs {
    a: i64,
    b: i64,
}

fn add_tool() -> ToolDef {
    tool_typed(
        "add_numbers",
        "Add two integers and return their sum",
        |args: AddArgs| async move { Ok(json!({ "sum": args.a + args.b })) },
    )
}

#[derive(Debug, Deserialize, JsonSchema)]
struct MultiplyArgs {
    a: i64,
    b: i64,
}

fn multiply_tool() -> ToolDef {
    tool_typed(
        "multiply_numbers",
        "Multiply two integers and return their product",
        |args: MultiplyArgs| async move { Ok(json!({ "product": args.a * args.b })) },
    )
}

#[derive(Default, Clone)]
struct PartialToolCall {
    id: Option<String>,
    name: Option<String>,
    arguments: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    if std::env::var("OPENAI_API_KEY").is_err() {
        eprintln!("OPENAI_API_KEY is not set – skipping example.");
        return Ok(());
    }

    let client = Arc::new(Client::new());

    // Build our toolset and router for executing tool invocations.
    let add = add_tool();
    let multiply = multiply_tool();
    let (router, tool_specs) = ToolRouter::new(vec![add, multiply]);
    let mut tool_service = tower::buffer::Buffer::new(router, 8);

    let system = ChatCompletionRequestSystemMessageArgs::default()
        .content("You are a helpful assistant. When arithmetic is requested, pick the most appropriate tool (add_numbers or multiply_numbers) before responding, and then explain the results clearly.")
        .build()?;
    let user = ChatCompletionRequestUserMessageArgs::default()
        .content("Add 2 and 40, and also multiply 6 and 7. Narrate the steps, call whichever tools you need, and summarize both results when done.")
        .build()?;

    let mut messages = vec![
        ChatCompletionRequestMessage::from(system),
        ChatCompletionRequestMessage::from(user),
    ];

    let show_raw = std::env::var("STREAMING_AGENT_SHOW_RAW").is_ok();

    println!("--- streaming agent with real OpenAI ---");
    if show_raw {
        println!("(STREAMING_AGENT_SHOW_RAW=1 → raw SSE events are printed)\n");
    } else {
        println!("(set STREAMING_AGENT_SHOW_RAW=1 to also dump raw SSE events)\n");
    }
    println!("Streaming tokens and tool activity will appear inline.\n");

    for step_index in 1..=3 {
        println!("=== request step {step_index} ===");
        let request = CreateChatCompletionRequestArgs::default()
            .model("gpt-4o")
            .messages(messages.clone())
            .tools(tool_specs.clone())
            .max_tokens(256u16)
            .temperature(0.0)
            .stream(true)
            .build()?;

        let mut stream = client.chat().create_stream(request).await?;
        let mut stdout = io::stdout();

        let mut content_buffer = String::new();
        let mut tool_calls: BTreeMap<u32, PartialToolCall> = BTreeMap::new();
        let mut finish_reason: Option<FinishReason> = None;

        while let Some(chunk) = stream.next().await {
            let chunk: CreateChatCompletionStreamResponse = chunk?;
            if show_raw {
                emit_raw_event(&chunk)?;
            }

            if let Some(choice) = chunk.choices.into_iter().next() {
                if finish_reason.is_none() {
                    finish_reason = choice.finish_reason;
                }

                process_delta(&choice.delta, &mut content_buffer, &mut stdout)?;
                if let Some(calls) = choice.delta.tool_calls {
                    for delta_call in calls {
                        let summary = merge_tool_call(&mut tool_calls, delta_call);
                        println!("{summary}");
                    }
                }
            }
        }

        println!();

        match finish_reason {
            Some(FinishReason::ToolCalls) if !tool_calls.is_empty() => {
                println!("tool call requested – executing locally");
                let (assistant_message, invocations) = finalize_tool_calls(tool_calls)?;
                messages.push(assistant_message);

                for (idx, invocation) in invocations.into_iter().enumerate() {
                    let ToolOutput { id, result } = ServiceExt::ready(&mut tool_service)
                        .await?
                        .call(invocation)
                        .await?;
                    println!(
                        "[tool result #{idx}] id={id} output={}",
                        render_json(result.clone())
                    );

                    let tool_msg = ChatCompletionRequestToolMessageArgs::default()
                        .tool_call_id(id)
                        .content(result.to_string())
                        .build()?;
                    messages.push(tool_msg.into());
                }
            }
            Some(FinishReason::Stop) | None => {
                if !content_buffer.trim().is_empty() {
                    println!("assistant> {content_buffer}\n");
                    let assistant = ChatCompletionRequestAssistantMessageArgs::default()
                        .content(content_buffer.clone())
                        .build()?;
                    messages.push(assistant.into());
                    break;
                }
            }
            other => {
                println!("finish_reason={:?}; stopping", other);
                break;
            }
        }
    }

    println!("\n--- final transcript ---");
    for (idx, msg) in messages.iter().enumerate() {
        println!("[{idx}] {msg:?}");
    }

    Ok(())
}

fn emit_raw_event(
    event: &CreateChatCompletionStreamResponse,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let json = serde_json::to_string(event)?;
    println!("[raw event] {json}");
    Ok(())
}

fn process_delta(
    delta: &ChatCompletionStreamResponseDelta,
    buffer: &mut String,
    stdout: &mut io::Stdout,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    if let Some(text) = &delta.content {
        buffer.push_str(text);
        print!("{text}");
        let _ = stdout.flush();
    }
    if let Some(refusal) = &delta.refusal {
        println!("[refusal] {refusal}");
    }
    Ok(())
}

fn merge_tool_call(
    map: &mut BTreeMap<u32, PartialToolCall>,
    delta_call: ChatCompletionMessageToolCallChunk,
) -> String {
    let entry = map.entry(delta_call.index).or_default();
    if let Some(id) = delta_call.id {
        entry.id = Some(id);
    }
    if let Some(function) = delta_call.function {
        if let Some(name) = function.name {
            entry.name = Some(name);
        }
        if let Some(args) = function.arguments {
            entry.arguments.push_str(&args);
        }
    }
    summarize_tool_call(delta_call.index, entry)
}

fn finalize_tool_calls(
    calls: BTreeMap<u32, PartialToolCall>,
) -> Result<(ChatCompletionRequestMessage, Vec<ToolInvocation>), BoxError> {
    let mut invocations = Vec::new();
    let mut tool_calls_for_msg = Vec::new();

    for (_idx, partial) in calls {
        let id = partial
            .id
            .ok_or_else(|| "missing tool_call id".to_string())?;
        let name = partial
            .name
            .ok_or_else(|| "missing tool_call name".to_string())?;
        let arguments: Value = if partial.arguments.trim().is_empty() {
            Value::Null
        } else {
            serde_json::from_str(&partial.arguments)
                .map_err(|e| format!("invalid JSON arguments: {e}"))?
        };

        invocations.push(ToolInvocation {
            id: id.clone(),
            name: name.clone(),
            arguments: arguments.clone(),
        });

        let tool_call = async_openai::types::ChatCompletionMessageToolCall {
            id: id.clone(),
            r#type: async_openai::types::ChatCompletionToolType::Function,
            function: async_openai::types::FunctionCall {
                name,
                arguments: arguments.to_string(),
            },
        };
        tool_calls_for_msg.push(tool_call);
    }

    let assistant_msg = ChatCompletionRequestAssistantMessageArgs::default()
        .tool_calls(tool_calls_for_msg)
        .build()?;

    Ok((assistant_msg.into(), invocations))
}

fn summarize_tool_call(index: u32, call: &PartialToolCall) -> String {
    let mut segments = Vec::new();
    if let Some(id) = &call.id {
        segments.push(format!("id={id}"));
    }
    if let Some(name) = &call.name {
        segments.push(format!("name={name}"));
    }
    let args = call.arguments.trim();
    if !args.is_empty() {
        segments.push(format!("args={}", clip_text(args, 96)));
    }

    if segments.is_empty() {
        format!("[tool-call delta #{index}] waiting for details")
    } else {
        format!("[tool-call delta #{index}] {}", segments.join(" | "))
    }
}

fn clip_text(input: &str, limit: usize) -> String {
    if input.chars().count() <= limit {
        return input.to_string();
    }
    let mut out = String::with_capacity(limit + 1);
    for (i, ch) in input.chars().enumerate() {
        if i >= limit {
            out.push('…');
            break;
        }
        out.push(ch);
    }
    out
}

fn render_json(value: Value) -> String {
    serde_json::to_string_pretty(&value).unwrap_or_else(|_| value.to_string())
}
