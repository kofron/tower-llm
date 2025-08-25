use schemars::JsonSchema;
use serde::Deserialize;
use std::sync::Arc;
use tower::{BoxError, Layer, Service, ServiceExt};

use tower_llm::provider::OpenAIProvider;
use tower_llm::{
    policies, tool_typed, Client, CreateChatCompletionRequest, CreateChatCompletionRequestArgs,
    OpenAIConfig, ReasoningEffort, StepLayer, StepOutcome, ToolDef, ToolInvocation, ToolOutput,
    ToolRouter,
};

#[derive(Debug, Deserialize, JsonSchema)]
struct AddArgs {
    a: i64,
    b: i64,
}

fn add_tool() -> ToolDef {
    tool_typed(
        "add",
        "Add two integers and return their sum",
        |args: AddArgs| async move { Ok(args.a + args.b) },
    )
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Tracing for visibility of taps
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_target(false)
        .with_thread_ids(false)
        .with_thread_names(false)
        .with_file(false)
        .with_line_number(false)
        .with_level(true)
        .with_ansi(true)
        .compact()
        .init();

    // Simple tool set
    let add = add_tool();
    let (router, specs) = ToolRouter::new(vec![add]);

    // Tap around tool service to observe ToolInvocation ↔ ToolOutput
    let tool_tap = tower_llm::tap::TapLayer::<ToolInvocation, ToolOutput, BoxError>::new()
        .on_request(|inv: &ToolInvocation| {
            tracing::info!(tool = %inv.name, id = %inv.id, args = ?inv.arguments, "tap: tool request");
        })
        .on_response(|out: &ToolOutput| {
            tracing::info!(id = %out.id, result = ?out.result, "tap: tool response");
        })
        .on_error(|e: &BoxError| {
            tracing::error!(error = %e, "tap: tool error");
        });

    // Layer the tap over the router; make it cloneable for Step via Buffer
    let tapped_tools = tool_tap.layer(router);
    let tool_svc = tower::buffer::Buffer::new(tapped_tools, 16);

    // Base step with real OpenAI provider
    let client = Arc::new(Client::<OpenAIConfig>::new());
    let provider = OpenAIProvider::new(client);
    let step = StepLayer::new(provider, "gpt-4o", specs)
        .reasoning_effort(ReasoningEffort::Medium)
        .layer(tool_svc);

    // Tap around the step service to observe chat requests and outcomes
    let step_tap =
        tower_llm::tap::TapLayer::<CreateChatCompletionRequest, StepOutcome, BoxError>::new()
            .on_request(|req: &CreateChatCompletionRequest| {
                let model: Option<String> = req.model.clone().into();
                tracing::debug!(model = ?model, messages = req.messages.len(), "tap: step request");
            })
            .on_response(|out: &StepOutcome| match out {
                StepOutcome::Next { invoked_tools, .. } => {
                    tracing::debug!(tools = ?invoked_tools, "tap: step NEXT");
                }
                StepOutcome::Done { .. } => {
                    tracing::debug!("tap: step DONE");
                }
            })
            .on_error(|e: &BoxError| {
                tracing::error!(error = %e, "tap: step error");
            });
    let step = step_tap.layer(step);

    // Agent loop: stop after a few steps
    let policy = tower_llm::CompositePolicy::new(vec![policies::max_steps(4)]);
    let agent = tower_llm::AgentLoopLayer::new(policy).layer(step);
    let mut agent = tower::util::BoxService::new(agent);

    // Seed request encouraging tool usage
    let user = async_openai::types::ChatCompletionRequestUserMessageArgs::default()
        .content("Please compute 2 + 3 by calling the add tool and report the result.")
        .build()
        .unwrap();
    let req = CreateChatCompletionRequestArgs::default()
        .model("gpt-4o")
        .messages(vec![user.into()])
        .build()
        .unwrap();

    let out = ServiceExt::ready(&mut agent).await?.call(req).await?;
    println!("Steps: {} | Stop: {:?}", out.steps, out.stop);
    for m in out.messages {
        match m {
            async_openai::types::ChatCompletionRequestMessage::System(s) => {
                let txt = match s.content {
                    async_openai::types::ChatCompletionRequestSystemMessageContent::Text(t) => t,
                    _ => String::from("[non-text system content]"),
                };
                println!("SYSTEM: {}", txt);
            }
            async_openai::types::ChatCompletionRequestMessage::User(u) => {
                let txt = match u.content {
                    async_openai::types::ChatCompletionRequestUserMessageContent::Text(t) => t,
                    _ => String::from("[non-text user content]"),
                };
                println!("USER: {}", txt);
            }
            async_openai::types::ChatCompletionRequestMessage::Assistant(a) => {
                let txt = if let Some(content) = a.content {
                    match content {
                        async_openai::types::ChatCompletionRequestAssistantMessageContent::Text(
                            t,
                        ) => t,
                        _ => String::from(""),
                    }
                } else {
                    String::new()
                };
                if !txt.is_empty() {
                    println!("ASSISTANT: {}", txt);
                }
                if let Some(calls) = a.tool_calls {
                    for c in calls {
                        println!(
                            "ASSISTANT requested tool: {} args={} (id={})",
                            c.function.name, c.function.arguments, c.id
                        );
                    }
                }
            }
            async_openai::types::ChatCompletionRequestMessage::Tool(t) => {
                let txt = match t.content {
                    async_openai::types::ChatCompletionRequestToolMessageContent::Text(s) => s,
                    _ => String::from("[non-text tool content]"),
                };
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&txt) {
                    println!(
                        "TOOL[{}]: {}",
                        t.tool_call_id,
                        serde_json::to_string_pretty(&v).unwrap()
                    );
                } else {
                    println!("TOOL[{}]: {}", t.tool_call_id, txt);
                }
            }
            _ => {}
        }
    }

    Ok(())
}
