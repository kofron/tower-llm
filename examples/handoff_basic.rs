//! Basic handoff example demonstrating picker vs policy distinction.
//!
//! This example shows:
//! - AgentPicker: WHO starts the conversation (initial routing)
//! - HandoffPolicy: HOW agents collaborate (runtime handoffs)
//! - Simple explicit handoff between two specialized agents

use async_openai::types::{
    ChatCompletionRequestMessage, ChatCompletionRequestAssistantMessageArgs,
    ChatCompletionRequestUserMessageArgs, ChatCompletionMessageToolCall,
    ChatCompletionToolType, FunctionCall, FunctionObject, ChatCompletionTool,
    CreateChatCompletionRequest, CreateChatCompletionRequestArgs, Role,
};
use serde_json::json;
use std::collections::HashMap;
use std::pin::Pin;
use std::future::Future;
use tower::{Service, ServiceExt};
use tower_llm::{
    AgentRun, AgentStopReason, AgentSvc,
    groups::{
        AgentName, AgentPicker, MultiExplicitHandoffPolicy, GroupBuilder, HandoffCoordinator,
        PickRequest,
    },
};

// Mock agent services for demonstration
#[derive(Clone)]
struct MathAgent;

impl Service<CreateChatCompletionRequest> for MathAgent {
    type Response = AgentRun;
    type Error = tower::BoxError;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(
        &mut self,
        _: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        std::task::Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: CreateChatCompletionRequest) -> Self::Future {
        Box::pin(async move {
            // Math agent can handle calculations but hands off to writer for explanations
            let needs_explanation = req.messages.iter().any(|msg| {
                if let ChatCompletionRequestMessage::User(user_msg) = msg {
                    match &user_msg.content {
                        async_openai::types::ChatCompletionRequestUserMessageContent::Text(text) => {
                            text.contains("explain") || text.contains("write")
                        }
                        _ => false,
                    }
                } else {
                    false
                }
            });

            if needs_explanation {
                // Hand off to writer agent for explanations
                let tool_call = ChatCompletionMessageToolCall {
                    id: "handoff_1".to_string(),
                    r#type: ChatCompletionToolType::Function,
                    function: FunctionCall {
                        name: "handoff_to_writer".to_string(),
                        arguments: json!({"reason": "Need explanation of calculation"}).to_string(),
                    },
                };

                let message = ChatCompletionRequestAssistantMessageArgs::default()
                    .content("I can do the math, but let me hand this off to the writer for a clear explanation.")
                    .tool_calls(vec![tool_call])
                    .build()?;

                return Ok(AgentRun {
                    messages: vec![ChatCompletionRequestMessage::Assistant(message)],
                    steps: 1,
                    stop: AgentStopReason::ToolCalled("handoff_to_writer".to_string()),
                });
            }

            // Handle math directly
            let message = ChatCompletionRequestAssistantMessageArgs::default()
                .content("25 * 4 = 100. The calculation is complete.")
                .build()?;

            Ok(AgentRun {
                messages: vec![ChatCompletionRequestMessage::Assistant(message)],
                steps: 1,
                stop: AgentStopReason::DoneNoToolCalls,
            })
        })
    }
}

#[derive(Clone)]
struct WriterAgent;

impl Service<CreateChatCompletionRequest> for WriterAgent {
    type Response = AgentRun;
    type Error = tower::BoxError;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(
        &mut self,
        _: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        std::task::Poll::Ready(Ok(()))
    }

    fn call(&mut self, _req: CreateChatCompletionRequest) -> Self::Future {
        Box::pin(async move {
            let message = ChatCompletionRequestAssistantMessageArgs::default()
                .content("Let me explain this calculation step by step:\n\n25 × 4 = 100\n\nThis is because we're multiplying 25 by 4, which means adding 25 four times: 25 + 25 + 25 + 25 = 100. Alternatively, you can think of it as 20 × 4 + 5 × 4 = 80 + 20 = 100.")
                .build()?;

            Ok(AgentRun {
                messages: vec![ChatCompletionRequestMessage::Assistant(message)],
                steps: 1,
                stop: AgentStopReason::DoneNoToolCalls,
            })
        })
    }
}

// Simple picker that routes math questions to math agent, writing to writer
#[derive(Clone)]
struct TopicPicker;

impl Service<PickRequest> for TopicPicker {
    type Response = AgentName;
    type Error = tower::BoxError;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(
        &mut self,
        _: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        std::task::Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: PickRequest) -> Self::Future {
        Box::pin(async move {
            let contains_math = req.messages.iter().any(|msg| {
                if let ChatCompletionRequestMessage::User(user_msg) = msg {
                    match &user_msg.content {
                        async_openai::types::ChatCompletionRequestUserMessageContent::Text(text) => {
                            text.contains("calculate") || text.contains("math") || text.contains("*")
                        }
                        _ => false,
                    }
                } else {
                    false
                }
            });

            if contains_math {
                Ok("math_agent".to_string())
            } else {
                Ok("writer_agent".to_string())
            }
        })
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("=== Basic Handoff Example ===\n");

    println!("This example demonstrates the key distinction:");
    println!("• AgentPicker: WHO starts the conversation (initial routing)");
    println!("• HandoffPolicy: HOW agents collaborate (runtime handoffs)\n");

    // Create handoff policy - defines HOW agents can collaborate
    // This allows math_agent to hand off to writer_agent for explanations
    let mut handoffs = HashMap::new();
    handoffs.insert(
        "handoff_to_writer".to_string(),
        "writer_agent".to_string(),
    );
    let handoff_policy = MultiExplicitHandoffPolicy::new(handoffs);

    // Create picker - defines WHO starts based on input
    let picker = TopicPicker;

    // Build coordinator with both picker and policy
    let mut coordinator = GroupBuilder::new()
        .agent("math_agent", tower::util::BoxService::new(MathAgent) as AgentSvc)
        .agent("writer_agent", tower::util::BoxService::new(WriterAgent) as AgentSvc)
        .picker(picker)
        .handoff_policy(handoff_policy)
        .build();

    println!("--- Scenario 1: Simple Math (No Handoff) ---");
    println!("Input: 'What is 25 * 4?'");
    println!("Expected: Picker routes to math_agent, math_agent handles directly\n");

    let user_message1 = ChatCompletionRequestUserMessageArgs::default()
        .content("What is 25 * 4?")
        .build()?;

    let req1 = CreateChatCompletionRequestArgs::default()
        .model("gpt-4o")
        .messages(vec![ChatCompletionRequestMessage::User(user_message1)])
        .build()?;

    match coordinator.ready().await?.call(req1).await {
        Ok(result) => {
            println!("✅ Result: {} messages, {} steps", result.messages.len(), result.steps);
            if let Some(ChatCompletionRequestMessage::Assistant(msg)) = result.messages.first() {
                if let Some(content) = &msg.content {
                    let text = match content {
                        async_openai::types::ChatCompletionRequestAssistantMessageContent::Text(t) => t,
                        _ => "(non-text content)",
                    };
                    println!("✅ Response: {}\n", text);
                }
            }
        }
        Err(e) => println!("❌ Error: {}\n", e),
    }

    println!("--- Scenario 2: Math with Explanation Request (With Handoff) ---");
    println!("Input: 'Calculate 25 * 4 and explain how you got the answer'");
    println!("Expected: Picker routes to math_agent, math_agent hands off to writer_agent\n");

    let user_message2 = ChatCompletionRequestUserMessageArgs::default()
        .content("Calculate 25 * 4 and explain how you got the answer")
        .build()?;

    let req2 = CreateChatCompletionRequestArgs::default()
        .model("gpt-4o")
        .messages(vec![ChatCompletionRequestMessage::User(user_message2)])
        .build()?;

    match coordinator.ready().await?.call(req2).await {
        Ok(result) => {
            println!("✅ Result: {} messages, {} steps", result.messages.len(), result.steps);
            println!("✅ Math agent initiated handoff to writer agent");
            for msg in &result.messages {
                if let ChatCompletionRequestMessage::Assistant(assistant_msg) = msg {
                    if let Some(content) = &assistant_msg.content {
                        let text = match content {
                            async_openai::types::ChatCompletionRequestAssistantMessageContent::Text(t) => t.as_str(),
                            _ => "(non-text content)",
                        };
                        let preview = if text.len() > 100 {
                            format!("{}...", &text[..100])
                        } else {
                            text.to_string()
                        };
                        println!("   Response: {}", preview);
                    }
                }
            }
            println!();
        }
        Err(e) => println!("❌ Error: {}\n", e),
    }

    println!("=== Key Concepts Demonstrated ===");
    println!("1. **AgentPicker (TopicPicker)**:");
    println!("   - Analyzes initial request");
    println!("   - Routes math questions → math_agent");
    println!("   - Routes writing questions → writer_agent");
    println!("   - Runs ONCE at start of conversation\n");

    println!("2. **HandoffPolicy (ExplicitHandoffPolicy)**:");
    println!("   - Defines handoff_to_writer tool");
    println!("   - Allows math_agent to transfer control to writer_agent");
    println!("   - Runs DURING conversation when agents call handoff tools\n");

    println!("3. **Specialization Benefits**:");
    println!("   - Math agent: Fast calculations, knows when to delegate");
    println!("   - Writer agent: Detailed explanations and documentation");
    println!("   - Each agent focuses on their core competency\n");

    Ok(())
}