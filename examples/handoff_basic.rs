//! Basic handoff example demonstrating picker vs policy distinction with real LLM agents.
//!
//! This example shows:
//! - AgentPicker: WHO starts the conversation (initial routing)
//! - HandoffPolicy: HOW agents collaborate (runtime handoffs)
//! - Real LLM agents with specialized capabilities

use async_openai::{
    config::OpenAIConfig,
    types::{
        ChatCompletionRequestMessage, ChatCompletionRequestUserMessageArgs,
        ChatCompletionRequestSystemMessageArgs,
        CreateChatCompletionRequestArgs,
    },
    Client,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use tower::{Service, ServiceExt};
use tower_llm::{
    groups::{GroupBuilder, MultiExplicitHandoffPolicy, PickRequest},
    policies, Agent, AgentSvc, CompositePolicy,
};
use tracing::{info, debug};

// Math-focused calculator tool
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
enum Operation {
    Add,
    Subtract,
    Multiply,
    Divide,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct CalculatorArgs {
    /// The mathematical operation to perform (add, subtract, multiply, or divide)
    operation: Operation,
    /// First number
    a: f64,
    /// Second number
    b: f64,
}

// Writing analysis tool
#[derive(Debug, Deserialize, JsonSchema)]
struct TextAnalysisArgs {
    text: String,
    analysis_type: String, // "complexity", "readability", "word_count"
}

/// Topic-based picker that routes to appropriate specialist
#[derive(Clone)]
struct TopicPicker;

impl Service<PickRequest> for TopicPicker {
    type Response = String;
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
            // Analyze the user's message to determine the appropriate agent
            let content = req
                .messages
                .iter()
                .find_map(|msg| {
                    if let ChatCompletionRequestMessage::User(user_msg) = msg {
                        match &user_msg.content {
                            async_openai::types::ChatCompletionRequestUserMessageContent::Text(
                                text,
                            ) => Some(text.to_lowercase()),
                            _ => None,
                        }
                    } else {
                        None
                    }
                })
                .unwrap_or_default();

            // Route based on content
            if content.contains("calculate")
                || content.contains("math")
                || content.contains("compute")
                || content.contains("solve")
                || content.contains("+")
                || content.contains("-")
                || content.contains("*")
                || content.contains("/")
            {
                info!("🎯 Picker: Routing to math_agent (detected mathematical content)");
                Ok("math_agent".to_string())
            } else if content.contains("write")
                || content.contains("explain")
                || content.contains("describe")
                || content.contains("essay")
                || content.contains("story")
            {
                info!("🎯 Picker: Routing to writer_agent (detected writing task)");
                Ok("writer_agent".to_string())
            } else {
                info!("🎯 Picker: Routing to general_agent (general query)");
                Ok("general_agent".to_string())
            }
        })
    }
}

fn create_math_agent(client: Arc<Client<OpenAIConfig>>) -> AgentSvc {
    // Calculator tool for the math agent
    let calculator = tower_llm::tool_typed(
        "calculator",
        "Perform basic arithmetic operations. Use 'add', 'subtract', 'multiply', or 'divide' as the operation.",
        |args: CalculatorArgs| async move {
            let (result, op_str) = match args.operation {
                Operation::Add => (args.a + args.b, "+"),
                Operation::Subtract => (args.a - args.b, "-"),
                Operation::Multiply => (args.a * args.b, "*"),
                Operation::Divide => {
                    if args.b == 0.0 {
                        return Err("Division by zero".into());
                    }
                    (args.a / args.b, "/")
                }
            };

            println!("  🧮 Calculator: {} {} {} = {}", args.a, op_str, args.b, result);
            Ok::<_, tower::BoxError>(json!({ 
                "result": result,
                "expression": format!("{} {} {} = {}", args.a, op_str, args.b, result)
            }))
        },
    );

    let agent = Agent::builder(client)
        .model("gpt-4o-mini")
        .temperature(0.2) // Lower temperature for mathematical precision
        .tool(calculator)
        .policy(CompositePolicy::new(vec![
            policies::until_no_tool_calls(),
            policies::max_steps(3),
        ]))
        .build();

    agent
}

fn create_writer_agent(client: Arc<Client<OpenAIConfig>>) -> AgentSvc {
    // Text analysis tool for the writer agent
    let text_analyzer = tower_llm::tool_typed(
        "text_analyzer",
        "Analyze text for various metrics",
        |args: TextAnalysisArgs| async move {
            let result = match args.analysis_type.as_str() {
                "word_count" => {
                    let count = args.text.split_whitespace().count();
                    json!({ "word_count": count })
                }
                "complexity" => {
                    let avg_word_len: f64 = args.text.split_whitespace()
                        .map(|w| w.len() as f64)
                        .sum::<f64>() / args.text.split_whitespace().count() as f64;
                    json!({ "average_word_length": avg_word_len })
                }
                "readability" => {
                    let sentences = args.text.matches('.').count().max(1);
                    let words = args.text.split_whitespace().count();
                    let avg_sentence_len = words as f64 / sentences as f64;
                    json!({ "average_sentence_length": avg_sentence_len })
                }
                _ => json!({ "error": "Unknown analysis type" })
            };
            
            println!("  📝 Text Analyzer: {} = {:?}", args.analysis_type, result);
            Ok::<_, tower::BoxError>(result)
        },
    );

    let agent = Agent::builder(client)
        .model("gpt-4o-mini")
        .temperature(0.7) // Higher temperature for creativity
        .tool(text_analyzer)
        .policy(CompositePolicy::new(vec![
            policies::until_no_tool_calls(),
            policies::max_steps(3),
        ]))
        .build();

    agent
}

fn create_general_agent(client: Arc<Client<OpenAIConfig>>) -> AgentSvc {
    let agent = Agent::builder(client)
        .model("gpt-4o-mini")
        .temperature(0.5)
        .policy(CompositePolicy::new(vec![
            policies::until_no_tool_calls(),
            policies::max_steps(2),
        ]))
        .build();

    agent
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Initialize tracing with human-readable output
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .with_thread_ids(false)
        .with_thread_names(false)
        .with_file(false)
        .with_line_number(false)
        .with_level(true)
        .with_ansi(true)
        .compact()
        .init();

    println!("=== Basic Handoff Example with Real LLM Agents ===\n");

    println!("This example demonstrates the key distinction:");
    println!("• AgentPicker: WHO starts the conversation (initial routing)");
    println!("• HandoffPolicy: HOW agents collaborate (runtime handoffs)\n");

    // Create OpenAI client
    let client = Arc::new(Client::<OpenAIConfig>::new());

    // Create specialized agents
    let math_agent = create_math_agent(client.clone());
    let writer_agent = create_writer_agent(client.clone());
    let general_agent = create_general_agent(client.clone());

    // Create handoff policy - defines HOW agents can collaborate
    // These tools will be injected into agents to enable handoffs
    let mut handoffs = HashMap::new();
    handoffs.insert("handoff_to_math".to_string(), "math_agent".to_string());
    handoffs.insert("handoff_to_writer".to_string(), "writer_agent".to_string());
    handoffs.insert("handoff_to_general".to_string(), "general_agent".to_string());
    let handoff_policy = MultiExplicitHandoffPolicy::new(handoffs);

    // Create picker - defines WHO starts based on input
    let picker = TopicPicker;

    // Build coordinator with both picker and policy
    let mut coordinator = GroupBuilder::new()
        .agent("math_agent", math_agent)
        .agent("writer_agent", writer_agent)
        .agent("general_agent", general_agent)
        .picker(picker)
        .handoff_policy(handoff_policy)
        .build();

    println!("--- Scenario 1: Direct Math Question ---");
    println!("User: 'Calculate 25 * 4 + 15'\n");

    let system_message = async_openai::types::ChatCompletionRequestSystemMessageArgs::default()
        .content("You are a mathematics specialist. Use the calculator tool for calculations. For non-math tasks like writing, you can use the handoff_to_writer tool to delegate to the writing specialist.")
        .build()?;
    
    let user_message1 = ChatCompletionRequestUserMessageArgs::default()
        .content("Calculate 25 * 4 + 15")
        .build()?;

    let req1 = CreateChatCompletionRequestArgs::default()
        .model("gpt-4o-mini")
        .messages(vec![
            ChatCompletionRequestMessage::System(system_message),
            ChatCompletionRequestMessage::User(user_message1)
        ])
        .build()?;

    match coordinator.ready().await?.call(req1).await {
        Ok(result) => {
            println!("\n✅ Result: {} messages, {} steps", result.messages.len(), result.steps);
            if let Some(ChatCompletionRequestMessage::Assistant(msg)) = result.messages.last() {
                if let Some(content) = &msg.content {
                    let text = match content {
                        async_openai::types::ChatCompletionRequestAssistantMessageContent::Text(
                            t,
                        ) => t,
                        _ => "(non-text content)",
                    };
                    println!("📊 Final response: {}\n", text);
                }
            }
        }
        Err(e) => println!("❌ Error: {}\n", e),
    }

    println!("--- Scenario 2: Writing Task ---");
    println!("User: 'Write a short poem about coding'\n");

    let system_message2 = ChatCompletionRequestSystemMessageArgs::default()
        .content("You are a writing specialist. Use the text_analyzer tool to analyze any text. For math calculations, use the handoff_to_math tool to delegate to the math specialist.")
        .build()?;
    
    let user_message2 = ChatCompletionRequestUserMessageArgs::default()
        .content("Write a short poem about coding")
        .build()?;

    let req2 = CreateChatCompletionRequestArgs::default()
        .model("gpt-4o-mini")
        .messages(vec![
            ChatCompletionRequestMessage::System(system_message2),
            ChatCompletionRequestMessage::User(user_message2)
        ])
        .build()?;

    match coordinator.ready().await?.call(req2).await {
        Ok(result) => {
            println!("\n✅ Result: {} messages, {} steps", result.messages.len(), result.steps);
            if let Some(ChatCompletionRequestMessage::Assistant(msg)) = result.messages.last() {
                if let Some(content) = &msg.content {
                    let text = match content {
                        async_openai::types::ChatCompletionRequestAssistantMessageContent::Text(
                            t,
                        ) => t,
                        _ => "(non-text content)",
                    };
                    println!("📊 Final response: {}\n", text);
                }
            }
        }
        Err(e) => println!("❌ Error: {}\n", e),
    }

    println!("--- Scenario 3: Mixed Task (Potential Handoff) ---");
    println!("User: 'Calculate 100/5 and then write a haiku about the number twenty'\n");

    let system_message3 = ChatCompletionRequestSystemMessageArgs::default()
        .content("You are a mathematics specialist. Use the calculator tool for any math calculations (the operation must be 'add', 'subtract', 'multiply', or 'divide'). For creative writing tasks like poems or haikus, use the handoff_to_writer tool to delegate to the writing specialist.")
        .build()?;
    
    let user_message3 = ChatCompletionRequestUserMessageArgs::default()
        .content("Calculate 100 divided by 5, then write a haiku about the result (twenty)")
        .build()?;

    let req3 = CreateChatCompletionRequestArgs::default()
        .model("gpt-4o-mini")
        .messages(vec![
            ChatCompletionRequestMessage::System(system_message3),
            ChatCompletionRequestMessage::User(user_message3)
        ])
        .build()?;

    match coordinator.ready().await?.call(req3).await {
        Ok(result) => {
            println!("\n✅ Result: {} messages, {} steps", result.messages.len(), result.steps);
            println!("📊 Conversation flow:");
            for (i, msg) in result.messages.iter().enumerate() {
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
                        println!("   Step {}: {}", i + 1, preview);
                    }
                }
            }
        }
        Err(e) => println!("❌ Error: {}\n", e),
    }

    println!("\n=== Key Concepts Demonstrated ===");
    println!("1. **AgentPicker (TopicPicker)**:");
    println!("   - Analyzes initial request content");
    println!("   - Routes math questions → math_agent");
    println!("   - Routes writing tasks → writer_agent");
    println!("   - Routes general queries → general_agent");
    println!("   - Runs ONCE at start of conversation\n");

    println!("2. **HandoffPolicy (MultiExplicitHandoffPolicy)**:");
    println!("   - Provides handoff tools to agents");
    println!("   - Enables math_agent to hand off to writer_agent");
    println!("   - Enables writer_agent to hand off to math_agent");
    println!("   - Runs DURING conversation when agents need to collaborate\n");

    println!("3. **Real LLM Agents**:");
    println!("   - Each agent has specialized system prompts");
    println!("   - Math agent: Calculator tool, low temperature");
    println!("   - Writer agent: Text analysis tool, high temperature");
    println!("   - General agent: No tools, balanced approach\n");

    println!("4. **Specialization Benefits**:");
    println!("   - Agents focus on their core competencies");
    println!("   - Can delegate tasks outside their expertise");
    println!("   - Coordinator manages seamless transitions");
    println!("   - Conversation context preserved across handoffs\n");

    Ok(())
}