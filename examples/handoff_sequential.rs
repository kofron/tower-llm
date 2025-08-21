//! Sequential handoff example showing workflow orchestration with real LLM agents.
//!
//! This example demonstrates:
//! - Sequential workflow where agents pass work in a predefined order
//! - How picker vs policy work together for complex workflows
//! - Research → Analysis → Report generation pipeline

use async_openai::{
    config::OpenAIConfig,
    types::{
        ChatCompletionRequestMessage, ChatCompletionRequestUserMessageArgs,
        CreateChatCompletionRequestArgs,
    },
    Client,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use tower::{Service, ServiceExt};
use tower_llm::{
    groups::{GroupBuilder, PickRequest, SequentialHandoffPolicy},
    policies, Agent, AgentSvc, CompositePolicy,
};

// Research tools
#[derive(Debug, Deserialize, JsonSchema)]
struct WebSearchArgs {
    query: String,
    #[allow(dead_code)]
    max_results: Option<u32>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct AnalysisArgs {
    data: String,
    analysis_type: String, // "sentiment", "summary", "key_points"
}

/// Always starts with research agent for the workflow
#[derive(Clone)]
struct WorkflowPicker;

impl Service<PickRequest> for WorkflowPicker {
    type Response = String;
    type Error = tower::BoxError;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(
        &mut self,
        _: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        std::task::Poll::Ready(Ok(()))
    }

    fn call(&mut self, _req: PickRequest) -> Self::Future {
        Box::pin(async move {
            // Always start with research agent for this workflow
            println!("🎯 Picker: Starting workflow with research_agent");
            Ok("research_agent".to_string())
        })
    }
}

fn create_research_agent(client: Arc<Client<OpenAIConfig>>) -> AgentSvc {
    // Mock web search tool
    let web_search = tower_llm::tool_typed(
        "web_search",
        "Search the web for information",
        |args: WebSearchArgs| async move {
            println!("  🔍 Searching for: {}", args.query);
            let results = json!({
                "results": [
                    {
                        "title": "Recent Market Analysis",
                        "snippet": "The AI industry has grown 40% year-over-year with significant investments in LLM technology...",
                        "url": "https://example.com/market-analysis"
                    },
                    {
                        "title": "Technology Trends 2024",
                        "snippet": "Foundation models are becoming the cornerstone of enterprise AI adoption...",
                        "url": "https://example.com/tech-trends"
                    },
                    {
                        "title": "Industry Report",
                        "snippet": "Companies investing in AI see average productivity gains of 25%...",
                        "url": "https://example.com/industry-report"
                    }
                ],
                "query": args.query,
                "count": 3
            });
            Ok::<_, tower::BoxError>(results)
        },
    );

    Agent::builder(client)
        .model("gpt-4o-mini")
        .temperature(0.3)
        .tool(web_search)
        .policy(CompositePolicy::new(vec![
            policies::until_no_tool_calls(),
            policies::max_steps(3),
        ]))
        .build()
}

fn create_analysis_agent(client: Arc<Client<OpenAIConfig>>) -> AgentSvc {
    // Data analysis tool
    let analyzer = tower_llm::tool_typed(
        "analyze_data",
        "Analyze data for insights",
        |args: AnalysisArgs| async move {
            println!(
                "  📊 Analyzing: {} (type: {})",
                &args.data[..args.data.len().min(50)],
                args.analysis_type
            );

            let result = match args.analysis_type.as_str() {
                "sentiment" => json!({
                    "sentiment": "positive",
                    "confidence": 0.85,
                    "key_factors": ["growth", "investment", "productivity"]
                }),
                "summary" => json!({
                    "summary": "Strong growth in AI sector with enterprise adoption accelerating",
                    "main_points": 3
                }),
                "key_points" => json!({
                    "points": [
                        "40% YoY growth in AI industry",
                        "Foundation models driving adoption",
                        "25% average productivity gains"
                    ]
                }),
                _ => json!({"error": "Unknown analysis type"}),
            };

            Ok::<_, tower::BoxError>(result)
        },
    );

    Agent::builder(client)
        .model("gpt-4o-mini")
        .temperature(0.4)
        .tool(analyzer)
        .policy(CompositePolicy::new(vec![
            policies::until_no_tool_calls(),
            policies::max_steps(3),
        ]))
        .build()
}

fn create_report_agent(client: Arc<Client<OpenAIConfig>>) -> AgentSvc {
    // Report generator doesn't need tools - it synthesizes the findings
    Agent::builder(client)
        .model("gpt-4o-mini")
        .temperature(0.6) // Higher for creative report writing
        .policy(CompositePolicy::new(vec![
            policies::until_no_tool_calls(),
            policies::max_steps(2),
        ]))
        .build()
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

    println!("=== Sequential Handoff Example with Real LLM Agents ===\n");

    println!("This example demonstrates sequential workflow orchestration:");
    println!("• Picker: Always routes to research_agent (workflow entry point)");
    println!("• Policy: Enforces Research → Analysis → Report sequence");
    println!("• Each agent contributes specialized expertise to the pipeline\n");

    // Create OpenAI client
    let client = Arc::new(Client::<OpenAIConfig>::new());

    // Create specialized agents
    let research_agent = create_research_agent(client.clone());
    let analysis_agent = create_analysis_agent(client.clone());
    let report_agent = create_report_agent(client.clone());

    // Create sequential handoff policy
    // This will automatically hand off when each agent completes
    let sequence = vec![
        "research_agent".to_string(),
        "analysis_agent".to_string(),
        "report_agent".to_string(),
    ];
    let handoff_policy = SequentialHandoffPolicy::new(sequence);

    // Create picker that always starts with research
    let picker = WorkflowPicker;

    // Build coordinator
    let mut coordinator = GroupBuilder::new()
        .agent("research_agent", research_agent)
        .agent("analysis_agent", analysis_agent)
        .agent("report_agent", report_agent)
        .picker(picker)
        .handoff_policy(handoff_policy)
        .build();

    println!("--- Running Complete Workflow ---");
    println!("User: 'Analyze the current state of AI technology and create a brief report'\n");

    let user_message = ChatCompletionRequestUserMessageArgs::default()
        .content("Analyze the current state of AI technology and create a brief report about market trends, key developments, and future outlook.")
        .build()?;

    let req = CreateChatCompletionRequestArgs::default()
        .model("gpt-4o-mini")
        .messages(vec![ChatCompletionRequestMessage::User(user_message)])
        .build()?;

    let start_time = std::time::Instant::now();

    match coordinator.ready().await?.call(req).await {
        Ok(result) => {
            let duration = start_time.elapsed();
            println!("\n✅ Workflow completed successfully!");
            println!("📊 Statistics:");
            println!("   - Total messages: {}", result.messages.len());
            println!("   - Total steps: {}", result.steps);
            println!("   - Duration: {:?}", duration);

            println!("\n📝 Workflow progression:");
            for (i, msg) in result.messages.iter().enumerate() {
                if let ChatCompletionRequestMessage::Assistant(assistant_msg) = msg {
                    if let Some(content) = &assistant_msg.content {
                        let text = match content {
                            async_openai::types::ChatCompletionRequestAssistantMessageContent::Text(t) => t.as_str(),
                            _ => "(non-text content)",
                        };

                        // Identify which agent this is from based on content patterns
                        let agent = if i == 0 || text.contains("research") || text.contains("found")
                        {
                            "Research"
                        } else if text.contains("analysis") || text.contains("insights") {
                            "Analysis"
                        } else {
                            "Report"
                        };

                        let preview = if text.len() > 150 {
                            format!("{}...", &text[..150])
                        } else {
                            text.to_string()
                        };
                        println!("\n   [{}] {}", agent, preview);
                    }
                }
            }

            // Show the final report
            if let Some(ChatCompletionRequestMessage::Assistant(msg)) = result.messages.last() {
                if let Some(content) = &msg.content {
                    let text = match content {
                        async_openai::types::ChatCompletionRequestAssistantMessageContent::Text(
                            t,
                        ) => t,
                        _ => "(non-text content)",
                    };
                    println!("\n=== Final Report ===");
                    println!("{}", text);
                }
            }
        }
        Err(e) => println!("❌ Workflow failed: {}", e),
    }

    println!("\n=== Key Workflow Benefits ===");
    println!("1. **Specialization**: Each agent focuses on their expertise");
    println!("   - Research agent: Information gathering with web search");
    println!("   - Analysis agent: Data processing and insight extraction");
    println!("   - Report agent: Professional documentation and synthesis");
    println!();
    println!("2. **Quality Gates**: Sequential handoffs ensure complete work");
    println!("   - Research must complete before analysis begins");
    println!("   - Analysis must complete before report generation");
    println!("   - Each step builds on previous work");
    println!();
    println!("3. **Consistency**: Enforced workflow prevents shortcuts");
    println!("   - SequentialHandoffPolicy ensures proper order");
    println!("   - No skipping steps or ad-hoc routing");
    println!("   - Predictable, repeatable process");
    println!();
    println!("4. **Separation of Concerns**:");
    println!("   - **Picker**: Entry point routing (always → research_agent)");
    println!("   - **Policy**: Workflow orchestration (research → analysis → report)");
    println!("   - Clear architectural boundaries\n");

    Ok(())
}
