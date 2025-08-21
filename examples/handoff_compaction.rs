//! Example demonstrating automatic conversation compaction during agent handoffs.
//!
//! This example shows how to use the CompactingHandoffPolicy to automatically
//! compact conversation history when transferring control between agents.
//! This helps manage context length and improves performance with long conversations.

use async_openai::{config::OpenAIConfig, Client};
use std::collections::HashMap;
use std::sync::Arc;
use tower_llm::{
    auto_compaction::{
        CompactionPolicy, CompactionStrategy, OrphanedToolCallStrategy, ProactiveThreshold,
    },
    groups::{CompactingHandoffPolicy, HandoffCoordinator, MultiExplicitHandoffPolicy},
    Agent, CompositePolicy, Service, ServiceExt,
};

/// Simple picker that always starts with the research agent
#[derive(Clone)]
struct ResearchFirstPicker;

impl tower::Service<tower_llm::groups::PickRequest> for ResearchFirstPicker {
    type Response = String;
    type Error = tower::BoxError;
    type Future = std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Self::Response, Self::Error>> + Send>,
    >;

    fn poll_ready(
        &mut self,
        _: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        std::task::Poll::Ready(Ok(()))
    }

    fn call(&mut self, _req: tower_llm::groups::PickRequest) -> Self::Future {
        Box::pin(async move { Ok("research".to_string()) })
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Initialize tracing for visibility
    tracing_subscriber::fmt::init();

    println!("=== Handoff with Compaction Example ===\n");
    println!("This example demonstrates automatic conversation compaction during handoffs.");
    println!("Watch as long conversations are automatically summarized when agents hand off.\n");

    let client = Arc::new(Client::<OpenAIConfig>::new());

    // Create research agent - gathers information
    let research_agent = Agent::builder(client.clone())
        .model("gpt-4o-mini")
        .temperature(0.7)
        .policy(CompositePolicy::new(vec![
            tower_llm::policies::until_no_tool_calls(),
            tower_llm::policies::max_steps(3),
        ]))
        .build();

    // Create analysis agent - processes gathered information
    let analysis_agent = Agent::builder(client.clone())
        .model("gpt-4o-mini")
        .temperature(0.3)
        .policy(CompositePolicy::new(vec![
            tower_llm::policies::until_no_tool_calls(),
            tower_llm::policies::max_steps(3),
        ]))
        .build();

    // Create report agent - generates final output
    let report_agent = Agent::builder(client.clone())
        .model("gpt-4o")
        .temperature(0.5)
        .policy(CompositePolicy::new(vec![
            tower_llm::policies::until_no_tool_calls(),
            tower_llm::policies::max_steps(2),
        ]))
        .build();

    // Build the group with agents
    let agents = HashMap::from([
        ("research".to_string(), research_agent),
        ("analysis".to_string(), analysis_agent),
        ("report".to_string(), report_agent),
    ]);

    // Create handoff policy with explicit handoff tools
    let mut handoffs = HashMap::new();
    handoffs.insert("handoff_to_analysis".to_string(), "analysis".to_string());
    handoffs.insert("handoff_to_report".to_string(), "report".to_string());
    handoffs.insert("handoff_to_research".to_string(), "research".to_string());

    let base_handoff_policy = MultiExplicitHandoffPolicy::new(handoffs);

    // Configure compaction policy
    let compaction_policy = CompactionPolicy {
        // Use smaller model for compaction to save costs
        compaction_model: "gpt-4o-mini".to_string(),

        // Very low threshold to trigger compaction for demo purposes
        // In production, you'd use a higher threshold like 8000-16000
        proactive_threshold: Some(ProactiveThreshold {
            token_threshold: 100, // Artificially low for demo
            percentage_threshold: None,
        }),

        // Keep system message and last 2 messages, compact the middle
        compaction_strategy: CompactionStrategy::PreserveSystemAndRecent { recent_count: 2 },

        // Custom prompt for research compaction
        compaction_prompt: tower_llm::auto_compaction::CompactionPrompt::Custom(
            "Summarize the research and analysis so far, preserving all key findings, \
             data points, and conclusions. Format as a brief but comprehensive summary."
                .to_string(),
        ),

        max_compaction_attempts: 2,
        orphaned_tool_call_strategy: OrphanedToolCallStrategy::DropAndReappend,
    };

    // Create provider for compaction
    let provider = Arc::new(tokio::sync::Mutex::new(
        tower_llm::provider::OpenAIProvider::new(client.clone()),
    ));

    // Wrap the handoff policy with compaction
    let compacting_policy =
        CompactingHandoffPolicy::new(base_handoff_policy, compaction_policy, provider);

    // Create the handoff coordinator
    let mut coordinator = HandoffCoordinator::new(agents, ResearchFirstPicker, compacting_policy);

    // Create a research-heavy request that will build up context
    let request = tower_llm::simple_chat_request(
        "You are a helpful research and analysis system. Work through the task step by step.",
        "Research the history and impact of the Tower pattern in software architecture. \
         Start by gathering information about its origins, then analyze its benefits and drawbacks, \
         and finally prepare a comprehensive report. \
         \
         For the research phase, provide detailed findings about: \
         1. The origins and evolution of the Tower pattern \
         2. Key implementations and frameworks \
         3. Comparison with other architectural patterns \
         4. Real-world use cases and success stories \
         \
         Make sure to gather extensive information before handing off to analysis. \
         After research is complete, use handoff_to_analysis to proceed. \
         After analysis, use handoff_to_report for the final output.",
    );

    println!("Starting multi-agent workflow with automatic compaction on handoff...\n");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Execute the coordinator
    let result = ServiceExt::ready(&mut coordinator)
        .await?
        .call(request)
        .await?;

    println!("\n═══════════════════════════════════════════════════════════════\n");
    println!("Workflow complete!");
    println!("Total messages: {}", result.messages.len());
    println!("Total steps: {}", result.steps);
    println!("Stop reason: {:?}", result.stop);

    // Display the final message
    if let Some(async_openai::types::ChatCompletionRequestMessage::Assistant(asst)) = result.messages.last() {
        if let Some(content) = &asst.content {
            println!("\n=== Final Output ===");
            match content {
                async_openai::types::ChatCompletionRequestAssistantMessageContent::Text(t) => {
                    println!("{}", t);
                }
                _ => println!("(Non-text content)"),
            }
        }
    }

    println!("\n=== Key Features Demonstrated ===");
    println!("✓ Multi-agent coordination with explicit handoffs");
    println!("✓ Automatic conversation compaction during handoffs");
    println!("✓ Context length management across agent boundaries");
    println!("✓ Preservation of orphaned tool calls during compaction");
    println!("✓ Graceful fallback if compaction fails");

    println!("\nNote: With the low token threshold (100), compaction likely triggered");
    println!("during handoffs. Check the logs to see compaction in action!");

    Ok(())
}
