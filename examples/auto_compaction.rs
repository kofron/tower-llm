//! Example demonstrating auto-compaction for managing long conversations
//!
//! This example shows how to use the auto-compaction layer to automatically
//! summarize conversation history when approaching context limits.

use std::sync::Arc;

use async_openai::{config::OpenAIConfig, Client};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;
use tower_llm::{
    auto_compaction::{CompactionPolicy, CompactionPrompt, CompactionStrategy, ProactiveThreshold},
    Agent, CompositePolicy, Service, ServiceExt,
};

#[derive(Deserialize, JsonSchema)]
struct StoryArgs {
    topic: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    tracing_subscriber::fmt::init();

    println!("=== Auto-Compaction Example ===\n");
    println!("This example demonstrates automatic conversation compaction.");
    println!("We'll have a long conversation that triggers compaction.\n");

    // Tool for continuing a story
    let story_tool = tower_llm::tool_typed(
        "continue_story",
        "Continue the story with the given topic",
        |args: StoryArgs| async move {
            Ok::<_, tower::BoxError>(json!({
                "continuation": format!("The story continues with {}...", args.topic)
            }))
        },
    );

    let client = Arc::new(Client::<OpenAIConfig>::new());

    // Configure auto-compaction policy
    let compaction_policy = CompactionPolicy {
        // Use a smaller model for compaction to save costs
        compaction_model: "gpt-4o-mini".to_string(),

        // Proactive threshold - compact when we hit 500 tokens (very low for demo)
        proactive_threshold: Some(ProactiveThreshold {
            token_threshold: 500,
            percentage_threshold: None,
        }),

        // Keep system message and last 4 messages, compact the middle
        compaction_strategy: CompactionStrategy::PreserveSystemAndRecent { recent_count: 4 },

        // Custom prompt for story compaction
        compaction_prompt: CompactionPrompt::Custom(
            "Summarize the story so far, preserving all key plot points, \
             character developments, and important details. \
             Format as a brief narrative summary."
                .to_string(),
        ),

        max_compaction_attempts: 2,
    };

    // Build agent with auto-compaction
    let mut agent = Agent::builder(client)
        .model("gpt-4o")
        .tool(story_tool)
        .auto_compaction(compaction_policy)
        .policy(CompositePolicy::new(vec![
            tower_llm::policies::until_no_tool_calls(),
            tower_llm::policies::max_steps(10),
        ]))
        .build();

    println!("Starting a collaborative story that will trigger compaction...\n");

    // First interaction - establish the story
    let run1 = tower_llm::run(
        &mut agent,
        "You are a creative storyteller. Help me write an epic fantasy story. \
         Start with a mysterious beginning involving a forgotten kingdom.",
        "Begin our story with a mysterious prophecy discovered in ancient ruins.",
    )
    .await?;

    println!("=== Part 1: The Beginning ===");
    if let Some(async_openai::types::ChatCompletionRequestMessage::Assistant(msg)) =
        run1.messages.last()
    {
        if let Some(content) = &msg.content {
            // Extract text from content enum
            let text = match content {
                async_openai::types::ChatCompletionRequestAssistantMessageContent::Text(t) => t,
                _ => "...",
            };
            println!("{}\n", text);
        }
    }

    // Continue the story multiple times to build up context
    let prompts = [
        "Continue the story. The protagonist finds a magical artifact.",
        "Add a twist where the artifact reveals a hidden truth about the protagonist's past.",
        "Introduce a wise mentor character who knows about the prophecy.",
        "The mentor reveals there's a dark force awakening. Add tension.",
        "Describe the journey to the ancient library where more answers lie.",
        "At the library, they discover a map to three sacred temples.",
        "Detail their preparation for the dangerous journey ahead.",
        "They encounter their first major obstacle - a cursed forest.",
    ];

    for (i, prompt) in prompts.iter().enumerate() {
        println!("=== Part {}: Continuing... ===", i + 2);
        println!("User: {}\n", prompt);

        // Build request manually to simulate ongoing conversation
        let request = tower_llm::simple_chat_request(
            "You are a creative storyteller continuing an epic fantasy story.",
            prompt,
        );

        let run = ServiceExt::ready(&mut agent).await?.call(request).await?;

        if let Some(async_openai::types::ChatCompletionRequestMessage::Assistant(msg)) =
            run.messages.last()
        {
            if let Some(content) = &msg.content {
                // Extract text from content enum
                let text = match content {
                    async_openai::types::ChatCompletionRequestAssistantMessageContent::Text(t) => t,
                    _ => "...",
                };

                // Check if this is a compacted summary
                if text.starts_with("[Previous conversation summary]") {
                    println!("📝 **COMPACTION OCCURRED** 📝");
                    println!(
                        "The conversation was automatically compacted to manage context length."
                    );
                    println!("{}\n", text);
                } else {
                    println!("Assistant: {}\n", text);
                }
            }
        }

        // Small delay to make the output readable
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
    }

    println!("\n=== Example Complete ===");
    println!("The auto-compaction layer automatically managed the conversation length.");
    println!("When the token count exceeded the threshold, it:");
    println!("1. Summarized the older parts of the conversation");
    println!("2. Preserved recent messages for continuity");
    println!("3. Continued the conversation seamlessly");
    println!("\nThis is especially useful for:");
    println!("- Long-running conversations");
    println!("- Chat applications with context limits");
    println!("- Cost optimization (using smaller models for compaction)");
    println!("- Maintaining conversation coherence over many turns");

    Ok(())
}
