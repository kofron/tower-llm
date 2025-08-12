//! Example demonstrating persistent session storage with SQLite
//!
//! This shows how sessions persist across program runs using SQLite storage.
//!
//! Run with: cargo run --example persistent_session

use openai_agents_rs::{
    memory::Session, runner::RunConfig, sqlite_session::SqliteSession, Agent, Runner,
};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Persistent Session Example ===\n");
    println!("This example demonstrates persistent conversation history using SQLite.");
    println!("The conversation will be saved and restored on restart.\n");

    // Create or connect to a persistent SQLite database
    // Note: For this example, we'll use in-memory database for simplicity
    // In production, you'd use a file path like "chat_history.db"
    let session_id = "main_user";

    // Create the session (in-memory for demo, use SqliteSession::new for file-based)
    let session = Arc::new(SqliteSession::new_in_memory(session_id).await?);

    // Check if we have previous conversation history
    let previous_items = session.get_items(Some(10)).await?;
    if !previous_items.is_empty() {
        println!(
            "📚 Found {} previous conversation items",
            previous_items.len()
        );
        println!("Continuing from where we left off...\n");
    } else {
        println!("🆕 Starting a new conversation\n");
    }

    // Create an agent
    let agent = Agent::simple(
        "PersistentAssistant",
        "You are a helpful assistant with memory of past conversations. \
         If you remember previous interactions, acknowledge them.",
    );

    // Configure with session
    let config = RunConfig {
        max_turns: Some(5),
        stream: false,
        session: Some(session.clone()),
        model_provider: None, // Use default OpenAI provider
    };

    // Simulate multiple interactions
    let queries = vec![
        "Hello! My name is Alice and I love programming in Rust.",
        "What's my name and what do I like?",
        "Can you recommend some Rust learning resources?",
    ];

    for (i, query) in queries.iter().enumerate() {
        println!("👤 User (Turn {}): {}", i + 1, query);

        let result = Runner::run(agent.clone(), query.to_string(), config.clone()).await?;

        if result.is_success() {
            println!("🤖 Assistant: {}\n", result.final_output);

            // The session is automatically updated by the runner
            let total_items = session.get_items(None).await?.len();
            println!("  💾 Session now contains {} items", total_items);
        } else {
            println!("❌ Error: {:?}", result.error);
        }

        if i < queries.len() - 1 {
            println!("{}", "-".repeat(50));
        }
    }

    // Demonstrate session persistence
    println!("\n{}", "=".repeat(60));
    println!("📊 Session Summary:");

    let all_items = session.get_items(None).await?;
    println!("  Total items stored: {}", all_items.len());

    let messages = session.get_messages(None).await?;
    println!("  Total messages: {}", messages.len());

    // Show recent messages
    if messages.len() > 0 {
        println!("\n  Recent messages:");
        for (i, msg) in messages.iter().rev().take(3).enumerate() {
            println!(
                "    {}. [{:?}] {}",
                i + 1,
                msg.role,
                if msg.content.len() > 50 {
                    format!("{}...", &msg.content[..50])
                } else {
                    msg.content.clone()
                }
            );
        }
    }

    println!("\n✨ Session data is persisted and will be available on next run!");
    println!("   (Note: This example uses in-memory DB for demo purposes)");

    Ok(())
}
