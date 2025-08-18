//! # Example: Persistent Sessions with SQLite
//!
//! This example demonstrates how to use the `SqliteSession` to create a
//! persistent conversation history that is saved to a database file. This
//! allows an agent to maintain context across multiple application runs.
//!
//! ## Key Concepts Demonstrated
//!
//! - **`SqliteSession`**: The core component for persistent storage. It saves
//!   the entire conversation trace to a SQLite database.
//! - **Session Continuity**: The example shows how to check for an existing
//!   session and continue the conversation from where it left off.
//! - **Automatic State Management**: The `Runner` automatically handles the
//!   loading and saving of session data when a `Session` is provided in the
//!   `RunConfig`.
//!
//! **Note**: For simplicity, this example uses an in-memory SQLite database.
//! To make the session truly persistent, you would provide a file path when
//! creating the `SqliteSession`, like so:
//! `SqliteSession::new("user_session", "my_chat_history.db").await?`
//!
//! To run this example, you first need to set your `OPENAI_API_KEY` environment
//! variable.
//!
//! ```bash
//! export OPENAI_API_KEY="your-api-key"
//! cargo run --example persistent_session
//! ```
//!
//! Expected: The agent should be able to continue the conversation from where it left off.  
//! It might also contain other filler text, but as long as the conversation is continued, this example has succeeded.

use openai_agents_rs::{
    memory::Session, runner::RunConfig, sqlite_session::SqliteSession, Agent, Runner,
};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Persistent Session Example ===\n");
    println!("This example demonstrates persistent conversation history using SQLite.");
    println!("The conversation will be saved and restored on restart.\n");

    // 1. Create or connect to a persistent session.
    //
    // We use a unique session ID to identify the conversation. If a database
    // with this session ID already exists, the history will be loaded.
    let session_id = "main_user";

    // For this example, we use an in-memory database for simplicity.
    // In a real application, you would provide a file path to create a
    // persistent database on disk.
    let session = Arc::new(SqliteSession::new_in_memory(session_id).await?);

    // 2. Check for and display previous conversation history.
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

    // 3. Create an agent designed to use its memory.
    //
    // The agent's instructions encourage it to acknowledge past interactions,
    // demonstrating that it has access to the conversation history.
    let agent = Agent::simple(
        "PersistentAssistant",
        "You are a helpful assistant with memory of past conversations. \
         If you remember previous interactions, acknowledge them.",
    );

    // 4. Configure the runner to use the session.
    //
    // By providing the `session` in the `RunConfig`, we instruct the `Runner`
    // to automatically load the history before each run and save the new
    // items after each run.
    let config = RunConfig {
        max_turns: Some(5),
        stream: false,
        session: Some(session.clone()),
        model_provider: None, // Use default OpenAI provider
        run_context: None,
    };

    // 5. Simulate a multi-turn conversation.
    let queries = [
        "Hello! My name is Alice and I love programming in Rust.",
        "What's my name and what do I like?",
        "Can you recommend some Rust learning resources?",
    ];

    for (i, query) in queries.iter().enumerate() {
        println!("👤 User (Turn {}): {}", i + 1, query);

        let result = Runner::run(agent.clone(), query.to_string(), config.clone()).await?;

        if result.is_success() {
            println!(
                "🤖 Assistant: {}",
                result.final_output.to_string().trim_matches('"')
            );

            // The session is automatically updated by the runner.
            let total_items = session.get_items(None).await?.len();
            println!("  💾 Session now contains {} items\n", total_items);
        } else {
            println!("❌ Error: {:?}", result.error);
        }

        if i < queries.len() - 1 {
            println!("{}", "-".repeat(50));
        }
    }

    // 6. Demonstrate session persistence by summarizing the stored data.
    println!("\n{}", "=".repeat(60));
    println!("📊 Session Summary:");

    let all_items = session.get_items(None).await?;
    println!("  Total items stored: {}", all_items.len());

    let messages = session.get_messages(None).await?;
    println!("  Total messages: {}", messages.len());

    // Show the last few messages to demonstrate the stored history.
    if !messages.is_empty() {
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
    println!("   (Note: This example uses an in-memory DB for demo purposes)");

    Ok(())
}
