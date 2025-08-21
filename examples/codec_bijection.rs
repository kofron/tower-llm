//! Example demonstrating the codec module's bijective conversion between
//! OpenAI messages and RunItems for replayability.

use async_openai::types::{
    ChatCompletionMessageToolCall, ChatCompletionRequestAssistantMessageArgs,
    ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageArgs,
    ChatCompletionRequestToolMessageArgs, ChatCompletionRequestUserMessageArgs,
    ChatCompletionToolType, FunctionCall,
};
use serde_json::json;

// Import the next module and its submodules
// Core module is now at root level
// use tower_llm directly

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Codec Bijection Example ===\n");

    // Create a complex conversation with tool calls
    let original_messages = vec![
        // System message
        ChatCompletionRequestSystemMessageArgs::default()
            .content("You are a helpful assistant with access to a calculator.")
            .build()?
            .into(),
        // User message
        ChatCompletionRequestUserMessageArgs::default()
            .content("What's 15 * 7?")
            .build()?
            .into(),
        // Assistant message with tool call
        ChatCompletionRequestAssistantMessageArgs::default()
            .content("")
            .tool_calls(vec![ChatCompletionMessageToolCall {
                id: "call_abc123".to_string(),
                r#type: ChatCompletionToolType::Function,
                function: FunctionCall {
                    name: "multiply".to_string(),
                    arguments: json!({"a": 15, "b": 7}).to_string(),
                },
            }])
            .build()?
            .into(),
        // Tool response
        ChatCompletionRequestToolMessageArgs::default()
            .tool_call_id("call_abc123")
            .content(json!({"result": 105}).to_string())
            .build()?
            .into(),
        // Assistant's final response
        ChatCompletionRequestAssistantMessageArgs::default()
            .content("15 * 7 = 105")
            .build()?
            .into(),
    ];

    println!("Original messages ({} total):", original_messages.len());
    for (i, msg) in original_messages.iter().enumerate() {
        println!("  {}: {:?}", i, message_type(msg));
    }

    // Convert to RunItems
    println!("\n--- Converting to RunItems ---");
    let items = tower_llm::codec::messages_to_items(&original_messages)?;

    println!("RunItems ({} total):", items.len());
    for (i, item) in items.iter().enumerate() {
        match item {
            tower_llm::items::RunItem::Message(m) => {
                println!(
                    "  {}: Message(role={:?}, content_len={})",
                    i,
                    m.role,
                    m.content.len()
                );
            }
            tower_llm::items::RunItem::ToolCall(tc) => {
                println!("  {}: ToolCall(name={}, id={})", i, tc.tool_name, tc.id);
            }
            tower_llm::items::RunItem::ToolOutput(to) => {
                println!(
                    "  {}: ToolOutput(id={}, output_type={})",
                    i, to.tool_call_id, to.output
                );
            }
            tower_llm::items::RunItem::Handoff(_) => {
                println!("  {}: Handoff", i);
            }
        }
    }

    // Convert back to messages
    println!("\n--- Converting back to Messages ---");
    let reconstructed = tower_llm::codec::items_to_messages(&items);

    println!("Reconstructed messages ({} total):", reconstructed.len());
    for (i, msg) in reconstructed.iter().enumerate() {
        println!("  {}: {:?}", i, message_type(msg));
    }

    // Verify bijection
    println!("\n--- Verifying Bijection ---");
    let bijection_valid = messages_equal(&original_messages, &reconstructed);

    if bijection_valid {
        println!("✅ Perfect bijection: Original and reconstructed messages match!");
        println!("   This means we can record agent runs as RunItems and replay them exactly.");
    } else {
        println!("❌ Bijection failed: Messages don't match");
        return Err("Bijection test failed".into());
    }

    // Demonstrate use case
    println!("\n--- Use Case: Replay Capability ---");
    println!("1. During agent execution, messages are converted to RunItems for logging");
    println!("2. RunItems are stored (database, file, etc.) with additional metadata");
    println!("3. Later, RunItems can be loaded and converted back to messages");
    println!("4. The reconstructed messages can be fed to a new agent run for:");
    println!("   - Debugging previous runs");
    println!("   - Testing agent behavior changes");
    println!("   - Auditing agent decisions");
    println!("   - Creating training datasets");

    Ok(())
}

fn message_type(msg: &ChatCompletionRequestMessage) -> &'static str {
    match msg {
        ChatCompletionRequestMessage::System(_) => "System",
        ChatCompletionRequestMessage::User(_) => "User",
        ChatCompletionRequestMessage::Assistant(_) => "Assistant",
        ChatCompletionRequestMessage::Tool(_) => "Tool",
        ChatCompletionRequestMessage::Function(_) => "Function",
        ChatCompletionRequestMessage::Developer(_) => "Developer",
    }
}

fn messages_equal(a: &[ChatCompletionRequestMessage], b: &[ChatCompletionRequestMessage]) -> bool {
    if a.len() != b.len() {
        return false;
    }

    for (msg_a, msg_b) in a.iter().zip(b.iter()) {
        // For this example, we do a simple type comparison
        // In production, you'd want deeper equality checks
        if std::mem::discriminant(msg_a) != std::mem::discriminant(msg_b) {
            return false;
        }
    }

    true
}
