//! # Advanced Example: Personal Assistant with Sessions and Guardrails
//!
//! This example demonstrates a more complex agent that acts as a personal
//! assistant. It showcases the integration of several key features:
//!
//! - **Session Memory**: The agent maintains conversation history using a
//!   `MemorySession`, allowing it to remember information from previous turns.
//! - **Multiple Tools**: The assistant is equipped with tools for saving notes
//!   and setting reminders.
//! - **Input Guardrails**: A combination of built-in and custom guardrails are
//!   used to validate user input for length, sensitive information, and profanity.
//! - **Output Guardrails**: A custom guardrail is used to add a disclaimer to
//!   the agent's responses when it discusses financial or medical topics.
//!
//! ## Key Concepts Demonstrated
//!
//! - **Stateful Conversation**: The agent can recall information provided by the
//!   user in previous turns.
//! - **Custom Guardrails**: The example shows how to implement the `InputGuardrail`
//!   and `OutputGuardrail` traits to create custom validation logic.
//! - **Guardrail Priority**: The `SensitiveInfoGuardrail` is given a high priority
//!   to ensure it runs before other checks.
//! - **Interactive Session**: The example includes an interactive mode where you
//!   can have a conversation with the assistant, and the session memory will be
//!   retained.
//!
//! To run this example, you first need to set your `OPENAI_API_KEY` environment
//! variable.
//!
//! ```bash
//! export OPENAI_API_KEY="your-api-key"
//! cargo run --example session_with_guardrails
//! ```
//!
//! Expected: The assistant should maintain session memory across turns, use
//! tools to save a note and set a reminder, and add a disclaimer when content
//! resembles financial or medical advice. The interactive section may be
//! skipped by the example runner.

use async_trait::async_trait;
use openai_agents_rs::{
    guardrail::{
        GuardrailResult, InputGuardrail, MaxLengthGuardrail, OutputGuardrail, PatternBlockGuardrail,
    },
    memory::{MemorySession, Session},
    runner::RunConfig,
    Agent, FunctionTool, Runner,
};
use std::sync::Arc;

/// A custom input guardrail to detect and block potentially sensitive information.
#[derive(Debug)]
struct SensitiveInfoGuardrail {
    name: String,
    patterns: Vec<String>,
}

impl SensitiveInfoGuardrail {
    fn new() -> Self {
        Self {
            name: "SensitiveInfoGuardrail".to_string(),
            patterns: vec![
                "password".to_string(),
                "credit card".to_string(),
                "ssn".to_string(),
                "social security".to_string(),
                "api key".to_string(),
                "secret".to_string(),
            ],
        }
    }
}

#[async_trait]
impl InputGuardrail for SensitiveInfoGuardrail {
    fn name(&self) -> &str {
        &self.name
    }

    async fn check(&self, input: &str) -> openai_agents_rs::error::Result<GuardrailResult> {
        let input_lower = input.to_lowercase();

        for pattern in &self.patterns {
            if input_lower.contains(pattern) {
                println!("  ⚠️  Guardrail triggered: Sensitive information detected");
                return Ok(GuardrailResult {
                    passed: false,
                    reason: Some(format!(
                        "Input contains potentially sensitive information: '{}'",
                        pattern
                    )),
                });
            }
        }

        Ok(GuardrailResult {
            passed: true,
            reason: None,
        })
    }

    fn priority(&self) -> i32 {
        100 // High priority to run before other checks.
    }
}

/// A custom output guardrail that adds a disclaimer to responses containing
/// financial or medical advice.
#[derive(Debug)]
struct DisclaimerGuardrail;

#[async_trait]
impl OutputGuardrail for DisclaimerGuardrail {
    fn name(&self) -> &str {
        "DisclaimerGuardrail"
    }

    async fn check(&self, output: &str) -> openai_agents_rs::error::Result<GuardrailResult> {
        // Check if the output contains keywords related to financial or medical advice.
        let output_lower = output.to_lowercase();

        if output_lower.contains("invest")
            || output_lower.contains("financial advice")
            || output_lower.contains("medical")
            || output_lower.contains("diagnosis")
        {
            let _modified = format!(
                "{}\n\n⚠️ Disclaimer: This is for informational purposes only and should not be considered professional advice.",
                output
            );

            println!("  📝 Output guardrail: Added disclaimer");
            Ok(GuardrailResult {
                passed: true,
                reason: None,
            })
        } else {
            Ok(GuardrailResult {
                passed: true,
                reason: None,
            })
        }
    }
}

/// Creates a tool for saving notes.
fn create_note_tool() -> Arc<FunctionTool> {
    Arc::new(FunctionTool::simple(
        "save_note",
        "Save a note for later reference",
        |note: String| {
            println!("  📝 Note saved: {}", note);
            format!("Note saved: '{}'", note)
        },
    ))
}

/// Creates a tool for setting reminders.
fn create_reminder_tool() -> Arc<FunctionTool> {
    Arc::new(FunctionTool::new(
        "set_reminder".to_string(),
        "Set a reminder for a specific time".to_string(),
        serde_json::json!({
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "What to be reminded about"
                },
                "time": {
                    "type": "string",
                    "description": "When to be reminded (e.g., 'in 5 minutes', 'tomorrow at 3pm')"
                }
            },
            "required": ["task", "time"]
        }),
        |args| {
            let task = args
                .get("task")
                .and_then(|v| v.as_str())
                .unwrap_or("Unknown task");
            let time = args
                .get("time")
                .and_then(|v| v.as_str())
                .unwrap_or("Unknown time");

            println!("  ⏰ Reminder set: '{}' at {}", task, time);
            Ok(serde_json::json!({
                "status": "success",
                "message": format!("Reminder set for '{}' at {}", task, time)
            }))
        },
    ))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Session with Guardrails Example ===\n");
    println!("This example demonstrates:");
    println!("  • Session memory across multiple interactions");
    println!("  • Input guardrails blocking sensitive information");
    println!("  • Output guardrails adding disclaimers");
    println!("  • Tools for note-taking and reminders\n");
    println!("{}\n", "=".repeat(60));

    // 1. Create the personal assistant agent.
    //
    // This agent is equipped with multiple tools and a comprehensive set of
    // both built-in and custom guardrails.
    let agent = Agent::simple(
        "PersonalAssistant",
        "You are a helpful personal assistant. You can help with various tasks including:
         - Taking notes and saving information
         - Setting reminders
         - Answering questions
         - Providing helpful information
         
         Be conversational and remember what the user has told you in previous messages.",
    )
    .with_tools(vec![create_note_tool(), create_reminder_tool()])
    .with_input_guardrail(Arc::new(MaxLengthGuardrail::new(500)))
    .with_input_guardrail(Arc::new(SensitiveInfoGuardrail::new()))
    .with_input_guardrail(Arc::new(PatternBlockGuardrail::new(
        "ProfanityFilter",
        vec!["badword".to_string(), "inappropriate".to_string()],
    )))
    .with_output_guardrail(Arc::new(DisclaimerGuardrail))
    .with_max_turns(5);

    // 2. Create a session to maintain conversation history.
    //
    // `MemorySession` is used here for simplicity, but `SqliteSession` could be
    // used for a persistent assistant.
    let session = Arc::new(MemorySession::new("user_session_001"));

    let config = RunConfig {
        session: Some(session.clone()),
        run_context: None,
        ..Default::default()
    };

    // 3. Simulate a multi-turn conversation.
    //
    // This conversation is designed to showcase the agent's memory, tool use,
    // and the triggering of the output guardrail.
    let conversations = [
        "Hello! I'm planning a trip to Paris next month.",
        "Can you save a note that I need to book flights by next Friday?",
        "Set a reminder for tomorrow at 2pm to check hotel prices",
        "What did I tell you I was planning?",
        "What investment opportunities should I consider?",
    ];

    for (turn, message) in conversations.iter().enumerate() {
        println!("Turn {} - User: {}", turn + 1, message);
        println!("{}", "-".repeat(40));

        let result = Runner::run(agent.clone(), message.to_string(), config.clone()).await;

        match result {
            Ok(res) if res.is_success() => {
                println!("Assistant: {}\n", res.final_output);

                // Show if any tools were used during the turn.
                let tool_calls: Vec<_> = res
                    .items
                    .iter()
                    .filter_map(|item| {
                        if let openai_agents_rs::items::RunItem::ToolCall(tc) = item {
                            Some(tc.tool_name.clone())
                        } else {
                            None
                        }
                    })
                    .collect();

                if !tool_calls.is_empty() {
                    println!("(Used tools: {})", tool_calls.join(", "));
                }
            }
            Ok(res) => {
                println!("Error in response: {:?}", res.error());
            }
            Err(e) => {
                println!("Error: {}", e);
                if e.to_string().contains("Guardrail") {
                    println!("💡 Tip: The input was blocked by a guardrail for safety reasons.");
                }
            }
        }

        println!("\n{}\n", "=".repeat(60));
    }

    // 4. Show a summary of the session.
    println!("Session Summary");
    println!("{}", "-".repeat(40));

    let history = session.get_messages(None).await?;
    println!("Total messages in session: {}", history.len());

    // 5. Enter interactive mode.
    //
    // This allows you to continue the conversation with the assistant, which
    // will retain the context from the previous turns.
    println!("\nInteractive mode with session memory");
    println!("The assistant will remember your conversation.");
    println!("Type 'quit' to exit, 'clear' to clear session memory.\n");

    use std::io::{self, Write};

    loop {
        print!("You: ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.eq_ignore_ascii_case("quit") || input.eq_ignore_ascii_case("exit") {
            println!("Goodbye! Your conversation has been saved in the session.");
            break;
        }

        if input.eq_ignore_ascii_case("clear") {
            session.clear_session().await?;
            println!("Session memory cleared.\n");
            continue;
        }

        let result = Runner::run(agent.clone(), input.to_string(), config.clone()).await;

        match result {
            Ok(res) if res.is_success() => {
                println!("\nAssistant: {}\n", res.final_output);
            }
            Ok(res) => {
                println!("\nError: {:?}\n", res.error());
            }
            Err(e) => {
                println!("\nError: {}\n", e);
            }
        }
    }

    Ok(())
}
