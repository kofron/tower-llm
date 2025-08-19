//! Tests demonstrating the new Tower-based patterns for tools and layers.

use openai_agents_rs::{layers, Agent, FunctionTool, Tool};
use std::sync::Arc;

#[test]
fn test_tool_with_layers() {
    // Create a tool and add layers to it
    let tool = FunctionTool::simple("uppercase", "Converts to uppercase", |s: String| {
        s.to_uppercase()
    })
    .layer(layers::boxed_timeout_secs(5))
    .layer(layers::boxed_retry_times(3));

    // Verify the tool has the expected name
    assert_eq!(tool.name(), "uppercase");
    assert_eq!(tool.description(), "Converts to uppercase");
}

#[test]
fn test_tool_with_custom_name() {
    // Create a tool with a custom name
    let tool = FunctionTool::simple("original", "A tool", |s: String| s).with_name("custom_name");

    assert_eq!(tool.name(), "custom_name");
}

#[test]
fn test_layered_tool_preserves_interface() {
    // Create a layered tool
    let base_tool = FunctionTool::simple("test", "Test tool", |s: String| s);
    let layered = base_tool.layer(layers::boxed_timeout_secs(10));

    // LayeredTool implements Tool trait
    assert_eq!(layered.name(), "test");
    assert_eq!(layered.description(), "Test tool");

    // Can be used as Arc<dyn Tool>
    let _tool_arc: Arc<dyn Tool> = Arc::new(layered);
}

#[test]
fn test_tool_default_implementation() {
    // Test that FunctionTool implements Default
    let tool = FunctionTool::default();
    assert_eq!(tool.name(), "example");
    assert_eq!(tool.description(), "An example tool");
}

#[test]
fn test_agent_with_layered_tools() {
    // Create tools with their own layers
    let tool1 = FunctionTool::simple("tool1", "First tool", |s: String| s)
        .layer(layers::boxed_timeout_secs(5));

    let tool2 = FunctionTool::simple("tool2", "Second tool", |s: String| s.to_uppercase())
        .layer(layers::boxed_retry_times(3));

    // Agent just composes tools, doesn't configure them
    let agent = Agent::simple("TestAgent", "A test agent")
        .with_tool(Arc::new(tool1))
        .with_tool(Arc::new(tool2));

    // Verify the agent has both tools
    assert_eq!(agent.tools().len(), 2);
    assert_eq!(agent.name(), "TestAgent");
}

#[test]
fn test_clean_separation_of_concerns() {
    // Tools manage their own configuration
    let db_tool = FunctionTool::new(
        "database".to_string(),
        "Database operations".to_string(),
        serde_json::json!({"type": "object"}),
        |_args| Ok(serde_json::json!({"result": "success"})),
    )
    .with_name("user_db")
    .layer(layers::boxed_timeout_secs(30))
    .layer(layers::boxed_retry_times(2));

    // Agent manages its own configuration (would add agent layers here)
    let agent = Agent::simple("DataAgent", "Handles data operations")
        .with_tool(Arc::new(db_tool));

    // Clean boundaries - each level manages itself
    assert_eq!(agent.tools().len(), 1);
    assert_eq!(agent.tools()[0].name(), "user_db");
}
