//! Tests demonstrating the new Tower-based patterns for tools and layers.
//! 
//! Step 8 completed: LayeredTool removed, tools now use uniform Tower service composition

use openai_agents_rs::{layers, Agent, FunctionTool, Tool, tool_service::IntoToolService, service::DefaultEnv};
use std::sync::Arc;
use tower::Layer;

#[test]
fn test_tool_with_service_layers() {
    // Create a tool and compose it with layers via Tower services
    let tool = FunctionTool::simple("uppercase", "Converts to uppercase", |s: String| {
        s.to_uppercase()
    });

    // Tools now compose via .into_service().layer() for uniform Tower patterns
    let _service = tool.clone().into_service::<DefaultEnv>();
    let _layered = layers::TimeoutLayer::secs(5).layer(
        layers::RetryLayer::times(3).layer(_service)
    );

    // Original tool interface is preserved
    assert_eq!(tool.name(), "uppercase");
    assert_eq!(tool.description(), "Converts to uppercase");
    
    // Service composition creates typed Tower services
    // (Service trait methods would be tested via integration tests)
}

#[test]
fn test_tool_with_custom_name() {
    // Create a tool with a custom name
    let tool = FunctionTool::simple("original", "A tool", |s: String| s).with_name("custom_name");

    assert_eq!(tool.name(), "custom_name");
}

#[test]
fn test_service_tool_preserves_interface() {
    // Tools maintain their interface when converted to services
    let base_tool = FunctionTool::simple("test", "Test tool", |s: String| s);
    
    // Original tool interface is preserved
    assert_eq!(base_tool.name(), "test");
    assert_eq!(base_tool.description(), "Test tool");

    // Tool can still be used as Arc<dyn Tool> in agents
    let _tool_arc: Arc<dyn Tool> = Arc::new(base_tool.clone());
    
    // And can be composed as a service with typed layers
    let _service = base_tool.into_service::<DefaultEnv>();
    let _layered = layers::TimeoutLayer::secs(10).layer(_service);
}

#[test]
fn test_tool_default_implementation() {
    // Test that FunctionTool implements Default
    let tool = FunctionTool::default();
    assert_eq!(tool.name(), "example");
    assert_eq!(tool.description(), "An example tool");
}

#[test]
fn test_agent_with_service_based_tools() {
    // Tools are now used directly in agents - layering happens via services
    let tool1 = FunctionTool::simple("tool1", "First tool", |s: String| s);
    let tool2 = FunctionTool::simple("tool2", "Second tool", |s: String| s.to_uppercase());

    // Agent composes tools without configuring their layers
    // Layer configuration happens when tools are converted to services
    let agent = Agent::simple("TestAgent", "A test agent")
        .with_tool(Arc::new(tool1))
        .with_tool(Arc::new(tool2));

    // Verify the agent has both tools
    assert_eq!(agent.tools().len(), 2);
    assert_eq!(agent.name(), "TestAgent");
    
    // Tools can be individually layered when converted to services:
    // let service1 = tool1.into_service::<DefaultEnv>().layer(TimeoutLayer::secs(5));
    // let service2 = tool2.into_service::<DefaultEnv>().layer(RetryLayer::times(3));
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
    .with_name("user_db");
    
    // Service-based layering would be:
    // let _service = db_tool.clone().into_service::<DefaultEnv>()
    //     .layer(TimeoutLayer::secs(30))
    //     .layer(RetryLayer::times(2));

    // Agent manages its own configuration (would add agent layers here)
    let agent = Agent::simple("DataAgent", "Handles data operations")
        .with_tool(Arc::new(db_tool));

    // Clean boundaries - each level manages itself
    assert_eq!(agent.tools().len(), 1);
    assert_eq!(agent.tools()[0].name(), "user_db");
}
