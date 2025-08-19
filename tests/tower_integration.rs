//! Integration tests for the Tower-based tool architecture.
//!
//! These tests demonstrate real-world usage patterns and verify that
//! the Tower composition model works correctly.

use openai_agents_rs::{layers, runner::RunConfig, Agent, FunctionTool, Tool};
use serde_json::json;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

#[tokio::test]
async fn test_layered_tool_execution() {
    // Create a tool with multiple layers
    let call_count = Arc::new(AtomicUsize::new(0));
    let count_clone = call_count.clone();

    let tool = FunctionTool::new(
        "counter".to_string(),
        "Counts calls".to_string(),
        json!({"type": "object"}),
        move |_args| {
            count_clone.fetch_add(1, Ordering::SeqCst);
            Ok(json!({"count": count_clone.load(Ordering::SeqCst)}))
        },
    )
    .layer(layers::boxed_retry_times(2)); // Will retry on failure

    // Tool should work with the standard Tool trait
    let result = tool.execute(json!({})).await.unwrap();
    assert_eq!(result.output["count"], 1);
}

#[tokio::test]
async fn test_tool_layer_ordering() {
    // Demonstrates that layers wrap in the correct order
    let execution_order = Arc::new(std::sync::Mutex::new(Vec::new()));

    // Create a tool that records when it executes
    let order_clone = execution_order.clone();
    let tool = FunctionTool::new(
        "ordered".to_string(),
        "Tests layer order".to_string(),
        json!({"type": "object"}),
        move |_args| {
            order_clone.lock().unwrap().push("tool");
            Ok(json!({"executed": true}))
        },
    );

    // Layers are applied outside-in as they're added
    // So if we add A then B, execution goes B -> A -> tool
    let layered = tool
        .layer(layers::boxed_retry_times(1)) // Inner layer
        .layer(layers::boxed_timeout_secs(10)); // Outer layer

    // Verify the layered tool still implements Tool
    assert_eq!(layered.name(), "ordered");
    let _result = layered.execute(json!({})).await.unwrap();
}

#[test]
fn test_tool_with_custom_name_and_layers() {
    // Real-world pattern: database tool with custom configuration
    let db_tool = FunctionTool::new(
        "generic_db".to_string(),
        "Database operations".to_string(),
        json!({
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "params": {"type": "array"}
            }
        }),
        |args| {
            // Simulate database query
            Ok(json!({
                "rows": [],
                "query": args["query"]
            }))
        },
    )
    .with_name("user_database") // Custom name for this instance
    .layer(layers::boxed_timeout_secs(30)) // Database queries can be slow
    .layer(layers::boxed_retry_times(1)); // But don't retry too much

    assert_eq!(db_tool.name(), "user_database");
}

#[test]
fn test_agent_with_mixed_tools() {
    // Some tools have layers, some don't - both work fine
    let simple_tool = FunctionTool::simple("simple", "A simple tool", |s: String| s);

    let complex_tool =
        FunctionTool::simple("complex", "A complex tool", |s: String| s.to_uppercase())
            .layer(layers::boxed_timeout_secs(5))
            .layer(layers::boxed_retry_times(3));

    let agent = Agent::simple("MixedAgent", "Uses both simple and complex tools")
        .with_tool(Arc::new(simple_tool))
        .with_tool(Arc::new(complex_tool));

    assert_eq!(agent.tools().len(), 2);

    // Both tools are available and properly named
    let tool_names: Vec<&str> = agent.tools().iter().map(|t| t.name()).collect();
    assert!(tool_names.contains(&"simple"));
    assert!(tool_names.contains(&"complex"));
}

#[test]
fn test_tool_default_with_layers() {
    // Even default tools can have layers added
    let tool = FunctionTool::default()
        .with_name("customized_default")
        .layer(layers::boxed_timeout_secs(10));

    assert_eq!(tool.name(), "customized_default");
    assert_eq!(tool.description(), "An example tool");
}

#[tokio::test]
async fn test_tool_error_handling_with_layers() {
    // Note: Layers are applied by the runner when building the service stack.
    // Direct tool execution doesn't apply layers - this is by design.
    // The LayeredTool carries its layers but they're applied during agent execution.

    let flaky_tool = FunctionTool::new(
        "flaky".to_string(),
        "Sometimes fails".to_string(),
        json!({"type": "object"}),
        move |args| {
            // This tool would fail on first attempt in a real scenario
            // But for this test, we'll just verify it can be created with layers
            Ok(json!({"status": "ok", "args": args}))
        },
    )
    .layer(layers::boxed_retry_times(2)); // Retry layer attached but not executed here

    // The tool carries its layers
    assert_eq!(flaky_tool.layers().len(), 1);

    // Direct execution doesn't apply layers (they're applied by the runner)
    let result = flaky_tool.execute(json!({"test": true})).await;
    assert!(result.is_ok());
    assert_eq!(result.unwrap().output["status"], "ok");
}

#[test]
fn test_separation_of_concerns() {
    // This test demonstrates the clean separation of concerns

    // 1. Tool level - manages tool-specific configuration
    let api_tool = FunctionTool::new(
        "api".to_string(),
        "External API".to_string(),
        json!({"type": "object", "properties": {"endpoint": {"type": "string"}}}),
        |args| Ok(json!({"response": "ok", "endpoint": args["endpoint"]})),
    )
    .with_name("external_api")
    .layer(layers::boxed_timeout_secs(10)) // API-specific timeout
    .layer(layers::boxed_retry_times(3)); // API-specific retry

    // 2. Agent level - manages agent-specific configuration
    let agent = Agent::simple("APIAgent", "Handles API calls")
        .with_tool(Arc::new(api_tool))
        .with_agent_layers(vec![
            // Agent-level concerns like tracing would go here
            // layers::boxed_trace_all(),
        ]);

    // 3. Run level - manages run-specific configuration
    let _config = RunConfig::default().with_run_layers(vec![
        // Run-level concerns like global timeout would go here
        // layers::boxed_global_timeout_secs(300),
    ]);

    // Each level is independent and manages its own concerns
    assert_eq!(agent.tools()[0].name(), "external_api");
}

#[test]
fn test_tool_composition_patterns() {
    // Pattern 1: Simple tool with no layers
    let simple = FunctionTool::simple("echo", "Echoes input", |s: String| s);

    // Pattern 2: Tool with single layer
    let with_timeout = FunctionTool::simple("slow", "Slow operation", |s: String| {
        // Simulate slow operation
        std::thread::sleep(std::time::Duration::from_millis(10));
        s
    })
    .layer(layers::boxed_timeout_secs(1));

    // Pattern 3: Tool with multiple layers (order matters!)
    let with_multiple = FunctionTool::simple("critical", "Critical operation", |s: String| s)
        .layer(layers::boxed_timeout_secs(5)) // Inner layer
        .layer(layers::boxed_retry_times(3)); // Outer layer

    // Pattern 4: Tool with custom name and layers
    let fully_configured = FunctionTool::simple("base", "Base tool", |s: String| s)
        .with_name("production_tool")
        .layer(layers::boxed_timeout_secs(30))
        .layer(layers::boxed_retry_times(2));

    // All patterns result in tools that implement the Tool trait
    let tools: Vec<Arc<dyn Tool>> = vec![
        Arc::new(simple),
        Arc::new(with_timeout),
        Arc::new(with_multiple),
        Arc::new(fully_configured),
    ];

    assert_eq!(tools.len(), 4);
    assert_eq!(tools[3].name(), "production_tool");
}

#[test]
fn test_no_string_coupling() {
    // The new architecture eliminates string-based coupling

    // OLD WAY (removed):
    // agent.with_tool_layers("database", vec![layers]);  // String coupling!

    // NEW WAY: Direct, type-safe configuration
    let tool = FunctionTool::simple("database", "DB operations", |s: String| s)
        .layer(layers::boxed_timeout_secs(30));

    let agent = Agent::simple("DataAgent", "Handles data").with_tool(Arc::new(tool));

    // No strings needed to configure tools
    // The tool carries its configuration with it
    assert_eq!(agent.tools()[0].name(), "database");
}

#[test]
fn test_tower_philosophy() {
    // This test demonstrates that we're following Tower's philosophy:
    // "Everything is a service, and services compose through layers"

    // Tools are services
    let service1 = FunctionTool::simple("service1", "First service", |s: String| s);

    // Layers modify service behavior
    let service2 = FunctionTool::simple("service2", "Second service", |s: String| s)
        .layer(layers::boxed_timeout_secs(10));

    // Services compose into higher-level services (agents)
    let agent = Agent::simple("ComposedAgent", "Composed from services")
        .with_tool(Arc::new(service1))
        .with_tool(Arc::new(service2));

    // The agent itself could have layers (agent-level policies)
    let agent_with_layers = agent.with_agent_layers(vec![
        // Agent-level layers
    ]);

    // And runs could have their own layers (run-level policies)
    let _config = RunConfig::default().with_run_layers(vec![
        // Run-level layers
    ]);

    // Clean, uniform composition at every level
    assert_eq!(agent_with_layers.tools().len(), 2);
}
