//! Integration test verifying runner uses service-based tool path
//!
//! This test ensures the runner properly converts tools to services 
//! without using the deprecated BaseToolService adapter.

use openai_agents_rs::{
    Agent, Runner,
    runner::RunConfig,
    tool::FunctionTool,
    model::MockProvider,
};
use std::sync::Arc;
use serde_json::json;

#[tokio::test] 
async fn test_runner_uses_service_tools() {
    // Create a simple tool that the runner should convert to a service
    let calculator = Arc::new(FunctionTool::simple(
        "add", 
        "Adds two numbers",
        |input: String| {
            if let Some((a, b)) = input.split_once(',') {
                let a: f64 = a.trim().parse().unwrap_or(0.0);
                let b: f64 = b.trim().parse().unwrap_or(0.0);
                format!("{}", a + b)
            } else {
                "Invalid input".to_string()
            }
        }
    ));

    // Create an agent with the tool
    let agent = Agent::simple("Calculator", "I can add numbers")
        .with_tool(calculator.clone());

    // Mock provider that requests a tool call
    let provider = Arc::new(
        MockProvider::new("test-model")
            .with_tool_call("add", json!({"input": "3,4"}))
            .with_message("The result is 7")
    );

    let config = RunConfig {
        max_turns: Some(2),
        model_provider: Some(provider),
        ..RunConfig::default()
    };

    // This should work without any references to BaseToolService
    let result = Runner::run(
        agent, 
        "What is 3 + 4?", 
        config
    ).await;

    assert!(result.is_ok());
    let run_result = result.unwrap();
    
    // Verify we got a successful response 
    // The important part is that this compiles and runs without BaseToolService
    assert!(run_result.is_success());
}

#[tokio::test]
async fn test_runner_multiple_tool_calls() {
    // Test that multiple tool calls work properly with service-based tools
    let add_tool = Arc::new(FunctionTool::simple(
        "add", 
        "Adds two numbers",
        |input: String| {
            if let Some((a, b)) = input.split_once(',') {
                let a: f64 = a.trim().parse().unwrap_or(0.0);
                let b: f64 = b.trim().parse().unwrap_or(0.0);
                format!("{}", a + b)
            } else {
                "Invalid input".to_string()
            }
        }
    ));

    let multiply_tool = Arc::new(FunctionTool::simple(
        "multiply", 
        "Multiplies two numbers",
        |input: String| {
            if let Some((a, b)) = input.split_once(',') {
                let a: f64 = a.trim().parse().unwrap_or(0.0);
                let b: f64 = b.trim().parse().unwrap_or(0.0);
                format!("{}", a * b)
            } else {
                "Invalid input".to_string()
            }
        }
    ));

    let agent = Agent::simple("Calculator", "I can do math")
        .with_tool(add_tool)
        .with_tool(multiply_tool);

    let provider = Arc::new(
        MockProvider::new("test-model")
            .with_tool_call("add", json!({"input": "2,3"}))
            .with_tool_call("multiply", json!({"input": "5,6"}))
            .with_message("2 + 3 = 5 and 5 * 6 = 30")
    );

    let config = RunConfig {
        max_turns: Some(3),
        model_provider: Some(provider),
        ..RunConfig::default()
    };

    // This verifies multiple tool calls work with the service-based tool path
    let result = Runner::run(
        agent,
        "Calculate 2+3 and then 5*6", 
        config
    ).await;

    assert!(result.is_ok());
    assert!(result.unwrap().is_success());
}