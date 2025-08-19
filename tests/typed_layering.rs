//! Tests for typed agent and run config layering APIs.
//!
//! These tests verify that the new fluent `.layer()` APIs work correctly
//! and provide compile-time type safety for layer composition.

use openai_agents_rs::{
    Agent, 
    runner::{RunConfig, RunConfigLike},
    agent::AgentLike,
    service::{TimeoutLayer, RetryLayer, ApprovalLayer, InputSchemaLayer},
    tool::FunctionTool,
};
use serde_json::json;
use std::time::Duration;
use std::sync::Arc;

#[test]
fn test_agent_layer_chaining_compiles() {
    // Test that agent layer chaining compiles with proper types
    let _agent = Agent::simple("TestAgent", "A test agent")
        .layer(TimeoutLayer::from_duration(Duration::from_secs(30)))
        .layer(RetryLayer::times(3))
        .layer(ApprovalLayer);

    // Test that we can access agent properties through the trait
    let layered = Agent::simple("Another", "instructions")
        .layer(TimeoutLayer::secs(10));
    
    assert_eq!(layered.name(), "Another");
    assert_eq!(layered.instructions(), "instructions");
    assert_eq!(layered.tools().len(), 0);
}

#[test]
fn test_runconfig_layer_chaining_compiles() {
    // Test that run config layer chaining compiles with proper types  
    let _config = RunConfig::default()
        .layer(TimeoutLayer::from_duration(Duration::from_secs(45)))
        .layer(RetryLayer::times(2))
        .layer(ApprovalLayer);

    // Test that we can access config properties through the trait
    let layered = RunConfig::default()
        .layer(TimeoutLayer::secs(20));
    
    assert_eq!(layered.max_turns(), Some(10)); // Default is Some(10)
    assert_eq!(layered.stream(), false);
    assert_eq!(layered.parallel_tools(), true);
}

#[test]
fn test_mixed_layer_types() {
    // Test that different layer types can be chained together
    let tool = Arc::new(FunctionTool::simple(
        "test", 
        "test tool", 
        |s: String| s.to_uppercase()
    ));
    
    let _agent = Agent::simple("Mixed", "test")
        .with_tool(tool)
        .layer(InputSchemaLayer::strict(json!({"type": "object"})))
        .layer(TimeoutLayer::secs(15))
        .layer(RetryLayer::times(1))
        .layer(ApprovalLayer);

    // This should compile without issues, demonstrating type safety
}

#[test]
fn test_agent_like_trait_usage() {

    // Test that both Agent and LayeredAgent implement AgentLike
    let base_agent = Agent::simple("Base", "base instructions");
    let layered_agent = base_agent.clone().layer(TimeoutLayer::secs(5));

    // Both should work with AgentLike trait methods
    fn check_agent<A: AgentLike>(agent: &A) -> &str {
        agent.name()
    }

    assert_eq!(check_agent(&base_agent), "Base");
    assert_eq!(check_agent(&layered_agent), "Base");
}

#[test]
fn test_runconfig_like_trait_usage() {

    // Test that both RunConfig and LayeredRunConfig implement RunConfigLike
    let base_config = RunConfig::default();
    let layered_config = base_config.clone().layer(RetryLayer::times(2));

    // Both should work with RunConfigLike trait methods
    fn check_config<R: RunConfigLike>(config: &R) -> Option<usize> {
        config.max_turns()
    }

    assert_eq!(check_config(&base_config), Some(10));
    assert_eq!(check_config(&layered_config), Some(10));
}

#[test] 
fn test_deeply_nested_layers() {
    // Test that many layers can be chained without issues
    let _deeply_layered = Agent::simple("Deep", "deep test")
        .layer(TimeoutLayer::secs(1))
        .layer(RetryLayer::times(1))
        .layer(ApprovalLayer)
        .layer(TimeoutLayer::secs(2))  // Different timeout layer
        .layer(RetryLayer::times(2))   // Different retry layer
        .layer(InputSchemaLayer::lenient(json!({"type": "string"})));

    // If this compiles, the type system is handling nested LayeredAgent types correctly
}