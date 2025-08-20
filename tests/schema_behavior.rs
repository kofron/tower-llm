//! Tests to verify schema validation behavior is not enforced by default
//! and only applied when explicitly layered (Step 6 verification).

use openai_agents_rs::{
    service::{DefaultEnv, InputSchemaLayer, ToolRequest},
    tool::FunctionTool,
    tool_service::IntoToolService,
    Tool,
};
use serde_json::json;
use std::sync::Arc;
use tower::{Layer, Service};

#[tokio::test]
async fn test_schema_not_enforced_without_layer() {
    // Test that tools without schema layers accept invalid arguments
    // This verifies Step 6: runner no longer injects schema validation by default
    
    let tool = Arc::new(FunctionTool::new(
        "strict_tool".to_string(),
        "A tool with strict schema requirements".to_string(),
        json!({
            "type": "object",
            "properties": {
                "required_field": {"type": "string"}
            },
            "required": ["required_field"]
        }),
        |args| {
            // This should receive the raw arguments, even if they're invalid
            Ok(json!({
                "received": args,
                "status": "processed_without_validation"
            }))
        },
    ));
    
    // Create tool service WITHOUT schema layer (this is the new default)
    let mut tool_service = <FunctionTool as Clone>::clone(&tool).into_service::<DefaultEnv>();
    
    // Create a request with invalid arguments (missing required_field)
    let req = ToolRequest::<DefaultEnv> {
        env: DefaultEnv,
        run_id: "test_run".to_string(),
        agent: "test_agent".to_string(),
        tool_call_id: "test_call".to_string(),
        tool_name: "strict_tool".to_string(),
        arguments: json!({"wrong_field": "value"}), // Invalid according to schema
    };
    
    // Execute without schema validation
    let result = tool_service.call(req).await;
    assert!(result.is_ok(), "Tool should execute without schema validation: {:?}", result);
    
    let response = result.unwrap();
    assert!(response.error.is_none(), "Tool should not error without schema validation");
    
    // Verify the tool received the invalid arguments
    assert_eq!(
        response.output["received"]["wrong_field"], 
        "value",
        "Tool should receive raw arguments without validation"
    );
    assert_eq!(
        response.output["status"], 
        "processed_without_validation",
        "Tool should process without validation"
    );
}

#[tokio::test]
async fn test_schema_enforced_with_strict_layer() {
    // Test that when schema layer is explicitly added, validation is enforced
    
    let tool = Arc::new(FunctionTool::new(
        "validated_tool".to_string(),
        "A tool with validation".to_string(),
        json!({
            "type": "object",
            "properties": {
                "required_field": {"type": "string"}
            },
            "required": ["required_field"]
        }),
        |_args| {
            // This should not be called if validation fails
            Ok(json!({"status": "validation_passed"}))
        },
    ));
    
    // Create tool service WITH strict schema layer
    let tool_service = <FunctionTool as Clone>::clone(&tool).into_service::<DefaultEnv>();
    let schema = tool.parameters_schema();
    let mut layered_service = InputSchemaLayer::strict(schema).layer(tool_service);
    
    // Create a request with invalid arguments
    let req = ToolRequest::<DefaultEnv> {
        env: DefaultEnv,
        run_id: "test_run".to_string(),
        agent: "test_agent".to_string(),
        tool_call_id: "test_call".to_string(),
        tool_name: "validated_tool".to_string(),
        arguments: json!({"wrong_field": "value"}), // Invalid according to schema
    };
    
    // Execute with strict schema validation
    let result = layered_service.call(req).await;
    assert!(result.is_ok(), "Service should return Ok with error payload: {:?}", result);
    
    let response = result.unwrap();
    assert!(response.error.is_some(), "Schema validation should produce an error");
    assert!(
        response.error.unwrap().contains("schema validation failed"),
        "Error should indicate schema validation failure"
    );
}

#[tokio::test]
async fn test_schema_enforced_with_lenient_layer() {
    // Test that lenient schema layer allows invalid arguments to pass through
    
    let tool = Arc::new(FunctionTool::new(
        "lenient_tool".to_string(),
        "A tool with lenient validation".to_string(),
        json!({
            "type": "object",
            "properties": {
                "required_field": {"type": "string"}
            },
            "required": ["required_field"]
        }),
        |args| {
            Ok(json!({
                "received": args,
                "status": "processed_with_lenient_validation"
            }))
        },
    ));
    
    // Create tool service WITH lenient schema layer
    let tool_service = <FunctionTool as Clone>::clone(&tool).into_service::<DefaultEnv>();
    let schema = tool.parameters_schema();
    let mut layered_service = InputSchemaLayer::lenient(schema).layer(tool_service);
    
    // Create a request with invalid arguments
    let req = ToolRequest::<DefaultEnv> {
        env: DefaultEnv,
        run_id: "test_run".to_string(),
        agent: "test_agent".to_string(),
        tool_call_id: "test_call".to_string(),
        tool_name: "lenient_tool".to_string(),
        arguments: json!({"wrong_field": "value"}), // Invalid according to schema
    };
    
    // Execute with lenient schema validation
    let result = layered_service.call(req).await;
    assert!(result.is_ok(), "Lenient validation should allow invalid args: {:?}", result);
    
    let response = result.unwrap();
    assert!(response.error.is_none(), "Lenient validation should not produce errors");
    assert_eq!(
        response.output["status"],
        "processed_with_lenient_validation",
        "Tool should execute with lenient validation"
    );
}

#[tokio::test]
async fn test_valid_arguments_work_with_and_without_validation() {
    // Test that valid arguments work both with and without schema validation
    
    let tool = Arc::new(FunctionTool::new(
        "flexible_tool".to_string(),
        "A tool that works with or without validation".to_string(),
        json!({
            "type": "object",
            "properties": {
                "input": {"type": "string"}
            },
            "required": ["input"]
        }),
        |args| {
            Ok(json!({
                "processed": args["input"],
                "status": "success"
            }))
        },
    ));
    
    let valid_req = ToolRequest::<DefaultEnv> {
        env: DefaultEnv,
        run_id: "test_run".to_string(),
        agent: "test_agent".to_string(),
        tool_call_id: "test_call".to_string(),
        tool_name: "flexible_tool".to_string(),
        arguments: json!({"input": "valid_value"}), // Valid according to schema
    };
    
    // Test without schema validation
    let mut unvalidated_service = <FunctionTool as Clone>::clone(&tool).into_service::<DefaultEnv>();
    let result1 = unvalidated_service.call(valid_req.clone()).await.unwrap();
    assert!(result1.error.is_none());
    assert_eq!(result1.output["processed"], "valid_value");
    
    // Test with strict schema validation
    let validated_service = <FunctionTool as Clone>::clone(&tool).into_service::<DefaultEnv>();
    let schema = tool.parameters_schema();
    let mut strict_service = InputSchemaLayer::strict(schema).layer(validated_service);
    let result2 = strict_service.call(valid_req.clone()).await.unwrap();
    assert!(result2.error.is_none());
    assert_eq!(result2.output["processed"], "valid_value");
    
    // Both should produce the same successful result
    assert_eq!(result1.output, result2.output);
}