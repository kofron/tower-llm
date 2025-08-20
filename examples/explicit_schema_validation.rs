//! Example demonstrating explicit schema validation with tools.
//! 
//! This shows how tools can explicitly opt into schema validation
//! using Tower layers, following the Step 6 changes where the runner
//! no longer automatically applies schema validation.

use openai_agents_rs::{
    layers, tool::FunctionTool, tool_service::IntoToolService, service::{DefaultEnv, ToolRequest},
    Tool,
};
use serde_json::json;
use std::sync::Arc;
use tower::{Layer, Service};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a tool with strict schema requirements
    let schema_tool = Arc::new(FunctionTool::new(
        "process_user".to_string(),
        "Process user information with strict validation".to_string(),
        json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number", "minimum": 0},
                "email": {"type": "string", "format": "email"}
            },
            "required": ["name", "age"]
        }),
        |args| {
            let name = args["name"].as_str().unwrap_or("Unknown");
            let age = args["age"].as_u64().unwrap_or(0);
            Ok(json!({
                "message": format!("Processed user {} (age {})", name, age),
                "status": "valid"
            }))
        },
    ));

    println!("=== Explicit Schema Validation Examples ===\n");
    
    // Example 1: Tool without schema validation (new default)
    println!("1. Tool without schema validation (accepts invalid args):");
    {
        let mut tool_service = <FunctionTool as Clone>::clone(&schema_tool)
            .into_service::<DefaultEnv>();
            
        let req = ToolRequest::<DefaultEnv> {
            env: DefaultEnv,
            run_id: "example".to_string(),
            agent: "demo".to_string(), 
            tool_call_id: "call1".to_string(),
            tool_name: "process_user".to_string(),
            arguments: json!({"invalid": "data"}), // Invalid according to schema
        };
        
        match tool_service.call(req).await {
            Ok(response) => {
                if response.error.is_some() {
                    println!("   ❌ Tool rejected: {}", response.error.unwrap());
                } else {
                    println!("   ✅ Tool accepted invalid args and processed them");
                }
            },
            Err(e) => println!("   ❌ Service error: {}", e),
        }
    }
    
    // Example 2: Tool with explicit strict validation
    println!("\n2. Tool with explicit STRICT schema validation:");
    {
        let tool_service = <FunctionTool as Clone>::clone(&schema_tool)
            .into_service::<DefaultEnv>();
        let schema = schema_tool.parameters_schema();
        let mut strict_service = layers::InputSchemaLayer::strict(schema)
            .layer(tool_service);
            
        let req = ToolRequest::<DefaultEnv> {
            env: DefaultEnv,
            run_id: "example".to_string(),
            agent: "demo".to_string(),
            tool_call_id: "call2".to_string(),
            tool_name: "process_user".to_string(),
            arguments: json!({"invalid": "data"}), // Invalid according to schema
        };
        
        match strict_service.call(req).await {
            Ok(response) => {
                if response.error.is_some() {
                    println!("   ✅ Strict validation rejected: {}", response.error.unwrap());
                } else {
                    println!("   ❌ Strict validation unexpectedly allowed invalid args");
                }
            },
            Err(e) => println!("   ❌ Service error: {}", e),
        }
    }
    
    // Example 3: Tool with explicit lenient validation
    println!("\n3. Tool with explicit LENIENT schema validation:");
    {
        let tool_service = <FunctionTool as Clone>::clone(&schema_tool)
            .into_service::<DefaultEnv>();
        let schema = schema_tool.parameters_schema();
        let mut lenient_service = layers::InputSchemaLayer::lenient(schema)
            .layer(tool_service);
            
        let req = ToolRequest::<DefaultEnv> {
            env: DefaultEnv,
            run_id: "example".to_string(),
            agent: "demo".to_string(),
            tool_call_id: "call3".to_string(),
            tool_name: "process_user".to_string(),
            arguments: json!({"invalid": "data"}), // Invalid according to schema
        };
        
        match lenient_service.call(req).await {
            Ok(response) => {
                if response.error.is_some() {
                    println!("   ❌ Lenient validation rejected: {}", response.error.unwrap());
                } else {
                    println!("   ✅ Lenient validation allowed invalid args to pass through");
                }
            },
            Err(e) => println!("   ❌ Service error: {}", e),
        }
    }
    
    // Example 4: Tool with valid arguments works with any validation
    println!("\n4. Tool with VALID arguments (works with any validation level):");
    {
        let tool_service = <FunctionTool as Clone>::clone(&schema_tool)
            .into_service::<DefaultEnv>();
        let schema = schema_tool.parameters_schema();  
        let mut strict_service = layers::InputSchemaLayer::strict(schema)
            .layer(tool_service);
            
        let req = ToolRequest::<DefaultEnv> {
            env: DefaultEnv,
            run_id: "example".to_string(),
            agent: "demo".to_string(),
            tool_call_id: "call4".to_string(),
            tool_name: "process_user".to_string(),
            arguments: json!({
                "name": "Alice",
                "age": 30
            }), // Valid according to schema
        };
        
        match strict_service.call(req).await {
            Ok(response) => {
                if response.error.is_some() {
                    println!("   ❌ Unexpected error: {}", response.error.unwrap());
                } else {
                    println!("   ✅ Valid args processed successfully: {}", response.output);
                }
            },
            Err(e) => println!("   ❌ Service error: {}", e),
        }
    }
    
    println!("\n=== Key Takeaways ===");
    println!("• Tools no longer have automatic schema validation (Step 6 completed)");
    println!("• Use layers::InputSchemaLayer::strict() for validation that rejects invalid args");
    println!("• Use layers::InputSchemaLayer::lenient() for validation that warns but allows invalid args"); 
    println!("• No layer = no validation (tools receive raw arguments)");
    println!("• Tools can define their own defaults via Default implementations");
    
    Ok(())
}