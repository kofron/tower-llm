//! Tool system for agents
//!
//! Tools are the primary way agents interact with the external world.
//! Following the "functional core, imperative shell" principle.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fmt::Debug;
use std::sync::Arc;

use crate::error::Result;

/// Result from a tool execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// The output from the tool
    pub output: Value,
    /// Whether this result should be considered the final output
    pub is_final: bool,
    /// Optional error message if the tool failed
    pub error: Option<String>,
}

impl ToolResult {
    /// Create a successful tool result
    pub fn success(output: Value) -> Self {
        Self {
            output,
            is_final: false,
            error: None,
        }
    }

    /// Create a final output result
    pub fn final_output(output: Value) -> Self {
        Self {
            output,
            is_final: true,
            error: None,
        }
    }

    /// Create an error result
    pub fn error(message: String) -> Self {
        Self {
            output: Value::Null,
            is_final: false,
            error: Some(message),
        }
    }
}

/// Parameters for a tool call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Name of the tool to call
    pub name: String,
    /// Arguments to pass to the tool
    pub arguments: Value,
    /// Unique ID for this tool call
    pub id: String,
}

/// Trait for all tools that can be used by agents
#[async_trait]
pub trait Tool: Send + Sync + Debug {
    /// Get the name of the tool
    fn name(&self) -> &str;

    /// Get the description of the tool
    fn description(&self) -> &str;

    /// Get the JSON schema for the tool's parameters
    fn parameters_schema(&self) -> Value;

    /// Execute the tool with the given arguments
    async fn execute(&self, arguments: Value) -> Result<ToolResult>;

    /// Whether this tool requires approval before execution
    fn requires_approval(&self) -> bool {
        false
    }
}

/// A function-based tool
#[derive(Clone)]
pub struct FunctionTool {
    name: String,
    description: String,
    parameters_schema: Value,
    function: Arc<dyn Fn(Value) -> Result<Value> + Send + Sync>,
}

impl std::fmt::Debug for FunctionTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FunctionTool")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("parameters_schema", &self.parameters_schema)
            .finish()
    }
}

impl FunctionTool {
    /// Create a new function tool
    pub fn new<F>(name: String, description: String, parameters_schema: Value, function: F) -> Self
    where
        F: Fn(Value) -> Result<Value> + Send + Sync + 'static,
    {
        Self {
            name,
            description,
            parameters_schema,
            function: Arc::new(function),
        }
    }

    /// Create a function tool with a simple string-to-string function
    pub fn simple<F>(name: &str, description: &str, function: F) -> Self
    where
        F: Fn(String) -> String + Send + Sync + 'static,
    {
        let name = name.to_string();
        let wrapped = move |args: Value| {
            let input = args
                .get("input")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let output = function(input);
            Ok(Value::String(output))
        };

        Self {
            name: name.clone(),
            description: description.to_string(),
            parameters_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "Input to the function"
                    }
                },
                "required": ["input"]
            }),
            function: Arc::new(wrapped),
        }
    }
}

#[async_trait]
impl Tool for FunctionTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters_schema(&self) -> Value {
        self.parameters_schema.clone()
    }

    async fn execute(&self, arguments: Value) -> Result<ToolResult> {
        match (self.function)(arguments) {
            Ok(output) => Ok(ToolResult::success(output)),
            Err(e) => Ok(ToolResult::error(e.to_string())),
        }
    }
}

/// Macro to create a function tool from a Rust function
#[macro_export]
macro_rules! function_tool {
    ($name:expr, $description:expr, $func:expr) => {
        $crate::tool::FunctionTool::simple($name, $description, $func)
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_tool_result_creation() {
        let result = ToolResult::success(serde_json::json!({"data": "test"}));
        assert!(!result.is_final);
        assert!(result.error.is_none());
        assert_eq!(result.output, serde_json::json!({"data": "test"}));

        let final_result = ToolResult::final_output(serde_json::json!("done"));
        assert!(final_result.is_final);
        assert!(final_result.error.is_none());

        let error_result = ToolResult::error("Something went wrong".to_string());
        assert!(!error_result.is_final);
        assert_eq!(error_result.error, Some("Something went wrong".to_string()));
    }

    #[test]
    fn test_function_tool_simple() {
        let tool = FunctionTool::simple("uppercase", "Converts text to uppercase", |s: String| {
            s.to_uppercase()
        });

        assert_eq!(tool.name(), "uppercase");
        assert_eq!(tool.description(), "Converts text to uppercase");

        let schema = tool.parameters_schema();
        assert!(schema.is_object());
        assert_eq!(schema["type"], "object");
    }

    #[tokio::test]
    async fn test_function_tool_execution() {
        let tool = FunctionTool::simple("reverse", "Reverses a string", |s: String| {
            s.chars().rev().collect()
        });

        let args = serde_json::json!({"input": "hello"});
        let result = tool.execute(args).await.unwrap();

        assert_eq!(result.output, Value::String("olleh".to_string()));
        assert!(!result.is_final);
        assert!(result.error.is_none());
    }

    #[test]
    fn test_function_tool_with_complex_schema() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city name"
                },
                "units": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature units"
                }
            },
            "required": ["city"]
        });

        let tool = FunctionTool::new(
            "get_weather".to_string(),
            "Get weather for a city".to_string(),
            schema.clone(),
            |args| {
                let city = args
                    .get("city")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");
                Ok(serde_json::json!({
                    "temperature": 22,
                    "city": city,
                    "condition": "sunny"
                }))
            },
        );

        assert_eq!(tool.parameters_schema(), schema);
    }

    #[tokio::test]
    async fn test_function_tool_error_handling() {
        let tool = FunctionTool::new(
            "failing_tool".to_string(),
            "A tool that fails".to_string(),
            serde_json::json!({}),
            |_| {
                Err(crate::error::AgentsError::ToolExecutionError {
                    message: "Intentional failure".to_string(),
                })
            },
        );

        let result = tool.execute(serde_json::json!({})).await.unwrap();
        assert!(result.error.is_some());
        assert!(result.error.unwrap().contains("Intentional failure"));
    }

    #[test]
    fn test_function_tool_macro() {
        let tool = function_tool!("echo", "Echoes the input", |s: String| format!(
            "Echo: {}",
            s
        ));

        assert_eq!(tool.name(), "echo");
        assert_eq!(tool.description(), "Echoes the input");
    }

    #[test]
    fn test_tool_call_serialization() {
        let tool_call = ToolCall {
            name: "test_tool".to_string(),
            arguments: serde_json::json!({"key": "value"}),
            id: "call_123".to_string(),
        };

        let serialized = serde_json::to_string(&tool_call).unwrap();
        let deserialized: ToolCall = serde_json::from_str(&serialized).unwrap();

        assert_eq!(tool_call.name, deserialized.name);
        assert_eq!(tool_call.arguments, deserialized.arguments);
        assert_eq!(tool_call.id, deserialized.id);
    }
}
