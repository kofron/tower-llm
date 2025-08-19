//! # Tool System for Agents
//!
//! Tools are the primary mechanism by which agents interact with the external
//! world. They provide a structured way to extend an agent's capabilities,
//! allowing it to perform actions, retrieve information, and interface with
//! other systems. This approach aligns with the "functional core, imperative
//! shell" principle, where the agent's reasoning (the functional core) is
//! decoupled from the side effects of its actions (the imperative shell).
//!
//! The [`Tool`] trait defines the common interface for all tools, requiring
//! methods for its name, description, and parameter schema. The [`FunctionTool`]
//! provides a convenient way to create tools from simple Rust functions.
//!
//! ## Creating and Using a Tool
//!
//! The easiest way to create a tool is by using the [`FunctionTool::simple`]
//! method for functions that take a `String` and return a `String`, or the
//! [`FunctionTool::new`] method for more complex signatures.
//!
//! ### Example: A Simple Weather Tool
//!
//! ```rust
//! # use openai_agents_rs::tool::{FunctionTool, Tool, ToolResult};
//! # use openai_agents_rs::error::Result;
//! # use serde_json::json;
//! # use std::sync::Arc;
//!
//! # // Define a simple function to act as our tool's logic.
//! # let get_weather = |location: String| -> String {
//! #     if location.to_lowercase().contains("san francisco") {
//! #         "The weather in San Francisco is 70°F and sunny.".to_string()
//! #     } else {
//! #         format!("I don't have the weather for {}.", location)
//! #     }
//! # };
//!
//! // Create a tool from the function.
//! let weather_tool = Arc::new(FunctionTool::simple(
//!     "get_weather",
//!     "Gets the current weather for a specified location.",
//!     get_weather
//! ));
//!
//! # tokio_test::block_on(async {
//! // Simulate an agent executing the tool.
//! let arguments = json!({"input": "What's the weather in San Francisco?"});
//! let result = weather_tool.execute(arguments).await.unwrap();
//!
//! assert_eq!(
//!     result.output,
//!     json!("The weather in San Francisco is 70°F and sunny.")
//! );
//! # });
//! ```
//!
//! For more complex tools with multiple parameters, you can define a custom
//! schema and use [`FunctionTool::new`].
//!
//! ## Tool Execution
//!
//! When an agent calls a tool, it expects a [`ToolResult`]. A successful
//! execution returns a `ToolResult` with an `output` and `is_final` set to
//! `false`. If the tool's output should be treated as the final response,
//! the `is_final` flag should be set to `true`. If an error occurs during
//! execution, the `error` field in the `ToolResult` will contain a descriptive
//! message.
//!
//! ## Approval Required
//!
//! Some tools might require user approval before execution. Implement the
//! `requires_approval` method to indicate if a tool needs approval. If
//! `requires_approval` returns `true`, the agent will pause and wait for
//! confirmation before executing the tool.
//!
//! ## Tool Call Serialization
//!
//! When communicating with an LLM, tool calls are serialized to JSON. Use
//! [`items::ToolCall`](crate::items::ToolCall) for representing calls (name,
//! arguments, id).
//!
//! ```rust
//! use openai_agents_rs::items::ToolCall;
//! use serde_json::json;
//!
//! let tool_call = ToolCall {
//!     id: "call_123".to_string(),
//!     name: "test_tool".to_string(),
//!     arguments: json!({"key": "value"}),
//! };
//!
//! let serialized = serde_json::to_string(&tool_call).unwrap();
//! let deserialized: ToolCall = serde_json::from_str(&serialized).unwrap();
//!
//! assert_eq!(tool_call.name, deserialized.name);
//! assert_eq!(tool_call.arguments, deserialized.arguments);
//! assert_eq!(tool_call.id, deserialized.id);
//! ```
//!
//! The [`function_tool!`] macro simplifies the creation of `FunctionTool`
//! instances, making it easy to define basic tools.
//!
//! ```rust
//! use openai_agents_rs::{function_tool, tool::Tool};
//!
//! // Create a tool using the macro.
//! let echo_tool = function_tool!(
//!     "echo",
//!     "Echoes the input string back to the user.",
//!     |s: String| format!("You said: {}", s)
//! );
//!
//! assert_eq!(echo_tool.name(), "echo");
//! assert!(echo_tool.description().contains("Echoes the input"));
//! ```
//!
//! The [`Tool`] trait is implemented by [`FunctionTool`], which provides the
//! actual execution logic.
//!
//! ```rust
//! # use openai_agents_rs::tool::{Tool, ToolResult, FunctionTool};
//! # use openai_agents_rs::error::Result;
//! # use serde_json::json;
//! # use std::sync::Arc;
//!
//! // Define a simple function to act as our tool's logic.
//! let get_weather = |args: serde_json::Value| -> Result<serde_json::Value> {
//!     let city = args
//!         .get("city")
//!         .and_then(|v| v.as_str())
//!         .unwrap_or("unknown");
//!     Ok(json!({
//!         "temperature": 22,
//!         "city": city,
//!         "condition": "sunny"
//!     }))
//! };
//!
//! // Create a tool from the function.
//! let weather_tool = Arc::new(FunctionTool::new(
//!     "get_weather".to_string(),
//!     "Get weather for a city".to_string(),
//!     json!({
//!         "type": "object",
//!         "properties": {
//!             "city": {
//!                 "type": "string",
//!                 "description": "The city name"
//!             },
//!             "units": {
//!                 "type": "string",
//!                 "enum": ["celsius", "fahrenheit"],
//!                 "description": "Temperature units"
//!             }
//!         },
//!         "required": ["city"]
//!     }),
//!     get_weather
//! ));
//!
//! # tokio_test::block_on(async {
//! // Simulate an agent executing the tool.
//! let arguments = json!({"city": "San Francisco", "units": "fahrenheit"});
//! let result = weather_tool.execute(arguments).await.unwrap();
//!
//! assert_eq!(
//!     result.output,
//!     json!({
//!         "temperature": 22,
//!         "city": "San Francisco",
//!         "condition": "sunny"
//!     })
//! );
//! # });
//! ```
//!
//! The `Tool` trait is designed to be flexible, allowing for different
//! implementations of tool execution logic.

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::Arc;

use crate::error::Result;

/// Represents the result of a tool's execution.
///
/// A `ToolResult` encapsulates the output of a tool, along with metadata
/// indicating whether the result is final and any errors that occurred. This
/// allows the agent to process the tool's output and decide on the next steps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// The output generated by the tool, represented as a [`serde_json::Value`].
    /// This allows for structured data to be returned from tools.
    pub output: Value,
    /// A flag indicating whether this result should be considered the final
    /// output of the agent's run. If `true`, the agent will stop execution
    /// and return this output.
    pub is_final: bool,
    /// An optional error message that provides details if the tool execution
    /// failed. If `None`, the tool executed successfully.
    pub error: Option<String>,
}

impl ToolResult {
    /// Creates a successful tool result with the given output.
    ///
    /// This is used when the tool executes successfully and the conversation
    /// should continue.
    pub fn success(output: Value) -> Self {
        Self {
            output,
            is_final: false,
            error: None,
        }
    }

    /// Creates a final output result.
    ///
    /// This is used when the tool's output should be treated as the final
    /// response from the agent, terminating the run.
    pub fn final_output(output: Value) -> Self {
        Self {
            output,
            is_final: true,
            error: None,
        }
    }

    /// Creates an error result with a descriptive message.
    ///
    /// This is used when the tool fails to execute, allowing the agent to
    /// handle the error gracefully.
    pub fn error(message: String) -> Self {
        Self {
            output: Value::Null,
            is_final: false,
            error: Some(message),
        }
    }
}

// NOTE: ToolCall definition removed; use crate::items::ToolCall

/// Defines the interface for all tools that can be used by an agent.
///
/// The `Tool` trait provides a common structure for defining external
/// capabilities that an agent can leverage. Implementors of this trait must
/// provide a name, a description, a parameter schema, and the execution logic.
///
/// The `async_trait` macro is used to allow async functions in the trait.
#[async_trait]
pub trait Tool: Send + Sync + Debug {
    /// Returns the name of the tool.
    ///
    /// The name should be a unique identifier for the tool, as it is used by
    /// the agent to select which tool to execute.
    fn name(&self) -> &str;

    /// Returns a description of what the tool does.
    ///
    /// This description is provided to the LLM to help it understand the tool's
    /// purpose and when to use it. A clear and concise description is crucial
    /// for the agent's performance.
    fn description(&self) -> &str;

    /// Returns the JSON schema for the tool's parameters.
    ///
    /// The schema defines the structure of the arguments that the tool expects.
    /// This is used by the LLM to generate the correct arguments for a tool call.
    fn parameters_schema(&self) -> Value;

    /// Executes the tool with the given arguments.
    ///
    /// This method contains the core logic of the tool. It takes the arguments
    /// as a [`serde_json::Value`] and returns a [`ToolResult`].
    async fn execute(&self, arguments: Value) -> Result<ToolResult>;

    /// Indicates whether the tool requires user approval before execution.
    ///
    /// If this method returns `true`, the agent will pause and wait for
    /// confirmation before executing the tool. This is useful for tools that
    /// perform sensitive or irreversible actions.
    fn requires_approval(&self) -> bool {
        false
    }
}

/// A concrete implementation of the [`Tool`] trait that wraps a Rust function.
///
/// `FunctionTool` simplifies the process of creating tools by allowing you to
/// use standard Rust functions as the execution logic. It handles the
/// serialization and deserialization of arguments and results.
///
/// Use [`FunctionTool::simple`] for basic tools with a single string input and
/// output, or [`FunctionTool::new`] for more complex tools with custom parameter
/// schemas.
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
    /// Creates a new `FunctionTool` with a custom parameter schema and function.
    ///
    /// This constructor is suitable for tools that require multiple arguments or
    /// more complex data structures. The provided function should take a
    /// [`serde_json::Value`] as input and return a `Result<Value>`.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the tool.
    /// * `description` - A description of the tool's purpose.
    /// * `parameters_schema` - The JSON schema for the tool's parameters.
    /// * `function` - The function that implements the tool's logic.
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

    /// Creates a simple `FunctionTool` for a function that takes a `String` and
    /// returns a `String`.
    ///
    /// This is a convenience method for creating tools with a single string
    /// input. The parameter schema is automatically generated to expect an
    /// object with a single "input" property.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the tool.
    /// * `description` - A description of the tool's purpose.
    /// * `function` - The function to be wrapped, with a `String -> String` signature.
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
        let func = self.function.clone();
        let args = arguments;
        let join = tokio::task::spawn_blocking(move || (func)(args)).await;
        match join {
            Ok(res) => match res {
                Ok(output) => Ok(ToolResult::success(output)),
                Err(e) => Ok(ToolResult::error(e.to_string())),
            },
            Err(e) => Ok(ToolResult::error(format!(
                "tool panicked or was cancelled: {}",
                e
            ))),
        }
    }
}

/// A typed function tool that (de)serializes inputs/outputs via serde.
///
/// This wrapper allows you to write a strongly-typed tool function and provide
/// an explicit JSON schema for its arguments. Output is serialized back to
/// `serde_json::Value`.
pub struct TypedFunctionTool<I, O, F>
where
    I: serde::de::DeserializeOwned + Send + 'static,
    O: serde::Serialize + Send + 'static,
    F: Fn(I) -> crate::error::Result<O> + Send + Sync + 'static,
{
    name: String,
    description: String,
    schema: Value,
    function: Arc<F>,
    _in: PhantomData<I>,
    _out: PhantomData<O>,
}

impl<I, O, F> std::fmt::Debug for TypedFunctionTool<I, O, F>
where
    I: serde::de::DeserializeOwned + Send + 'static,
    O: serde::Serialize + Send + 'static,
    F: Fn(I) -> crate::error::Result<O> + Send + Sync + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TypedFunctionTool")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("schema", &self.schema)
            .finish()
    }
}

impl<I, O, F> TypedFunctionTool<I, O, F>
where
    I: serde::de::DeserializeOwned + Send + Sync + 'static,
    O: serde::Serialize + Send + Sync + 'static,
    F: Fn(I) -> crate::error::Result<O> + Send + Sync + 'static,
{
    /// Create a new typed tool with an explicit JSON schema for the input type.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        schema: Value,
        f: F,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            schema,
            function: Arc::new(f),
            _in: PhantomData,
            _out: PhantomData,
        }
    }

    /// Create a new typed tool with schema inferred from the input type `I`
    /// using `schemars::JsonSchema`.
    pub fn new_inferred(name: impl Into<String>, description: impl Into<String>, f: F) -> Self
    where
        I: JsonSchema,
    {
        let schema = schemars::schema_for!(I);
        let schema_value = serde_json::to_value(schema.schema)
            .unwrap_or_else(|_| serde_json::json!({"type":"object"}));
        Self {
            name: name.into(),
            description: description.into(),
            schema: schema_value,
            function: Arc::new(f),
            _in: PhantomData,
            _out: PhantomData,
        }
    }
}

#[async_trait]
impl<I, O, F> Tool for TypedFunctionTool<I, O, F>
where
    I: serde::de::DeserializeOwned + Send + Sync + 'static,
    O: serde::Serialize + Send + Sync + 'static,
    F: Fn(I) -> crate::error::Result<O> + Send + Sync + 'static,
{
    fn name(&self) -> &str {
        &self.name
    }
    fn description(&self) -> &str {
        &self.description
    }
    fn parameters_schema(&self) -> Value {
        self.schema.clone()
    }

    async fn execute(&self, arguments: Value) -> Result<ToolResult> {
        let parsed: I = match serde_json::from_value(arguments) {
            Ok(v) => v,
            Err(e) => {
                return Ok(ToolResult::error(format!("invalid arguments: {}", e)));
            }
        };
        match (self.function)(parsed) {
            Ok(out) => match serde_json::to_value(out) {
                Ok(val) => Ok(ToolResult::success(val)),
                Err(e) => Ok(ToolResult::error(format!("serialization error: {}", e))),
            },
            Err(e) => Ok(ToolResult::error(e.to_string())),
        }
    }
}

/// A macro to simplify the creation of a [`FunctionTool`] from a simple function.
///
/// This macro is a shorthand for [`FunctionTool::simple`], making it even easier
/// to define basic tools.
///
/// ## Example
///
/// ```rust
/// use openai_agents_rs::{function_tool, tool::Tool};
///
/// // Create a tool using the macro.
/// let echo_tool = function_tool!(
///     "echo",
///     "Echoes the input string back to the user.",
///     |s: String| format!("You said: {}", s)
/// );
///
/// assert_eq!(echo_tool.name(), "echo");
/// assert!(echo_tool.description().contains("Echoes the input"));
/// ```
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
        let tool_call = crate::items::ToolCall {
            id: "call_123".to_string(),
            name: "test_tool".to_string(),
            arguments: serde_json::json!({"key": "value"}),
        };

        let serialized = serde_json::to_string(&tool_call).unwrap();
        let deserialized: crate::items::ToolCall = serde_json::from_str(&serialized).unwrap();

        assert_eq!(tool_call.name, deserialized.name);
        assert_eq!(tool_call.arguments, deserialized.arguments);
        assert_eq!(tool_call.id, deserialized.id);
    }

    #[tokio::test]
    async fn test_typed_function_tool() {
        #[derive(Deserialize)]
        struct AddArgs {
            x: i32,
            y: i32,
        }
        #[derive(Serialize)]
        struct Sum {
            sum: i32,
        }

        let schema = serde_json::json!({
            "type":"object",
            "properties":{ "x": {"type":"integer"}, "y": {"type":"integer"} },
            "required":["x","y"]
        });
        let tool = TypedFunctionTool::new("add", "Adds two numbers", schema, |a: AddArgs| {
            Ok(Sum { sum: a.x + a.y })
        });
        let args = serde_json::json!({"x": 2, "y": 5});
        let result = tool.execute(args).await.unwrap();
        assert!(result.error.is_none());
        assert_eq!(result.output, serde_json::json!({"sum":7}));
    }

    #[tokio::test]
    async fn test_typed_function_tool_inferred_schema() {
        #[derive(Deserialize, JsonSchema)]
        struct Args {
            v: String,
        }
        #[derive(Serialize)]
        struct Out {
            len: usize,
        }
        let tool = TypedFunctionTool::<Args, Out, _>::new_inferred(
            "len",
            "Returns string length",
            |a: Args| Ok(Out { len: a.v.len() }),
        );
        let args = serde_json::json!({"v":"abc"});
        let res = tool.execute(args).await.unwrap();
        assert!(res.error.is_none());
        assert_eq!(res.output, serde_json::json!({"len":3}));
        let schema = tool.parameters_schema();
        assert!(schema.is_object());
    }

    #[tokio::test]
    async fn test_tool_args_and_output_macros_compile_and_run() {
        use crate::{tool_args, tool_output};
        #[tool_args]
        struct Args {
            v: String,
        }
        #[tool_output]
        struct Out {
            len: usize,
        }

        let tool = TypedFunctionTool::<Args, Out, _>::new_inferred(
            "len",
            "Returns string length",
            |a: Args| Ok(Out { len: a.v.len() }),
        );
        let args = serde_json::json!({"v":"abcd"});
        let res = tool.execute(args).await.unwrap();
        assert!(res.error.is_none());
        assert_eq!(res.output, serde_json::json!({"len":4}));
        let schema = tool.parameters_schema();
        assert!(schema.is_object());
    }
}
