//! Tools as Tower Services.
//!
//! This module provides the new architecture where tools directly implement
//! Tower's Service trait, making them first-class citizens in the Tower ecosystem.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use serde_json::Value;
use tower::{Layer, Service};

use crate::env::Env;
use crate::error::Result;
use crate::service::{Effect, ToolRequest, ToolResponse};
use crate::tool::{Tool, ToolResult};

/// A tool that implements Tower Service directly.
///
/// This is the new pattern where tools are services that can be composed
/// with layers just like any other Tower service.
pub struct ServiceTool<F, E = crate::env::DefaultEnv>
where
    E: Env,
{
    name: String,
    description: String,
    parameters_schema: Value,
    function: Arc<F>,
    _phantom: std::marker::PhantomData<E>,
}

impl<F, E> ServiceTool<F, E>
where
    F: Fn(Value) -> Result<Value> + Send + Sync + 'static,
    E: Env,
{
    /// Create a new service-based tool.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters_schema: Value,
        function: F,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters_schema,
            function: Arc::new(function),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create a simple tool that takes and returns strings.
    pub fn simple(
        name: impl Into<String>,
        description: impl Into<String>,
        function: impl Fn(String) -> String + Send + Sync + 'static,
    ) -> ServiceTool<impl Fn(Value) -> Result<Value> + Send + Sync + 'static, E> {
        let wrapped = move |args: Value| {
            let input = args
                .get("input")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let output = function(input);
            Ok(Value::String(output))
        };

        ServiceTool::new(
            name,
            description,
            serde_json::json!({
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "The input string"
                    }
                },
                "required": ["input"]
            }),
            wrapped,
        )
    }

    /// Set a custom name for this tool.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }
}

impl<F, E> Clone for ServiceTool<F, E>
where
    E: Env,
{
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            description: self.description.clone(),
            parameters_schema: self.parameters_schema.clone(),
            function: self.function.clone(),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<F, E> std::fmt::Debug for ServiceTool<F, E>
where
    E: Env,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ServiceTool")
            .field("name", &self.name)
            .field("description", &self.description)
            .finish()
    }
}

#[async_trait::async_trait]
impl<F, E> Tool for ServiceTool<F, E>
where
    F: Fn(Value) -> Result<Value> + Send + Sync + 'static,
    E: Env,
{
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
        let result = (self.function)(arguments)?;
        Ok(ToolResult::success(result))
    }
}

/// Tower Service implementation for ServiceTool.
impl<F, E> Service<ToolRequest<E>> for ServiceTool<F, E>
where
    F: Fn(Value) -> Result<Value> + Send + Sync + 'static,
    E: Env,
{
    type Response = ToolResponse;
    type Error = tower::BoxError;
    type Future =
        Pin<Box<dyn Future<Output = std::result::Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<std::result::Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: ToolRequest<E>) -> Self::Future {
        let function = self.function.clone();
        let tool_name = self.name.clone();

        Box::pin(async move {
            // Use capabilities from the environment if needed
            if let Some(logger) = req.env.capability::<crate::env::LoggingCapability>() {
                use crate::env::Logging;
                logger.debug(&format!("Executing tool: {}", tool_name));
            }

            match function(req.arguments) {
                Ok(result) => Ok(ToolResponse {
                    output: result,
                    error: None,
                    effect: Effect::Continue,
                }),
                Err(e) => Ok(ToolResponse {
                    output: Value::Null,
                    error: Some(e.to_string()),
                    effect: Effect::Continue,
                }),
            }
        })
    }
}

/// A builder for creating tools with layers already applied.
pub struct ToolServiceBuilder<S> {
    service: S,
}

impl<S> ToolServiceBuilder<S> {
    /// Create a new builder with a service.
    pub fn new(service: S) -> Self {
        Self { service }
    }

    /// Apply a layer to the service.
    pub fn layer<L>(self, layer: L) -> ToolServiceBuilder<L::Service>
    where
        L: Layer<S>,
    {
        ToolServiceBuilder {
            service: layer.layer(self.service),
        }
    }

    /// Build the final service.
    pub fn build(self) -> S {
        self.service
    }
}

/// Extension trait to make any tool into a Tower service.
pub trait IntoToolService: Tool {
    /// Convert this tool into a Tower service.
    fn into_service<E: Env>(self) -> ToolServiceAdapter<Self, E>
    where
        Self: Sized,
    {
        ToolServiceAdapter::new(self)
    }
}

impl<T: Tool> IntoToolService for T {}

/// Adapter that makes any Tool implement Service.
pub struct ToolServiceAdapter<T, E = crate::env::DefaultEnv>
where
    T: Tool,
    E: Env,
{
    tool: T,
    _phantom: std::marker::PhantomData<E>,
}

impl<T, E> ToolServiceAdapter<T, E>
where
    T: Tool,
    E: Env,
{
    pub fn new(tool: T) -> Self {
        Self {
            tool,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T, E> Clone for ToolServiceAdapter<T, E>
where
    T: Tool + Clone,
    E: Env,
{
    fn clone(&self) -> Self {
        Self {
            tool: self.tool.clone(),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T, E> Service<ToolRequest<E>> for ToolServiceAdapter<T, E>
where
    T: Tool + Clone + 'static,
    E: Env,
{
    type Response = ToolResponse;
    type Error = tower::BoxError;
    type Future =
        Pin<Box<dyn Future<Output = std::result::Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<std::result::Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: ToolRequest<E>) -> Self::Future {
        let tool = self.tool.clone();

        Box::pin(async move {
            match tool.execute(req.arguments).await {
                Ok(result) => Ok(ToolResponse {
                    output: result.output.clone(),
                    error: result.error,
                    effect: if result.is_final {
                        Effect::Final(result.output)
                    } else {
                        Effect::Continue
                    },
                }),
                Err(e) => Ok(ToolResponse {
                    output: Value::Null,
                    error: Some(e.to_string()),
                    effect: Effect::Continue,
                }),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::DefaultEnv;
    use serde_json::json;

    #[tokio::test]
    async fn test_service_tool() {
        let tool = ServiceTool::<_, DefaultEnv>::simple(
            "uppercase",
            "Converts to uppercase",
            |s: String| s.to_uppercase(),
        );

        let mut service = tool.clone();

        let req = ToolRequest {
            env: DefaultEnv,
            run_id: "test".to_string(),
            agent: "test".to_string(),
            tool_call_id: "1".to_string(),
            tool_name: "uppercase".to_string(),
            arguments: json!({"input": "hello"}),
        };

        let response = service.call(req).await.unwrap();
        assert_eq!(response.output, json!("HELLO"));
        assert_eq!(response.error, None);
    }

    #[tokio::test]
    async fn test_tool_service_adapter() {
        use crate::tool::FunctionTool;

        let tool = FunctionTool::simple("reverse", "Reverses string", |s: String| {
            s.chars().rev().collect()
        });

        let mut service = tool.into_service::<DefaultEnv>();

        let req = ToolRequest {
            env: DefaultEnv,
            run_id: "test".to_string(),
            agent: "test".to_string(),
            tool_call_id: "2".to_string(),
            tool_name: "reverse".to_string(),
            arguments: json!({"input": "hello"}),
        };

        let response = service.call(req).await.unwrap();
        assert_eq!(response.output, json!("olleh"));
        assert_eq!(response.error, None);
    }

    #[test]
    fn test_service_tool_with_layers() {
        use tower::ServiceBuilder;

        let tool = ServiceTool::<_, DefaultEnv>::simple("echo", "Echoes input", |s: String| s);

        // Tools can be composed with Tower layers directly
        let _service = ServiceBuilder::new()
            .timeout(std::time::Duration::from_secs(5))
            .service(tool);
    }
}
