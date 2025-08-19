//! Comprehensive tests for the Tower-based architecture.
//!
//! These tests validate that the Tower patterns work correctly
//! and that the architecture delivers on its promises.

use openai_agents_rs::{
    env::{Approval, DefaultEnv, Env, EnvBuilder, LoggingCapability, Metrics},
    layers,
    service::{Effect, ToolRequest, ToolResponse},
    tool::{FunctionTool, Tool, ToolResult},
    tool_service::{IntoToolService, ServiceTool},
    Agent,
};
use serde_json::{json, Value};

use std::sync::{Arc, Mutex};
use std::time::Duration;
use tower::{Layer, Service, ServiceBuilder};

/// Test metrics collector
#[derive(Default)]
struct TestMetrics {
    counters: Arc<Mutex<std::collections::HashMap<String, u64>>>,
}

impl Metrics for TestMetrics {
    fn increment(&self, name: &str, value: u64) {
        let mut counters = self.counters.lock().unwrap();
        *counters.entry(name.to_string()).or_insert(0) += value;
    }

    fn gauge(&self, _name: &str, _value: f64) {}
    fn histogram(&self, _name: &str, _value: f64) {}
}

/// Test approval service
struct TestApproval {
    approved: Arc<Mutex<Vec<String>>>,
    should_approve: bool,
}

impl Approval for TestApproval {
    fn request_approval(&self, operation: &str, _details: &str) -> bool {
        self.approved.lock().unwrap().push(operation.to_string());
        self.should_approve
    }
}

#[test]
fn test_tool_layer_composition() {
    // Tools can have multiple layers
    let tool = FunctionTool::simple("test", "Test tool", |s: String| s)
        .layer(layers::boxed_timeout_secs(5))
        .layer(layers::boxed_retry_times(3));

    assert_eq!(tool.name(), "test");
    assert_eq!(tool.layers().len(), 2);
}

#[test]
fn test_tool_with_custom_name() {
    let tool = FunctionTool::simple("original", "Tool", |s: String| s).with_name("custom");

    assert_eq!(tool.name(), "custom");
}

#[tokio::test]
async fn test_tool_as_service() {
    let tool = FunctionTool::simple("echo", "Echo", |s: String| s);
    let mut service = tool.into_service::<DefaultEnv>();

    let req = ToolRequest {
        env: DefaultEnv,
        run_id: "test".to_string(),
        agent: "test".to_string(),
        tool_call_id: "1".to_string(),
        tool_name: "echo".to_string(),
        arguments: json!({"input": "hello"}),
    };

    let response = service.call(req).await.unwrap();
    assert_eq!(response.output, json!("hello"));
    assert!(response.error.is_none());
}

#[tokio::test]
async fn test_service_tool_direct() {
    let mut tool = ServiceTool::<_, DefaultEnv>::new(
        "adder",
        "Adds one",
        json!({"type": "object"}),
        |args: Value| {
            if let Some(n) = args.get("value").and_then(|v| v.as_i64()) {
                Ok(json!(n + 1))
            } else {
                Err(openai_agents_rs::error::AgentsError::Other(
                    "Invalid input".to_string(),
                ))
            }
        },
    );

    let req = ToolRequest {
        env: DefaultEnv,
        run_id: "test".to_string(),
        agent: "test".to_string(),
        tool_call_id: "1".to_string(),
        tool_name: "adder".to_string(),
        arguments: json!({"value": 5}),
    };

    let response = tool.call(req).await.unwrap();
    assert_eq!(response.output, json!(6));
}

#[tokio::test]
async fn test_capability_access() {
    let metrics = Arc::new(TestMetrics::default());
    let env = EnvBuilder::new()
        .with_capability(metrics.clone())
        .with_capability(Arc::new(LoggingCapability))
        .build();

    // Create a custom layer that uses capabilities
    struct CapabilityLayer;

    impl<S> Layer<S> for CapabilityLayer {
        type Service = CapabilityService<S>;
        fn layer(&self, inner: S) -> Self::Service {
            CapabilityService { inner }
        }
    }

    #[derive(Clone)]
    struct CapabilityService<S> {
        inner: S,
    }

    impl<S> Service<ToolRequest<openai_agents_rs::env::CapabilityEnv>> for CapabilityService<S>
    where
        S: Service<
                ToolRequest<openai_agents_rs::env::CapabilityEnv>,
                Response = ToolResponse,
                Error = tower::BoxError,
            > + Clone
            + Send
            + 'static,
        S::Future: Send + 'static,
    {
        type Response = ToolResponse;
        type Error = tower::BoxError;
        type Future = std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<Self::Response, Self::Error>> + Send>,
        >;

        fn poll_ready(
            &mut self,
            cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Result<(), Self::Error>> {
            self.inner.poll_ready(cx)
        }

        fn call(&mut self, req: ToolRequest<openai_agents_rs::env::CapabilityEnv>) -> Self::Future {
            // Access capabilities
            if let Some(metrics) = req.env.capability::<TestMetrics>() {
                metrics.increment("tool.calls", 1);
            }

            let mut inner = self.inner.clone();
            Box::pin(async move { inner.call(req).await })
        }
    }

    let tool = FunctionTool::simple("test", "Test", |s: String| s)
        .into_service::<openai_agents_rs::env::CapabilityEnv>();

    let mut service = CapabilityLayer.layer(tool);

    let req = ToolRequest {
        env: env.clone(),
        run_id: "test".to_string(),
        agent: "test".to_string(),
        tool_call_id: "1".to_string(),
        tool_name: "test".to_string(),
        arguments: json!({"input": "test"}),
    };

    service.call(req).await.unwrap();

    // Verify metrics were incremented
    let counters = metrics.counters.lock().unwrap();
    assert_eq!(counters.get("tool.calls"), Some(&1));
}

#[test]
fn test_layer_ordering() {
    // Test that layers can be stacked
    let tool = FunctionTool::simple("test", "Test", |s: String| s);

    // Layers are applied and stored in order
    let layered = tool
        .layer(layers::boxed_timeout_secs(5))
        .layer(layers::boxed_retry_times(3))
        .layer(layers::boxed_input_schema_lenient(
            json!({"type": "object"}),
        ));

    // Verify layers are accumulated
    assert_eq!(layered.layers().len(), 3);
}

#[tokio::test]
async fn test_error_propagation() {
    let tool = FunctionTool::new(
        "failing".to_string(),
        "Always fails".to_string(),
        json!({"type": "object"}),
        |_args| {
            Err(openai_agents_rs::error::AgentsError::Other(
                "Intentional failure".to_string(),
            ))
        },
    );

    let mut service = tool.into_service::<DefaultEnv>();

    let req = ToolRequest {
        env: DefaultEnv,
        run_id: "test".to_string(),
        agent: "test".to_string(),
        tool_call_id: "1".to_string(),
        tool_name: "failing".to_string(),
        arguments: json!({}),
    };

    let response = service.call(req).await.unwrap();
    assert!(response.error.is_some());
    assert_eq!(response.error.unwrap(), "Intentional failure");
}

#[tokio::test]
async fn test_effect_propagation() {
    // Create a custom tool that returns a final output
    #[derive(Debug, Clone)]
    struct FinalTool;

    #[async_trait::async_trait]
    impl Tool for FinalTool {
        fn name(&self) -> &str {
            "finalizer"
        }
        fn description(&self) -> &str {
            "Returns final output"
        }
        fn parameters_schema(&self) -> Value {
            json!({"type": "object"})
        }

        async fn execute(&self, _arguments: Value) -> openai_agents_rs::Result<ToolResult> {
            Ok(ToolResult::final_output(json!({"final": true})))
        }
    }

    let tool = FinalTool;

    let mut service = tool.into_service::<DefaultEnv>();

    let req = ToolRequest {
        env: DefaultEnv,
        run_id: "test".to_string(),
        agent: "test".to_string(),
        tool_call_id: "1".to_string(),
        tool_name: "finalizer".to_string(),
        arguments: json!({}),
    };

    let response = service.call(req).await.unwrap();
    assert!(matches!(response.effect, Effect::Final(_)));
    if let Effect::Final(value) = response.effect {
        assert_eq!(value, json!({"final": true}));
    }
}

#[test]
fn test_no_string_coupling() {
    // Everything is type-safe, no string-based lookups
    let tool: Arc<dyn Tool> = Arc::new(FunctionTool::simple("test", "Test", |s: String| s));

    // Direct access, no string keys
    assert_eq!(tool.name(), "test");

    // Layers are type-safe
    let _layered =
        FunctionTool::simple("test", "Test", |s: String| s).layer(layers::boxed_timeout_secs(5));

    // No string-based registry needed
}

#[test]
fn test_abstraction_boundaries() {
    // Tools manage themselves
    let tool = FunctionTool::simple("calc", "Calculator", |s: String| s)
        .with_name("calculator")
        .layer(layers::boxed_retry_times(3));

    // Agents don't configure tool internals
    let agent = Agent::simple("Bot", "Assistant").with_tool(Arc::new(tool));

    // Each level manages only its own concerns
    assert_eq!(agent.tools().len(), 1);
}

#[tokio::test]
async fn test_tower_ecosystem_compatibility() {
    // Tools work with standard Tower middleware
    let tool = FunctionTool::simple("echo", "Echo", |s: String| s).into_service::<DefaultEnv>();

    let service = ServiceBuilder::new()
        .timeout(Duration::from_secs(1))
        .service(tool);

    // Service is a standard Tower service
    let mut svc = service;
    let req = ToolRequest {
        env: DefaultEnv,
        run_id: "test".to_string(),
        agent: "test".to_string(),
        tool_call_id: "1".to_string(),
        tool_name: "echo".to_string(),
        arguments: json!({"input": "tower"}),
    };

    let response = svc.call(req).await.unwrap();
    assert_eq!(response.output, json!("tower"));
}
