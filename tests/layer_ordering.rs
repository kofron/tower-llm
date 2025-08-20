//! Tests for cross-scope layer ordering verification.
//!
//! These tests use probe layers that record their entry/exit points
//! to verify the canonical execution order: Run → Agent → Tool → Base.
//!
//! Step 8 completed: ErasedToolLayer removed, using Tower layers directly

use openai_agents_rs::{
    service::{Effect, ToolRequest, ToolResponse, DefaultEnv, InputSchemaLayer},
    tool::FunctionTool,
    tool_service::IntoToolService,
    Tool,
};
use serde_json::json;
use std::sync::{Arc, Mutex};
use std::collections::VecDeque;
use tower::{Layer, Service, util::BoxService, BoxError, ServiceExt};
use std::future::Future;
use std::pin::Pin;

/// Shared probe log to record layer entry/exit
type ProbeLog = Arc<Mutex<VecDeque<String>>>;

/// A probe layer that records when it enters and exits using Tower Layer trait
#[derive(Clone)]
struct ProbeLayer {
    scope: String,
    log: ProbeLog,
}

impl ProbeLayer {
    fn new(scope: impl Into<String>, log: ProbeLog) -> Self {
        Self {
            scope: scope.into(),
            log,
        }
    }
}

impl<S> Layer<S> for ProbeLayer {
    type Service = ProbeService<S>;
    fn layer(&self, inner: S) -> Self::Service {
        ProbeService {
            inner,
            scope: self.scope.clone(),
            log: self.log.clone(),
        }
    }
}

#[derive(Clone)]
struct ProbeService<S> {
    inner: S,
    scope: String,
    log: ProbeLog,
}

impl<S> Service<ToolRequest<DefaultEnv>> for ProbeService<S>
where
    S: Service<ToolRequest<DefaultEnv>, Response = ToolResponse, Error = BoxError> + Clone + Send + 'static,
    S::Future: Send + 'static,
{
    type Response = ToolResponse;
    type Error = BoxError;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(
        &mut self,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: ToolRequest<DefaultEnv>) -> Self::Future {
        let mut inner = self.inner.clone();
        let scope = self.scope.clone();
        let log = self.log.clone();
        
        Box::pin(async move {
            // Record entry
            {
                let mut log_guard = log.lock().unwrap();
                log_guard.push_back(format!("{}_enter", scope));
            }

            // Call inner service
            let result = inner.call(req).await;
            
            // Record exit  
            {
                let mut log_guard = log.lock().unwrap();
                log_guard.push_back(format!("{}_exit", scope));
            }

            result
        })
    }
}

#[tokio::test] 
async fn test_cross_scope_layer_ordering() {
    // This test directly composes a tool service stack to verify layer ordering
    // without requiring OpenAI API calls
    
    // Shared log to capture execution order
    let log: ProbeLog = Arc::new(Mutex::new(VecDeque::new()));

    // Create a simple test tool
    let tool = Arc::new(FunctionTool::simple(
        "test_tool",
        "A test tool",
        |_s: String| "processed".to_string()
    ));

    // Build the tool stack manually like the runner does using service-based approach
    let service = <FunctionTool as Clone>::clone(&tool).into_service::<DefaultEnv>();
    let schema = tool.parameters_schema();
    let base_stack = InputSchemaLayer::lenient(schema).layer(service);
    
    // Apply layers in the fixed order: Tool → Agent → Run (inner-to-outer)
    // Tool layers applied first (innermost, closest to base)
    let tool_layered = ProbeLayer::new("tool", log.clone()).layer(base_stack);
    
    // Agent layers wrap tool layers
    let agent_layered = ProbeLayer::new("agent", log.clone()).layer(tool_layered);
    
    // Run layers wrap everything (outermost)
    let mut stack = BoxService::new(ProbeLayer::new("run", log.clone()).layer(agent_layered));

    // Create a test request
    let req = ToolRequest::<DefaultEnv> {
        env: DefaultEnv,
        run_id: "test_run".to_string(),
        agent: "test_agent".to_string(),
        tool_call_id: "test_call".to_string(),
        tool_name: "test_tool".to_string(),
        arguments: json!({"input": "test"}),
    };

    // Execute the stack
    let result = stack.oneshot(req).await;
    assert!(result.is_ok(), "Tool stack should execute successfully: {:?}", result);

    // Check the execution order
    let execution_log = {
        let log_guard = log.lock().unwrap();
        log_guard.clone().into_iter().collect::<Vec<_>>()
    };

    // Expected order: Run (outermost) → Agent → Tool → Base → Tool exit → Agent exit → Run exit
    let expected_order = vec![
        "run_enter",
        "agent_enter", 
        "tool_enter",
        "tool_exit",
        "agent_exit",
        "run_exit"
    ];

    assert_eq!(
        execution_log, expected_order,
        "Layer execution order should be Run → Agent → Tool → Base. Got: {:?}",
        execution_log
    );
}

#[tokio::test]
async fn test_final_effect_precedence() {
    // Test that run-scope finalization takes precedence over tool-scope
    // by directly composing tool stacks

    /// A layer that produces a final effect with a specific value using Tower Layer trait
    #[derive(Clone)]
    struct FinalizingLayer {
        final_value: serde_json::Value,
    }

    impl FinalizingLayer {
        fn new(final_value: serde_json::Value) -> Self {
            Self { final_value }
        }
    }

    impl<S> Layer<S> for FinalizingLayer {
        type Service = FinalizingService<S>;
        fn layer(&self, inner: S) -> Self::Service {
            FinalizingService {
                inner,
                final_value: self.final_value.clone(),
            }
        }
    }

    #[derive(Clone)]
    struct FinalizingService<S> {
        inner: S,
        final_value: serde_json::Value,
    }

    impl<S> Service<ToolRequest<DefaultEnv>> for FinalizingService<S>
    where
        S: Service<ToolRequest<DefaultEnv>, Response = ToolResponse, Error = BoxError> + Clone + Send + 'static,
        S::Future: Send + 'static,
    {
        type Response = ToolResponse;
        type Error = BoxError;
        type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

        fn poll_ready(
            &mut self,
            cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Result<(), Self::Error>> {
            self.inner.poll_ready(cx)
        }

        fn call(&mut self, req: ToolRequest<DefaultEnv>) -> Self::Future {
            let mut inner = self.inner.clone();
            let final_value = self.final_value.clone();
            
            Box::pin(async move {
                let _result = inner.call(req).await?;
                
                // Always produce a final effect
                Ok(ToolResponse {
                    output: final_value.clone(),
                    error: None,
                    effect: Effect::Final(final_value),
                })
            })
        }
    }

    // Create a test tool
    let tool = Arc::new(FunctionTool::simple(
        "finalizing_tool",
        "A tool that gets finalized",
        |_s: String| "base response".to_string()
    ));

    // Build the tool stack manually using service-based approach
    let service = <FunctionTool as Clone>::clone(&tool).into_service::<DefaultEnv>();
    let schema = tool.parameters_schema();
    let base_stack = InputSchemaLayer::lenient(schema).layer(service);
    
    // Apply layers in the fixed order: Tool → Agent → Run (inner-to-outer)
    // Tool layer that finalizes (innermost)
    let tool_layered = FinalizingLayer::new(json!({"source": "tool_layer"})).layer(base_stack);
    
    // Run layer that also finalizes (outermost - should win)
    let mut stack = BoxService::new(FinalizingLayer::new(json!({"source": "run_layer"})).layer(tool_layered));

    // Create a test request
    let req = ToolRequest::<DefaultEnv> {
        env: DefaultEnv,
        run_id: "test_run".to_string(),
        agent: "test_agent".to_string(),
        tool_call_id: "test_call".to_string(),
        tool_name: "finalizing_tool".to_string(),
        arguments: json!({"input": "test"}),
    };

    // Execute the stack
    let result = stack.oneshot(req).await;
    assert!(result.is_ok(), "Tool stack should execute successfully: {:?}", result);
    
    let response = result.unwrap();
    
    // The run-scope layer should win because it's outermost
    assert_eq!(
        response.output,
        json!({"source": "run_layer"}),
        "Run-scope finalization should take precedence over tool-scope"
    );
    
    // Verify it's a final effect
    assert!(matches!(response.effect, Effect::Final(_)), "Response should have final effect");
}