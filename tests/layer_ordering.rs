//! Tests for cross-scope layer ordering verification.
//!
//! These tests use probe layers that record their entry/exit points
//! to verify the canonical execution order: Run → Agent → Tool → Base.

use openai_agents_rs::{
    service::{Effect, ErasedToolLayer, ToolBoxService, ToolRequest, ToolResponse},
    tool::FunctionTool,
    Tool,
};
use serde_json::json;
use std::sync::{Arc, Mutex};
use std::collections::VecDeque;
use tower::{service_fn, Service};

/// Shared probe log to record layer entry/exit
type ProbeLog = Arc<Mutex<VecDeque<String>>>;

/// A probe layer that records when it enters and exits
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

impl ErasedToolLayer for ProbeLayer {
    fn layer_boxed(&self, inner: ToolBoxService) -> ToolBoxService {
        let scope = self.scope.clone();
        let log = self.log.clone();
        let shared = Arc::new(tokio::sync::Mutex::new(inner));

        let service = service_fn(move |req: ToolRequest<openai_agents_rs::env::DefaultEnv>| {
            let scope = scope.clone();
            let log = log.clone();
            let shared = shared.clone();
            
            async move {
                // Record entry
                {
                    let mut log_guard = log.lock().unwrap();
                    log_guard.push_back(format!("{}_enter", scope));
                }

                // Call inner service
                let mut inner = shared.lock().await;
                let result = inner.call(req).await;
                
                // Record exit  
                {
                    let mut log_guard = log.lock().unwrap();
                    log_guard.push_back(format!("{}_exit", scope));
                }

                result
            }
        });

        ToolBoxService::new(service)
    }
}

#[tokio::test] 
async fn test_cross_scope_layer_ordering() {
    // This test directly composes a tool service stack to verify layer ordering
    // without requiring OpenAI API calls
    
    use openai_agents_rs::service::{DefaultEnv, ToolRequest, InputSchemaLayer};
    use openai_agents_rs::tool_service::IntoToolService;
    use tower::{ServiceExt, Layer, util::BoxService};

    // Shared log to capture execution order
    let log: ProbeLog = Arc::new(Mutex::new(VecDeque::new()));

    // Create a simple test tool
    let tool = Arc::new(FunctionTool::simple(
        "test_tool",
        "A test tool",
        |_s: String| "processed".to_string()
    ));

    // Build the tool stack manually like the runner does using service-based approach
    let tool_service = <FunctionTool as Clone>::clone(&tool).into_service::<DefaultEnv>();
    let schema = tool.parameters_schema();
    let base_stack = InputSchemaLayer::lenient(schema).layer(tool_service);
    let mut stack = BoxService::new(base_stack);
    
    // Apply layers in the fixed order: Tool → Agent → Run (inner-to-outer)
    // Tool layers applied first (innermost, closest to base)
    stack = ProbeLayer::new("tool", log.clone()).layer_boxed(stack);
    
    // Agent layers wrap tool layers
    stack = ProbeLayer::new("agent", log.clone()).layer_boxed(stack);
    
    // Run layers wrap everything (outermost)
    stack = ProbeLayer::new("run", log.clone()).layer_boxed(stack);

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

    use openai_agents_rs::service::{DefaultEnv, ToolRequest, InputSchemaLayer};
    use openai_agents_rs::tool_service::IntoToolService;
    use tower::{ServiceExt, Layer, util::BoxService};

    /// A layer that produces a final effect with a specific value
    #[derive(Clone)]
    struct FinalizingLayer {
        final_value: serde_json::Value,
    }

    impl FinalizingLayer {
        fn new(final_value: serde_json::Value) -> Self {
            Self { final_value }
        }
    }

    impl ErasedToolLayer for FinalizingLayer {
        fn layer_boxed(&self, inner: ToolBoxService) -> ToolBoxService {
            let final_value = self.final_value.clone();
            let shared = Arc::new(tokio::sync::Mutex::new(inner));

            let service = service_fn(move |req: ToolRequest<openai_agents_rs::env::DefaultEnv>| {
                let final_value = final_value.clone();
                let shared = shared.clone();
                
                async move {
                    let mut inner = shared.lock().await;
                    let _result = inner.call(req).await?;
                    
                    // Always produce a final effect
                    Ok(ToolResponse {
                        output: final_value.clone(),
                        error: None,
                        effect: Effect::Final(final_value),
                    })
                }
            });

            ToolBoxService::new(service)
        }
    }

    // Create a test tool
    let tool = Arc::new(FunctionTool::simple(
        "finalizing_tool",
        "A tool that gets finalized",
        |_s: String| "base response".to_string()
    ));

    // Build the tool stack manually using service-based approach
    let tool_service = <FunctionTool as Clone>::clone(&tool).into_service::<DefaultEnv>();
    let schema = tool.parameters_schema();
    let base_stack = InputSchemaLayer::lenient(schema).layer(tool_service);
    let mut stack = BoxService::new(base_stack);
    
    // Apply layers in the fixed order: Tool → Agent → Run (inner-to-outer)
    // Tool layer that finalizes (innermost)
    stack = FinalizingLayer::new(json!({"source": "tool_layer"})).layer_boxed(stack);
    
    // Run layer that also finalizes (outermost - should win)
    stack = FinalizingLayer::new(json!({"source": "run_layer"})).layer_boxed(stack);

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