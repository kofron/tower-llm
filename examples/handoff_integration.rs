//! Integration example showing handoff system with existing Tower layers.
//!
//! This example demonstrates:
//! - Handoff coordination with full Tower ecosystem
//! - Integration with policies, budgets, resilience, and observability
//! - Production-ready multi-agent system with complete middleware stack

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use async_openai::types::{ChatCompletionRequestMessage, CreateChatCompletionRequest};
use serde_json::json;
use tower::{Service, ServiceExt, Layer};
use tower_llm::groups::{
    AgentName, AgentPicker, HandoffCoordinator, ExplicitHandoffPolicy
};

// Content creation agent
#[derive(Clone)]
struct ContentAgent;

impl Service<CreateChatCompletionRequest> for ContentAgent {
    type Response = tower_llm::StepOutcome;
    type Error = tower::BoxError;
    type Future = std::pin::Pin<Box<dyn std::future::Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, _: &mut std::task::Context<'_>) -> std::task::Poll<Result<(), Self::Error>> {
        std::task::Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: CreateChatCompletionRequest) -> Self::Future {
        Box::pin(async move {
            let content = req.messages.last()
                .and_then(|m| m.content.as_ref())
                .unwrap_or("");

            if content.contains("review") || content.contains("feedback") || content.contains("check") {
                return Ok(tower_llm::StepOutcome::Next {
                    messages: vec![ChatCompletionRequestMessage {
                        role: async_openai::types::Role::Assistant,
                        content: Some("I've drafted the content:\n\n# Getting Started with Our API\n\nOur REST API provides programmatic access to your data...\n\nLet me hand this off to our editor for review and refinement.".to_string()),
                        name: None,
                        tool_calls: Some(vec![async_openai::types::ChatCompletionMessageToolCall {
                            id: "content_review".to_string(),
                            r#type: async_openai::types::ChatCompletionMessageToolCallType::Function,
                            function: async_openai::types::FunctionCall {
                                name: "handoff_to_editor".to_string(),
                                arguments: json!({"reason": "Content ready for editorial review"}).to_string(),
                            },
                        }]),
                        tool_call_id: None,
                    }],
                    aux: tower_llm::StepAux {
                        prompt_tokens: 200,
                        completion_tokens: 100,
                        tool_invocations: 1,
                    },
                    invoked_tools: vec![],
                });
            }

            Ok(tower_llm::StepOutcome::Done {
                messages: vec![ChatCompletionRequestMessage {
                    role: async_openai::types::Role::Assistant,
                    content: Some("Here's your content:\n\n# Product Launch Guide\n\nLaunching a successful product requires careful planning and execution. This guide covers the essential steps to ensure your launch drives maximum impact and user adoption.".to_string()),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                }],
                aux: tower_llm::StepAux {
                    prompt_tokens: 150,
                    completion_tokens: 75,
                    tool_invocations: 0,
                },
            })
        })
    }
}

// Editorial review agent
#[derive(Clone)]
struct EditorAgent;

impl Service<CreateChatCompletionRequest> for EditorAgent {
    type Response = tower_llm::StepOutcome;
    type Error = tower::BoxError;
    type Future = std::pin::Pin<Box<dyn std::future::Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, _: &mut std::task::Context<'_>) -> std::task::Poll<Result<(), Self::Error>> {
        std::task::Poll::Ready(Ok(()))
    }

    fn call(&mut self, _req: CreateChatCompletionRequest) -> Self::Future {
        Box::pin(async move {
            Ok(tower_llm::StepOutcome::Done {
                messages: vec![ChatCompletionRequestMessage {
                    role: async_openai::types::Role::Assistant,
                    content: Some("Editorial review complete! Here's the refined version:\n\n# Getting Started with Our API\n\nOur comprehensive REST API empowers developers with seamless programmatic access to your data. Whether you're building integrations, dashboards, or custom applications, our API provides the robust foundation you need.\n\n## Quick Start\n1. Generate your API key from the dashboard\n2. Make your first request\n3. Explore our interactive documentation\n\n## Key Features\n- Rate limiting: 1000 requests/hour\n- Real-time webhooks\n- Comprehensive error handling\n- SDKs available in Python, JavaScript, and Go\n\n*Edited for clarity, flow, and technical accuracy.*".to_string()),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                }],
                aux: tower_llm::StepAux {
                    prompt_tokens: 250,
                    completion_tokens: 150,
                    tool_invocations: 0,
                },
            })
        })
    }
}

// Content-aware picker
#[derive(Clone)]
struct ContentPicker;

impl AgentPicker<CreateChatCompletionRequest> for ContentPicker {
    fn pick(&self, req: &CreateChatCompletionRequest) -> Result<AgentName, tower::BoxError> {
        let content = req.messages.last()
            .and_then(|m| m.content.as_ref())
            .unwrap_or("")
            .to_lowercase();

        if content.contains("edit") || content.contains("review") || content.contains("proofread") {
            Ok(AgentName("editor_agent".to_string()))
        } else {
            Ok(AgentName("content_agent".to_string()))
        }
    }
}

// Mock tool that simulates processing
fn create_mock_tool() -> tower::util::BoxService<String, serde_json::Value, tower::BoxError> {
    tower::util::BoxService::new(tower::service_fn(|input: String| async move {
        tokio::time::sleep(Duration::from_millis(10)).await;
        Ok::<_, tower::BoxError>(json!({
            "result": format!("Processed: {}", input),
            "timestamp": chrono::Utc::now().to_rfc3339()
        }))
    }))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Initialize tracing for observability
    tracing_subscriber::fmt()
        .with_target(false)
        .with_level(true)
        .init();

    println!("=== Integration Example: Handoffs + Tower Ecosystem ===\n");
    
    println!("This example demonstrates complete integration:");
    println!("• Handoff coordination with full Tower middleware stack");
    println!("• Policy enforcement for both agents and handoffs");
    println!("• Budget tracking across multi-agent workflows");
    println!("• Resilience patterns for robust operation");
    println!("• Observability and metrics collection\n");

    // Set up base agents
    let mut base_agents = HashMap::new();
    base_agents.insert(
        AgentName("content_agent".to_string()),
        tower::util::BoxService::new(ContentAgent)
    );
    base_agents.insert(
        AgentName("editor_agent".to_string()),
        tower::util::BoxService::new(EditorAgent)
    );

    // Create handoff policy
    let mut handoffs = HashMap::new();
    handoffs.insert("handoff_to_editor".to_string(), AgentName("editor_agent".to_string()));
    let handoff_policy = ExplicitHandoffPolicy::new(handoffs);

    // Create picker
    let picker = ContentPicker;

    // Create base handoff coordinator
    let base_coordinator = HandoffCoordinator::new(
        Arc::new(tokio::sync::Mutex::new(base_agents)),
        picker,
        handoff_policy,
    );

    println!("--- Layer 1: Base Handoff Coordination ---");
    println!("✅ HandoffCoordinator with ContentAgent + EditorAgent");
    println!("✅ ExplicitHandoffPolicy for content → editor workflow");
    println!("✅ ContentPicker for intelligent initial routing\n");

    // Add budget policy layer
    let budget_policy = tower_llm::budgets::budget_policy(tower_llm::budgets::Budget {
        max_prompt_tokens: Some(1000),
        max_completion_tokens: Some(500),
        max_tool_invocations: Some(10),
        max_time: Some(Duration::from_secs(30)),
    });

    // Create composite policy with handoff awareness
    let composite_policy = tower_llm::CompositePolicy::new(vec![
        tower_llm::policies::until_no_tool_calls(),
        tower_llm::policies::max_steps(5),
        budget_policy,
    ]);

    println!("--- Layer 2: Policy Integration ---");
    println!("✅ Budget policy: 1000 prompt + 500 completion tokens");
    println!("✅ Max 5 steps across entire handoff workflow");
    println!("✅ Until no tool calls (including handoff tools)");
    println!("✅ 30 second timeout protection\n");

    // Add agent loop layer for policy enforcement
    let policy_enforced_coordinator = tower_llm::AgentLoopLayer::new(composite_policy)
        .layer(base_coordinator);

    println!("--- Layer 3: Agent Loop Integration ---");
    println!("✅ AgentLoopLayer enforces policies across handoffs");
    println!("✅ Multi-agent workflows respect global constraints");
    println!("✅ Automatic termination on budget exhaustion\n");

    // Add resilience layers
    let retry_policy = tower_llm::resilience::RetryPolicy {
        max_retries: 3,
        backoff: tower_llm::resilience::Backoff::exponential(
            Duration::from_millis(100),
            Duration::from_secs(5)
        ),
    };

    let resilient_coordinator = tower_llm::resilience::TimeoutLayer::new(Duration::from_secs(60))
        .layer(tower_llm::resilience::RetryLayer::new(
            retry_policy,
            tower_llm::resilience::AlwaysRetry
        ).layer(policy_enforced_coordinator));

    println!("--- Layer 4: Resilience Integration ---");
    println!("✅ 60 second global timeout for entire workflow");
    println!("✅ Exponential backoff retry (100ms → 5s)");
    println!("✅ Up to 3 retries for transient failures");
    println!("✅ Resilient handoff coordination\n");

    // Add observability layers
    let metrics_collector = {
        let metrics = Arc::new(std::sync::Mutex::new(HashMap::<String, u64>::new()));
        let metrics_clone = metrics.clone();
        
        tower::service_fn(move |record: tower_llm::observability::MetricRecord| {
            let metrics = metrics_clone.clone();
            async move {
                match record {
                    tower_llm::observability::MetricRecord::Counter { name, value } => {
                        let mut m = metrics.lock().unwrap();
                        *m.entry(name).or_insert(0) += value;
                    }
                    _ => {}
                }
                Ok::<_, tower::BoxError>(())
            }
        })
    };

    let full_stack_coordinator = tower_llm::observability::TracingLayer::new()
        .layer(tower_llm::observability::MetricsLayer::new(metrics_collector)
            .layer(resilient_coordinator));

    println!("--- Layer 5: Observability Integration ---");
    println!("✅ Distributed tracing across handoffs");
    println!("✅ Metrics collection for multi-agent workflows");
    println!("✅ Performance monitoring and debugging\n");

    // Test the complete stack
    println!("--- Testing Complete Production Stack ---");
    println!("Input: 'Create API documentation that needs review'");
    println!("Expected: content_agent → handoff → editor_agent (with full middleware)\n");

    let req = async_openai::types::CreateChatCompletionRequestArgs::default()
        .model("gpt-4o")
        .messages(vec![ChatCompletionRequestMessage {
            role: async_openai::types::Role::User,
            content: Some("Create API documentation that needs review".to_string()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }])
        .build()?;

    let mut production_coordinator = full_stack_coordinator;
    let start_time = std::time::Instant::now();

    match production_coordinator.ready().await?.call(req).await {
        Ok(result) => {
            let duration = start_time.elapsed();
            println!("✅ Production workflow completed successfully!");
            println!("📊 Duration: {:?}", duration);
            println!("📊 Steps: {}", result.steps);
            println!("📊 Stop reason: {:?}", result.stop);
            println!("📊 Total tokens: prompt={}, completion={}", 
                result.aux.prompt_tokens, result.aux.completion_tokens);
        },
        Err(e) => println!("❌ Production workflow failed: {}", e),
    }

    println!("\n=== Production Architecture Benefits ===");
    println!("1. **Complete Middleware Stack**:");
    println!("   - Observability (tracing + metrics)");
    println!("   - Resilience (timeout + retry)");
    println!("   - Policy enforcement (budgets + termination)");
    println!("   - Handoff coordination (routing + collaboration)");
    println!();
    println!("2. **Multi-Agent Workflow Management**:");
    println!("   - Global policy enforcement across all agents");
    println!("   - Budget tracking for entire conversation");
    println!("   - Resilient handoff execution");
    println!("   - End-to-end observability");
    println!();
    println!("3. **Tower Ecosystem Integration**:");
    println!("   - Handoffs as first-class Tower services");
    println!("   - Composable middleware layers");
    println!("   - Consistent error handling");
    println!("   - Zero-cost abstractions");
    println!();
    println!("4. **Production Readiness**:");
    println!("   - Fault tolerance and recovery");
    println!("   - Resource consumption monitoring");
    println!("   - Performance optimization");
    println!("   - Debugging and troubleshooting");

    Ok(())
}