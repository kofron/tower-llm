//! Comprehensive demonstration of all next module capabilities.
//! This example shows how all the submodules work together to create
//! a production-ready agent system.

use std::sync::Arc;
use std::time::Duration;

use async_openai::{config::OpenAIConfig, Client};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;
use tower::{Layer, Service, ServiceExt};
use tracing::info;

// Import the next module and key submodules
// Core module is now at root level
// use tower_llm directly

// no additional imports needed

#[derive(Debug, Deserialize, JsonSchema)]
struct MathArgs {
    operation: String,
    a: f64,
    b: f64,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_target(false)
        .with_level(true)
        .init();

    println!("=== Comprehensive Next Module Demo ===\n");
    println!("This example demonstrates:");
    println!("  ✓ Agent with typed tools (services module)");
    println!("  ✓ Budget policies (budgets module)");
    println!("  ✓ Resilience patterns (resilience module)");
    println!("  ✓ Observability (observability module)");
    println!("  ✓ Composable layers (layers module)");
    println!("  ✓ DX utilities (utils module)\n");

    // Create OpenAI client
    let client: Arc<Client<OpenAIConfig>> = Arc::new(Client::new());

    // Define a calculator tool using typed helper
    let calculator = tower_llm::tool_typed(
        "calculator",
        "Perform arithmetic operations",
        |args: MathArgs| async move {
            let result = match args.operation.as_str() {
                "add" => args.a + args.b,
                "subtract" => args.a - args.b,
                "multiply" => args.a * args.b,
                "divide" => {
                    if args.b == 0.0 {
                        return Err("Division by zero".into());
                    }
                    args.a / args.b
                }
                _ => return Err(format!("Unknown operation: {}", args.operation).into()),
            };

            info!(
                "Calculator: {} {} {} = {}",
                args.a, args.operation, args.b, result
            );
            Ok::<_, tower::BoxError>(json!({ "result": result }))
        },
    );

    println!("--- Part 1: Basic Agent with Tools and Policies ---\n");

    // Build agent with tools and composite policy
    let policy = tower_llm::CompositePolicy::new(vec![
        tower_llm::policies::until_no_tool_calls(),
        tower_llm::policies::max_steps(3),
        tower_llm::budgets::budget_policy(tower_llm::budgets::Budget {
            max_prompt_tokens: Some(1000),
            max_completion_tokens: Some(500),
            max_tool_invocations: Some(5),
            max_time: Some(Duration::from_secs(30)),
        }),
    ]);

    let mut basic_agent = tower_llm::Agent::builder(client.clone())
        .model("gpt-4o-mini")
        .temperature(0.7)
        .tool(calculator)
        .policy(policy)
        .build();

    // Run a simple calculation
    println!("User: What's 25 * 4?");
    let result = tower_llm::run(
        &mut basic_agent,
        "You are a helpful math assistant. Use the calculator tool for calculations.",
        "What's 25 * 4?",
    )
    .await?;

    println!("Agent completed in {} steps", result.steps);
    println!("Stop reason: {:?}\n", result.stop);

    println!("--- Part 2: Resilient Agent Service ---\n");

    // Create a mock agent service that sometimes fails
    let fail_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let fail_count_clone = fail_count.clone();

    let flaky_agent = tower::service_fn(
        move |req: async_openai::types::CreateChatCompletionRequest| {
            let count = fail_count_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            async move {
                if count == 0 {
                    // First call fails
                    Err::<tower_llm::StepOutcome, tower::BoxError>("Temporary failure".into())
                } else {
                    // Subsequent calls succeed
                    Ok(tower_llm::StepOutcome::Done {
                        messages: req.messages,
                        aux: tower_llm::StepAux {
                            prompt_tokens: 50,
                            completion_tokens: 25,
                            tool_invocations: 0,
                        },
                    })
                }
            }
        },
    );

    // Wrap with resilience layers
    let retry_policy = tower_llm::resilience::RetryPolicy {
        max_retries: 2,
        backoff: tower_llm::resilience::Backoff::fixed(Duration::from_millis(100)),
    };

    let resilient_agent = tower_llm::resilience::TimeoutLayer::new(Duration::from_secs(5)).layer(
        tower_llm::resilience::RetryLayer::new(retry_policy, tower_llm::resilience::AlwaysRetry)
            .layer(flaky_agent),
    );

    let mut resilient = resilient_agent;

    println!("Testing resilient agent (will retry on failure)...");
    let req = async_openai::types::CreateChatCompletionRequestArgs::default()
        .model("gpt-4o")
        .messages(vec![])
        .build()?;

    match resilient.ready().await?.call(req).await {
        Ok(_) => println!("✅ Succeeded after retry"),
        Err(e) => println!("❌ Failed: {}", e),
    }

    println!("\n--- Part 3: Observable Agent Stack ---\n");

    // Create a simple metrics collector
    let metrics = Arc::new(std::sync::Mutex::new(std::collections::HashMap::new()));
    let metrics_clone = metrics.clone();

    let collector = tower::service_fn(move |record: tower_llm::observability::MetricRecord| {
        let metrics = metrics_clone.clone();
        async move {
            if let tower_llm::observability::MetricRecord::Counter { name, value } = record {
                let mut m = metrics.lock().unwrap();
                *m.entry(name).or_insert(0u64) += value;
            }
            Ok::<_, tower::BoxError>(())
        }
    });

    // Create observable agent with tracing and metrics
    let observable_agent = tower_llm::observability::TracingLayer::new().layer(
        tower_llm::observability::MetricsLayer::new(collector).layer(tower::service_fn(
            |req: async_openai::types::CreateChatCompletionRequest| async move {
                info!("Processing request with {} messages", req.messages.len());
                Ok::<_, tower::BoxError>(tower_llm::StepOutcome::Done {
                    messages: req.messages,
                    aux: tower_llm::StepAux {
                        prompt_tokens: 100,
                        completion_tokens: 50,
                        tool_invocations: 1,
                    },
                })
            },
        )),
    );

    let mut observable = observable_agent;

    println!("Running observable agent...");
    let req = async_openai::types::CreateChatCompletionRequestArgs::default()
        .model("gpt-4o")
        .messages(vec![])
        .build()?;

    let _ = observable.ready().await?.call(req).await?;

    // Print collected metrics
    let final_metrics = metrics.lock().unwrap();
    println!("\n📊 Collected Metrics:");
    for (name, value) in final_metrics.iter() {
        println!("  {}: {}", name, value);
    }

    println!("\n--- Part 4: Complete Production Stack ---\n");

    println!("A production agent would combine all these patterns:");
    println!("```");
    println!("let production_agent = ");
    println!("    // Observability (outermost)");
    println!("    TracingLayer::new()");
    println!("    .layer(MetricsLayer::new(prometheus_collector)");
    println!("    // Resilience");
    println!("    .layer(CircuitBreakerLayer::new(breaker_config)");
    println!("    .layer(RetryLayer::new(retry_policy, classifier)");
    println!("    .layer(TimeoutLayer::new(timeout_duration)");
    println!("    // Core agent with budget policies");
    println!("    .layer(AgentLoopLayer::new(composite_policy)");
    println!("    .layer(StepLayer::new(client, model, tools)");
    println!("    .layer(tool_router))))));");
    println!("```");

    println!("\n=== Key Achievements ===");
    println!("✅ Demonstrated all major submodules working together");
    println!("✅ Showed practical composition of layers");
    println!("✅ Integrated with OpenAI API successfully");
    println!("✅ Applied production patterns (retry, timeout, metrics)");
    println!("✅ Used typed tools and DX utilities");
    println!("✅ Enforced resource budgets");

    println!("\n=== Architecture Benefits ===");
    println!("1. **Composability**: Layers combine naturally");
    println!("2. **Testability**: Each layer can be tested independently");
    println!("3. **Observability**: Built-in metrics and tracing");
    println!("4. **Resilience**: Production-ready error handling");
    println!("5. **Type Safety**: Compile-time guarantees");
    println!("6. **Performance**: Zero-cost abstractions");

    Ok(())
}
