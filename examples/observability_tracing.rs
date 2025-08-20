//! Example demonstrating observability features: metrics and tracing.
//! Shows how to instrument agent services for production monitoring.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use tower::{Layer, Service, ServiceExt};
use tracing::{info, info_span};

// Import the next module and its submodules
// Core module is now at root level
// use tower_llm directly

// Simple metrics collector for demonstration
#[derive(Clone)]
struct DemoMetricsCollector {
    counters: Arc<std::sync::Mutex<std::collections::HashMap<&'static str, u64>>>,
    histograms: Arc<std::sync::Mutex<Vec<(&'static str, u64)>>>,
}

impl DemoMetricsCollector {
    fn new() -> Self {
        Self {
            counters: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
            histograms: Arc::new(std::sync::Mutex::new(Vec::new())),
        }
    }

    fn print_summary(&self) {
        println!("\n📊 Metrics Summary:");
        let counters = self.counters.lock().unwrap();
        for (name, value) in counters.iter() {
            println!("  Counter '{}': {}", name, value);
        }

        let histograms = self.histograms.lock().unwrap();
        if !histograms.is_empty() {
            println!("  Histogram samples:");
            for (name, value) in histograms.iter().take(5) {
                println!("    {} = {}ms", name, value);
            }
            if histograms.len() > 5 {
                println!("    ... and {} more samples", histograms.len() - 5);
            }
        }
    }
}

impl Service<tower_llm::observability::MetricRecord> for DemoMetricsCollector {
    type Response = ();
    type Error = tower::BoxError;
    type Future =
        std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), tower::BoxError>> + Send>>;

    fn poll_ready(
        &mut self,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        std::task::Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: tower_llm::observability::MetricRecord) -> Self::Future {
        let counters = self.counters.clone();
        let histograms = self.histograms.clone();

        Box::pin(async move {
            match req {
                tower_llm::observability::MetricRecord::Counter { name, value } => {
                    let mut c = counters.lock().unwrap();
                    *c.entry(name).or_insert(0) += value;
                }
                tower_llm::observability::MetricRecord::Histogram { name, value } => {
                    let mut h = histograms.lock().unwrap();
                    h.push((name, value));
                }
            }
            Ok(())
        })
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Initialize tracing subscriber for structured logging
    tracing_subscriber::fmt()
        .with_target(false)
        .with_level(true)
        .init();

    println!("=== Observability Example ===\n");

    // Create a mock service that simulates agent steps
    let step_count = Arc::new(AtomicU64::new(0));
    let step_count_clone = step_count.clone();

    let mock_step = tower::service_fn(
        move |req: async_openai::types::CreateChatCompletionRequest| {
            let count = step_count_clone.fetch_add(1, Ordering::SeqCst);
            async move {
                // Simulate some work
                tokio::time::sleep(tokio::time::Duration::from_millis(50 + count * 10)).await;

                info!(
                    "Processed step {} with {} messages",
                    count + 1,
                    req.messages.len()
                );

                Ok::<_, tower::BoxError>(tower_llm::StepOutcome::Done {
                    messages: req.messages,
                    aux: tower_llm::StepAux {
                        prompt_tokens: 100 + count as usize * 20,
                        completion_tokens: 50 + count as usize * 10,
                        tool_invocations: if count % 2 == 0 { 1 } else { 0 },
                    },
                })
            }
        },
    );

    println!("--- Example 1: Tracing Layer ---");
    println!("Instrumenting service calls with spans...\n");

    // Add tracing layer
    let tracing_layer = tower_llm::observability::TracingLayer::new();
    let mut traced_service = tracing_layer.layer(mock_step.clone());

    // Make some calls within a span
    let span = info_span!("agent_run", run_id = "test-123");
    let _enter = span.enter();

    for i in 1..=3 {
        info!("Starting step {}", i);

        let req = async_openai::types::CreateChatCompletionRequestArgs::default()
            .model("gpt-4o")
            .messages(vec![])
            .build()?;

        let outcome = traced_service.ready().await?.call(req).await?;

        match outcome {
            tower_llm::StepOutcome::Done { aux, .. } => {
                info!(
                    "Step {} complete: {} prompt, {} completion tokens",
                    i, aux.prompt_tokens, aux.completion_tokens
                );
            }
            _ => {}
        }
    }

    println!("\n✅ Tracing spans logged (check output above)");

    println!("\n--- Example 2: Metrics Layer ---");
    println!("Collecting metrics from service calls...\n");

    // Create metrics collector
    let collector = DemoMetricsCollector::new();

    // Add metrics layer
    let metrics_layer = tower_llm::observability::MetricsLayer::new(collector.clone());
    let mut metered_service = metrics_layer.layer(mock_step.clone());

    // Reset step counter
    step_count.store(0, Ordering::SeqCst);

    // Make several calls to generate metrics
    for i in 1..=5 {
        let req = async_openai::types::CreateChatCompletionRequestArgs::default()
            .model("gpt-4o")
            .messages(vec![])
            .build()?;

        let start = std::time::Instant::now();
        let _ = metered_service.ready().await?.call(req).await?;
        let duration = start.elapsed();

        println!("  Call {}: {}ms", i, duration.as_millis());
    }

    // Print collected metrics
    collector.print_summary();

    println!("\n--- Example 3: Combined Observability Stack ---");
    println!("Layering tracing and metrics together...\n");

    // Create a new collector for the combined example
    let combined_collector = DemoMetricsCollector::new();

    // Stack the layers: tracing -> metrics -> service
    let combined = tower_llm::observability::TracingLayer::new().layer(
        tower_llm::observability::MetricsLayer::new(combined_collector.clone()).layer(mock_step),
    );

    let mut observable_service = combined;

    // Reset counter
    step_count.store(0, Ordering::SeqCst);

    // Run with full observability
    let span = info_span!("production_run", env = "demo");
    let _enter = span.enter();

    for i in 1..=3 {
        let req = async_openai::types::CreateChatCompletionRequestArgs::default()
            .model("gpt-4o")
            .messages(vec![])
            .build()?;

        info!("Request {}", i);
        let _ = observable_service.ready().await?.call(req).await?;
    }

    combined_collector.print_summary();

    println!("\n=== Key Takeaways ===");
    println!("1. TracingLayer adds structured logging spans to service calls");
    println!("2. MetricsLayer collects counters and histograms for monitoring");
    println!("3. Layers compose naturally for comprehensive observability");
    println!("4. Essential for debugging and monitoring production agents");
    println!("5. Integrates with standard observability tools (Prometheus, Jaeger, etc.)");
    println!("6. Zero-cost abstraction when disabled");

    Ok(())
}
