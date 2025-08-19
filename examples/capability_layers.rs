//! Example demonstrating capability-based layers.
//!
//! This example shows how layers can use capabilities from the environment
//! to access shared resources like logging, metrics, and approval services.

use openai_agents_rs::{
    env::{EnvBuilder, InMemoryMetrics, Logging, LoggingCapability, Metrics},
    layers,
    service::{ErasedToolLayer, ToolRequest, ToolResponse},
    Agent, FunctionTool, Tool,
};
use serde_json::json;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use tower::{BoxError, Layer, Service};

/// A custom layer that uses logging and metrics capabilities.
#[derive(Clone)]
struct ObservabilityLayer;

impl<S> Layer<S> for ObservabilityLayer {
    type Service = ObservabilityService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        ObservabilityService { inner }
    }
}

#[derive(Clone)]
struct ObservabilityService<S> {
    inner: S,
}

impl<S, E> Service<ToolRequest<E>> for ObservabilityService<S>
where
    S: Service<ToolRequest<E>, Response = ToolResponse, Error = BoxError> + Clone + Send + 'static,
    S::Future: Send + 'static,
    E: openai_agents_rs::env::Env,
{
    type Response = ToolResponse;
    type Error = BoxError;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: ToolRequest<E>) -> Self::Future {
        let mut inner = self.inner.clone();

        Box::pin(async move {
            // Use logging capability if available
            if let Some(logger) = req.env.capability::<LoggingCapability>() {
                logger.info(&format!("Executing tool: {}", req.tool_name));
            }

            // Use metrics capability if available
            if let Some(metrics) = req.env.capability::<InMemoryMetrics>() {
                metrics.increment(&format!("tool.{}.calls", req.tool_name), 1);
            }

            let start = std::time::Instant::now();
            let result = inner.call(req.clone()).await;
            let duration = start.elapsed();

            // Log the result
            if let Some(logger) = req.env.capability::<LoggingCapability>() {
                match &result {
                    Ok(_) => logger.info(&format!(
                        "Tool {} succeeded in {:?}",
                        req.tool_name, duration
                    )),
                    Err(e) => logger.error(&format!("Tool {} failed: {}", req.tool_name, e)),
                }
            }

            // Record metrics
            if let Some(metrics) = req.env.capability::<InMemoryMetrics>() {
                metrics.histogram(
                    &format!("tool.{}.duration_ms", req.tool_name),
                    duration.as_millis() as f64,
                );
                if result.is_ok() {
                    metrics.increment(&format!("tool.{}.success", req.tool_name), 1);
                } else {
                    metrics.increment(&format!("tool.{}.failure", req.tool_name), 1);
                }
            }

            result
        })
    }
}

/// Helper to box the observability layer
fn boxed_observability() -> Arc<dyn ErasedToolLayer> {
    struct Erased;
    impl ErasedToolLayer for Erased {
        fn layer_boxed(
            &self,
            svc: openai_agents_rs::service::ToolBoxService,
        ) -> openai_agents_rs::service::ToolBoxService {
            tower::util::BoxService::new(ObservabilityLayer.layer(svc))
        }
    }
    Arc::new(Erased)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    // Create an environment with logging and metrics capabilities
    let env = EnvBuilder::new()
        .with_capability(Arc::new(LoggingCapability))
        .with_capability(Arc::new(InMemoryMetrics::default()))
        .build();

    // Create tools with observability layer
    let calculator = FunctionTool::simple("add", "Adds two numbers", |input: String| {
        // Parse input as "a + b"
        let parts: Vec<&str> = input.split('+').collect();
        if parts.len() == 2 {
            let a: f64 = parts[0].trim().parse().unwrap_or(0.0);
            let b: f64 = parts[1].trim().parse().unwrap_or(0.0);
            format!("{}", a + b)
        } else {
            "Invalid input".to_string()
        }
    })
    .layer(boxed_observability())
    .layer(layers::boxed_retry_times(2));

    let weather = FunctionTool::simple("weather", "Gets weather", |city: String| {
        format!("The weather in {} is sunny", city)
    })
    .layer(boxed_observability())
    .layer(layers::boxed_timeout_secs(5));

    // Create an agent with the tools
    let agent = Agent::simple("Assistant", "A helpful assistant with observability")
        .with_tool(Arc::new(calculator))
        .with_tool(Arc::new(weather));

    println!("Created agent with capability-aware layers!");
    println!("The tools will use logging and metrics from the environment.");

    // In a real application, you would run the agent with:
    // let runner = Runner::new(env);
    // let result = runner.run(agent, "What's 5 + 3?", RunConfig::default()).await?;

    Ok(())
}
