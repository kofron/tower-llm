//! Example demonstrating tools as Tower services.
//!
//! This shows the new pattern where tools directly implement Tower's Service trait,
//! making them first-class citizens that can be composed with any Tower middleware.

use openai_agents_rs::{
    env::{DefaultEnv, EnvBuilder, LoggingCapability},
    service::{ToolRequest, ToolResponse},
    tool::FunctionTool,
    tool_service::{IntoToolService, ServiceTool},
};
use serde_json::json;
use std::sync::Arc;
use std::time::Duration;
use tower::{
    retry::{Policy, RetryLayer},
    timeout::TimeoutLayer,
    Service, ServiceBuilder,
};

/// Custom retry policy for demonstration
#[derive(Clone)]
struct SimpleRetryPolicy {
    remaining_attempts: usize,
}

impl SimpleRetryPolicy {
    fn new(max_attempts: usize) -> Self {
        Self {
            remaining_attempts: max_attempts,
        }
    }
}

impl<E> Policy<ToolRequest<E>, ToolResponse, tower::BoxError> for SimpleRetryPolicy
where
    E: openai_agents_rs::env::Env,
{
    type Future = std::future::Ready<()>;

    fn retry(
        &mut self,
        _req: &mut ToolRequest<E>,
        result: &mut Result<ToolResponse, tower::BoxError>,
    ) -> Option<Self::Future> {
        match result {
            Ok(response) if response.error.is_some() && self.remaining_attempts > 0 => {
                self.remaining_attempts -= 1;
                Some(std::future::ready(()))
            }
            _ => None,
        }
    }

    fn clone_request(&mut self, req: &ToolRequest<E>) -> Option<ToolRequest<E>> {
        Some(req.clone())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    println!("=== Tools as Tower Services Demo ===\n");

    // Create a ServiceTool - the new pattern
    let calc_fn = |input: String| {
        // Simple calculator that evaluates "a + b" format
        if let Some((a, b)) = input.split_once('+') {
            let a: f64 = a.trim().parse().unwrap_or(0.0);
            let b: f64 = b.trim().parse().unwrap_or(0.0);
            format!("{}", a + b)
        } else if let Some((a, b)) = input.split_once('-') {
            let a: f64 = a.trim().parse().unwrap_or(0.0);
            let b: f64 = b.trim().parse().unwrap_or(0.0);
            format!("{}", a - b)
        } else {
            "Invalid expression".to_string()
        }
    };
    // Use the adapter pattern for simpler construction
    let calculator = openai_agents_rs::tool::FunctionTool::simple(
        "calculator",
        "Performs calculations",
        calc_fn,
    )
    .into_service::<DefaultEnv>();

    // Compose with Tower middleware directly!
    let service = ServiceBuilder::new()
        .layer(TimeoutLayer::new(Duration::from_secs(1)))
        .layer(RetryLayer::new(SimpleRetryPolicy::new(3)))
        .service(calculator);

    println!("1. Created a ServiceTool with Tower middleware");
    println!("   - Timeout: 1 second");
    println!("   - Retry: up to 3 attempts\n");

    // Test the service
    let mut svc = service.clone();
    let req = ToolRequest {
        env: DefaultEnv,
        run_id: "test-run".to_string(),
        agent: "test-agent".to_string(),
        tool_call_id: "calc-1".to_string(),
        tool_name: "calculator".to_string(),
        arguments: json!({"input": "5 + 3"}),
    };

    let response = svc.call(req).await.map_err(|e| e.to_string())?;
    println!(
        "2. Calculator result: output={:?}, error={:?}\n",
        response.output, response.error
    );

    // Demonstrate adapter pattern for existing tools
    let existing_tool = FunctionTool::simple("uppercase", "Converts to uppercase", |s: String| {
        s.to_uppercase()
    });

    // Convert to service using the adapter
    let adapted_service = existing_tool.into_service::<DefaultEnv>();

    // Can compose with Tower middleware
    let enhanced = ServiceBuilder::new()
        .timeout(Duration::from_secs(2))
        .service(adapted_service);

    println!("3. Adapted existing FunctionTool to Service");
    println!("   - Applied timeout middleware");
    println!("   - Fully compatible with Tower ecosystem\n");

    // Test the adapted service
    let mut svc = enhanced;
    let req = ToolRequest {
        env: DefaultEnv,
        run_id: "test-run".to_string(),
        agent: "test-agent".to_string(),
        tool_call_id: "upper-1".to_string(),
        tool_name: "uppercase".to_string(),
        arguments: json!({"input": "hello tower"}),
    };

    let response = svc.call(req).await.map_err(|e| e.to_string())?;
    println!(
        "4. Uppercase result: output={:?}, error={:?}\n",
        response.output, response.error
    );

    // Demonstrate with capabilities
    let env = EnvBuilder::new()
        .with_capability(Arc::new(LoggingCapability))
        .build();

    let log_fn = |s: String| {
        println!("   Tool received: {}", s);
        s
    };
    let logging_tool =
        openai_agents_rs::tool::FunctionTool::simple("logger", "Logs and echoes", log_fn)
            .into_service();

    let mut svc = logging_tool;
    let req = ToolRequest {
        env: env.clone(),
        run_id: "test-run".to_string(),
        agent: "test-agent".to_string(),
        tool_call_id: "log-1".to_string(),
        tool_name: "logger".to_string(),
        arguments: json!({"input": "Hello with capabilities!"}),
    };

    println!("5. Tool with capability-aware environment:");
    let response = svc.call(req).await.map_err(|e| e.to_string())?;
    println!(
        "   Result: output={:?}, error={:?}\n",
        response.output, response.error
    );

    println!("=== Summary ===");
    println!("✅ Tools are now first-class Tower services");
    println!("✅ Direct composition with Tower middleware");
    println!("✅ Existing tools can be adapted");
    println!("✅ Full capability system integration");
    println!("✅ Type-safe, no string coupling");

    Ok(())
}
