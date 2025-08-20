//! Example demonstrating resilience patterns: retry, timeout, and circuit breaker.
//! Shows how to make agent services more robust against failures.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tokio::time::sleep;
use tower::{Layer, Service, ServiceExt};

// Import the next module and its submodules
#[path = "../src/next/mod.rs"]
mod next;

#[path = "../src/next/resilience/mod.rs"]
mod resilience;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("=== Resilience Patterns Example ===\n");

    // Example 1: Retry with exponential backoff
    println!("--- Example 1: Retry Pattern ---");

    let fail_count = Arc::new(AtomicUsize::new(0));
    let fail_count_clone = fail_count.clone();

    // Service that fails twice then succeeds
    let flaky_service = tower::service_fn(move |_req: ()| {
        let count = fail_count_clone.fetch_add(1, Ordering::SeqCst);
        async move {
            if count < 2 {
                println!("  Attempt {}: ❌ Failed", count + 1);
                Err::<String, tower::BoxError>("temporary failure".into())
            } else {
                println!("  Attempt {}: ✅ Success!", count + 1);
                Ok("Success after retries".to_string())
            }
        }
    });

    let retry_policy = resilience::RetryPolicy {
        max_retries: 3,
        backoff: resilience::Backoff::exponential(
            Duration::from_millis(100),
            2.0,
            Duration::from_secs(1),
        ),
    };

    let retry_layer = resilience::RetryLayer::new(retry_policy, resilience::AlwaysRetry);
    let mut retrying_service = retry_layer.layer(flaky_service);

    println!("  Starting request with retry (max 3 attempts)...");
    let result = retrying_service.ready().await?.call(()).await?;
    println!("  Final result: {}\n", result);

    // Example 2: Timeout protection
    println!("--- Example 2: Timeout Pattern ---");

    // Service that takes too long
    let slow_service = tower::service_fn(|_req: ()| async {
        println!("  Service starting slow operation...");
        sleep(Duration::from_secs(2)).await;
        Ok::<_, tower::BoxError>("This took too long")
    });

    let timeout_layer = resilience::TimeoutLayer::new(Duration::from_millis(500));
    let mut timeout_service = timeout_layer.layer(slow_service.clone());

    println!("  Calling service with 500ms timeout...");
    match timeout_service.ready().await?.call(()).await {
        Ok(_) => println!("  Unexpected success"),
        Err(e) => println!("  ⏱️ Timed out as expected: {}", e),
    }

    // Now with longer timeout
    let longer_timeout = resilience::TimeoutLayer::new(Duration::from_secs(3));
    let mut longer_service = longer_timeout.layer(slow_service);

    println!("\n  Calling same service with 3s timeout...");
    match longer_service.ready().await?.call(()).await {
        Ok(msg) => println!("  ✅ Completed: {}", msg),
        Err(e) => println!("  Failed: {}", e),
    }

    // Example 3: Circuit breaker
    println!("\n--- Example 3: Circuit Breaker Pattern ---");

    let call_count = Arc::new(AtomicUsize::new(0));
    let call_count_clone = call_count.clone();

    // Service that always fails
    let failing_service = tower::service_fn(move |_req: ()| {
        let count = call_count_clone.fetch_add(1, Ordering::SeqCst);
        async move {
            println!("    Service called (attempt #{})", count + 1);
            Err::<(), tower::BoxError>("service error".into())
        }
    });

    let breaker_config = resilience::BreakerConfig {
        failure_threshold: 2,
        reset_timeout: Duration::from_millis(500),
    };

    let breaker_layer = resilience::CircuitBreakerLayer::new(breaker_config);
    let mut breaker_service = breaker_layer.layer(failing_service);

    println!("  Circuit breaker with failure threshold: 2");

    // First two calls should go through and fail
    for i in 1..=2 {
        println!("\n  Call #{}", i);
        match breaker_service.ready().await?.call(()).await {
            Ok(_) => println!("    Unexpected success"),
            Err(e) => println!("    Failed (expected): {}", e),
        }
    }

    // Third call should be rejected by circuit breaker
    println!("\n  Call #3 (circuit should be open)");
    match breaker_service.ready().await?.call(()).await {
        Ok(_) => println!("    Unexpected success"),
        Err(e) => println!("    🔌 Circuit breaker prevented call: {}", e),
    }

    // Wait for reset timeout
    println!("\n  Waiting 500ms for circuit reset...");
    sleep(Duration::from_millis(600)).await;

    // Circuit should be half-open, allowing one test call
    println!("\n  Call #4 (circuit half-open, testing)");
    match breaker_service.ready().await?.call(()).await {
        Ok(_) => println!("    Unexpected success"),
        Err(e) => println!("    Failed again, circuit reopens: {}", e),
    }

    println!("\n=== Combining Patterns ===");
    println!("In production, you would typically combine these patterns:");
    println!("1. Circuit breaker (outermost) - prevents cascading failures");
    println!("2. Retry (middle) - handles transient failures");
    println!("3. Timeout (innermost) - ensures bounded latency");
    println!("\nExample stack:");
    println!("  CircuitBreakerLayer::new(config)");
    println!("    .layer(RetryLayer::new(policy, classifier)");
    println!("      .layer(TimeoutLayer::new(duration)");
    println!("        .layer(your_service)))");

    println!("\n=== Key Takeaways ===");
    println!("1. Retry patterns handle transient failures automatically");
    println!("2. Exponential backoff prevents overwhelming failed services");
    println!("3. Timeouts ensure operations complete in bounded time");
    println!("4. Circuit breakers prevent cascading failures");
    println!("5. These patterns compose naturally with Tower layers");
    println!("6. Essential for production agent deployments");

    Ok(())
}
