//! Example demonstrating concurrent tool execution with the concurrency module.
//! Shows how to run multiple tools in parallel with rate limiting and policies.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::json;
use tokio::time::sleep;
use tower::{Service, ServiceExt};

// Import the next module and its submodules
#[path = "../src/next/mod.rs"]
mod next;

#[path = "../src/next/concurrency/mod.rs"]
mod concurrency;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("=== Concurrent Tool Execution Example ===\n");

    // Track concurrent executions
    let concurrent_count = Arc::new(AtomicUsize::new(0));
    let max_concurrent = Arc::new(AtomicUsize::new(0));

    // Create multiple tool services that simulate work
    let tool1 = create_tool(
        "weather",
        500,
        concurrent_count.clone(),
        max_concurrent.clone(),
    );
    let tool2 = create_tool(
        "stock",
        300,
        concurrent_count.clone(),
        max_concurrent.clone(),
    );
    let tool3 = create_tool(
        "news",
        400,
        concurrent_count.clone(),
        max_concurrent.clone(),
    );
    let tool4 = create_tool(
        "translate",
        200,
        concurrent_count.clone(),
        max_concurrent.clone(),
    );

    // Create a simple router that dispatches based on tool name
    let tools = Arc::new(vec![
        ("weather", Arc::new(tokio::sync::Mutex::new(tool1))),
        ("stock", Arc::new(tokio::sync::Mutex::new(tool2))),
        ("news", Arc::new(tokio::sync::Mutex::new(tool3))),
        ("translate", Arc::new(tokio::sync::Mutex::new(tool4))),
    ]);

    let create_router = move || {
        let tools = tools.clone();
        tower::util::BoxService::new(tower::service_fn(move |inv: next::ToolInvocation| {
            let tools = tools.clone();
            Box::pin(async move {
                for (name, tool) in tools.iter() {
                    if inv.name == *name {
                        let mut tool_guard = tool.lock().await;
                        return tool_guard.ready().await?.call(inv).await;
                    }
                }
                Err::<next::ToolOutput, tower::BoxError>(
                    format!("Unknown tool: {}", inv.name).into(),
                )
            })
        }))
    };

    println!("--- Test 1: Sequential Execution (Baseline) ---");
    let start = Instant::now();

    // Execute tools sequentially
    let mut seq_router = create_router();
    for (i, tool_name) in ["weather", "stock", "news", "translate"].iter().enumerate() {
        let inv = next::ToolInvocation {
            id: format!("seq_{}", i),
            name: tool_name.to_string(),
            arguments: json!({}),
        };
        let _ = seq_router.ready().await?.call(inv).await?;
    }

    let seq_duration = start.elapsed();
    println!("Sequential execution took: {:?}", seq_duration);
    println!(
        "Peak concurrent executions: {}\n",
        max_concurrent.load(Ordering::SeqCst)
    );

    // Reset counter
    max_concurrent.store(0, Ordering::SeqCst);

    println!("--- Test 2: Parallel Execution with JoinAll Policy ---");

    // Wrap router with parallel execution layer (limit 3 concurrent)
    // Use Buffer to make the router clonable for concurrent access
    let buffered_router = tower::buffer::Buffer::new(create_router(), 10);
    let mut parallel_router = concurrency::ParallelToolRouter::new(
        buffered_router,
        concurrency::ConcurrencyLimit(3),
        concurrency::ToolJoinPolicy::JoinAll,
    );

    // Create multiple tool invocations
    let invocations = vec![
        next::ToolInvocation {
            id: "p1".into(),
            name: "weather".into(),
            arguments: json!({"city": "New York"}),
        },
        next::ToolInvocation {
            id: "p2".into(),
            name: "stock".into(),
            arguments: json!({"symbol": "AAPL"}),
        },
        next::ToolInvocation {
            id: "p3".into(),
            name: "news".into(),
            arguments: json!({"topic": "tech"}),
        },
        next::ToolInvocation {
            id: "p4".into(),
            name: "translate".into(),
            arguments: json!({"text": "hello"}),
        },
    ];

    let start = Instant::now();

    // Execute all tools in parallel
    let results = parallel_router
        .ready()
        .await?
        .call(invocations.clone())
        .await?;

    let par_duration = start.elapsed();
    println!("Parallel execution took: {:?}", par_duration);
    println!(
        "Peak concurrent executions: {}",
        max_concurrent.load(Ordering::SeqCst)
    );
    println!("Results received: {}", results.len());
    println!(
        "Speedup: {:.2}x\n",
        seq_duration.as_secs_f64() / par_duration.as_secs_f64()
    );

    // Reset counter
    max_concurrent.store(0, Ordering::SeqCst);

    println!("--- Test 3: Parallel with FailFast Policy ---");

    // Add a failing tool
    let failing_invocations = vec![
        next::ToolInvocation {
            id: "f1".into(),
            name: "weather".into(),
            arguments: json!({}),
        },
        next::ToolInvocation {
            id: "fail".into(),
            name: "unknown_tool".into(), // This will fail
            arguments: json!({}),
        },
        next::ToolInvocation {
            id: "f3".into(),
            name: "news".into(),
            arguments: json!({}),
        },
    ];

    let buffered_router2 = tower::buffer::Buffer::new(create_router(), 10);
    let mut fail_fast_router = concurrency::ParallelToolRouter::new(
        buffered_router2,
        concurrency::ConcurrencyLimit(3),
        concurrency::ToolJoinPolicy::FailFast,
    );

    let start = Instant::now();

    match fail_fast_router
        .ready()
        .await?
        .call(failing_invocations)
        .await
    {
        Ok(_) => println!("Unexpected success"),
        Err(e) => {
            let duration = start.elapsed();
            println!("FailFast stopped execution after: {:?}", duration);
            println!("Error: {}", e);
            println!("(Other tools may have been cancelled)\n");
        }
    }

    println!("--- Test 4: Concurrency Limit Enforcement ---");

    // Test with limit of 2
    let buffered_router3 = tower::buffer::Buffer::new(create_router(), 10);
    let limited_router = concurrency::ParallelToolRouter::new(
        buffered_router3,
        concurrency::ConcurrencyLimit(2), // Only 2 at a time
        concurrency::ToolJoinPolicy::JoinAll,
    );

    max_concurrent.store(0, Ordering::SeqCst);
    let mut lim_router = limited_router;

    let _ = lim_router.ready().await?.call(invocations).await?;

    println!("With concurrency limit of 2:");
    println!(
        "Peak concurrent executions: {}",
        max_concurrent.load(Ordering::SeqCst)
    );
    println!("(Should not exceed 2)\n");

    println!("=== Key Takeaways ===");
    println!("1. ParallelToolRouter enables concurrent tool execution");
    println!("2. ConcurrencyLimit prevents resource exhaustion");
    println!("3. JoinAll waits for all tools to complete");
    println!("4. FailFast aborts on first error for fast failure");
    println!("5. Parallel execution can significantly reduce latency");
    println!("6. Order of results is preserved despite parallel execution");

    Ok(())
}

fn create_tool(
    name: &'static str,
    delay_ms: u64,
    counter: Arc<AtomicUsize>,
    max_counter: Arc<AtomicUsize>,
) -> tower::util::BoxService<next::ToolInvocation, next::ToolOutput, tower::BoxError> {
    tower::util::BoxService::new(tower::service_fn(move |inv: next::ToolInvocation| {
        let counter = counter.clone();
        let max_counter = max_counter.clone();
        Box::pin(async move {
            // Increment concurrent count
            let current = counter.fetch_add(1, Ordering::SeqCst) + 1;

            // Update max if needed
            let mut max = max_counter.load(Ordering::SeqCst);
            while current > max {
                match max_counter.compare_exchange_weak(
                    max,
                    current,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                ) {
                    Ok(_) => break,
                    Err(x) => max = x,
                }
            }

            println!("  [{}] Starting (concurrent: {})", name, current);

            // Simulate work
            sleep(Duration::from_millis(delay_ms)).await;

            // Decrement concurrent count
            counter.fetch_sub(1, Ordering::SeqCst);

            println!("  [{}] Completed", name);

            Ok(next::ToolOutput {
                id: inv.id,
                result: json!({ "tool": name, "status": "completed" }),
            })
        })
    }))
}
