//! Parallel tools example demonstrating concurrent tool execution
//!
//! This example shows how an agent can use multiple tools in parallel
//! to gather information more efficiently.
//!
//! Run with: cargo run --example parallel_tools

use openai_agents_rs::{Agent, FunctionTool, Runner};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::sleep;

/// Create a weather tool that simulates API delay
fn create_weather_tool() -> Arc<FunctionTool> {
    Arc::new(FunctionTool::new(
        "get_weather".to_string(),
        "Get current weather for a city".to_string(),
        serde_json::json!({
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name"
                }
            },
            "required": ["city"]
        }),
        Box::new(move |args: serde_json::Value| {
            let city = args["city"].as_str().unwrap_or("Unknown");

            // Simulate API delay
            std::thread::sleep(Duration::from_millis(500));

            // Return mock weather data
            Ok(serde_json::json!({
                "city": city,
                "temperature": 22 + (city.len() % 10) as i32,
                "condition": if city.len() % 2 == 0 { "sunny" } else { "cloudy" },
                "humidity": 45 + (city.len() % 20) as i32,
                "wind_speed": 10 + (city.len() % 15) as i32,
            }))
        }),
    ))
}

/// Create a news tool that simulates API delay
fn create_news_tool() -> Arc<FunctionTool> {
    Arc::new(FunctionTool::new(
        "get_news".to_string(),
        "Get latest news headlines for a topic".to_string(),
        serde_json::json!({
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "News topic or city"
                }
            },
            "required": ["topic"]
        }),
        Box::new(move |args: serde_json::Value| {
            let topic = args["topic"].as_str().unwrap_or("general");

            // Simulate API delay
            std::thread::sleep(Duration::from_millis(700));

            // Return mock news data
            Ok(serde_json::json!({
                "topic": topic,
                "headlines": [
                    format!("{} sees record growth in tech sector", topic),
                    format!("New infrastructure project announced in {}", topic),
                    format!("{} hosts international conference next month", topic),
                ],
                "source": "Global News Network",
                "updated": chrono::Utc::now().to_rfc3339(),
            }))
        }),
    ))
}

/// Create a statistics tool that simulates database query
fn create_stats_tool() -> Arc<FunctionTool> {
    Arc::new(FunctionTool::new(
        "get_statistics".to_string(),
        "Get demographic and economic statistics for a city".to_string(),
        serde_json::json!({
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name"
                }
            },
            "required": ["city"]
        }),
        Box::new(move |args: serde_json::Value| {
            let city = args["city"].as_str().unwrap_or("Unknown");

            // Simulate database query delay
            std::thread::sleep(Duration::from_millis(300));

            // Return mock statistics
            Ok(serde_json::json!({
                "city": city,
                "population": 100000 + (city.len() * 50000),
                "gdp_per_capita": 40000 + (city.len() * 2000),
                "unemployment_rate": 3.5 + (city.len() % 5) as f64 * 0.5,
                "growth_rate": 2.0 + (city.len() % 3) as f64,
            }))
        }),
    ))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Parallel Tools Example ===\n");
    println!("This example demonstrates how agents can use multiple tools");
    println!("in parallel to gather information more efficiently.\n");

    // Create an agent with multiple tools
    let agent = Agent::simple(
        "CityInfoAgent",
        "You are a helpful city information assistant. When asked about cities, \
         gather comprehensive information using all available tools: weather, news, \
         and statistics. Present the information in a well-organized format.",
    )
    .with_tool(create_weather_tool())
    .with_tool(create_news_tool())
    .with_tool(create_stats_tool());

    // Test queries that should trigger parallel tool usage
    let queries = ["Tell me everything about Tokyo - weather, news, and statistics",
        "Compare London and Paris - I need weather and population data for both",
        "What's happening in New York? Get me weather, news, and economic stats"];

    for (i, query) in queries.iter().enumerate() {
        println!("\n📝 Query {}: {}", i + 1, query);
        println!("{}", "=".repeat(60));

        let start = Instant::now();

        // Note: The actual parallelization depends on the LLM's decision
        // to call multiple tools in a single response
        let result = Runner::run(agent.clone(), query.to_string(), Default::default()).await?;

        let elapsed = start.elapsed();

        if result.is_success() {
            println!("\n✅ Response:");
            println!("{}", result.final_output);

            // Count tool calls
            let tool_calls = result
                .items
                .iter()
                .filter(|item| matches!(item, openai_agents_rs::items::RunItem::ToolCall(_)))
                .count();

            println!("\n📊 Execution Statistics:");
            println!("  Time taken: {:.2?}", elapsed);
            println!("  Tool calls made: {}", tool_calls);
            println!("  Total tokens: {}", result.usage.total.total_tokens);

            // Estimate time saved by parallel execution
            // (assuming tools were called in parallel vs sequential)
            if tool_calls > 1 {
                let estimated_sequential = Duration::from_millis(500 * tool_calls as u64);
                let time_saved = estimated_sequential.saturating_sub(elapsed);
                if time_saved > Duration::ZERO {
                    println!(
                        "  Estimated time saved by parallel execution: {:.2?}",
                        time_saved
                    );
                }
            }
        } else {
            println!("\n❌ Error: {:?}", result.error);
        }

        if i < queries.len() - 1 {
            println!("\n{}", "-".repeat(60));
            println!("Waiting before next query...");
            sleep(Duration::from_secs(2)).await;
        }
    }

    println!("\n{}", "=".repeat(60));
    println!("✨ Example completed!");

    Ok(())
}
