//! Calculator example demonstrating multiple tool calls and rounds
//!
//! This example shows how an agent can use multiple calculator tools
//! to solve complex mathematical problems step by step.
//!
//! Run with: cargo run --example calculator

use openai_agents_rs::{runner::RunConfig, Agent, FunctionTool, Runner};
use serde_json::Value;
use std::sync::Arc;

/// Create a calculator tool for basic arithmetic
fn create_add_tool() -> Arc<FunctionTool> {
    Arc::new(FunctionTool::new(
        "add".to_string(),
        "Add two numbers together".to_string(),
        serde_json::json!({
            "type": "object",
            "properties": {
                "a": {
                    "type": "number",
                    "description": "First number"
                },
                "b": {
                    "type": "number",
                    "description": "Second number"
                }
            },
            "required": ["a", "b"]
        }),
        |args| {
            let a = args.get("a").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let b = args.get("b").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let result = a + b;
            println!("  [Tool] Adding {} + {} = {}", a, b, result);
            Ok(Value::Number(serde_json::Number::from_f64(result).unwrap()))
        },
    ))
}

fn create_subtract_tool() -> Arc<FunctionTool> {
    Arc::new(FunctionTool::new(
        "subtract".to_string(),
        "Subtract one number from another".to_string(),
        serde_json::json!({
            "type": "object",
            "properties": {
                "a": {
                    "type": "number",
                    "description": "Number to subtract from"
                },
                "b": {
                    "type": "number",
                    "description": "Number to subtract"
                }
            },
            "required": ["a", "b"]
        }),
        |args| {
            let a = args.get("a").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let b = args.get("b").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let result = a - b;
            println!("  [Tool] Subtracting {} - {} = {}", a, b, result);
            Ok(Value::Number(serde_json::Number::from_f64(result).unwrap()))
        },
    ))
}

fn create_multiply_tool() -> Arc<FunctionTool> {
    Arc::new(FunctionTool::new(
        "multiply".to_string(),
        "Multiply two numbers".to_string(),
        serde_json::json!({
            "type": "object",
            "properties": {
                "a": {
                    "type": "number",
                    "description": "First number"
                },
                "b": {
                    "type": "number",
                    "description": "Second number"
                }
            },
            "required": ["a", "b"]
        }),
        |args| {
            let a = args.get("a").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let b = args.get("b").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let result = a * b;
            println!("  [Tool] Multiplying {} × {} = {}", a, b, result);
            Ok(Value::Number(serde_json::Number::from_f64(result).unwrap()))
        },
    ))
}

fn create_divide_tool() -> Arc<FunctionTool> {
    Arc::new(FunctionTool::new(
        "divide".to_string(),
        "Divide one number by another".to_string(),
        serde_json::json!({
            "type": "object",
            "properties": {
                "a": {
                    "type": "number",
                    "description": "Dividend (number to be divided)"
                },
                "b": {
                    "type": "number",
                    "description": "Divisor (number to divide by)"
                }
            },
            "required": ["a", "b"]
        }),
        |args| {
            let a = args.get("a").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let b = args.get("b").and_then(|v| v.as_f64()).unwrap_or(1.0);
            if b == 0.0 {
                println!("  [Tool] Error: Division by zero!");
                return Ok(Value::String("Error: Cannot divide by zero".to_string()));
            }
            let result = a / b;
            println!("  [Tool] Dividing {} ÷ {} = {}", a, b, result);
            Ok(Value::Number(serde_json::Number::from_f64(result).unwrap()))
        },
    ))
}

fn create_power_tool() -> Arc<FunctionTool> {
    Arc::new(FunctionTool::new(
        "power".to_string(),
        "Raise a number to a power".to_string(),
        serde_json::json!({
            "type": "object",
            "properties": {
                "base": {
                    "type": "number",
                    "description": "Base number"
                },
                "exponent": {
                    "type": "number",
                    "description": "Exponent"
                }
            },
            "required": ["base", "exponent"]
        }),
        |args| {
            let base = args.get("base").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let exponent = args.get("exponent").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let result = base.powf(exponent);
            println!("  [Tool] Calculating {}^{} = {}", base, exponent, result);
            Ok(Value::Number(serde_json::Number::from_f64(result).unwrap()))
        },
    ))
}

fn create_sqrt_tool() -> Arc<FunctionTool> {
    Arc::new(FunctionTool::new(
        "sqrt".to_string(),
        "Calculate the square root of a number".to_string(),
        serde_json::json!({
            "type": "object",
            "properties": {
                "number": {
                    "type": "number",
                    "description": "Number to find the square root of"
                }
            },
            "required": ["number"]
        }),
        |args| {
            let number = args.get("number").and_then(|v| v.as_f64()).unwrap_or(0.0);
            if number < 0.0 {
                println!("  [Tool] Error: Cannot take square root of negative number!");
                return Ok(Value::String("Error: Cannot take square root of negative number".to_string()));
            }
            let result = number.sqrt();
            println!("  [Tool] Calculating √{} = {}", number, result);
            Ok(Value::Number(serde_json::Number::from_f64(result).unwrap()))
        },
    ))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Calculator Agent Example ===\n");

    // Create an agent with calculator tools
    let agent = Agent::simple(
        "MathBot",
        "You are a helpful mathematics assistant. Use the calculator tools to solve mathematical problems step by step. 
         Always show your work by using the tools for each calculation.
         When given a complex problem, break it down into steps and solve each part."
    )
    .with_tools(vec![
        create_add_tool(),
        create_subtract_tool(),
        create_multiply_tool(),
        create_divide_tool(),
        create_power_tool(),
        create_sqrt_tool(),
    ])
    .with_max_turns(15); // Allow more turns for complex calculations

    // Example problems that require multiple steps
    let problems = ["What is (15 + 25) * 3?",
        "Calculate the area of a rectangle with length 12.5 and width 8.3, then add 15 to the result.",
        "If I have $100 and spend $35.50, then earn $20.75, how much do I have?",
        "What is 2^3 + sqrt(16) - 5?",
        "Calculate: ((10 + 5) * 2) / 3"];

    for (i, problem) in problems.iter().enumerate() {
        println!("Problem {}: {}", i + 1, problem);
        println!("{}", "-".repeat(50));

        let result = Runner::run(
            agent.clone(),
            problem.to_string(),
            RunConfig::default(),
        )
        .await?;

        if result.is_success() {
            println!("\nFinal Answer: {}\n", result.final_output);
            
            // Count how many tool calls were made
            let tool_calls = result.items.iter().filter(|item| {
                matches!(item, openai_agents_rs::items::RunItem::ToolCall(_))
            }).count();
            
            println!("(Used {} tool calls across {} turns)", 
                tool_calls, 
                result.items.iter().filter(|item| {
                    matches!(item, openai_agents_rs::items::RunItem::Message(m) 
                        if m.role == openai_agents_rs::items::Role::Assistant)
                }).count()
            );
        } else {
            println!("Error: {:?}", result.error());
        }
        
        println!("\n{}\n", "=".repeat(60));
    }

    // Interactive mode
    println!("Now entering interactive mode. Type 'quit' to exit.\n");
    
    use std::io::{self, Write};
    
    loop {
        print!("Enter a math problem: ");
        io::stdout().flush()?;
        
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();
        
        if input.eq_ignore_ascii_case("quit") || input.eq_ignore_ascii_case("exit") {
            println!("Goodbye!");
            break;
        }
        
        println!("\nCalculating...\n");
        
        let result = Runner::run(
            agent.clone(),
            input,
            RunConfig::default(),
        )
        .await?;
        
        if result.is_success() {
            println!("\nAnswer: {}\n", result.final_output);
        } else {
            println!("\nError: {:?}\n", result.error());
        }
    }

    Ok(())
}
