//! # Advanced Example: Multi-Tool Calculator Agent
//!
//! This example showcases an agent that can solve complex mathematical problems
//! by using a suite of calculator tools. It demonstrates how an agent can be
//! instructed to break down a problem into smaller steps and use different tools
//! in sequence to arrive at a final answer.
//!
//! ## Key Concepts Demonstrated
//!
//! - **Multiple Tools**: The agent is equipped with a variety of tools for
//!   basic arithmetic operations (add, subtract, multiply, divide), as well as
//!   more advanced functions like powers and square roots.
//! - **Complex Problem Solving**: The agent's instructions guide it to decompose
//!   complex problems and solve them step-by-step, showing its work.
//! - **Multi-Turn Tool Use**: The agent can perform multiple tool calls across
//!   several turns to solve a single problem.
//! - **Interactive Mode**: After running through a set of predefined problems,
//!   the example enters an interactive mode where you can input your own math
//!   problems for the agent to solve.
//!
//! To run this example, you first need to set your `OPENAI_API_KEY` environment
//! variable.
//!
//! ```bash
//! export OPENAI_API_KEY="your-api-key"
//! cargo run --example calculator
//! ```

use openai_agents_rs::{runner::RunConfig, Agent, FunctionTool, Runner};
use serde_json::Value;
use std::sync::Arc;

/// Creates a tool for adding two numbers.
///
/// This function demonstrates the use of `FunctionTool::new` to create a tool
/// with a custom parameter schema, allowing for multiple, named arguments.
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

/// Creates a tool for subtracting one number from another.
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

/// Creates a tool for multiplying two numbers.
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

/// Creates a tool for dividing one number by another.
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

/// Creates a tool for raising a number to a power.
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

/// Creates a tool for calculating the square root of a number.
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
                return Ok(Value::String(
                    "Error: Cannot take square root of negative number".to_string(),
                ));
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

    // 1. Create the agent with a suite of calculator tools.
    //
    // The agent's instructions are crucial here. They guide the agent to
    // break down problems and use the tools in a step-by-step manner.
    // We also increase `max_turns` to allow for more complex calculations.
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

    // 2. Run the agent with a set of predefined problems.
    //
    // These problems are designed to test the agent's ability to use multiple
    // tools in sequence to arrive at a solution.
    let problems = ["What is (15 + 25) * 3?",
        "Calculate the area of a rectangle with length 12.5 and width 8.3, then add 15 to the result.",
        "If I have $100 and spend $35.50, then earn $20.75, how much do I have?",
        "What is 2^3 + sqrt(16) - 5?",
        "Calculate: ((10 + 5) * 2) / 3"];

    for (i, problem) in problems.iter().enumerate() {
        println!("Problem {}: {}", i + 1, problem);
        println!("{}", "-".repeat(50));

        let result = Runner::run(agent.clone(), problem.to_string(), RunConfig::default()).await?;

        if result.is_success() {
            println!("\nFinal Answer: {}\n", result.final_output);

            // Count how many tool calls were made to show the agent's work.
            let tool_calls = result
                .items
                .iter()
                .filter(|item| matches!(item, openai_agents_rs::items::RunItem::ToolCall(_)))
                .count();

            println!(
                "(Used {} tool calls across {} turns)",
                tool_calls,
                result
                    .items
                    .iter()
                    .filter(|item| {
                        matches!(item, openai_agents_rs::items::RunItem::Message(m)
                        if m.role == openai_agents_rs::items::Role::Assistant)
                    })
                    .count()
            );
        } else {
            println!("Error: {:?}", result.error());
        }

        println!("\n{}\n", "=".repeat(60));
    }

    // 3. Enter interactive mode.
    //
    // After the predefined problems, the example enters a loop where you can
    // provide your own math problems for the agent to solve.
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

        let result = Runner::run(agent.clone(), input, RunConfig::default()).await?;

        if result.is_success() {
            println!("\nAnswer: {}\n", result.final_output);
        } else {
            println!("\nError: {:?}\n", result.error());
        }
    }

    Ok(())
}
