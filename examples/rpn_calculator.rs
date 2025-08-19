//! # Example: RPN Calculator with Stateful Tool
//!
//! This example demonstrates a Reverse Polish Notation (RPN) calculator
//! implemented as a stateful tool that maintains its own stack.
//!
//! To run this example:
//!
//! ```bash
//! export OPENAI_API_KEY="your-api-key"
//! cargo run --example rpn_calculator
//! ```
//!
//! Expected: The agent should be able to compute the RPN expression and return the result.
//! The answer should be 16.

use openai_agents_rs::{runner::RunConfig, Agent, FunctionTool, Runner};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};

/// Stateful RPN calculator that maintains a stack
struct RpnCalculator {
    stack: Arc<Mutex<Vec<f64>>>,
}

impl RpnCalculator {
    fn new() -> Self {
        Self {
            stack: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn execute(
        &self,
        op: &str,
        value: Option<f64>,
    ) -> Result<Value, openai_agents_rs::error::AgentsError> {
        let mut stack = self.stack.lock().unwrap();

        match op {
            "push" => {
                if let Some(v) = value {
                    stack.push(v);
                    Ok(json!({
                        "operation": "push",
                        "value": v,
                        "stack": stack.clone(),
                        "top": stack.last().cloned()
                    }))
                } else {
                    Err(openai_agents_rs::error::AgentsError::Other(
                        "Push operation requires a value".to_string(),
                    ))
                }
            }
            "add" => {
                if stack.len() < 2 {
                    Err(openai_agents_rs::error::AgentsError::Other(
                        "Add requires at least 2 values on stack".to_string(),
                    ))
                } else {
                    let b = stack.pop().unwrap();
                    let a = stack.pop().unwrap();
                    let result = a + b;
                    stack.push(result);
                    Ok(json!({
                        "operation": "add",
                        "operands": [a, b],
                        "result": result,
                        "stack": stack.clone(),
                        "top": result
                    }))
                }
            }
            "sub" => {
                if stack.len() < 2 {
                    Err(openai_agents_rs::error::AgentsError::Other(
                        "Subtract requires at least 2 values on stack".to_string(),
                    ))
                } else {
                    let b = stack.pop().unwrap();
                    let a = stack.pop().unwrap();
                    let result = a - b;
                    stack.push(result);
                    Ok(json!({
                        "operation": "sub",
                        "operands": [a, b],
                        "result": result,
                        "stack": stack.clone(),
                        "top": result
                    }))
                }
            }
            "mul" => {
                if stack.len() < 2 {
                    Err(openai_agents_rs::error::AgentsError::Other(
                        "Multiply requires at least 2 values on stack".to_string(),
                    ))
                } else {
                    let b = stack.pop().unwrap();
                    let a = stack.pop().unwrap();
                    let result = a * b;
                    stack.push(result);
                    Ok(json!({
                        "operation": "mul",
                        "operands": [a, b],
                        "result": result,
                        "stack": stack.clone(),
                        "top": result
                    }))
                }
            }
            "div" => {
                if stack.len() < 2 {
                    Err(openai_agents_rs::error::AgentsError::Other(
                        "Divide requires at least 2 values on stack".to_string(),
                    ))
                } else {
                    let b = stack.pop().unwrap();
                    let a = stack.pop().unwrap();
                    if b == 0.0 {
                        // Put values back on divide by zero
                        stack.push(a);
                        stack.push(b);
                        Err(openai_agents_rs::error::AgentsError::Other(
                            "Division by zero".to_string(),
                        ))
                    } else {
                        let result = a / b;
                        stack.push(result);
                        Ok(json!({
                            "operation": "div",
                            "operands": [a, b],
                            "result": result,
                            "stack": stack.clone(),
                            "top": result
                        }))
                    }
                }
            }
            "peek" => Ok(json!({
                "operation": "peek",
                "stack": stack.clone(),
                "top": stack.last().cloned(),
                "size": stack.len()
            })),
            "clear" => {
                stack.clear();
                Ok(json!({
                    "operation": "clear",
                    "stack": stack.clone(),
                    "message": "Stack cleared"
                }))
            }
            _ => Err(openai_agents_rs::error::AgentsError::Other(format!(
                "Unknown operation: {}",
                op
            ))),
        }
    }

    fn create_tool(self: Arc<Self>) -> Arc<FunctionTool> {
        let schema = json!({
            "type": "object",
            "properties": {
                "op": {
                    "type": "string",
                    "enum": ["push", "add", "sub", "mul", "div", "peek", "clear"],
                    "description": "RPN operation to perform"
                },
                "value": {
                    "type": ["number", "null"],
                    "description": "Value to push when op == 'push'"
                }
            },
            "required": ["op"]
        });

        let calculator = self.clone();
        let func = move |args: Value| -> Result<Value, openai_agents_rs::error::AgentsError> {
            let op = args.get("op").and_then(|v| v.as_str()).unwrap_or("peek");
            let value = args.get("value").and_then(|v| v.as_f64());

            calculator.execute(op, value)
        };

        Arc::new(FunctionTool::new(
            "rpn".to_string(),
            "RPN calculator operations: push values onto stack, perform arithmetic operations (add, sub, mul, div), peek at stack state, or clear stack".to_string(),
            schema,
            func,
        ))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create the RPN calculator
    let calculator = Arc::new(RpnCalculator::new());

    // Create the tool
    let rpn_tool = calculator.clone().create_tool();

    // Create the agent with the RPN tool
    let agent = Agent::simple(
        "RpnCalculator",
        "You are an RPN calculator assistant. Help the user compute expressions using Reverse Polish Notation.

RPN works by pushing values onto a stack, then performing operations that pop operands and push results.
For example, to compute (3 + 5) * 2:
1. Push 3
2. Push 5  
3. Add (pops 3 and 5, pushes 8)
4. Push 2
5. Multiply (pops 8 and 2, pushes 16)

Always peek at the stack after operations to show the current state."
    )
    .with_tool(rpn_tool);

    // Test the RPN calculator
    let input = "Calculate (3 + 5) * 2 using RPN. Show me each step.";

    println!("🧮 RPN Calculator Example");
    println!("Input: {}\n", input);

    let config = RunConfig::default();
    let result = Runner::run(agent, input, config).await?;

    if result.is_success() {
        println!("\n✅ Calculation completed!");
        println!("Final output: {}", result.final_output);

        // Show the final stack state
        println!("\n📊 Final stack state:");
        let final_state = calculator.execute("peek", None)?;
        println!("{}", serde_json::to_string_pretty(&final_state)?);
    } else {
        println!("\n❌ Calculation failed");
        if let Some(error) = result.error {
            println!("Error: {}", error);
        }
    }

    Ok(())
}
