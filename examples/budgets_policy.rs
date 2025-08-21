//! Example demonstrating budget policies for controlling agent execution.
//! Shows how to limit agent runs based on token usage, tool invocations, and time.

use std::time::Duration;
use tower_llm::AgentPolicy;

// Import the next module and its submodules
// Core module is now at root level
// use tower_llm directly

#[tokio::main]
async fn main() {
    println!("=== Budget Policies Example ===\n");

    // Create different budget configurations
    let token_budget = tower_llm::budgets::Budget {
        max_prompt_tokens: Some(1000),
        max_completion_tokens: Some(500),
        max_tool_invocations: None,
        max_time: None,
    };

    let tool_budget = tower_llm::budgets::Budget {
        max_prompt_tokens: None,
        max_completion_tokens: None,
        max_tool_invocations: Some(5),
        max_time: None,
    };

    let _time_budget = tower_llm::budgets::Budget {
        max_prompt_tokens: None,
        max_completion_tokens: None,
        max_tool_invocations: None,
        max_time: Some(Duration::from_secs(30)),
    };

    let combined_budget = tower_llm::budgets::Budget {
        max_prompt_tokens: Some(2000),
        max_completion_tokens: Some(1000),
        max_tool_invocations: Some(10),
        max_time: Some(Duration::from_secs(60)),
    };

    println!("--- Token Budget Policy ---");
    println!("Max prompt tokens: 1000");
    println!("Max completion tokens: 500");

    let token_policy = tower_llm::budgets::budget_policy(token_budget);

    // Simulate multiple steps
    let mut total_prompt = 0;
    let mut total_completion = 0;

    for step in 1..=5 {
        total_prompt += 250;
        total_completion += 150;

        let state = tower_llm::LoopState { steps: step };
        let outcome = tower_llm::StepOutcome::Next {
            messages: vec![],
            aux: tower_llm::StepAux {
                prompt_tokens: 250,
                completion_tokens: 150,
                tool_invocations: 0,
            },
            invoked_tools: vec![],
        };

        // Check if policy would stop
        let stop_reason = token_policy.decide(&state, &outcome);

        println!(
            "  Step {}: {} prompt, {} completion tokens total",
            step, total_prompt, total_completion
        );

        if let Some(reason) = stop_reason {
            println!("  ❌ Would stop due to: {:?}", reason);
            break;
        } else {
            println!("  ✅ Within budget");
        }
    }

    println!("\n--- Tool Invocation Budget Policy ---");
    println!("Max tool invocations: 5");

    let tool_policy = tower_llm::budgets::budget_policy(tool_budget);
    let mut total_tools = 0;

    for step in 1..=7 {
        total_tools += 1;

        let state = tower_llm::LoopState { steps: step };
        let outcome = tower_llm::StepOutcome::Next {
            messages: vec![],
            aux: tower_llm::StepAux {
                prompt_tokens: 0,
                completion_tokens: 0,
                tool_invocations: 1,
            },
            invoked_tools: vec!["tool".to_string()],
        };

        let stop_reason = tool_policy.decide(&state, &outcome);

        println!("  Step {}: {} tools invoked total", step, total_tools);

        if let Some(reason) = stop_reason {
            println!("  ❌ Would stop due to: {:?}", reason);
            break;
        } else {
            println!("  ✅ Within budget");
        }
    }

    println!("\n--- Time Budget Policy ---");
    println!("Max time: 30 seconds");

    // In a real scenario, this would track actual execution time
    println!("  (Would stop execution after 30 seconds)");

    println!("\n--- Combined Budget Policy ---");
    println!("Multiple limits active simultaneously:");
    println!("  - Max prompt tokens: 2000");
    println!("  - Max completion tokens: 1000");
    println!("  - Max tool invocations: 10");
    println!("  - Max time: 60 seconds");

    let _combined_policy = tower_llm::budgets::budget_policy(combined_budget);

    println!("\n=== Key Takeaways ===");
    println!("1. Budget policies provide fine-grained control over agent execution");
    println!("2. Token budgets prevent excessive LLM costs");
    println!("3. Tool invocation limits prevent runaway tool usage");
    println!("4. Time budgets ensure bounded execution time");
    println!("5. Budgets can be combined for comprehensive resource control");
    println!("6. Integrates seamlessly with CompositePolicy for agent loops");
}
