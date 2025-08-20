//! Example demonstrating multi-agent routing with the groups module.
//! Shows how to route requests between specialized agents.

use std::sync::Arc;

use tower::{Service, ServiceExt};

// Import the next module and its submodules
// Core module is now at root level
// use tower_llm directly

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("=== Multi-Agent Groups Example ===\n");

    println!("Setting up specialized agents:");

    // Create specialized mock agents using factories

    // Math agent factory
    let create_math_agent = || -> tower_llm::AgentSvc {
        tower::util::BoxService::new(tower::service_fn(
            |_req: async_openai::types::CreateChatCompletionRequest| async move {
                println!("    [Math Agent] Processing request");
                Ok::<_, tower::BoxError>(tower_llm::AgentRun {
                    messages: vec![
                        async_openai::types::ChatCompletionRequestAssistantMessageArgs::default()
                            .content("I'm the math specialist. I can help with calculations.")
                            .build()?
                            .into(),
                    ],
                    steps: 1,
                    stop: tower_llm::AgentStopReason::DoneNoToolCalls,
                })
            },
        ))
    };

    // Writing agent factory
    let create_writing_agent = || -> tower_llm::AgentSvc {
        tower::util::BoxService::new(tower::service_fn(
            |_req: async_openai::types::CreateChatCompletionRequest| async move {
                println!("    [Writing Agent] Processing request");
                Ok::<_, tower::BoxError>(tower_llm::AgentRun {
                    messages: vec![
                        async_openai::types::ChatCompletionRequestAssistantMessageArgs::default()
                            .content("I'm the writing specialist. I excel at creative text.")
                            .build()?
                            .into(),
                    ],
                    steps: 1,
                    stop: tower_llm::AgentStopReason::DoneNoToolCalls,
                })
            },
        ))
    };

    // Code agent factory
    let create_code_agent = || -> tower_llm::AgentSvc {
        tower::util::BoxService::new(tower::service_fn(
            |_req: async_openai::types::CreateChatCompletionRequest| async move {
                println!("    [Code Agent] Processing request");
                Ok::<_, tower::BoxError>(tower_llm::AgentRun {
                    messages: vec![
                        async_openai::types::ChatCompletionRequestAssistantMessageArgs::default()
                            .content("I'm the coding specialist. I can help with programming.")
                            .build()?
                            .into(),
                    ],
                    steps: 1,
                    stop: tower_llm::AgentStopReason::DoneNoToolCalls,
                })
            },
        ))
    };

    println!("  ✅ Math Agent - Handles calculations");
    println!("  ✅ Writing Agent - Handles text generation");
    println!("  ✅ Code Agent - Handles programming\n");

    println!("--- Example 1: Content-Based Routing ---");

    // Create a content-based picker
    let content_picker = tower::service_fn(|req: tower_llm::groups::PickRequest| async move {
        // Examine the last user message to determine routing
        let last_user_msg = req.messages.iter().rev().find(|m| {
            matches!(
                m,
                async_openai::types::ChatCompletionRequestMessage::User(_)
            )
        });

        let agent_name = if let Some(msg) = last_user_msg {
            // Simple keyword-based routing
            let content = format!("{:?}", msg).to_lowercase();
            if content.contains("calculate")
                || content.contains("math")
                || content.contains("number")
            {
                "math"
            } else if content.contains("write")
                || content.contains("story")
                || content.contains("poem")
            {
                "writing"
            } else if content.contains("code")
                || content.contains("program")
                || content.contains("function")
            {
                "code"
            } else {
                "math" // default
            }
        } else {
            "math"
        };

        println!("  Picker selected: {} agent", agent_name);
        Ok::<_, tower::BoxError>(agent_name.to_string())
    });

    // Build the group router
    let router = tower_llm::groups::GroupBuilder::new()
        .agent("math", create_math_agent())
        .agent("writing", create_writing_agent())
        .agent("code", create_code_agent())
        .picker(content_picker)
        .build();

    let mut group_service = router;

    // Test different types of requests
    let test_requests = vec![
        ("Calculate 15 * 7", "math"),
        ("Write a haiku about spring", "writing"),
        ("Create a Python function", "code"),
    ];

    for (user_msg, expected_agent) in test_requests {
        println!("\n  User: '{}'", user_msg);
        println!("  Expected routing: {} agent", expected_agent);

        let req = async_openai::types::CreateChatCompletionRequestArgs::default()
            .model("gpt-4o")
            .messages(vec![
                async_openai::types::ChatCompletionRequestUserMessageArgs::default()
                    .content(user_msg)
                    .build()?
                    .into(),
            ])
            .build()?;

        let result = group_service.ready().await?.call(req).await?;
        println!("  Result: {} steps taken", result.steps);
    }

    println!("\n--- Example 2: Round-Robin Routing ---");

    let round_robin_counter = Arc::new(tokio::sync::Mutex::new(0usize));
    let agents = vec!["math", "writing", "code"];

    let round_robin_picker = {
        let counter = round_robin_counter.clone();
        let agents = agents.clone();
        tower::service_fn(move |_req: tower_llm::groups::PickRequest| {
            let counter = counter.clone();
            let agents = agents.clone();
            async move {
                let mut count = counter.lock().await;
                let agent = agents[*count % agents.len()].to_string();
                *count += 1;
                println!("  Round-robin selected: {} agent", agent);
                Ok::<_, tower::BoxError>(agent)
            }
        })
    };

    let round_robin_router = tower_llm::groups::GroupBuilder::new()
        .agent("math", create_math_agent())
        .agent("writing", create_writing_agent())
        .agent("code", create_code_agent())
        .picker(round_robin_picker)
        .build();

    let mut rr_service = round_robin_router;

    println!("\n  Testing round-robin distribution:");
    for i in 1..=6 {
        let req = async_openai::types::CreateChatCompletionRequestArgs::default()
            .model("gpt-4o")
            .messages(vec![
                async_openai::types::ChatCompletionRequestUserMessageArgs::default()
                    .content(format!("Request {}", i))
                    .build()?
                    .into(),
            ])
            .build()?;

        let _ = rr_service.ready().await?.call(req).await?;
    }

    println!("\n--- Example 3: Load-Based Routing ---");

    // Track agent loads
    let agent_loads = Arc::new(tokio::sync::Mutex::new(std::collections::HashMap::from([
        ("math".to_string(), 0),
        ("writing".to_string(), 0),
        ("code".to_string(), 0),
    ])));

    let load_picker = {
        let loads = agent_loads.clone();
        tower::service_fn(move |_req: tower_llm::groups::PickRequest| {
            let loads = loads.clone();
            async move {
                let mut load_map = loads.lock().await;

                // Find agent with lowest load
                let agent = load_map
                    .iter()
                    .min_by_key(|(_, &load)| load)
                    .map(|(name, _)| name.clone())
                    .unwrap_or_else(|| "math".to_string());

                // Increment load for selected agent
                *load_map.get_mut(&agent).unwrap() += 1;

                println!(
                    "  Load balancer selected: {} (current loads: {:?})",
                    agent, *load_map
                );
                Ok::<_, tower::BoxError>(agent)
            }
        })
    };

    let load_balanced_router = tower_llm::groups::GroupBuilder::new()
        .agent("math", create_math_agent())
        .agent("writing", create_writing_agent())
        .agent("code", create_code_agent())
        .picker(load_picker)
        .build();

    let mut lb_service = load_balanced_router;

    println!("\n  Testing load-balanced distribution:");
    for i in 1..=6 {
        let req = async_openai::types::CreateChatCompletionRequestArgs::default()
            .model("gpt-4o")
            .messages(vec![
                async_openai::types::ChatCompletionRequestUserMessageArgs::default()
                    .content(format!("Task {}", i))
                    .build()?
                    .into(),
            ])
            .build()?;

        let _ = lb_service.ready().await?.call(req).await?;

        // Simulate some agents finishing work
        if i % 2 == 0 {
            let mut loads = agent_loads.lock().await;
            // Decrement a random agent's load
            if let Some(load) = loads.get_mut("math") {
                *load = (*load as usize).saturating_sub(1);
            }
        }
    }

    println!("\n=== Routing Strategies ===");
    println!("1. **Content-Based**: Route by analyzing message content");
    println!("2. **Round-Robin**: Distribute evenly across agents");
    println!("3. **Load-Based**: Route to least busy agent");
    println!("4. **Capability-Based**: Route based on agent capabilities");
    println!("5. **Priority-Based**: Route critical requests to best agents");

    println!("\n=== Key Takeaways ===");
    println!("1. GroupRouter enables multi-agent architectures");
    println!("2. Picker services implement custom routing logic");
    println!("3. Agents can be specialized for different domains");
    println!("4. Routing strategies can be swapped dynamically");
    println!("5. Enables horizontal scaling of agent workloads");
    println!("6. Perfect for building agent teams and workflows");

    Ok(())
}
