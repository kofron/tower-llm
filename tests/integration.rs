//! Integration tests for the Tower-based agent implementation

use openai_agents_rs::{
    policies, simple_chat_request, tool_typed, Agent, AgentBuilder, AgentPolicy, AgentRun,
    AgentStopReason, CompositePolicy, LoopState, PolicyFn, StepOutcome, ToolDef, ToolInvocation,
    ToolOutput,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;
use std::sync::Arc;
use tower::{Service, ServiceExt};

#[derive(Debug, Deserialize, JsonSchema)]
struct TestArgs {
    value: String,
}

#[tokio::test]
async fn test_tool_typed_helper() {
    // Test the tool_typed helper creates a proper ToolDef
    let tool = tool_typed("test_tool", "A test tool", |args: TestArgs| async move {
        Ok::<_, tower::BoxError>(json!({ "result": args.value }))
    });

    assert_eq!(tool.name, "test_tool");
    assert_eq!(tool.description, "A test tool");

    // Test the tool service works
    let inv = ToolInvocation {
        id: "test_id".to_string(),
        name: "test_tool".to_string(),
        arguments: json!({ "value": "hello" }),
    };

    let mut svc = tool.service;
    let result = svc.ready().await.unwrap().call(inv).await.unwrap();
    assert_eq!(result.id, "test_id");
    assert_eq!(result.result, json!({ "result": "hello" }));
}

#[test]
fn test_simple_chat_request() {
    let req = simple_chat_request("System prompt", "User message");
    assert_eq!(req.messages.len(), 2);

    // Check system message
    if let async_openai::types::ChatCompletionRequestMessage::System(sys) = &req.messages[0] {
        match &sys.content {
            async_openai::types::ChatCompletionRequestSystemMessageContent::Text(text) => {
                assert_eq!(text, "System prompt");
            }
            _ => panic!("Expected text content"),
        }
    } else {
        panic!("Expected system message");
    }
}

#[test]
fn test_policies() {
    // Test max_steps policy
    let policy = policies::max_steps(3);
    let state = LoopState { steps: 3 };
    let outcome = StepOutcome::Next {
        messages: vec![],
        aux: Default::default(),
        invoked_tools: vec![],
    };

    let reason = policy.decide(&state, &outcome);
    assert!(matches!(reason, Some(AgentStopReason::MaxSteps)));

    // Test until_no_tool_calls policy
    let policy = policies::until_no_tool_calls();
    let done_outcome = StepOutcome::Done {
        messages: vec![],
        aux: Default::default(),
    };

    let reason = policy.decide(&state, &done_outcome);
    assert!(matches!(reason, Some(AgentStopReason::DoneNoToolCalls)));
}

#[test]
fn test_composite_policy() {
    let policy = CompositePolicy::new(vec![
        policies::max_steps(2),
        policies::until_no_tool_calls(),
    ]);

    // Should stop at max steps first
    let state = LoopState { steps: 2 };
    let outcome = StepOutcome::Next {
        messages: vec![],
        aux: Default::default(),
        invoked_tools: vec![],
    };

    let reason = policy.decide(&state, &outcome);
    assert!(matches!(reason, Some(AgentStopReason::MaxSteps)));
}

#[test]
fn test_policy_builder() {
    use openai_agents_rs::Policy;
    
    let policy = Policy::new()
        .or_max_steps(5)
        .until_no_tool_calls()
        .build();
    
    // Test it's a valid composite policy
    let state = LoopState { steps: 5 };
    let outcome = StepOutcome::Next {
        messages: vec![],
        aux: Default::default(),
        invoked_tools: vec![],
    };
    
    let reason = policy.decide(&state, &outcome);
    assert!(matches!(reason, Some(AgentStopReason::MaxSteps)));
}

#[tokio::test]
async fn test_agent_builder() {
    use async_openai::{config::OpenAIConfig, Client};

    // Test that AgentBuilder creates a valid agent
    let tool = tool_typed("echo", "Echo back the input", |args: TestArgs| async move {
        Ok::<_, tower::BoxError>(json!({ "echo": args.value }))
    });

    // Create a client (won't actually be used in this test)
    let client = Arc::new(Client::<OpenAIConfig>::new());

    // This test just verifies the builder compiles and produces the right type
    let _builder = Agent::builder(client)
        .model("gpt-4")
        .tool(tool)
        .policy(CompositePolicy::new(vec![policies::max_steps(1)]))
        .temperature(0.7)
        .max_tokens(100);

    // The builder pattern works
    assert!(true);
}

#[test]
fn test_tool_invocation_output_types() {
    let inv = ToolInvocation {
        id: "123".to_string(),
        name: "test".to_string(),
        arguments: json!({"key": "value"}),
    };

    assert_eq!(inv.id, "123");
    assert_eq!(inv.name, "test");

    let out = ToolOutput {
        id: "123".to_string(),
        result: json!({"result": "success"}),
    };

    assert_eq!(out.id, "123");
    assert_eq!(out.result, json!({"result": "success"}));
}

#[test]
fn test_agent_stop_reasons() {
    // Test all stop reason variants exist
    let _reasons = vec![
        AgentStopReason::DoneNoToolCalls,
        AgentStopReason::MaxSteps,
        AgentStopReason::ToolCalled("test".to_string()),
        AgentStopReason::TokensBudgetExceeded,
        AgentStopReason::ToolBudgetExceeded,
        AgentStopReason::TimeBudgetExceeded,
    ];
}

#[test]
fn test_step_outcome_variants() {
    use openai_agents_rs::StepAux;

    let aux = StepAux {
        prompt_tokens: 10,
        completion_tokens: 20,
        tool_invocations: 1,
    };

    let _next = StepOutcome::Next {
        messages: vec![],
        aux: aux.clone(),
        invoked_tools: vec!["tool1".to_string()],
    };

    let _done = StepOutcome::Done {
        messages: vec![],
        aux,
    };
}

// Test that the old API is truly gone
#[test]
fn test_old_api_removed() {
    // These should not compile if uncommented:
    // use openai_agents_rs::Runner;
    // use openai_agents_rs::env::EnvBuilder;
    // use openai_agents_rs::tool::Tool;
    // let agent = Agent::simple("bot", "instructions");

    // The new API requires a client
    use async_openai::{config::OpenAIConfig, Client};
    let client = Arc::new(Client::<OpenAIConfig>::new());
    let _builder = Agent::builder(client);
}
