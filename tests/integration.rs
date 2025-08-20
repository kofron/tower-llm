#![allow(deprecated)]
//! Integration tests for the Tower-based agent implementation

use async_openai::types::{
    ChatCompletionMessageToolCall, ChatCompletionResponseMessage, ChatCompletionToolType,
    FunctionCall, Role as RespRole,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;
use std::sync::Arc;
use tokio::time::{sleep, Duration};
use tower::{Service, ServiceExt};

use tower::ServiceBuilder;
use tower_llm::provider::{FixedProvider, ProviderResponse};
use tower_llm::AgentPolicy;
use tower_llm::ToolJoinPolicy;
use tower_llm::{
    policies, simple_chat_request, tool_typed, Agent, AgentStopReason, CompositePolicy, LoopState,
    StepOutcome, ToolInvocation, ToolOutput,
};

#[derive(Debug, Deserialize, JsonSchema)]
struct TestArgs {
    value: String,
}

#[tokio::test]
async fn test_step_parallel_tools_preserve_order() {
    use async_openai::{config::OpenAIConfig, Client};

    let slow = tool_typed("slow", "slow tool", |_args: serde_json::Value| async move {
        sleep(Duration::from_millis(50)).await;
        Ok::<_, tower::BoxError>(serde_json::json!({"label":"slow"}))
    });
    let fast = tool_typed("fast", "fast tool", |_args: serde_json::Value| async move {
        sleep(Duration::from_millis(5)).await;
        Ok::<_, tower::BoxError>(serde_json::json!({"label":"fast"}))
    });

    let tc1 = ChatCompletionMessageToolCall {
        id: "c1".to_string(),
        r#type: ChatCompletionToolType::Function,
        function: FunctionCall {
            name: "slow".to_string(),
            arguments: "{}".to_string(),
        },
    };
    let tc2 = ChatCompletionMessageToolCall {
        id: "c2".to_string(),
        r#type: ChatCompletionToolType::Function,
        function: FunctionCall {
            name: "fast".to_string(),
            arguments: "{}".to_string(),
        },
    };
    let assistant = ChatCompletionResponseMessage {
        content: None,
        role: RespRole::Assistant,
        tool_calls: Some(vec![tc1, tc2]),
        function_call: None,
        refusal: None,
        audio: None,
    };
    let provider = FixedProvider::new(ProviderResponse {
        assistant,
        prompt_tokens: 1,
        completion_tokens: 1,
    });

    let client = Arc::new(Client::<OpenAIConfig>::new());
    let mut agent = Agent::builder(client)
        .model("gpt-4o")
        .tool(slow)
        .tool(fast)
        .with_provider(provider)
        .parallel_tools(true)
        .policy(CompositePolicy::new(vec![policies::max_steps(1)]))
        .build();

    let out = ServiceExt::ready(&mut agent)
        .await
        .unwrap()
        .call(simple_chat_request("sys", "hi"))
        .await
        .unwrap();

    let mut labels: Vec<String> = Vec::new();
    for m in out.messages.iter() {
        if let async_openai::types::ChatCompletionRequestMessage::Tool(t) = m {
            if let async_openai::types::ChatCompletionRequestToolMessageContent::Text(txt) =
                &t.content
            {
                let v: serde_json::Value =
                    serde_json::from_str(txt).unwrap_or(serde_json::Value::Null);
                if let Some(lbl) = v.get("label").and_then(|x| x.as_str()) {
                    labels.push(lbl.to_string());
                }
            }
        }
    }
    assert_eq!(labels, vec!["slow".to_string(), "fast".to_string()]);
}

#[tokio::test]
async fn test_step_parallel_tools_error_propagation() {
    use async_openai::{config::OpenAIConfig, Client};
    let good = tool_typed("good", "good", |_args: serde_json::Value| async move {
        Ok::<_, tower::BoxError>(serde_json::json!({"ok":true}))
    });
    let bad = tool_typed("bad", "bad", |_args: serde_json::Value| async move {
        Err::<serde_json::Value, _>("boom".into())
    });

    let tc1 = ChatCompletionMessageToolCall {
        id: "g1".to_string(),
        r#type: ChatCompletionToolType::Function,
        function: FunctionCall {
            name: "good".to_string(),
            arguments: "{}".to_string(),
        },
    };
    let tc2 = ChatCompletionMessageToolCall {
        id: "b1".to_string(),
        r#type: ChatCompletionToolType::Function,
        function: FunctionCall {
            name: "bad".to_string(),
            arguments: "{}".to_string(),
        },
    };
    let assistant = ChatCompletionResponseMessage {
        content: None,
        role: RespRole::Assistant,
        tool_calls: Some(vec![tc1, tc2]),
        refusal: None,
        audio: None,
        function_call: None,
    };
    let provider = FixedProvider::new(ProviderResponse {
        assistant,
        prompt_tokens: 1,
        completion_tokens: 1,
    });

    let client = Arc::new(Client::<OpenAIConfig>::new());
    let mut agent = Agent::builder(client)
        .model("gpt-4o")
        .tool(good)
        .tool(bad)
        .with_provider(provider)
        .parallel_tools(true)
        .policy(CompositePolicy::new(vec![policies::max_steps(1)]))
        .build();

    let err = ServiceExt::ready(&mut agent)
        .await
        .unwrap()
        .call(simple_chat_request("sys", "hi"))
        .await
        .unwrap_err();
    assert!(format!("{}", err).contains("boom"));
}

#[tokio::test]
async fn test_step_parallel_tools_concurrency_limit() {
    use async_openai::{config::OpenAIConfig, Client};
    use std::sync::atomic::{AtomicUsize, Ordering};
    static CURRENT: AtomicUsize = AtomicUsize::new(0);
    static MAX_OBSERVED: AtomicUsize = AtomicUsize::new(0);

    let gate = tool_typed("gate", "gate", |_args: serde_json::Value| async move {
        let now = CURRENT.fetch_add(1, Ordering::SeqCst) + 1;
        let max = MAX_OBSERVED.load(Ordering::SeqCst);
        if now > max {
            MAX_OBSERVED
                .compare_exchange(max, now, Ordering::SeqCst, Ordering::SeqCst)
                .ok();
        }
        sleep(Duration::from_millis(15)).await;
        CURRENT.fetch_sub(1, Ordering::SeqCst);
        Ok::<_, tower::BoxError>(serde_json::json!({}))
    });

    // Provider that requests the same tool many times
    let mut calls = Vec::new();
    for i in 0..6 {
        calls.push(ChatCompletionMessageToolCall {
            id: format!("c{}", i),
            r#type: ChatCompletionToolType::Function,
            function: FunctionCall {
                name: "gate".to_string(),
                arguments: "{}".to_string(),
            },
        });
    }
    let assistant = ChatCompletionResponseMessage {
        content: None,
        role: RespRole::Assistant,
        tool_calls: Some(calls),
        function_call: None,
        refusal: None,
        audio: None,
    };
    let provider = FixedProvider::new(ProviderResponse {
        assistant,
        prompt_tokens: 1,
        completion_tokens: 1,
    });

    let client = Arc::new(Client::<OpenAIConfig>::new());
    let mut agent = Agent::builder(client)
        .model("gpt-4o")
        .tool(gate)
        .with_provider(provider)
        .parallel_tools(true)
        .tool_concurrency_limit(2)
        .policy(CompositePolicy::new(vec![policies::max_steps(1)]))
        .build();

    let _ = ServiceExt::ready(&mut agent)
        .await
        .unwrap()
        .call(simple_chat_request("sys", "hi"))
        .await
        .unwrap();
    assert!(MAX_OBSERVED.load(Ordering::SeqCst) <= 2);
}

#[tokio::test]
async fn test_step_parallel_tools_join_policy_failfast_vs_joinall() {
    use async_openai::{config::OpenAIConfig, Client};
    use std::sync::atomic::{AtomicUsize, Ordering};
    static FINISHED: AtomicUsize = AtomicUsize::new(0);

    let good = tool_typed("good", "good", |_args: serde_json::Value| async move {
        tokio::time::sleep(Duration::from_millis(30)).await;
        FINISHED.fetch_add(1, Ordering::SeqCst);
        Ok::<_, tower::BoxError>(serde_json::json!({"ok":true}))
    });
    let bad = tool_typed("bad", "bad", |_args: serde_json::Value| async move {
        Err::<serde_json::Value, _>("boom".into())
    });

    let tc_bad = ChatCompletionMessageToolCall {
        id: "b1".to_string(),
        r#type: ChatCompletionToolType::Function,
        function: FunctionCall {
            name: "bad".to_string(),
            arguments: "{}".to_string(),
        },
    };
    let tc_good = ChatCompletionMessageToolCall {
        id: "g1".to_string(),
        r#type: ChatCompletionToolType::Function,
        function: FunctionCall {
            name: "good".to_string(),
            arguments: "{}".to_string(),
        },
    };
    let assistant = ChatCompletionResponseMessage {
        content: None,
        role: RespRole::Assistant,
        tool_calls: Some(vec![tc_bad.clone(), tc_good.clone()]),
        function_call: None,
        refusal: None,
        audio: None,
    };
    let provider = FixedProvider::new(ProviderResponse {
        assistant,
        prompt_tokens: 1,
        completion_tokens: 1,
    });

    // FailFast: with concurrency limit 1, second tool should not complete
    FINISHED.store(0, Ordering::SeqCst);
    let client = Arc::new(Client::<OpenAIConfig>::new());
    let mut agent_ff = Agent::builder(client.clone())
        .model("gpt-4o")
        .tool(good)
        .tool(bad)
        .with_provider(provider.clone())
        .parallel_tools(true)
        .tool_concurrency_limit(1)
        .tool_join_policy(ToolJoinPolicy::FailFast)
        .policy(CompositePolicy::new(vec![policies::max_steps(1)]))
        .build();
    let _ = ServiceExt::ready(&mut agent_ff)
        .await
        .unwrap()
        .call(simple_chat_request("sys", "hi"))
        .await
        .unwrap_err();
    // small grace to allow dropped futures to cancel
    tokio::time::sleep(Duration::from_millis(5)).await;
    assert_eq!(FINISHED.load(Ordering::SeqCst), 0);

    // JoinAll: waits for the good tool to finish even if another fails
    FINISHED.store(0, Ordering::SeqCst);
    let assistant2 = ChatCompletionResponseMessage {
        content: None,
        role: RespRole::Assistant,
        tool_calls: Some(vec![tc_bad, tc_good]),
        function_call: None,
        refusal: None,
        audio: None,
    };
    let provider2 = FixedProvider::new(ProviderResponse {
        assistant: assistant2,
        prompt_tokens: 1,
        completion_tokens: 1,
    });
    // Recreate tools since ToolDef is not Clone
    let good2 = tool_typed("good", "good", |_args: serde_json::Value| async move {
        tokio::time::sleep(Duration::from_millis(30)).await;
        FINISHED.fetch_add(1, Ordering::SeqCst);
        Ok::<_, tower::BoxError>(serde_json::json!({"ok":true}))
    });
    let bad2 = tool_typed("bad", "bad", |_args: serde_json::Value| async move {
        Err::<serde_json::Value, _>("boom".into())
    });
    let mut agent_ja = Agent::builder(client)
        .model("gpt-4o")
        .tool(good2)
        .tool(bad2)
        .with_provider(provider2)
        .parallel_tools(true)
        .tool_concurrency_limit(1)
        .tool_join_policy(ToolJoinPolicy::JoinAll)
        .policy(CompositePolicy::new(vec![policies::max_steps(1)]))
        .build();
    let _ = ServiceExt::ready(&mut agent_ja)
        .await
        .unwrap()
        .call(simple_chat_request("sys", "hi"))
        .await
        .unwrap_err();
    assert_eq!(FINISHED.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn test_builder_map_agent_service() {
    use async_openai::{config::OpenAIConfig, Client};
    let assistant = ChatCompletionResponseMessage {
        content: None,
        role: RespRole::Assistant,
        tool_calls: None,
        function_call: None,
        refusal: None,
        audio: None,
    };
    let provider = FixedProvider::new(ProviderResponse {
        assistant,
        prompt_tokens: 1,
        completion_tokens: 1,
    });
    let client = Arc::new(Client::<OpenAIConfig>::new());
    let mut agent = Agent::builder(client)
        .model("gpt-4o")
        .with_provider(provider)
        .policy(CompositePolicy::new(vec![policies::max_steps(1)]))
        .map_agent_service(|svc| ServiceBuilder::new().service(svc))
        .build();
    let _ = ServiceExt::ready(&mut agent)
        .await
        .unwrap()
        .call(simple_chat_request("sys", "hi"))
        .await
        .unwrap();
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
    use tower_llm::Policy;

    let policy = Policy::new().or_max_steps(5).until_no_tool_calls().build();

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

    // The builder pattern compiles and constructs a builder
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
    let _reasons = [
        AgentStopReason::DoneNoToolCalls,
        AgentStopReason::MaxSteps,
        AgentStopReason::ToolCalled("test".to_string()),
        AgentStopReason::TokensBudgetExceeded,
        AgentStopReason::ToolBudgetExceeded,
        AgentStopReason::TimeBudgetExceeded,
    ];
}

#[tokio::test]
async fn test_agentbuilder_handoff_with_custom_provider() {
    use async_openai::{config::OpenAIConfig, Client};
    use tower_llm::groups::explicit_handoff_to;

    // Provider crafts an assistant message that calls the handoff tool
    let tool_name = "handoff_to_specialist".to_string();
    let tc = ChatCompletionMessageToolCall {
        id: "call_1".to_string(),
        r#type: ChatCompletionToolType::Function,
        function: FunctionCall {
            name: tool_name.clone(),
            arguments: "{\"reason\":\"escalate\"}".to_string(),
        },
    };
    let assistant = ChatCompletionResponseMessage {
        content: None,
        role: RespRole::Assistant,
        tool_calls: Some(vec![tc]),
        function_call: None,
        refusal: None,
        audio: None,
    };
    let provider = FixedProvider::new(ProviderResponse {
        assistant,
        prompt_tokens: 1,
        completion_tokens: 1,
    });

    // Build agent with handoff policy and custom provider (no real network)
    let client = Arc::new(Client::<OpenAIConfig>::new());
    let mut agent = Agent::builder(client)
        .model("gpt-4o")
        .handoff_policy(explicit_handoff_to("specialist").into())
        .policy(CompositePolicy::new(vec![policies::until_tool_called(
            tool_name.clone(),
        )]))
        .with_provider(provider)
        .build();

    let out = ServiceExt::ready(&mut agent)
        .await
        .unwrap()
        .call(simple_chat_request("sys", "hi"))
        .await
        .unwrap();

    assert!(matches!(out.stop, AgentStopReason::ToolCalled(n) if n == tool_name));
    assert!(out.messages.iter().any(|m| matches!(
        m,
        async_openai::types::ChatCompletionRequestMessage::Assistant(_)
    )));
    assert!(out.messages.iter().any(|m| matches!(
        m,
        async_openai::types::ChatCompletionRequestMessage::Tool(_)
    )));
}

#[test]
fn test_step_outcome_variants() {
    use tower_llm::StepAux;

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
    // use tower_llm::Runner;
    // use tower_llm::env::EnvBuilder;
    // use tower_llm::tool::Tool;
    // let agent = Agent::simple("bot", "instructions");

    // The new API requires a client
    use async_openai::{config::OpenAIConfig, Client};
    let client = Arc::new(Client::<OpenAIConfig>::new());
    let _builder = Agent::builder(client);
}
