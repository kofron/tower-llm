# tower-llm

This crate implements LLM agents as a set of _composable_ primitives, built on top of tower services and layers. The core idea is that by imagining LLMs and the tools they call as tower services, we can compose them together easily, but also modify their behavior via layers (middleware) to build up almost arbitrarily complex behaviors.

## Overview: Tower-native building blocks

- Core service flow:
  - `Step<S>`: one non-streaming LLM call plus tool execution routed through `S: Service<ToolInvocation>`
  - `AgentLoop`: runs steps until a policy decides to stop
  - `AgentBuilder`: ergonomic builder for assembling tools, model, policy, sessions
- Composition as layers (middleware):
  - Sessions: `sessions::MemoryLayer` with `LoadSession`/`SaveSession` services (e.g., `InMemorySessionStore`, `SqliteSessionStore`)
  - Observability: `observability::{TracingLayer, MetricsLayer}`
  - Approvals: `approvals::{ModelApprovalLayer, ToolApprovalLayer}`
  - Resilience: `resilience` (retry, timeout, circuit-breaker)
  - Recording/Replay: `recording::{RecorderLayer, ReplayService}`
  - Handoffs (multi-agent): `groups::{HandoffPolicy, HandoffCoordinator}` and `AgentBuilder::handoff_policy(...)`

## Status

### Version 0.0.x - Just past experimental

I have used this crate in some software systems that I've built, and it works and is reliable. With that said, it's still a bit of an experiment, so YMMV!

## Contributing

Feel free to open PRs. In the future I'd like to make this more of an ecosystem in the same way that tower is an ecosystem and
leverage more 3rd party contributions. Before doing that I'd like for the core interfaces and design to settle down a bit more.

## License

MIT

## Quickstart

```rust
use std::sync::Arc;
use async_openai::{config::OpenAIConfig, Client};
use tower_llm::{Agent, policies, tool_typed};
use serde::Deserialize;

#[derive(Deserialize)]
struct EchoArgs { value: String }

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let client = Arc::new(Client::<OpenAIConfig>::new());
    let echo = tool_typed("echo", "Echo back the value", |args: EchoArgs| async move {
        Ok::<_, tower::BoxError>(serde_json::json!({"echo": args.value}))
    });

    let mut agent = Agent::builder(client)
        .model("gpt-4o")
        .tool(echo)
        .policy(policies::max_steps(1).into())
        .build();

    // Use run(system, user) convenience or call service with your own request
    let _ = tower_llm::run(&mut agent, "You are helpful", "Say hi").await?;
    Ok(())
}
```

## Sessions (stateful agents)

Attach `MemoryLayer` at build time using store services that implement `Service<LoadSession>` and `Service<SaveSession>`.

```rust
use std::sync::Arc;
use tower_llm::sessions::{InMemorySessionStore, SessionId};

let load = Arc::new(InMemorySessionStore::default());
let save = Arc::new(InMemorySessionStore::default());
let session = SessionId("my-session".into());

let mut agent = Agent::builder(client)
    .model("gpt-4o")
    .tool(echo)
    .policy(policies::max_steps(1).into())
    .build_with_session(load, save, session);
```

For persistence, use the SQLite-backed store:

```rust
use tower_llm::sqlite_session::SqliteSessionStore;

let load = Arc::new(SqliteSessionStore::new("sessions.db").await?);
let save = load.clone();
let mut agent = Agent::builder(client)
    .model("gpt-4o")
    .tool(echo)
    .policy(policies::max_steps(1).into())
    .build_with_session(load, save, SessionId("s1".into()));
```

## Observability (tracing and metrics)

Wrap the step/agent with layers:

```rust
use tower::{ServiceBuilder, Service};
use tower_llm::observability::{TracingLayer, MetricsLayer, MetricRecord};

let metrics_sink = tower::service_fn(|_m: MetricRecord| async move { Ok::<(), tower::BoxError>(()) });
let mut agent = ServiceBuilder::new()
    .layer(TracingLayer::default())
    .layer(MetricsLayer::new(metrics_sink))
    .service(agent);
```

## Budgeting (tokens, tools, time)

Budgeting is expressed as a policy and composed with others:

```rust
use tower_llm::budgets::{Budget, budget_policy};
use tower_llm::core::{CompositePolicy, policies};

let budget = Budget { max_prompt_tokens: Some(8000), max_tool_invocations: Some(5), ..Default::default() };
let policy = CompositePolicy::new(vec![
    policies::until_no_tool_calls(),
    budget_policy(budget),
]);

let mut agent = Agent::builder(client).policy(policy).build();
```

## Recording and Replay

```rust
use tower_llm::recording::{InMemoryTraceStore, RecorderLayer, ReplayService, WriteTrace};
use tower::{Service, ServiceExt};

// Record
let writer = InMemoryTraceStore::default();
let mut recorded = RecorderLayer::new(writer.clone(), "trace-1").layer(agent);
let _ = ServiceExt::ready(&mut recorded).await?.call(tower_llm::simple_chat_request("sys","hi")).await?;

// Replay (no model calls)
let mut replay = ReplayService::new(writer, "trace-1", "gpt-4o");
let _out = ServiceExt::ready(&mut replay).await?.call(tower_llm::simple_chat_request("sys","ignored")).await?;
```

## Handoffs (multi-agent)

Advertise a handoff policy in the agent builder, or orchestrate multiple agents with `HandoffCoordinator`.

```rust
use tower_llm::groups::{explicit_handoff_to, GroupBuilder};

// Builder-level interception on tools
let mut agent = Agent::builder(client)
    .model("gpt-4o")
    .handoff_policy(explicit_handoff_to("specialist").into())
    .build();

// Group-level coordination
let coordinator = GroupBuilder::new()
    .agent("triage", agent)
    .agent("specialist", Agent::builder(client.clone()).model("gpt-4o").build())
    .picker(tower::service_fn(|_req| async { Ok::<_, tower::BoxError>("triage".to_string()) }))
    .handoff_policy(explicit_handoff_to("specialist"))
    .build();
```

## Development

- Run tests: `cargo test`
- Lints: `cargo clippy -D warnings`
- Doctests: `cargo test --doc`

## Conversation validation (tests/examples)

This crate includes a pure validation facility to assert that conversations are well-formed for testing and examples.

- API: `tower_llm::validation::{validate_conversation, ValidationPolicy}`
- Examples: `examples/validate_conversation.rs`, `examples/generate_conversations.rs`

Minimal usage:

```rust
use tower_llm::validation::{validate_conversation, ValidationPolicy};
use async_openai::types::*;

let sys = ChatCompletionRequestSystemMessageArgs::default().content("sys").build().unwrap();
let usr = ChatCompletionRequestUserMessageArgs::default().content("hi").build().unwrap();
let asst = ChatCompletionRequestAssistantMessageArgs::default().content("ok").build().unwrap();
let msgs = vec![sys.into(), usr.into(), asst.into()];

assert!(validate_conversation(&msgs, &ValidationPolicy::default()).is_none());
```

Generators and mutators are available under `tower_llm::validation::gen` and `tower_llm::validation::mutate` for property tests.

## Parallel tool execution

You can enable concurrent execution of tool calls within a single step. Order of outputs is preserved.

Non-streaming:

```rust
let mut agent = Agent::builder(client)
  .model("gpt-4o")
  .tools(vec![slow, fast])
  .parallel_tools(true)
  .tool_concurrency_limit(4) // optional
  .tool_join_policy(tower_llm::ToolJoinPolicy::JoinAll) // optional: wait for all tools even if one fails
  .policy(CompositePolicy::new(vec![policies::max_steps(1)]))
  .build();
```

Streaming (when constructing the service directly):

```rust
use tower_llm::streaming::StepStreamService;

let step_stream = StepStreamService::new(provider, tool_router)
  .parallel_tools(true)
  .tool_concurrency_limit(4)
  .tool_join_policy(tower_llm::ToolJoinPolicy::JoinAll);
```

## Provider override (no-network testing)

You can inject a custom non-streaming provider into `AgentBuilder` for testing or to adapt other backends. Use `with_provider(...)` to supply any service that implements the `ModelService` trait.

```rust
use tower_llm::provider::{FixedProvider, ProviderResponse};
use async_openai::types::{ChatCompletionResponseMessage, Role as RespRole};

// Fixed assistant response from your provider
let assistant = ChatCompletionResponseMessage {
  content: Some("ok".into()),
  role: RespRole::Assistant,
  tool_calls: None,
  refusal: None,
  audio: None,
};
let provider = FixedProvider::new(ProviderResponse { assistant, prompt_tokens: 1, completion_tokens: 1 });

let mut agent = Agent::builder(client)
  .model("gpt-4o")
  .with_provider(provider)
  .policy(CompositePolicy::new(vec![policies::max_steps(1)]))
  .build();
```

## Wrapping the built agent with custom Tower layers

You can wrap the final agent service with arbitrary Tower composition using `map_agent_service(...)`:

```rust
use tower::ServiceBuilder;
use tower::timeout::TimeoutLayer;
use std::time::Duration;

let mut agent = Agent::builder(client)
  .model("gpt-4o")
  .map_agent_service(|svc| ServiceBuilder::new().layer(TimeoutLayer::new(Duration::from_secs(2))).service(svc))
  .build();
```
