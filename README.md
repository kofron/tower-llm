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

## Architecture and rationale

At its core, tower-llm treats agents, tools, and coordinators as Tower services you compose with layers. This keeps business logic pure and testable while pushing side effects (I/O, retries, tracing, storage) to the edges.

- Agents are services that turn chat requests into outcomes.
- Tools are services invoked by the agent when the model asks for them.
- Layers add cross-cutting behavior without coupling concerns.

Why this design works well:

- Composable by default: you assemble exactly what you need.
- Static dependency injection: no global registries or runtime lookups.
- Functional core, imperative shell: easy to reason about and to test.

## A layered story: from hello world to production

Start small, add power as you go:

1. Hello world agent: see [Quickstart](#quickstart). One model, one tool, a simple stop policy.

2. Keep state between turns: add [Sessions](#sessions-stateful-agents) with `MemoryLayer` and an in-memory or SQLite store.

3. See what's happening: add [Observability](#observability-tracing-and-metrics) via `TracingLayer` and `MetricsLayer` to emit spans and metrics.

4. Bound cost and behavior: compose a [Budget policy](#budgeting-tokens-tools-time) with your stop policies.

5. Be resilient: wrap the agent with resilience layers (retry/timeout/circuit-breaker) from `resilience`.

6. Record and reproduce: tap runs with [Recording/Replay](#recording-and-replay) to debug or build fixtures without model calls.

7. Speed up tools: enable [Parallel tool execution](#parallel-tool-execution) and pick a join policy when multiple tools can run concurrently.

8. Stream in real time: use `streaming::StepStreamService` and `AgentLoopStreamLayer` for token-by-token UIs (see the streaming snippet below).

9. Go multi-agent: coordinate specialists with [Handoffs](#handoffs-multi-agent). Start with explicit or sequential policies; compose them as needed.

10. Keep context tight: add `auto_compaction::AutoCompactionLayer` or `groups::CompactingHandoffPolicy` to trim history during long runs (see `examples/handoff_compaction.rs`).

11. Validate conversations: use [Conversation validation](#conversation-validation-testsexamples) to assert invariants in tests and examples.

Throughout, you can swap providers or run entirely offline using [Provider override](#provider-override-no-network-testing).

## Layer catalog at a glance

- Sessions: persist and reload history around each call.

  - When: you need stateful conversations or persistence.
  - How: `sessions::MemoryLayer`, with `InMemorySessionStore` or `SqliteSessionStore`.

- Observability: trace spans and emit metrics for every step/agent call.

  - When: you want visibility in dev and prod.
  - How: `observability::{TracingLayer, MetricsLayer}` with your metrics sink.

- Approvals: gate model outputs and tool invocations.

  - When: you need review or policy enforcement.
  - How: `approvals::{ModelApprovalLayer, ToolApprovalLayer}` plus an `Approver` service.

- Resilience: retry, time out, and break circuits around calls.

  - When: you want robust error handling for flaky dependencies.
  - How: `resilience::{RetryLayer, TimeoutLayer, CircuitBreakerLayer}`.

- Recording/Replay: capture runs, replay deterministically without network.

  - When: you want debuggable, repeatable scenarios or tests.
  - How: `recording::{RecorderLayer, ReplayService}` with a trace store.

- Budgets: constrain tokens, tools, or steps.

  - When: you want cost and behavior bounds.
  - How: `budgets::budget_policy(...)`, composed in `CompositePolicy`.

- Concurrency: execute multiple tools concurrently; preserve output order.

  - When: independent tools can run in parallel.
  - How: enable parallel tools on the builder or use `concurrency::ParallelToolRouter`; select a join policy.

- Streaming: emit tokens and tool events incrementally for UIs.

  - When: you need real-time output.
  - How: `streaming::{StepStreamService, AgentLoopStreamLayer, StreamTapLayer}`.

- Handoffs (multi-agent): coordinate multiple agents at runtime.

  - When: you have specialists or workflows.
  - How: `groups::{AgentPicker, HandoffPolicy, HandoffCoordinator, GroupBuilder}`; see also `CompactingHandoffPolicy`.

- Auto compaction: trim conversation history safely.

  - When: you approach token limits or want faster iterations.
  - How: `auto_compaction::{AutoCompactionLayer, CompactionPolicy}`; or wrap handoffs with `CompactingHandoffPolicy`.

- Provider override: swap the model implementation.
  - When: offline tests or custom backends.
  - How: `provider::{FixedProvider, SequenceProvider, OpenAIProvider}` with the `ModelService` trait.

## Agent-level instructions (system)

Attach the system prompt at the agent level so it applies consistently across steps and handoffs.

```rust
let mut agent = Agent::builder(client)
  .model("gpt-4o")
  .instructions("You are a helpful assistant.")
  .tool(echo)
  .policy(policies::max_steps(1).into())
  .build();

// Send only a user message; the agent injects its instructions as a system message
let _ = tower_llm::run_user(&mut agent, "Say hi").await?;
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
