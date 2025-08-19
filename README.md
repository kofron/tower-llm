# OpenAI Agents SDK for Rust

A Rust implementation of the OpenAI Agents SDK, providing a lightweight yet powerful framework for building multi-agent workflows. This SDK wraps the `async-openai` crate for LLM interactions.

> Note: The library has fully cut over to a Tower-driven execution path for tools and policies. Upgrade across minor versions to adopt the new behavior.

## Features

- ✅ **Agent System**: LLMs configured with instructions, tools, and handoffs
- ✅ **Tool System**: Extensible tool framework for agent capabilities
- ✅ **Handoffs**: Transfer control between specialized agents
- ✅ **Agent Groups**: Compose multiple agents into a single, cohesive unit
- ✅ **Guardrails**: Input/output validation for safety
- ✅ **Session Memory**: Maintain conversation history across runs
- ✅ **Tracing**: Built-in observability using the `tracing` crate
- ✅ **Async/Sync Support**: Both async and blocking execution modes
- ✅ **Streaming**: Real-time event streaming for long-running operations
- ✅ **Contextual Runs**: Per-run context that can rewrite tool outputs or finalize early

## Getting Started

### Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
openai-agents-rs = "0.1.0"
```

### Environment Setup

Before running the examples or your own code, you need to set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY="your-api-key"
```

## Quick Start

Here's a simple example of how to create and run an agent that writes haikus:

````rust,no_run
use openai_agents_rs::{Agent, Runner, runner::RunConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create an agent with a name and instructions.
    let agent = Agent::simple(
        "HaikuBot",
        "You are a helpful assistant that writes haikus about programming."
    );

    // Run the agent with a prompt.
    let result = Runner::run(
        agent,
        "Write a haiku about Rust",
        RunConfig::default(),
    ).await?;

    if result.is_success() {
        println!("Haiku about Rust:");
        println!("{}", result.final_output);
    } else {
        println!("Error: {:?}", result.error());
    }

    Ok(())
}

## Core Concepts

The SDK is built around a few core concepts that work together to create powerful agentic workflows. For more detailed information, please refer to the [crate documentation](https://docs.rs/openai-agents-rs).

- **[`Agent`]**: The fundamental building block, representing an entity that can process input and generate a response. Agents are defined by their configuration, including their identity, instructions, and tools.
- **[`Runner`]**: The engine that executes an agent's logic. It manages the interaction loop with the LLM, handles tool calls, and orchestrates the overall workflow.
- **[`AgentGroup`]**: Compose multiple agents with declared handoffs so they act as a single agent. Ideal for multi-agent workflows.
- **[`Tool`]**: A function or capability that an agent can use to interact with the outside world, such as calling an API or accessing a database.
- **[`Session`]**: Manages the state of an interaction, including the history of messages. A [`SqliteSession`] is provided for persistent state.
- **[`Guardrail`]**: A mechanism for validating and sanitizing the input and output of an agent, ensuring safety and reliability.

[`Agent`]: https://docs.rs/openai-agents-rs/latest/openai_agents_rs/agent/struct.Agent.html
[`Runner`]: https://docs.rs/openai-agents-rs/latest/openai_agents_rs/runner/struct.Runner.html
[`AgentGroup`]: https://docs.rs/openai-agents-rs/latest/openai_agents_rs/group/struct.AgentGroup.html
[`AgentGroupBuilder`]: https://docs.rs/openai-agents-rs/latest/openai_agents_rs/group/struct.AgentGroupBuilder.html
[`Tool`]: https://docs.rs/openai-agents-rs/latest/openai_agents_rs/tool/trait.Tool.html
[`Session`]: https://docs.rs/openai-agents-rs/latest/openai_agents_rs/memory/trait.Session.html
[`SqliteSession`]: https://docs.rs/openai-agents-rs/latest/openai_agents_rs/sqlite_session/struct.SqliteSession.html
[`Guardrail`]: https://docs.rs/openai-agents-rs/latest/openai_agents_rs/guardrail/index.html

## Examples

The `examples/` directory contains a rich set of demonstrations. Follow this progression to get a coherent tour of the SDK:

### Progression

- **Getting Started**
  - `hello_world.rs`: A simple agent that writes haikus
- **Adding Capabilities**
  - `tool_example.rs`: An agent that uses a tool to fetch information
- **Composing Agents**
  - `group_no_shared.rs`: Compose multiple agents with explicit handoffs, no shared state
- **Managing State**
  - `group_shared.rs`: Share run-scoped context across a group to accumulate state
- **Advanced Concepts**
  - `persistent_session.rs`: SQLite-backed persistent conversation history
  - `session_with_guardrails.rs`: Session memory with safety guardrails
  - `parallel_tools.rs`: Execute multiple tools concurrently
  - `tool_scope_timeout.rs`: Tool-scope policy layer (per-tool timeout)
  - `typed_tool.rs`: Strongly-typed tool inputs/outputs using `TypedFunctionTool`
  - `typed_tool_derive.rs`: Derive macros `#[tool_args]`, `#[tool_output]` for typed tools
  - `typed_env_approval.rs`: Advanced typed Env approval using `ApprovalLayer`
  - `approval.rs`: Predicate-based approval policy via `boxed_approval_with`
  - `calculator.rs`: Multi-tool calculator with step-by-step reasoning
  - `db_migrator.rs`: Transactional DB migrator with run-context commit/rollback
  - `rpn_calculator.rs`: RPN calculator with handler-managed execution stack
- **Case Study**
  - `multi_agent_research.rs`: Multi-agent research system with specialized roles

To run the examples:

```bash
cargo run --example <example_name>
````

For instance, to run the `hello_world` example:

```bash
cargo run --example hello_world
```

## Architecture

The SDK follows these design principles:

1. **Functional Core, Imperative Shell**: Business logic is pure and testable
2. **Type Safety**: Leverages Rust's type system for correctness
3. **Async-First**: Built on Tokio for efficient async operations
4. **Extensible**: Easy to add custom tools, guardrails, and model providers
5. **Context-aware**: Optional per-run context hook for tool output shaping

### Tower-based execution and layers

Tools execute through a Tower stack with fixed ordering: Agent layers → Run layers → Tool layers → Base tool. Policy layers (schema, timeouts, retries, approval, tracing) are scope-agnostic: the same type can be attached at any of the three positions. Replies are always appended in the provider/tool-call order; per-turn tool calls may run in parallel internally.

### Behavior invariants (v1)

- Messages constructed from run items preserve provider tool-call order.
- Tool replies are emitted as `Message::tool(content, tool_call_id)`; `content` is JSON stringified from tool output.
- Handoff is exposed as a tool call to the provider; the runner injects a single ACK tool reply and switches to the target agent. Run-scoped context persists across handoffs; per-agent context resets per agent.
- Context ordering: run-scoped context processes outputs before per-agent context. Outermost `Final` short-circuits the run.
- Errors are carried internally in typed form and stringified only at the message/protocol boundary.

Note: Legacy direct context handlers in `context.rs` remain internally wired but should be considered deprecated in favor of Tower layers. Public deprecation notes will follow.

Minimal usage remains unchanged:

```rust
use openai_agents_rs::{Agent, Runner, runner::RunConfig, FunctionTool};
use std::sync::Arc;

let tool = Arc::new(FunctionTool::simple("uppercase", "Upper", |s: String| s.to_uppercase()));
let agent = Agent::simple("Writer", "Be helpful").with_tool(tool);

let result = Runner::run(agent, "hello", RunConfig::default()).await?;
```

Parallel execution controls:

```rust
use openai_agents_rs::runner::RunConfig;

let config = RunConfig::default()
  .with_parallel_tools(true) // run multiple tool calls per turn concurrently
  .with_max_concurrency(4);  // optional limit

let result = Runner::run(agent, "hello", config).await?;
```

Attach scope-agnostic policy layers dynamically (run-scope example):

```rust
use openai_agents_rs::{runner::RunConfig, layers};

let layers_vec = vec![
  layers::boxed_timeout_secs(10),
  layers::boxed_retry_times(3),
  layers::boxed_input_schema_lenient(serde_json::json!({
    "type": "object",
    "properties": {"input": {"type":"string"}},
    "required": ["input"]
  })),
];

let config = RunConfig::default()
  .with_parallel_tools(true)
  .with_run_layers(layers_vec);
```

Attach layers at agent-scope (wraps run-scope):

```rust
use openai_agents_rs::{Agent, layers};

let agent = Agent::simple("CityInfoAgent", "...")
  .with_agent_layers(vec![
    layers::boxed_timeout_secs(5),
    layers::boxed_retry_times(2),
  ]);
```

The runner preserves reply order even when executing tools concurrently per turn.

### DX cheatsheet: attaching policy layers

```rust
use openai_agents_rs::{layers, runner::RunConfig, Agent, FunctionTool, Runner};
use std::sync::Arc;

// Define a tool
let tool = Arc::new(FunctionTool::simple("uppercase", "Upper", |s: String| s.to_uppercase()));

// Agent-scope layers (wrap run-scope)
let agent = Agent::simple("Writer", "Be helpful")
  .with_tool(tool.clone())
  .with_agent_layers(vec![
    layers::boxed_timeout_secs(10),
    layers::boxed_retry_times(3),
  ])
  // Tool-scope layers (wrap agent/run for this tool)
  .with_tool_layers("uppercase", vec![ layers::boxed_input_schema_lenient(serde_json::json!({
    "type": "object",
    "properties": {"input": {"type":"string"}},
    "required": ["input"]
  })) ]);

// Run-scope layers
let cfg = RunConfig::default().with_run_layers(vec![
  layers::boxed_timeout_secs(5),
]);

let _result = Runner::run(agent, "hello", cfg).await?;
```

#### Approval policy layer (predicate-based)

```rust
use openai_agents_rs::{layers, Agent, FunctionTool, Runner, runner::RunConfig};
use std::sync::Arc;

let safe = Arc::new(FunctionTool::simple("safe", "ok", |s: String| s));
let danger = Arc::new(FunctionTool::simple("danger", "", |s: String| s));
let agent = Agent::simple("Approve", "Use tools").with_tool(safe).with_tool(danger);

let cfg = RunConfig::default().with_run_layers(vec![
  // Deny the tool named "danger"; allow all others
  layers::boxed_approval_with(|_agent, tool, _args| tool != "danger"),
]);

let _ = Runner::run(agent, "run", cfg).await?;
```

### Typed tools (early API)

```rust
use openai_agents_rs::{TypedFunctionTool, Tool};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Deserialize, JsonSchema)]
struct AddArgs { x: i32, y: i32 }
#[derive(Serialize)]
struct Sum { sum: i32 }

let schema = serde_json::json!({
  "type":"object",
  "properties":{ "x": {"type":"integer"}, "y": {"type":"integer"} },
  "required":["x","y"]
});
let tool = TypedFunctionTool::new("add", "Adds two numbers", schema, |a: AddArgs| Ok(Sum { sum: a.x + a.y }));
```

## Performance notes

- Parallel tool execution can significantly reduce end-to-end latency when the model emits multiple tool calls per turn. In a synthetic benchmark with 8 slow tools (10ms each), sequential execution took ~111ms vs parallel ~15ms on a Mac (M-series). Your numbers will vary.
- Use `RunConfig::with_parallel_tools(true)` to enable parallel execution (default is true). To limit concurrency, set `with_max_concurrency(n)`.
- Reply ordering is preserved regardless of parallelism; the runner always appends tool replies in the provider-specified order.
- Consider applying per-tool timeouts with `layers::boxed_timeout_secs` to guard slow or unreliable tools.

Locking guidance:

- Run/Agent context layers are stateful and guarded by internal `Mutex`es for correctness across parallel calls. Keep work inside handlers minimal and non-blocking; prefer cloning small state and doing heavier work in tools instead.
- If you attach additional boxed layers that serialize access (e.g., via `tokio::sync::Mutex`), be mindful of throughput under high parallelism; use `with_max_concurrency(...)` to tune.

#### Common policy layers (scope-agnostic)

```rust
use openai_agents_rs::{layers, runner::RunConfig};

// Schema validation (lenient by default in the stack)
// Policy layers (schema/timeout/retry/approval) are available internally and applied by default conservatively.
// Public helpers to attach layers at run/agent/tool scopes are planned.
```

### Typed Env (advanced)

For advanced users, compose Tower layers directly with a typed Env that implements capability traits (e.g., `HasApproval`). This does not affect the default runner.

```rust,no_run
use openai_agents_rs::{layers::ApprovalLayer, service::BaseToolService, service::ToolRequest};
use openai_agents_rs::{layers::InputSchemaLayer, layers::RetryLayer, tool::FunctionTool};
use openai_agents_rs::service::HasApproval;
use std::sync::Arc;
use tower::{Layer, ServiceExt};

#[derive(Clone, Default)]
struct EnvAllowSafe;
impl HasApproval for EnvAllowSafe {
    fn approve(&self, _agent: &str, tool: &str, _args: &serde_json::Value) -> bool {
        tool != "danger"
    }
}

let safe = Arc::new(FunctionTool::simple("safe", "ok", |s: String| s));
let base = BaseToolService::new(safe.clone());
let stack = ApprovalLayer.layer(RetryLayer::times(2).layer(InputSchemaLayer::lenient(
    safe.parameters_schema(),
).layer(base))));

let req = ToolRequest::<EnvAllowSafe> {
    env: EnvAllowSafe,
    run_id: "r".into(), agent: "A".into(), tool_call_id: "id1".into(),
    tool_name: "safe".into(), arguments: serde_json::json!({"input":"ok"})
};
let resp = stack.oneshot(req).await?;
```

You can attach the same layers at agent- or tool-scope if you want more granular behavior.

Attach layers at tool-scope (wraps agent/run for a specific tool name):

```rust
use openai_agents_rs::{Agent, layers};

let agent = Agent::simple("Writer", "...")
  .with_tool_layers(
    "uppercase",
    vec![layers::boxed_timeout_secs(2)],
  );
```

### Context and Contextual Runs

Attach a context to observe and shape tool outputs. There are two complementary ways to do this:

#### Per-agent context

- Attach at agent construction time
- Applies only while that agent is active

```rust
use openai_agents_rs::{Agent, ToolContext, ContextStep};
use serde_json::Value;

#[derive(Clone, Default)]
struct MyCtx;
struct MyHandler;

impl ToolContext<MyCtx> for MyHandler {
    fn on_tool_output(
        &self,
        ctx: MyCtx,
        tool_name: &str,
        arguments: &Value,
        result: Result<Value, String>,
    ) -> openai_agents_rs::Result<ContextStep<MyCtx>> {
        let _ = (tool_name, arguments);
        match result {
            Ok(v) => Ok(ContextStep::rewrite(ctx, v)),
            Err(_e) => Ok(ContextStep::final_output(ctx, serde_json::json!("stopped"))),
        }
    }
}

let agent = Agent::simple("Ctx", "...")
    .with_context_factory(|| MyCtx::default(), MyHandler);
```

#### Run-scoped context (spans handoffs)

- Attach at run time so it applies across all agents, including handoffs
- Runs before any per-agent handler; its rewrite/final decisions take precedence

```rust,no_run
use openai_agents_rs::{Agent, Runner, runner::RunConfig, ContextStep, ToolContext, RunResultWithContext};
use serde_json::Value;

#[derive(Clone, Default)]
struct RunCtx { calls: usize }
struct RunHandler;

impl ToolContext<RunCtx> for RunHandler {
    fn on_tool_output(
        &self,
        mut ctx: RunCtx,
        _tool: &str,
        _args: &Value,
        result: Result<Value, String>,
    ) -> openai_agents_rs::Result<ContextStep<RunCtx>> {
        ctx.calls += 1;
        Ok(ContextStep::rewrite(ctx, result.unwrap_or(Value::Null)))
    }
}

let agent = Agent::simple("Planner", "Use tools and possibly hand off …");
let config = RunConfig::default();
let out: RunResultWithContext<RunCtx> = Runner::run_with_run_context(
    agent,
    "Do the thing",
    config,
    || RunCtx::default(),
    RunHandler,
).await?;
assert!(out.result.is_success());
println!("run-scoped calls: {}", out.context.calls);
```

Notes:

- Run-scoped context spans handoffs automatically
- Ordering: run-scoped handler runs first, then any per-agent handler
- Finalization: either handler can return `Final` to stop the run immediately

## Testing

The SDK includes comprehensive tests. Run them with:

```bash
cargo test
```

## Roadmap

- [ ] Context DI, it's one of the best things

## License

MIT
