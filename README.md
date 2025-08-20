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
- ✅ **Tower Layers**: Composable middleware for cross-cutting concerns (timeouts, retries, validation)

## Getting Started

### Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
tower-llm = "0.1.0"
```

### Environment Setup

Before running the examples or your own code, you need to set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY="your-api-key"
```

## Quick Start

Here's a simple example of how to create and run an agent that writes haikus:

````rust,no_run
use tower_llm::{Agent, Runner, runner::RunConfig};

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

The SDK is built around a few core concepts that work together to create powerful agentic workflows. For more detailed information, please refer to the [crate documentation](https://docs.rs/tower-llm).

- **[`Agent`]**: The fundamental building block, representing an entity that can process input and generate a response. Agents are defined by their configuration, including their identity, instructions, and tools.
- **[`Runner`]**: The engine that executes an agent's logic. It manages the interaction loop with the LLM, handles tool calls, and orchestrates the overall workflow.
- **[`AgentGroup`]**: Compose multiple agents with declared handoffs so they act as a single agent. Ideal for multi-agent workflows.
- **[`Tool`]**: A function or capability that an agent can use to interact with the outside world, such as calling an API or accessing a database.
- **[`Session`]**: Manages the state of an interaction, including the history of messages. A [`SqliteSession`] is provided for persistent state.
- **[`Guardrail`]**: A mechanism for validating and sanitizing the input and output of an agent, ensuring safety and reliability.

[`Agent`]: https://docs.rs/tower-llm/latest/tower_llm/agent/struct.Agent.html
[`Runner`]: https://docs.rs/tower-llm/latest/tower_llm/runner/struct.Runner.html
[`AgentGroup`]: https://docs.rs/tower-llm/latest/tower_llm/group/struct.AgentGroup.html
[`AgentGroupBuilder`]: https://docs.rs/tower-llm/latest/tower_llm/group/struct.AgentGroupBuilder.html
[`Tool`]: https://docs.rs/tower-llm/latest/tower_llm/tool/trait.Tool.html
[`Session`]: https://docs.rs/tower-llm/latest/tower_llm/memory/trait.Session.html
[`SqliteSession`]: https://docs.rs/tower-llm/latest/tower_llm/sqlite_session/struct.SqliteSession.html
[`Guardrail`]: https://docs.rs/tower-llm/latest/tower_llm/guardrail/index.html

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
  - `group_shared.rs`: Share state across a group using stateful Tower layers
- **Advanced Concepts**
  - `persistent_session.rs`: SQLite-backed persistent conversation history
  - `session_with_guardrails.rs`: Session memory with safety guardrails
  - `parallel_tools.rs`: Execute multiple tools concurrently
  - `tool_scope_timeout.rs`: Tool-scope policy layer (per-tool timeout)
  - `typed_tool.rs`: Strongly-typed tool inputs/outputs using `TypedFunctionTool`
  - `typed_tool_derive.rs`: Derive macros `#[tool_args]`, `#[tool_output]` for typed tools
  - `typed_env_approval.rs`: Advanced typed Env approval using `ApprovalLayer`
  - `approval.rs`: Capability-based approval policy using Tower layers
  - `calculator.rs`: Multi-tool calculator with step-by-step reasoning
  - `db_migrator.rs`: Transactional DB migrator with stateful layer commit/rollback
  - `rpn_calculator.rs`: RPN calculator with stateful execution stack layers
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
5. **Composable**: Tower-based middleware for cross-cutting concerns

### Tower-based execution and layers

Tools execute through a Tower stack with fixed ordering: Run layers → Agent layers → Tool layers → Base tool. Policy layers (schema, timeouts, retries, approval, tracing) are scope-agnostic: the same type can be attached at any of the three positions. Replies are always appended in the provider/tool-call order; per-turn tool calls may run in parallel internally.

### Behavior invariants (v1)

- Messages constructed from run items preserve provider tool-call order.
- Tool replies are emitted as `Message::tool(content, tool_call_id)`; `content` is JSON stringified from tool output.
- Handoff is exposed as a tool call to the provider; the runner injects a single ACK tool reply and switches to the target agent. Run-scoped context persists across handoffs; per-agent context resets per agent.
- Context ordering: run-scoped context processes outputs before per-agent context. Outermost `Final` short-circuits the run.
- Errors are carried internally in typed form and stringified only at the message/protocol boundary.

Note: All policy and cross-cutting concerns are now handled through Tower layers for uniform, type-safe composition.

Minimal usage remains unchanged:

```rust
use tower_llm::{Agent, Runner, runner::RunConfig, FunctionTool};
use std::sync::Arc;

let tool = Arc::new(FunctionTool::simple("uppercase", "Upper", |s: String| s.to_uppercase()));
let agent = Agent::simple("Writer", "Be helpful").with_tool(tool);

let result = Runner::run(agent, "hello", RunConfig::default()).await?;
```

Parallel execution controls:

```rust
use tower_llm::runner::RunConfig;

let config = RunConfig::default()
  .with_parallel_tools(true) // run multiple tool calls per turn concurrently
  .with_max_concurrency(4);  // optional limit

let result = Runner::run(agent, "hello", config).await?;
```

Compose typed layers fluently at run-scope:

```rust
use tower_llm::{runner::RunConfig, layers};

let config = RunConfig::default()
  .with_parallel_tools(true)
  .layer(layers::TimeoutLayer::secs(10))
  .layer(layers::RetryLayer::times(3));
```

Attach layers at agent-scope (wraps run-scope):

```rust
use tower_llm::{Agent, layers};

let agent = Agent::simple("CityInfoAgent", "...")
  .layer(layers::TimeoutLayer::secs(5))
  .layer(layers::RetryLayer::times(2));
```

The runner preserves reply order even when executing tools concurrently per turn.

### DX cheatsheet: composing typed policy layers

```rust
use tower_llm::{layers, runner::RunConfig, Agent, FunctionTool, Runner};
use std::sync::Arc;

// Define a tool with explicit schema validation
let tool = Arc::new(
    FunctionTool::simple("uppercase", "Upper", |s: String| s.to_uppercase())
        .layer(layers::InputSchemaLayer::strict(serde_json::json!({
            "type": "object",
            "properties": {"input": {"type": "string"}},
            "required": ["input"]
        })))
);

// Agent-scope layers (wrap run-scope)
let agent = Agent::simple("Writer", "Be helpful")
  .with_tool(tool)
  .layer(layers::TimeoutLayer::secs(10))
  .layer(layers::RetryLayer::times(3));

// Run-scope layers (outermost)
let cfg = RunConfig::default()
  .layer(layers::TimeoutLayer::secs(5));

let _result = Runner::run(agent, "hello", cfg).await?;
```

#### Approval policy layer (capability-based)

```rust
use tower_llm::{layers, Agent, FunctionTool, Runner, runner::RunConfig, env::EnvBuilder};
use std::sync::Arc;

// Custom approval implementation
#[derive(Default)]
struct SafetyApproval;
impl tower_llm::env::Approval for SafetyApproval {
    fn request_approval(&self, operation: &str, details: &str) -> bool {
        !operation.contains("danger") // Deny dangerous operations
    }
}

let safe = Arc::new(FunctionTool::simple("safe", "ok", |s: String| s));
let agent = Agent::simple("Approve", "Use tools").with_tool(safe);

let env = EnvBuilder::new()
    .with_capability(Arc::new(tower_llm::env::ApprovalCapability::new(SafetyApproval)))
    .build();

let cfg = RunConfig::default()
    .layer(layers::ApprovalLayer);

let _ = Runner::run(agent, "run", cfg).await?;
```

### Typed tools (early API)

```rust
use tower_llm::{TypedFunctionTool, Tool};
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
- Consider applying timeouts via Tower services: `tool.into_service().layer(TimeoutLayer::secs(30))` to guard slow or unreliable tools.

Locking guidance:

- Stateful Tower layers are thread-safe and can maintain state across tool executions. Keep work inside layers minimal and non-blocking; prefer cloning small state and doing heavier work in tools instead.
- Custom layers that serialize access (e.g., via `tokio::sync::Mutex`) should be mindful of throughput under high parallelism; use `with_max_concurrency(...)` to tune.

#### Common policy layers (scope-agnostic)

```rust
use tower_llm::{layers, runner::RunConfig};

// Schema validation (lenient by default in the stack)
// Policy layers (schema/timeout/retry/approval) are available internally and applied by default conservatively.
// Public helpers to attach layers at run/agent/tool scopes are planned.
```

### Advanced Tower Service Composition

For advanced users, compose Tower layers directly with tools using the service-based approach:

```rust,no_run
use tower_llm::{
    layers::{ApprovalLayer, InputSchemaLayer, RetryLayer},
    service::ToolRequest,
    tool::FunctionTool,
    tool_service::IntoToolService,
    env::{EnvBuilder, ApprovalCapability}
};
use std::sync::Arc;
use tower::{Layer, ServiceExt};

// Custom approval capability
#[derive(Default)]
struct SafetyApproval;
impl tower_llm::env::Approval for SafetyApproval {
    fn request_approval(&self, _operation: &str, _details: &str) -> bool { true }
}

let env = EnvBuilder::new()
    .with_capability(Arc::new(ApprovalCapability::new(SafetyApproval)))
    .build();

let safe = Arc::new(FunctionTool::simple("safe", "ok", |s: String| s));
let tool_service = <FunctionTool as Clone>::clone(&safe).into_service::<_>();
let stack = ApprovalLayer.layer(
    RetryLayer::times(2).layer(
        InputSchemaLayer::lenient(safe.parameters_schema()).layer(tool_service)
    )
);

let req = ToolRequest {
    env: env.clone(),
    run_id: "r".into(), agent: "A".into(), tool_call_id: "id1".into(),
    tool_name: "safe".into(), arguments: serde_json::json!({"input":"ok"})
};
let resp = stack.oneshot(req).await?;
```

Tools can have their own layers applied directly during construction for self-contained behavior.

### Stateful Layers (Advanced)

For cross-cutting concerns that require state, implement custom Tower layers:

```rust,no_run
use tower_llm::{service::{ToolRequest, ToolResponse}, layers};
use tower::{Layer, Service};
use std::sync::{Arc, Mutex};

// Custom stateful layer that counts tool calls
#[derive(Clone)]
pub struct CallCounterLayer {
    counter: Arc<Mutex<usize>>,
}

impl CallCounterLayer {
    pub fn new() -> Self {
        Self { counter: Arc::new(Mutex::new(0)) }
    }

    pub fn get_count(&self) -> usize {
        *self.counter.lock().unwrap()
    }
}

impl<S> Layer<S> for CallCounterLayer {
    type Service = CallCounterService<S>;

    fn layer(&self, service: S) -> Self::Service {
        CallCounterService {
            inner: service,
            counter: self.counter.clone(),
        }
    }
}

// Apply to agents, tools, or runs as needed
let counter = CallCounterLayer::new();
let agent = Agent::simple("Counter", "Uses tools")
    .layer(counter.clone());
```

This provides type-safe, composable state management using standard Tower patterns.

## Testing

The SDK includes comprehensive tests. Run them with:

```bash
cargo test
```

## Roadmap

- [ ] Context DI, it's one of the best things

## License

MIT
