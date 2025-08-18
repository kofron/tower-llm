# OpenAI Agents SDK for Rust

A Rust implementation of the OpenAI Agents SDK, providing a lightweight yet powerful framework for building multi-agent workflows. This SDK wraps the `async-openai` crate for LLM interactions.

## Features

- ✅ **Agent System**: LLMs configured with instructions, tools, and handoffs
- ✅ **Tool System**: Extensible tool framework for agent capabilities
- ✅ **Handoffs**: Transfer control between specialized agents
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
- **[`Tool`]**: A function or capability that an agent can use to interact with the outside world, such as calling an API or accessing a database.
- **[`Session`]**: Manages the state of an interaction, including the history of messages. A [`SqliteSession`] is provided for persistent state.
- **[`Guardrail`]**: A mechanism for validating and sanitizing the input and output of an agent, ensuring safety and reliability.

[`Agent`]: https://docs.rs/openai-agents-rs/latest/openai_agents_rs/agent/struct.Agent.html
[`Runner`]: https://docs.rs/openai-agents-rs/latest/openai_agents_rs/runner/struct.Runner.html
[`Tool`]: https://docs.rs/openai-agents-rs/latest/openai_agents_rs/tool/trait.Tool.html
[`Session`]: https://docs.rs/openai-agents-rs/latest/openai_agents_rs/memory/trait.Session.html
[`SqliteSession`]: https://docs.rs/openai-agents-rs/latest/openai_agents_rs/sqlite_session/struct.SqliteSession.html
[`Guardrail`]: https://docs.rs/openai-agents-rs/latest/openai_agents_rs/guardrail/index.html

## Examples

The `examples/` directory contains a rich set of demonstrations that showcase the capabilities of the SDK.

### Basic Examples

- **`hello_world.rs`**: A simple agent that writes haikus.
- **`tool_example.rs`**: An agent that uses a tool to fetch weather information.

### Advanced Examples

- **`calculator.rs`**: A multi-tool calculator that solves complex math problems step-by-step.
- **`multi_agent_research.rs`**: A research system with specialized agents for coordination, research, analysis, and archiving.
- **`session_with_guardrails.rs`**: A personal assistant with session memory and safety guardrails.
- **`persistent_session.rs`**: Demonstrates the use of SQLite for persistent conversation history.
- **`parallel_tools.rs`**: Shows how to execute multiple tools concurrently for improved performance.
 - **`contextual.rs`**: Demonstrates contextual handling of tool outputs (rewrite/finalize)
 - **`rpn_calculator.rs`**: RPN calculator where the handler maintains the execution stack and the final stack is extracted after the run

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

### Contextual Runs

You can attach a contextual handler that observes tool outputs and decides to:

- Forward the output unchanged
- Rewrite the output fed back to the model
- Finalize the run immediately with a value

Builder API:

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

## Testing

The SDK includes comprehensive tests. Run them with:

```bash
cargo test
```

## Roadmap

- [ ] Context DI, it's one of the best things

## License

MIT
