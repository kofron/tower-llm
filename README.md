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

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
openai-agents-rs = "0.1.0"
```

## Quick Start

```rust
use openai_agents_rs::{Agent, Runner, RunConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create an agent
    let agent = Agent::simple(
        "Assistant",
        "You are a helpful assistant"
    );

    // Run the agent
    let result = Runner::run(
        agent,
        "Write a haiku about recursion",
        RunConfig::default(),
    ).await?;

    println!("{}", result.final_output);
    Ok(())
}
```

## Core Concepts

### Agents

Agents are LLMs configured with specific instructions and capabilities:

```rust
let agent = Agent::simple("MyAgent", "You are helpful")
    .with_model("gpt-4")
    .with_temperature(0.7)
    .with_max_turns(10);
```

### Tools

Tools allow agents to interact with external systems:

```rust
use openai_agents_rs::{FunctionTool, Tool};
use std::sync::Arc;

let tool = Arc::new(FunctionTool::simple(
    "uppercase",
    "Converts text to uppercase",
    |text: String| text.to_uppercase(),
));

let agent = Agent::simple("Agent", "Use tools when needed")
    .with_tool(tool);
```

### Handoffs

Agents can transfer control to specialized agents:

```rust
use openai_agents_rs::Handoff;

let specialist = Agent::simple("Specialist", "I handle special cases");
let handoff = Handoff::new(specialist, "Handles complex queries");

let main_agent = Agent::simple("Main", "I delegate when needed")
    .with_handoff(handoff);
```

### Guardrails

Validate input and output for safety:

```rust
use openai_agents_rs::guardrail::{MaxLengthGuardrail, InputGuardrail};
use std::sync::Arc;

let guard = Arc::new(MaxLengthGuardrail::new(1000));
let agent = Agent::simple("Safe", "I'm safe")
    .with_input_guardrail(guard);
```

### Sessions

Maintain conversation history:

```rust
use openai_agents_rs::{SqliteSession, Session};
use std::sync::Arc;

let session = Arc::new(SqliteSession::new_default("user_123"));

let config = RunConfig {
    session: Some(session),
    ..Default::default()
};

// First message
let result1 = Runner::run(agent.clone(), "Hello", config.clone()).await?;

// Second message - remembers context
let result2 = Runner::run(agent, "What did I just say?", config).await?;
```

## Examples

The `examples/` directory contains comprehensive demonstrations:

### Basic Examples

- **`hello_world.rs`** - Simple agent that writes haikus
- **`tool_example.rs`** - Using tools to fetch weather information

### Advanced Examples

- **`calculator.rs`** - Multi-tool calculator that solves complex math problems step-by-step

  - Demonstrates multiple tool calls across rounds
  - Shows how agents break down complex problems
  - Includes interactive mode for custom calculations

- **`multi_agent_research.rs`** - Research system with specialized agents

  - Coordinator agent that delegates to specialists
  - Research agent for searching knowledge base
  - Analyst agent for synthesizing information
  - Archivist agent for storing new facts
  - Demonstrates agent handoffs and collaboration

- **`session_with_guardrails.rs`** - Personal assistant with safety features

  - Session memory for conversation history
  - Input guardrails blocking sensitive information
  - Output guardrails adding disclaimers
  - Custom guardrail implementations
  - Note-taking and reminder tools

- **`persistent_session.rs`** - SQLite-based persistent sessions

  - Conversation history that survives application restarts
  - Database-backed session storage with SQLite
  - Demonstrates session continuity across runs
  - Automatic session recovery

- **`parallel_tools.rs`** - Concurrent tool execution
  - Multiple tools working in parallel for efficiency
  - Weather, news, and statistics gathering
  - Performance optimization through parallelization
  - Demonstrates time savings with concurrent execution

Run examples with:

```bash
# Basic examples
cargo run --example hello_world
cargo run --example tool_example

# Advanced examples with interaction
cargo run --example calculator
cargo run --example multi_agent_research
cargo run --example session_with_guardrails
cargo run --example persistent_session
cargo run --example parallel_tools
```

Each example includes both pre-configured scenarios and interactive modes where you can experiment with the agents.

## Architecture

The SDK follows these design principles:

1. **Functional Core, Imperative Shell**: Business logic is pure and testable
2. **Type Safety**: Leverages Rust's type system for correctness
3. **Async-First**: Built on Tokio for efficient async operations
4. **Extensible**: Easy to add custom tools, guardrails, and model providers

## Testing

The SDK includes comprehensive tests. Run them with:

```bash
cargo test
```

## License

MIT
