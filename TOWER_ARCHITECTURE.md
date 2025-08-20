# Tower-Based Architecture Guide

## Overview

This codebase implements a pure Tower-based architecture for OpenAI agents. Everything is a Tower service, layers compose uniformly via typed `.layer()` APIs, and all cross-cutting concerns are handled through Tower middleware.

> **Important**: This document reflects the current architecture after the Tower migration completion. All legacy APIs have been removed in favor of uniform Tower patterns.

## Core Concepts

### 1. Three-Level Abstraction

```
Runs → Agents → Tools
```

- **Runs** orchestrate multiple agent turns
- **Agents** manage multiple tool calls
- **Tools** execute specific functions

Each level can have its own layers, applied in the canonical execution order: **Run → Agent → Tool → Base**.

### 2. Tools as Services

Tools are first-class Tower services that implement `Service<ToolRequest<E>, Response = ToolResponse>`.

```rust
use openai_agents_rs::{FunctionTool, tool_service::IntoToolService};
use std::sync::Arc;

// Create function tool
let tool = Arc::new(FunctionTool::simple(
    "upper", 
    "Uppercase", 
    |s: String| s.to_uppercase()
));

// Convert to service for Tower composition
let service = <FunctionTool as Clone>::clone(&tool).into_service::<DefaultEnv>();
```

### 3. Uniform Composition with Layers

Everything uses Tower's typed `.layer()` pattern:

```rust
use openai_agents_rs::{Agent, FunctionTool, layers, runner::RunConfig};
use tower::{ServiceBuilder, Layer};
use std::{sync::Arc, time::Duration};

// Service-based tool composition  
let tool = Arc::new(FunctionTool::simple("slow", "Slow operation", slow_fn));
let service = <FunctionTool as Clone>::clone(&tool).into_service::<DefaultEnv>();
let layered_service = layers::TimeoutLayer::secs(5).layer(
    layers::RetryLayer::times(3).layer(service)
);

// Agent with typed layers
let agent = Agent::simple("Assistant", "Helpful assistant")
    .with_tool(tool)
    .layer(layers::TracingLayer);

// Run config with typed layers  
let config = RunConfig::default()
    .layer(layers::TimeoutLayer::secs(30));
```

### 4. Capability-Based Environment

Layers can access shared resources through type-safe capabilities:

```rust
use openai_agents_rs::env::{EnvBuilder, LoggingCapability, Metrics};

// Build environment with capabilities
let env = EnvBuilder::new()
    .with_capability(Arc::new(LoggingCapability))
    .with_capability(Arc::new(MetricsCollector))
    .build();

// In a layer, access capabilities
if let Some(logger) = req.env.capability::<LoggingCapability>() {
    logger.info("Processing request");
}
```

## Legacy APIs Removed

The following APIs have been **completely removed** in favor of uniform Tower patterns:

### Removed Vector-Based APIs
- `with_agent_layers(vec![...])` → Use typed `.layer()` chaining
- `with_run_layers(vec![...])` → Use typed `.layer()` chaining  
- `with_tool_layers("name", vec![...])` → Configure layers at tool creation

### Removed Erased Layer System
- `ErasedToolLayer` trait → Use typed `Layer<S>` implementations
- `boxed_timeout_secs()`, `boxed_retry_times()` → Use `TimeoutLayer::secs()`, `RetryLayer::times()`
- `LayeredTool` struct → Use `.into_service().layer()` composition

### Removed Context System
- `ToolContext` handlers → Use stateful Tower layers
- `with_context_factory()` → Use capability-based environment
- `RunResultWithContext<C>` → Use stateful layers for cross-cutting state

### Removed Adapter Pattern
- `BaseToolService` → Tools implement `Service` directly via `.into_service()`

## Migration Patterns

### Tool Composition: Before vs After

```rust
// BEFORE: LayeredTool with erased layers
let tool = FunctionTool::simple("calc", "Calculator", calc_fn)
    .layer(layers::boxed_timeout_secs(5))
    .layer(layers::boxed_retry_times(3));

// AFTER: Service-based composition  
let tool = Arc::new(FunctionTool::simple("calc", "Calculator", calc_fn));
let service = <FunctionTool as Clone>::clone(&tool).into_service::<DefaultEnv>();
let layered = layers::TimeoutLayer::secs(5).layer(
    layers::RetryLayer::times(3).layer(service)
);
```

### Agent/Run Layering: Before vs After

```rust
// BEFORE: Vector-based APIs
let agent = Agent::simple("Bot", "Assistant")
    .with_agent_layers(vec![
        layers::boxed_timeout_secs(10),
        layers::boxed_retry_times(3),
    ]);

let config = RunConfig::default()
    .with_run_layers(vec![layers::boxed_timeout_secs(30)]);

// AFTER: Typed fluent chaining
let agent = Agent::simple("Bot", "Assistant")
    .layer(layers::TimeoutLayer::secs(10))
    .layer(layers::RetryLayer::times(3));

let config = RunConfig::default()
    .layer(layers::TimeoutLayer::secs(30));
```

### Environment Capabilities: Before vs After

```rust
// BEFORE: Context handlers
impl ToolContext<State> for Handler {
    fn on_tool_output(&self, ctx, tool, args, result) -> ContextStep {
        // Transform output
    }
}

// AFTER: Capability-based layers
let env = EnvBuilder::new()
    .with_capability(Arc::new(LoggingCapability))
    .build();

// Layer accesses capability
if let Some(logger) = req.env.capability::<LoggingCapability>() {
    logger.info("Processing request");
}
```

### Layer Ordering: Canonical Order

The execution order is always: **Run → Agent → Tool → Base**

```rust
// Runtime execution flow:
Run Layer (outermost)
  ↓
Agent Layer  
  ↓
Tool Layer
  ↓  
Base Tool Execution
```

## Standard Layers

### Timeout Layer

```rust
// On agent or run config
let agent = agent.layer(layers::TimeoutLayer::secs(30));

// On service directly  
let service = layers::TimeoutLayer::secs(30).layer(tool_service);
```

### Retry Layer

```rust
// On agent or run config
let agent = agent.layer(layers::RetryLayer::times(3));

// On service directly
let service = layers::RetryLayer::times(3).layer(tool_service);
```

### Approval Layer

```rust
// Requires capability in environment
let agent = agent.layer(layers::ApprovalLayer);
let config = config.layer(layers::ApprovalLayer);
```

### Schema Validation

```rust
// Applied directly to service
let schema = serde_json::json!({"type": "object"});
let service = layers::InputSchemaLayer::strict(schema).layer(tool_service);
```

## Creating Custom Layers

Layers are just Tower middleware:

```rust
use tower::{Layer, Service};

#[derive(Clone)]
struct LoggingLayer;

impl<S> Layer<S> for LoggingLayer {
    type Service = LoggingService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        LoggingService { inner }
    }
}

struct LoggingService<S> {
    inner: S,
}

impl<S, E> Service<ToolRequest<E>> for LoggingService<S>
where
    S: Service<ToolRequest<E>, Response = ToolResponse>,
    E: Env,
{
    type Response = ToolResponse;
    type Error = S::Error;
    type Future = /* ... */;

    fn call(&mut self, req: ToolRequest<E>) -> Self::Future {
        println!("Calling tool: {}", req.tool_name);
        self.inner.call(req)
    }
}
```

## Best Practices

### 1. Tools Compose Via Services

```rust
// ✅ GOOD: Service-based layering at tool creation
let tool = Arc::new(DatabaseTool::new(db_pool));
let service = tool.clone().into_service::<DefaultEnv>();
let layered = layers::TimeoutLayer::secs(30).layer(
    layers::RetryLayer::times(3).layer(service)
);

// ❌ BAD: External configuration of tool internals (removed)
// agent.configure_tool("database", layers);
```

### 2. Use Capabilities for Shared Resources

```rust
// ✅ GOOD: Type-safe capability access
if let Some(metrics) = env.capability::<MetricsCollector>() {
    metrics.increment("tool.calls", 1);
}

// ❌ BAD: String-based registry
let metrics = registry.get("metrics").unwrap();
```

### 3. Compose with Tower Patterns

```rust
// ✅ GOOD: Standard Tower composition
let service = layers::TimeoutLayer::secs(30).layer(
    layers::RetryLayer::times(3).layer(tool_service)
);

// Or via ServiceBuilder
use tower::ServiceBuilder;
let service = ServiceBuilder::new()
    .layer(layers::TimeoutLayer::secs(30))
    .layer(layers::RetryLayer::times(3))
    .service(tool_service);

// ❌ BAD: Custom composition mechanisms (removed)
// tool.with_timeout(30).with_retry(3)
```

### 4. Layer Order Matters

Layers wrap from outside-in as you add them:

```rust
tool
    .layer(A)  // A is innermost
    .layer(B)  // B wraps A
    .layer(C)  // C wraps B, outermost

// Execution order: C → B → A → Tool → A → B → C
```

## Examples

### Simple Agent with Layered Tools

```rust
use openai_agents_rs::{Agent, FunctionTool, layers};
use std::sync::Arc;

// Create tool
let calculator = Arc::new(FunctionTool::simple(
    "calc",
    "Calculator", 
    |input: String| calculate(input)
));

// Service-based layering for advanced usage
let service = <FunctionTool as Clone>::clone(&calculator).into_service::<DefaultEnv>();
let layered_service = layers::TimeoutLayer::secs(5).layer(
    layers::RetryLayer::times(2).layer(service)
);

// Agent with typed layers  
let agent = Agent::simple("Bot", "Assistant")
    .with_tool(calculator)
    .layer(layers::TimeoutLayer::secs(30));
```

### Tool as Tower Service

```rust
use openai_agents_rs::{FunctionTool, tool_service::IntoToolService, layers};
use tower::ServiceBuilder;
use std::time::Duration;

let tool = FunctionTool::simple("upper", "Uppercase", |s: String| s.to_uppercase());
let service = tool.into_service::<DefaultEnv>();

// Direct layer composition
let layered = layers::TimeoutLayer::secs(1).layer(service);

// Or via ServiceBuilder
let service2 = tool.into_service::<DefaultEnv>();
let composed = ServiceBuilder::new()
    .layer(layers::TimeoutLayer::secs(1))
    .service(service2);
```

### Custom Layer with Capabilities

```rust
struct MetricsLayer;

impl<S, E> Service<ToolRequest<E>> for MetricsService<S>
where
    E: Env,
{
    fn call(&mut self, req: ToolRequest<E>) -> Self::Future {
        if let Some(metrics) = req.env.capability::<Metrics>() {
            metrics.increment("tool.calls", 1);
        }
        self.inner.call(req)
    }
}
```

## Architecture Benefits

1. **Uniform Composition**: One pattern for everything
2. **Type Safety**: No string coupling or dynamic lookups
3. **Clear Boundaries**: Each abstraction level manages itself
4. **Tower Ecosystem**: Use any Tower middleware
5. **Testability**: Easy to test layers in isolation
6. **Performance**: Tower's optimized async execution

## Further Reading

- [Tower Documentation](https://docs.rs/tower)
- [DESIGN_UPDATES.md](./DESIGN_UPDATES.md) - Design diary
- [MIGRATION_GUIDE.md](./MIGRATION_GUIDE.md) - Migration guide
- [Examples](./examples/) - Working examples
