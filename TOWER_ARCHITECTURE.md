# Tower-Based Architecture Guide

## Overview

This codebase implements a pure Tower-based architecture for OpenAI agents. Everything is a Tower service, layers compose uniformly, and all cross-cutting concerns are handled through Tower middleware.

## Core Concepts

### 1. Three-Level Abstraction

```
Runs → Agents → Tools
```

- **Runs** orchestrate multiple agent turns
- **Agents** manage multiple tool calls
- **Tools** execute specific functions

Each level can have its own layers, applied in order from most specific to most general.

### 2. Tools as Services

Tools are first-class Tower services that implement `Service<ToolRequest<E>, Response = ToolResponse>`.

```rust
use openai_agents_rs::tool_service::{ServiceTool, IntoToolService};

// Direct service tool
let tool = ServiceTool::new(
    "calculator",
    "Performs calculations",
    json!({"type": "object"}),
    |args| {
        // Tool logic here
        Ok(json!({"result": 42}))
    }
);

// Adapt existing tool
let existing = FunctionTool::simple("upper", "Uppercase", |s: String| s.to_uppercase());
let service = existing.into_service::<DefaultEnv>();
```

### 3. Uniform Composition with Layers

Everything uses Tower's `.layer()` pattern:

```rust
use tower::ServiceBuilder;

// Tools with layers
let tool = FunctionTool::simple("slow", "Slow operation", slow_fn)
    .layer(layers::boxed_timeout_secs(5))
    .layer(layers::boxed_retry_times(3));

// Direct Tower composition
let service = ServiceBuilder::new()
    .timeout(Duration::from_secs(5))
    .retry(retry_policy)
    .service(tool.into_service());

// Agent with layers
let agent = Agent::simple("Assistant", "Helpful assistant")
    .with_tool(Arc::new(tool))
    .layer(layers::boxed_trace());
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

## Migration from Old Architecture

### Before: Multiple Ways to Modify Behavior

```rust
// OLD: Agent configures tool internals (boundary violation)
let agent = agent.with_tool_layers("calculator", vec![retry_layer]);

// OLD: Context handlers (spooky action at a distance)
let agent = agent.with_context(initial_ctx, handler);

// OLD: String-based coupling
if let Some(tool) = tools.get("calculator") { ... }
```

### After: Uniform Tower Composition

```rust
// NEW: Tools manage themselves
let tool = calculator.layer(retry_layer);

// NEW: Layers for cross-cutting concerns
let tool = calculator.layer(transform_output_layer);

// NEW: Type-safe throughout
let tool: Arc<dyn Tool> = Arc::new(calculator);
```

## Standard Layers

### Timeout Layer

```rust
tool.layer(layers::boxed_timeout_secs(30))
```

### Retry Layer

```rust
tool.layer(layers::boxed_retry_times(3))
```

### Approval Layer

```rust
tool.layer(layers::boxed_approval_with(approval_fn))
```

### Schema Validation

```rust
tool.layer(layers::boxed_input_schema_strict(schema))
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

### 1. Tools Own Their Layers

```rust
// ✅ GOOD: Tool manages its own behavior
let tool = DatabaseTool::new(db_pool)
    .layer(timeout_layer)
    .layer(retry_layer);

// ❌ BAD: External configuration of tool internals
agent.configure_tool("database", layers);
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
ServiceBuilder::new()
    .layer(TimeoutLayer::new(duration))
    .layer(RetryLayer::new(policy))
    .service(tool)

// ❌ BAD: Custom composition mechanisms
tool.with_timeout(30).with_retry(3)
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

### Simple Tool with Layers

```rust
let calculator = FunctionTool::simple(
    "calc",
    "Calculator",
    |input: String| calculate(input)
)
.layer(layers::boxed_timeout_secs(5))
.layer(layers::boxed_retry_times(2));

let agent = Agent::simple("Bot", "Assistant")
    .with_tool(Arc::new(calculator));
```

### Tool as Tower Service

```rust
let tool = FunctionTool::simple("upper", "Uppercase", |s: String| s.to_uppercase())
    .into_service::<DefaultEnv>();

let service = ServiceBuilder::new()
    .timeout(Duration::from_secs(1))
    .service(tool);
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
