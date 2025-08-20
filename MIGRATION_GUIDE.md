# Migration Guide: Tower-Based Architecture

This guide explains the migration from the old context-based system to the new Tower-based architecture.

> **Status**: This guide reflects the final Tower architecture after all migration phases are complete. All legacy APIs have been removed.

## What Changed

### Before: Multiple Ways to Modify Tool Behavior

The old system had three different ways to modify tool behavior:

1. Context handlers (`ToolContext` trait)
2. Tool layers attached via agents (`with_tool_layers`)
3. Direct tool implementation

This created confusion and violated the principle of separation of concerns.

### After: Unified Tower-Based Composition

Now there's one clear pattern:

- **Tools do work** - They execute their core functionality
- **Layers modify behavior** - All cross-cutting concerns (retry, timeout, transformation) are layers
- **Each level manages itself** - Tools, agents, and runs each manage their own layers

## Migration Examples

### Old Pattern: Agent Configures Tool Layers (REMOVED)

```rust
// ❌ REMOVED: Agent reaches into tool configuration
let tool = Arc::new(FunctionTool::simple("db", "Database", |s| s));
let agent = Agent::simple("Bot", "...")
    .with_tool(tool)
    .with_tool_layers("db", vec![
        layers::boxed_timeout_secs(30),
        layers::boxed_retry_times(3),
    ]);
```

### New Pattern: Service-Based Composition

```rust
// ✅ NEW: Service-based layering 
let tool = Arc::new(FunctionTool::simple("db", "Database", |s| s));

// For advanced layering, use service composition
let service = <FunctionTool as Clone>::clone(&tool).into_service::<DefaultEnv>();
let layered = layers::TimeoutLayer::secs(30).layer(
    layers::RetryLayer::times(3).layer(service)
);

// Agent uses tool directly
let agent = Agent::simple("Bot", "...")
    .with_tool(tool);
```

## Key Concepts

### 1. Service-Based Tool Composition (Replaces LayeredTool)

LayeredTool has been removed. Tools now use uniform Tower service composition:

```rust
// Create base tool
let base_tool = Arc::new(FunctionTool::simple("test", "Test", |s| s));

// Convert to service and apply layers
let service = <FunctionTool as Clone>::clone(&base_tool).into_service::<DefaultEnv>();
let layered_service = layers::TimeoutLayer::secs(5).layer(service);
```

### 2. Tool Naming

Tools can have custom names:

```rust
let tool = FunctionTool::simple("original", "Description", |s| s)
    .with_name("custom_name");
assert_eq!(tool.name(), "custom_name");
```

### 3. Clean Boundaries

Each abstraction level manages only its own concerns:

```rust
// Tool level - compose layers via service when needed
let tool = Arc::new(DatabaseTool::new(db_pool));
let service = tool.clone().into_service::<DefaultEnv>();
let layered = layers::TimeoutLayer::secs(30).layer(
    layers::RetryLayer::times(3).layer(service)
);

// Agent level - manages agent-specific policies via typed layers
let agent = Agent::simple("Bot", "...")
    .with_tool(tool)
    .layer(layers::TracingLayer);

// Run level - manages run-specific policies  
let config = RunConfig::default()
    .layer(layers::TimeoutLayer::secs(300));
```

## Removed APIs

The following APIs have been removed:

### Context System

- `ToolContext` trait - Removed entirely
- `with_context()` - Use layers instead
- `with_context_factory()` - Use layers instead
- `with_context_typed()` - Use layers instead
- `RunContextLayer` - Use run-level layers instead
- `AgentContextLayer` - Use agent-level layers instead

### Agent Tool Configuration  

- `Agent::with_tool_layers()` - Tools now manage their own layers
- `with_agent_layers(vec![...])` - Use typed `.layer()` chaining instead
- `with_run_layers(vec![...])` - Use typed `.layer()` chaining instead

### LayeredTool System (Step 8 Removal)

- `LayeredTool` struct - Use service composition via `.into_service().layer()`
- `Tool::layer()` methods - Use service composition instead
- `ErasedToolLayer` trait - Use typed `Layer<S>` implementations  
- All `boxed_*` helpers - Use typed layers directly
  - `boxed_timeout_secs()` → `TimeoutLayer::secs()`
  - `boxed_retry_times()` → `RetryLayer::times()`
  - `boxed_input_schema_*()` → `InputSchemaLayer::strict/lenient()`

### Adapter System

- `BaseToolService` - Tools implement `Service` directly via `.into_service()`

### Run Context Methods

- `Runner::run_with_context()` - Use layers instead
- `Runner::run_with_run_context()` - Use layers instead
- `RunConfig::with_run_context()` - Use run-level layers instead

## Common Migration Patterns

### Output Transformation

**Old (Context Handler):**

```rust
impl ToolContext<MyCtx> for MyHandler {
    fn on_tool_output(&self, ctx, tool, args, result) -> ContextStep {
        let wrapped = json!({"wrapped": result?});
        Ok(ContextStep::rewrite(ctx, wrapped))
    }
}
agent.with_context(MyCtx::default(), MyHandler);
```

**New (Transformation Layer):**

```rust
// Create a custom layer or use a provided one
let tool = my_tool.layer(OutputTransformLayer::new(|output| {
    json!({"wrapped": output})
}));
```

### State Accumulation

**Old (Context with State):**

```rust
struct MyCtx { count: usize }
impl ToolContext<MyCtx> for Counter {
    fn on_tool_output(&self, mut ctx, ...) -> ContextStep {
        ctx.count += 1;
        // ...
    }
}
```

**New (Stateful Layer):**

```rust
// Use a stateful layer at the appropriate level
let agent = agent.layer(layers::AccumulatorLayer::new(initial_state));
```

### Tool-Specific Configuration

**Old (REMOVED):**

```rust
agent.with_tool_layers("slow_api", vec![
    layers::boxed_timeout_secs(60),
    layers::boxed_retry_times(5),
]);
```

**New:**

```rust
let slow_api = Arc::new(ApiTool::new(client).with_name("slow_api"));
// For advanced layering, use service composition
let service = <ApiTool as Clone>::clone(&slow_api).into_service::<DefaultEnv>();
let layered = layers::TimeoutLayer::secs(60).layer(
    layers::RetryLayer::times(5).layer(service)
);
```

## Benefits of the New Architecture

1. **Simplicity**: One pattern for all behavior modification
2. **Type Safety**: No string-based lookups or configuration
3. **Clear Boundaries**: Each level manages only its own concerns
4. **Tower Compatibility**: Follows Tower's proven patterns
5. **Composability**: Layers compose predictably and uniformly

## Testing the New Patterns

See `tests/tower_patterns.rs` for examples of testing tools with layers:

```rust
#[test]
fn test_tool_with_layers() {
    let base_tool = Arc::new(FunctionTool::simple("test", "Test", |s| s));
    
    // Service composition for layered tools
    let service = <FunctionTool as Clone>::clone(&base_tool).into_service::<DefaultEnv>();
    let layered = layers::TimeoutLayer::secs(5).layer(
        layers::RetryLayer::times(3).layer(service)
    );

    assert_eq!(base_tool.name(), "test");
    // Layering happens via service composition
}
```

## Need Help?

If you're migrating existing code and need help:

1. Look for uses of removed APIs (context handlers, `with_tool_layers`)
2. Move tool configuration to the tool itself using `.layer()`
3. Use the appropriate level (tool/agent/run) for each policy
4. Test using the patterns in `tests/tower_patterns.rs`

The new architecture is simpler and more powerful - embrace the Tower way!
