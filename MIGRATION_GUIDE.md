# Migration Guide: Tower-Based Architecture

This guide explains the migration from the old context-based system to the new Tower-based architecture.

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

### Old Pattern: Agent Configures Tool Layers

```rust
// ❌ OLD: Agent reaches into tool configuration
let tool = Arc::new(FunctionTool::simple("db", "Database", |s| s));
let agent = Agent::simple("Bot", "...")
    .with_tool(tool)
    .with_tool_layers("db", vec![
        layers::boxed_timeout_secs(30),
        layers::boxed_retry_times(3),
    ]);
```

### New Pattern: Tools Manage Themselves

```rust
// ✅ NEW: Tool manages its own layers
let tool = FunctionTool::simple("db", "Database", |s| s)
    .with_name("user_db")  // Optional custom name
    .layer(layers::boxed_timeout_secs(30))
    .layer(layers::boxed_retry_times(3));

let agent = Agent::simple("Bot", "...")
    .with_tool(Arc::new(tool));
```

## Key Concepts

### 1. LayeredTool

When you call `.layer()` on a tool, it returns a `LayeredTool` that wraps the original tool with its layers:

```rust
let base_tool = FunctionTool::simple("test", "Test", |s| s);
let layered_tool = base_tool.layer(layers::boxed_timeout_secs(5));
// layered_tool is a LayeredTool that implements the Tool trait
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
// Tool level - manages tool-specific policies
let tool = DatabaseTool::new(db_pool)
    .layer(TimeoutLayer::new(Duration::from_secs(30)))
    .layer(RetryLayer::times(3));

// Agent level - manages agent-specific policies
let agent = Agent::simple("Bot", "...")
    .with_tool(Arc::new(tool))
    .with_agent_layers(vec![
        layers::boxed_trace_all(),
    ]);

// Run level - manages run-specific policies
let config = RunConfig::default()
    .with_run_layers(vec![
        layers::boxed_global_timeout(300),
    ]);
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
agent.with_agent_layers(vec![
    layers::boxed_accumulator(initial_state),
]);
```

### Tool-Specific Configuration

**Old:**

```rust
agent.with_tool_layers("slow_api", vec![
    layers::boxed_timeout_secs(60),
    layers::boxed_retry_times(5),
]);
```

**New:**

```rust
let slow_api = ApiTool::new(client)
    .with_name("slow_api")
    .layer(layers::boxed_timeout_secs(60))
    .layer(layers::boxed_retry_times(5));
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
    let tool = FunctionTool::simple("test", "Test", |s| s)
        .layer(layers::boxed_timeout_secs(5))
        .layer(layers::boxed_retry_times(3));

    assert_eq!(tool.name(), "test");
    // Tool carries its layers internally
}
```

## Need Help?

If you're migrating existing code and need help:

1. Look for uses of removed APIs (context handlers, `with_tool_layers`)
2. Move tool configuration to the tool itself using `.layer()`
3. Use the appropriate level (tool/agent/run) for each policy
4. Test using the patterns in `tests/tower_patterns.rs`

The new architecture is simpler and more powerful - embrace the Tower way!
