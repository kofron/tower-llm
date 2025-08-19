# Tower Architecture Design Guidelines

## Core Philosophy

The Tower-based architecture treats everything as a composable service. Tools, agents, and runs are all services that can be wrapped with layers to modify behavior. This creates a uniform, predictable, and type-safe system.

## Fundamental Rules

### 1. Separation of Concerns

Each abstraction level manages ONLY its own concerns:

```rust
// ✅ GOOD: Tool manages its own layers
let tool = DatabaseTool::new(db_pool)
    .layer(TimeoutLayer::new(Duration::from_secs(30)));

// ❌ BAD: Agent configuring tool internals
let agent = agent.with_tool_layers("database", vec![...]);
```

### 2. Explicit Dependencies

Dependencies are injected at construction time, not discovered through context:

```rust
// ✅ GOOD: Tool gets what it needs upfront
let tool = DatabaseTool::new(db_pool, logger);

// ❌ BAD: Tool discovers dependencies through ambient context
impl Tool for DatabaseTool {
    fn execute(&self, ctx: Context) {
        let db = ctx.get::<Database>();  // Spooky action
    }
}
```

### 3. Uniform Composition

Everything uses Tower's `.layer()` pattern:

```rust
// ✅ GOOD: Consistent API everywhere
let tool = tool.layer(TimeoutLayer::new(...));
let agent = agent.layer(TracingLayer::new());
let config = config.layer(ApprovalLayer::new());

// ❌ BAD: Special methods for different concerns
let tool = tool.with_timeout(...).with_retry(...);
```

### 4. Type Safety Over Strings

No string-based lookups or coupling:

```rust
// ✅ GOOD: Type-safe composition
let tool: DatabaseTool = DatabaseTool::new(pool);
agent.with_tool(Arc::new(tool));

// ❌ BAD: String-based registration
registry.register("database", tool);
let tool = registry.get("database");
```

## Layer Design Patterns

### Layer Order Matters

Layers wrap outside-in as they're chained:

```rust
let service = base_service
    .layer(A)  // A wraps base
    .layer(B)  // B wraps A
    .layer(C); // C wraps B

// Execution order: C → B → A → base_service
```

### Capability Requirements

Layers declare requirements through trait bounds:

```rust
impl<S, E> Layer<S> for ApprovalLayer
where
    E: HasApproval,  // Layer requires approval capability
    S: Service<ToolRequest<E>>,
{
    // Implementation
}
```

### Stateless Preferred

Prefer stateless layers when possible:

```rust
// ✅ GOOD: Stateless transformation
pub struct OutputTransformLayer<F> {
    transform: F,  // Pure function
}

// ⚠️ USE CAREFULLY: Stateful layer
pub struct AccumulatorLayer {
    state: Arc<Mutex<State>>,  // Shared mutable state
}
```

## Tool Design Patterns

### Self-Contained Tools

Tools should be complete, self-managing units:

```rust
pub struct DatabaseTool {
    name: String,
    pool: DbPool,
    // Tool owns everything it needs
}

impl DatabaseTool {
    pub fn new(pool: DbPool) -> Self {
        Self {
            name: "DatabaseTool".to_string(),  // Default name
            pool,
        }
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }
}
```

### Default Implementations

Use `Default` for recommended configurations:

```rust
impl Default for DatabaseTool {
    fn default() -> Self {
        DatabaseTool::new_uninit()
            .layer(TimeoutLayer::new(Duration::from_secs(30)))
            .layer(RetryLayer::new(RetryPolicy::times(1)))
    }
}

// Users can use defaults or customize
let tool = DatabaseTool::default().with_connection(pool);
let tool = DatabaseTool::new(pool);  // No default layers
```

### Direct Service Implementation

Tools implement `Service` directly, no adapters:

```rust
impl<E> Service<ToolRequest<E>> for DatabaseTool
where
    E: Clone + Send + 'static,
{
    type Response = ToolResponse;
    type Error = BoxError;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn call(&mut self, req: ToolRequest<E>) -> Self::Future {
        let pool = self.pool.clone();
        Box::pin(async move {
            // Direct implementation
            let result = query_database(&pool, req.arguments).await?;
            Ok(ToolResponse::success(result))
        })
    }
}
```

## Agent Design Patterns

### Agents as Orchestrators

Agents compose tools without knowing their internals:

```rust
impl Agent {
    pub fn with_tool(mut self, tool: Arc<dyn Tool>) -> Self {
        self.tools.push(tool);
        self
    }

    // Agent has its own layers, doesn't touch tool layers
    pub fn layer<L>(self, layer: L) -> Layered<L, Self>
    where
        L: Layer<Self>,
    {
        Layered::new(layer, self)
    }
}
```

### Builder Pattern for Complex Agents

```rust
let agent = Agent::builder()
    .name("Assistant")
    .instructions("You are helpful")
    .tool(Arc::new(tool1))
    .tool(Arc::new(tool2))
    .layer(TracingLayer::new())
    .build();
```

## Environment Design Patterns

### Capability Traits

Define clear capability interfaces:

```rust
pub trait HasApproval {
    fn approve(&self, agent: &str, tool: &str, args: &Value) -> bool;
}

pub trait HasLogger {
    fn logger(&self) -> &dyn Logger;
}

pub trait HasMetrics {
    fn metrics(&self) -> &dyn MetricsCollector;
}
```

### Composable Environments

Environments can compose multiple capabilities:

```rust
struct ProductionEnv {
    approver: ApprovalService,
    logger: LogService,
    metrics: MetricsService,
}

impl HasApproval for ProductionEnv { ... }
impl HasLogger for ProductionEnv { ... }
impl HasMetrics for ProductionEnv { ... }
```

### Minimal Environments

Not every environment needs every capability:

```rust
struct TestEnv;  // Minimal env for testing

impl HasApproval for TestEnv {
    fn approve(&self, _: &str, _: &str, _: &Value) -> bool {
        true  // Auto-approve in tests
    }
}
```

## Anti-Patterns to Avoid

### ❌ Context Handlers

Don't use context handlers for output transformation:

```rust
// ❌ BAD: Context handler
impl ToolContext<MyContext> for MyHandler {
    fn on_tool_output(&self, ctx, tool_name, args, result) -> ContextStep {
        // Transform output
    }
}

// ✅ GOOD: Layer
pub struct OutputTransformLayer<F> {
    transform: F,
}
```

### ❌ String-Based Configuration

Don't use strings to identify or configure:

```rust
// ❌ BAD: String-based
agent.with_tool_layers("database", vec![...]);

// ✅ GOOD: Type-based
let tool = database_tool.layer(...);
```

### ❌ Action at a Distance

Don't reach across abstraction boundaries:

```rust
// ❌ BAD: Agent configuring tool internals
agent.configure_tool("database", |tool| {
    tool.set_timeout(30);
});

// ✅ GOOD: Tool configures itself
let tool = DatabaseTool::new(pool)
    .layer(TimeoutLayer::new(Duration::from_secs(30)));
```

### ❌ Hidden Dependencies

Don't hide dependencies in ambient context:

```rust
// ❌ BAD: Hidden dependency
fn execute(&self) {
    let db = somehow_get_database();  // Where does this come from?
}

// ✅ GOOD: Explicit dependency
fn new(db: Database) -> Self {
    Self { db }
}
```

## Testing Guidelines

### Test Layers in Isolation

```rust
#[test]
fn test_timeout_layer() {
    let base = MockService::new();
    let wrapped = base.layer(TimeoutLayer::new(Duration::from_millis(10)));

    // Test timeout behavior
}
```

### Test Composition

```rust
#[test]
fn test_layer_composition() {
    let service = base_service
        .layer(LayerA)
        .layer(LayerB)
        .layer(LayerC);

    // Verify execution order
}
```

### Test Capabilities

```rust
#[test]
fn test_approval_capability() {
    struct TestEnv;
    impl HasApproval for TestEnv {
        fn approve(&self, _, tool, _) -> bool {
            tool != "dangerous"
        }
    }

    // Test approval logic
}
```

## Performance Considerations

### Minimize Allocations

```rust
// ✅ GOOD: Clone cheap handles
let pool = self.pool.clone();  // Arc<Pool>

// ❌ BAD: Clone expensive data
let data = self.large_data.clone();  // Vec<BigStruct>
```

### Avoid Blocking in Async

```rust
// ✅ GOOD: Async all the way
Box::pin(async move {
    let result = async_operation().await?;
    Ok(ToolResponse::success(result))
})

// ❌ BAD: Blocking in async
Box::pin(async move {
    let result = blocking_operation();  // Blocks executor
    Ok(ToolResponse::success(result))
})
```

### Layer Overhead

Be mindful of layer overhead in hot paths:

```rust
// Consider the cost of many layers
let service = base
    .layer(A)  // Each layer adds overhead
    .layer(B)
    .layer(C)
    .layer(D);
```

## Migration Patterns

### From Context Handler to Layer

```rust
// OLD: Context handler
impl ToolContext<State> for Handler {
    fn on_tool_output(&self, state, tool, args, result) -> ContextStep {
        let new_state = update_state(state, &result);
        let transformed = transform_output(result);
        ContextStep::rewrite(new_state, transformed)
    }
}

// NEW: Stateful layer
pub struct StateLayer {
    state: Arc<Mutex<State>>,
}

impl<S> Layer<S> for StateLayer {
    // Transform both state and output
}
```

### From Special Methods to Layers

```rust
// OLD: Special methods
let tool = tool
    .with_timeout(30)
    .with_retry(3)
    .with_approval();

// NEW: Uniform layers
let tool = tool
    .layer(TimeoutLayer::new(Duration::from_secs(30)))
    .layer(RetryLayer::new(RetryPolicy::times(3)))
    .layer(ApprovalLayer::new());
```

## Future Considerations

### Custom Layer Combinators

Consider convenience combinators for common patterns:

```rust
// Potential future API
let standard_stack = LayerStack::new()
    .timeout(Duration::from_secs(30))
    .retry(3)
    .trace();

let tool = tool.layer(standard_stack);
```

### Dynamic Layer Configuration

Consider patterns for runtime configuration:

```rust
// Potential future API
let layers = if config.debug {
    vec![Box::new(TracingLayer::verbose())]
} else {
    vec![Box::new(TracingLayer::minimal())]
};
```

### Cross-Cutting Policies

Consider how to apply policies across many tools:

```rust
// Potential future API
let policy = StandardPolicy::new()
    .timeout(Duration::from_secs(30))
    .retry(3);

tools.apply_policy(policy);
```
