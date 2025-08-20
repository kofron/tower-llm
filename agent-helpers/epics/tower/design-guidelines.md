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

## Static DI and Raw OpenAI I/O (New Policy)

### Goals

- Eliminate dynamic environments and ambient context for now
- Use raw OpenAI chat request/message types at service boundaries
- Make the agent loop a first-class Tower layer over a single-step service
- Ensure replayability by making `RunItem` a projection of raw OpenAI messages (bijective mapping)

### Policy Overview

- **Static DI only**: All dependencies are injected via constructors (usually `Arc<...>`). No request-carried dynamic envs or extension bags. Layers/services declare trait-bounded requirements statically at compile time.
- **Standardized I/O**: The inner “one-step” agent service accepts a raw OpenAI chat request and returns either an updated set of raw messages (continue) or a final set (done). The outer agent loop layer drives iteration and produces an event stream.

### Canonical Types (shape)

```rust
use async_openai::types::{
    CreateChatCompletionRequest as RawChatRequest,
    ChatCompletionRequestMessage as RawChatMessage,
};
use tower::{Service, Layer, BoxError, ServiceExt};
use std::sync::Arc;

// Auxiliary accounting captured per step
pub struct StepAux {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub tool_invocations: usize,
}

pub enum StepOutcome {
    Next { messages: Vec<RawChatMessage>, aux: StepAux },
    Done { messages: Vec<RawChatMessage>, aux: StepAux },
}

// Single-step service: one LLM completion + optional tool execution
pub struct AgentStepService {
    pub openai: Arc<async_openai::Client<async_openai::config::OpenAIConfig>>,
    pub model: String,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    // typed tool router held here (constructor-injected)
    pub tool_router: Arc<ToolRouter>,
}

impl Service<RawChatRequest> for AgentStepService {
    type Response = StepOutcome;
    type Error = BoxError;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn call(&mut self, req: RawChatRequest) -> Self::Future {
        // One model call; if assistant asks for tools, execute, append tool outputs as raw messages,
        // and return Next with updated messages; otherwise Done.
        unimplemented!()
    }
}
```

### Agent Loop as a Layer

```rust
use crate::items::RunItem; // see bijection section below

pub struct AgentRun {
    pub events: Vec<RunItem>,           // replayable event stream
    pub messages: Vec<RawChatMessage>,  // final message history
}

pub struct AgentLoopLayer {
    pub max_turns: usize,
}

pub struct AgentLoop<S> { inner: S, max_turns: usize }

impl<S> Layer<S> for AgentLoopLayer {
    type Service = AgentLoop<S>;
    fn layer(&self, inner: S) -> Self::Service { AgentLoop { inner, max_turns: self.max_turns } }
}

impl<S> Service<RawChatRequest> for AgentLoop<S>
where
    S: Service<RawChatRequest, Response = StepOutcome, Error = BoxError>,
{
    type Response = AgentRun;
    type Error = BoxError;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn call(&mut self, req: RawChatRequest) -> Self::Future {
        // Loop: repeatedly call inner until StepOutcome::Done or budget exhausted.
        // Convert raw messages <-> RunItem as we go to produce a replayable event stream.
        unimplemented!()
    }
}
```

### Idiomatic Composition

```rust
use tower::ServiceBuilder;

let agent = ServiceBuilder::new()
    .layer(AgentLoopLayer { max_turns: 16 })
    .service(AgentStepService { /* injected deps */ });

let run: AgentRun = agent.oneshot(initial_raw_request).await?;
```

## Bijection: RunItem ↔ Raw OpenAI Messages

To enable perfect replays, the event type we log must be a lossless projection of the raw OpenAI chat messages that go across the wire.

### Principles

- Every assistant/user/tool message in raw OpenAI maps to exactly one `RunItem::Message`/`RunItem::ToolOutput`/`RunItem::ToolCall` and back.
- Assistant function/tool calls (`tool_calls`) are represented explicitly so we can reconstruct them byte-for-byte.
- Tool outputs are represented as tool-role messages associated to the originating `tool_call_id`.
- Agent-only concepts (e.g., handoffs) are logged as separate event kinds that do not participate in the bijection; they are orthogonal to chat replay.

### Event Shapes (conceptual)

```rust
// Chat events that are bijective with raw OpenAI
#[serde(tag = "type")]
pub enum RunItem {
    // user/system/assistant messages (assistant may have tool_calls)
    Message { role: Role, content: String, tool_calls: Option<Vec<ToolCall>> },
    // outputs from tools (tool role messages)
    ToolOutput { tool_call_id: String, content: String },
    // agent-level events (not part of bijection)
    Handoff { from_agent: String, to_agent: String, reason: Option<String> },
}

pub struct ToolCall { pub id: String, pub name: String, pub arguments: serde_json::Value }
pub enum Role { System, User, Assistant, Tool }
```

### Conversions (lossless)

```rust
fn run_items_to_raw_messages(items: &[RunItem]) -> Vec<RawChatMessage> {
    // 1) Message(role=User|System|Assistant) → raw role message
    // 2) Message(role=Assistant, tool_calls=Some(_)) → assistant message with tool_calls
    // 3) ToolOutput(tool_call_id, content) → tool role message with matching tool_call_id
    unimplemented!()
}

fn raw_messages_to_run_items(messages: &[RawChatMessage]) -> Vec<RunItem> {
    // Inverse of the above mapping; preserves order and payloads
    unimplemented!()
}
```

### Tests

```rust
// Property: Bijective roundtrip over chat messages
proptest! {
    #[test]
    fn chat_roundtrip_is_identity(raw in arbitrary_raw_chat_history()) {
        let items = raw_messages_to_run_items(&raw);
        let back = run_items_to_raw_messages(&items);
        prop_assert_eq!(back, raw);
    }
}
```

## DI Guidance (Static Only)

- Constructor injection for all services and layers; hold shared infra in `Arc<...>`.
- Declare capability requirements using trait bounds on the concrete service/layer types. No dynamic registries, no string lookups.
- If per-request knobs are needed, carry them as explicit fields on the raw request (e.g., temperature, max_tokens) rather than hidden context.

## Summary of Changes to Previous Guidance

- Environment objects and dynamic lookups are removed for now; use static DI exclusively.
- Standardize on raw OpenAI chat request/messages at the service boundary.
- The agent loop is implemented as a Tower layer (`AgentLoopLayer`) over a single-step service (`AgentStepService`).
- `RunItem` is defined as a lossless projection of raw chat messages, enabling capture-and-replay via bijective conversions.
