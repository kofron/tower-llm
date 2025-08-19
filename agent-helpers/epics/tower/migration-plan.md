# Tower-Based Architecture Migration Plan

## Executive Summary

Complete migration from the current mixed-abstraction model to a pure Tower-based architecture where tools, agents, and runs compose uniformly through layers. This migration eliminates `ToolContext`, string-based coupling, and action-at-a-distance patterns in favor of explicit, type-safe composition.

## Quick Orientation for Engineers

**If you're picking this up fresh, here's what you need to know:**

1. **We're adopting Tower's service model completely** - If you know Tower, you know this architecture
2. **The mantra: "Tools do work. Layers modify behavior."** - That's the entire mental model
3. **Every abstraction level manages itself** - Tools configure tools, agents configure agents, runs configure runs
4. **No backwards compatibility** - We're doing a clean break to get to a better place
5. **When in doubt, follow Tower's patterns** - They've solved these problems already

**Start by reading:**

- "Why This Refactor?" section below for motivation
- "Understanding the Architecture" for the mental model
- Then tackle phases in order

## Why This Refactor?

### The Problem

Our current codebase has evolved to support multiple ways of achieving the same goals, creating confusion and violating the principle of simplicity:

1. **Three ways to modify tool behavior:**

   - Tools themselves (`FunctionTool`, `TypedFunctionTool`)
   - Context handlers (`ToolContext` trait with `with_context_factory`)
   - Policy layers (timeout, retry, approval, schema validation)

2. **Mixed metaphors:**

   - Tools can be added traditionally via `Agent::with_tool()`
   - OR composed as layers (e.g., in `typed_env_approval.rs`)
   - Unclear which approach to use when

3. **Abstraction violations:**

   - Agents reach into tools to configure them (`with_tool_layers`)
   - Context handlers create "spooky action at a distance"
   - String-based coupling everywhere (tool names, registry lookups)

4. **Inconsistent APIs:**
   - Some things use `.with_retry()`, others use layers
   - Vector-based layer APIs vs chained composition
   - Special methods for different concerns

### The Vision

Tower is a battle-tested Rust library for building composable network services. Its key insight: **everything is a service, and services compose uniformly through layers**. We're adopting this philosophy completely.

In our new world:

- **Tools** are self-contained services that know how to execute their function
- **Layers** are the ONLY way to modify behavior (retry, timeout, transform output, etc.)
- **Composition** is uniform - everything uses `.layer()` the same way
- **Dependencies** are explicit - injected at construction, not discovered through context

### What Success Looks Like

An engineer should be able to:

1. Build a tool with its dependencies
2. Add layers to modify its behavior
3. Compose it into an agent
4. Never wonder "should I use a context handler or a layer for this?"
5. Never deal with string-based lookups or configuration

The mental model becomes dead simple: **Tools do work. Layers modify behavior. That's it.**

## Current State vs. Target State

### Current State (Problematic)

```rust
// Multiple ways to do the same thing - confusing!

// Method 1: Context handler for output transformation
agent.with_context_factory(|| MyContext::default(), MyHandler);

// Method 2: Layer on the agent
agent.with_agent_layers(vec![transform_layer]);

// Method 3: Configure tool from agent (abstraction violation!)
agent.with_tool_layers("database", vec![retry_layer]);

// String-based coupling everywhere
let tool_name = "database";  // Hope this matches!
```

### Target State (Clean)

```rust
// One way to do things - clear and consistent!

// Tools manage themselves
let db_tool = DatabaseTool::new(db_pool)
    .layer(TimeoutLayer::new(Duration::from_secs(30)))
    .layer(RetryLayer::new(RetryPolicy::times(1)));

// Agents just compose tools
let agent = Agent::simple("Bot", "Instructions...")
    .with_tool(Arc::new(db_tool))
    .layer(TracingLayer::new());  // Agent's own concerns

// No strings, no reaching across boundaries, no confusion
```

## Design Principles

1. **Uniform Composition**: Everything uses Tower's `.layer()` pattern
2. **Explicit Dependencies**: Tools receive dependencies at construction time
3. **Clear Boundaries**: Each abstraction level (Run → Agent → Tool) manages only its own concerns
4. **No String Coupling**: Type-safe composition throughout
5. **No Backwards Compatibility**: Clean break, full refactor

## Target Architecture

### Core Abstractions

```rust
// Tools are self-contained services with their own layers
let db_tool = DatabaseTool::new(db_pool)
    .with_name("database")  // Optional, defaults to type name
    .layer(TimeoutLayer::new(Duration::from_secs(30)))
    .layer(RetryLayer::new(RetryPolicy::times(1)));

// Agents compose tools without knowing their internals
let agent = Agent::simple("Bot", "Instructions...")
    .with_tool(Arc::new(db_tool))
    .layer(TracingLayer::new())
    .layer(StateAccumulatorLayer::new());

// Runs orchestrate agents
let config = RunConfig::default()
    .layer(GlobalTimeoutLayer::new(Duration::from_secs(300)))
    .layer(ApprovalLayer::new());

// Env provides cross-cutting capabilities via traits
let env = AppEnv {
    transport: log_transport,
    approver: approval_service,
};
```

### Layer Composition Model

- Layers wrap outside-in as they're chained (Tower standard)
- Each abstraction level applies its own layers
- Execution order: Run layers → Agent layers → Tool layers → Base execution
- Capabilities provided through Env + trait bounds

## Understanding the Architecture

### The Three-Level Abstraction Model

We have three levels of abstraction, each with a clear purpose:

1. **Runs** abstract over multiple agents/turns
   - Orchestrate the overall workflow
   - Apply global policies (e.g., total timeout, approval for all tools)
2. **Agents** abstract over multiple tool calls
   - Manage conversation flow with the LLM
   - Apply agent-specific policies (e.g., tracing, state accumulation)
3. **Tools** abstract over external services
   - Execute specific functions (database queries, API calls, calculations)
   - Apply tool-specific policies (e.g., retries for flaky APIs, timeouts for slow operations)

Each level can have layers, but **each level only manages its own layers**. This is crucial for maintaining clean boundaries.

### How Layers Compose

Layers wrap from outside-in following Tower's model:

```rust
// When you write this:
let service = base
    .layer(A)
    .layer(B)
    .layer(C);

// Execution flows: C → B → A → base

// In our architecture:
// Run layers → Agent layers → Tool layers → Tool execution
```

### Key Insight: Everything is a Service

This is Tower's superpower. A tool is a service. An agent is a service that uses tool services. A run is a service that uses agent services. Layers can wrap any service uniformly.

## How to Read This Migration Plan

1. **Start with "Why This Refactor?"** above to understand the problems we're solving
2. **Each phase has clear goals** - understand what we're trying to achieve before diving into implementation
3. **Look for "Changes Required"** sections - these are your action items
4. **"Interface Design"** shows the target API - this is what we're building toward
5. **"Acceptance Criteria"** are your checklist - all must be checked before moving on

## A Practical Example: Database Tool Evolution

Here's how a database tool evolves through this migration:

### Before (Current State - Confusing)

```rust
// Tool doesn't manage itself
let db_tool = DatabaseTool::new(pool);

// Agent reaches into tool's business
let agent = Agent::simple("Bot", "...")
    .with_tool(Arc::new(db_tool))
    .with_tool_layers("database", vec![  // String coupling!
        BoxedRetryLayer::times(3),
        BoxedTimeoutLayer::secs(30),
    ]);

// Or maybe use context handler?
agent.with_context_factory(|| State::default(), OutputHandler);

// Or maybe agent layers?
agent.with_agent_layers(vec![...]);

// Too many ways to do the same thing!
```

### After (Target State - Clean)

```rust
// Tool is self-contained and self-configured
let db_tool = DatabaseTool::new(pool)
    .with_name("user_db")  // Optional, defaults to "DatabaseTool"
    .layer(TimeoutLayer::new(Duration::from_secs(30)))
    .layer(RetryLayer::times(3));

// Agent just composes, doesn't configure
let agent = Agent::simple("Bot", "...")
    .with_tool(Arc::new(db_tool))
    .layer(TracingLayer::new());  // Agent's own concerns

// One way to do things, clear ownership, no strings
```

The tool owns its configuration. The agent owns its configuration. Clean boundaries. No confusion.

## Migration Phases

### Phase 1: Tool Self-Management

**Goal**: Tools become self-contained services that manage their own layers

#### Why This Phase?

Currently, agents can reach into tools to configure them (`with_tool_layers`), violating abstraction boundaries. Tools should be responsible for their own configuration, just like any other service.

#### Changes Required

1. Remove `Agent::with_tool_layers()` method
2. Add `Tool::layer()` method following Tower pattern
3. Add `Tool::with_name()` for explicit naming (defaults to type name)
4. Implement `Default` for tools with recommended layer stacks

#### Interface Design

```rust
pub trait Tool: Service<ToolRequest<E>, Response = ToolResponse> {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters_schema(&self) -> Value;

    // Layer composition
    fn layer<L>(self, layer: L) -> Layered<L, Self>
    where
        L: Layer<Self>,
        Self: Sized,
    {
        Layered::new(layer, self)
    }
}

impl<F> FunctionTool<F> {
    pub fn new(name: impl Into<String>, func: F) -> Self { ... }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }
}

// Default provides recommended layers
impl Default for DatabaseTool {
    fn default() -> Self {
        DatabaseTool::new_uninit()
            .layer(TimeoutLayer::new(Duration::from_secs(30)))
            .layer(RetryLayer::new(RetryPolicy::times(1)))
    }
}
```

#### Acceptance Criteria

- [ ] All tools can add their own layers via `.layer()`
- [ ] Tools have intrinsic names (with defaults)
- [ ] `Agent::with_tool_layers()` is removed
- [ ] All examples updated to show tools managing themselves

### Phase 2: Eliminate ToolContext

**Goal**: Remove `ToolContext` trait and all context handler infrastructure

#### Why This Phase?

`ToolContext` is a parallel system to layers that creates confusion. It allows "spooky action at a distance" where handlers can observe and modify tool outputs globally. This same functionality should be achieved through explicit layers, maintaining our principle of uniform composition.

#### Changes Required

1. Delete `src/context.rs` entirely
2. Remove `with_context_factory` from Agent
3. Remove `ToolContextSpec` from AgentConfig
4. Remove `RunContextSpec` and related types
5. Convert all context handler logic to layers

#### Layer Replacements for Context Patterns

```rust
// OLD: Context handler for output transformation
impl ToolContext<MyContext> for MyHandler {
    fn on_tool_output(&self, ctx, tool_name, args, result) -> ContextStep {
        // Transform output
    }
}

// NEW: Output transformation layer
pub struct OutputTransformLayer<F> {
    transform: F,
}

impl<S, F> Layer<S> for OutputTransformLayer<F>
where
    F: Fn(Value) -> Value + Clone,
{
    // Standard Tower layer implementation
}

// OLD: Context handler for state accumulation
agent.with_context_factory(|| MyState::default(), StateHandler);

// NEW: State accumulation layer
agent.layer(StateAccumulatorLayer::new(initial_state))
```

#### Acceptance Criteria

- [ ] `context.rs` deleted
- [ ] No references to `ToolContext` anywhere
- [ ] All context handler examples converted to layers
- [ ] `RunContextLayer` and `AgentContextLayer` removed from service.rs

### Phase 3: Uniform Layer API

**Goal**: All entities (Tool, Agent, RunConfig) use identical `.layer()` pattern

#### Why This Phase?

Currently we have `with_agent_layers(vec![...])` and `with_run_layers(vec![...])` which is inconsistent with Tower's chaining pattern. Tower uses `.layer()` for a reason - it makes composition order explicit and type-safe. We should follow their lead.

#### Changes Required

1. Replace `with_agent_layers(vec![...])` with chained `.layer()` calls
2. Replace `with_run_layers(vec![...])` with chained `.layer()` calls
3. Remove all `Vec<Box<dyn ErasedToolLayer>>` usage
4. Implement proper `Layer` trait for all layer types

#### Interface Design

```rust
impl Agent {
    pub fn layer<L>(self, layer: L) -> Layered<L, Self>
    where
        L: Layer<Self>,
        Self: Sized,
    {
        Layered::new(layer, self)
    }
}

impl RunConfig {
    pub fn layer<L>(self, layer: L) -> Layered<L, Self>
    where
        L: Layer<Self>,
        Self: Sized,
    {
        Layered::new(layer, self)
    }
}

// Usage
let agent = Agent::simple("Bot", "...")
    .with_tool(Arc::new(tool))
    .layer(TracingLayer::new())
    .layer(MetricsLayer::new());
```

#### Acceptance Criteria

- [ ] All entities use `.layer()` method
- [ ] No vector-based layer APIs remain
- [ ] Layer order is explicit through chaining
- [ ] All examples use fluent chaining pattern

### Phase 4: Capability-Based Env

**Goal**: Env provides capabilities through trait implementations

#### Why This Phase?

Some layers need access to shared resources or policies (like approval services, loggers, etc.). Rather than passing these through constructors or ambient context, we use Tower's pattern of capability traits. This allows layers to declare what they need, and the environment to provide it. It's type-safe, explicit, and composable.

#### Changes Required

1. Define capability traits (HasApproval, HasTransport, etc.)
2. Layers declare requirements via trait bounds
3. Remove `DefaultEnv` in favor of user-defined Envs
4. Make Env a generic parameter throughout

#### Interface Design

```rust
// Capability traits
pub trait HasApproval {
    fn approve(&self, agent: &str, tool: &str, args: &Value) -> bool;
}

pub trait HasTransport {
    fn transport(&self) -> &dyn Transport;
}

// Layers declare requirements
impl<S, E> Layer<S> for ApprovalLayer
where
    E: HasApproval,
{
    // Implementation
}

// User provides Env with capabilities
struct MyEnv {
    approver: Box<dyn Approver>,
    transport: Box<dyn Transport>,
}

impl HasApproval for MyEnv { ... }
impl HasTransport for MyEnv { ... }
```

#### Acceptance Criteria

- [ ] Capability traits defined for all cross-cutting concerns
- [ ] Layers use trait bounds instead of concrete types
- [ ] `DefaultEnv` removed
- [ ] Examples show custom Env implementations

### Phase 5: Clean Service Implementation

**Goal**: Tools are Tower Services, not wrapped by adapters

#### Why This Phase?

Currently, tools implement a custom `Tool` trait and are wrapped by `BaseToolService` to become Tower services. This indirection adds complexity without value. Tools should just BE Tower services directly. This simplifies the mental model and removes unnecessary abstraction layers.

#### Changes Required

1. Remove `BaseToolService` adapter
2. Tools directly implement `Service<ToolRequest<E>>`
3. Remove `ToolResult` in favor of `ToolResponse`
4. Simplify execution path

#### Interface Design

```rust
impl<E> Service<ToolRequest<E>> for DatabaseTool
where
    E: Clone + Send + 'static,
{
    type Response = ToolResponse;
    type Error = BoxError;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: ToolRequest<E>) -> Self::Future {
        // Direct implementation, no adapter
    }
}
```

#### Acceptance Criteria

- [ ] All tools directly implement `Service`
- [ ] `BaseToolService` deleted
- [ ] No adapter layers between tools and Tower
- [ ] Simplified execution path

### Phase 6: Remove Legacy Infrastructure

**Goal**: Delete all deprecated code and simplify module structure

#### Why This Phase?

Once we've migrated to the new patterns, we need to clean house. Old code that's no longer used creates confusion and maintenance burden. This phase ensures we end with a clean, minimal codebase that only contains what's necessary.

#### Changes Required

1. Delete unused modules and types
2. Consolidate service types
3. Remove backward compatibility shims
4. Clean up module exports

#### Files to Delete/Modify

- `src/context.rs` - DELETE entirely
- `src/service.rs` - Remove RunContextLayer, AgentContextLayer, ErasedToolLayer
- `src/agent.rs` - Remove context-related methods
- `src/runner.rs` - Simplify to use only Tower services

#### Acceptance Criteria

- [ ] No deprecated code remains
- [ ] Module structure is clean and logical
- [ ] All tests pass with new architecture
- [ ] Documentation updated

### Phase 7: Examples and Documentation

**Goal**: All examples showcase the new patterns

#### Why This Phase?

Examples are often the first thing developers look at. They need to showcase best practices and make the new patterns crystal clear. Good examples are better than documentation because they show real, working code.

#### Required Updates

1. Convert all examples to new API
2. Add examples for common patterns
3. Update README with new architecture
4. Write migration guide for users

#### New Example Patterns

```rust
// examples/simple_tool.rs
let tool = CalculatorTool::new()
    .layer(TimeoutLayer::new(Duration::from_secs(5)));

// examples/tool_with_dependencies.rs
let tool = DatabaseTool::new(db_pool)
    .with_name("user_db")
    .layer(RetryLayer::new(RetryPolicy::times(3)));

// examples/custom_env.rs
struct MyEnv {
    approver: ApprovalService,
}

impl HasApproval for MyEnv { ... }

// examples/complex_composition.rs
let agent = Agent::builder()
    .name("Assistant")
    .instructions("...")
    .tool(Arc::new(tool1))
    .tool(Arc::new(tool2))
    .layer(TracingLayer::new())
    .layer(MetricsLayer::new())
    .build();
```

#### Acceptance Criteria

- [ ] All examples use new patterns
- [ ] No examples use deprecated APIs
- [ ] README reflects new architecture
- [ ] Clear documentation of patterns

## Testing Strategy

### Unit Tests

- Each layer tested in isolation
- Tool service implementation tests
- Capability trait tests

### Integration Tests

- End-to-end agent runs with layers
- Complex layer composition
- Env capability provision

### Migration Tests

- Ensure no old patterns remain
- Verify layer execution order
- Test capability requirements

## Risk Mitigation

1. **Incremental Refactoring**: Each phase is independently testable
2. **Type Safety**: Compiler catches most issues
3. **Clear Boundaries**: Each phase has clear acceptance criteria
4. **No Feature Flags**: Clean breaks prevent confusion

## Success Metrics

1. **Code Simplification**: Fewer concepts, clearer boundaries
2. **Type Safety**: No string-based lookups
3. **Composability**: Uniform layer composition
4. **Performance**: No overhead from adapters
5. **Developer Experience**: Intuitive, follows Rust/Tower idioms

## Final State

### What's Gone

- `ToolContext` trait and all context handlers
- `with_context_factory` methods
- `with_tool_layers` on Agent
- `BaseToolService` adapter
- `ErasedToolLayer` and boxed layer vectors
- String-based tool configuration
- `DefaultEnv`

### What's New

- Tools as self-contained Tower services
- Uniform `.layer()` composition
- Capability-based Env with traits
- Direct dependency injection at construction
- Type-safe throughout

### Code Example - Final State

```rust
// Tool with dependencies and layers
let db_tool = DatabaseTool::new(db_pool)
    .with_name("database")
    .layer(TimeoutLayer::new(Duration::from_secs(30)))
    .layer(RetryLayer::new(RetryPolicy::times(1)));

// Agent composition
let agent = Agent::builder()
    .name("Assistant")
    .instructions("You are a helpful assistant")
    .tool(Arc::new(db_tool))
    .tool(Arc::new(api_tool))
    .layer(TracingLayer::new())
    .layer(StateLayer::new())
    .build();

// Run configuration
let config = RunConfig::default()
    .layer(GlobalTimeoutLayer::new(Duration::from_secs(300)))
    .layer(ApprovalLayer::new());

// Environment with capabilities
let env = AppEnv {
    approver: my_approver,
    logger: my_logger,
};

// Execute
let result = Runner::run_with_env(agent, "prompt", config, env).await?;
```

## Timeline Estimate

- Phase 1: 2-3 days (Tool self-management)
- Phase 2: 2-3 days (Eliminate ToolContext)
- Phase 3: 1-2 days (Uniform layer API)
- Phase 4: 2-3 days (Capability-based Env)
- Phase 5: 2-3 days (Clean service implementation)
- Phase 6: 1-2 days (Remove legacy)
- Phase 7: 2-3 days (Examples and docs)

**Total: 2-3 weeks for complete migration**

## Definition of Done

- [ ] All tests pass
- [ ] All examples run successfully
- [ ] No deprecated code remains
- [ ] Documentation updated
- [ ] Performance benchmarks show no regression
- [ ] Code follows Tower idioms throughout
- [ ] Type-safe composition everywhere
- [ ] Clear separation of concerns at each abstraction level
