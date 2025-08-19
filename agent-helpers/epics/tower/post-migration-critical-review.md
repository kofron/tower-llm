# Tower Migration Critical Review (Follow-on Epic)

This document is a self-contained critical review of the initial refactor to a Tower-oriented design. It identifies concrete drifts from the agreed design, presents evidence from the current codebase, proposes targeted solutions, and explains how those solutions realign the implementation with the design guidelines and migration plan.

References:

- Design guidelines: `agent-helpers/epics/tower/design-guidelines.md`
- Migration plan: `agent-helpers/epics/tower/migration-plan.md`

---

## 1) Layer order inconsistency (actual vs documented vs design)

### Problem (disagreement with design)

The migration plan specifies a canonical execution order: Run layers → Agent layers → Tool layers → Base execution. The runner module header contradicts this with "Agent → Run → Tool → BaseTool", and the current composition order in code results in Tool layers becoming outermost, not innermost, violating the intended order.

Design intent (migration plan):

- “Execution order: Run layers → Agent layers → Tool layers → Base execution.”

### Evidence

- Runner header documents a conflicting order:

```4:6:src/runner.rs
//! tool-calls through the Tower stack (Agent → Run → Tool → BaseTool), and
//! maintains ordering and state across turns and handoffs. Policy layers and
//! tool execution are composed in `service.rs`; this module focuses on orchestration.
```

- Actual code applies layers in a loop that makes Tool layers the outermost wrapper (applied last):

```607:619:src/runner.rs
// Apply dynamic layers: run-scope then agent-scope (Agent wraps Run)
for l in &config.run_layers {
    stack = l.layer_boxed(stack);
}
let agent_layers = agent.config.agent_layers.clone();
for l in &agent_layers {
    stack = l.layer_boxed(stack);
}
// Apply tool's own layers (from LayeredTool)
for l in &tool_layers {
    stack = l.layer_boxed(stack);
}
```

Since the last applied wrapper is outermost, the current runtime order is Tool → Agent → Run → Base, which inverts the design intent and the migration plan’s “outside-in” model for chaining.

### Solution

- Standardize the canonical order to: Run (outermost) → Agent → Tool → Base.
- In the runner’s stacking logic, apply Tool layers first, then Agent layers, then Run layers last, so Run wraps everything.
- Update runner module header comments to reflect the canonical order.
- Add an integration test that asserts the actual execution order by recording entry/exit across injected probe layers at run-, agent-, and tool-scope.

### Why this better fits the design

This aligns with the migration plan’s “Layer Composition Model” and Tower’s outside-in wrapping semantics, ensuring global policies (run) can uniformly wrap agent- and tool-level behavior, as intended.

---

## 2) `BaseToolService` adapter remains (Phase 5 not completed)

### Problem (disagreement with design)

Phase 5 of the migration plan mandates removing the `BaseToolService` adapter and having tools implement Tower `Service` directly (or be adapted via a single consistent path). The adapter is still central in execution paths.

Design intent (migration plan):

- “Remove `BaseToolService` adapter.”
- “Tools directly implement `Service<ToolRequest<E>>`.”

### Evidence

- Adapter present and used:

```67:79:src/service.rs
/// Base tool executor adapting `dyn Tool` to a Tower Service.
#[derive(Clone)]
pub struct BaseToolService {
    tool: Arc<dyn Tool>,
}

impl BaseToolService {
    pub fn new(tool: Arc<dyn Tool>) -> Self {
        Self { tool }
    }
}
```

- Runner builds stacks via the adapter:

```351:359:src/service.rs
/// Utility to build a boxed service stack for a given tool.
pub fn build_tool_stack<E: crate::env::Env>(
    tool: Arc<dyn Tool>,
) -> BoxService<ToolRequest<E>, ToolResponse, BoxError> {
    // Default lenient schema validation
    let schema = tool.parameters_schema();
    let base = BaseToolService::new(tool);
    let with_schema = InputSchemaLayer::lenient(schema).layer(base);
    BoxService::new(with_schema)
}
```

### Solution

- Replace all uses of `BaseToolService` in the runner with the service-based tool path:
  - Convert tools to services via `tool_service::IntoToolService::into_service::<E>()`.
  - Compose Tower-typed layers directly on the returned service.
- Option A (preferred): move `FunctionTool` (and typed tools) to implement `Service<ToolRequest<E>>` directly, removing the need for adapters.
- Remove `BaseToolService` entirely once the runner and examples are updated.

### Why this better fits the design

This fulfills Phase 5’s goal: “Tools are Tower Services, not wrapped by adapters,” simplifying the mental model and avoiding redundant indirection.

---

## 3) Dynamic erased layers and vector APIs persist (Phase 3 not completed)

### Problem (disagreement with design)

The migration plan requires replacing vector-based, erased-layer APIs with fluent, typed `.layer()` chaining across entities. The codebase still exposes `with_agent_layers(Vec<Arc<dyn ErasedToolLayer>>)` and `with_run_layers(...)`, and relies heavily on `ErasedToolLayer` and boxed helpers.

Design intent (migration plan):

- “Replace `with_agent_layers(vec![...])` and `with_run_layers(vec![...])` with chained `.layer()` calls.”
- “Remove all `Vec<Box<dyn ErasedToolLayer>>` usage.”

### Evidence

- Erased layers and helpers:

```129:138:src/service.rs
pub trait ErasedToolLayer: Send + Sync {
    fn layer_boxed(&self, inner: ToolBoxService) -> ToolBoxService;
}

#[derive(Clone, Copy, Debug)]
pub struct BoxedTimeoutLayer(pub TimeoutLayer);
```

- Tools store erased layers:

```79:83:src/tool.rs
pub struct LayeredTool {
    tool: Arc<dyn Tool>,
    layers: Vec<Arc<dyn ErasedToolLayer>>,
}
```

- Vector-based agent/run APIs:

```186:190:src/agent.rs
pub fn with_agent_layers(mut self, layers: Vec<Arc<dyn ErasedToolLayer>>) -> Self { ... }
```

```204:211:src/runner.rs
pub fn with_run_layers(
    mut self,
    layers: Vec<Arc<dyn crate::service::ErasedToolLayer>>,
) -> Self { ... }
```

### Solution

- Introduce fluent, typed `.layer<L>(...)` on `Agent` and `RunConfig` that return a typed wrapper (mirroring Tower’s `Layered` pattern), eliminating vectors.
- Migrate common policies to typed layers reusable at all scopes (e.g., `TimeoutLayer`, `RetryLayer`, `ApprovalLayer`).
- Deprecate `with_agent_layers` and `with_run_layers` (docs + attributes), then remove after a transition.
- Remove `ErasedToolLayer` and boxed helper functions after all call sites are migrated.

### Why this better fits the design

It restores uniform, type-safe composition throughout the stack and removes dynamic/erased indirection that contradicts Tower’s patterns and the migration plan.

---

## 4) Context system remains in documentation (Phase 2 not completed)

### Problem (disagreement with design)

Phase 2 removes the `ToolContext` system. While the code path is removed, the public documentation and crate docs still teach context handlers and run-context APIs.

Design intent (migration plan):

- “Delete `src/context.rs` entirely… Remove `with_context_factory` from Agent… Convert all context handler logic to layers.”

### Evidence

- Crate docs still reference `ToolContext`:

```23:26:src/lib.rs
//! - **Contextual Runs**: An optional per-run context hook that can observe and
//!   shape tool outputs. See [`ToolContext`](crate::context::ToolContext) and
//!   `examples/contextual.rs`.
```

- README includes large sections on context handlers and run-scoped context functions, none of which exist in code anymore.

- `RunResultWithContext` docs reference removed APIs:

```264:271:src/result.rs
/// Returned by `Runner::run_with_context` when using a typed contextual agent.
pub struct RunResultWithContext<C> { ... }
```

(There is no `run_with_context` in the runner.)

### Solution

- Remove all public references to `ToolContext` and context handlers from crate docs and README.
- Replace those sections with layer-based examples (e.g., stateful accumulation via a stateful Tower layer at agent- or run-scope).
- Update or remove `RunResultWithContext` (or re-document it properly if retained for a different purpose).

### Why this better fits the design

It eliminates mixed metaphors, clarifies the single layering model, and prevents users from adopting deprecated patterns.

---

## 5) Approval capability duplication (`service::HasApproval` vs `env::Approval`)

### Problem (disagreement with design)

The codebase defines approval twice: a `service::HasApproval` bound used by `ApprovalLayer` and an `env::Approval` capability in the Env module. The migration plan calls for capability-based Env with trait bounds.

Design intent (migration plan):

- “Env provides capabilities through trait implementations… Layers declare requirements via trait bounds.”

### Evidence

- `service::HasApproval` is the trait bound for `ApprovalLayer`:

```566:575:src/service.rs
pub trait HasApproval {
    fn approve(&self, agent: &str, tool: &str, args: &Value) -> bool;
}
#[derive(Clone, Copy, Debug, Default)]
pub struct ApprovalLayer;
```

- Separate capability traits exist in `env.rs`:

```139:156:src/env.rs
/// Capability for approval workflows.
pub trait Approval: Send + Sync { ... }
```

### Solution

- Remove `service::HasApproval` entirely.
- Update `ApprovalLayer` to require `E: crate::env::Env` and to call `req.env.capability::<dyn Approval>()` to decide approval; if absent, either allow-all (DX) or deny-by-default (safer) per design choice.
- Update `examples/typed_env_approval.rs` to use the Env capability.

### Why this better fits the design

It unifies capability provision via Env, making requirements explicit and type-safe as intended by Phase 4.

---

## 6) Hard-coded default schema validation in Runner

### Problem (disagreement with design)

`build_tool_stack` unconditionally injects a lenient `InputSchemaLayer` around every tool. The design stresses “Tools manage themselves” and that layers modify behavior; defaults should live with tool constructors, not be forced at run-time.

Design intent (design guidelines + migration plan):

- Tools should own their default layer stacks (see examples using `Default` impl adding timeouts/retries).

### Evidence

- Default schema applied in Runner utility:

```351:359:src/service.rs
let schema = tool.parameters_schema();
let base = BaseToolService::new(tool);
let with_schema = InputSchemaLayer::lenient(schema).layer(base);
BoxService::new(with_schema)
```

### Solution

- Move schema validation defaults to tool constructors/builders or a `Tool::default_layers()` pattern.
- Remove the unconditional schema layer in the runner stack builder.
- Offer explicit public helpers to attach schema layers where desired.

### Why this better fits the design

It preserves the separation of concerns: tools configure tool policy; the runner orchestrates but does not silently alter tool behavior.

---

## 7) Documentation and examples teach deprecated APIs

### Problem (disagreement with design)

README and examples continue to present `with_run_layers`, `with_agent_layers`, `with_tool_layers("name", ...)`, and context handlers. The design and migration plan require uniform `.layer()` chaining and no string-based configuration.

Design intent (design guidelines + migration plan):

- “Everything uses Tower’s `.layer()` pattern.”
- “No string-based lookups or coupling.”

### Evidence

- README shows vector-based and string-coupled usage:

```223:236:README.md
let agent = Agent::simple("Writer", "Be helpful")
  .with_tool(tool.clone())
  .with_agent_layers(vec![
    layers::boxed_timeout_secs(10),
    layers::boxed_retry_times(3),
  ])
  .with_tool_layers("uppercase", vec![ layers::boxed_input_schema_lenient(...)) ]);
```

- README and crate docs include large Context sections.

### Solution

- Rewrite README examples to:
  - Use `.into_service()` and standard Tower `ServiceBuilder` where appropriate.
  - Or, once available, use fluent `.layer(...)` on `Agent` and `RunConfig` (typed variants).
  - Remove `with_tool_layers` completely; show configuring layers at tool creation.
  - Replace context handler examples with stateful Tower layers.

### Why this better fits the design

It gives users a clear, single mental model and prevents reintroduction of deprecated, string-based patterns.

---

## 8) `LayeredTool` carries erased layers and defers composition to Runner

### Problem (disagreement with design)

`LayeredTool` stores `Vec<Arc<dyn ErasedToolLayer>>` and relies on the runner to apply them dynamically. The design calls for uniform, typed composition where tools are services or layered directly, not via erased indirection.

Design intent (design guidelines):

- “Everything uses Tower’s `.layer()` pattern” with typed `Layer<S>`.

### Evidence

- Erased layers in `LayeredTool`:

```79:99:src/tool.rs
pub struct LayeredTool {
    tool: Arc<dyn Tool>,
    layers: Vec<Arc<dyn ErasedToolLayer>>,
}
...
pub fn layer(mut self, layer: Arc<dyn ErasedToolLayer>) -> Self {
    self.layers.push(layer);
    self
}
```

### Solution

- Option A: Remove `LayeredTool` and standardize on `.into_service::<E>().layer(...)` so tools compose like any Tower service.
- Option B: Rework `LayeredTool` to immediately produce a typed service stack internally rather than deferring to runner and erasing types.
- In either case, remove dynamic/erased layering and centralize composition where the layers are declared.

### Why this better fits the design

It eliminates dynamic indirection, restores typed composition, and keeps composition local to where the behavior is declared, improving clarity and safety.

---

## 9) Conflicting/duplicated architecture narratives

### Problem (disagreement with design)

Different documents present conflicting orders and APIs (e.g., runner header vs migration plan order; context presence vs removal; boxed dynamic layers shown as first-class). This confuses engineers adopting the new model.

### Evidence

- Conflicting order documented (see Section 1 evidence).
- Context and string-coupled APIs present in README and crate docs (see Sections 4 and 7).
- Boxed helpers positioned as public DX while migration plan targets removal.

### Solution

- Canonicalize one architecture narrative in `TOWER_ARCHITECTURE.md` and align all other docs to it.
- Add a short “Legacy APIs removed” section listing `with_*_layers`, contexts, and string-based tool configuration as removed.
- Provide a “Before vs After” migration snippet per area (Tool composition, Agent/Run layers, Env capabilities, Ordering), all in Tower idioms.

### Why this better fits the design

A single, consistent story reduces cognitive load and prevents future drift, reinforcing the Tower-first mental model.

---

## 10) Test coverage gaps for new invariants

### Problem (disagreement with design)

The migration plan emphasizes testing; however, we lack explicit tests for cross-scope layer ordering in the runner, for runner integration with service-based tools, and for capability consumption by layers across scopes.

### Evidence

- Existing tests exercise service tools and capability layers in isolation, but runner tests still use the adapter path and vector-based layer APIs.

### Solution

Add tests that verify:

- Cross-scope ordering: inject three probe layers (run, agent, tool) that record entry/exit to assert Run (outermost) → Agent → Tool → Base order.
- Runner + service tools: run an agent whose tools are used via `.into_service::<E>()` and composed with typed layers; verify behavior and usage.
- Capability-based layer: a layer that reads a counter capability and increments across scopes.
- Remove or rewrite tests using vector-based erased-layer APIs.

### Why this better fits the design

It ensures the most important architectural invariants don’t regress and documents behavior through executable specs.

---

## Implementation plan (incremental)

1. Fix runner order and comments; add a cross-scope order test.
2. Add fluent `.layer<L>(...)` to `Agent` and `RunConfig`; mark vector APIs as deprecated.
3. Replace `BaseToolService` with service-based tools in runner; remove adapter.
4. Unify approval capability on `env::Approval`; update `ApprovalLayer`.
5. Remove `ErasedToolLayer` and boxed helpers after migrating call sites.
6. Move default schema from runner to tool constructors/builders.
7. Purge context and string-coupled APIs from README/crate docs; update examples to Tower idioms.
8. Decide and implement the final tool composition path (remove or rework `LayeredTool`).

Each step is independently testable and reduces risk while converging on the target design.

---

## Acceptance criteria

- Layer order is Run → Agent → Tool → Base in both code and docs; verified by tests.
- Tools are Tower Services (or adapted once consistently); `BaseToolService` removed.
- No `ErasedToolLayer` or vector-based layer APIs remain; fluent `.layer()` present on all entities.
- Approval uses Env capabilities only; removal of `service::HasApproval`.
- Runner no longer injects default schema; tools own defaults.
- README, crate docs, and examples reflect only Tower patterns; no contexts or string-coupled APIs.
- All tests green.

---

## Appendix: Quick-reference to design intent

- Uniform composition via Tower’s `.layer()` chaining.
- Explicit dependencies via Env capabilities and constructor injection.
- Clear boundaries: Runs manage run concerns; Agents manage agent concerns; Tools self-manage.
- No string-based coupling.
- Tools as Tower Services.

This review should equip the engineering team with a precise set of changes to align the codebase with the Tower-first architecture and eliminate legacy patterns.
