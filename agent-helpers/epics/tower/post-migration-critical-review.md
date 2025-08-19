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

### Design smells

- Contradictory documentation vs implementation creates ambiguity about guarantees.
- Applying tool layers outermost breaks specificity: the most local policies should be closest to the base.
- Finalization precedence is undefined without a canonical, tested order, leading to surprising outcomes.

### Design smells

- Documentation enabling deprecated APIs creates “broken windows” and perpetuates drift.
- String-based configuration contradicts type-safety goals and invites runtime errors.

### Solution

- Standardize the canonical order to: Run (outermost) → Agent → Tool → Base.
- In the runner’s stacking logic, apply Tool layers first, then Agent layers, then Run layers last, so Run wraps everything.
- Update runner module header comments to reflect the canonical order.
- Add an integration test that asserts the actual execution order by recording entry/exit across injected probe layers at run-, agent-, and tool-scope.

### Why this better fits the design

This aligns with the migration plan’s “Layer Composition Model” and Tower’s outside-in wrapping semantics, ensuring global policies (run) can uniformly wrap agent- and tool-level behavior, as intended.

### TODOs

- Update `src/runner.rs` to apply layers in order: Tool → Agent → Run (applied inner-to-outer so runtime execution is Run → Agent → Tool → Base).
- Update the runner module header comment to document the canonical order.
- Add a dedicated test module (e.g., `tests/layer_ordering.rs`) with probe layers for run/agent/tool scopes that append to a shared log to capture call order.
- Update any docs mentioning a different order (runner header, `TOWER_ARCHITECTURE.md`).

### Acceptance criteria & tests

- A test `test_cross_scope_layer_ordering` asserts the recorded call order is exactly: Run (enter) → Agent (enter) → Tool (enter) → Base → Tool (exit) → Agent (exit) → Run (exit).
- A test `test_final_effect_precedence` sets a tool-scope layer and a run-scope layer to each produce `Effect::Final(_)` and asserts the run-scope finalization wins (outermost precedence).
- Grep check: no remaining references to the older documented order in code comments or docs.

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

### Design smells

- Adapter indirection hides the true execution surface and fragments the mental model.
- Dual execution paths (adapter vs direct service) complicate testing and maintenance.
- Inconsistent composition semantics across tools vs services increase cognitive load.

### Solution

- Replace all uses of `BaseToolService` in the runner with the service-based tool path:
  - Convert tools to services via `tool_service::IntoToolService::into_service::<E>()`.
  - Compose Tower-typed layers directly on the returned service.
- Option A (preferred): move `FunctionTool` (and typed tools) to implement `Service<ToolRequest<E>>` directly, removing the need for adapters.
- Remove `BaseToolService` entirely once the runner and examples are updated.

### Why this better fits the design

This fulfills Phase 5’s goal: “Tools are Tower Services, not wrapped by adapters,” simplifying the mental model and avoiding redundant indirection.

### TODOs

- Replace all usages of `BaseToolService` and `build_tool_stack` with the service-based tool path using `tool_service::IntoToolService` (or direct `Service` impls on tools).
- Remove `BaseToolService` and `build_tool_stack` from `src/service.rs`.
- Update runner execution to compose Tower layers on service tools directly.
- Update examples (`examples/typed_env_approval.rs`, others) and tests to use the service-based path.

### Acceptance criteria & tests

- Grep checks: `BaseToolService` and `build_tool_stack` no longer exist; no references remain.
- Integration test `test_runner_uses_service_tools` executes a tool through the runner and asserts expected output; this test imports no adapter types.
- Examples compile and run using service tools with standard Tower middleware composition.

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

### Design smells

- Erased trait objects (`ErasedToolLayer`) undermine type safety and Tower’s typed composition.
- Vector-based APIs obscure order and discourage fluent, readable chaining.
- Boxed helper catalog (`boxed_*`) becomes a parallel composition system, diverging from Tower idioms.

### Solution

- Introduce fluent, typed `.layer<L>(...)` on `Agent` and `RunConfig` that return a typed wrapper (mirroring Tower’s `Layered` pattern), eliminating vectors.
- Migrate common policies to typed layers reusable at all scopes (e.g., `TimeoutLayer`, `RetryLayer`, `ApprovalLayer`).
- Deprecate `with_agent_layers` and `with_run_layers` (docs + attributes), then remove after a transition.
- Remove `ErasedToolLayer` and boxed helper functions after all call sites are migrated.

### Why this better fits the design

It restores uniform, type-safe composition throughout the stack and removes dynamic/erased indirection that contradicts Tower’s patterns and the migration plan.

### TODOs

- Add fluent, typed `.layer<L>(...)` chaining APIs for `Agent` and `RunConfig` (returning typed wrappers mirroring Tower’s pattern).
- Deprecate `with_agent_layers` and `with_run_layers` (doc deprecation + `#[deprecated]`), and migrate all call sites across examples/tests.
- Remove `ErasedToolLayer` and all `boxed_*` helpers after migration; replace with typed layers used directly.
- Update `LayeredTool` (see Section 8) or migrate off it to typed/service layering.

### Acceptance criteria & tests

- Grep checks: no occurrences of `with_agent_layers`, `with_run_layers`, `ErasedToolLayer`, or `boxed_timeout_secs`/`boxed_retry_times`/`boxed_input_schema_*` remain.
- New tests: `test_agent_layer_chaining_compiles` and `test_runconfig_layer_chaining_compiles` compose two typed layers and assert both effects occur in expected order.
- All examples compile using fluent `.layer(...)` APIs without vectors.

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

### Design smells

- Mixed metaphors between layers and contexts encourage “action at a distance”.
- Docs advertising removed APIs lead to dead ends and onboarding friction.
- Conceptual surface area expands beyond the intended uniform Tower model.

### Solution

- Remove all public references to `ToolContext` and context handlers from crate docs and README.
- Replace those sections with layer-based examples (e.g., stateful accumulation via a stateful Tower layer at agent- or run-scope).
- Update or remove `RunResultWithContext` (or re-document it properly if retained for a different purpose).

### Why this better fits the design

It eliminates mixed metaphors, clarifies the single layering model, and prevents users from adopting deprecated patterns.

### TODOs

- Remove all references to `ToolContext`, `with_context_factory`, and run-context APIs from `src/lib.rs` docs and `README.md`.
- Remove or rewrite any examples that reference contexts; replace with stateful Tower layers.
- Update or remove `RunResultWithContext` and associated docs; if retained for different semantics, provide up-to-date examples and tests.

### Acceptance criteria & tests

- Grep checks: no `ToolContext`, `with_context`, or `with_context_factory` references across the repo.
- README and crate docs sections replaced with layer-based examples; `cargo doc` builds without broken references.
- Any doctests or examples added for stateful layers compile and pass.

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

### Design smells

- Duplicated approval concepts (`HasApproval` vs Env `Approval`) create confusion and divergent extension points.
- Layers bound to bespoke traits bypass the Env capability system, undermining uniformity.

### Solution

- Remove `service::HasApproval` entirely.
- Update `ApprovalLayer` to require `E: crate::env::Env` and to call `req.env.capability::<dyn Approval>()` to decide approval; if absent, either allow-all (DX) or deny-by-default (safer) per design choice.
- Update `examples/typed_env_approval.rs` to use the Env capability.

### Why this better fits the design

It unifies capability provision via Env, making requirements explicit and type-safe as intended by Phase 4.

### TODOs

- Delete `service::HasApproval`.
- Refactor `ApprovalLayer` to require `E: crate::env::Env` and use `req.env.capability::<dyn env::Approval>()` to decide.
- Update examples (`examples/typed_env_approval.rs`) and tests to provide `env::Approval` via `EnvBuilder` or typed Env.
- Decide deny-by-default vs allow-by-default when capability is absent; document behavior and add tests.

### Acceptance criteria & tests

- Grep checks: no references to `HasApproval` remain.
- Test `approval_layer_denies_without_capability` uses an Env without `Approval` and asserts denial per chosen policy.
- Test `approval_layer_allows_with_capability` provides an Env with `Approval` that approves and asserts success.
- Example `typed_env_approval.rs` runs and demonstrates capability-driven approval.

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

### Design smells

- Hidden policy injection in the runner violates separation of concerns.
- Surprising defaults make behavior non-local to the tool definition.

### Solution

- Move schema validation defaults to tool constructors/builders or a `Tool::default_layers()` pattern.
- Remove the unconditional schema layer in the runner stack builder.
- Offer explicit public helpers to attach schema layers where desired.

### Why this better fits the design

It preserves the separation of concerns: tools configure tool policy; the runner orchestrates but does not silently alter tool behavior.

### TODOs

- Remove unconditional `InputSchemaLayer` insertion from the runner (`build_tool_stack`); migrate any remaining stack builder to not apply schema by default.
- Provide explicit helpers to attach `InputSchemaLayer::{lenient,strict}` at tool/agent/run scopes as needed.
- Optionally, define recommended defaults in tool constructors (e.g., `FunctionTool::default()` adds conservative layers) and document them clearly.

### Acceptance criteria & tests

- Test `schema_not_enforced_without_layer` constructs a tool whose schema would reject arguments under strict mode; without attaching schema layer, the call succeeds.
- Test `schema_enforced_with_strict_layer` attaches strict `InputSchemaLayer` and asserts invalid args return an error.
- Grep checks: runner code no longer injects schema layers implicitly.

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

### TODOs

- Rewrite README to remove `with_*layers`, `with_tool_layers("name", ...)`, and all context content; replace with Tower `.layer(...)` examples and service-based tools.
- Update all examples to use typed `.layer(...)` or Tower `ServiceBuilder` composition; remove string-based per-tool configuration.
- Add an example that demonstrates typed Env capability consumption inside a custom layer.

### Acceptance criteria & tests

- Grep checks: no occurrences of `with_tool_layers`, `with_agent_layers`, `with_run_layers`, or `ToolContext` in README and examples.
- Run `cargo run --example <name>` across all examples; each compiles and runs successfully with the new APIs.
- README quick-start code compiles as a doctest (if enabled) or matches a real example.

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

### Design smells

- Layering deferred to the runner splits responsibility and makes behavior non-local to the tool.
- Erased layers stored out-of-band prevent typed, compile-time-checked composition.

### Solution

- Option A: Remove `LayeredTool` and standardize on `.into_service::<E>().layer(...)` so tools compose like any Tower service.
- Option B: Rework `LayeredTool` to immediately produce a typed service stack internally rather than deferring to runner and erasing types.
- In either case, remove dynamic/erased layering and centralize composition where the layers are declared.

### Why this better fits the design

It eliminates dynamic indirection, restores typed composition, and keeps composition local to where the behavior is declared, improving clarity and safety.

### TODOs

- Decide final approach:
  - Remove `LayeredTool` in favor of service-based layering; or
  - Rework `LayeredTool` to build a typed service stack immediately and stop storing erased layers.
- Migrate runner away from reading erased layers from tools; tools should expose already-layered services or typed layer builders.
- Remove `ErasedToolLayer` and its boxed helpers if the service-based approach is chosen.

### Acceptance criteria & tests

- Grep checks: no references to `LayeredTool` storing `ErasedToolLayer` remain; either `LayeredTool` is removed or reworked to typed composition.
- Test `tool_level_layer_applies_in_service_path` creates a tool with a tool-scope layer and verifies the layer’s effect when executed via runner.
- No dynamic/erased layering is required for tool behavior.

---

## 9) Conflicting/duplicated architecture narratives

### Problem (disagreement with design)

Different documents present conflicting orders and APIs (e.g., runner header vs migration plan order; context presence vs removal; boxed dynamic layers shown as first-class). This confuses engineers adopting the new model.

### Evidence

- Conflicting order documented (see Section 1 evidence).
- Context and string-coupled APIs present in README and crate docs (see Sections 4 and 7).
- Boxed helpers positioned as public DX while migration plan targets removal.

### Design smells

- Multiple contradictory narratives lead to cargo-culting and regressions.
- Lack of a single canonical source of truth undermines design clarity and enforcement.

### Solution

- Canonicalize one architecture narrative in `TOWER_ARCHITECTURE.md` and align all other docs to it.
- Add a short “Legacy APIs removed” section listing `with_*_layers`, contexts, and string-based tool configuration as removed.
- Provide a “Before vs After” migration snippet per area (Tool composition, Agent/Run layers, Env capabilities, Ordering), all in Tower idioms.

### Why this better fits the design

A single, consistent story reduces cognitive load and prevents future drift, reinforcing the Tower-first mental model.

### TODOs

- Update `TOWER_ARCHITECTURE.md` to state the canonical layer order and remove references to deprecated APIs.
- Update `DESIGN.md` and `MIGRATION_GUIDE.md` to remove mentions of context handlers, string-coupled configuration, erased layers, and adapter paths.
- Add a “Legacy APIs removed” section listing removed APIs and their replacements.
- Add "Before vs After" snippets for: tool composition, agent/run layering, Env capabilities, and ordering.

### Acceptance criteria & tests

- Grep checks: no references to removed APIs (`with_*layers`, contexts, `ErasedToolLayer`, `BaseToolService`) across docs.
- Documents are internally consistent regarding layer order and composition models.
- Spot-check by a reviewer: following docs alone, a developer can implement a tool with layers and run it successfully.

---

## 10) Test coverage gaps for new invariants

### Problem (disagreement with design)

The migration plan emphasizes testing; however, we lack explicit tests for cross-scope layer ordering in the runner, for runner integration with service-based tools, and for capability consumption by layers across scopes.

### Evidence

- Existing tests exercise service tools and capability layers in isolation, but runner tests still use the adapter path and vector-based layer APIs.

### Design smells

- Missing tests for core invariants allow silent drift from the architecture.
- Reliance on adapter-path tests obscures the intended service-based path.

### Solution

Add tests that verify:

- Cross-scope ordering: inject three probe layers (run, agent, tool) that record entry/exit to assert Run (outermost) → Agent → Tool → Base order.
- Runner + service tools: run an agent whose tools are used via `.into_service::<E>()` and composed with typed layers; verify behavior and usage.
- Capability-based layer: a layer that reads a counter capability and increments across scopes.
- Remove or rewrite tests using vector-based erased-layer APIs.

### Why this better fits the design

It ensures the most important architectural invariants don’t regress and documents behavior through executable specs.

### TODOs

- Add the following tests:
  - `test_cross_scope_layer_ordering`
  - `test_final_effect_precedence`
  - `test_runner_uses_service_tools`
  - `test_agent_layer_chaining_compiles`
  - `test_runconfig_layer_chaining_compiles`
  - `approval_layer_denies_without_capability` and `approval_layer_allows_with_capability`
  - `schema_not_enforced_without_layer` and `schema_enforced_with_strict_layer`
  - `tool_level_layer_applies_in_service_path`
- Remove or rewrite tests that depend on vector-based erased layers or `BaseToolService`.

### Acceptance criteria & tests

- All new tests compile and pass in CI.
- No tests import or reference removed APIs.
- Coverage (conceptual): cross-scope ordering, capability usage, service tools in runner, schema behavior, fluent agent/run layering.

---

## Order of operations (with rationale)

1. Fix runner layer order and document it; add cross-scope order tests.

   - Rationale: Establishes the global invariant other changes depend on; prevents rework by locking the contract early.

2. Introduce fluent, typed `.layer<L>(...)` APIs for `Agent` and `RunConfig`; deprecate vector APIs.

   - Rationale: Enables migration off erased layers and vectors while keeping code building; prepares for removal steps.

3. Switch runner to service-based tools and remove the `BaseToolService` adapter.

   - Rationale: Unifies the execution model; reduces dual-path complexity before broad removals; aligns tests and examples.

4. Unify approval on Env capability in `ApprovalLayer`.

   - Rationale: Small, contained change that clarifies capability access patterns used by subsequent layers and examples.

5. Remove `ErasedToolLayer` and boxed helpers once typed APIs and runner are in place.

   - Rationale: Avoids premature deletion; ensures a smooth cutover with compile-time guidance in place.

6. Move default schema behavior out of the runner into tool constructors/builders.

   - Rationale: Separation of concerns; low risk once the execution path is stable.

7. Purge deprecated APIs and contexts from docs/examples; align architecture docs.

   - Rationale: Documentation last to reflect finalized APIs and prevent churn; reduces drift going forward.

8. Decide and implement the final `LayeredTool` approach (remove or rework to typed composition).
   - Rationale: Safer to resolve after typed APIs and execution path are stable to avoid conflicting layering flows.

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
