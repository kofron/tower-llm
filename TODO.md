## Migration Plan: Tower-driven Execution (Tools, Layers, Env)

Goal: Transition the codebase to a Tower-based architecture for tool execution and policies (layers), while maintaining a continuously working repo (examples and tests) throughout. Prioritize an early end-to-end (E2E) path that runs agents with tools via Tower stacks.

Principles

- Keep the simple path simple. Maintain a non-generic default runner API while introducing typed variants for advanced use.
- Ship incremental E2E value early and keep all examples runnable (with minimal interface changes) at each major milestone.
- Maintain green tests; update or add tests alongside each change.
- Prefer additive changes first; remove/deprecate only after parity.
- Layers are UNIFIED and reusable: generic policy layers (timeouts, retries, schema, approval, tracing) are a single type that works at any stack position (run, agent, or tool) without separate variants.

Decisions (locked-in)

- Preserve provider/tool-call order in replies; no reordering. Concurrency is an internal optimization only.
- Fixed layer order: Run → Agent → Tool → BaseTool; not configurable in v1. Outermost `Final` wins.
- Generic policy layers are scope-agnostic (same type usable in run-, agent-, or tool-segments).
- Schema validation defaults lenient (coercion best-effort); strict mode opt-in.
- Timeouts/retries are opt-in (off by default) with ergonomic helpers.
- Approval defaults to auto-approve for parity; warn if a tool marks `requires_approval()` and no approval policy is configured.
- Internal error type through stack; convert to strings only at message/protocol boundary.
- Typed Env APIs remain optional (advanced); the primary runner stays non-generic.

Milestones Overview

1. Scaffold core Tower primitives (no behavior change).
2. Hard cutover: replace runner tool execution with a Tower stack (DefaultEnv); get E2E working.
3. Parallel tool calls via Tower stack (preserve provider protocol ordering).
4. Context handlers as layers (run-scoped, per-agent) with parity tests; remove old path.
5. Handoffs in Tower path; parity with current handoff behavior.
6. Built-in layers: schema validation, timeouts, retries, approval.
7. Examples migrated to Tower APIs where needed; keep them all working.
8. Unify `ToolCall` types and remove any legacy execution code.
9. Ergonomics: typed tool I/O, optional typed Env APIs, docs, and cleanup.

Detailed Plan & Checklists

Phase 0 — Inventory and guardrails (Day 0)

- [x] Tag current `main` and enable CI to run full test suite on PRs.
- [x] Document current behavior invariants (messages format, tool call protocol, handoff shape, context ordering) in `README.md` (Architecture section) for reference.
- [x] Add a short top-level note in `README.md` that a Tower hard cutover is in progress (users adopt by upgrading the crate version).

Phase 1 — Core Tower scaffolding (no behavior change)
Deliverables:

- Introduce Tower dependency and core types (new module `src/service.rs` or `src/engine/`):
  - [x] `Effect` enum: `Continue | Rewrite(Value) | Final(Value) | Handoff(String)`.
  - [x] `ToolRequest<E>` and `ToolResponse` structs (generic Env only used by typed path initially).
  - [x] `BaseToolService` adapter for existing `dyn Tool` → Tower `Service<ToolRequest<E>>`.
  - [x] `ModelService` adapter for `ModelProvider` → Tower `Service<ModelRequest>`.
- [x] Unit tests for services/layers covering behavior and pass-through.

Acceptance criteria:

- All existing tests pass unchanged.
- New module compiles; not yet wired into `Runner`.

Phase 2 — Hard cutover: Tower in the primary runner
Deliverables:

- [x] Introduce an internal default env type `DefaultEnv` (no capabilities required yet).
- [x] Replace the tool execution branch in `Runner` with a Tower stack using `DefaultEnv`: `AgentContextLayer` → `RunContextLayer` → `BaseToolService`.
- [x] Reuse current `ModelProvider` unchanged.
- [x] Map existing context handlers to layers internally to preserve behavior.
- [x] Update examples and tests as needed so everything passes using the Tower-backed runner.
  - NOTE: Policy layers (timeouts/retries/schema/approval) are single, reusable types that can be plugged into run-, agent-, or tool-segments without separate variants.

Acceptance criteria:

- All tool execution goes through Tower.
- Tests and examples run successfully end-to-end on the new path.
- Parity tests: run-layer then agent-layer ordering; Final/Rewrite semantics match prior tests.

Phase 3 — Parallel tool calls with ordering preservation
Deliverables:

- [x] In the Tower-based tool execution, launch multiple tool-call futures concurrently per turn.
- [x] Preserve reply ordering according to provider tool_calls order (buffer responses and emit in sequence).
- [x] Ensure run-scoped and agent layers remain concurrency-safe (document expectation; guard state with `Mutex` where needed).
- [x] Add tests that assert: (a) parallel execution happens, (b) replies are ordered, (c) context layers see all calls.

Acceptance criteria:

- [x] New tests for concurrency and ordering pass consistently.
- [x] `examples/parallel_tools.rs` demonstrates improved latency while preserving reply order (manual verification acceptable).
  - [x] Add a note to the example output indicating ordering preservation under parallelism.
  - [x] Run-scoped rewrite verified under parallel calls (runner-level test).

Phase 4 — Context-to-Layer parity and removal of old path
Deliverables:

- [x] Replace internal usage of `context.rs` handlers by Tower `RunContextLayer` and `AgentContextLayer` everywhere.
- [x] Keep public context APIs (`with_context`, `with_context_factory`, `RunConfig::with_run_context`) but implement them by constructing layers.
- [x] Add tests verifying ordering: run-layer first, then agent-layer; `Final` short-circuits properly.
- [x] Remove now-unused direct handler paths.

Acceptance criteria:

- [x] All existing context-related tests ported to call into Tower layers; behavior unchanged.

Phase 5 — Context-to-Layer parity and deprecation (merged with Phase 4)
Deliverables:

- [x] Replace internal usage of `context.rs` handlers by Tower `RunContextLayer` and `AgentContextLayer` everywhere.
- [x] Keep public context APIs (`with_context`, `with_context_factory`, `RunConfig::with_run_context`) but implement them by constructing layers.
- [x] Add tests verifying ordering: run-layer then agent-layer wrapping semantics; `Final` short-circuits properly.
- [x] Mark old handler types as deprecated in docs (not code yet) with guidance to layers.

Acceptance criteria:

- All existing context-related tests ported to call into Tower layers; behavior unchanged.

Phase 6 — Handoffs in Tower path
Deliverables:

- [x] Decide: keep runner-level handoff special-case (Path A) for now.
  - Path A (fast): keep runner interception of handoff tool names, returning `Effect::Handoff` internally.
  - Path B (tower-native): add `HandoffLayer` that recognizes handoff tools and returns `Effect::Handoff` (future).
- [x] Ensure the tool reply ACK for handoff is inserted and the agent switch happens exactly as before.
- [x] Tests: handoff exposed as tool to provider; tool reply ACK present; final agent recorded.

Acceptance criteria:

- All handoff tests remain green; `examples/group_*` run successfully.

Phase 7 — Built-in cross-cutting layers
Deliverables:

- [x] `InputSchemaLayer` (validate/coerce args vs `Tool::parameters_schema`).
- [x] `TimeoutLayer` (per-tool/per-policy; cancellation-safe).
- [x] `RetryLayer` (e.g., retry on transient errors; with optional fixed delay). Tests added.
- [x] `ApprovalLayer` (enforce external approval via `HasApproval`). Tests added.
- [x] Wire defaults conservatively: off by default or minimal policies to avoid surprises.
- [x] Add targeted tests for each layer (success, timeout, retry logic, approval denied path).
  - NOTE: Each is a single type implementing `Layer<E>` and is scope-agnostic; you can attach the same type at run-, agent-, or tool-level.
  - [x] DX: dynamic run-scope layer attachment via `RunConfig::with_run_layers` and boxed layer helpers (timeout/retry/schema).
  - [x] DX: agent-scope dynamic layers via `Agent::with_agent_layers`.
  - [x] DX: tool-scope dynamic layers via `Agent::with_tool_layers` (per-tool).

Acceptance criteria:

- Layers compile and are opt-in; examples demonstrate usage; tests cover behaviors.

Phase 8 — Examples migration and DX polish
Deliverables:

- [x] Update examples to Tower-based runner path without breaking user flow. Minimal changes:
  - `tool_example.rs` (no change or add comment “Tower-backed”).
  - `contextual.rs` (uses context via layers).
  - `parallel_tools.rs` (shows concurrency benefit with ordering preservation).
  - Group/handoff examples unchanged conceptually.
- [x] Documentation snippets in `README.md` explaining the Tower architecture and layer composition.
  - [x] Initial README updates describing Tower execution and scope-agnostic layers.
  - [x] Expand with layer usage examples (schema/timeout/retry/approval).
  - [x] Add a tool-scope layer example (`examples/tool_scope_timeout.rs`).

Acceptance criteria:

- All examples compile and run. CI executes `cargo run --example ...` where feasible.

Phase 9 — Type cleanup and unification
Deliverables:

- [x] Unify duplicate `ToolCall` definitions (keep the `items::ToolCall` and remove or privatize the one in `tool.rs`).
- [x] Move any protocol-only types to a single module (`items.rs`) used by model and runner.
- [x] Audit stringifying of tool outputs in `Message::tool` to ensure JSON is consistently preserved/encoded.

Acceptance criteria:

- No duplicate `ToolCall` types; build remains green.

Phase 10 — Ergonomics and optional typed Env APIs
Deliverables:

- [x] Derive macros for tool input/output types to generate schema + serde: attribute macros `#[tool_args]`, `#[tool_output]` in new `tower-llm-derive` crate.
- [x] A `TypedFunctionTool<Input, Output>` wrapper adapting typed functions to `Value` + schema.
- [x] Introduce optional typed Env and capability traits for advanced users (generic APIs), keeping the core Tower runner unaffected.
- [x] Examples showing typed tools and, optionally, typed Env usage.

Acceptance criteria:

- Examples compile; schema validation layer passes typed schemas; tests for (de)serialization added.
  - [x] New example `typed_tool_derive.rs` demonstrates macros with `TypedFunctionTool`.
  - [x] `typed_env_approval.rs` demonstrates advanced typed Env with `ApprovalLayer` and `HasApproval` trait.

Phase 11 — Performance, locks, and QA matrix
Deliverables:

- [x] Audit lock contention in run/agent layers under parallel calls; refactor to minimize blocking.
  - [x] Reduced mutex hold time in context layers by dropping the lock before handler execution and reacquiring only to store state.
  - [x] Context layers hold a short critical section without `.await`; README updated with locking guidance and tuning tips.
- [x] Add benchmarks (criterion) for sequential vs parallel tool calls.
- [ ] Expand test matrix: with/without parallelism, with timeouts/retries, with handoffs, session persistence on.
  - [x] Max concurrency limiting test (`with_max_concurrency`) under parallel execution.
  - [x] Retry layer interplay under parallel execution; preserved ordering.
  - [x] Timeout layer interplay under parallel execution.
  - [x] Approval layer interplay under parallel execution.
  - [x] Handoff behavior under parallel execution.
  - [x] Session persistence path with parallel execution.

Acceptance criteria:

- [x] Benchmarks recorded; no regressions beyond acceptable thresholds; documentation updated with guidance.

Cutover Checklist (100% Tower)

- [x] Runner uses Tower stack for tool execution exclusively.
- [x] Run-scoped / per-agent context mapped to layers; old handler path removed.
- [x] Handoffs work identically (ACK tool message, agent switch, session history).
- [x] Parallel tool calls supported and ordered; can be toggled.
- [x] Examples and tests cover tools, context rewriting, finalization, handoffs, parallelism, layers (timeout/retry/approval/schema).
- [x] Duplicated types removed; public API documented and stable.

Risk Mitigations & Notes

- Cut over the runner to Tower early; keep examples/tests green as we iterate.
- Guard stateful layers with `Send + Sync` and clear docs; prefer stateless layers where possible.
- Validate layer ordering and `Final` precedence with explicit tests (run-layer before agent-layer; outermost `Final` wins).

Developer Quickstart (during migration)

```bash
cargo test
cargo run --example tool_example
cargo run --example contextual
cargo run --example parallel_tools
```

File/Module map (proposed additions)

- `src/service.rs` (or `src/engine/`): Tower traits, `Effect`, `ToolRequest<E>`, `ToolResponse>`, adapters.
- `src/layers/*.rs`: `run_context.rs`, `agent_context.rs`, `timeout.rs`, `retry.rs`, `approval.rs`, `schema.rs`, `tracing.rs`.
- `src/runner_tower.rs` (optional) or integrate into `src/runner.rs` incrementally.
- Keep `src/agent.rs`, `src/items.rs`, `src/model.rs` as the public façade; wire through Tower under the hood.

Final API Surface (spec and examples)

High-level concepts

- Unified layers: same policy layer types can be added at run, agent, or tool scope. The stack is composed as Run → Agent → Tool → BaseTool.
- Ordering and precedence are fixed (not configurable). Outermost `Final` wins.
- Reply ordering preserves provider/tool-call sequence even with parallel calls.
- Default behavior parity: if no layers are attached, behavior matches today’s runner.
- Approval defaults to auto-approve; warn on tools flagged `requires_approval()` without an approval policy.
- Schema validation is lenient by default; strict is opt-in.
- Errors are typed internally; only stringified at the protocol boundary.

Run configuration (simple)

```rust
use tower_llm::{Agent, Runner, runner::RunConfig, FunctionTool};
use std::sync::Arc;

let tool = Arc::new(FunctionTool::simple("uppercase", "Upper", |s: String| s.to_uppercase()));
let agent = Agent::simple("Writer", "Be helpful").with_tool(tool);

let result = Runner::run(agent, "hello", RunConfig::default()).await?;
```

Adding run-level policies (scope-agnostic layers)

```rust
use tower_llm::{runner::RunConfig, layers};

let layers_vec = vec![
  layers::boxed_timeout_secs(10),
  layers::boxed_retry_times(3),
  layers::boxed_input_schema_lenient(serde_json::json!({
    "type": "object",
    "properties": {"input": {"type":"string"}},
    "required": ["input"]
  })),
];

let config = RunConfig::default()
  .with_parallel_tools(true)
  .with_run_layers(layers_vec);
```

Adding agent-level shaping

```rust
use tower_llm::layers::AgentContextLayer;

let agent = agent.with_agent_layers(vec![
  AgentContextLayer::rewrite(|mut ctx, req, res| {
    // ctx is per-agent context; res exposes map_value/forward/final helpers
    let res = res.map_value(|v| serde_json::json!({ "agent": req.agent, "value": v }));
    ctx.count += 1;
    (ctx, res)
  })
]);
```

Adding tool-level last-mile shaping

```rust
use tower_llm::{Agent, layers, FunctionTool, Runner, runner::RunConfig};
use std::sync::Arc;

let echo = Arc::new(FunctionTool::simple("echo", "echoes", |s: String| s));
let agent = Agent::simple("ToolShape", "Use tools")
  .with_tool(echo.clone())
  // Attach a per-tool timeout only for "echo"
  .with_tool_layers("echo", vec![layers::boxed_timeout_secs(2)]);

let cfg = RunConfig::default();
let _ = Runner::run(agent, "hi", cfg).await?;
```
