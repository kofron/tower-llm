# Tower Migration Critical Review (Current State)

## Progress snapshot

- Architecture largely aligns with Tower-first design: tools-as-services are available, Env capabilities exist, typed `.layer()` surfaces for `Agent` and `RunConfig` compile, runner has custom env entry points, tests cover most layers.
- Remaining drifts: hidden policy injection in runner, sequential path Env/type usage, un-integrated typed Agent/Run layering, mixed adapter vs direct `Service` story for tools, doc/examples mismatches, small legacy debris.

---

## 1) Runner injects schema validation implicitly

### Problem

Implicit `InputSchemaLayer` is added inside runner stacks. This violates the design principle “Tools do work. Layers modify behavior.” and “Each level manages its own concerns.” Policy must be explicit.

### Evidence

```912:916:src/runner.rs
// Add default schema validation (to be moved to tool constructors in Step 6)
let with_schema =
    InputSchemaLayer::lenient(schema).layer(tool_service);
BoxService::new(with_schema)
```

```1261:1274:src/runner.rs
// Start with schema validation (to be moved to tool constructors in Step 6)
let with_schema = InputSchemaLayer::lenient(schema).layer(tool_service);

// Auto-apply ApprovalLayer if environment has approval capability
if env.capability::<crate::env::ApprovalCapability>().is_some()
    || env.capability::<crate::env::AutoApprove>().is_some()
    || env.capability::<crate::env::ManualApproval>().is_some() {
    let with_approval = ApprovalLayer.layer(with_schema);
    BoxService::new(with_approval)
} else {
    BoxService::new(with_schema)
}
```

### Design smell

- Hidden defaults in orchestration violate separation of concerns and surprise users.
- Docs claim explicit layering; code enforces implicit lenient validation.

### Recommendation

- Remove all implicit `InputSchemaLayer` insertion in runner.
- If desired, provide documented helpers or defaults at tool construction, not at run-time orchestration.

### TODOs

- [ ] Remove `InputSchemaLayer::lenient(...)` usage in runner sequential path:
  - Edit `src/runner.rs`: delete the schema wrapping block near the sequential tool execution path (~lines 912–916) and return the base service instead.
- [ ] Remove schema wrapping from `create_tool_service_stack::<E>` (~lines 1261–1274) and return the base service unchanged.
- [ ] Update tests so schema behavior is explicit through attached layers (keep `tests/schema_behavior.rs` as the canonical spec; add runner-path variant, see §8).
- [ ] Update README/Migration Guide to state no default schema enforcement occurs unless layer is attached.

### Acceptance criteria

- Grep shows no `InputSchemaLayer` usage inside runner tool stack creation.
- A runner-path test shows invalid args pass without schema layer and fail when strict layer is attached.

---

## 2) Sequential runner path ignores user Env (pins DefaultEnv)

### Problem

Sequential path constructs `ToolRequest<DefaultEnv>` and a `Service<ToolRequest<DefaultEnv>>`, ignoring `E` provided via `run_with_env`.

### Evidence

```923:931:src/runner.rs
let req = ToolRequest::<DefaultEnv> {
    env: DefaultEnv,
    run_id: trace_id.clone(),
    agent: agent.name().to_string(),
    tool_call_id: id.clone(),
    tool_name: name.clone(),
    arguments: args.clone(),
};
```

### Design smell

- Capability-based layers cannot observe Env in sequential mode; parity with parallel path is broken.

### Recommendation

- Make sequential execution use the same generic `E: Env` stack as parallel execution.

### TODOs

- [ ] Replace the sequential inline `ToolService<ToolRequest<DefaultEnv>>` with a call to a generic stack factory (e.g., `create_tool_service_stack::<E>(&tool, &env)`), returning `BoxService<ToolRequest<E>, ToolResponse, BoxError>`.
- [ ] Construct `ToolRequest::<E>` using the provided `env` instead of `DefaultEnv`.
- [ ] Delete the entire `impl Service<ToolRequest<DefaultEnv>> for ToolService` block in the sequential path to avoid divergent code.

### Acceptance criteria

- No construction of `ToolRequest::<DefaultEnv>` in runner execution paths.
- New tests demonstrate that a custom Env capability (e.g., approval) is honored in both sequential and parallel modes.

---

## 3) Runner auto-applies ApprovalLayer based on capabilities (hidden policy)

### Problem

The runner injects `ApprovalLayer` when an approval capability exists in Env. Layers must be explicit.

### Evidence

```1265:1273:src/runner.rs
if env.capability::<crate::env::ApprovalCapability>().is_some()
    || env.capability::<crate::env::AutoApprove>().is_some()
    || env.capability::<crate::env::ManualApproval>().is_some() {
    let with_approval = ApprovalLayer.layer(with_schema);
    BoxService::new(with_approval)
}
```

### Design smell

- Action-at-a-distance: enabling a capability silently changes runtime behavior.
- Contradiction with “Uniform Composition” and explicit dependency model.

### Recommendation

- Remove automatic `ApprovalLayer` insertion. Require users to attach `ApprovalLayer` explicitly at their desired scope.

### TODOs

- [ ] Delete the conditional `ApprovalLayer` wrapping in `create_tool_service_stack::<E>`.
- [ ] Update `examples/approval.rs` to attach `ApprovalLayer` explicitly (run-scope or agent-scope), and to illustrate deny-by-default within the layer (when present) and pass-through when the layer is absent.
- [ ] Update tests that currently pass due to auto-injection to instead attach the layer in the test setup.

### Acceptance criteria

- Approval is enforced only when `ApprovalLayer` is explicitly present.
- Tests cover approval allowed/denied with and without the layer.

---

## 4) Typed `.layer()` on Agent/RunConfig not applied during execution

### Problem

`Agent::layer` and `RunConfig::layer` produce typed wrappers, but runner accepts concrete `Agent` and `RunConfig` and ignores these layers. Comments mention removed erased vectors, but the typed composition isn’t integrated.

### Evidence

```361:363:src/agent.rs
pub fn layer<L>(self, layer: L) -> LayeredAgent<L, Self> {
    LayeredAgent::new(layer, self)
}
```

```451:456:src/runner.rs
pub async fn run_with_env<E: crate::env::Env>(
    agent: Agent,
    input: impl Into<String>,
    config: RunConfig,
    env: E,
) -> Result<RunResult> {
```

### Design smell

- API suggests typed chaining works at run/agent scope, but composition is not in effect at runtime.

### Recommendation (choose one path and make it consistent)

- Option A: Integrate typed wrappers.
  - Introduce traits like `AgentPolicy` and `RunPolicy` with a `wrap<S>(self, service: S) -> L::Service` that apply the chain of layers.
  - Accept generic `A: AgentLike + AgentPolicy` and `R: RunConfigLike + RunPolicy` in runner, and apply in order Run → Agent → Tool → Base.
- Option B: Remove typed wrappers for now; document service-based layering as the single composition path (`tool.into_service::<E>().layer(...)`) and keep runner orchestration minimal.

### TODOs

- [ ] Decide A or B in DESIGN_UPDATES.md.
- If A:
  - [ ] Define `RunPolicy` and `AgentPolicy` traits that can be implemented for `LayeredRunConfig<..>` and `LayeredAgent<..>` recursively to compose layers.
  - [ ] Overload runner entry points to accept generics that implement these traits, or convert layered types to a policy object before calling runner.
  - [ ] Add runner-path probe tests that verify Run → Agent → Tool → Base ordering via explicit layers chained through these wrappers.
- If B:
  - [ ] Deprecate and remove `LayeredAgent` and `LayeredRunConfig` types and associated APIs; delete placeholder `run_layers()`/`agent_layers` Vec<()> remnants.
  - [ ] Update docs and examples to only show service-based layering.

### Acceptance criteria

- Clear, working story for agent/run layering: either applied by runner via typed wrappers, or removed and replaced by service-based composition only.
- Tests demonstrate the applied order at runtime through the runner path.

---

## 5) Mixed tools-as-service story: `ToolResult` + adapter vs direct `Service`

### Problem

The original “remove adapter and have tools be Services” target isn’t fully realized. The code removed `BaseToolService` but introduced a `ToolServiceAdapter`. `FunctionTool` still returns `ToolResult` instead of `ToolResponse` and does not implement `Service` directly.

### Evidence

```23:69:src/tool.rs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult { ... }
```

```264:300:src/tool_service.rs
impl<T, E> Service<ToolRequest<E>> for ToolServiceAdapter<T, E> { ... match tool.execute(req.arguments).await { ... } }
```

### Design smell

- Two parallel narratives: adapter vs direct `Service` tools. Increases cognitive load and inconsistency.

### Recommendation (pick one and standardize)

- Option A: Embrace adapter as the standard path. Document it clearly; ensure all examples/tests use `tool.into_service::<E>()`; keep `ToolResult` simple.
- Option B: Complete the migration: implement `Service<ToolRequest<E>>` directly on `FunctionTool`/`TypedFunctionTool`; refactor `Tool` to return a simpler type (e.g., `Result<Value, Error>` plus optional `final` marker if needed) or return `ToolResponse` directly; remove `ToolServiceAdapter`.

### TODOs

- [ ] Decide in DESIGN_UPDATES.md.
- If A:
  - [ ] Audit examples and tests to use only `IntoToolService` path; remove mentions of direct `ServiceTool` unless we keep and document it clearly.
  - [ ] Adjust docs to present adapter as canonical.
- If B:
  - [ ] Add `impl<E> Service<ToolRequest<E>> for FunctionTool` and `TypedFunctionTool`.
  - [ ] Replace `ToolResult` with direct construction of `ToolResponse` (or align `Tool` API to match the direct service response path).
  - [ ] Remove `ToolServiceAdapter` and update examples/tests.

### Acceptance criteria

- One canonical way to compose tools with Tower, consistently taught and used throughout.

---

## 6) Runner header/docs mismatch

### Problem

Runner header mentions “BaseTool” and has an outdated wording on stack ordering terminology.

### Evidence

```1:6:src/runner.rs
//! # Runner (orientation)
//!
//! The `Runner` coordinates an agent run: it interacts with the model, routes
//! tool-calls through the Tower stack (Run → Agent → Tool → BaseTool), and
//! maintains ordering and state across turns and handoffs. Policy layers and
//! tool execution are composed in `service.rs`; this module focuses on orchestration.
```

### Recommendation

- Update header to “Run → Agent → Tool → Base” and remove legacy references.

### TODOs

- [ ] Edit `src/runner.rs` module header accordingly.

### Acceptance criteria

- Header matches canonical order and terminology; grep shows no “BaseTool”.

---

## 7) Docs/examples not aligned with current APIs

### Problems

- README shows `.layer(...)` called directly on a tool instance (no longer supported).
- Docs suggest implicit policy application and claim all legacy APIs removed while deprecated stubs remain.
- TOWER_ARCHITECTURE.md references `TracingLayer` which doesn’t exist and uses typed agent/run layering that is not currently applied at runtime.

### Evidence

```180:189:README.md
let tool = Arc::new(
    FunctionTool::simple("uppercase", "Upper", |s: String| s.to_uppercase())
        .layer(layers::InputSchemaLayer::strict(serde_json::json!({
            "type": "object",
            "properties": {"input": {"type": "string"}},
            "required": ["input"]
        })))
);
```

### Recommendation

- Rewrite README/MIGRATION_GUIDE/TOWER_ARCHITECTURE to:
  - Use service-based composition: `tool.into_service::<E>().layer(...)`.
  - Show explicit `ApprovalLayer` attachment; no claims of default policy injection.
  - If typed wrappers are kept and integrated, show how they apply; otherwise, omit them from quickstart to avoid confusion.
  - Remove or implement `TracingLayer` consistently; if not implementing now, remove from docs.

### TODOs

- [ ] README: update Quick Start/Architecture/Examples snippets to compile:
  - Replace tool `.layer(...)` with service composition.
  - Remove/replace `TracingLayer` references.
- [ ] MIGRATION_GUIDE.md: adjust language from “layer tools via `.layer()`” to “convert to service and layer”, and reflect that some deprecated APIs still exist (with deprecation markers) until removed.
- [ ] TOWER_ARCHITECTURE.md: align to the actual composition story and remove invalid code.
- [ ] Run `cargo test --doc` or compile doc snippets where practical.

### Acceptance criteria

- All code snippets in docs compile or are clearly marked `no_run` but syntactically valid.
- No examples rely on non-existent APIs.

---

## 8) Sequential vs parallel parity and runner-path tests

### Problems

- Before fixes in §2–§3, sequential path behavior diverges (Env ignored, schema auto-applied).
- Ordering tests currently verify manual stacks; add runner-path probe tests.

### Recommendation

- After unifying the runner stack, add tests that execute through runner to verify layer ordering and Env capability handling in both execution modes.

### TODOs

- [ ] Add `tests/runner_layer_ordering.rs` with probe layers attached at run/agent/tool scopes (explicitly) and assert order via the runner (not only manual stacks).
- [ ] Add sequential-mode tests for `Runner::run_with_env` honoring Env capabilities (approval allowed/denied) with and without `ApprovalLayer` attached.
- [ ] Add runner-path schema tests showing no enforcement without layer; strict failure with layer.

### Acceptance criteria

- New tests pass and cover runner behavior parity.

---

## 9) Minor cleanup and consistency

### Issues

- Deprecated remnants: `AgentConfig.agent_layers: Vec<()>`, `RunConfig.run_layers: Vec<()>`, `with_*_layers` no-ops.
- `ManualApproval` blocks stdin in async context; clarify demo-only.
- `Tool::requires_approval()` unused.
- Inline migration breadcrumbs (“Step 6/8” comments) in production files.

### TODOs

- [ ] Remove deprecated `with_agent_layers`/`with_run_layers` APIs and the Vec<()> fields once typed story is decided; or hide behind `#[cfg(test)]` if needed temporarily.
- [ ] Add a doc comment to `env::ManualApproval` stating it is a blocking demo and not suitable for async services; consider `tokio::io` alternative or mark as example-only.
- [ ] Drop `Tool::requires_approval()` or wire it to a documented layer-based policy if we find a strong use case.
- [ ] Move migration breadcrumbs to `DESIGN_UPDATES.md`; clean production code comments.

### Acceptance criteria

- `rg '(with_agent_layers|with_run_layers|agent_layers: Vec|run_layers: Vec)'` yields no hits outside history/docs.
- `ManualApproval` warning present.

---

## Order of operations (recommended)

1. Unify runner stack and remove hidden policies (Sections 1–3). Locks global invariant and eliminates surprises.
2. Decide and finalize the layering surface (Section 4) and tool service story (Section 5). Update docs accordingly.
3. Refresh docs/examples (Section 7) after APIs settle.
4. Add runner-path tests (Section 8) to protect invariants.
5. Cleanup deprecated remnants and comments (Section 9).

---

## Appendix: Quick-reference code anchors

- Runner header terminology

```1:6:src/runner.rs
//! # Runner (orientation)
//!
//! The `Runner` coordinates an agent run: it interacts with the model, routes
//! tool-calls through the Tower stack (Run → Agent → Tool → BaseTool), and
//! maintains ordering and state across turns and handoffs. Policy layers and
//! tool execution are composed in `service.rs`; this module focuses on orchestration.
```

- Sequential path DefaultEnv pin & schema injection

```923:931:src/runner.rs
let req = ToolRequest::<DefaultEnv> {
    env: DefaultEnv,
    run_id: trace_id.clone(),
    agent: agent.name().to_string(),
    tool_call_id: id.clone(),
    tool_name: name.clone(),
    arguments: args.clone(),
};
```

```912:916:src/runner.rs
let with_schema =
    InputSchemaLayer::lenient(schema).layer(tool_service);
BoxService::new(with_schema)
```

- Auto approval injection

```1265:1273:src/runner.rs
if env.capability::<crate::env::ApprovalCapability>().is_some()
    || env.capability::<crate::env::AutoApprove>().is_some()
    || env.capability::<crate::env::ManualApproval>().is_some() {
    let with_approval = ApprovalLayer.layer(with_schema);
    BoxService::new(with_approval)
}
```

- Typed layering wrapper creation (not applied by runner)

```361:363:src/agent.rs
pub fn layer<L>(self, layer: L) -> LayeredAgent<L, Self> {
    LayeredAgent::new(layer, self)
}
```

- Tool adapter path

```264:300:src/tool_service.rs
impl<T, E> Service<ToolRequest<E>> for ToolServiceAdapter<T, E>
where
    T: Tool + Clone + 'static,
    E: Env,
{
    type Response = ToolResponse;
    type Error = tower::BoxError;
    type Future =
        Pin<Box<dyn Future<Output = std::result::Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<std::result::Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: ToolRequest<E>) -> Self::Future { ... }
}
```

---

## Tracking and ownership

- Create a GitHub tracking issue per section; link commits/PRs back to the corresponding section here.
- Update this document as changes land; remove resolved sections and capture final decisions in `DESIGN_UPDATES.md`.
