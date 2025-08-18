# Run-scoped Context and Agent Groups – Plan & Checklist

## Objective

Enable a run-scoped context (owned by the runner) that spans handoffs, and introduce an `AgentGroup` that acts like a single agent. Support:

- No shared state (default)
- Shared run-scoped context across multiple agents
- Per-agent context only (status quo)
- Mixed: run-scoped and per-agent contexts applied together

## API Sketch (non-binding)

- Run-scoped context (attached at run time):
  - `RunConfig::with_run_context<C, H, F>(factory: F, handler: H) -> Self`
  - `Runner::run_with_run_context<C, A: Into<Agent>>(agent_or_group: A, input, config) -> Result<RunResultWithContext<C>>`
- Agent group (acts like one agent):
  - `AgentGroup::builder(name)` → `.root(agent)` → `.handoff(from_name, to_agent)` → `.build()`

## Design Notes

- Run-scoped context is created once per run via factory.
- After every tool execution (success or error), the runner applies the run-scoped handler if configured.
- If a per-agent context handler is also configured, apply both in a deterministic order (run-scoped first, then per-agent), so the agent sees the rewritten output from the run context.
- Handoffs are transparent: the run-scoped context persists; per-agent context resets when changing agents (as it does today).
- Typed retrieval: expose final `C` only at run completion via `RunResultWithContext<C>`.
- `AgentGroup` is a thin composition that builds a top-level agent with declared handoffs, retaining the same interface the runner expects.

## Implementation Tasks

- [x] Add run-scoped context fields to `RunConfig` and re-export types as needed
- [x] Extend `Runner` to initialize and carry run-scoped context state
- [x] In tool handling, apply handlers in order: run-scoped → per-agent
- [x] Add typed entrypoint `Runner::run_with_run_context<C, A: Into<Agent>>`
- [ ] Implement `AgentGroup` (builder + conversion into a single top-level `Agent` with handoffs)
- [x] Ensure streaming path also respects run-scoped context
- [x] Add runtime errors for type mismatches or missing state
- [x] Update docs: crate docs + README (new “Run-scoped Context” and “Agent Groups” sections)

## Tests (robust coverage)

- [x] Run-scoped context, single agent: Forward/Rewrite/Final
- [x] Run-scoped context across handoffs: accumulation verified across 2–3 agents
- [x] Per-agent-only (existing behavior sanity) unaffected by run-scoped off
- [x] Mixed: run-scoped + per-agent both configured; verify order (run-scoped first), and combined effect
- [x] Typed API: final `C` returned, inaccessible otherwise
- [x] Error cases: handler failure falls back safely; type mismatch yields clear error
- [ ] Streaming: run-scoped rewrite applied (note: `collect()` returns `RunResult`, not context)
- [ ] `AgentGroup` behaves like `Agent`: no run context (no sharing), then with run context (sharing)

## Examples

- [ ] `examples/group_no_shared.rs`: group of agents, no shared context
- [ ] `examples/group_shared.rs`: group sharing a simple counter/accumulator via run-scoped context
- [ ] `examples/mixed_shared_individual.rs`: run-scoped shared + per-agent specific handler
- [x] Add `//! Expected:` blocks for the example runner (all current examples covered)

## CI & Tooling

- [x] Ensure example runner picks up new examples automatically
- [x] All clippy/check/test targets remain green

## Migration / Backwards Compatibility

- No breaking changes to existing per-agent context APIs
- New features are opt-in via `RunConfig::with_run_context` and `AgentGroup`
