# Tower Static-DI + Raw OpenAI I/O Epic

Goal: Implement a new, parallel module that realizes the updated design guidelines without modifying existing runtime code. All work happens in a separate namespace/crate so we can iterate, test, and benchmark independently.

Note: This is a fresh implementation that may borrow code patterns from the current repo but must not refactor or break existing code paths.

## Scaffolding

- [ ] Decide packaging: standalone crate (preferred) vs. in-repo experimental module
  - [ ] Option A (recommended): new crate `agents_tower_next` under `experiments/agents_tower_next/` with its own `Cargo.toml`
  - [ ] Option B: new module under `src_next/` guarded by `#[cfg(feature = "next")]`
  - [ ] Do not alter existing APIs; keep new work behind separate entry points
- [ ] Add README with goals, architecture diagram, and examples

## Core I/O Types and Conversions (Bijection)

- [ ] Define `RunItem` (next) as a lossless projection of raw OpenAI chat messages
  - [ ] Variants: `Message { role, content, tool_calls? }`, `ToolOutput { tool_call_id, content }`, `Handoff { from, to, reason? }`
  - [ ] Define `ToolCall { id, name, arguments: serde_json::Value }`
- [ ] Implement conversions
  - [ ] `fn run_items_to_raw_messages(items: &[RunItem]) -> Vec<RawChatMessage>`
  - [ ] `fn raw_messages_to_run_items(messages: &[RawChatMessage]) -> Vec<RunItem>`
  - [ ] Ensure stable ordering of tool_calls and their corresponding tool outputs
- [ ] Property tests
  - [ ] Roundtrip: `raw -> items -> raw == raw`
  - [ ] Determinism across serialization boundaries
  - [ ] Edge cases: empty content, multiple tool_calls, interleaved tool outputs

## Static DI Primitives

- [ ] Define constructor-injected dependencies (`Arc` fields)
  - [ ] OpenAI client + model configuration
  - [ ] Typed `ToolRouter` (no string lookups)
  - [ ] Optional tracing/metrics handles
- [ ] No dynamic env; no request extension bags

## One-Step Agent Service

- [ ] `AgentStepService: Service<RawChatRequest, Response = StepOutcome>`
  - [ ] Perform one model call
  - [ ] If assistant returns `tool_calls`, dispatch tools via `ToolRouter`
  - [ ] Append tool outputs as raw `tool` messages
  - [ ] Return `StepOutcome::Next` with updated messages; otherwise `Done`
- [ ] `StepAux` structure for usage/tool accounting
- [ ] Unit tests with a fake `ModelProvider` (no network)

## Agent Loop Layer

- [ ] `AgentLoopLayer: Layer<S>`; `AgentLoop<S>: Service<RawChatRequest, Response = AgentRun>`
  - [ ] Drive iteration until `Done` or `max_turns`/budget reached
  - [ ] Convert between `RunItem` and raw messages to build a replayable event stream
- [ ] Cancellation/time budgets (config knobs)
- [ ] Tests: loop termination, budget enforcement, step error propagation

## Tools and Router

- [ ] Define `Tool` trait for the next module (pure, minimal)
  - [ ] `fn name(&self) -> &str`, `fn description(&self) -> &str`, `fn parameters_schema(&self) -> Value`, `async fn call(args: Value) -> Result<Value>`
- [ ] `ToolRouter`
  - [ ] Typed registration API (no strings at call sites; string names only at boundary for OpenAI function spec)
  - [ ] Dispatch by name coming from `tool_calls`
  - [ ] Tests: unknown tool, schema mismatch, propagation of tool errors

## Examples

- [ ] Minimal example: single tool + 2-turn loop
- [ ] Multi-tool example with interleaved tool calls and outputs
- [ ] Replay example: capture `RunItem` stream to disk, reconstruct raw messages, and re-run deterministically

## Observability

- [ ] Optional tracing layer(s) that wrap step/loop services
- [ ] Usage accounting aggregation into `AgentRun`
- [ ] Structured logs for tool dispatch and model responses

## Docs and Design Hygiene

- [ ] Update `agent-helpers/epics/tower/design-guidelines.md` references with the new shapes (done in this PR)
- [ ] Add architecture notes to the new module’s README
- [ ] Document the bijection contract and known limitations

## Integration (Non-breaking)

- [ ] Bridge adapters (optional): map existing `Message`/`RunItem` types to the new ones for replay/testing only
- [ ] Keep all new code opt-in; do not modify existing examples/tests

## Stretch Goals

- [ ] Budget-aware loop (tokens/time/tool invocations)
- [ ] Pluggable retry policies for transient model/tool errors
- [ ] Offline golden tests using recorded `RunItem` traces
