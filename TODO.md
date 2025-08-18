# Documentation and Examples Update Plan

This plan tracks the documentation improvements and related example updates.

## Goals

- Elevate `AgentGroup` to a core concept and reflect it across docs
- Clarify contextual runs by unifying per-agent and run-scoped context docs
- Provide a clearer narrative and progression through examples
- Expand doctest coverage for key public APIs
- Consolidate/retire ad hoc TODO files once integrated

## Tasks

### Core Concepts and Crate Docs

- [ ] `README.md`: add `AgentGroup` to Features and Core Concepts
- [ ] `README.md`: add link refs for `AgentGroup` and `AgentGroupBuilder`
- [ ] `src/lib.rs`: mention `AgentGroup` in Core Concepts list
- [ ] `src/group.rs`: add module-level docs with minimal examples (builder + handoff)

### Contextual Runs (unify and clarify)

- [ ] `README.md`: merge "Contextual Runs" and "Run-scoped Context" into one section with two subsections: per-agent and run-scoped
- [ ] Ensure examples in the section compile and mirror `examples/contextual.rs`
- [ ] Clearly state ordering and precedence: run-scoped handler runs before per-agent and can finalize

### Examples Narrative and Organization

- [ ] `README.md`: restructure Examples into a progression:
  - Getting Started: `hello_world.rs`
  - Adding Capabilities: `tool_example.rs`
  - Composing Agents: `group_no_shared.rs`
  - Managing State: `group_shared.rs`
  - Advanced Concepts: `persistent_session.rs`, `session_with_guardrails.rs`, `parallel_tools.rs`, `calculator.rs`, `db_migrator.rs`, `rpn_calculator.rs`
  - Case Study: `multi_agent_research.rs`
- [ ] Consider renaming examples for clarity:
  - `group_no_shared.rs` → `group_basic.rs`
  - `group_shared.rs` → `group_with_context.rs`
  - If renamed, update `README.md`, `TODO_RUN_CONTEXT_AND_GROUPS.md`, and any references

### Doctests Expansion

- [ ] `src/runner.rs`: add doctests for `Runner::run` and `Runner::run_with_run_context`
- [ ] `src/agent.rs`: add doctests for `Agent::simple`, `Agent::with_tool`, `Agent::with_handoff`
- [ ] `src/group.rs`: add doctests for `AgentGroupBuilder::new`, `.root`, `.handoff`, `.build`
- [ ] `src/tool.rs`: add doctests for `FunctionTool::simple`, `FunctionTool::new`

### Integrate and Retire TODO Files

- [ ] Review `TODO_CONTEXT.md` and integrate any remaining notes into crate docs
- [ ] Review `TODO_RUN_CONTEXT_AND_GROUPS.md` and mark completed items; move outstanding items into this `TODO.md`
- [ ] Delete `TODO_CONTEXT.md` and `TODO_RUN_CONTEXT_AND_GROUPS.md` after integration to reduce duplication

### Polish and CI

- [ ] Run `cargo fmt`, `cargo clippy`, and `cargo test`
- [ ] Verify examples via `src/bin/run_examples.rs`
- [ ] Bump `README.md` links to docs.rs if versions changed

## Stretch (optional)

- [ ] Refactor `examples/multi_agent_research.rs` to use `AgentGroupBuilder` instead of manual handoffs, and add brief commentary at top of file
- [ ] Add a short "Design Diary" section in `README.md` highlighting functional core/imperative shell decisions and how groups/contexts embody that
