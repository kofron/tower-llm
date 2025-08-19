## Design Diary

This document captures the architectural decisions and their evolution during the migration to a Tower-driven execution model. Entries are chronological and append-only.

### 2025-01 — Tool Self-Management and Context Removal

Context

- The codebase had evolved to support multiple ways of achieving the same goals (context handlers, tool layers, direct implementation), creating confusion and violating simplicity.
- Agents could reach into tools to configure them (`with_tool_layers`), violating abstraction boundaries.
- Context handlers created "spooky action at a distance" with global observation and modification of tool outputs.

Decisions

- Tools now manage their own layers via `.layer()` method that returns a `LayeredTool` wrapper.
- Removed `ToolContext` trait and all context handler infrastructure entirely.
- Removed `Agent::with_tool_layers()` - agents no longer configure tool internals.
- Each abstraction level (tool/agent/run) manages only its own layers.
- Tools can have custom names via `.with_name()` method.

Implementation Notes

- `LayeredTool` wraps a tool with its layers and implements the `Tool` trait.
- Runner recognizes `LayeredTool` and applies its layers when building the service stack.
- All context-based APIs removed in favor of Tower layers.
- Clean separation: Tools do work, layers modify behavior.

Benefits

- Single, unified pattern for behavior modification.
- No string-based coupling (removed tool name lookups).
- Each abstraction maintains its boundaries.
- Follows Tower's composition model exactly.

### 2025-08 — Tower-driven execution, scope-agnostic layers, generic Env

Context

- The legacy runner executed tools directly and threaded custom context handlers. We needed a more modular, composable approach with strong DX and clear control of cross-cutting concerns.

Decisions

- Adopt Tower as the execution substrate for tools. Tools are adapted to `Service<ToolRequest<E>> -> ToolResponse` with `Effect` to control flow (`Continue`, `Rewrite`, `Final`, `Handoff`).
- Fix layer order to ensure predictable behavior: Agent → Run → Tool → BaseTool. Outermost `Final` wins. No configurability in v1 for simplicity.
- Make generic policy layers scope-agnostic: a single `Layer<S>` works at run-, agent-, or tool-scope.
- Preserve provider/tool-call reply order even under parallel execution. Concurrency is an internal optimization only.
- Keep a non-generic default runner (`DefaultEnv`) while allowing typed Env for advanced users.
- Handoffs remain a runner-level special case in v1 for simplicity and parity.

Implementation Notes

- `src/service.rs` houses: `Effect`, `ToolRequest<E>`, `ToolResponse`, `BaseToolService`, `RunContextLayer`, `AgentContextLayer`, `InputSchemaLayer`, `TimeoutLayer`, `RetryLayer`, `ApprovalLayer`, and `ModelService`.
- Dynamic layer composition for DX via `ErasedToolLayer` and boxed helpers: `boxed_timeout_secs`, `boxed_retry_times`, `boxed_input_schema_{lenient,strict}`, and `boxed_approval_with`.
- Parallel tool execution per turn with `join_all`, preserving original ordering of outputs.
- Stateful layers (`RunContextLayer`, `AgentContextLayer`) use an internal `Mutex` to manage state transitions safely under parallelism. The lock is held only for short critical sections and dropped before invoking the handler.

DX and APIs

- `RunConfig::with_run_layers`, `Agent::with_agent_layers`, `Agent::with_tool_layers` for dynamic policy attachment.
- `TypedFunctionTool<I, O, F>` to author strongly-typed tools with schema inference.
- `#[tool_args]` and `#[tool_output]` attribute macros to derive serde and schemars for tool I/O types.
- `layers::boxed_approval_with` for predicate-based approval without a typed Env; and `ApprovalLayer` + `HasApproval` for typed Env scenarios.

Behavioral Invariants

- Reply order mirrors model-provided tool-call order even under parallelism.
- Run-scoped context evaluates before agent-scoped; outermost `Final` terminates execution.
- Tool outputs are encoded as JSON strings in tool messages; errors are stringified only at the message/protocol boundary.

Performance

- Benchmarks (Criterion) show material wins from parallel tool execution when the model emits multiple calls in a turn. Use `with_max_concurrency` to tune.
- Locking guidance added to README; keep handler work light, avoid awaits under locks.

Future Work

- Explore a Tower-native handoff layer when handoff semantics stabilize.
- Consider richer schema validation and tracing layers.
