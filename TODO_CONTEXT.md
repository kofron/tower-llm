Contextual Tool Output Handling – Implementation Checklist

- [x] Add `src/context.rs` with: `ContextDecision`, `ContextStep<C>`, `ToolContext<C>` trait, erased handler trait, and typed adapter
- [x] Export `context` module in `src/lib.rs`
- [x] Extend `AgentConfig` with `tool_context: Option<ToolContextSpec>` and add `ToolContextSpec`
- [x] Add builders on `Agent`: `with_context` and `with_context_factory`
- [x] Initialize per-run context instance in `Runner::run`
- [x] Intercept tool outputs in `Runner::run_loop` and apply context decisions (Forward/Rewrite/Final), including error paths
- [x] Ensure `RunItem::ToolOutput` and `Message::tool` reflect rewrites
- [x] Honor `ToolResult.is_final` with possible rewrite override
- [x] Unit tests: Forward (no-op), Rewrite, Final, and error rewrite cases
- [x] Unit tests: handler failure fallback resets factory and forwards
- [x] Unit tests: multiple tool calls in one turn accumulate context
- [x] Unit tests: streaming collect with context rewrite
- [x] Add example demonstrating contextual handling (e.g., `examples/contextual.rs`)
- [x] Update README with brief section on contextual runs

Notes

- Keep API additive and avoid breaking changes.
- Prefer general `tool_name` and `arguments` over concrete tool types.
