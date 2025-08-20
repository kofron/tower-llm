# Migration Plan: Promoting `next` Module to Primary Implementation

## Overview

This document outlines the plan to migrate from the existing agent implementation to the Tower-based `next` module implementation, making it the primary and only implementation.

## Current State Analysis

### Existing Implementation (to be replaced)

- **Core Files:**
  - `src/agent.rs` - Agent configuration and structure
  - `src/runner.rs` - Agent execution loop
  - `src/group.rs` - Multi-agent orchestration
  - `src/tool.rs` - Tool trait and implementations
  - `src/env.rs` - Environment and capability system
  - `src/service.rs` - Service abstractions
  - `src/tool_service.rs` - Tool service implementations

### Next Module (to be promoted)

- **Core Components:**
  - `src/next/mod.rs` - Core types and implementations
  - `src/next/layers/` - Tower layers for agent loop
  - `src/next/services/` - Tower services for tools and steps
  - `src/next/utils/` - DX helper functions
  - **Feature Modules:**
    - `codec/` - Message conversion
    - `sessions/` - Conversation memory
    - `streaming/` - Streaming responses
    - `concurrency/` - Parallel tool execution
    - `budgets/` - Resource limits
    - `resilience/` - Retry/timeout/circuit breaker
    - `approvals/` - Tool approval flows
    - `observability/` - Metrics and tracing
    - `recording/` - Record and replay
    - `provider/` - LLM provider abstraction
    - `groups/` - Multi-agent routing

## Migration Strategy

### Phase 1: Preparation

1. Create backup branch for rollback
2. Document public API changes
3. Create compatibility mapping

### Phase 2: Core Module Promotion

1. **Move next module to root:**

   - Move `src/next/mod.rs` contents to appropriate root modules
   - Promote submodules to root level
   - Update module declarations in `lib.rs`

2. **Rename and reorganize:**
   - `next::Agent` → `Agent`
   - `next::AgentBuilder` → `AgentBuilder`
   - `next::Tool*` → `Tool*`
   - `next::Step` → core execution service
   - `next::AgentLoop` → core loop implementation

### Phase 3: Remove Old Implementation

1. **Delete old files:**

   - `src/agent.rs` (replaced by next implementation)
   - `src/runner.rs` (replaced by AgentLoop)
   - `src/group.rs` (replaced by next/groups)
   - `src/tool.rs` (replaced by next ToolDef)
   - `src/env.rs` (capability system removed in favor of static DI)
   - `src/service.rs` (replaced by Tower services)
   - `src/tool_service.rs` (replaced by next tool services)

2. **Keep/adapt:**
   - `src/items.rs` - Used by both implementations
   - `src/sqlite_session.rs` - Can be adapted to new session trait
   - `src/error.rs` - Keep and extend as needed
   - `src/config.rs` - Adapt for new structure
   - `src/model.rs` - Keep model configurations
   - `src/guardrail.rs` - Integrate with approvals module
   - `src/handoff.rs` - Integrate with groups module

### Phase 4: Update Examples

1. **Convert existing examples to use new API:**

   - Update imports from `next::` to root
   - Replace `Runner::run()` with `AgentBuilder` pattern
   - Update tool definitions to use `tool_typed`
   - Replace environment/capabilities with static DI

2. **Remove `next_` prefix from new examples:**
   - Rename files to replace old examples
   - Update imports and documentation

### Phase 5: Update Tests

1. Run all tests and fix compilation errors
2. Update test imports
3. Verify all functionality is preserved

### Phase 6: Documentation

1. Update README with new architecture
2. Update module documentation
3. Create migration guide for users

## API Mapping

| Old API           | New API                               |
| ----------------- | ------------------------------------- |
| `Agent::simple()` | `Agent::builder().instructions()`     |
| `Runner::run()`   | `agent.run()` or `ServiceExt::call()` |
| `Tool` trait      | `ToolDef` with `tool_typed()` helper  |
| `AgentGroup`      | `GroupRouter` with `GroupBuilder`     |
| `Session` trait   | `SessionStore` trait                  |
| `EnvBuilder`      | Removed - use static DI               |
| Capabilities      | Removed - use Tower layers            |

## Risk Mitigation

1. Create comprehensive test suite before migration
2. Keep backup branch until migration is verified
3. Run parallel testing of old vs new implementation
4. Document all breaking changes

## Success Criteria

- [ ] All tests pass
- [ ] All examples work
- [ ] No references to `next::` module remain
- [ ] Documentation is updated
- [ ] Performance is equal or better
- [ ] API is cleaner and more idiomatic

## Timeline Estimate

- Phase 1: 30 minutes
- Phase 2: 2 hours
- Phase 3: 1 hour
- Phase 4: 2 hours
- Phase 5: 1 hour
- Phase 6: 1 hour
- **Total: ~7-8 hours**
