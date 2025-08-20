//! Parallel tool execution and concurrency controls
//!
//! What this module provides (spec)
//! - A layer that fan-outs tool_calls concurrently and fan-ins results deterministically
//! - Configurable concurrency limits and join/failure policies
//!
//! Exports
//! - Models
//!   - `ConcurrencyLimit(usize)`
//!   - `ToolJoinPolicy::{JoinAll, FailFast, TimeoutPerTool(Duration)}`
//! - Layers
//!   - `ParallelToolsLayer<S, R>` where `S: Service<RawChatRequest, Response=StepOutcome>` and `R: Service<ToolInvocation,...>`
//! - Utils
//!   - Ordering helper to map completed outputs back to requested order
//!
//! Implementation strategy
//! - Wrap the tool router with `tower::buffer::Buffer` to acquire readiness per invocation
//! - On `StepOutcome::Next` with `invoked_tools`, spawn invocations concurrently:
//!   - Use `FuturesUnordered` or `join_all` with a semaphore set by `ConcurrencyLimit`
//!   - Apply `ToolJoinPolicy` (wait all, fail fast, per-invocation timeout)
//! - Serialize outputs as `tool` messages in the same order as original tool_calls
//! - Return a rewritten `StepOutcome::Next` with appended tool messages
//!
//! Composition
//! - `ServiceBuilder::new().layer(ParallelToolsLayer::new(limit, policy)).service(step)`
//! - Combine with resilience layers for per-tool retry/timeout if desired
//!
//! Testing strategy
//! - Fake tools with injected latency and error behavior
//! - Assert that with `JoinAll` all succeed and order is preserved
//! - Assert that with `FailFast` layer aborts on first error and surfaces it
//! - Assert that limit `N` bounds concurrent calls (use atomic counters)
