//! Budgets: tokens, time, and tool invocations
//!
//! What this module provides (spec)
//! - A layer that terminates the agent loop when budgets are exceeded
//!
//! Exports
//! - Models
//!   - `Budget { max_prompt_tokens, max_completion_tokens, max_time, max_tool_invocations }`
//!   - `BudgetUsage { prompt_tokens, completion_tokens, tools, start_time }`
//! - Layers
//!   - `BudgetLayer<S>` that wraps `AgentLoop` (or `Step` for per-step limits)
//! - Utils
//!   - Accounting helpers applied to `StepAux` and streaming deltas
//!
//! Implementation strategy
//! - Maintain `BudgetUsage` inside the layer (constructor-injected initial state)
//! - On each `StepOutcome`, update usage and compare against thresholds
//! - On breach, return `AgentStopReason::{MaxSteps|Tokens|Time|Tools}` accordingly
//!
//! Composition
//! - `ServiceBuilder::new().layer(BudgetLayer::new(budget)).service(agent_loop)`
//!
//! Testing strategy
//! - Fake `Step` producing controlled `StepAux` values across iterations
//! - Assert stop reasons at exact thresholds and that below-threshold runs continue


