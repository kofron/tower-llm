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

use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::core::{AgentStopReason, LoopState, PolicyFn, StepOutcome};

/// Budget thresholds for an agent run.
#[derive(Debug, Clone, Copy, Default)]
pub struct Budget {
    pub max_prompt_tokens: Option<usize>,
    pub max_completion_tokens: Option<usize>,
    pub max_tool_invocations: Option<usize>,
    pub max_time: Option<Duration>,
}

/// Running usage counters during an agent run.
#[derive(Debug, Clone)]
pub struct BudgetUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub tools: usize,
    pub start_time: Instant,
}

impl Default for BudgetUsage {
    fn default() -> Self {
        Self {
            prompt_tokens: 0,
            completion_tokens: 0,
            tools: 0,
            start_time: Instant::now(),
        }
    }
}

/// Create a policy that enforces the provided budget across steps.
///
/// Internally maintains counters using interior mutability.
pub fn budget_policy(b: Budget) -> PolicyFn {
    let usage = Arc::new(std::sync::Mutex::new(BudgetUsage::default()));
    let usage_cl = usage.clone();
    PolicyFn(Arc::new(move |_state: &LoopState, last: &StepOutcome| {
        let usage = usage_cl.clone();
        let mut u = usage.lock().unwrap();
        match last {
            StepOutcome::Next { aux, .. } | StepOutcome::Done { aux, .. } => {
                u.prompt_tokens += aux.prompt_tokens;
                u.completion_tokens += aux.completion_tokens;
                u.tools += aux.tool_invocations;
            }
        }
        // Check time budget first
        if let Some(max) = b.max_time {
            if u.start_time.elapsed() >= max {
                return Some(AgentStopReason::TimeBudgetExceeded);
            }
        }
        if let Some(max) = b.max_prompt_tokens {
            if u.prompt_tokens >= max {
                return Some(AgentStopReason::TokensBudgetExceeded);
            }
        }
        if let Some(max) = b.max_completion_tokens {
            if u.completion_tokens >= max {
                return Some(AgentStopReason::TokensBudgetExceeded);
            }
        }
        if let Some(max) = b.max_tool_invocations {
            if u.tools >= max {
                return Some(AgentStopReason::ToolBudgetExceeded);
            }
        }
        None
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{AgentPolicy, CompositePolicy, StepAux};

    fn fake_next_step(prompt: usize, completion: usize, tools: usize) -> StepOutcome {
        StepOutcome::Next {
            messages: vec![],
            aux: StepAux {
                prompt_tokens: prompt,
                completion_tokens: completion,
                tool_invocations: tools,
            },
            invoked_tools: vec![],
        }
    }

    #[tokio::test]
    async fn stops_on_token_budget() {
        let budget = Budget {
            max_prompt_tokens: Some(10),
            ..Default::default()
        };
        let policy = budget_policy(budget);
        let comp = CompositePolicy::new(vec![policy]);
        let state = LoopState { steps: 1 };
        // below threshold
        assert!(comp.decide(&state, &fake_next_step(5, 0, 0)).is_none());
        // at threshold triggers
        assert!(matches!(
            comp.decide(&state, &fake_next_step(5, 0, 0)),
            Some(AgentStopReason::TokensBudgetExceeded)
        ));
    }

    #[tokio::test]
    async fn stops_on_tool_budget() {
        let budget = Budget {
            max_tool_invocations: Some(2),
            ..Default::default()
        };
        let policy = budget_policy(budget);
        let comp = CompositePolicy::new(vec![policy]);
        let state = LoopState { steps: 1 };
        assert!(comp.decide(&state, &fake_next_step(0, 0, 1)).is_none());
        assert!(matches!(
            comp.decide(&state, &fake_next_step(0, 0, 1)),
            Some(AgentStopReason::ToolBudgetExceeded)
        ));
    }
}
