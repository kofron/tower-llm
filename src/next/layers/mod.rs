//! Layers and loop/policy abstractions for the experimental `next` module
//!
//! What this module provides (spec)
//! - The building blocks for the agent control loop and its termination policies
//! - A thin, Tower-native adapter (`StepLayer`) to lift a tool router into the step service
//! - A composable loop layer (`AgentLoopLayer`) that drives single-step services until a policy says stop
//!
//! Exports
//! - `StepLayer<S>`: wraps a routed tool service `S` into a one-step model+tools `Service<RawChatRequest, StepOutcome>`
//! - `AgentLoopLayer<S, P>`: drives `S: Service<RawChatRequest, StepOutcome>` according to `P: AgentPolicy`
//! - `StepOutcome`: `{Next{messages, aux, invoked_tools} | Done{messages, aux}}`
//! - Policy DSL: `Policy`, `CompositePolicy`, and built-ins in `policies::{until_no_tool_calls, until_tool_called, max_steps}`
//! - `AgentRun` and `AgentStopReason` as the agent-level response contract
//!
//! Implementation strategy (high-level)
//! - `StepLayer` injects: model configuration, OpenAI tool specs, and a tool service into a self-contained step
//!   - On call: construct model request; call provider; append assistant message; if tool_calls → invoke tools and append tool messages
//! - `AgentLoopLayer` wraps a `Step` and repeats until policy returns a stop reason
//!   - Maintains loop state (turn count), reuses model/tool knobs each iteration, rebuilds the request from `messages`
//! - Policies remain pure; the loop invokes them after every step
//!
//! Composition examples
//! - `ServiceBuilder::new().layer(AgentLoopLayer::new(policy)).service(StepLayer::new(...).layer(tool_router))`
//! - Combine with other cross-cutting layers (budget, tracing, approvals) around the loop or step as needed
//!
//! Testing strategy
//! - Unit-test policies (already provided) for basic stop conditions and composition
//! - Build a fake `Step` service that returns `Next` then `Done`; assert `AgentLoopLayer` termination and step count
//! - Add tests for policy evaluation order and short-circuit
//!
//! Example ideas
//! - "Toys": a `DummyStep` that toggles `Next`/`Done` to show policy behavior
//! - "Tools": a step with a fake tool router; policy `until_tool_called("handoff")` halts on a particular tool
//! - "Budgets": show `AgentLoopLayer` wrapped by a budget layer to stop after N tools or tokens

pub use crate::next::{AgentRun, AgentStopReason, CompositePolicy, LoopState, Policy, StepOutcome};

pub use crate::next::policies;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::next::AgentPolicy;

    #[test]
    fn policy_until_no_tool_calls_stops_on_done() {
        let p = policies::until_no_tool_calls();
        let state = LoopState { steps: 1 };
        let last = StepOutcome::Done {
            messages: vec![],
            aux: Default::default(),
        };
        let reason = p.decide(&state, &last);
        assert!(matches!(reason, Some(AgentStopReason::DoneNoToolCalls)));
    }

    #[test]
    fn policy_max_steps_triggers_when_reached() {
        let p = policies::max_steps(3);
        let state = LoopState { steps: 3 };
        let last = StepOutcome::Next {
            messages: vec![],
            aux: Default::default(),
            invoked_tools: vec![],
        };
        let reason = p.decide(&state, &last);
        assert!(matches!(reason, Some(AgentStopReason::MaxSteps)));
    }

    #[test]
    fn composite_policy_short_circuits_first_match() {
        let comp = CompositePolicy::new(vec![
            policies::max_steps(1),
            policies::until_no_tool_calls(),
        ]);
        let state = LoopState { steps: 1 };
        let last = StepOutcome::Next {
            messages: vec![],
            aux: Default::default(),
            invoked_tools: vec![],
        };
        let reason = comp.decide(&state, &last);
        assert!(matches!(reason, Some(AgentStopReason::MaxSteps)));
    }
}
