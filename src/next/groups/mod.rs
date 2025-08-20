//! Multi-agent orchestration and handoffs
//!
//! What this module provides (spec)
//! - A Tower-native router between multiple agent services with explicit handoff events
//!
//! Exports
//! - Models
//!   - `AgentName` newtype
//!   - `PickRequest { messages, last_stop: AgentStopReason }`
//! - Services
//!   - `GroupRouter: Service<RawChatRequest, Response=AgentRun>`
//!   - `AgentPicker: Service<PickRequest, Response=AgentName>`
//! - Layers
//!   - `HandoffLayer` that annotates runs with AgentStart/AgentEnd/Handoff events
//! - Utils
//!   - `GroupBuilder` to assemble named `AgentSvc`s and a picker strategy
//!
//! Implementation strategy
//! - Use `tower::steer` or a small name→index map, routing to boxed `AgentSvc`s
//! - `AgentPicker` decides next agent based on the current transcript and stop reason
//! - `HandoffLayer` wraps the router to emit handoff events into the run
//!
//! Composition
//! - `GroupBuilder::new().agent("triage", a).agent("specialist", b).picker(p).build()`
//! - Can be wrapped by resilience/observability layers as needed
//!
//! Testing strategy
//! - Build two fake agents that return deterministic responses
//! - A picker that selects based on a message predicate
//! - Assert the handoff events sequence and final run aggregation


