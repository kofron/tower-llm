//! Recording and replay I/O surfaces
//!
//! What this module provides (spec)
//! - Pluggable I/O surfaces for capturing and replaying runs
//! - Integrates with `codec` for lossless message/event fidelity
//!
//! Exports
//! - Services
//!   - `TraceWriter: Service<WriteTrace { id, items }, Response=()>`
//!   - `TraceReader: Service<ReadTrace { id }, Response=Trace>`
//!   - `ReplayService: Service<RawChatRequest, Response=StepOutcome>` (reads from a `Trace`)
//! - Layers
//!   - `RecorderLayer<S>` taps `StepOutcome`/`AgentRun` and writes via `TraceWriter`
//! - Utils
//!   - Trace format (ndjson), `TraceVersion`, integrity checks (hashes)
//!
//! Implementation strategy
//! - `RecorderLayer` calls `codec::messages_to_items`/`items_to_messages` as needed
//! - `ReplayService` reads precomputed outcomes and serves them in sequence (for step) or as a final run (for agent)
//! - Writers/readers are constructor-injected services supporting file/db backends
//!
//! Composition
//! - `ServiceBuilder::new().layer(RecorderLayer::new(writer)).service(step)`
//! - Or: `let agent = ReplayService::new(reader, trace_id);`
//!
//! Testing strategy
//! - Roundtrip with fake provider: live run → record → replay; assert same final messages/events
//! - Corruption tests: invalid trace produces explicit error


