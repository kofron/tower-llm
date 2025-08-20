//! Sessions and persistence (memory/replay) for the `next` stack
//!
//! What this module provides (spec)
//! - A clear, Tower-native way to persist and replay conversational state
//! - No dynamic lookups; all dependencies are constructor-injected
//! - Interoperates with the codec (RunItem ↔ raw messages) and recording
//!
//! Exports (public API surface)
//! - Models
//!   - `SessionId` (newtype)
//!   - `LoadSession { id: SessionId }`, `SaveSession { id: SessionId, history: History }`
//!   - `History` = `Vec<RawChatMessage>` (or a smalltype wrapper)
//! - Services
//!   - `SessionLoadStore: Service<LoadSession, Response=History, Error=BoxError>`
//!   - `SessionSaveStore: Service<SaveSession, Response=(), Error=BoxError>`
//!     - Impl examples: `SqliteSessionStore`, `InMemorySessionStore`
//! - Layers
//!   - `MemoryLayer<S>` where `S: Service<RawChatRequest, Response=StepOutcome>`
//!     - On call: loads `History`, merges into request messages, forwards, then appends new messages and saves
//!   - `RecorderLayer<S>` (see recording module)
//!   - `ReplayLayer<S>` (short-circuits with canned outcomes)
//! - Utils (sugar)
//!   - AgentBuilder: `.session(load_store, save_store, session_id)`
//!   - Helpers: `merge_history(history, request_messages)`
//!
//! Implementation strategy
//! - Session stores are plain services with typed requests (no global registries)
//! - `MemoryLayer` holds `Arc<SessionLoadStore>`, `Arc<SessionSaveStore>`, and `SessionId`
//! - On each call:
//!   1) `load_store.call(LoadSession { id })` → `History`
//!   2) Compose `RawChatRequest` by prefixing `History` to current messages
//!   3) Forward to inner step/agent
//!   4) Extract newly produced messages from `StepOutcome`/`AgentRun` and append to `History`
//!   5) `save_store.call(SaveSession { id, history })`
//! - Errors bubble up; store errors are surfaced explicitly
//!
//! Composition examples
//! - `ServiceBuilder::new().layer(MemoryLayer::new(load, save, session_id)).service(step)`
//! - Combine with `RecorderLayer` if you want both persistence and replay traces
//!
//! Testing strategy
//! - Unit tests
//!   - Fake stores using `tower::service_fn` to simulate load/save
//!   - Assert correct merge order (history first) and that saves receive appended messages
//!   - Error propagation when load/save fails
//! - Integration tests
//!   - With a fake model provider and a real `InMemorySessionStore`, verify multi-turn accumulation
//!   - With `ReplayLayer`, verify deterministic reproduction of a captured trace
//!
//! Notes and constraints
//! - Keep the session I/O isolated behind services; do not push DB/file logic into layers
//! - Prefer separate load/save services to keep signatures simple and testable
//! - The replay logic defers to the `recording` and `codec` modules for trace fidelity
