//! Bijection codec between RunItem and raw OpenAI chat messages
//!
//! What this module provides (spec)
//! - Deterministic, lossless conversions used by recording/replay and persistence layers
//! - Pure utility functions decoupled from services
//!
//! Exports
//! - Models (reuse from `items.rs` if desired)
//!   - `RunItem::{Message, ToolCall, ToolOutput, Handoff}`
//! - Utils (pure)
//!   - `messages_to_items(messages: &[RawChatMessage]) -> Vec<RunItem>`
//!   - `items_to_messages(items: &[RunItem]) -> Vec<RawChatMessage>`
//!   - `CodecError` for invalid sequences (e.g., tool output without prior tool_call)
//!
//! Implementation strategy
//! - One-pass state machine for `messages_to_items`:
//!   - Accumulate assistant `tool_calls` to attach to the preceding assistant message
//!   - Immediately emit `ToolOutput` for `tool` role messages with resolved `tool_call_id`
//! - The inverse reconstructs messages, ensuring assistant tool_calls coalesce correctly
//! - Avoid allocations by pre-sizing vectors and reusing buffers
//!
//! Composition
//! - Recorder and session layers call these functions at the edge
//! - Does not depend on Tower; purely functional
//!
//! Testing strategy
//! - Property tests (e.g., with `proptest`) asserting round-trip identity
//! - Golden-case unit tests for:
//!   - Multiple tool_calls in a single assistant message
//!   - Interleaved ToolCall/ToolOutput
//!   - Empty content and error outputs
//! - Fuzz invalid sequences to ensure `CodecError` surfaces clearly
