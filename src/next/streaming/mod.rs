//! Streaming step/agent variants
//!
//! What this module provides (spec)
//! - A streaming version of the step and loop that emits tokens/tool events incrementally
//! - Tap APIs for UIs without breaking Tower composition
//!
//! Exports
//! - Models
//!   - `StepChunk::{Token(String), ToolCallStart{ id, name, args }, ToolCallEnd{ id, output }, UsageDelta{...} }`
//!   - `AgentEvent` mirroring the above at agent layer boundaries
//! - Services
//!   - `StepStream: Service<RawChatRequest, Response=impl Stream<Item=StepChunk>>`
//! - Layers
//!   - `AgentLoopStreamLayer<S>` where `S: Service<RawChatRequest, Response=Stream<StepChunk>>`
//!   - `StreamTapLayer<S>` to tee events to an injected sink (observer)
//! - Utils
//!   - `collect_final(stream) -> AgentRun` to remain API-compatible with non-streaming callers
//!
//! Implementation strategy
//! - Provider adapter translates SSE/streaming API into `StepChunk` stream
//! - Loop layer buffers minimal state (e.g., current messages, pending tool_calls), evaluates policy on-the-fly
//! - Ensure back-pressure: do not buffer entire streams; forward as items arrive
//! - Error handling: surface provider/tool errors as terminal `AgentEvent::Error`
//!
//! Composition
//! - `ServiceBuilder::new().layer(StreamTapLayer::new(sink)).layer(AgentLoopStreamLayer::new(policy)).service(step_stream)`
//!
//! Testing strategy
//! - Fake provider that yields a scripted sequence of chunks (tokens → tool_call → outputs → final)
//! - Assert policy-controlled termination (e.g., until tool_called("x"))
//! - Verify tap receives the exact event sequence; no extra buffering or reordering
