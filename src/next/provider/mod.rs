//! Model provider abstraction
//!
//! What this module provides (spec)
//! - An interface for LLM providers (OpenAI, local mocks) decoupled from the step logic
//!
//! Exports
//! - Models
//!   - `ModelRequest { messages, tools, temperature, max_tokens }`
//!   - `ModelResponse { assistant: AssistantMessage, usage: Usage }`
//! - Services
//!   - `ModelService: Service<ModelRequest, Response=ModelResponse, Error=BoxError>`
//!   - Implementations: `OpenAIProvider`, `MockProvider`
//! - Layers
//!   - `RequestMapLayer<S>`: `Service<RawChatRequest> -> Service<ModelRequest>` adapter
//! - Utils
//!   - Provider presets: `providers::openai(model)`, `providers::mock(script)`
//!
//! Implementation strategy
//! - `Step` is made generic over `M: ModelService`
//! - The OpenAI adapter maps our message/types to async-openai types
//! - Tools are mapped to function specs on demand
//!
//! Composition
//! - For simple usage, keep `Step` with an OpenAI provider injected
//! - For testing, swap in `MockProvider` returning scripted `ModelResponse`s
//!
//! Testing strategy
//! - Unit tests for mapping correctness; round-trip assistant tool_calls through adapter
//! - Integration tests: Step+MockProvider to test tool routing and loop logic independent of network


