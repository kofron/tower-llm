//! Model provider abstractions (streaming and non-streaming)
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

use std::future::Future;
use std::pin::Pin;

use async_openai::{
    config::OpenAIConfig,
    types::{ChatCompletionResponseMessage, CreateChatCompletionRequest},
    Client,
};
use futures::stream;
use futures::Stream;
use tower::BoxError;

pub use crate::streaming::{StepChunk, StepProvider};

/// A provider that always yields a fixed sequence of chunks.
#[derive(Clone)]
pub struct SequenceProvider {
    items: Vec<StepChunk>,
}
impl SequenceProvider {
    pub fn new(items: Vec<StepChunk>) -> Self {
        Self { items }
    }
}

impl StepProvider for SequenceProvider {
    type Stream = Pin<Box<dyn Stream<Item = StepChunk> + Send>>;
    fn stream_step(
        &self,
        _req: CreateChatCompletionRequest,
    ) -> Pin<Box<dyn Future<Output = Result<Self::Stream, BoxError>> + Send>> {
        let iter = stream::iter(self.items.clone());
        Box::pin(async move { Ok(Box::pin(iter) as Pin<Box<dyn Stream<Item = StepChunk> + Send>>) })
    }
}

// =============================
// Non-streaming provider (ModelService)
// =============================

#[derive(Debug, Clone)]
pub struct ProviderResponse {
    pub assistant: ChatCompletionResponseMessage,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
}

/// Trait alias for non-streaming model services
pub trait ModelService:
    tower::Service<CreateChatCompletionRequest, Response = ProviderResponse, Error = BoxError>
{
}
impl<T> ModelService for T where
    T: tower::Service<CreateChatCompletionRequest, Response = ProviderResponse, Error = BoxError>
{
}

/// OpenAI adapter for the non-streaming provider interface
#[derive(Clone)]
pub struct OpenAIProvider {
    client: std::sync::Arc<Client<OpenAIConfig>>,
}
impl OpenAIProvider {
    pub fn new(client: std::sync::Arc<Client<OpenAIConfig>>) -> Self {
        Self { client }
    }
}

impl tower::Service<CreateChatCompletionRequest> for OpenAIProvider {
    type Response = ProviderResponse;
    type Error = BoxError;
    type Future = std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Self::Response, Self::Error>> + Send>,
    >;

    fn poll_ready(
        &mut self,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        std::task::Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: CreateChatCompletionRequest) -> Self::Future {
        let client = self.client.clone();
        Box::pin(async move {
            let resp = client.chat().create(req).await?;
            let usage = resp.usage.unwrap_or_default();
            let choice = resp
                .choices
                .into_iter()
                .next()
                .ok_or_else(|| "no choices".to_string())?;
            Ok(ProviderResponse {
                assistant: choice.message,
                prompt_tokens: usage.prompt_tokens as usize,
                completion_tokens: usage.completion_tokens as usize,
            })
        })
    }
}

/// Fixed-response provider for tests
#[derive(Clone)]
pub struct FixedProvider {
    output: ProviderResponse,
}
impl FixedProvider {
    pub fn new(output: ProviderResponse) -> Self {
        Self { output }
    }
}

impl tower::Service<CreateChatCompletionRequest> for FixedProvider {
    type Response = ProviderResponse;
    type Error = BoxError;
    type Future = std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Self::Response, Self::Error>> + Send>,
    >;

    fn poll_ready(
        &mut self,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        std::task::Poll::Ready(Ok(()))
    }

    fn call(&mut self, _req: CreateChatCompletionRequest) -> Self::Future {
        let out = self.output.clone();
        Box::pin(async move { Ok(out) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_openai::types::CreateChatCompletionRequestArgs;
    use futures::StreamExt;

    #[tokio::test]
    async fn sequence_provider_streams_items() {
        let p = SequenceProvider::new(vec![
            StepChunk::Token("a".into()),
            StepChunk::Token("b".into()),
        ]);
        let req = CreateChatCompletionRequestArgs::default()
            .model("gpt-4o")
            .messages(vec![])
            .build()
            .unwrap();
        let mut s = p.stream_step(req).await.unwrap();
        let items: Vec<_> = s.by_ref().collect().await;
        assert_eq!(items.len(), 2);
    }
}
