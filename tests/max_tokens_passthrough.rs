//! Test that max_tokens and max_completion_tokens pass through unmolested when not set
//! This is critical for models like GPT-5 that handle defaults differently

use std::sync::Arc;

use async_openai::{
    config::OpenAIConfig,
    types::{
        ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
        CreateChatCompletionRequest, CreateChatCompletionRequestArgs,
    },
    Client,
};
use tower::{BoxError, Service, ServiceExt};
use tower_llm::{provider::ProviderResponse, Agent};

/// A test provider that captures the exact request it receives
#[derive(Clone)]
struct RequestCapturingProvider {
    captured: Arc<tokio::sync::Mutex<Option<CreateChatCompletionRequest>>>,
}

impl RequestCapturingProvider {
    fn new() -> Self {
        Self {
            captured: Arc::new(tokio::sync::Mutex::new(None)),
        }
    }

    async fn get_captured(&self) -> Option<CreateChatCompletionRequest> {
        self.captured.lock().await.clone()
    }
}

impl Service<CreateChatCompletionRequest> for RequestCapturingProvider {
    type Response = ProviderResponse;
    type Error = BoxError;
    type Future = std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<ProviderResponse, BoxError>> + Send>,
    >;

    fn poll_ready(
        &mut self,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), BoxError>> {
        std::task::Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: CreateChatCompletionRequest) -> Self::Future {
        let captured = self.captured.clone();
        Box::pin(async move {
            // Capture the request
            *captured.lock().await = Some(req.clone());

            // Return a dummy response
            #[allow(deprecated)]
            let assistant = async_openai::types::ChatCompletionResponseMessage {
                content: Some("test response".into()),
                role: async_openai::types::Role::Assistant,
                tool_calls: None,
                refusal: None,
                audio: None,
                function_call: None,
            };

            Ok(ProviderResponse {
                assistant,
                prompt_tokens: 10,
                completion_tokens: 10,
            })
        })
    }
}

#[tokio::test]
async fn test_max_tokens_not_set_by_default() {
    let client = Arc::new(Client::<OpenAIConfig>::new());
    let provider = RequestCapturingProvider::new();
    let captured_provider = provider.clone();

    // Build agent WITHOUT setting max_tokens
    let mut agent = Agent::builder(client)
        .model("gpt-4o")
        .with_provider(provider)
        .policy(tower_llm::CompositePolicy::new(vec![
            tower_llm::policies::max_steps(1),
        ]))
        .build();

    // Create a request without max_tokens
    let request = CreateChatCompletionRequestArgs::default()
        .model("gpt-4o")
        .messages(vec![
            ChatCompletionRequestSystemMessageArgs::default()
                .content("You are a helpful assistant")
                .build()
                .unwrap()
                .into(),
            ChatCompletionRequestUserMessageArgs::default()
                .content("Hello")
                .build()
                .unwrap()
                .into(),
        ])
        .build()
        .unwrap();

    // Call the agent
    let _ = agent.ready().await.unwrap().call(request).await.unwrap();

    // Check captured request - max_tokens should be None
    let captured = captured_provider.get_captured().await.unwrap();
    #[allow(deprecated)]
    {
        assert_eq!(
            captured.max_tokens, None,
            "max_tokens should not be set when not specified"
        );
    }
}

#[tokio::test]
async fn test_max_tokens_preserved_when_set() {
    let client = Arc::new(Client::<OpenAIConfig>::new());
    let provider = RequestCapturingProvider::new();
    let captured_provider = provider.clone();

    // Build agent WITH max_tokens set
    let mut agent = Agent::builder(client)
        .model("gpt-4o")
        .max_tokens(1000)
        .with_provider(provider)
        .policy(tower_llm::CompositePolicy::new(vec![
            tower_llm::policies::max_steps(1),
        ]))
        .build();

    // Create a request
    let request = CreateChatCompletionRequestArgs::default()
        .model("gpt-4o")
        .messages(vec![
            ChatCompletionRequestSystemMessageArgs::default()
                .content("You are a helpful assistant")
                .build()
                .unwrap()
                .into(),
            ChatCompletionRequestUserMessageArgs::default()
                .content("Hello")
                .build()
                .unwrap()
                .into(),
        ])
        .build()
        .unwrap();

    // Call the agent
    let _ = agent.ready().await.unwrap().call(request).await.unwrap();

    // Check captured request - max_tokens should be 1000
    let captured = captured_provider.get_captured().await.unwrap();
    #[allow(deprecated)]
    {
        assert_eq!(
            captured.max_tokens,
            Some(1000),
            "max_tokens should be preserved when set"
        );
    }
}

#[tokio::test]
async fn test_request_max_tokens_preserved() {
    let client = Arc::new(Client::<OpenAIConfig>::new());
    let provider = RequestCapturingProvider::new();
    let captured_provider = provider.clone();

    // Build agent without setting max_tokens
    let mut agent = Agent::builder(client)
        .model("gpt-4o")
        .with_provider(provider)
        .policy(tower_llm::CompositePolicy::new(vec![
            tower_llm::policies::max_steps(1),
        ]))
        .build();

    // Create a request WITH max_tokens set
    let request = CreateChatCompletionRequestArgs::default()
        .model("gpt-4o")
        .messages(vec![
            ChatCompletionRequestSystemMessageArgs::default()
                .content("You are a helpful assistant")
                .build()
                .unwrap()
                .into(),
            ChatCompletionRequestUserMessageArgs::default()
                .content("Hello")
                .build()
                .unwrap()
                .into(),
        ])
        .max_tokens(2000u32)
        .build()
        .unwrap();

    // Call the agent
    let _ = agent.ready().await.unwrap().call(request).await.unwrap();

    // Check captured request - max_tokens from request should be preserved
    let captured = captured_provider.get_captured().await.unwrap();
    #[allow(deprecated)]
    {
        assert_eq!(
            captured.max_tokens,
            Some(2000),
            "max_tokens from request should be preserved"
        );
    }
}

#[tokio::test]
async fn test_max_completion_tokens_not_set_by_default() {
    let client = Arc::new(Client::<OpenAIConfig>::new());
    let provider = RequestCapturingProvider::new();
    let captured_provider = provider.clone();

    // Build agent without setting max_completion_tokens
    let mut agent = Agent::builder(client)
        .model("gpt-4o")
        .with_provider(provider)
        .policy(tower_llm::CompositePolicy::new(vec![
            tower_llm::policies::max_steps(1),
        ]))
        .build();

    // Create a request
    let request = CreateChatCompletionRequestArgs::default()
        .model("gpt-4o")
        .messages(vec![
            ChatCompletionRequestSystemMessageArgs::default()
                .content("You are a helpful assistant")
                .build()
                .unwrap()
                .into(),
            ChatCompletionRequestUserMessageArgs::default()
                .content("Hello")
                .build()
                .unwrap()
                .into(),
        ])
        .build()
        .unwrap();

    // Call the agent
    let _ = agent.ready().await.unwrap().call(request).await.unwrap();

    // Check captured request - max_completion_tokens should be None
    let captured = captured_provider.get_captured().await.unwrap();
    assert_eq!(
        captured.max_completion_tokens, None,
        "max_completion_tokens should not be set when not specified"
    );
}
