//! Model abstraction for LLM interactions
//!
//! Wraps the async-openai crate to provide a clean interface for agent-LLM communication.

use async_openai::{
    config::OpenAIConfig,
    types::{
        ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage,
        ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestToolMessageArgs,
        ChatCompletionRequestUserMessageArgs, ChatCompletionTool, ChatCompletionToolArgs,
        ChatCompletionToolType, CreateChatCompletionRequestArgs, FunctionObjectArgs,
    },
    Client,
};
use async_trait::async_trait;
use serde_json::Value;
use std::sync::Arc;

use crate::error::{AgentsError, Result};
use crate::items::{Message, ModelResponse, Role, ToolCall};
use crate::tool::Tool;
use crate::usage::Usage;

/// Trait for model providers
#[async_trait]
pub trait ModelProvider: Send + Sync {
    /// Generate a completion
    async fn complete(
        &self,
        messages: Vec<Message>,
        tools: Vec<Arc<dyn Tool>>,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
    ) -> Result<(ModelResponse, Usage)>;

    /// Get the model name
    fn model_name(&self) -> &str;
}

/// OpenAI model provider using async-openai
pub struct OpenAIProvider {
    client: Client<OpenAIConfig>,
    model: String,
}

impl OpenAIProvider {
    /// Create a new OpenAI provider
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            model: model.into(),
        }
    }

    /// Create with a custom client
    pub fn with_client(client: Client<OpenAIConfig>, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
        }
    }

    /// Convert our Message to OpenAI's format
    fn convert_message(&self, msg: &Message) -> ChatCompletionRequestMessage {
        match msg.role {
            Role::System => ChatCompletionRequestSystemMessageArgs::default()
                .content(msg.content.clone())
                .build()
                .unwrap()
                .into(),
            Role::User => ChatCompletionRequestUserMessageArgs::default()
                .content(msg.content.clone())
                .build()
                .unwrap()
                .into(),
            Role::Assistant => {
                let mut builder = ChatCompletionRequestAssistantMessageArgs::default();
                builder.content(msg.content.clone());
                
                // Add tool calls if present
                if let Some(tool_calls) = &msg.tool_calls {
                    let openai_tool_calls: Vec<_> = tool_calls.iter().map(|tc| {
                        async_openai::types::ChatCompletionMessageToolCall {
                            id: tc.id.clone(),
                            r#type: async_openai::types::ChatCompletionToolType::Function,
                            function: async_openai::types::FunctionCall {
                                name: tc.name.clone(),
                                arguments: tc.arguments.to_string(),
                            },
                        }
                    }).collect();
                    builder.tool_calls(openai_tool_calls);
                }
                
                builder.build().unwrap().into()
            }
            Role::Tool => ChatCompletionRequestToolMessageArgs::default()
                .content(msg.content.clone())
                .tool_call_id(msg.tool_call_id.clone().unwrap_or_default())
                .build()
                .unwrap()
                .into(),
        }
    }

    /// Convert tools to OpenAI format
    fn convert_tools(&self, tools: &[Arc<dyn Tool>]) -> Vec<ChatCompletionTool> {
        tools
            .iter()
            .map(|tool| {
                ChatCompletionToolArgs::default()
                    .r#type(ChatCompletionToolType::Function)
                    .function(
                        FunctionObjectArgs::default()
                            .name(tool.name())
                            .description(tool.description())
                            .parameters(tool.parameters_schema())
                            .build()
                            .unwrap(),
                    )
                    .build()
                    .unwrap()
            })
            .collect()
    }
}

#[async_trait]
impl ModelProvider for OpenAIProvider {
    async fn complete(
        &self,
        messages: Vec<Message>,
        tools: Vec<Arc<dyn Tool>>,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
    ) -> Result<(ModelResponse, Usage)> {
        let openai_messages: Vec<ChatCompletionRequestMessage> = messages
            .iter()
            .map(|msg| self.convert_message(msg))
            .collect();

        let mut request = CreateChatCompletionRequestArgs::default();
        request.model(&self.model).messages(openai_messages);

        if !tools.is_empty() {
            request.tools(self.convert_tools(&tools));
        }

        if let Some(temp) = temperature {
            request.temperature(temp);
        }

        if let Some(max) = max_tokens {
            request.max_tokens(max);
        }

        let response = self.client.chat().create(request.build()?).await?;

        // Extract the first choice
        let choice = response
            .choices
            .first()
            .ok_or_else(|| AgentsError::ModelBehaviorError {
                message: "No choices in response".to_string(),
            })?;

        // Convert tool calls if any
        let tool_calls = if let Some(tool_calls) = &choice.message.tool_calls {
            tool_calls
                .iter()
                .map(|tc| ToolCall {
                    id: tc.id.clone(),
                    name: tc.function.name.clone(),
                    arguments: serde_json::from_str(&tc.function.arguments).unwrap_or(Value::Null),
                })
                .collect()
        } else {
            vec![]
        };

        let model_response = ModelResponse {
            id: response.id.clone(),
            content: choice.message.content.clone(),
            tool_calls,
            finish_reason: choice.finish_reason.as_ref().map(|r| format!("{:?}", r)),
            created_at: chrono::Utc::now(),
        };

        let usage = if let Some(usage) = response.usage {
            Usage::new(
                usage.prompt_tokens as usize,
                usage.completion_tokens as usize,
            )
        } else {
            Usage::empty()
        };

        Ok((model_response, usage))
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}

/// Mock model provider for testing
#[cfg(test)]
pub struct MockProvider {
    model: String,
    responses: std::sync::Mutex<Vec<ModelResponse>>,
}

#[cfg(test)]
impl MockProvider {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            responses: std::sync::Mutex::new(vec![]),
        }
    }

    pub fn with_response(self, response: ModelResponse) -> Self {
        self.responses.lock().unwrap().push(response);
        self
    }

    pub fn with_message(self, content: impl Into<String>) -> Self {
        self.with_response(ModelResponse::new_message(content))
    }

    pub fn with_tool_call(self, tool_name: impl Into<String>, args: Value) -> Self {
        let tool_call = ToolCall {
            id: uuid::Uuid::new_v4().to_string(),
            name: tool_name.into(),
            arguments: args,
        };
        self.with_response(ModelResponse::new_tool_calls(vec![tool_call]))
    }
}

#[cfg(test)]
#[async_trait]
impl ModelProvider for MockProvider {
    async fn complete(
        &self,
        _messages: Vec<Message>,
        _tools: Vec<Arc<dyn Tool>>,
        _temperature: Option<f32>,
        _max_tokens: Option<u32>,
    ) -> Result<(ModelResponse, Usage)> {
        let mut responses = self.responses.lock().unwrap();
        if responses.is_empty() {
            return Ok((
                ModelResponse::new_message("Default response"),
                Usage::new(10, 5),
            ));
        }

        let response = responses.remove(0);
        Ok((response, Usage::new(10, 5)))
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::FunctionTool;

    #[test]
    fn test_openai_provider_creation() {
        let provider = OpenAIProvider::new("gpt-4");
        assert_eq!(provider.model_name(), "gpt-4");
    }

    #[test]
    fn test_message_conversion() {
        let provider = OpenAIProvider::new("gpt-4");

        let system_msg = Message::system("You are helpful");
        let _converted = provider.convert_message(&system_msg);
        // Just verify it doesn't panic

        let user_msg = Message::user("Hello");
        let _ = provider.convert_message(&user_msg);

        let assistant_msg = Message::assistant("Hi there");
        let _ = provider.convert_message(&assistant_msg);

        let tool_msg = Message::tool("Result", "call_123");
        let _ = provider.convert_message(&tool_msg);
    }

    #[test]
    fn test_tool_conversion() {
        let provider = OpenAIProvider::new("gpt-4");

        let tool = Arc::new(FunctionTool::simple(
            "test_tool",
            "Test description",
            |s: String| s,
        ));

        let tools: Vec<Arc<dyn Tool>> = vec![tool];
        let converted = provider.convert_tools(&tools);

        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0].function.name, "test_tool");
        assert_eq!(
            converted[0].function.description.as_ref().unwrap(),
            "Test description"
        );
    }

    #[tokio::test]
    async fn test_mock_provider() {
        let provider = MockProvider::new("mock-model").with_message("Test response");

        assert_eq!(provider.model_name(), "mock-model");

        let (response, usage) = provider.complete(vec![], vec![], None, None).await.unwrap();

        assert_eq!(response.content, Some("Test response".to_string()));
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 5);
    }

    #[tokio::test]
    async fn test_mock_provider_tool_call() {
        let provider = MockProvider::new("mock-model").with_tool_call(
            "calculator",
            serde_json::json!({"operation": "add", "a": 1, "b": 2}),
        );

        let (response, _) = provider.complete(vec![], vec![], None, None).await.unwrap();

        assert_eq!(response.tool_calls.len(), 1);
        assert_eq!(response.tool_calls[0].name, "calculator");
    }

    #[tokio::test]
    async fn test_mock_provider_multiple_responses() {
        let provider = MockProvider::new("mock-model")
            .with_message("First")
            .with_message("Second");

        let (response1, _) = provider.complete(vec![], vec![], None, None).await.unwrap();
        assert_eq!(response1.content, Some("First".to_string()));

        let (response2, _) = provider.complete(vec![], vec![], None, None).await.unwrap();
        assert_eq!(response2.content, Some("Second".to_string()));

        // Default response when queue is empty
        let (response3, _) = provider.complete(vec![], vec![], None, None).await.unwrap();
        assert_eq!(response3.content, Some("Default response".to_string()));
    }
}
