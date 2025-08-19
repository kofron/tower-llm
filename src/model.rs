//! # Model abstraction (orientation)
//!
//! A thin boundary for LLM interactions. `ModelProvider` adapts different backends
//! behind a consistent interface; the runner depends on this trait. The default
//! implementation is `OpenAIProvider`; tests use `MockProvider`.

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
use std::sync::Arc;

use crate::error::Result;
use crate::items::{Message, ModelResponse, ToolCall};
use crate::tool::Tool;
use crate::usage::Usage;

/// A trait that defines the interface for a Language Model (LLM) provider.
#[async_trait]
pub trait ModelProvider: Send + Sync {
    /// Generates a model response for a given set of messages and tools.
    async fn complete(
        &self,
        messages: Vec<Message>,
        tools: Vec<Arc<dyn Tool>>,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
    ) -> Result<(ModelResponse, Usage)>;

    /// Name of the underlying model.
    fn model_name(&self) -> &str;
}

/// A [`ModelProvider`] implementation for OpenAI's API, using the `async-openai` crate.
pub struct OpenAIProvider {
    #[allow(dead_code)]
    client: Client<OpenAIConfig>,
    model: String,
}

impl OpenAIProvider {
    /// Creates a new `OpenAIProvider` with the specified model.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            model: model.into(),
        }
    }

    /// Creates a new `OpenAIProvider` with a custom `async-openai` client.
    pub fn with_client(client: Client<OpenAIConfig>, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
        }
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
        // Map messages
        let mut req_messages: Vec<ChatCompletionRequestMessage> =
            Vec::with_capacity(messages.len());
        for m in messages {
            match m.role {
                crate::items::Role::System => {
                    let msg = ChatCompletionRequestSystemMessageArgs::default()
                        .content(m.content)
                        .build()
                        .map_err(|e| crate::error::AgentsError::Other(e.to_string()))?;
                    req_messages.push(msg.into());
                }
                crate::items::Role::User => {
                    let msg = ChatCompletionRequestUserMessageArgs::default()
                        .content(m.content)
                        .build()
                        .map_err(|e| crate::error::AgentsError::Other(e.to_string()))?;
                    req_messages.push(msg.into());
                }
                crate::items::Role::Assistant => {
                    // For request messages we only forward assistant content.
                    // Replaying previous tool_calls is not required for correctness here.
                    let mut builder = ChatCompletionRequestAssistantMessageArgs::default();
                    builder.content(m.content);
                    let msg = builder
                        .build()
                        .map_err(|e| crate::error::AgentsError::Other(e.to_string()))?;
                    req_messages.push(msg.into());
                }
                crate::items::Role::Tool => {
                    let msg = ChatCompletionRequestToolMessageArgs::default()
                        .content(m.content)
                        .tool_call_id(m.tool_call_id.unwrap_or_default())
                        .build()
                        .map_err(|e| crate::error::AgentsError::Other(e.to_string()))?;
                    req_messages.push(msg.into());
                }
            }
        }

        // Map tools
        let mut tool_specs: Vec<ChatCompletionTool> = Vec::with_capacity(tools.len());
        for t in tools {
            let func = FunctionObjectArgs::default()
                .name(t.name())
                .description(t.description())
                .parameters(t.parameters_schema())
                .build()
                .map_err(|e| crate::error::AgentsError::Other(e.to_string()))?;
            let tool = ChatCompletionToolArgs::default()
                .r#type(ChatCompletionToolType::Function)
                .function(func)
                .build()
                .map_err(|e| crate::error::AgentsError::Other(e.to_string()))?;
            tool_specs.push(tool);
        }

        let mut req_builder = CreateChatCompletionRequestArgs::default();
        req_builder.model(&self.model);
        req_builder.messages(req_messages);
        if !tool_specs.is_empty() {
            req_builder.tools(tool_specs);
        }
        if let Some(temp) = temperature {
            req_builder.temperature(temp);
        }
        if let Some(mt) = max_tokens {
            req_builder.max_tokens(mt);
        }
        let req = req_builder
            .build()
            .map_err(|e| crate::error::AgentsError::Other(e.to_string()))?;

        let resp = self.client.chat().create(req).await?;

        // Map response
        let choice = resp
            .choices
            .get(0)
            .ok_or_else(|| crate::error::AgentsError::Other("no choices".into()))?;
        let msg = &choice.message;
        let usage = resp.usage.as_ref().cloned().unwrap_or_default();
        let usage_out = Usage::new(
            usage.prompt_tokens as usize,
            usage.completion_tokens as usize,
        );

        // If tool_calls are present, return ModelResponse with tool calls. Otherwise content.
        if let Some(calls) = &msg.tool_calls {
            let mapped = calls
                .iter()
                .map(|c| ToolCall {
                    id: c.id.clone(),
                    name: c.function.name.clone(),
                    arguments: serde_json::from_str(&c.function.arguments)
                        .unwrap_or_else(|_| serde_json::json!({})),
                })
                .collect::<Vec<_>>();
            Ok((ModelResponse::new_tool_calls(mapped), usage_out))
        } else {
            let content = msg.content.clone().unwrap_or_default();
            Ok((ModelResponse::new_message(content), usage_out))
        }
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}

/// A simple mock model provider for tests and examples.
pub struct MockProvider {
    model: String,
    responses: std::sync::Mutex<std::collections::VecDeque<ModelResponse>>,
}

impl MockProvider {
    pub fn new(model: &str) -> Self {
        Self {
            model: model.to_string(),
            responses: std::sync::Mutex::new(std::collections::VecDeque::new()),
        }
    }

    pub fn with_message(mut self, content: &str) -> Self {
        self.responses
            .get_mut()
            .unwrap()
            .push_back(ModelResponse::new_message(content));
        self
    }

    pub fn with_tool_call(mut self, name: &str, arguments: serde_json::Value) -> Self {
        let call = ToolCall {
            id: uuid::Uuid::new_v4().to_string(),
            name: name.to_string(),
            arguments,
        };
        self.responses
            .get_mut()
            .unwrap()
            .push_back(ModelResponse::new_tool_calls(vec![call]));
        self
    }

    pub fn with_response(mut self, resp: ModelResponse) -> Self {
        self.responses.get_mut().unwrap().push_back(resp);
        self
    }
}

#[async_trait]
impl ModelProvider for MockProvider {
    async fn complete(
        &self,
        _messages: Vec<Message>,
        _tools: Vec<Arc<dyn Tool>>,
        _temperature: Option<f32>,
        _max_tokens: Option<u32>,
    ) -> Result<(ModelResponse, Usage)> {
        let mut guard = self.responses.lock().unwrap();
        if let Some(resp) = guard.pop_front() {
            Ok((resp, Usage::new(0, 0)))
        } else {
            Ok((ModelResponse::new_message("ok"), Usage::new(0, 0)))
        }
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}
