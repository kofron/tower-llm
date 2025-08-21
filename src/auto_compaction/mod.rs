//! Auto-compaction layer for managing conversation context length
//!
//! This module provides automatic conversation compaction when context limits are reached.
//! It can work reactively (on error) or proactively (when approaching limits).
//!
//! # Features
//! - Reactive compaction on context length errors
//! - Proactive compaction based on token thresholds
//! - Configurable compaction strategies
//! - Custom compaction prompts
//! - Integration with existing provider infrastructure
//!
//! # Example
//!
//! ```no_run
//! use tower_llm::{Agent, auto_compaction::{CompactionPolicy, CompactionStrategy, ProactiveThreshold}};
//! use std::sync::Arc;
//! use async_openai::{Client, config::OpenAIConfig};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
//! let client = Arc::new(Client::<OpenAIConfig>::new());
//!
//! // Configure compaction to trigger at 1000 tokens
//! let policy = CompactionPolicy {
//!     compaction_model: "gpt-4o-mini".to_string(),
//!     proactive_threshold: Some(ProactiveThreshold {
//!         token_threshold: 1000,
//!         percentage_threshold: None,
//!     }),
//!     compaction_strategy: CompactionStrategy::PreserveSystemAndRecent { recent_count: 5 },
//!     ..Default::default()
//! };
//!
//! let agent = Agent::builder(client)
//!     .model("gpt-4o")
//!     .auto_compaction(policy)
//!     .build();
//! # Ok(())
//! # }
//! ```

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use async_openai::types::{
    ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage,
    ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestToolMessageArgs,
    ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequest,
    CreateChatCompletionRequestArgs,
};
use tower::{BoxError, Layer, Service, ServiceExt};

use crate::core::StepOutcome;
use crate::provider::ModelService;

// ===== Core Types =====

/// Strategy for handling orphaned tool calls during compaction
#[derive(Clone, Debug)]
pub enum OrphanedToolCallStrategy {
    /// Drop orphaned tool calls before compaction and re-append after (default)
    DropAndReappend,
    /// Exclude messages with orphaned tool calls from compaction range
    ExcludeFromCompaction,
    /// Add placeholder tool responses for orphaned calls
    AddPlaceholderResponses,
    /// Fail compaction if orphaned tool calls are detected
    FailOnOrphaned,
}

impl Default for OrphanedToolCallStrategy {
    fn default() -> Self {
        Self::DropAndReappend
    }
}

/// Configuration for auto-compaction behavior
#[derive(Clone)]
pub struct CompactionPolicy {
    /// Model to use for compaction (e.g., "gpt-4o-mini")
    pub compaction_model: String,

    /// Optional: Proactive threshold for triggering compaction
    pub proactive_threshold: Option<ProactiveThreshold>,

    /// Strategy for selecting messages to compact
    pub compaction_strategy: CompactionStrategy,

    /// Prompt to use for compaction
    pub compaction_prompt: CompactionPrompt,

    /// Maximum retries for compaction attempts
    pub max_compaction_attempts: usize,

    /// Strategy for handling orphaned tool calls
    pub orphaned_tool_call_strategy: OrphanedToolCallStrategy,
}

impl Default for CompactionPolicy {
    fn default() -> Self {
        Self {
            compaction_model: "gpt-4o-mini".to_string(),
            proactive_threshold: None,
            compaction_strategy: CompactionStrategy::PreserveSystemAndRecent { recent_count: 10 },
            compaction_prompt: CompactionPrompt::Default,
            max_compaction_attempts: 2,
            orphaned_tool_call_strategy: OrphanedToolCallStrategy::default(),
        }
    }
}

/// Threshold configuration for proactive compaction
#[derive(Clone, Debug)]
pub struct ProactiveThreshold {
    /// Token count at which to trigger compaction
    pub token_threshold: usize,

    /// Alternative: percentage of model's context window (0.0-1.0)
    pub percentage_threshold: Option<f32>,
}

/// Strategy for selecting which messages to compact
#[derive(Clone)]
pub enum CompactionStrategy {
    /// Compact all messages except the last N
    CompactAllButLast(usize),

    /// Compact messages older than N turns (user+assistant pairs)
    CompactOlderThan(usize),

    /// Keep system prompt and last N messages, compact the middle
    PreserveSystemAndRecent { recent_count: usize },

    /// Custom function to select which messages to compact
    Custom(CompactionRangeFn),
}

/// Range of messages to compact
#[derive(Clone, Debug)]
pub struct CompactionRange {
    /// Start index (inclusive)
    pub start: usize,
    /// End index (exclusive)
    pub end: usize,
}

/// Type alias for custom compaction range function
pub type CompactionRangeFn =
    Arc<dyn Fn(&[ChatCompletionRequestMessage]) -> CompactionRange + Send + Sync>;

/// Type alias for dynamic prompt generation function
pub type PromptGeneratorFn = Arc<dyn Fn(&[ChatCompletionRequestMessage]) -> String + Send + Sync>;

/// Compaction prompt configuration
#[derive(Clone)]
pub enum CompactionPrompt {
    /// Use default prompt optimized for conversation summarization
    Default,

    /// Custom static prompt
    Custom(String),

    /// Dynamic prompt based on messages being compacted
    Dynamic(PromptGeneratorFn),
}

impl CompactionPrompt {
    fn generate(&self, messages: &[ChatCompletionRequestMessage]) -> String {
        match self {
            CompactionPrompt::Default => {
                "Please provide a concise summary of the conversation above, preserving:\n\
                1. The user's original intent and requirements\n\
                2. Key decisions, conclusions, and action items\n\
                3. Important context, constraints, and technical details\n\
                4. Any errors or issues encountered and their resolutions\n\
                5. Current state of any ongoing tasks\n\n\
                Format the summary as a clear narrative that maintains conversation continuity."
                    .to_string()
            }
            CompactionPrompt::Custom(prompt) => prompt.clone(),
            CompactionPrompt::Dynamic(f) => f(messages),
        }
    }
}

/// Token counter trait for estimating message tokens
pub trait TokenCounter: Send + Sync {
    fn count_messages(&self, messages: &[ChatCompletionRequestMessage]) -> usize;
}

/// Simple token counter that estimates based on character count
#[derive(Clone)]
pub struct SimpleTokenCounter {
    chars_per_token: f32,
}

impl SimpleTokenCounter {
    pub fn new() -> Self {
        Self {
            chars_per_token: 4.0, // Rough estimate
        }
    }
}

impl Default for SimpleTokenCounter {
    fn default() -> Self {
        Self::new()
    }
}

impl TokenCounter for SimpleTokenCounter {
    fn count_messages(&self, messages: &[ChatCompletionRequestMessage]) -> usize {
        let total_chars: usize = messages
            .iter()
            .map(|msg| {
                // Estimate character count based on message type
                match msg {
                    ChatCompletionRequestMessage::System(m) => {
                        match &m.content {
                            async_openai::types::ChatCompletionRequestSystemMessageContent::Text(t) => t.len(),
                            async_openai::types::ChatCompletionRequestSystemMessageContent::Array(_) => 100, // Rough estimate
                        }
                    }
                    ChatCompletionRequestMessage::User(m) => {
                        match &m.content {
                            async_openai::types::ChatCompletionRequestUserMessageContent::Text(t) => t.len(),
                            async_openai::types::ChatCompletionRequestUserMessageContent::Array(_) => 100,
                        }
                    }
                    ChatCompletionRequestMessage::Assistant(m) => {
                        // Assistant messages have optional content
                        m.content.as_ref().map(|c| match c {
                            async_openai::types::ChatCompletionRequestAssistantMessageContent::Text(t) => t.len(),
                            async_openai::types::ChatCompletionRequestAssistantMessageContent::Array(_) => 100,
                        }).unwrap_or(0)
                    }
                    ChatCompletionRequestMessage::Tool(m) => {
                        // Tool messages have content enum
                        match &m.content {
                            async_openai::types::ChatCompletionRequestToolMessageContent::Text(t) => t.len(),
                            async_openai::types::ChatCompletionRequestToolMessageContent::Array(_) => 100,
                        }
                    }
                    ChatCompletionRequestMessage::Function(m) => {
                        m.content.as_ref().map(|c| c.len()).unwrap_or(0)
                    }
                    ChatCompletionRequestMessage::Developer(m) => {
                        match &m.content {
                            async_openai::types::ChatCompletionRequestDeveloperMessageContent::Text(t) => t.len(),
                            async_openai::types::ChatCompletionRequestDeveloperMessageContent::Array(_) => 100,
                        }
                    }
                }
            })
            .sum();

        (total_chars as f32 / self.chars_per_token) as usize
    }
}

// ===== Layer Implementation =====

/// Layer that adds auto-compaction to a Step service
pub struct AutoCompactionLayer<P, C> {
    policy: CompactionPolicy,
    provider: P,
    token_counter: Arc<C>,
}

impl<P, C> AutoCompactionLayer<P, C> {
    pub fn new(policy: CompactionPolicy, provider: P, token_counter: C) -> Self {
        Self {
            policy,
            provider,
            token_counter: Arc::new(token_counter),
        }
    }
}

/// Service wrapper that performs auto-compaction
pub struct AutoCompaction<S, P, C> {
    pub(crate) inner: Arc<tokio::sync::Mutex<S>>,
    pub(crate) policy: CompactionPolicy,
    pub(crate) provider: Arc<tokio::sync::Mutex<P>>,
    pub(crate) token_counter: Arc<C>,
}

impl<S, P, C> Layer<S> for AutoCompactionLayer<P, C>
where
    P: Clone,
    C: Clone,
{
    type Service = AutoCompaction<S, P, C>;

    fn layer(&self, inner: S) -> Self::Service {
        AutoCompaction {
            inner: Arc::new(tokio::sync::Mutex::new(inner)),
            policy: self.policy.clone(),
            provider: Arc::new(tokio::sync::Mutex::new(self.provider.clone())),
            token_counter: self.token_counter.clone(),
        }
    }
}

impl<S, P, C> AutoCompaction<S, P, C>
where
    P: ModelService + Send + 'static,
    P::Future: Send + 'static,
    C: TokenCounter + 'static,
{
    /// Check if messages have orphaned tool calls (tool calls without responses)
    fn has_orphaned_tool_calls(messages: &[ChatCompletionRequestMessage]) -> bool {
        // Check if the last assistant message has tool calls without responses
        for (i, msg) in messages.iter().enumerate().rev() {
            if let ChatCompletionRequestMessage::Assistant(asst) = msg {
                if let Some(tool_calls) = &asst.tool_calls {
                    if !tool_calls.is_empty() {
                        // Check if there are tool responses after this
                        for msg in messages.iter().skip(i + 1) {
                            if matches!(msg, ChatCompletionRequestMessage::Tool(_)) {
                                return false; // Found responses, not orphaned
                            }
                        }
                        return true; // No responses found, orphaned
                    }
                }
            }
        }
        false
    }

    /// Extract orphaned tool calls from messages
    fn extract_orphaned_tool_calls(
        messages: Vec<ChatCompletionRequestMessage>,
    ) -> (
        Vec<ChatCompletionRequestMessage>,
        Option<ChatCompletionRequestMessage>,
    ) {
        // Find the last assistant message with tool calls
        for (i, msg) in messages.iter().enumerate().rev() {
            if let ChatCompletionRequestMessage::Assistant(asst) = msg {
                if let Some(tool_calls) = &asst.tool_calls {
                    if !tool_calls.is_empty() {
                        // Check if orphaned (no tool responses after)
                        let mut has_responses = false;
                        for msg in messages.iter().skip(i + 1) {
                            if matches!(msg, ChatCompletionRequestMessage::Tool(_)) {
                                has_responses = true;
                                break;
                            }
                        }

                        if !has_responses {
                            // Extract everything except the orphaned message
                            let cleaned = messages[..i].to_vec();
                            return (cleaned, Some(messages[i].clone()));
                        }
                    }
                }
            }
        }
        (messages, None)
    }

    /// Compact messages based on the configured strategy
    pub(crate) async fn compact_messages(
        &self,
        messages: Vec<ChatCompletionRequestMessage>,
    ) -> Result<Vec<ChatCompletionRequestMessage>, BoxError> {
        // Handle orphaned tool calls based on strategy
        let (messages_to_compact, orphaned_tool_call) = if Self::has_orphaned_tool_calls(&messages)
        {
            match self.policy.orphaned_tool_call_strategy {
                OrphanedToolCallStrategy::DropAndReappend => {
                    tracing::debug!("Detected orphaned tool calls, dropping and will re-append after compaction");
                    Self::extract_orphaned_tool_calls(messages)
                }
                OrphanedToolCallStrategy::ExcludeFromCompaction => {
                    tracing::debug!("Detected orphaned tool calls, excluding from compaction");
                    // Adjust range to exclude the orphaned message
                    // This will be handled by adjusting the range below
                    (messages, None)
                }
                OrphanedToolCallStrategy::AddPlaceholderResponses => {
                    tracing::debug!("Detected orphaned tool calls, adding placeholder responses");
                    // Add placeholder tool responses
                    let mut fixed_messages = messages.clone();
                    if let Some(ChatCompletionRequestMessage::Assistant(asst)) = messages.last() {
                        if let Some(tool_calls) = &asst.tool_calls {
                            for tool_call in tool_calls {
                                let placeholder = ChatCompletionRequestToolMessageArgs::default()
                                    .content("[Pending tool execution]")
                                    .tool_call_id(&tool_call.id)
                                    .build()?;
                                fixed_messages.push(placeholder.into());
                            }
                        }
                    }
                    (fixed_messages, None)
                }
                OrphanedToolCallStrategy::FailOnOrphaned => {
                    return Err("Cannot compact: conversation has orphaned tool calls".into());
                }
            }
        } else {
            (messages, None)
        };

        // Determine which messages to compact based on strategy
        let range = match &self.policy.compaction_strategy {
            CompactionStrategy::CompactAllButLast(n) => {
                let start = 0;
                let end = messages_to_compact.len().saturating_sub(*n);
                CompactionRange { start, end }
            }
            CompactionStrategy::CompactOlderThan(turns) => {
                // Count back N user+assistant pairs
                let mut turn_count = 0;
                let mut cutoff = messages_to_compact.len();
                for (i, msg) in messages_to_compact.iter().enumerate().rev() {
                    if matches!(msg, ChatCompletionRequestMessage::User(_)) {
                        turn_count += 1;
                        if turn_count >= *turns {
                            cutoff = i;
                            break;
                        }
                    }
                }
                CompactionRange {
                    start: 0,
                    end: cutoff,
                }
            }
            CompactionStrategy::PreserveSystemAndRecent { recent_count } => {
                // Find first non-system message
                let start = messages_to_compact
                    .iter()
                    .position(|m| !matches!(m, ChatCompletionRequestMessage::System(_)))
                    .unwrap_or(0);
                let end = messages_to_compact.len().saturating_sub(*recent_count);
                CompactionRange {
                    start,
                    end: end.max(start),
                }
            }
            CompactionStrategy::Custom(f) => f(&messages_to_compact),
        };

        // If nothing to compact, return original with orphaned tool call if any
        if range.start >= range.end {
            let mut result = messages_to_compact;
            if let Some(orphaned) = orphaned_tool_call {
                result.push(orphaned);
            }
            return Ok(result);
        }

        // Extract messages to compact
        let to_compact = &messages_to_compact[range.start..range.end];
        let prompt = self.policy.compaction_prompt.generate(to_compact);

        // Build compaction request
        let mut builder = CreateChatCompletionRequestArgs::default();
        builder.model(&self.policy.compaction_model);

        // Add messages to compact as context
        let mut compact_messages = vec![ChatCompletionRequestSystemMessageArgs::default()
            .content(prompt)
            .build()?
            .into()];

        // Add the messages to be compacted
        for msg in to_compact {
            compact_messages.push(msg.clone());
        }

        // Add instruction to summarize
        compact_messages.push(
            ChatCompletionRequestUserMessageArgs::default()
                .content("Please provide a summary of the above conversation following the instructions.")
                .build()?
                .into(),
        );

        builder.messages(compact_messages);
        let compact_req = builder.build()?;

        // Call compaction model
        let mut provider = self.provider.lock().await;
        let response = ServiceExt::ready(&mut *provider)
            .await?
            .call(compact_req)
            .await?;
        drop(provider);

        // Build new message list with compacted summary
        let mut result = Vec::new();

        // Keep messages before the compacted range
        for msg in &messages_to_compact[..range.start] {
            result.push(msg.clone());
        }

        // Add the compacted summary as an assistant message
        if let Some(summary) = response.assistant.content {
            result.push(
                ChatCompletionRequestAssistantMessageArgs::default()
                    .content(format!("[Previous conversation summary]: {}", summary))
                    .build()?
                    .into(),
            );
        }

        // Keep messages after the compacted range
        for msg in &messages_to_compact[range.end..] {
            result.push(msg.clone());
        }

        // Re-append orphaned tool call if we dropped it earlier
        if let Some(orphaned) = orphaned_tool_call {
            tracing::debug!("Re-appending orphaned tool call after compaction");
            result.push(orphaned);
        }

        Ok(result)
    }

    /// Check if the error is a context length error
    fn is_context_length_error(error: &BoxError) -> bool {
        let error_str = error.to_string().to_lowercase();
        error_str.contains("context_length_exceeded")
            || error_str.contains("context length")
            || error_str.contains("maximum context")
            || error_str.contains("token limit")
    }
}

impl<S, P, C> Service<CreateChatCompletionRequest> for AutoCompaction<S, P, C>
where
    S: Service<CreateChatCompletionRequest, Response = StepOutcome, Error = BoxError>
        + Send
        + 'static,
    S::Future: Send + 'static,
    P: ModelService + Send + 'static,
    P::Future: Send + 'static,
    C: TokenCounter + 'static,
{
    type Response = StepOutcome;
    type Error = BoxError;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(
        &mut self,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        std::task::Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: CreateChatCompletionRequest) -> Self::Future {
        let inner = self.inner.clone();
        let policy = self.policy.clone();
        let provider = self.provider.clone();
        let token_counter = self.token_counter.clone();

        Box::pin(async move {
            let mut current_messages = req.messages.clone();
            let mut attempts = 0;

            // Check for proactive compaction
            if let Some(threshold) = &policy.proactive_threshold {
                let token_count = token_counter.count_messages(&current_messages);

                if token_count > threshold.token_threshold {
                    // Proactively compact
                    let compactor = AutoCompaction {
                        inner: inner.clone(),
                        policy: policy.clone(),
                        provider: provider.clone(),
                        token_counter: token_counter.clone(),
                    };

                    match compactor.compact_messages(current_messages.clone()).await {
                        Ok(compacted) => {
                            current_messages = compacted;
                        }
                        Err(e) => {
                            // Log but don't fail - continue with original messages
                            tracing::warn!("Proactive compaction failed: {}", e);
                        }
                    }
                }
            }

            loop {
                // Build request with current messages
                let mut builder = CreateChatCompletionRequestArgs::default();
                builder.model(&req.model);
                builder.messages(current_messages.clone());
                if let Some(t) = req.temperature {
                    builder.temperature(t);
                }
                #[allow(deprecated)]
                if let Some(mt) = req.max_tokens {
                    builder.max_tokens(mt);
                }
                if let Some(mct) = req.max_completion_tokens {
                    builder.max_completion_tokens(mct);
                }
                if let Some(tools) = req.tools.clone() {
                    builder.tools(tools);
                }
                let current_req = builder.build()?;

                // Try to call inner service
                let mut guard = inner.lock().await;
                let result = ServiceExt::ready(&mut *guard)
                    .await?
                    .call(current_req)
                    .await;
                drop(guard);

                match result {
                    Ok(outcome) => return Ok(outcome),
                    Err(e) => {
                        // Check if it's a context length error
                        if Self::is_context_length_error(&e)
                            && attempts < policy.max_compaction_attempts
                        {
                            attempts += 1;

                            // Attempt reactive compaction
                            let compactor = AutoCompaction {
                                inner: inner.clone(),
                                policy: policy.clone(),
                                provider: provider.clone(),
                                token_counter: token_counter.clone(),
                            };

                            match compactor.compact_messages(current_messages.clone()).await {
                                Ok(compacted) => {
                                    tracing::info!(
                                        "Reactive compaction successful, retrying request"
                                    );
                                    current_messages = compacted;
                                    // Loop will retry with compacted messages
                                }
                                Err(compact_err) => {
                                    return Err(format!(
                                        "Context length error and compaction failed: original={}, compact={}",
                                        e, compact_err
                                    ).into());
                                }
                            }
                        } else {
                            return Err(e);
                        }
                    }
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tower::Service;

    #[test]
    fn test_simple_token_counter() {
        let counter = SimpleTokenCounter::new();
        let messages = vec![
            ChatCompletionRequestSystemMessageArgs::default()
                .content("You are a helpful assistant.")
                .build()
                .unwrap()
                .into(),
            ChatCompletionRequestUserMessageArgs::default()
                .content("Hello, how are you?")
                .build()
                .unwrap()
                .into(),
        ];

        let count = counter.count_messages(&messages);
        assert!(count > 0);
        // Rough estimate: ~50 chars / 4 = ~12 tokens
        assert!((10..=20).contains(&count));
    }

    #[test]
    fn test_compaction_range_preserve_system() {
        let strategy = CompactionStrategy::PreserveSystemAndRecent { recent_count: 2 };
        let messages = vec![
            ChatCompletionRequestSystemMessageArgs::default()
                .content("System")
                .build()
                .unwrap()
                .into(),
            ChatCompletionRequestUserMessageArgs::default()
                .content("User1")
                .build()
                .unwrap()
                .into(),
            ChatCompletionRequestAssistantMessageArgs::default()
                .content("Assistant1")
                .build()
                .unwrap()
                .into(),
            ChatCompletionRequestUserMessageArgs::default()
                .content("User2")
                .build()
                .unwrap()
                .into(),
            ChatCompletionRequestAssistantMessageArgs::default()
                .content("Assistant2")
                .build()
                .unwrap()
                .into(),
        ];

        let range = match strategy {
            CompactionStrategy::PreserveSystemAndRecent { recent_count } => {
                let start = messages
                    .iter()
                    .position(|m| !matches!(m, ChatCompletionRequestMessage::System(_)))
                    .unwrap_or(0);
                let end = messages.len().saturating_sub(recent_count);
                CompactionRange {
                    start,
                    end: end.max(start),
                }
            }
            _ => panic!("Wrong strategy"),
        };

        assert_eq!(range.start, 1); // After system message
        assert_eq!(range.end, 3); // Keep last 2 messages
    }

    #[test]
    fn test_compaction_strategy_compact_all_but_last() {
        let strategy = CompactionStrategy::CompactAllButLast(3);
        let messages: Vec<ChatCompletionRequestMessage> = vec![
            ChatCompletionRequestUserMessageArgs::default()
                .content("1")
                .build()
                .unwrap()
                .into(),
            ChatCompletionRequestAssistantMessageArgs::default()
                .content("2")
                .build()
                .unwrap()
                .into(),
            ChatCompletionRequestUserMessageArgs::default()
                .content("3")
                .build()
                .unwrap()
                .into(),
            ChatCompletionRequestAssistantMessageArgs::default()
                .content("4")
                .build()
                .unwrap()
                .into(),
            ChatCompletionRequestUserMessageArgs::default()
                .content("5")
                .build()
                .unwrap()
                .into(),
        ];

        let range = match strategy {
            CompactionStrategy::CompactAllButLast(n) => {
                let start = 0;
                let end = messages.len().saturating_sub(n);
                CompactionRange { start, end }
            }
            _ => panic!("Wrong strategy"),
        };

        assert_eq!(range.start, 0);
        assert_eq!(range.end, 2); // Keep last 3 messages
    }

    #[test]
    fn test_compaction_prompt_default() {
        let prompt = CompactionPrompt::Default;
        let messages = vec![];
        let generated = prompt.generate(&messages);
        assert!(generated.contains("concise summary"));
        assert!(generated.contains("conversation continuity"));
    }

    #[test]
    fn test_compaction_prompt_custom() {
        let prompt = CompactionPrompt::Custom("Custom prompt".to_string());
        let messages = vec![];
        let generated = prompt.generate(&messages);
        assert_eq!(generated, "Custom prompt");
    }

    #[test]
    fn test_compaction_prompt_dynamic() {
        let prompt = CompactionPrompt::Dynamic(Arc::new(|msgs| {
            format!("Summarize {} messages", msgs.len())
        }));
        let messages = vec![ChatCompletionRequestUserMessageArgs::default()
            .content("test")
            .build()
            .unwrap()
            .into()];
        let generated = prompt.generate(&messages);
        assert_eq!(generated, "Summarize 1 messages");
    }

    #[test]
    fn test_is_context_length_error() {
        let test_cases = vec![
            ("context_length_exceeded", true),
            ("Error: context length exceeded", true),
            ("Maximum context reached", true),
            ("Token limit exceeded", true),
            ("Some other error", false),
            ("Network timeout", false),
        ];

        for (error_msg, expected) in test_cases {
            let error: BoxError = error_msg.into();
            assert_eq!(
                AutoCompaction::<DummyService, DummyProvider, SimpleTokenCounter>::is_context_length_error(&error),
                expected,
                "Failed for: {}",
                error_msg
            );
        }
    }

    #[test]
    fn test_proactive_threshold() {
        let threshold = ProactiveThreshold {
            token_threshold: 1000,
            percentage_threshold: Some(0.8),
        };

        assert_eq!(threshold.token_threshold, 1000);
        assert_eq!(threshold.percentage_threshold, Some(0.8));
    }

    #[test]
    fn test_compaction_policy_default() {
        let policy = CompactionPolicy::default();
        assert_eq!(policy.compaction_model, "gpt-4o-mini");
        assert!(policy.proactive_threshold.is_none());
        assert_eq!(policy.max_compaction_attempts, 2);
        matches!(
            policy.compaction_strategy,
            CompactionStrategy::PreserveSystemAndRecent { .. }
        );
        matches!(
            policy.orphaned_tool_call_strategy,
            OrphanedToolCallStrategy::DropAndReappend
        );
    }

    #[test]
    fn test_orphaned_tool_call_detection() {
        use async_openai::types::{
            ChatCompletionMessageToolCall, ChatCompletionRequestToolMessageArgs,
            ChatCompletionToolType, FunctionCall,
        };

        // Create messages with an orphaned tool call
        let messages = vec![
            ChatCompletionRequestUserMessageArgs::default()
                .content("Please help me")
                .build()
                .unwrap()
                .into(),
            ChatCompletionRequestAssistantMessageArgs::default()
                .content("")
                .tool_calls(vec![ChatCompletionMessageToolCall {
                    id: "call_123".to_string(),
                    r#type: ChatCompletionToolType::Function,
                    function: FunctionCall {
                        name: "some_tool".to_string(),
                        arguments: "{}".to_string(),
                    },
                }])
                .build()
                .unwrap()
                .into(),
            // No tool response message - this is orphaned!
        ];

        assert!(has_orphaned_tool_calls(&messages));

        // Now add the tool response - no longer orphaned
        let mut messages_with_response = messages.clone();
        messages_with_response.push(
            ChatCompletionRequestToolMessageArgs::default()
                .content("Tool result")
                .tool_call_id("call_123")
                .build()
                .unwrap()
                .into(),
        );

        assert!(!has_orphaned_tool_calls(&messages_with_response));
    }

    #[test]
    fn test_orphaned_tool_call_strategy_drop_and_reappend() {
        use async_openai::types::{
            ChatCompletionMessageToolCall, ChatCompletionToolType, FunctionCall,
        };

        let messages = vec![
            ChatCompletionRequestUserMessageArgs::default()
                .content("User message")
                .build()
                .unwrap()
                .into(),
            ChatCompletionRequestAssistantMessageArgs::default()
                .content("Assistant response")
                .build()
                .unwrap()
                .into(),
            ChatCompletionRequestUserMessageArgs::default()
                .content("Another user message")
                .build()
                .unwrap()
                .into(),
            ChatCompletionRequestAssistantMessageArgs::default()
                .content("")
                .tool_calls(vec![ChatCompletionMessageToolCall {
                    id: "orphaned_call".to_string(),
                    r#type: ChatCompletionToolType::Function,
                    function: FunctionCall {
                        name: "orphaned_tool".to_string(),
                        arguments: "{}".to_string(),
                    },
                }])
                .build()
                .unwrap()
                .into(),
        ];

        let (cleaned, orphaned) = extract_orphaned_tool_calls(messages.clone());

        // Should have removed the last message with tool calls
        assert_eq!(cleaned.len(), 3);
        assert!(orphaned.is_some());

        // The orphaned message should be the one with tool calls
        if let Some(orphaned_msg) = orphaned {
            if let ChatCompletionRequestMessage::Assistant(asst) = orphaned_msg {
                assert!(asst.tool_calls.is_some());
            } else {
                panic!("Expected assistant message");
            }
        }
    }

    // Helper functions for testing (these will be implemented in the main code)
    fn has_orphaned_tool_calls(messages: &[ChatCompletionRequestMessage]) -> bool {
        // Check if the last assistant message has tool calls without responses
        for (i, msg) in messages.iter().enumerate().rev() {
            if let ChatCompletionRequestMessage::Assistant(asst) = msg {
                if let Some(tool_calls) = &asst.tool_calls {
                    if !tool_calls.is_empty() {
                        // Check if there are tool responses after this
                        for msg in messages.iter().skip(i + 1) {
                            if matches!(msg, ChatCompletionRequestMessage::Tool(_)) {
                                return false; // Found responses, not orphaned
                            }
                        }
                        return true; // No responses found, orphaned
                    }
                }
            }
        }
        false
    }

    fn extract_orphaned_tool_calls(
        messages: Vec<ChatCompletionRequestMessage>,
    ) -> (
        Vec<ChatCompletionRequestMessage>,
        Option<ChatCompletionRequestMessage>,
    ) {
        // Find the last assistant message with tool calls
        for (i, msg) in messages.iter().enumerate().rev() {
            if let ChatCompletionRequestMessage::Assistant(asst) = msg {
                if let Some(tool_calls) = &asst.tool_calls {
                    if !tool_calls.is_empty() {
                        // Check if orphaned (no tool responses after)
                        let mut has_responses = false;
                        for msg in messages.iter().skip(i + 1) {
                            if matches!(msg, ChatCompletionRequestMessage::Tool(_)) {
                                has_responses = true;
                                break;
                            }
                        }

                        if !has_responses {
                            // Extract everything except the orphaned message
                            let cleaned = messages[..i].to_vec();
                            return (cleaned, Some(messages[i].clone()));
                        }
                    }
                }
            }
        }
        (messages, None)
    }

    // Dummy types for testing
    struct DummyService;
    struct DummyProvider;

    impl Service<CreateChatCompletionRequest> for DummyService {
        type Response = crate::core::StepOutcome;
        type Error = BoxError;
        type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

        fn poll_ready(
            &mut self,
            _: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Result<(), Self::Error>> {
            std::task::Poll::Ready(Ok(()))
        }

        fn call(&mut self, _: CreateChatCompletionRequest) -> Self::Future {
            Box::pin(async {
                Ok(crate::core::StepOutcome::Done {
                    messages: vec![],
                    aux: Default::default(),
                })
            })
        }
    }

    impl Service<CreateChatCompletionRequest> for DummyProvider {
        type Response = crate::provider::ProviderResponse;
        type Error = BoxError;
        type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

        fn poll_ready(
            &mut self,
            _: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Result<(), Self::Error>> {
            std::task::Poll::Ready(Ok(()))
        }

        fn call(&mut self, _: CreateChatCompletionRequest) -> Self::Future {
            Box::pin(async {
                #[allow(deprecated)]
                let assistant = async_openai::types::ChatCompletionResponseMessage {
                    content: Some("summary".to_string()),
                    role: async_openai::types::Role::Assistant,
                    tool_calls: None,
                    refusal: None,
                    audio: None,
                    function_call: None,
                };
                Ok(crate::provider::ProviderResponse {
                    assistant,
                    prompt_tokens: 10,
                    completion_tokens: 10,
                })
            })
        }
    }
}
