//! # SDK Configuration System
//!
//! This module provides a flexible and comprehensive configuration system for
//! the OpenAI Agents SDK. It allows you to fine-tune the behavior of agents,
//! runners, and other components through a unified interface.
//!
//! The configuration is managed through a set of structs:
//!
//! - [`SdkConfig`]: The main configuration struct that holds global settings
//!   for the SDK, such as default model parameters, API timeouts, and retry
//!   strategies.
//! - [`RetryConfig`]: Defines the parameters for handling transient network
//!   errors, including the number of retries and backoff delays.
//! - [`RateLimitConfig`]: Configures how the SDK should handle rate limiting
//!   from the API.
//! - [`AgentConfigOptions`]: Provides agent-specific overrides for model
//!   parameters, allowing for fine-grained control over individual agents.
//!
//! ## Configuration Methods
//!
//! There are multiple ways to create and manage configurations:
//!
//! 1.  **Builder Pattern**: The [`ConfigBuilder`] provides a fluent interface for
//!     constructing an `SdkConfig` programmatically.
//! 2.  **Environment Variables**: The `from_env` function loads configuration
//!     settings from environment variables, such as `OPENAI_MODEL` and
//!     `AGENTS_DEBUG`.
//! 3.  **TOML File**: The `from_file` function allows you to load a complete
//!     configuration from a TOML file, which is ideal for production environments.
//!
//! ### Example: Using the `ConfigBuilder`
//!
//! ```rust
//! use openai_agents_rs::config::ConfigBuilder;
//! use std::time::Duration;
//!
//! let config = ConfigBuilder::new()
//!     .model("gpt-3.5-turbo")
//!     .temperature(0.8)
//!     .timeout(Duration::from_secs(60))
//!     .debug(true)
//!     .build();
//!
//! assert_eq!(config.default_model, "gpt-3.5-turbo");
//! assert_eq!(config.api_timeout, Duration::from_secs(60));
//! assert!(config.debug_mode);
//! ```

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;

/// The main configuration struct for the entire SDK.
///
/// `SdkConfig` holds all the global settings that control the behavior of the
/// agents and the underlying API client. It provides default values that can be
/// customized as needed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SdkConfig {
    /// The default model to be used by agents if not otherwise specified.
    pub default_model: String,

    /// The default temperature for the LLM's response generation.
    pub default_temperature: f32,

    /// The default maximum number of tokens to generate in a response.
    pub default_max_tokens: Option<usize>,

    /// The timeout for API calls to the LLM provider.
    pub api_timeout: Duration,

    /// Configuration for handling retries on network errors.
    pub retry_config: RetryConfig,

    /// Configuration for managing API rate limits.
    pub rate_limit_config: RateLimitConfig,

    /// The default path for storing persistent session data.
    pub session_storage_path: PathBuf,

    /// A flag to enable or disable debug logging throughout the SDK.
    pub debug_mode: bool,
}

impl Default for SdkConfig {
    fn default() -> Self {
        Self {
            default_model: "gpt-5".to_string(),
            default_temperature: 1.0,
            default_max_tokens: None,
            api_timeout: Duration::from_secs(30),
            retry_config: RetryConfig::default(),
            rate_limit_config: RateLimitConfig::default(),
            session_storage_path: PathBuf::from("./sessions"),
            debug_mode: false,
        }
    }
}

/// Defines the retry strategy for handling transient network errors.
///
/// This struct allows you to configure an exponential backoff strategy with
/// optional jitter to prevent thundering herd problems.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// The maximum number of times to retry a failed API request.
    pub max_retries: usize,

    /// The initial delay to wait before the first retry.
    pub initial_delay: Duration,

    /// The maximum delay to wait between retries.
    pub max_delay: Duration,

    /// The multiplier for increasing the delay between retries. A value of 2.0
    /// means the delay will double with each retry.
    pub backoff_multiplier: f32,

    /// A flag to enable or disable jitter, which adds a small amount of
    /// randomness to the delay to avoid synchronized retries.
    pub jitter: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            backoff_multiplier: 2.0,
            jitter: true,
        }
    }
}

/// Configuration for handling API rate limits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    /// The maximum number of requests allowed per minute.
    pub requests_per_minute: Option<usize>,

    /// The maximum number of tokens that can be processed per minute.
    pub tokens_per_minute: Option<usize>,

    /// A flag to enable or disable automatic throttling. When enabled, the SDK
    /// will automatically pause to stay within the specified rate limits.
    pub auto_throttle: bool,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            requests_per_minute: Some(60),
            tokens_per_minute: Some(90000),
            auto_throttle: true,
        }
    }
}

/// Provides agent-specific overrides for model parameters.
///
/// This struct allows you to fine-tune the behavior of individual agents,
/// overriding the global settings in `SdkConfig`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfigOptions {
    /// The specific model to be used for this agent.
    pub model: Option<String>,

    /// The temperature for this agent's response generation.
    pub temperature: Option<f32>,

    /// The maximum number of tokens to generate for this agent.
    pub max_tokens: Option<usize>,

    /// The nucleus sampling parameter, which controls the diversity of the
    /// generated text.
    pub top_p: Option<f32>,

    /// A penalty for repeating tokens, discouraging the model from generating
    /// repetitive text.
    pub frequency_penalty: Option<f32>,

    /// A penalty for introducing new tokens, encouraging the model to stay on
    /// topic.
    pub presence_penalty: Option<f32>,

    /// A list of sequences that, when generated, will cause the model to stop.
    pub stop_sequences: Vec<String>,

    /// A flag to enable or disable the use of function calling (tools) for this
    /// agent.
    pub enable_functions: bool,

    /// The maximum number of tools that the agent can call in parallel.
    pub max_parallel_tools: usize,
}

impl Default for AgentConfigOptions {
    fn default() -> Self {
        Self {
            model: None,
            temperature: None,
            max_tokens: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop_sequences: vec![],
            enable_functions: true,
            max_parallel_tools: 5,
        }
    }
}

/// A builder for constructing an `SdkConfig` instance.
///
/// The `ConfigBuilder` provides a fluent API for creating a custom configuration,
/// allowing you to chain method calls to set the desired parameters.
pub struct ConfigBuilder {
    config: SdkConfig,
}

impl Default for ConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ConfigBuilder {
    /// Creates a new `ConfigBuilder` with default settings.
    pub fn new() -> Self {
        Self {
            config: SdkConfig::default(),
        }
    }

    /// Sets the default model for the configuration.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.config.default_model = model.into();
        self
    }

    /// Sets the default temperature for the configuration.
    pub fn temperature(mut self, temp: f32) -> Self {
        self.config.default_temperature = temp;
        self
    }

    /// Sets the default maximum number of tokens for the configuration.
    pub fn max_tokens(mut self, tokens: usize) -> Self {
        self.config.default_max_tokens = Some(tokens);
        self
    }

    /// Sets the API timeout for the configuration.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.config.api_timeout = timeout;
        self
    }

    /// Sets the maximum number of retries for the configuration.
    pub fn max_retries(mut self, retries: usize) -> Self {
        self.config.retry_config.max_retries = retries;
        self
    }

    /// Enables or disables debug mode for the configuration.
    pub fn debug(mut self, enabled: bool) -> Self {
        self.config.debug_mode = enabled;
        self
    }

    /// Sets the session storage path for the configuration.
    pub fn session_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.config.session_storage_path = path.into();
        self
    }

    /// Builds and returns the final `SdkConfig`.
    pub fn build(self) -> SdkConfig {
        self.config
    }
}

/// Loads the SDK configuration from environment variables.
///
/// This function looks for specific environment variables, such as `OPENAI_MODEL`
/// and `AGENTS_DEBUG`, to populate the `SdkConfig`.
pub fn from_env() -> SdkConfig {
    let mut config = SdkConfig::default();

    if let Ok(model) = std::env::var("OPENAI_MODEL") {
        config.default_model = model;
    }

    if let Ok(temp) = std::env::var("OPENAI_TEMPERATURE") {
        if let Ok(temp_f) = temp.parse::<f32>() {
            config.default_temperature = temp_f;
        }
    }

    if let Ok(debug) = std::env::var("AGENTS_DEBUG") {
        config.debug_mode = debug.to_lowercase() == "true" || debug == "1";
    }

    if let Ok(timeout) = std::env::var("AGENTS_TIMEOUT") {
        if let Ok(timeout_secs) = timeout.parse::<u64>() {
            config.api_timeout = Duration::from_secs(timeout_secs);
        }
    }

    config
}

/// Loads the SDK configuration from a TOML file.
///
/// This function reads the specified TOML file and deserializes it into an
/// `SdkConfig` struct, allowing for file-based configuration management.
pub fn from_file(
    path: impl AsRef<std::path::Path>,
) -> Result<SdkConfig, Box<dyn std::error::Error>> {
    let contents = std::fs::read_to_string(path)?;
    let config: SdkConfig = toml::from_str(&contents)?;
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = SdkConfig::default();
        assert_eq!(config.default_model, "gpt-5");
        assert_eq!(config.default_temperature, 1.0);
        assert!(config.rate_limit_config.auto_throttle);
    }

    #[test]
    fn test_config_builder() {
        let config = ConfigBuilder::new()
            .model("gpt-3.5-turbo")
            .temperature(0.5)
            .max_tokens(1000)
            .debug(true)
            .build();

        assert_eq!(config.default_model, "gpt-3.5-turbo");
        assert_eq!(config.default_temperature, 0.5);
        assert_eq!(config.default_max_tokens, Some(1000));
        assert!(config.debug_mode);
    }

    #[test]
    fn test_retry_config() {
        let retry = RetryConfig::default();
        assert_eq!(retry.max_retries, 3);
        assert_eq!(retry.backoff_multiplier, 2.0);
        assert!(retry.jitter);
    }

    #[test]
    fn test_agent_config_options() {
        let options = AgentConfigOptions {
            model: Some("gpt-5".to_string()),
            temperature: Some(0.9),
            max_parallel_tools: 10,
            ..Default::default()
        };

        assert_eq!(options.model, Some("gpt-5".to_string()));
        assert_eq!(options.temperature, Some(0.9));
        assert_eq!(options.max_parallel_tools, 10);
        assert!(options.enable_functions);
    }
}
