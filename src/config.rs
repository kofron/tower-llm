//! Configuration system for the Agents SDK
//!
//! Provides flexible configuration options for agents, runners, and tools.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;

/// Global SDK configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SdkConfig {
    /// Default model to use
    pub default_model: String,

    /// Default temperature for generation
    pub default_temperature: f32,

    /// Default max tokens
    pub default_max_tokens: Option<usize>,

    /// Timeout for API calls
    pub api_timeout: Duration,

    /// Retry configuration
    pub retry_config: RetryConfig,

    /// Rate limiting configuration
    pub rate_limit_config: RateLimitConfig,

    /// Default session storage path
    pub session_storage_path: PathBuf,

    /// Enable debug logging
    pub debug_mode: bool,
}

impl Default for SdkConfig {
    fn default() -> Self {
        Self {
            default_model: "gpt-4".to_string(),
            default_temperature: 0.7,
            default_max_tokens: None,
            api_timeout: Duration::from_secs(30),
            retry_config: RetryConfig::default(),
            rate_limit_config: RateLimitConfig::default(),
            session_storage_path: PathBuf::from("./sessions"),
            debug_mode: false,
        }
    }
}

/// Retry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum number of retries
    pub max_retries: usize,

    /// Initial retry delay
    pub initial_delay: Duration,

    /// Maximum retry delay
    pub max_delay: Duration,

    /// Exponential backoff multiplier
    pub backoff_multiplier: f32,

    /// Jitter to add randomness to retries
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

/// Rate limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    /// Requests per minute
    pub requests_per_minute: Option<usize>,

    /// Tokens per minute
    pub tokens_per_minute: Option<usize>,

    /// Enable automatic rate limit handling
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

/// Agent-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfigOptions {
    /// Model to use for this agent
    pub model: Option<String>,

    /// Temperature for generation
    pub temperature: Option<f32>,

    /// Maximum tokens to generate
    pub max_tokens: Option<usize>,

    /// Top-p sampling
    pub top_p: Option<f32>,

    /// Frequency penalty
    pub frequency_penalty: Option<f32>,

    /// Presence penalty
    pub presence_penalty: Option<f32>,

    /// Stop sequences
    pub stop_sequences: Vec<String>,

    /// Enable function calling
    pub enable_functions: bool,

    /// Maximum parallel tool calls
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

/// Configuration builder
pub struct ConfigBuilder {
    config: SdkConfig,
}

impl Default for ConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ConfigBuilder {
    pub fn new() -> Self {
        Self {
            config: SdkConfig::default(),
        }
    }

    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.config.default_model = model.into();
        self
    }

    pub fn temperature(mut self, temp: f32) -> Self {
        self.config.default_temperature = temp;
        self
    }

    pub fn max_tokens(mut self, tokens: usize) -> Self {
        self.config.default_max_tokens = Some(tokens);
        self
    }

    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.config.api_timeout = timeout;
        self
    }

    pub fn max_retries(mut self, retries: usize) -> Self {
        self.config.retry_config.max_retries = retries;
        self
    }

    pub fn debug(mut self, enabled: bool) -> Self {
        self.config.debug_mode = enabled;
        self
    }

    pub fn session_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.config.session_storage_path = path.into();
        self
    }

    pub fn build(self) -> SdkConfig {
        self.config
    }
}

/// Load configuration from environment variables
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

/// Load configuration from a TOML file
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
        assert_eq!(config.default_model, "gpt-4");
        assert_eq!(config.default_temperature, 0.7);
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
            model: Some("gpt-4".to_string()),
            temperature: Some(0.9),
            max_parallel_tools: 10,
            ..Default::default()
        };

        assert_eq!(options.model, Some("gpt-4".to_string()));
        assert_eq!(options.temperature, Some(0.9));
        assert_eq!(options.max_parallel_tools, 10);
        assert!(options.enable_functions);
    }
}
