//! # Agent Guardrails for Input and Output Validation
//!
//! Guardrails are a critical component for building safe and reliable agents.
//! They provide a mechanism to validate and sanitize the data that flows into
//! and out of an agent. This module introduces two types of guardrails:
//!
//! - **[`InputGuardrail`]**: Checks user input before it is processed by the agent.
//!   This can be used to prevent prompt injection, enforce content policies, or
//!   block malicious inputs.
//! - **[`OutputGuardrail`]**: Validates the agent's response before it is sent
//!   to the user. This can be used for content moderation, ensuring the output
//!   conforms to a specific format, or redacting sensitive information.
//!
//! ## How Guardrails Work
//!
//! Guardrails are implemented as traits and are executed by the [`GuardrailRunner`].
//! They are executed in order of their `priority`, with higher priority guardrails
//! running first. If a guardrail check fails, the agent run is immediately
//! halted, and an error is returned.
//!
//! Output guardrails also have the ability to modify the content before passing
//! it along to the next guardrail or back to the user.
//!
//! ### Example: Using a `MaxLengthGuardrail`
//!
//! ```rust
//! use openai_agents_rs::guardrail::{InputGuardrail, MaxLengthGuardrail};
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let guard = MaxLengthGuardrail::new(50);
//!
//! // This input is valid.
//! let result = guard.check("This is a reasonable length input.").await?;
//! assert!(result.passed);
//!
//! // This input will fail the check.
//! let long_input = "This input is definitely way too long and will not be \
//!                     allowed to pass through the guardrail.";
//! let result = guard.check(long_input).await?;
//! assert!(!result.passed);
//! assert!(result.reason.unwrap().contains("exceeds maximum length"));
//! # Ok(())
//! # }
//! ```
//!
//! ### Example: Using a `PatternBlockGuardrail`
//!
//! ```rust
//! use openai_agents_rs::guardrail::{InputGuardrail, PatternBlockGuardrail};
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let profanity_filter = PatternBlockGuardrail::new(
//!     "ProfanityFilter",
//!     vec!["darn".to_string(), "heck".to_string()],
//! );
//!
//! // This input is clean.
//! let result = profanity_filter.check("What a wonderful day!").await?;
//! assert!(result.passed);
//!
//! // This input contains a blocked pattern.
//! let result = profanity_filter.check("Oh, darn it!").await?;
//! assert!(!result.passed);
//! assert!(result.reason.unwrap().contains("darn"));
//! # Ok(())
//! # }
//! ```
//!
//! ## Custom Guardrails
//!
//! You can implement your own guardrails by implementing the [`InputGuardrail`]
//! and [`OutputGuardrail`] traits. This allows you to add specific validation
//! logic based on your requirements.
//!
//! ## Priority Ordering
//!
//! Guardrails are executed in descending order of priority. This means that
//! guardrails with a higher priority value will run before those with a lower
//! priority. This allows you to control the order in which checks are performed.
//!
//! ```rust
//! use openai_agents_rs::guardrail::{InputGuardrail, GuardrailResult, GuardrailRunner};
//! use openai_agents_rs::error::{Result, AgentsError};
//! use async_trait::async_trait;
//!
//! # #[derive(Debug)]
//! # struct AlwaysFailGuard {
//! #     name: String,
//! #     priority: i32,
//! # }
//!
//! # #[async_trait]
//! # impl InputGuardrail for AlwaysFailGuard {
//! #     fn name(&self) -> &str {
//! #         &self.name
//! #     }
//! #
//! #     async fn check(&self, _: &str) -> Result<GuardrailResult> {
//! #         Ok(GuardrailResult::fail(self.name.clone()))
//! #     }
//! #
//! #     fn priority(&self) -> i32 {
//! #         self.priority
//! #     }
//! # }
//!
//! # #[tokio::main]
//! # async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
//! #     let guards: Vec<std::sync::Arc<dyn InputGuardrail>> = vec![
//! #         std::sync::Arc::new(AlwaysFailGuard {
//! #             name: "LowPriority".to_string(),
//! #             priority: 1,
//! #         }),
//! #         std::sync::Arc::new(AlwaysFailGuard {
//! #             name: "HighPriority".to_string(),
//! #             priority: 10,
//! #         }),
//! #     ];
//! #
//! #     let result = GuardrailRunner::check_input(&guards, "test").await;
//! #     assert!(result.is_err());
//! #
//! #     // Should fail with high priority guard first
//! #     if let Err(AgentsError::InputGuardrailTriggered { message }) = result {
//! #         assert_eq!(message, "HighPriority");
//! #     } else {
//! #         panic!("Expected InputGuardrailTriggered error");
//! #     }
//! #     Ok(())
//! # }
//! ```
//!
//! ```
//!
//! ```

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use crate::error::{AgentsError, Result};

/// Represents the result of a single guardrail check.
///
/// A `GuardrailResult` indicates whether the check passed and, if not, provides
/// a reason for the failure. For output guardrails, it can also include a
/// modified version of the content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuardrailResult {
    /// A boolean indicating whether the content passed the guardrail check.
    pub passed: bool,

    /// An optional string providing a reason for the failure. This is `None` if
    /// the check passed.
    pub reason: Option<String>,

    /// For output guardrails, this field can contain a modified version of the
    /// content. If `None`, the original content is used.
    pub modified_content: Option<String>,
}

impl GuardrailResult {
    /// Creates a `GuardrailResult` indicating that the check passed.
    pub fn pass() -> Self {
        Self {
            passed: true,
            reason: None,
            modified_content: None,
        }
    }

    /// Creates a `GuardrailResult` indicating that the check failed, with a
    /// provided reason.
    pub fn fail(reason: impl Into<String>) -> Self {
        Self {
            passed: false,
            reason: Some(reason.into()),
            modified_content: None,
        }
    }

    /// Creates a `GuardrailResult` indicating that the check passed, but with
    /// modified content. This is typically used by output guardrails that
    /// sanitize or reformat the agent's response.
    pub fn pass_with_modification(content: impl Into<String>) -> Self {
        Self {
            passed: true,
            reason: None,
            modified_content: Some(content.into()),
        }
    }
}

/// A trait for implementing input guardrails.
///
/// Input guardrails are used to validate user input before it is sent to the
/// agent. This is a crucial step for preventing security vulnerabilities and
/// ensuring the agent behaves as expected.
#[async_trait]
pub trait InputGuardrail: Send + Sync + Debug {
    /// Returns the name of the guardrail, used for logging and identification.
    fn name(&self) -> &str;

    /// Performs the validation check on the input string.
    ///
    /// The method returns a [`GuardrailResult`] indicating whether the input
    /// is valid.
    async fn check(&self, input: &str) -> Result<GuardrailResult>;

    /// Defines the execution priority of the guardrail.
    ///
    /// Guardrails are executed in descending order of priority. Higher values
    /// are executed first.
    fn priority(&self) -> i32 {
        0
    }
}

/// A trait for implementing output guardrails.
///
/// Output guardrails are used to validate the agent's response before it is
/// sent to the user. They can also be used to modify the output, for example,
/// to redact sensitive information.
#[async_trait]
pub trait OutputGuardrail: Send + Sync + Debug {
    /// Returns the name of the guardrail.
    fn name(&self) -> &str;

    /// Performs the validation check on the output string.
    ///
    /// This method can either approve, reject, or modify the output. The
    /// result is returned as a [`GuardrailResult`].
    async fn check(&self, output: &str) -> Result<GuardrailResult>;

    /// Defines the execution priority of the guardrail.
    fn priority(&self) -> i32 {
        0
    }
}

/// An [`InputGuardrail`] that checks if the input length exceeds a maximum value.
#[derive(Debug, Clone)]
pub struct MaxLengthGuardrail {
    name: String,
    max_length: usize,
}

impl MaxLengthGuardrail {
    /// Creates a new `MaxLengthGuardrail` with the specified maximum length.
    pub fn new(max_length: usize) -> Self {
        Self {
            name: format!("MaxLength_{}", max_length),
            max_length,
        }
    }
}

#[async_trait]
impl InputGuardrail for MaxLengthGuardrail {
    fn name(&self) -> &str {
        &self.name
    }

    async fn check(&self, input: &str) -> Result<GuardrailResult> {
        if input.len() > self.max_length {
            Ok(GuardrailResult::fail(format!(
                "Input exceeds maximum length of {} characters",
                self.max_length
            )))
        } else {
            Ok(GuardrailResult::pass())
        }
    }
}

/// An [`InputGuardrail`] and [`OutputGuardrail`] that blocks content containing
/// specific patterns.
///
/// This guardrail is useful for filtering out inappropriate language, sensitive
/// information, or other undesirable content. The pattern matching is case-insensitive.
///
/// ### Example: Filtering Profanity
///
/// ```rust
/// use openai_agents_rs::guardrail::{InputGuardrail, PatternBlockGuardrail};
///
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let profanity_filter = PatternBlockGuardrail::new(
///     "ProfanityFilter",
///     vec!["darn".to_string(), "heck".to_string()],
/// );
///
/// // This input is clean.
/// let result = profanity_filter.check("What a wonderful day!").await?;
/// assert!(result.passed);
///
/// // This input contains a blocked pattern.
/// let result = profanity_filter.check("Oh, darn it!").await?;
/// assert!(!result.passed);
/// assert!(result.reason.unwrap().contains("darn"));
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct PatternBlockGuardrail {
    name: String,
    patterns: Vec<String>,
}

impl PatternBlockGuardrail {
    /// Creates a new `PatternBlockGuardrail` with a list of patterns to block.
    pub fn new(name: impl Into<String>, patterns: Vec<String>) -> Self {
        Self {
            name: name.into(),
            patterns,
        }
    }
}

#[async_trait]
impl InputGuardrail for PatternBlockGuardrail {
    fn name(&self) -> &str {
        &self.name
    }

    async fn check(&self, input: &str) -> Result<GuardrailResult> {
        let input_lower = input.to_lowercase();
        for pattern in &self.patterns {
            if input_lower.contains(&pattern.to_lowercase()) {
                return Ok(GuardrailResult::fail(format!(
                    "Input contains blocked pattern: {}",
                    pattern
                )));
            }
        }
        Ok(GuardrailResult::pass())
    }
}

#[async_trait]
impl OutputGuardrail for PatternBlockGuardrail {
    fn name(&self) -> &str {
        &self.name
    }

    async fn check(&self, output: &str) -> Result<GuardrailResult> {
        let output_lower = output.to_lowercase();
        for pattern in &self.patterns {
            if output_lower.contains(&pattern.to_lowercase()) {
                return Ok(GuardrailResult::fail(format!(
                    "Output contains blocked pattern: {}",
                    pattern
                )));
            }
        }
        Ok(GuardrailResult::pass())
    }
}

/// A helper struct for executing a sequence of guardrails.
///
/// The `GuardrailRunner` is responsible for sorting guardrails by priority and
/// executing them in the correct order.
pub struct GuardrailRunner;

impl GuardrailRunner {
    /// Executes a sequence of input guardrails.
    ///
    /// If any guardrail fails, the method returns an `Err` with a
    /// [`AgentsError::InputGuardrailTriggered`] error.
    pub async fn check_input(
        guardrails: &[std::sync::Arc<dyn InputGuardrail>],
        input: &str,
    ) -> Result<()> {
        // Sort by priority (descending)
        let mut sorted_guards: Vec<_> = guardrails.iter().collect();
        sorted_guards.sort_by_key(|g| -g.priority());

        for guardrail in sorted_guards {
            let result = guardrail.check(input).await?;
            if !result.passed {
                return Err(AgentsError::InputGuardrailTriggered {
                    message: result
                        .reason
                        .unwrap_or_else(|| "Input validation failed".to_string()),
                });
            }
        }
        Ok(())
    }

    /// Executes a sequence of output guardrails.
    ///
    /// This method processes the output through each guardrail, allowing for
    /// modifications. If a guardrail fails, it returns an `Err` with an
    /// [`AgentsError::OutputGuardrailTriggered`] error.
    pub async fn check_output(
        guardrails: &[std::sync::Arc<dyn OutputGuardrail>],
        output: &str,
    ) -> Result<String> {
        let mut current_output = output.to_string();

        // Sort by priority (descending)
        let mut sorted_guards: Vec<_> = guardrails.iter().collect();
        sorted_guards.sort_by_key(|g| -g.priority());

        for guardrail in sorted_guards {
            let result = guardrail.check(&current_output).await?;

            if !result.passed {
                return Err(AgentsError::OutputGuardrailTriggered {
                    message: result
                        .reason
                        .unwrap_or_else(|| "Output validation failed".to_string()),
                });
            }

            if let Some(modified) = result.modified_content {
                current_output = modified;
            }
        }

        Ok(current_output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_guardrail_result() {
        let pass = GuardrailResult::pass();
        assert!(pass.passed);
        assert!(pass.reason.is_none());
        assert!(pass.modified_content.is_none());

        let fail = GuardrailResult::fail("Invalid input");
        assert!(!fail.passed);
        assert_eq!(fail.reason, Some("Invalid input".to_string()));

        let modified = GuardrailResult::pass_with_modification("cleaned text");
        assert!(modified.passed);
        assert_eq!(modified.modified_content, Some("cleaned text".to_string()));
    }

    #[tokio::test]
    async fn test_max_length_guardrail() {
        let guard = MaxLengthGuardrail::new(10);

        let short_result = guard.check("short").await.unwrap();
        assert!(short_result.passed);

        let long_result = guard.check("this is a very long input").await.unwrap();
        assert!(!long_result.passed);
        assert!(long_result
            .reason
            .unwrap()
            .contains("exceeds maximum length"));
    }

    #[tokio::test]
    async fn test_pattern_block_guardrail() {
        let guard = PatternBlockGuardrail::new(
            "ProfanityFilter",
            vec!["badword".to_string(), "forbidden".to_string()],
        );

        let clean_result = InputGuardrail::check(&guard, "This is clean text")
            .await
            .unwrap();
        assert!(clean_result.passed);

        let blocked_result = InputGuardrail::check(&guard, "This contains badword here")
            .await
            .unwrap();
        assert!(!blocked_result.passed);
        assert!(blocked_result.reason.unwrap().contains("blocked pattern"));

        // Test case insensitive
        let blocked_caps = InputGuardrail::check(&guard, "This has FORBIDDEN content")
            .await
            .unwrap();
        assert!(!blocked_caps.passed);
    }

    #[tokio::test]
    async fn test_guardrail_runner_input() {
        let guards: Vec<Arc<dyn InputGuardrail>> = vec![
            Arc::new(MaxLengthGuardrail::new(100)),
            Arc::new(PatternBlockGuardrail::new(
                "Filter",
                vec!["spam".to_string()],
            )),
        ];

        // Should pass
        let result = GuardrailRunner::check_input(&guards, "Valid input").await;
        assert!(result.is_ok());

        // Should fail on length
        let long_input = "x".repeat(200);
        let result = GuardrailRunner::check_input(&guards, &long_input).await;
        assert!(result.is_err());

        // Should fail on pattern
        let result = GuardrailRunner::check_input(&guards, "This is spam").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_guardrail_runner_output() {
        let guards: Vec<Arc<dyn OutputGuardrail>> = vec![Arc::new(PatternBlockGuardrail::new(
            "Filter",
            vec!["secret".to_string()],
        ))];

        // Should pass
        let result = GuardrailRunner::check_output(&guards, "Normal output").await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "Normal output");

        // Should fail
        let result = GuardrailRunner::check_output(&guards, "This is secret info").await;
        assert!(result.is_err());
    }

    // Test custom guardrail implementation
    #[derive(Debug)]
    struct CustomGuardrail {
        name: String,
        priority: i32,
    }

    #[async_trait]
    impl InputGuardrail for CustomGuardrail {
        fn name(&self) -> &str {
            &self.name
        }

        async fn check(&self, input: &str) -> Result<GuardrailResult> {
            if input.starts_with("!") {
                Ok(GuardrailResult::fail("Commands not allowed"))
            } else {
                Ok(GuardrailResult::pass())
            }
        }

        fn priority(&self) -> i32 {
            self.priority
        }
    }

    #[tokio::test]
    async fn test_custom_guardrail() {
        let guard = CustomGuardrail {
            name: "CommandBlocker".to_string(),
            priority: 10,
        };

        let result = guard.check("!command").await.unwrap();
        assert!(!result.passed);

        let result = guard.check("normal text").await.unwrap();
        assert!(result.passed);
    }

    #[tokio::test]
    async fn test_guardrail_priority_ordering() {
        // Create a guardrail that always fails
        #[derive(Debug)]
        struct AlwaysFailGuard {
            name: String,
            priority: i32,
        }

        #[async_trait]
        impl InputGuardrail for AlwaysFailGuard {
            fn name(&self) -> &str {
                &self.name
            }

            async fn check(&self, _: &str) -> Result<GuardrailResult> {
                Ok(GuardrailResult::fail(self.name.clone()))
            }

            fn priority(&self) -> i32 {
                self.priority
            }
        }

        let guards: Vec<Arc<dyn InputGuardrail>> = vec![
            Arc::new(AlwaysFailGuard {
                name: "LowPriority".to_string(),
                priority: 1,
            }),
            Arc::new(AlwaysFailGuard {
                name: "HighPriority".to_string(),
                priority: 10,
            }),
        ];

        let result = GuardrailRunner::check_input(&guards, "test").await;
        assert!(result.is_err());

        // Should fail with high priority guard first
        if let Err(AgentsError::InputGuardrailTriggered { message }) = result {
            assert_eq!(message, "HighPriority");
        } else {
            panic!("Expected InputGuardrailTriggered error");
        }
    }
}
