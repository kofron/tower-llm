//! # Guardrails (orientation)
//!
//! Input and output guardrails provide lightweight, pluggable validation and
//! shaping for agent I/O. They are defined as traits and executed by the
//! `GuardrailRunner` in priority order. Use input guardrails to validate user
//! input, and output guardrails to enforce formats or sanitize responses.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::error::{AgentsError, Result};

/// Represents the outcome of a guardrail check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuardrailResult {
    pub passed: bool,
    pub reason: Option<String>,
}

/// Trait for input guardrails that validate user input before processing.
#[async_trait]
pub trait InputGuardrail: Send + Sync {
    fn name(&self) -> &str;
    fn priority(&self) -> i32 {
        0
    }
    async fn check(&self, input: &str) -> Result<GuardrailResult>;
}

/// Trait for output guardrails that validate/sanitize agent responses.
#[async_trait]
pub trait OutputGuardrail: Send + Sync {
    fn name(&self) -> &str;
    fn priority(&self) -> i32 {
        0
    }
    async fn check(&self, output: &str) -> Result<GuardrailResult>;
}

/// Executes guardrails in descending priority order.
pub struct GuardrailRunner;

impl GuardrailRunner {
    pub async fn check_input(guards: &[Arc<dyn InputGuardrail>], input: &str) -> Result<()> {
        let mut guards = guards.to_vec();
        guards.sort_by_key(|g| -g.priority());
        for g in guards {
            let res = g.check(input).await?;
            if !res.passed {
                return Err(AgentsError::InputGuardrailTriggered {
                    message: res.reason.unwrap_or_else(|| g.name().to_string()),
                });
            }
        }
        Ok(())
    }

    pub async fn check_output(guards: &[Arc<dyn OutputGuardrail>], output: &str) -> Result<String> {
        let mut guards = guards.to_vec();
        guards.sort_by_key(|g| -g.priority());
        let out = output.to_string();
        for g in guards {
            let res = g.check(&out).await?;
            if !res.passed {
                return Err(AgentsError::OutputGuardrailTriggered {
                    message: res.reason.unwrap_or_else(|| g.name().to_string()),
                });
            }
        }
        Ok(out)
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
            Ok(GuardrailResult {
                passed: false,
                reason: Some(format!(
                    "Input exceeds maximum length of {} characters",
                    self.max_length
                )),
            })
        } else {
            Ok(GuardrailResult {
                passed: true,
                reason: None,
            })
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
                return Ok(GuardrailResult {
                    passed: false,
                    reason: Some(format!("Input contains blocked pattern: {}", pattern)),
                });
            }
        }
        Ok(GuardrailResult {
            passed: true,
            reason: None,
        })
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
                return Ok(GuardrailResult {
                    passed: false,
                    reason: Some(format!("Output contains blocked pattern: {}", pattern)),
                });
            }
        }
        Ok(GuardrailResult {
            passed: true,
            reason: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_guardrail_result() {
        let pass = GuardrailResult {
            passed: true,
            reason: None,
        };
        assert!(pass.passed);
        assert!(pass.reason.is_none());

        let fail = GuardrailResult {
            passed: false,
            reason: Some("Invalid input".to_string()),
        };
        assert!(!fail.passed);
        assert_eq!(fail.reason, Some("Invalid input".to_string()));
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
                Ok(GuardrailResult {
                    passed: false,
                    reason: Some("Commands not allowed".to_string()),
                })
            } else {
                Ok(GuardrailResult {
                    passed: true,
                    reason: None,
                })
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
                Ok(GuardrailResult {
                    passed: false,
                    reason: Some(self.name.clone()),
                })
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
