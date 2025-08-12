//! Guardrails for input and output validation
//!
//! Guardrails provide safety checks to validate user input and agent output.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use crate::error::{AgentsError, Result};

/// Result of a guardrail check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuardrailResult {
    /// Whether the check passed
    pub passed: bool,

    /// Reason if the check failed
    pub reason: Option<String>,

    /// Optional modified content (for output guardrails)
    pub modified_content: Option<String>,
}

impl GuardrailResult {
    /// Create a passing result
    pub fn pass() -> Self {
        Self {
            passed: true,
            reason: None,
            modified_content: None,
        }
    }

    /// Create a failing result with a reason
    pub fn fail(reason: impl Into<String>) -> Self {
        Self {
            passed: false,
            reason: Some(reason.into()),
            modified_content: None,
        }
    }

    /// Create a passing result with modified content
    pub fn pass_with_modification(content: impl Into<String>) -> Self {
        Self {
            passed: true,
            reason: None,
            modified_content: Some(content.into()),
        }
    }
}

/// Trait for input guardrails
#[async_trait]
pub trait InputGuardrail: Send + Sync + Debug {
    /// Name of the guardrail
    fn name(&self) -> &str;

    /// Check if the input is valid
    async fn check(&self, input: &str) -> Result<GuardrailResult>;

    /// Priority of this guardrail (higher = checked first)
    fn priority(&self) -> i32 {
        0
    }
}

/// Trait for output guardrails
#[async_trait]
pub trait OutputGuardrail: Send + Sync + Debug {
    /// Name of the guardrail
    fn name(&self) -> &str;

    /// Check if the output is valid, optionally modifying it
    async fn check(&self, output: &str) -> Result<GuardrailResult>;

    /// Priority of this guardrail (higher = checked first)
    fn priority(&self) -> i32 {
        0
    }
}

/// Simple length-based input guardrail
#[derive(Debug, Clone)]
pub struct MaxLengthGuardrail {
    name: String,
    max_length: usize,
}

impl MaxLengthGuardrail {
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

/// Pattern-based content filter
#[derive(Debug, Clone)]
pub struct PatternBlockGuardrail {
    name: String,
    patterns: Vec<String>,
}

impl PatternBlockGuardrail {
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

/// Helper to run multiple guardrails
pub struct GuardrailRunner;

impl GuardrailRunner {
    /// Run input guardrails, returning error if any fail
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

    /// Run output guardrails, returning the (possibly modified) output
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
