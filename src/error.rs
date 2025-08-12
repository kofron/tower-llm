//! Error types for the Agents SDK

use thiserror::Error;

/// Result type alias for the Agents SDK
pub type Result<T> = std::result::Result<T, AgentsError>;

/// Main error type for the Agents SDK
#[derive(Debug, Error)]
pub enum AgentsError {
    /// Error from the OpenAI API
    #[error("OpenAI API error: {0}")]
    OpenAIError(#[from] async_openai::error::OpenAIError),

    /// Maximum turns exceeded
    #[error("Maximum turns exceeded: {max_turns}")]
    MaxTurnsExceeded { max_turns: usize },

    /// Input guardrail triggered
    #[error("Input guardrail triggered: {message}")]
    InputGuardrailTriggered { message: String },

    /// Output guardrail triggered
    #[error("Output guardrail triggered: {message}")]
    OutputGuardrailTriggered { message: String },

    /// Tool execution error
    #[error("Tool execution error: {message}")]
    ToolExecutionError { message: String },

    /// Handoff error
    #[error("Handoff error: {message}")]
    HandoffError { message: String },

    /// Model behavior error
    #[error("Model behavior error: {message}")]
    ModelBehaviorError { message: String },

    /// User error
    #[error("User error: {message}")]
    UserError { message: String },

    /// Session error
    #[error("Session error: {0}")]
    SessionError(String),

    /// Serialization/deserialization error
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Database error
    #[error("Database error: {0}")]
    DatabaseError(#[from] sqlx::Error),

    /// Other errors
    #[error("{0}")]
    Other(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = AgentsError::MaxTurnsExceeded { max_turns: 10 };
        assert_eq!(err.to_string(), "Maximum turns exceeded: 10");

        let err = AgentsError::InputGuardrailTriggered {
            message: "Inappropriate content".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "Input guardrail triggered: Inappropriate content"
        );
    }

    #[test]
    fn test_error_from_openai() {
        // This tests that the From trait is properly implemented
        let openai_err = async_openai::error::OpenAIError::InvalidArgument("test".to_string());
        let agents_err: AgentsError = openai_err.into();
        assert!(matches!(agents_err, AgentsError::OpenAIError(_)));
    }

    #[test]
    fn test_result_type() {
        fn example_function() -> Result<String> {
            Ok("success".to_string())
        }

        let result = example_function();
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "success");
    }

    #[test]
    fn test_error_chaining() {
        fn might_fail() -> Result<()> {
            Err(AgentsError::UserError {
                message: "Something went wrong".to_string(),
            })
        }

        let result = might_fail();
        assert!(result.is_err());

        if let Err(e) = result {
            assert!(matches!(e, AgentsError::UserError { .. }));
        }
    }
}
