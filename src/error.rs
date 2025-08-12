//! # Error Handling for the Agents SDK
//!
//! This module defines the centralized error handling system for the SDK. It
//! provides a unified `Result` type and a comprehensive `AgentsError` enum that
//! covers all potential issues that can arise during an agent's execution.
//!
//! ## The `AgentsError` Enum
//!
//! The [`AgentsError`] enum is the primary error type used throughout the SDK. It
//! encapsulates a wide range of possible failures, from API-level errors to
//! internal logic issues like guardrail violations or tool execution failures.
//! The use of `thiserror` allows for clean and descriptive error messages.
//!
//! ## The `Result` Type Alias
//!
//! For convenience, this module provides a `Result<T>` type alias, which is a
//! shorthand for `std::result::Result<T, AgentsError>`. This simplifies function
//! signatures throughout the codebase and ensures consistent error handling.
//!
//! ### Example: Using the Custom `Result` Type
//!
//! ```rust
//! use openai_agents_rs::error::{Result, AgentsError};
//!
//! fn check_input(input: &str) -> Result<()> {
//!     if input.is_empty() {
//!         Err(AgentsError::UserError {
//!             message: "Input cannot be empty.".to_string(),
//!         })
//!     } else {
//!         Ok(())
//!     }
//! }
//!
//! assert!(check_input("Hello").is_ok());
//! let error = check_input("").unwrap_err();
//! assert_eq!(error.to_string(), "User error: Input cannot be empty.");
//! ```
//! Error types for the Agents SDK

use thiserror::Error;

/// A specialized `Result` type for the Agents SDK.
///
/// This type alias simplifies function signatures by providing a default
/// error type of [`AgentsError`].
pub type Result<T> = std::result::Result<T, AgentsError>;

/// The main error enum for the Agents SDK.
///
/// `AgentsError` consolidates all possible errors that can occur within the
/// SDK into a single, comprehensive type. This allows for robust and centralized
/// error handling.
#[derive(Debug, Error)]
pub enum AgentsError {
    /// An error originating from the underlying `async-openai` crate.
    #[error("OpenAI API error: {0}")]
    OpenAIError(#[from] async_openai::error::OpenAIError),

    /// Indicates that the maximum number of turns for an agent run has been
    /// exceeded, preventing infinite loops.
    #[error("Maximum turns exceeded: {max_turns}")]
    MaxTurnsExceeded { max_turns: usize },

    /// An input guardrail was triggered, indicating that the user's input
    /// violated a predefined constraint.
    #[error("Input guardrail triggered: {message}")]
    InputGuardrailTriggered { message: String },

    /// An output guardrail was triggered, indicating that the agent's response
    /// violated a predefined constraint.
    #[error("Output guardrail triggered: {message}")]
    OutputGuardrailTriggered { message: String },

    /// An error occurred during the execution of a tool.
    #[error("Tool execution error: {message}")]
    ToolExecutionError { message: String },

    /// An error occurred during a handoff between agents.
    #[error("Handoff error: {message}")]
    HandoffError { message: String },

    /// An error indicating unexpected or invalid behavior from the LLM.
    #[error("Model behavior error: {message}")]
    ModelBehaviorError { message: String },

    /// An error caused by invalid user input or configuration.
    #[error("User error: {message}")]
    UserError { message: String },

    /// An error related to session management, such as a failure to read or
    /// write to the session store.
    #[error("Session error: {0}")]
    SessionError(String),

    /// An error that occurred during JSON serialization or deserialization.
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    /// An I/O error, typically related to file system operations.
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// An error from the database, used by persistent session stores.
    #[error("Database error: {0}")]
    DatabaseError(#[from] sqlx::Error),

    /// A catch-all for any other type of error.
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
