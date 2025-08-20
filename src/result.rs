//! Result types for agent execution

use serde_json::Value;

/// The result of running an agent.
#[derive(Debug, Clone)]
pub struct RunResult {
    /// The final output from the agent
    pub final_output: Value,
    /// Whether the run was successful
    pub success: bool,
    /// Error message if the run failed
    pub error: Option<String>,
}

impl RunResult {
    /// Check if the run was successful
    pub fn is_success(&self) -> bool {
        self.success
    }

    /// Get the error if the run failed
    pub fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }
}

/// Extended result with additional context
pub type RunResultWithContext = RunResult;

/// Result for streaming operations
pub type StreamingRunResult = RunResult;
