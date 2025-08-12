//! # Retry Mechanism with Exponential Backoff
//!
//! This module provides a robust and flexible retry mechanism for handling
//! transient failures, such as network errors or temporary API unavailability.
//! It implements an exponential backoff strategy with optional jitter to prevent
//! thundering herd issues.
//!
//! ## Core Components
//!
//! - **[`RetryPolicy`]**: A struct that encapsulates the configuration for a retry
//!   strategy, including the maximum number of retries, initial and maximum
//!   delays, and backoff multiplier.
//! - **[`RetryBuilder`]**: A fluent builder for creating and executing retryable
//!   operations.
//! - **`retry_async` and `retry_sync`**: Functions that wrap an operation in retry
//!   logic, for asynchronous and synchronous code respectively.
//!
//! ## Usage
//!
//! The `RetryBuilder` provides the most convenient way to use the retry logic.
//! You can configure the retry policy and then execute an async or sync operation.
//!
//! ### Example: Retrying an Asynchronous Operation
//!
//! ```rust
//! use openai_agents_rs::retry::RetryBuilder;
//! use openai_agents_rs::error::{Result, AgentsError};
//! use std::sync::atomic::{AtomicUsize, Ordering};
//! use std::sync::Arc;
//!
//! # #[tokio::main]
//! # async fn main() {
//! let counter = Arc::new(AtomicUsize::new(0));
//!
//! let result = RetryBuilder::new()
//!     .max_retries(3)
//!     .run_async(|| {
//!         let counter = counter.clone();
//!         async move {
//!             let count = counter.fetch_add(1, Ordering::SeqCst);
//!             if count < 2 {
//!                 Err(AgentsError::IoError(
//!                     std::io::Error::new(std::io::ErrorKind::TimedOut, "timeout")
//!                 ))
//!             } else {
//!                 Ok("Success!")
//!             }
//!         }
//!     })
//!     .await;
//!
//! assert!(result.is_ok());
//! assert_eq!(result.unwrap(), "Success!");
//! assert_eq!(counter.load(Ordering::SeqCst), 3); // 2 failures + 1 success
//! # }
//! ```
//! Retry mechanism with exponential backoff
//!
//! Provides robust retry logic for handling transient failures.

use crate::config::RetryConfig;
use crate::error::{AgentsError, Result};
use std::future::Future;
use std::time::Duration;
use tokio::time::sleep;
use tracing::{debug, warn};

/// Encapsulates the configuration and state for a retry strategy.
///
/// `RetryPolicy` manages the retry logic, including tracking the number of
/// attempts and calculating the delay for the next retry using exponential
/// backoff.
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    config: RetryConfig,
    attempt: usize,
    next_delay: Duration,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self::new(RetryConfig::default())
    }
}

impl RetryPolicy {
    /// Creates a new `RetryPolicy` with the given configuration.
    pub fn new(config: RetryConfig) -> Self {
        Self {
            next_delay: config.initial_delay,
            config,
            attempt: 0,
        }
    }

    /// Returns `true` if the operation should be retried based on the maximum
    /// number of attempts.
    pub fn should_retry(&self) -> bool {
        self.attempt < self.config.max_retries
    }

    /// Returns the current attempt number (0-indexed).
    pub fn attempt(&self) -> usize {
        self.attempt
    }

    /// Calculates and returns the delay for the next retry.
    ///
    /// This method implements exponential backoff with optional jitter. It also
    /// increments the internal attempt counter.
    pub fn next_delay(&mut self) -> Duration {
        let mut delay = self.next_delay;

        // Add jitter if enabled
        if self.config.jitter {
            use rand::Rng;
            let mut rng = rand::thread_rng();
            let jitter = rng.gen_range(0.0..0.3);
            let jitter_ms = (delay.as_millis() as f64 * jitter) as u64;
            delay += Duration::from_millis(jitter_ms);
        }

        // Update for next iteration
        self.attempt += 1;
        self.next_delay = Duration::from_secs_f32(
            (self.next_delay.as_secs_f32() * self.config.backoff_multiplier)
                .min(self.config.max_delay.as_secs_f32()),
        );

        delay
    }

    /// Resets the retry policy to its initial state.
    pub fn reset(&mut self) {
        self.attempt = 0;
        self.next_delay = self.config.initial_delay;
    }
}

/// Determines whether a given `AgentsError` is considered retryable.
///
/// This function is used by the retry logic to decide whether to attempt an
/// operation again. Generally, network-related and transient errors are
/// considered retryable, while logic errors are not.
pub fn is_retryable(error: &AgentsError) -> bool {
    match error {
        AgentsError::OpenAIError(_) => true, // API errors are often transient
        AgentsError::IoError(_) => true,     // Network issues
        AgentsError::DatabaseError(_) => true, // Database connection issues
        AgentsError::ToolExecutionError { .. } => true, // Tool failures might be transient
        AgentsError::MaxTurnsExceeded { .. } => false,
        AgentsError::InputGuardrailTriggered { .. } => false,
        AgentsError::OutputGuardrailTriggered { .. } => false,
        AgentsError::UserError { .. } => false,
        _ => false,
    }
}

/// Wraps an asynchronous operation with retry logic.
///
/// This function will repeatedly call the `operation` closure until it succeeds
/// or the retry policy is exhausted.
pub async fn retry_async<F, Fut, T>(mut operation: F, policy: &mut RetryPolicy) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T>>,
{
    loop {
        match operation().await {
            Ok(result) => {
                if policy.attempt() > 0 {
                    debug!(
                        "Operation succeeded after {} attempts",
                        policy.attempt() + 1
                    );
                }
                return Ok(result);
            }
            Err(error) => {
                if !is_retryable(&error) {
                    debug!("Non-retryable error: {}", error);
                    return Err(error);
                }

                if !policy.should_retry() {
                    warn!(
                        "Max retries ({}) exceeded. Last error: {}",
                        policy.config.max_retries, error
                    );
                    return Err(error);
                }

                let delay = policy.next_delay();
                warn!(
                    "Attempt {} failed: {}. Retrying in {:?}...",
                    policy.attempt(),
                    error,
                    delay
                );

                sleep(delay).await;
            }
        }
    }
}

/// Wraps a synchronous operation with retry logic.
///
/// This function is the synchronous equivalent of `retry_async`.
pub fn retry_sync<F, T>(mut operation: F, policy: &mut RetryPolicy) -> Result<T>
where
    F: FnMut() -> Result<T>,
{
    loop {
        match operation() {
            Ok(result) => {
                if policy.attempt() > 0 {
                    debug!(
                        "Operation succeeded after {} attempts",
                        policy.attempt() + 1
                    );
                }
                return Ok(result);
            }
            Err(error) => {
                if !is_retryable(&error) {
                    debug!("Non-retryable error: {}", error);
                    return Err(error);
                }

                if !policy.should_retry() {
                    warn!(
                        "Max retries ({}) exceeded. Last error: {}",
                        policy.config.max_retries, error
                    );
                    return Err(error);
                }

                let delay = policy.next_delay();
                warn!(
                    "Attempt {} failed: {}. Retrying in {:?}...",
                    policy.attempt(),
                    error,
                    delay
                );

                std::thread::sleep(delay);
            }
        }
    }
}

/// A builder for creating and executing retryable operations.
///
/// `RetryBuilder` provides a fluent interface for configuring the retry policy
/// and running an operation with that policy.
pub struct RetryBuilder {
    policy: RetryPolicy,
}

impl Default for RetryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl RetryBuilder {
    /// Creates a new `RetryBuilder` with a default `RetryPolicy`.
    pub fn new() -> Self {
        Self {
            policy: RetryPolicy::default(),
        }
    }

    /// Sets the maximum number of retries.
    pub fn max_retries(mut self, max: usize) -> Self {
        self.policy.config.max_retries = max;
        self
    }

    /// Sets the initial delay for the first retry.
    pub fn initial_delay(mut self, delay: Duration) -> Self {
        self.policy.config.initial_delay = delay;
        self.policy.next_delay = delay;
        self
    }

    /// Sets the maximum delay between retries.
    pub fn max_delay(mut self, delay: Duration) -> Self {
        self.policy.config.max_delay = delay;
        self
    }

    /// Sets the backoff multiplier for increasing the delay between retries.
    pub fn backoff_multiplier(mut self, multiplier: f32) -> Self {
        self.policy.config.backoff_multiplier = multiplier;
        self
    }

    /// Enables or disables jitter.
    pub fn with_jitter(mut self, enabled: bool) -> Self {
        self.policy.config.jitter = enabled;
        self
    }

    /// Executes an asynchronous operation with the configured retry policy.
    pub async fn run_async<F, Fut, T>(mut self, operation: F) -> Result<T>
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = Result<T>>,
    {
        retry_async(operation, &mut self.policy).await
    }

    /// Executes a synchronous operation with the configured retry policy.
    pub fn run_sync<F, T>(mut self, operation: F) -> Result<T>
    where
        F: FnMut() -> Result<T>,
    {
        retry_sync(operation, &mut self.policy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[test]
    fn test_retry_policy() {
        let config = RetryConfig {
            max_retries: 3,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(1),
            backoff_multiplier: 2.0,
            jitter: false,
        };

        let mut policy = RetryPolicy::new(config);

        assert!(policy.should_retry());
        assert_eq!(policy.attempt(), 0);

        let delay1 = policy.next_delay();
        assert_eq!(delay1, Duration::from_millis(100));
        assert_eq!(policy.attempt(), 1);

        let delay2 = policy.next_delay();
        // Use approximate comparison due to floating point precision
        assert!((delay2.as_millis() as i64 - 200).abs() <= 1);
        assert_eq!(policy.attempt(), 2);

        let delay3 = policy.next_delay();
        // Use approximate comparison due to floating point precision
        assert!((delay3.as_millis() as i64 - 400).abs() <= 1);
        assert_eq!(policy.attempt(), 3);

        assert!(!policy.should_retry());
    }

    #[test]
    fn test_is_retryable() {
        assert!(is_retryable(&AgentsError::IoError(std::io::Error::new(
            std::io::ErrorKind::TimedOut,
            "timeout"
        ))));

        assert!(!is_retryable(&AgentsError::MaxTurnsExceeded {
            max_turns: 5
        }));

        assert!(!is_retryable(&AgentsError::UserError {
            message: "Invalid input".to_string()
        }));
    }

    #[test]
    fn test_retry_sync_success() {
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();

        let mut policy = RetryPolicy::new(RetryConfig {
            max_retries: 3,
            initial_delay: Duration::from_millis(10),
            max_delay: Duration::from_millis(100),
            backoff_multiplier: 2.0,
            jitter: false,
        });

        let result = retry_sync(
            || {
                let count = counter_clone.fetch_add(1, Ordering::SeqCst);
                if count < 2 {
                    Err(AgentsError::IoError(std::io::Error::new(
                        std::io::ErrorKind::TimedOut,
                        "timeout",
                    )))
                } else {
                    Ok(42)
                }
            },
            &mut policy,
        );

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
        assert_eq!(counter.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn test_retry_async_max_retries() {
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();

        let mut policy = RetryPolicy::new(RetryConfig {
            max_retries: 2,
            initial_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(10),
            backoff_multiplier: 2.0,
            jitter: false,
        });

        let result = retry_async(
            || {
                let counter = counter_clone.clone();
                async move {
                    counter.fetch_add(1, Ordering::SeqCst);
                    Err::<i32, _>(AgentsError::IoError(std::io::Error::new(
                        std::io::ErrorKind::TimedOut,
                        "timeout",
                    )))
                }
            },
            &mut policy,
        )
        .await;

        assert!(result.is_err());
        assert_eq!(counter.load(Ordering::SeqCst), 3); // Initial + 2 retries
    }

    #[tokio::test]
    async fn test_retry_builder() {
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();

        let result = RetryBuilder::new()
            .max_retries(5)
            .initial_delay(Duration::from_millis(1))
            .with_jitter(false)
            .run_async(|| {
                let counter = counter_clone.clone();
                async move {
                    let count = counter.fetch_add(1, Ordering::SeqCst);
                    if count == 0 {
                        Err(AgentsError::IoError(std::io::Error::new(
                            std::io::ErrorKind::TimedOut,
                            "timeout",
                        )))
                    } else {
                        Ok("success")
                    }
                }
            })
            .await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "success");
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }
}
