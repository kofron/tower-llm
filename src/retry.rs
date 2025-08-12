//! Retry mechanism with exponential backoff
//!
//! Provides robust retry logic for handling transient failures.

use crate::config::RetryConfig;
use crate::error::{AgentsError, Result};
use std::future::Future;
use std::time::Duration;
use tokio::time::sleep;
use tracing::{debug, warn};

/// Retry policy for operations
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    config: RetryConfig,
    attempt: usize,
    next_delay: Duration,
}

impl RetryPolicy {
    /// Create a new retry policy
    pub fn new(config: RetryConfig) -> Self {
        Self {
            next_delay: config.initial_delay,
            config,
            attempt: 0,
        }
    }

    /// Create a default retry policy
    pub fn default() -> Self {
        Self::new(RetryConfig::default())
    }

    /// Check if we should retry
    pub fn should_retry(&self) -> bool {
        self.attempt < self.config.max_retries
    }

    /// Get the current attempt number
    pub fn attempt(&self) -> usize {
        self.attempt
    }

    /// Calculate next delay with exponential backoff
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

    /// Reset the retry policy
    pub fn reset(&mut self) {
        self.attempt = 0;
        self.next_delay = self.config.initial_delay;
    }
}

/// Determine if an error is retryable
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

/// Retry an async operation with exponential backoff
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

/// Retry a synchronous operation
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

/// Builder for retry operations
pub struct RetryBuilder {
    policy: RetryPolicy,
}

impl Default for RetryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl RetryBuilder {
    pub fn new() -> Self {
        Self {
            policy: RetryPolicy::default(),
        }
    }

    pub fn max_retries(mut self, max: usize) -> Self {
        self.policy.config.max_retries = max;
        self
    }

    pub fn initial_delay(mut self, delay: Duration) -> Self {
        self.policy.config.initial_delay = delay;
        self.policy.next_delay = delay;
        self
    }

    pub fn max_delay(mut self, delay: Duration) -> Self {
        self.policy.config.max_delay = delay;
        self
    }

    pub fn backoff_multiplier(mut self, multiplier: f32) -> Self {
        self.policy.config.backoff_multiplier = multiplier;
        self
    }

    pub fn with_jitter(mut self, enabled: bool) -> Self {
        self.policy.config.jitter = enabled;
        self
    }

    pub async fn run_async<F, Fut, T>(mut self, operation: F) -> Result<T>
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = Result<T>>,
    {
        retry_async(operation, &mut self.policy).await
    }

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
