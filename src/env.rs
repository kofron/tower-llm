//! Capability-based environment for Tower layers.
//!
//! The Env system allows layers to access shared resources and capabilities
//! through trait implementations. This provides a type-safe, extensible way
//! to inject dependencies into the Tower service stack.

use std::any::Any;
use std::sync::Arc;

/// Base environment trait that all environments must implement.
pub trait Env: Clone + Send + Sync + 'static {
    /// Get a typed capability from this environment.
    ///
    /// Returns None if the capability is not available.
    fn capability<T: Any + Send + Sync>(&self) -> Option<Arc<T>>;

    /// Check if this environment provides a specific capability.
    fn has_capability<T: Any + Send + Sync>(&self) -> bool {
        self.capability::<T>().is_some()
    }
}

/// Default environment with no capabilities.
///
/// This is the simplest environment that provides no additional capabilities.
/// It's useful for basic tool execution that doesn't require any shared resources.
#[derive(Debug, Clone, Default)]
pub struct DefaultEnv;

impl Env for DefaultEnv {
    fn capability<T: Any + Send + Sync>(&self) -> Option<Arc<T>> {
        None
    }
}

/// Builder for creating environments with capabilities.
///
/// # Example
///
/// ```
/// use openai_agents_rs::env::{EnvBuilder, LoggingCapability};
/// use std::sync::Arc;
///
/// let env = EnvBuilder::new()
///     .with_capability(Arc::new(LoggingCapability::default()))
///     .build();
/// ```
pub struct EnvBuilder {
    capabilities: Vec<Arc<dyn Any + Send + Sync>>,
}

impl EnvBuilder {
    /// Create a new environment builder.
    pub fn new() -> Self {
        Self {
            capabilities: Vec::new(),
        }
    }

    /// Add a capability to the environment.
    pub fn with_capability<T: Any + Send + Sync>(mut self, capability: Arc<T>) -> Self {
        self.capabilities
            .push(capability as Arc<dyn Any + Send + Sync>);
        self
    }

    /// Build the environment.
    pub fn build(self) -> CapabilityEnv {
        CapabilityEnv {
            capabilities: Arc::new(self.capabilities),
        }
    }
}

impl Default for EnvBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Environment that provides capabilities through a type-erased store.
#[derive(Clone)]
pub struct CapabilityEnv {
    capabilities: Arc<Vec<Arc<dyn Any + Send + Sync>>>,
}

impl Env for CapabilityEnv {
    fn capability<T: Any + Send + Sync>(&self) -> Option<Arc<T>> {
        for cap in self.capabilities.iter() {
            if let Some(typed) = cap.clone().downcast::<T>().ok() {
                return Some(typed);
            }
        }
        None
    }
}

// ============================================================================
// Standard Capabilities
// ============================================================================

/// Capability for logging operations.
pub trait Logging: Send + Sync {
    /// Log a debug message.
    fn debug(&self, message: &str);

    /// Log an info message.
    fn info(&self, message: &str);

    /// Log a warning message.
    fn warn(&self, message: &str);

    /// Log an error message.
    fn error(&self, message: &str);
}

/// Default logging capability that uses tracing.
#[derive(Default)]
pub struct LoggingCapability;

impl Logging for LoggingCapability {
    fn debug(&self, message: &str) {
        tracing::debug!("{}", message);
    }

    fn info(&self, message: &str) {
        tracing::info!("{}", message);
    }

    fn warn(&self, message: &str) {
        tracing::warn!("{}", message);
    }

    fn error(&self, message: &str) {
        tracing::error!("{}", message);
    }
}

/// Capability for approval workflows.
pub trait Approval: Send + Sync {
    /// Request approval for an operation.
    ///
    /// Returns true if approved, false if denied.
    fn request_approval(&self, operation: &str, details: &str) -> bool;
}

/// Auto-approve capability that always approves.
#[derive(Default)]
pub struct AutoApprove;

impl Approval for AutoApprove {
    fn request_approval(&self, _operation: &str, _details: &str) -> bool {
        true
    }
}

/// Manual approval capability that prompts the user.
pub struct ManualApproval;

impl Approval for ManualApproval {
    fn request_approval(&self, operation: &str, details: &str) -> bool {
        println!("Approval requested for: {}", operation);
        println!("Details: {}", details);
        println!("Approve? (y/n): ");

        use std::io::{self, BufRead};
        let stdin = io::stdin();
        let mut lines = stdin.lock().lines();

        if let Some(Ok(line)) = lines.next() {
            line.trim().eq_ignore_ascii_case("y") || line.trim().eq_ignore_ascii_case("yes")
        } else {
            false
        }
    }
}

/// Capability for metrics collection.
pub trait Metrics: Send + Sync {
    /// Record a counter metric.
    fn increment(&self, name: &str, value: u64);

    /// Record a gauge metric.
    fn gauge(&self, name: &str, value: f64);

    /// Record a histogram metric.
    fn histogram(&self, name: &str, value: f64);
}

/// In-memory metrics collector for testing.
#[derive(Default)]
pub struct InMemoryMetrics {
    // In a real implementation, these would be proper metric stores
}

impl Metrics for InMemoryMetrics {
    fn increment(&self, name: &str, value: u64) {
        tracing::trace!("Metric increment: {} += {}", name, value);
    }

    fn gauge(&self, name: &str, value: f64) {
        tracing::trace!("Metric gauge: {} = {}", name, value);
    }

    fn histogram(&self, name: &str, value: f64) {
        tracing::trace!("Metric histogram: {} -> {}", name, value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_env_has_no_capabilities() {
        let env = DefaultEnv;
        assert!(!env.has_capability::<LoggingCapability>());
        assert!(env.capability::<LoggingCapability>().is_none());
    }

    #[test]
    fn test_capability_env_provides_capabilities() {
        let logger = Arc::new(LoggingCapability);
        let env = EnvBuilder::new().with_capability(logger.clone()).build();

        assert!(env.has_capability::<LoggingCapability>());
        assert!(env.capability::<LoggingCapability>().is_some());
    }

    #[test]
    fn test_multiple_capabilities() {
        let env = EnvBuilder::new()
            .with_capability(Arc::new(LoggingCapability))
            .with_capability(Arc::new(AutoApprove))
            .with_capability(Arc::new(InMemoryMetrics::default()))
            .build();

        assert!(env.has_capability::<LoggingCapability>());
        assert!(env.has_capability::<AutoApprove>());
        assert!(env.has_capability::<InMemoryMetrics>());
    }

    #[test]
    fn test_capability_usage() {
        let env = EnvBuilder::new()
            .with_capability(Arc::new(AutoApprove))
            .build();

        if let Some(approval) = env.capability::<AutoApprove>() {
            assert!(approval.request_approval("test", "testing"));
        } else {
            panic!("Should have approval capability");
        }
    }
}
