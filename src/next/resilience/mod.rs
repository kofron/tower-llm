//! Resilience layers: timeout, retry, rate-limit, circuit-breaker
//!
//! What this module provides (spec)
//! - Cross-cutting, reusable Tower middleware for reliability under failure and load
//!
//! Exports
//! - Models
//!   - `RetryPolicy { max_retries, backoff: Backoff }`
//!   - `RateLimit { qps, burst }`
//!   - `BreakerConfig { failure_threshold, window, reset_timeout }`
//!   - `ErrorKind` and classifier function `fn classify(&Error) -> ErrorKind`
//! - Layers
//!   - `TimeoutLayer(Duration)` (thin wrapper around `tower::timeout::Timeout`)
//!   - `RetryLayer<Classifier, Policy>`
//!   - `RateLimitLayer` (token bucket)
//!   - `CircuitBreakerLayer` (stateful gate)
//! - Utils
//!   - Backoff builders (fixed, exponential, jitter), default classifiers
//!
//! Implementation strategy
//! - Timeout: use `tower::timeout` directly; just re-expose with our config type
//! - Retry: wrap inner service; on transient errors per classifier, retry with backoff
//! - Rate limit: keep a token bucket (Arc<Mutex>) and check/consume tokens per request
//! - Circuit breaker: track success/failure counts in a sliding window; on open, short-circuit with error; half-open lets a probe through
//!
//! Composition
//! - Builder sugar chooses where to apply: model only, tools only, or entire step/agent
//! - Example: `ServiceBuilder::new().layer(TimeoutLayer::new(dur)).layer(RetryLayer::new(policy)).service(step)`
//!
//! Testing strategy
//! - Use fake services that error in a scripted pattern (e.g., E E S) and assert retry timing and counts
//! - For breaker: simulate sustained failures, confirm open/half-open transitions with timers
//! - Rate limit: simulate bursts to validate token consumption and backoff behavior


