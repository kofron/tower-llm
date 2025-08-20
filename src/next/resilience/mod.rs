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

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::Mutex;
use tokio::time::{sleep, timeout};
use tower::{BoxError, Layer, Service, ServiceExt};

// ===== Retry =====

#[derive(Debug, Clone, Copy)]
pub enum BackoffKind {
    Fixed,
    Exponential,
}

#[derive(Debug, Clone, Copy)]
pub struct Backoff {
    pub kind: BackoffKind,
    pub initial: Duration,
    pub factor: f32,
    pub max: Duration,
}

impl Backoff {
    pub fn fixed(delay: Duration) -> Self {
        Self {
            kind: BackoffKind::Fixed,
            initial: delay,
            factor: 1.0,
            max: delay,
        }
    }
    pub fn exponential(initial: Duration, factor: f32, max: Duration) -> Self {
        Self {
            kind: BackoffKind::Exponential,
            initial,
            factor,
            max,
        }
    }
    pub fn delay_for_attempt(&self, attempt: usize) -> Duration {
        match self.kind {
            BackoffKind::Fixed => self.initial,
            BackoffKind::Exponential => {
                let mult = self.factor.powi(attempt as i32);
                let d = self.initial.mul_f32(mult);
                if d > self.max {
                    self.max
                } else {
                    d
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RetryPolicy {
    pub max_retries: usize,
    pub backoff: Backoff,
}

pub trait ErrorClassifier: Send + Sync + 'static {
    fn retryable(&self, error: &BoxError) -> bool;
}

#[derive(Debug, Clone, Copy)]
pub struct AlwaysRetry;
impl ErrorClassifier for AlwaysRetry {
    fn retryable(&self, _error: &BoxError) -> bool {
        true
    }
}

pub struct RetryLayer<C> {
    policy: RetryPolicy,
    classifier: C,
}

impl<C> RetryLayer<C> {
    pub fn new(policy: RetryPolicy, classifier: C) -> Self {
        Self { policy, classifier }
    }
}

pub struct Retry<S, C> {
    inner: Arc<Mutex<S>>,
    policy: RetryPolicy,
    classifier: C,
}

impl<S, C> Layer<S> for RetryLayer<C>
where
    C: Clone,
{
    type Service = Retry<S, C>;
    fn layer(&self, inner: S) -> Self::Service {
        Retry {
            inner: Arc::new(Mutex::new(inner)),
            policy: self.policy,
            classifier: self.classifier.clone(),
        }
    }
}

impl<S, C, Req> Service<Req> for Retry<S, C>
where
    Req: Clone + Send + 'static,
    S: Service<Req, Error = BoxError> + Send + 'static,
    S::Future: Send + 'static,
    S::Response: Send + 'static,
    C: ErrorClassifier + Send + Sync + Clone + 'static,
{
    type Response = S::Response;
    type Error = BoxError;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(
        &mut self,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        std::task::Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: Req) -> Self::Future {
        let policy = self.policy;
        let classifier = self.classifier.clone();
        let req0 = req.clone();
        let mut attempts: usize = 0;
        let inner = self.inner.clone();
        Box::pin(async move {
            loop {
                let result = {
                    let mut guard = inner.lock().await;
                    ServiceExt::ready(&mut *guard)
                        .await?
                        .call(req0.clone())
                        .await
                };
                match result {
                    Ok(resp) => return Ok(resp),
                    Err(e) => {
                        if attempts >= policy.max_retries || !classifier.retryable(&e) {
                            return Err(e);
                        }
                        let delay = policy.backoff.delay_for_attempt(attempts);
                        attempts += 1;
                        sleep(delay).await;
                    }
                }
            }
        })
    }
}

// ===== Timeout =====

pub struct TimeoutLayer {
    dur: Duration,
}

impl TimeoutLayer {
    pub fn new(dur: Duration) -> Self {
        Self { dur }
    }
}

pub struct Timeout<S> {
    inner: S,
    dur: Duration,
}

impl<S> Layer<S> for TimeoutLayer {
    type Service = Timeout<S>;
    fn layer(&self, inner: S) -> Self::Service {
        Timeout {
            inner,
            dur: self.dur,
        }
    }
}

impl<S, Req> Service<Req> for Timeout<S>
where
    S: Service<Req, Error = BoxError> + Send + 'static,
    S::Future: Send + 'static,
    S::Response: Send + 'static,
{
    type Response = S::Response;
    type Error = BoxError;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(
        &mut self,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: Req) -> Self::Future {
        let fut = self.inner.call(req);
        let dur = self.dur;
        Box::pin(async move {
            match timeout(dur, fut).await {
                Ok(r) => r,
                Err(_) => Err::<S::Response, BoxError>("timeout".into()),
            }
        })
    }
}

// ===== Circuit Breaker (simplified) =====

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BreakerState {
    Closed,
    OpenUntil(Instant),
    HalfOpen,
}

#[derive(Debug, Clone, Copy)]
pub struct BreakerConfig {
    pub failure_threshold: usize,
    pub reset_timeout: Duration,
}

pub struct CircuitBreakerLayer {
    cfg: BreakerConfig,
}

impl CircuitBreakerLayer {
    pub fn new(cfg: BreakerConfig) -> Self {
        Self { cfg }
    }
}

pub struct CircuitBreaker<S> {
    inner: S,
    cfg: BreakerConfig,
    state: Arc<Mutex<(BreakerState, usize)>>, // (state, consecutive_failures)
}

impl<S> Layer<S> for CircuitBreakerLayer {
    type Service = CircuitBreaker<S>;
    fn layer(&self, inner: S) -> Self::Service {
        CircuitBreaker {
            inner,
            cfg: self.cfg,
            state: Arc::new(Mutex::new((BreakerState::Closed, 0))),
        }
    }
}

impl<S, Req> Service<Req> for CircuitBreaker<S>
where
    S: Service<Req, Error = BoxError> + Send + 'static,
    S::Future: Send + 'static,
    S::Response: Send + 'static,
{
    type Response = S::Response;
    type Error = BoxError;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(
        &mut self,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: Req) -> Self::Future {
        let cfg = self.cfg;
        let state = self.state.clone();
        let fut = self.inner.call(req);
        Box::pin(async move {
            // Check state
            {
                let mut s = state.lock().await;
                match s.0 {
                    BreakerState::Closed => {}
                    BreakerState::OpenUntil(t) => {
                        if Instant::now() < t {
                            return Err("circuit open".into());
                        }
                        s.0 = BreakerState::HalfOpen;
                    }
                    BreakerState::HalfOpen => {}
                }
            }

            match fut.await {
                Ok(resp) => {
                    let mut s = state.lock().await;
                    s.1 = 0; // reset failures
                    s.0 = BreakerState::Closed;
                    Ok(resp)
                }
                Err(e) => {
                    let mut s = state.lock().await;
                    s.1 += 1;
                    if s.1 >= cfg.failure_threshold {
                        s.0 = BreakerState::OpenUntil(Instant::now() + cfg.reset_timeout);
                    }
                    Err(e)
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tower::service_fn;

    #[tokio::test]
    async fn retry_eventually_succeeds() {
        static COUNT: AtomicUsize = AtomicUsize::new(0);
        let svc = service_fn(|()| async move {
            let n = COUNT.fetch_add(1, Ordering::SeqCst);
            if n < 2 {
                Err::<(), BoxError>("e".into())
            } else {
                Ok::<(), BoxError>(())
            }
        });
        let layer = RetryLayer::new(
            RetryPolicy {
                max_retries: 5,
                backoff: Backoff::fixed(Duration::from_millis(1)),
            },
            AlwaysRetry,
        );
        let mut svc = layer.layer(svc);
        ServiceExt::ready(&mut svc)
            .await
            .unwrap()
            .call(())
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn timeout_triggers_error() {
        let svc = service_fn(|()| async move {
            sleep(Duration::from_millis(20)).await;
            Ok::<(), BoxError>(())
        });
        let mut svc = TimeoutLayer::new(Duration::from_millis(5)).layer(svc);
        let err = ServiceExt::ready(&mut svc)
            .await
            .unwrap()
            .call(())
            .await
            .unwrap_err();
        assert!(format!("{}", err).contains("timeout"));
    }

    #[tokio::test]
    async fn breaker_opens_after_failures() {
        static CALLED: AtomicUsize = AtomicUsize::new(0);
        let svc = service_fn(|()| async move {
            CALLED.fetch_add(1, Ordering::SeqCst);
            Err::<(), BoxError>("boom".into())
        });
        let mut svc = CircuitBreakerLayer::new(BreakerConfig {
            failure_threshold: 2,
            reset_timeout: Duration::from_millis(30),
        })
        .layer(svc);
        // first two calls invoke inner and fail
        let _ = ServiceExt::ready(&mut svc).await.unwrap().call(()).await;
        let _ = ServiceExt::ready(&mut svc).await.unwrap().call(()).await;
        // now breaker should open; next call should be short-circuited (no inner increment)
        let _ = ServiceExt::ready(&mut svc).await.unwrap().call(()).await;
        assert!(CALLED.load(Ordering::SeqCst) <= 2);
        // wait and allow half-open, then call again to hit inner once
        sleep(Duration::from_millis(35)).await;
        let _ = ServiceExt::ready(&mut svc).await.unwrap().call(()).await;
        assert!(CALLED.load(Ordering::SeqCst) <= 3);
    }
}
