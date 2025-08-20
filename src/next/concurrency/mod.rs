//! Parallel tool execution and concurrency controls
//!
//! What this module provides (spec)
//! - A layer that fan-outs tool_calls concurrently and fan-ins results deterministically
//! - Configurable concurrency limits and join/failure policies
//!
//! Exports
//! - Models
//!   - `ConcurrencyLimit(usize)`
//!   - `ToolJoinPolicy::{JoinAll, FailFast, TimeoutPerTool(Duration)}`
//! - Layers
//!   - `ParallelToolsLayer<S, R>` where `S: Service<RawChatRequest, Response=StepOutcome>` and `R: Service<ToolInvocation,...>`
//! - Utils
//!   - Ordering helper to map completed outputs back to requested order
//!
//! Implementation strategy
//! - Wrap the tool router with `tower::buffer::Buffer` to acquire readiness per invocation
//! - On `StepOutcome::Next` with `invoked_tools`, spawn invocations concurrently:
//!   - Use `FuturesUnordered` or `join_all` with a semaphore set by `ConcurrencyLimit`
//!   - Apply `ToolJoinPolicy` (wait all, fail fast, per-invocation timeout)
//! - Serialize outputs as `tool` messages in the same order as original tool_calls
//! - Return a rewritten `StepOutcome::Next` with appended tool messages
//!
//! Composition
//! - `ServiceBuilder::new().layer(ParallelToolsLayer::new(limit, policy)).service(step)`
//! - Combine with resilience layers for per-tool retry/timeout if desired
//!
//! Testing strategy
//! - Fake tools with injected latency and error behavior
//! - Assert that with `JoinAll` all succeed and order is preserved
//! - Assert that with `FailFast` layer aborts on first error and surfaces it
//! - Assert that limit `N` bounds concurrent calls (use atomic counters)

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::{Mutex, Semaphore};
use tower::{BoxError, Service, ServiceExt};

use crate::next::{ToolInvocation, ToolOutput};

/// Maximum number of concurrent tool invocations.
#[derive(Debug, Clone, Copy)]
pub struct ConcurrencyLimit(pub usize);

/// Policy describing how to join multiple tool invocations.
#[derive(Debug, Clone, Copy)]
pub enum ToolJoinPolicy {
    /// Wait for all to complete; if any error occurs, return the first error.
    JoinAll,
    /// Return error as soon as one occurs (tasks still complete in background).
    FailFast,
}

/// Wraps a tool router service `R` to execute batches of tool invocations concurrently.
/// 
/// Note: The inner service R should be wrapped in tower::buffer::Buffer if it doesn't
/// support concurrent access (e.g., if it doesn't implement Clone).
#[derive(Clone)]
pub struct ParallelToolRouter<R> {
    inner: R,
    limit: ConcurrencyLimit,
    policy: ToolJoinPolicy,
}

impl<R> ParallelToolRouter<R> {
    pub fn new(inner: R, limit: ConcurrencyLimit, policy: ToolJoinPolicy) -> Self {
        Self {
            inner,
            limit,
            policy,
        }
    }
}

impl<R> Service<Vec<ToolInvocation>> for ParallelToolRouter<R>
where
    R: Service<ToolInvocation, Response = ToolOutput, Error = BoxError> + Clone + Send + 'static,
    R::Future: Send + 'static,
{
    type Response = Vec<ToolOutput>;
    type Error = BoxError;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(
        &mut self,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        std::task::Poll::Ready(Ok(()))
    }

    fn call(&mut self, reqs: Vec<ToolInvocation>) -> Self::Future {
        let limit = self.limit.0.max(1);
        let policy = self.policy;
        let router = self.inner.clone();
        Box::pin(async move {
            let sem = Arc::new(Semaphore::new(limit));
            let mut handles = Vec::with_capacity(reqs.len());
            for (idx, inv) in reqs.into_iter().enumerate() {
                let permit = sem.clone().acquire_owned().await.expect("semaphore");
                let mut svc = router.clone();
                handles.push(tokio::spawn(async move {
                    let _p = permit;
                    let out = svc.ready().await?.call(inv).await;
                    out.map(|o| (idx, o))
                }));
            }

            // Collect results; preserve original order
            let mut slots: Vec<Option<ToolOutput>> = vec![None; handles.len()];
            let mut first_err: Option<BoxError> = None;
            for h in handles {
                match h.await.expect("join") {
                    Ok((idx, out)) => {
                        slots[idx] = Some(out);
                    }
                    Err(e) => {
                        if first_err.is_none() {
                            first_err = Some(e);
                            if matches!(policy, ToolJoinPolicy::FailFast) {
                                break;
                            }
                        }
                    }
                }
            }
            if let Some(e) = first_err {
                return Err(e);
            }
            let mut outputs = Vec::with_capacity(slots.len());
            for s in slots.into_iter() {
                outputs.push(s.expect("missing output"));
            }
            Ok(outputs)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tower::service_fn;

    #[tokio::test]
    async fn preserves_order_with_concurrency() {
        // Fake router that delays based on invocation name
        let router = service_fn(|inv: ToolInvocation| async move {
            if inv.name == "slow" {
                sleep(Duration::from_millis(50)).await;
            } else {
                sleep(Duration::from_millis(5)).await;
            }
            Ok::<_, BoxError>(ToolOutput {
                id: inv.id,
                result: Value::String(inv.name),
            })
        });
        let mut svc = ParallelToolRouter::new(router, ConcurrencyLimit(2), ToolJoinPolicy::JoinAll);
        let reqs = vec![
            ToolInvocation {
                id: "a".into(),
                name: "slow".into(),
                arguments: Value::Null,
            },
            ToolInvocation {
                id: "b".into(),
                name: "fast".into(),
                arguments: Value::Null,
            },
        ];
        let outputs = svc.ready().await.unwrap().call(reqs).await.unwrap();
        assert_eq!(outputs.len(), 2);
        // Order must match inputs even though execution time differs
        assert_eq!(outputs[0].result, Value::String("slow".into()));
        assert_eq!(outputs[1].result, Value::String("fast".into()));
    }

    #[tokio::test]
    async fn fail_fast_returns_error() {
        let router = service_fn(|inv: ToolInvocation| async move {
            if inv.name == "bad" {
                Err::<ToolOutput, BoxError>("boom".into())
            } else {
                Ok::<_, BoxError>(ToolOutput {
                    id: inv.id,
                    result: Value::Null,
                })
            }
        });
        let mut svc =
            ParallelToolRouter::new(router, ConcurrencyLimit(4), ToolJoinPolicy::FailFast);
        let reqs = vec![
            ToolInvocation {
                id: "1".into(),
                name: "ok".into(),
                arguments: Value::Null,
            },
            ToolInvocation {
                id: "2".into(),
                name: "bad".into(),
                arguments: Value::Null,
            },
            ToolInvocation {
                id: "3".into(),
                name: "ok".into(),
                arguments: Value::Null,
            },
        ];
        let err = svc.ready().await.unwrap().call(reqs).await.unwrap_err();
        assert!(format!("{}", err).contains("boom"));
    }

    #[tokio::test]
    async fn enforces_concurrency_limit() {
        static CURRENT: AtomicUsize = AtomicUsize::new(0);
        static MAX_OBSERVED: AtomicUsize = AtomicUsize::new(0);
        let router = service_fn(|inv: ToolInvocation| async move {
            let now = CURRENT.fetch_add(1, Ordering::SeqCst) + 1;
            loop {
                let max = MAX_OBSERVED.load(Ordering::SeqCst);
                if now > max {
                    MAX_OBSERVED
                        .compare_exchange(max, now, Ordering::SeqCst, Ordering::SeqCst)
                        .ok();
                } else {
                    break;
                }
                break;
            }
            sleep(Duration::from_millis(10)).await;
            CURRENT.fetch_sub(1, Ordering::SeqCst);
            Ok::<_, BoxError>(ToolOutput {
                id: inv.id,
                result: Value::Null,
            })
        });

        let mut svc = ParallelToolRouter::new(router, ConcurrencyLimit(2), ToolJoinPolicy::JoinAll);
        let reqs: Vec<ToolInvocation> = (0..8)
            .map(|i| ToolInvocation {
                id: format!("{}", i),
                name: "n".into(),
                arguments: Value::Null,
            })
            .collect();
        let _ = svc.ready().await.unwrap().call(reqs).await.unwrap();
        assert!(MAX_OBSERVED.load(Ordering::SeqCst) <= 2);
    }
}
