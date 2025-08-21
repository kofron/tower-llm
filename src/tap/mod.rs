//! Generic, zero-interference tap layer for observing requests, responses, and errors.
//!
//! This module provides a lightweight Tower `Layer` that can wrap any `Service`
//! and invoke user-provided hooks on the request, response, and error paths
//! without altering the primary signal flow. The hooks are synchronous, cheap,
//! and receive references to the in-flight values so you can attach logging,
//! counters, or spawn background tasks for side-channel processing.
//!
//! Example (tapping a step service):
//! ```rust
//! use tower::{Service, ServiceExt, Layer, service_fn};
//! use tower_llm::tap::TapLayer;
//! use tower_llm::{StepOutcome};
//! use async_openai::types::{CreateChatCompletionRequest, CreateChatCompletionRequestArgs};
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
//! let on_req = |req: &CreateChatCompletionRequest| {
//!     tracing::debug!(model = ?req.model, "tap: request");
//! };
//! let on_res = |out: &StepOutcome| {
//!     tracing::debug!("tap: response");
//! };
//! let layer = TapLayer::<CreateChatCompletionRequest, StepOutcome, tower::BoxError>::new()
//!     .on_request(on_req)
//!     .on_response(on_res);
//!
//! let inner = service_fn(|_req: CreateChatCompletionRequest| async move {
//!     Ok::<_, tower::BoxError>(StepOutcome::Done { messages: vec![], aux: Default::default() })
//! });
//! let mut svc = layer.layer(inner);
//!
//! let req = CreateChatCompletionRequestArgs::default().model("gpt-4o").messages(vec![]).build()?;
//! let _ = ServiceExt::ready(&mut svc).await?.call(req).await?;
//! # Ok(())
//! # }
//! ```
//!
//! For fully non-blocking side-channel processing, create small owned events
//! inside the hook and `tokio::spawn` an async task to forward them to a sink.
//!
//! Design notes
//! - Hooks receive references to values; they must not hold them beyond the call.
//! - To avoid allocations and cloning, this layer does not capture both request
//!   and response together. If you need correlation, attach an ID in your request
//!   and emit it again in the response hook.

use std::future::Future;
use std::pin::Pin;
use tower::{Layer, Service};

/// Builder for a tap layer over a `Service<Req, Response = Res, Error = Err>`.
pub struct TapLayer<Req, Res, Err, OnReq = fn(&Req), OnRes = fn(&Res), OnErr = fn(&Err)> {
    on_request: Option<OnReq>,
    on_response: Option<OnRes>,
    on_error: Option<OnErr>,
    _phantom: std::marker::PhantomData<fn(Req, Res, Err)>,
}

impl<Req, Res, Err, OnReq, OnRes, OnErr> Default for TapLayer<Req, Res, Err, OnReq, OnRes, OnErr> {
    fn default() -> Self {
        Self::new()
    }
}

impl<Req, Res, Err, OnReq, OnRes, OnErr> TapLayer<Req, Res, Err, OnReq, OnRes, OnErr> {
    /// Create an empty tap with no hooks.
    pub fn new() -> Self {
        Self {
            on_request: None,
            on_response: None,
            on_error: None,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Attach a request hook, producing a new layer type with the closure installed.
    pub fn on_request<F>(self, f: F) -> TapLayer<Req, Res, Err, F, OnRes, OnErr>
    where
        F: Fn(&Req) + Send + Sync + 'static,
    {
        TapLayer {
            on_request: Some(f),
            on_response: self.on_response,
            on_error: self.on_error,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Attach a response hook, producing a new layer type with the closure installed.
    pub fn on_response<F>(self, f: F) -> TapLayer<Req, Res, Err, OnReq, F, OnErr>
    where
        F: Fn(&Res) + Send + Sync + 'static,
    {
        TapLayer {
            on_request: self.on_request,
            on_response: Some(f),
            on_error: self.on_error,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Attach an error hook, producing a new layer type with the closure installed.
    pub fn on_error<F>(self, f: F) -> TapLayer<Req, Res, Err, OnReq, OnRes, F>
    where
        F: Fn(&Err) + Send + Sync + 'static,
    {
        TapLayer {
            on_request: self.on_request,
            on_response: self.on_response,
            on_error: Some(f),
            _phantom: std::marker::PhantomData,
        }
    }
}

/// Service wrapper that invokes tap hooks around `call`.
pub struct Tap<S, Req, Res, Err, OnReq, OnRes, OnErr> {
    inner: S,
    on_request: Option<OnReq>,
    on_response: Option<OnRes>,
    on_error: Option<OnErr>,
    _phantom: std::marker::PhantomData<fn(Req, Res, Err)>,
}

impl<S, Req, Res, Err, OnReq, OnRes, OnErr> Layer<S>
    for TapLayer<Req, Res, Err, OnReq, OnRes, OnErr>
where
    OnReq: Clone,
    OnRes: Clone,
    OnErr: Clone,
{
    type Service = Tap<S, Req, Res, Err, OnReq, OnRes, OnErr>;

    fn layer(&self, inner: S) -> Self::Service {
        Tap {
            inner,
            on_request: self.on_request.clone(),
            on_response: self.on_response.clone(),
            on_error: self.on_error.clone(),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<S, Req, Res, Err, OnReq, OnRes, OnErr> Service<Req>
    for Tap<S, Req, Res, Err, OnReq, OnRes, OnErr>
where
    S: Service<Req, Response = Res, Error = Err> + Send + 'static,
    S::Future: Send + 'static,
    OnReq: Fn(&Req) + Send + Sync + Clone + 'static,
    OnRes: Fn(&Res) + Send + Sync + Clone + 'static,
    OnErr: Fn(&Err) + Send + Sync + Clone + 'static,
{
    type Response = Res;
    type Error = Err;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(
        &mut self,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: Req) -> Self::Future {
        if let Some(f) = &self.on_request {
            f(&req);
        }

        let on_response = self.on_response.clone();
        let on_error = self.on_error.clone();
        let fut = self.inner.call(req);

        Box::pin(async move {
            match fut.await {
                Ok(res) => {
                    if let Some(f) = &on_response {
                        f(&res);
                    }
                    Ok(res)
                }
                Err(err) => {
                    if let Some(f) = &on_error {
                        f(&err);
                    }
                    Err(err)
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_openai::types::{CreateChatCompletionRequest, CreateChatCompletionRequestArgs};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use tower::{service_fn, BoxError, ServiceExt};

    #[tokio::test]
    async fn tap_invokes_request_and_response_hooks() {
        let req_count = Arc::new(AtomicUsize::new(0));
        let res_count = Arc::new(AtomicUsize::new(0));

        let rc1 = req_count.clone();
        let rc2 = res_count.clone();
        let layer =
            TapLayer::<CreateChatCompletionRequest, crate::core::StepOutcome, BoxError>::new()
                .on_request(move |_r: &CreateChatCompletionRequest| {
                    rc1.fetch_add(1, Ordering::Relaxed);
                })
                .on_response(move |_o: &crate::core::StepOutcome| {
                    rc2.fetch_add(1, Ordering::Relaxed);
                });

        let inner = service_fn(|_req: CreateChatCompletionRequest| async move {
            Ok::<_, BoxError>(crate::core::StepOutcome::Done {
                messages: vec![],
                aux: Default::default(),
            })
        });

        let mut svc = layer.layer(inner);
        let req = CreateChatCompletionRequestArgs::default()
            .model("gpt-4o")
            .messages(vec![])
            .build()
            .unwrap();
        let _ = ServiceExt::ready(&mut svc)
            .await
            .unwrap()
            .call(req)
            .await
            .unwrap();

        assert_eq!(req_count.load(Ordering::Relaxed), 1);
        assert_eq!(res_count.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn tap_invokes_error_hook() {
        let err_count = Arc::new(AtomicUsize::new(0));
        let ec = err_count.clone();

        let layer =
            TapLayer::<CreateChatCompletionRequest, crate::core::StepOutcome, BoxError>::new()
                .on_error(move |_e: &BoxError| {
                    ec.fetch_add(1, Ordering::Relaxed);
                });

        let inner = service_fn(|_req: CreateChatCompletionRequest| async move {
            Err::<crate::core::StepOutcome, BoxError>("boom".into())
        });

        let mut svc = layer.layer(inner);
        let req = CreateChatCompletionRequestArgs::default()
            .model("gpt-4o")
            .messages(vec![])
            .build()
            .unwrap();
        let _ = ServiceExt::ready(&mut svc).await.unwrap().call(req).await;

        assert_eq!(err_count.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn tap_with_no_hooks_is_transparent() {
        let layer =
            TapLayer::<CreateChatCompletionRequest, crate::core::StepOutcome, BoxError>::new();
        let inner = service_fn(|_req: CreateChatCompletionRequest| async move {
            Ok::<_, BoxError>(crate::core::StepOutcome::Done {
                messages: vec![],
                aux: Default::default(),
            })
        });
        let mut svc = layer.layer(inner);
        let req = CreateChatCompletionRequestArgs::default()
            .model("gpt-4o")
            .messages(vec![])
            .build()
            .unwrap();
        let out = ServiceExt::ready(&mut svc)
            .await
            .unwrap()
            .call(req)
            .await
            .unwrap();
        match out {
            crate::core::StepOutcome::Done { .. } => {}
            _ => panic!("expected Done"),
        }
    }

    #[tokio::test]
    async fn tap_response_hook_fires_on_next() {
        let resp_count = Arc::new(AtomicUsize::new(0));
        let rc = resp_count.clone();
        let layer =
            TapLayer::<CreateChatCompletionRequest, crate::core::StepOutcome, BoxError>::new()
                .on_response(move |_o: &crate::core::StepOutcome| {
                    rc.fetch_add(1, Ordering::Relaxed);
                });

        let inner = service_fn(|_req: CreateChatCompletionRequest| async move {
            Ok::<_, BoxError>(crate::core::StepOutcome::Next {
                messages: vec![],
                aux: Default::default(),
                invoked_tools: vec![],
            })
        });
        let mut svc = layer.layer(inner);
        let req = CreateChatCompletionRequestArgs::default()
            .model("gpt-4o")
            .messages(vec![])
            .build()
            .unwrap();
        let _ = ServiceExt::ready(&mut svc)
            .await
            .unwrap()
            .call(req)
            .await
            .unwrap();
        assert_eq!(resp_count.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn tap_layers_can_be_chained_and_both_fire() {
        let req_a = Arc::new(AtomicUsize::new(0));
        let req_b = Arc::new(AtomicUsize::new(0));
        let ra = req_a.clone();
        let rb = req_b.clone();

        let l1 = TapLayer::<CreateChatCompletionRequest, crate::core::StepOutcome, BoxError>::new()
            .on_request(move |_r: &CreateChatCompletionRequest| {
                ra.fetch_add(1, Ordering::Relaxed);
            });
        let l2 = TapLayer::<CreateChatCompletionRequest, crate::core::StepOutcome, BoxError>::new()
            .on_request(move |_r: &CreateChatCompletionRequest| {
                rb.fetch_add(1, Ordering::Relaxed);
            });

        let inner = service_fn(|_req: CreateChatCompletionRequest| async move {
            Ok::<_, BoxError>(crate::core::StepOutcome::Done {
                messages: vec![],
                aux: Default::default(),
            })
        });

        // Chain l2 over l1 over inner
        let mut svc = l2.layer(l1.layer(inner));
        let req = CreateChatCompletionRequestArgs::default()
            .model("gpt-4o")
            .messages(vec![])
            .build()
            .unwrap();
        let _ = ServiceExt::ready(&mut svc)
            .await
            .unwrap()
            .call(req)
            .await
            .unwrap();
        assert_eq!(req_a.load(Ordering::Relaxed), 1);
        assert_eq!(req_b.load(Ordering::Relaxed), 1);
    }

    #[derive(Clone, Default)]
    struct CountingReady {
        calls: Arc<AtomicUsize>,
    }

    impl tower::Service<CreateChatCompletionRequest> for CountingReady {
        type Response = crate::core::StepOutcome;
        type Error = BoxError;
        type Future = std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<Self::Response, Self::Error>> + Send>,
        >;

        fn poll_ready(
            &mut self,
            _cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Result<(), Self::Error>> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            std::task::Poll::Ready(Ok(()))
        }

        fn call(&mut self, _req: CreateChatCompletionRequest) -> Self::Future {
            Box::pin(async move {
                Ok::<_, BoxError>(crate::core::StepOutcome::Done {
                    messages: vec![],
                    aux: Default::default(),
                })
            })
        }
    }

    #[tokio::test]
    async fn tap_poll_ready_is_delegated() {
        let inner = CountingReady::default();
        let layer =
            TapLayer::<CreateChatCompletionRequest, crate::core::StepOutcome, BoxError>::new();
        let calls = inner.calls.clone();

        let mut svc = layer.layer(inner);
        let req = CreateChatCompletionRequestArgs::default()
            .model("gpt-4o")
            .messages(vec![])
            .build()
            .unwrap();
        let _ = ServiceExt::ready(&mut svc)
            .await
            .unwrap()
            .call(req)
            .await
            .unwrap();
        assert!(calls.load(Ordering::Relaxed) >= 1);
    }
}
