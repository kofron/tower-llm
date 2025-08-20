//! Observability: tracing and metrics
//!
//! What this module provides (spec)
//! - Structured tracing and metrics around model calls, tools, and loop iterations
//!
//! Exports
//! - Models
//!   - `Usage { prompt_tokens, completion_tokens }`
//!   - `MetricRecord::{Counter{name, value}, Histogram{name, value}}`
//! - Layers
//!   - `TracingLayer<S>` creating spans with fields: model, tool_name, step_no, stop_reason
//!   - `MetricsLayer<S, C>` where `C: MetricsCollector`
//! - Services
//!   - `MetricsCollector: Service<MetricRecord, Response=()>`
//!
//! Implementation strategy
//! - `TracingLayer` decorates calls with `tracing::info_span` and structured fields
//! - `MetricsLayer` translates `StepAux`/`AgentRun` into metric updates via injected collector
//! - Keep overhead minimal; avoid heavy allocations in hot paths
//!
//! Composition
//! - `ServiceBuilder::new().layer(TracingLayer::new()).layer(MetricsLayer::new(collector)).service(step)`
//!
//! Testing strategy
//! - Use a fake collector capturing records; assert counts/histograms updated as expected
//! - Capture spans via `tracing-subscriber` test writer and assert key fields present

use std::future::Future;
use std::pin::Pin;

use async_openai::types::CreateChatCompletionRequest;
use tower::{BoxError, Layer, Service, ServiceExt};
use tracing::{info, info_span, Instrument};

use crate::core::{StepOutcome};

#[derive(Debug, Clone, Copy, Default)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
}

#[derive(Debug, Clone)]
pub enum MetricRecord {
    Counter { name: &'static str, value: u64 },
    Histogram { name: &'static str, value: u64 },
}

pub trait MetricsCollector: Service<MetricRecord, Response = (), Error = BoxError> {}
impl<T> MetricsCollector for T where T: Service<MetricRecord, Response = (), Error = BoxError> {}

/// Layer that adds tracing around step executions.
pub struct TracingLayer;
impl TracingLayer { pub fn new() -> Self { Self } }

pub struct Tracing<S> { inner: S }

impl<S> Layer<S> for TracingLayer {
    type Service = Tracing<S>;
    fn layer(&self, inner: S) -> Self::Service { Tracing { inner } }
}

impl<S> Service<CreateChatCompletionRequest> for Tracing<S>
where
    S: Service<CreateChatCompletionRequest, Response = StepOutcome, Error = BoxError> + Send + 'static,
    S::Future: Send + 'static,
{
    type Response = StepOutcome;
    type Error = BoxError;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(
        &mut self,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> { self.inner.poll_ready(cx) }

    fn call(&mut self, req: CreateChatCompletionRequest) -> Self::Future {
        let model = req.model.clone();
        let span = info_span!("step", model = %model);
        let fut = self.inner.call(req).instrument(span);
        Box::pin(async move {
            let out = fut.await?;
            match &out {
                StepOutcome::Next { aux, invoked_tools, .. } => info!(prompt = aux.prompt_tokens, completion = aux.completion_tokens, tools = aux.tool_invocations, invoked = ?invoked_tools, "step next"),
                StepOutcome::Done { aux, .. } => info!(prompt = aux.prompt_tokens, completion = aux.completion_tokens, tools = aux.tool_invocations, "step done"),
            }
            Ok(out)
        })
    }
}

/// Layer that translates step outcomes to metric updates via an injected collector.
pub struct MetricsLayer<C> { collector: C }
impl<C> MetricsLayer<C> { pub fn new(collector: C) -> Self { Self { collector } } }

pub struct Metrics<S, C> { inner: S, collector: C }

impl<S, C> Layer<S> for MetricsLayer<C>
where C: Clone {
    type Service = Metrics<S, C>;
    fn layer(&self, inner: S) -> Self::Service { Metrics { inner, collector: self.collector.clone() } }
}

impl<S, C> Service<CreateChatCompletionRequest> for Metrics<S, C>
where
    S: Service<CreateChatCompletionRequest, Response = StepOutcome, Error = BoxError> + Send + 'static,
    S::Future: Send + 'static,
    C: MetricsCollector + Clone + Send + 'static,
    C::Future: Send + 'static,
{
    type Response = StepOutcome;
    type Error = BoxError;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(
        &mut self,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> { self.inner.poll_ready(cx) }

    fn call(&mut self, req: CreateChatCompletionRequest) -> Self::Future {
        let mut collector = self.collector.clone();
        let fut = self.inner.call(req);
        Box::pin(async move {
            let out = fut.await?;
            let (prompt, completion, tools) = match &out {
                StepOutcome::Next { aux, .. } | StepOutcome::Done { aux, .. } => (aux.prompt_tokens, aux.completion_tokens, aux.tool_invocations),
            };
            let _ = ServiceExt::ready(&mut collector).await?.call(MetricRecord::Counter { name: "prompt_tokens", value: prompt as u64 }).await;
            let _ = ServiceExt::ready(&mut collector).await?.call(MetricRecord::Counter { name: "completion_tokens", value: completion as u64 }).await;
            let _ = ServiceExt::ready(&mut collector).await?.call(MetricRecord::Counter { name: "tool_invocations", value: tools as u64 }).await;
            Ok(out)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tower::service_fn;

    #[tokio::test]
    async fn metrics_layer_updates_collector() {
        let inner = service_fn(|_req: CreateChatCompletionRequest| async move {
            Ok::<_, BoxError>(StepOutcome::Done { messages: vec![], aux: crate::core::StepAux { prompt_tokens: 3, completion_tokens: 7, tool_invocations: 1 } })
        });
        let sink = std::sync::Arc::new(tokio::sync::Mutex::new(Vec::<(&'static str, u64)>::new()));
        let sink_cl = sink.clone();
        let collector = service_fn(move |rec: MetricRecord| {
            let sink = sink_cl.clone();
            async move {
                match rec { MetricRecord::Counter { name, value } => sink.lock().await.push((name, value)), _ => {} }
                Ok::<(), BoxError>(())
            }
        });
        let mut svc = MetricsLayer::new(collector).layer(inner);
        let req = CreateChatCompletionRequest { model: "gpt-4o".into(), messages: vec![], ..Default::default() };
        let _ = ServiceExt::ready(&mut svc).await.unwrap().call(req).await.unwrap();
        let data = sink.lock().await.clone();
        assert!(data.iter().any(|(n, v)| *n == "prompt_tokens" && *v == 3));
        assert!(data.iter().any(|(n, v)| *n == "completion_tokens" && *v == 7));
        assert!(data.iter().any(|(n, v)| *n == "tool_invocations" && *v == 1));
    }
}

