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


