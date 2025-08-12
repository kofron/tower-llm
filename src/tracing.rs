//! Tracing system for agent execution using the `tracing` crate
//!
//! Provides structured logging and observability for agent runs.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{debug, error, info, instrument, warn};
use uuid::Uuid;

use crate::error::Result;
use crate::usage::Usage;

/// Unique identifier for a trace
pub type TraceId = String;

/// Unique identifier for a span
pub type SpanId = String;

/// Generate a new trace ID
pub fn gen_trace_id() -> TraceId {
    Uuid::new_v4().to_string()
}

/// Generate a new span ID
pub fn gen_span_id() -> SpanId {
    Uuid::new_v4().to_string()
}

/// Different types of spans in the agent execution
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SpanType {
    Agent {
        agent_name: String,
        instructions: String,
    },
    Generation {
        model: String,
        prompt_tokens: usize,
        completion_tokens: usize,
    },
    Tool {
        tool_name: String,
        arguments: serde_json::Value,
    },
    Guardrail {
        guardrail_name: String,
        guardrail_type: String,
        passed: bool,
    },
    Handoff {
        from_agent: String,
        to_agent: String,
        reason: Option<String>,
    },
    Custom {
        name: String,
        metadata: serde_json::Value,
    },
}

/// A span representing a unit of work
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Span {
    pub id: SpanId,
    pub trace_id: TraceId,
    pub parent_id: Option<SpanId>,
    pub span_type: SpanType,
    pub start_time: DateTime<Utc>,
    pub end_time: Option<DateTime<Utc>>,
    pub error: Option<String>,
    pub metadata: serde_json::Value,
}

impl Span {
    /// Create a new span
    pub fn new(trace_id: TraceId, parent_id: Option<SpanId>, span_type: SpanType) -> Self {
        Self {
            id: gen_span_id(),
            trace_id,
            parent_id,
            span_type,
            start_time: Utc::now(),
            end_time: None,
            error: None,
            metadata: serde_json::json!({}),
        }
    }

    /// Complete the span
    pub fn complete(&mut self) {
        self.end_time = Some(Utc::now());
    }

    /// Mark the span as failed
    pub fn fail(&mut self, error: String) {
        self.error = Some(error);
        self.complete();
    }

    /// Get the duration of the span in milliseconds
    pub fn duration_ms(&self) -> Option<i64> {
        self.end_time
            .map(|end| (end - self.start_time).num_milliseconds())
    }
}

/// Context for tracing agent execution
pub struct TracingContext {
    trace_id: TraceId,
    current_span_id: Option<SpanId>,
    spans: Vec<Span>,
}

impl TracingContext {
    /// Create a new tracing context
    pub fn new() -> Self {
        let trace_id = gen_trace_id();
        info!(trace_id = %trace_id, "Starting new trace");

        Self {
            trace_id,
            current_span_id: None,
            spans: Vec::new(),
        }
    }

    /// Start a new span
    pub fn start_span(&mut self, span_type: SpanType) -> SpanId {
        let span = Span::new(
            self.trace_id.clone(),
            self.current_span_id.clone(),
            span_type.clone(),
        );

        let span_id = span.id.clone();

        match &span_type {
            SpanType::Agent { agent_name, .. } => {
                info!(span_id = %span_id, agent = %agent_name, "Starting agent span");
            }
            SpanType::Tool { tool_name, .. } => {
                debug!(span_id = %span_id, tool = %tool_name, "Starting tool span");
            }
            SpanType::Generation { model, .. } => {
                debug!(span_id = %span_id, model = %model, "Starting generation span");
            }
            _ => {
                debug!(span_id = %span_id, "Starting span");
            }
        }

        self.spans.push(span);
        self.current_span_id = Some(span_id.clone());
        span_id
    }

    /// End a span
    pub fn end_span(&mut self, span_id: &str) {
        if let Some(span) = self.spans.iter_mut().find(|s| s.id == span_id) {
            span.complete();

            if let Some(duration) = span.duration_ms() {
                debug!(span_id = %span_id, duration_ms = duration, "Span completed");
            }

            // Update current span to parent
            if self.current_span_id.as_deref() == Some(span_id) {
                self.current_span_id = span.parent_id.clone();
            }
        }
    }

    /// Record an error in a span
    pub fn record_error(&mut self, span_id: &str, error: String) {
        if let Some(span) = self.spans.iter_mut().find(|s| s.id == span_id) {
            error!(span_id = %span_id, error = %error, "Span failed");
            span.fail(error);
        }
    }

    /// Get all spans in the trace
    pub fn spans(&self) -> &[Span] {
        &self.spans
    }

    /// Get the trace ID
    pub fn trace_id(&self) -> &str {
        &self.trace_id
    }
}

impl Default for TracingContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for agent spans with tracing instrumentation
pub struct AgentSpan {
    context: Arc<std::sync::Mutex<TracingContext>>,
    span_id: SpanId,
}

impl AgentSpan {
    /// Create a new agent span
    #[instrument(skip(context))]
    pub fn new(
        context: Arc<std::sync::Mutex<TracingContext>>,
        agent_name: String,
        instructions: String,
    ) -> Self {
        let span_id = {
            let mut ctx = context.lock().unwrap();
            ctx.start_span(SpanType::Agent {
                agent_name,
                instructions,
            })
        };

        Self { context, span_id }
    }

    /// Complete the span
    pub fn complete(self) {
        let mut ctx = self.context.lock().unwrap();
        ctx.end_span(&self.span_id);
    }

    /// Record an error
    pub fn error(self, error: String) {
        let mut ctx = self.context.lock().unwrap();
        ctx.record_error(&self.span_id, error);
    }
}

/// Builder for tool spans
pub struct ToolSpan {
    context: Arc<std::sync::Mutex<TracingContext>>,
    span_id: SpanId,
}

impl ToolSpan {
    /// Create a new tool span
    #[instrument(skip(context, arguments))]
    pub fn new(
        context: Arc<std::sync::Mutex<TracingContext>>,
        tool_name: String,
        arguments: serde_json::Value,
    ) -> Self {
        let span_id = {
            let mut ctx = context.lock().unwrap();
            ctx.start_span(SpanType::Tool {
                tool_name: tool_name.clone(),
                arguments,
            })
        };

        debug!(tool = %tool_name, "Executing tool");

        Self { context, span_id }
    }

    /// Complete with success
    pub fn success(self) {
        let mut ctx = self.context.lock().unwrap();
        ctx.end_span(&self.span_id);
    }

    /// Complete with error
    pub fn error(self, error: String) {
        let mut ctx = self.context.lock().unwrap();
        ctx.record_error(&self.span_id, error);
    }
}

/// Builder for generation spans
pub struct GenerationSpan {
    context: Arc<std::sync::Mutex<TracingContext>>,
    span_id: SpanId,
}

impl GenerationSpan {
    /// Create a new generation span
    #[instrument(skip(context))]
    pub fn new(context: Arc<std::sync::Mutex<TracingContext>>, model: String) -> Self {
        let span_id = {
            let mut ctx = context.lock().unwrap();
            ctx.start_span(SpanType::Generation {
                model: model.clone(),
                prompt_tokens: 0,
                completion_tokens: 0,
            })
        };

        info!(model = %model, "Starting generation");

        Self { context, span_id }
    }

    /// Complete with usage information
    pub fn complete_with_usage(self, usage: Usage) {
        let mut ctx = self.context.lock().unwrap();

        // Update the span with usage info
        if let Some(span) = ctx.spans.iter_mut().find(|s| s.id == self.span_id) {
            if let SpanType::Generation {
                prompt_tokens,
                completion_tokens,
                ..
            } = &mut span.span_type
            {
                *prompt_tokens = usage.prompt_tokens;
                *completion_tokens = usage.completion_tokens;
            }

            info!(
                prompt_tokens = usage.prompt_tokens,
                completion_tokens = usage.completion_tokens,
                "Generation completed"
            );
        }

        ctx.end_span(&self.span_id);
    }

    /// Complete with error
    pub fn error(self, error: String) {
        let mut ctx = self.context.lock().unwrap();
        ctx.record_error(&self.span_id, error);
    }
}

/// Export traces to external systems (simplified for now)
pub trait TraceExporter: Send + Sync {
    /// Export a completed trace
    fn export(&self, trace_id: &str, spans: Vec<Span>) -> Result<()>;
}

/// Console exporter for debugging
pub struct ConsoleExporter;

impl TraceExporter for ConsoleExporter {
    fn export(&self, trace_id: &str, spans: Vec<Span>) -> Result<()> {
        println!("=== Trace {} ===", trace_id);
        for span in spans {
            println!(
                "  [{:?}] {} -> {} ({}ms)",
                span.span_type,
                span.start_time.format("%H:%M:%S%.3f"),
                span.end_time
                    .map(|t| t.format("%H:%M:%S%.3f").to_string())
                    .unwrap_or_else(|| "ongoing".to_string()),
                span.duration_ms().unwrap_or(0)
            );
            if let Some(error) = &span.error {
                println!("    ERROR: {}", error);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[test]
    fn test_trace_id_generation() {
        let id1 = gen_trace_id();
        let id2 = gen_trace_id();

        assert_ne!(id1, id2);
        assert!(!id1.is_empty());
    }

    #[test]
    fn test_span_creation() {
        let span = Span::new(
            gen_trace_id(),
            None,
            SpanType::Agent {
                agent_name: "TestAgent".to_string(),
                instructions: "Test instructions".to_string(),
            },
        );

        assert!(span.end_time.is_none());
        assert!(span.error.is_none());
        assert!(span.parent_id.is_none());
    }

    #[test]
    fn test_span_completion() {
        let mut span = Span::new(
            gen_trace_id(),
            None,
            SpanType::Custom {
                name: "test".to_string(),
                metadata: serde_json::json!({}),
            },
        );

        span.complete();
        assert!(span.end_time.is_some());
        assert!(span.duration_ms().is_some());
    }

    #[test]
    fn test_span_error() {
        let mut span = Span::new(
            gen_trace_id(),
            None,
            SpanType::Tool {
                tool_name: "failing_tool".to_string(),
                arguments: serde_json::json!({}),
            },
        );

        span.fail("Tool execution failed".to_string());
        assert!(span.end_time.is_some());
        assert_eq!(span.error, Some("Tool execution failed".to_string()));
    }

    #[test]
    fn test_tracing_context() {
        let mut context = TracingContext::new();

        let span_id = context.start_span(SpanType::Agent {
            agent_name: "Agent1".to_string(),
            instructions: "Instructions".to_string(),
        });

        assert_eq!(context.current_span_id, Some(span_id.clone()));
        assert_eq!(context.spans.len(), 1);

        context.end_span(&span_id);
        assert!(context.spans[0].end_time.is_some());
    }

    #[test]
    fn test_nested_spans() {
        let mut context = TracingContext::new();

        let parent_id = context.start_span(SpanType::Agent {
            agent_name: "Parent".to_string(),
            instructions: "Parent instructions".to_string(),
        });

        let child_id = context.start_span(SpanType::Tool {
            tool_name: "child_tool".to_string(),
            arguments: serde_json::json!({"key": "value"}),
        });

        assert_eq!(context.spans.len(), 2);
        assert_eq!(context.spans[1].parent_id, Some(parent_id.clone()));

        context.end_span(&child_id);
        assert_eq!(context.current_span_id, Some(parent_id.clone()));

        context.end_span(&parent_id);
        assert_eq!(context.current_span_id, None);
    }

    #[test]
    fn test_agent_span_builder() {
        let context = Arc::new(Mutex::new(TracingContext::new()));
        let agent_span = AgentSpan::new(
            context.clone(),
            "TestAgent".to_string(),
            "Test instructions".to_string(),
        );

        {
            let ctx = context.lock().unwrap();
            assert_eq!(ctx.spans.len(), 1);
        }

        agent_span.complete();

        {
            let ctx = context.lock().unwrap();
            assert!(ctx.spans[0].end_time.is_some());
        }
    }

    #[test]
    fn test_tool_span_builder() {
        let context = Arc::new(Mutex::new(TracingContext::new()));
        let tool_span = ToolSpan::new(
            context.clone(),
            "test_tool".to_string(),
            serde_json::json!({"param": "value"}),
        );

        tool_span.success();

        let ctx = context.lock().unwrap();
        assert_eq!(ctx.spans.len(), 1);
        assert!(ctx.spans[0].error.is_none());
    }

    #[test]
    fn test_generation_span_with_usage() {
        let context = Arc::new(Mutex::new(TracingContext::new()));
        let gen_span = GenerationSpan::new(context.clone(), "gpt-4".to_string());

        let usage = Usage::new(100, 50);
        gen_span.complete_with_usage(usage);

        let ctx = context.lock().unwrap();
        if let SpanType::Generation {
            prompt_tokens,
            completion_tokens,
            ..
        } = &ctx.spans[0].span_type
        {
            assert_eq!(*prompt_tokens, 100);
            assert_eq!(*completion_tokens, 50);
        } else {
            panic!("Expected Generation span type");
        }
    }

    #[test]
    fn test_console_exporter() {
        let exporter = ConsoleExporter;
        let trace_id = gen_trace_id();

        let mut span = Span::new(
            trace_id.clone(),
            None,
            SpanType::Custom {
                name: "test".to_string(),
                metadata: serde_json::json!({}),
            },
        );
        span.complete();

        let result = exporter.export(&trace_id, vec![span]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_span_serialization() {
        let span = Span::new(
            gen_trace_id(),
            None,
            SpanType::Handoff {
                from_agent: "Agent1".to_string(),
                to_agent: "Agent2".to_string(),
                reason: Some("Specialized task".to_string()),
            },
        );

        let serialized = serde_json::to_string(&span).unwrap();
        let deserialized: Span = serde_json::from_str(&serialized).unwrap();

        assert_eq!(span.id, deserialized.id);
        assert_eq!(span.trace_id, deserialized.trace_id);
    }
}
