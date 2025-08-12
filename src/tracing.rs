//! # Tracing System for Agent Execution
//!
//! This module provides a structured tracing system for monitoring and debugging
//! agent execution. It is built on the concepts of traces and spans, which are
//! common in distributed tracing systems. This allows for detailed observability
//! into the agent's behavior, including LLM interactions, tool calls, and
//! guardrail evaluations.
//!
//! ## Core Concepts
//!
//! - **Trace**: A trace represents the entire lifecycle of a single agent run,
//!   from the initial user input to the final response. Each trace is identified
//!   by a unique `TraceId`.
//! - **Span**: A span represents a single unit of work within a trace, such as
//!   an LLM call or a tool execution. Spans can be nested to create a causal
//!   chain of events. Each span has a unique `SpanId`.
//!
//! ## Tracing Components
//!
//! - **[`TracingContext`]**: Manages the state of a trace, including the creation
//!   and completion of spans.
//! - **[`Span`]**: The data structure that holds all the information about a
//!   single span, including its type, start and end times, and any associated
//!   errors or metadata. The `SpanType` enum defines the different kinds of
//!   operations that can be traced.
//! - **Span Builders**: Helper structs like [`AgentSpan`], [`ToolSpan`], and
//!   [`GenerationSpan`] provide a convenient, RAII-style interface for creating
//!   and managing spans.
//! - **[`TraceExporter`]**: A trait for exporting completed traces to external
//!   systems, such as logging platforms or observability tools. A simple
//!   [`ConsoleExporter`] is provided for debugging.
//!
//! ### Example: Manually Creating Spans
//!
//! ```rust
//! use openai_agents_rs::tracing::{TracingContext, SpanType};
//!
//! let mut context = TracingContext::new();
//!
//! // Start the parent span.
//! let agent_span_id = context.start_span(SpanType::Agent {
//!     agent_name: "MyAgent".to_string(),
//!     instructions: "Be helpful.".to_string(),
//! });
//!
//! // Start a nested span for a tool call.
//! let tool_span_id = context.start_span(SpanType::Tool {
//!     tool_name: "get_weather".to_string(),
//!     arguments: serde_json::json!({"location": "San Francisco"}),
//! });
//!
//! // Complete the spans.
//! context.end_span(&tool_span_id);
//! context.end_span(&agent_span_id);
//!
//! let spans = context.spans();
//! assert_eq!(spans.len(), 2);
//! assert_eq!(spans[1].parent_id, Some(agent_span_id));
//! ```
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

/// A unique identifier for a trace, representing a single end-to-end agent run.
pub type TraceId = String;

/// A unique identifier for a span, representing a single unit of work.
pub type SpanId = String;

/// Generates a new, unique trace ID using UUIDv4.
pub fn gen_trace_id() -> TraceId {
    Uuid::new_v4().to_string()
}

/// Generates a new, unique span ID using UUIDv4.
pub fn gen_span_id() -> SpanId {
    Uuid::new_v4().to_string()
}

/// An enum representing the different types of operations that can be traced as spans.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SpanType {
    /// A span representing the execution of a single agent turn.
    Agent {
        agent_name: String,
        instructions: String,
    },
    /// A span for a call to the LLM to generate a response.
    Generation {
        model: String,
        prompt_tokens: usize,
        completion_tokens: usize,
    },
    /// A span for the execution of a tool.
    Tool {
        tool_name: String,
        arguments: serde_json::Value,
    },
    /// A span for a guardrail check.
    Guardrail {
        guardrail_name: String,
        guardrail_type: String,
        passed: bool,
    },
    /// A span representing a handoff between agents.
    Handoff {
        from_agent: String,
        to_agent: String,
        reason: Option<String>,
    },
    /// A custom span for application-specific tracing.
    Custom {
        name: String,
        metadata: serde_json::Value,
    },
}

/// Represents a single span in a trace, capturing a unit of work.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Span {
    /// The unique identifier for this span.
    pub id: SpanId,
    /// The ID of the trace this span belongs to.
    pub trace_id: TraceId,
    /// The ID of the parent span, if this is a nested span.
    pub parent_id: Option<SpanId>,
    /// The type of operation this span represents.
    pub span_type: SpanType,
    /// The time when the span started.
    pub start_time: DateTime<Utc>,
    /// The time when the span ended. `None` if the span is still in progress.
    pub end_time: Option<DateTime<Utc>>,
    /// An error message if the operation failed.
    pub error: Option<String>,
    /// Additional, unstructured metadata associated with the span.
    pub metadata: serde_json::Value,
}

impl Span {
    /// Creates a new `Span`.
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

    /// Marks the span as completed, setting its end time.
    pub fn complete(&mut self) {
        self.end_time = Some(Utc::now());
    }

    /// Marks the span as failed, setting an error message and the end time.
    pub fn fail(&mut self, error: String) {
        self.error = Some(error);
        self.complete();
    }

    /// Calculates and returns the duration of the span in milliseconds.
    pub fn duration_ms(&self) -> Option<i64> {
        self.end_time
            .map(|end| (end - self.start_time).num_milliseconds())
    }
}

/// Manages the context for a single trace, including its spans.
///
/// `TracingContext` is responsible for creating new spans, tracking the current
/// active span, and collecting all spans for a given trace.
pub struct TracingContext {
    trace_id: TraceId,
    current_span_id: Option<SpanId>,
    spans: Vec<Span>,
}

impl TracingContext {
    /// Creates a new `TracingContext` with a new, unique trace ID.
    pub fn new() -> Self {
        let trace_id = gen_trace_id();
        info!(trace_id = %trace_id, "Starting new trace");

        Self {
            trace_id,
            current_span_id: None,
            spans: Vec::new(),
        }
    }

    /// Starts a new span within the current trace.
    ///
    /// The new span will be a child of the current active span.
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

    /// Ends the specified span, marking it as complete.
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

    /// Records an error for the specified span.
    pub fn record_error(&mut self, span_id: &str, error: String) {
        if let Some(span) = self.spans.iter_mut().find(|s| s.id == span_id) {
            error!(span_id = %span_id, error = %error, "Span failed");
            span.fail(error);
        }
    }

    /// Returns a slice of all spans recorded in this context.
    pub fn spans(&self) -> &[Span] {
        &self.spans
    }

    /// Returns the trace ID for this context.
    pub fn trace_id(&self) -> &str {
        &self.trace_id
    }
}

impl Default for TracingContext {
    fn default() -> Self {
        Self::new()
    }
}

/// An RAII-style builder for creating and managing agent spans.
///
/// When an `AgentSpan` is created, it starts a new span in the tracing context.
/// When it is dropped (or `complete` is called), the span is automatically ended.
pub struct AgentSpan {
    context: Arc<std::sync::Mutex<TracingContext>>,
    span_id: SpanId,
}

impl AgentSpan {
    /// Creates a new `AgentSpan`.
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

    /// Explicitly completes the span before it is dropped.
    pub fn complete(self) {
        let mut ctx = self.context.lock().unwrap();
        ctx.end_span(&self.span_id);
    }

    /// Records an error for the span before it is completed.
    pub fn error(self, error: String) {
        let mut ctx = self.context.lock().unwrap();
        ctx.record_error(&self.span_id, error);
    }
}

/// An RAII-style builder for creating and managing tool spans.
pub struct ToolSpan {
    context: Arc<std::sync::Mutex<TracingContext>>,
    span_id: SpanId,
}

impl ToolSpan {
    /// Creates a new `ToolSpan`.
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

    /// Marks the tool execution as successful and completes the span.
    pub fn success(self) {
        let mut ctx = self.context.lock().unwrap();
        ctx.end_span(&self.span_id);
    }

    /// Records an error for the tool execution and completes the span.
    pub fn error(self, error: String) {
        let mut ctx = self.context.lock().unwrap();
        ctx.record_error(&self.span_id, error);
    }
}

/// An RAII-style builder for creating and managing LLM generation spans.
pub struct GenerationSpan {
    context: Arc<std::sync::Mutex<TracingContext>>,
    span_id: SpanId,
}

impl GenerationSpan {
    /// Creates a new `GenerationSpan`.
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

    /// Completes the span and records the token usage.
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

    /// Records an error for the generation and completes the span.
    pub fn error(self, error: String) {
        let mut ctx = self.context.lock().unwrap();
        ctx.record_error(&self.span_id, error);
    }
}

/// A trait for exporting completed traces to an external system.
pub trait TraceExporter: Send + Sync {
    /// Exports a collection of spans for a given trace.
    fn export(&self, trace_id: &str, spans: Vec<Span>) -> Result<()>;
}

/// A simple `TraceExporter` that prints the trace to the console.
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
