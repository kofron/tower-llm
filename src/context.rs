//! # Contextual Tool Output Handling
//!
//! Per-run context lets you observe and shape tool outputs as the agent runs.
//! A handler receives each tool's name, arguments, and result, and can choose
//! to forward the value unchanged, rewrite it, or finalize the run. This is
//! useful for aggregating across multiple tool calls, standardizing tool
//! output envelopes, or terminating early once a condition is met.

use std::any::Any;
use std::fmt;
use std::marker::PhantomData;
use std::sync::Arc;

use serde_json::Value;

use crate::agent::Agent;
use crate::error::{AgentsError, Result};

/// A decision returned by a context handler for a tool output.
///
/// Use this to direct how tool outputs feed back into the conversation or
/// whether the run should conclude immediately with a final value.
#[derive(Debug, Clone)]
pub enum ContextDecision {
    /// Forward the original tool output unchanged to the model.
    Forward,
    /// Rewrite the tool output with a new value that will be fed back to the model.
    Rewrite(Value),
    /// Immediately finalize the run with the provided value.
    Final(Value),
}

/// A context step containing the updated context and the decision.
#[derive(Debug, Clone)]
pub struct ContextStep<C> {
    pub ctx: C,
    pub decision: ContextDecision,
}

impl<C> ContextStep<C> {
    pub fn forward(ctx: C) -> Self {
        Self {
            ctx,
            decision: ContextDecision::Forward,
        }
    }

    pub fn rewrite(ctx: C, value: Value) -> Self {
        Self {
            ctx,
            decision: ContextDecision::Rewrite(value),
        }
    }

    pub fn final_output(ctx: C, value: Value) -> Self {
        Self {
            ctx,
            decision: ContextDecision::Final(value),
        }
    }
}

/// A typed trait implemented by user handlers to process tool outputs with context.
pub trait ToolContext<C>: Send + Sync {
    /// Called on each tool output (or tool error).
    ///
    /// Receives the current per-run context, the executed tool's name and
    /// arguments, and the tool result (success or error). Return a
    /// [`ContextStep`] with an updated context and a [`ContextDecision`]
    /// to forward, rewrite, or finalize. See `examples/contextual.rs` for usage.
    fn on_tool_output(
        &self,
        ctx: C,
        tool_name: &str,
        arguments: &Value,
        result: std::result::Result<Value, String>,
    ) -> Result<ContextStep<C>>;
}

/// Erased handler to store any typed `ToolContext<C>` behind trait objects.
pub trait ErasedToolContextHandler: Send + Sync {
    /// Applies the handler to the erased context value.
    fn on_tool_output(
        &self,
        ctx: Box<dyn Any + Send>,
        tool_name: &str,
        arguments: &Value,
        result: std::result::Result<Value, String>,
    ) -> Result<(Box<dyn Any + Send>, ContextDecision)>;
}

/// Adapter from a typed handler to an erased handler.
pub struct TypedHandler<C, H>
where
    C: Send + Sync + 'static,
    H: ToolContext<C> + Send + Sync + 'static,
{
    handler: H,
    _phantom: PhantomData<C>,
}

impl<C, H> TypedHandler<C, H>
where
    C: Send + Sync + 'static,
    H: ToolContext<C> + Send + Sync + 'static,
{
    pub fn new(handler: H) -> Self {
        Self {
            handler,
            _phantom: PhantomData,
        }
    }
}

impl<C, H> ErasedToolContextHandler for TypedHandler<C, H>
where
    C: Send + Sync + 'static,
    H: ToolContext<C> + Send + Sync + 'static,
{
    fn on_tool_output(
        &self,
        ctx: Box<dyn Any + Send>,
        tool_name: &str,
        arguments: &Value,
        result: std::result::Result<Value, String>,
    ) -> Result<(Box<dyn Any + Send>, ContextDecision)> {
        let boxed_ctx = ctx
            .downcast::<C>()
            .map_err(|_| AgentsError::Other("Context type mismatch".to_string()))?;
        let ctx_val = *boxed_ctx;

        let step = self
            .handler
            .on_tool_output(ctx_val, tool_name, arguments, result)?;
        Ok((Box::new(step.ctx) as Box<dyn Any + Send>, step.decision))
    }
}

/// Specification stored in `AgentConfig` to enable contextual handling.
#[derive(Clone)]
pub struct ToolContextSpec {
    /// Factory to construct a fresh per-run context (erased).
    pub factory: Arc<dyn Fn() -> Box<dyn Any + Send> + Send + Sync>,
    /// The erased context handler.
    pub handler: Arc<dyn ErasedToolContextHandler>,
}

impl fmt::Debug for ToolContextSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ToolContextSpec").finish()
    }
}

/// Run-scoped context specification (owned by the runner for the entire run).
#[derive(Clone)]
pub struct RunContextSpec {
    /// Factory to construct a fresh per-run context (erased).
    pub factory: Arc<dyn Fn() -> Box<dyn Any + Send> + Send + Sync>,
    /// The erased context handler.
    pub handler: Arc<dyn ErasedToolContextHandler>,
}

impl fmt::Debug for RunContextSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RunContextSpec").finish()
    }
}

/// A typed agent wrapper that carries the compile-time context type.
///
/// Use the typed builders on [`Agent`](crate::agent::Agent) to create one,
/// then pass it to the corresponding runner functions to retrieve the final
/// context value once the run completes.
#[derive(Clone)]
pub struct ContextualAgent<C> {
    pub(crate) agent: Agent,
    pub(crate) _marker: std::marker::PhantomData<C>,
}

impl<C> ContextualAgent<C> {
    pub fn into_inner(self) -> Agent {
        self.agent
    }
}
