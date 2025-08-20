//! Multi-agent orchestration and handoffs
//!
//! What this module provides (spec)
//! - A Tower-native router between multiple agent services with explicit handoff events
//!
//! Exports
//! - Models
//!   - `AgentName` newtype
//!   - `PickRequest { messages, last_stop: AgentStopReason }`
//! - Services
//!   - `GroupRouter: Service<RawChatRequest, Response=AgentRun>`
//!   - `AgentPicker: Service<PickRequest, Response=AgentName>`
//! - Layers
//!   - `HandoffLayer` that annotates runs with AgentStart/AgentEnd/Handoff events
//! - Utils
//!   - `GroupBuilder` to assemble named `AgentSvc`s and a picker strategy
//!
//! Implementation strategy
//! - Use `tower::steer` or a small name→index map, routing to boxed `AgentSvc`s
//! - `AgentPicker` decides next agent based on the current transcript and stop reason
//! - `HandoffLayer` wraps the router to emit handoff events into the run
//!
//! Composition
//! - `GroupBuilder::new().agent("triage", a).agent("specialist", b).picker(p).build()`
//! - Can be wrapped by resilience/observability layers as needed
//!
//! Testing strategy
//! - Build two fake agents that return deterministic responses
//! - A picker that selects based on a message predicate
//! - Assert the handoff events sequence and final run aggregation

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

use async_openai::types::CreateChatCompletionRequest;
use tower::{BoxError, Service, ServiceExt};

use crate::core::{AgentRun, AgentStopReason, AgentSvc};

pub type AgentName = String;

#[derive(Debug, Clone)]
pub struct PickRequest {
    pub messages: Vec<async_openai::types::ChatCompletionRequestMessage>,
    pub last_stop: AgentStopReason,
}

pub trait AgentPicker: Service<PickRequest, Response = AgentName, Error = BoxError> {}
impl<T> AgentPicker for T where T: Service<PickRequest, Response = AgentName, Error = BoxError> {}

pub struct GroupBuilder<P> {
    agents: HashMap<AgentName, AgentSvc>,
    picker: Option<P>,
}

impl<P> GroupBuilder<P> {
    pub fn new() -> Self {
        Self {
            agents: HashMap::new(),
            picker: None,
        }
    }
    pub fn agent(mut self, name: impl Into<String>, svc: AgentSvc) -> Self {
        self.agents.insert(name.into(), svc);
        self
    }
    pub fn picker(mut self, p: P) -> Self {
        self.picker = Some(p);
        self
    }
    pub fn build(self) -> GroupRouter<P> {
        GroupRouter {
            agents: std::sync::Arc::new(tokio::sync::Mutex::new(self.agents)),
            picker: self.picker.expect("picker"),
        }
    }
}

pub struct GroupRouter<P> {
    agents: std::sync::Arc<tokio::sync::Mutex<HashMap<AgentName, AgentSvc>>>,
    picker: P,
}

impl<P> Service<CreateChatCompletionRequest> for GroupRouter<P>
where
    P: AgentPicker + Clone + Send + 'static,
    P::Future: Send + 'static,
{
    type Response = AgentRun;
    type Error = BoxError;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(
        &mut self,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        std::task::Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: CreateChatCompletionRequest) -> Self::Future {
        let mut picker = self.picker.clone();
        let agents = self.agents.clone();
        Box::pin(async move {
            let pick = ServiceExt::ready(&mut picker)
                .await?
                .call(PickRequest {
                    messages: req.messages.clone(),
                    last_stop: AgentStopReason::DoneNoToolCalls,
                })
                .await?;
            let mut guard = agents.lock().await;
            let agent = guard
                .get_mut(&pick)
                .ok_or_else(|| format!("unknown agent: {}", pick))?;
            let run = ServiceExt::ready(agent).await?.call(req).await?;
            Ok(run)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_openai::types::{
        ChatCompletionRequestMessage, CreateChatCompletionRequest, CreateChatCompletionRequestArgs,
    };
    use serde_json::json;
    use tower::util::BoxService;

    #[tokio::test]
    async fn routes_to_named_agent() {
        let a: AgentSvc = BoxService::new(tower::service_fn(
            |_r: CreateChatCompletionRequest| async move {
                Ok::<_, BoxError>(AgentRun {
                    messages: vec![],
                    steps: 1,
                    stop: AgentStopReason::DoneNoToolCalls,
                })
            },
        ));
        let picker =
            tower::service_fn(|_pr: PickRequest| async move { Ok::<_, BoxError>("a".to_string()) });
        let router = GroupBuilder::new().agent("a", a).picker(picker).build();
        let mut svc = router;
        let req = CreateChatCompletionRequestArgs::default()
            .model("gpt-4o")
            .messages(vec![])
            .build()
            .unwrap();
        let run = ServiceExt::ready(&mut svc)
            .await
            .unwrap()
            .call(req)
            .await
            .unwrap();
        assert_eq!(run.steps, 1);
    }
}
