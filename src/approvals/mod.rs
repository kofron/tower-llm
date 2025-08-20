//! Guardrails and approvals
//!
//! What this module provides (spec)
//! - Pluggable approval strategies applied before model/tool execution
//! - Ability to deny, allow, or rewrite requests
//!
//! Exports
//! - Models
//!   - `ApprovalRequest { stage: Stage, request: RawChatRequest | ToolInvocation }`
//!   - `Decision::{Allow, Deny{reason}, Modify{request}}`
//!   - `Stage::{Model, Tool}`
//! - Services
//!   - `Approver: Service<ApprovalRequest, Response=Decision>`
//! - Layers
//!   - `ApprovalLayer<S, A>` where `A: Approver`
//!     - On Model stage: evaluate before calling provider; on Modify, replace request; on Deny, short-circuit
//!     - On Tool stage: evaluate before invoking router
//! - Utils
//!   - Prebuilt approvers: `AllowListTools`, `MaxArgsSize`, `RequireReasoning`
//!
//! Implementation strategy
//! - Keep `Approver` pure and side-effect free (unless intentionally stateful)
//! - The layer inspects the stage and constructs `ApprovalRequest` appropriately
//! - Decisions flow control the inner service call
//!
//! Composition
//! - `ServiceBuilder::new().layer(ApprovalLayer::new(my_approver)).service(step)`
//! - For tools, wrap the router separately if needed
//!
//! Testing strategy
//! - Fake approver returning scripted decisions
//! - Unit tests per stage: Model denial prevents provider call; Tool denial prevents router call; Modify rewrites inputs

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use async_openai::types::CreateChatCompletionRequest;
use tower::{BoxError, Layer, Service, ServiceExt};

use crate::core::{StepOutcome, ToolInvocation, ToolOutput};

/// Stage at which an approval is evaluated.
#[derive(Debug, Clone)]
pub enum Stage {
    Model,
    Tool,
}

/// Approval request payload.
#[derive(Debug, Clone)]
pub enum ApprovalRequest {
    Model {
        request: CreateChatCompletionRequest,
    },
    Tool {
        invocation: ToolInvocation,
    },
}

/// Approval decision.
#[derive(Debug, Clone)]
pub enum Decision {
    Allow,
    Deny {
        reason: String,
    },
    ModifyModel {
        request: CreateChatCompletionRequest,
    },
    ModifyTool {
        invocation: ToolInvocation,
    },
}

/// Approver service trait alias: Service<ApprovalRequest, Decision>.
pub trait Approver: Service<ApprovalRequest, Response = Decision, Error = BoxError> {}
impl<T> Approver for T where T: Service<ApprovalRequest, Response = Decision, Error = BoxError> {}

/// Layer that evaluates approvals at the model stage before invoking the inner step service.
pub struct ModelApprovalLayer<A> {
    approver: A,
}

impl<A> ModelApprovalLayer<A> {
    pub fn new(approver: A) -> Self {
        Self { approver }
    }
}

pub struct ModelApproval<S, A> {
    inner: Arc<tokio::sync::Mutex<S>>,
    approver: A,
}

impl<S, A> Layer<S> for ModelApprovalLayer<A>
where
    A: Clone,
{
    type Service = ModelApproval<S, A>;
    fn layer(&self, inner: S) -> Self::Service {
        ModelApproval {
            inner: Arc::new(tokio::sync::Mutex::new(inner)),
            approver: self.approver.clone(),
        }
    }
}

impl<S, A> Service<CreateChatCompletionRequest> for ModelApproval<S, A>
where
    S: Service<CreateChatCompletionRequest, Response = StepOutcome, Error = BoxError>
        + Send
        + 'static,
    S::Future: Send + 'static,
    A: Approver + Clone + Send + 'static,
    A::Future: Send + 'static,
{
    type Response = StepOutcome;
    type Error = BoxError;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(
        &mut self,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        let _ = cx;
        std::task::Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: CreateChatCompletionRequest) -> Self::Future {
        let mut approver = self.approver.clone();
        let inner = self.inner.clone();
        Box::pin(async move {
            let decision = ServiceExt::ready(&mut approver)
                .await?
                .call(ApprovalRequest::Model {
                    request: req.clone(),
                })
                .await?;
            match decision {
                Decision::Allow | Decision::ModifyTool { .. } => {
                    let mut guard = inner.lock().await;
                    Service::call(&mut *guard, req).await
                }
                Decision::Deny { reason: _ } => Ok(StepOutcome::Done {
                    messages: vec![],
                    aux: Default::default(),
                }),
                Decision::ModifyModel { request } => {
                    let mut guard = inner.lock().await;
                    Service::call(&mut *guard, request).await
                }
            }
        })
    }
}

/// Wrapper for tool router that evaluates approvals per invocation.
pub struct ToolApprovalLayer<A> {
    approver: A,
}

impl<A> ToolApprovalLayer<A> {
    pub fn new(approver: A) -> Self {
        Self { approver }
    }
}

pub struct ToolApproval<R, A> {
    inner: Arc<tokio::sync::Mutex<R>>,
    approver: A,
}

impl<R, A> Layer<R> for ToolApprovalLayer<A>
where
    A: Clone,
{
    type Service = ToolApproval<R, A>;
    fn layer(&self, inner: R) -> Self::Service {
        ToolApproval {
            inner: Arc::new(tokio::sync::Mutex::new(inner)),
            approver: self.approver.clone(),
        }
    }
}

impl<R, A> Service<ToolInvocation> for ToolApproval<R, A>
where
    R: Service<ToolInvocation, Response = ToolOutput, Error = BoxError> + Send + 'static,
    R::Future: Send + 'static,
    A: Approver + Clone + Send + 'static,
    A::Future: Send + 'static,
{
    type Response = ToolOutput;
    type Error = BoxError;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(
        &mut self,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        let _ = cx;
        std::task::Poll::Ready(Ok(()))
    }

    fn call(&mut self, inv: ToolInvocation) -> Self::Future {
        let mut approver = self.approver.clone();
        let inner = self.inner.clone();
        Box::pin(async move {
            let decision = ServiceExt::ready(&mut approver)
                .await?
                .call(ApprovalRequest::Tool {
                    invocation: inv.clone(),
                })
                .await?;
            match decision {
                Decision::Allow | Decision::ModifyModel { .. } => {
                    let mut guard = inner.lock().await;
                    Service::call(&mut *guard, inv).await
                }
                Decision::Deny { reason } => {
                    Err::<ToolOutput, BoxError>(format!("denied: {}", reason).into())
                }
                Decision::ModifyTool { invocation } => {
                    let mut guard = inner.lock().await;
                    Service::call(&mut *guard, invocation).await
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tower::{service_fn, Service};

    #[tokio::test]
    async fn model_denial_short_circuits() {
        static CALLED: AtomicUsize = AtomicUsize::new(0);
        let inner = service_fn(|_req: CreateChatCompletionRequest| async move {
            CALLED.fetch_add(1, Ordering::SeqCst);
            Ok::<_, BoxError>(StepOutcome::Done {
                messages: vec![],
                aux: Default::default(),
            })
        });
        let approver = service_fn(|_req: ApprovalRequest| async move {
            Ok::<_, BoxError>(Decision::Deny {
                reason: "no".into(),
            })
        });
        let mut svc = ModelApprovalLayer::new(approver).layer(inner);
        let req = CreateChatCompletionRequest {
            model: "gpt-4o".into(),
            messages: vec![],
            ..Default::default()
        };
        let _ = Service::call(&mut svc, req).await.unwrap();
        assert_eq!(CALLED.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn tool_denial_prevents_call() {
        static CALLED: AtomicUsize = AtomicUsize::new(0);
        let inner = service_fn(|_inv: ToolInvocation| async move {
            CALLED.fetch_add(1, Ordering::SeqCst);
            Ok::<_, BoxError>(ToolOutput {
                id: "1".into(),
                result: Value::Null,
            })
        });
        let approver = service_fn(|_req: ApprovalRequest| async move {
            Ok::<_, BoxError>(Decision::Deny {
                reason: "no".into(),
            })
        });
        let mut svc = ToolApprovalLayer::new(approver).layer(inner);
        let err = Service::call(
            &mut svc,
            ToolInvocation {
                id: "1".into(),
                name: "x".into(),
                arguments: Value::Null,
            },
        )
        .await
        .unwrap_err();
        assert!(format!("{}", err).contains("denied"));
        assert_eq!(CALLED.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn model_modify_rewrites_request() {
        static CALLED: AtomicUsize = AtomicUsize::new(0);
        // Inner asserts it receives the modified request (system message content == "APPROVED")
        let inner = service_fn(|req: CreateChatCompletionRequest| async move {
            CALLED.fetch_add(1, Ordering::SeqCst);
            // First message should be system with content "APPROVED"
            if let Some(first) = req.messages.first() {
                match first {
                    async_openai::types::ChatCompletionRequestMessage::System(s) => {
                        if let async_openai::types::ChatCompletionRequestSystemMessageContent::Text(t) = &s.content {
                            assert_eq!(t, "APPROVED");
                        } else {
                            panic!("expected text content");
                        }
                    }
                    _ => panic!("expected system message first"),
                }
            } else {
                panic!("no messages");
            }
            Ok::<_, BoxError>(StepOutcome::Done {
                messages: req.messages,
                aux: Default::default(),
            })
        });
        // Approver transforms incoming request by injecting system message "APPROVED"
        let approver = service_fn(|req: ApprovalRequest| async move {
            match req {
                ApprovalRequest::Model { request: _ } => {
                    let mut b = async_openai::types::CreateChatCompletionRequestArgs::default();
                    b.model("gpt-4o");
                    let sys =
                        async_openai::types::ChatCompletionRequestSystemMessageArgs::default()
                            .content("APPROVED")
                            .build()
                            .unwrap();
                    b.messages(vec![sys.into()]);
                    let modified = b.build().unwrap();
                    Ok::<_, BoxError>(Decision::ModifyModel { request: modified })
                }
                _ => Ok::<_, BoxError>(Decision::Allow),
            }
        });
        let mut svc = ModelApprovalLayer::new(approver).layer(inner);
        let orig = async_openai::types::CreateChatCompletionRequestArgs::default()
            .model("gpt-4o")
            .messages(vec![])
            .build()
            .unwrap();
        let _ = Service::call(&mut svc, orig).await.unwrap();
        assert_eq!(CALLED.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn tool_modify_rewrites_invocation() {
        static CALLED: AtomicUsize = AtomicUsize::new(0);
        // Inner asserts invocation name was modified to "modified_tool"
        let inner = service_fn(|inv: ToolInvocation| async move {
            CALLED.fetch_add(1, Ordering::SeqCst);
            assert_eq!(inv.name, "modified_tool");
            Ok::<_, BoxError>(ToolOutput {
                id: inv.id,
                result: Value::Null,
            })
        });
        // Approver rewrites tool invocation name
        let approver = service_fn(|req: ApprovalRequest| async move {
            match req {
                ApprovalRequest::Tool { mut invocation } => {
                    invocation.name = "modified_tool".to_string();
                    Ok::<_, BoxError>(Decision::ModifyTool { invocation })
                }
                _ => Ok::<_, BoxError>(Decision::Allow),
            }
        });
        let mut svc = ToolApprovalLayer::new(approver).layer(inner);
        let _ = Service::call(
            &mut svc,
            ToolInvocation {
                id: "1".into(),
                name: "orig".into(),
                arguments: Value::Null,
            },
        )
        .await
        .unwrap();
        assert_eq!(CALLED.load(Ordering::SeqCst), 1);
    }
}
