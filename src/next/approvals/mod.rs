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


