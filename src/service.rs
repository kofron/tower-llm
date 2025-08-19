//! Tower-based tool execution primitives and layers.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use serde_json::Value;
use tokio::time::{sleep, timeout};
use tower::{service_fn, util::BoxService, BoxError, Layer, Service};

use crate::tool::{Tool, ToolResult};
use crate::usage::Usage;
use crate::{items::Message, model::ModelProvider};

// Re-export DefaultEnv from env module for backwards compatibility
pub use crate::env::DefaultEnv;

/// Control surface for layers and services to steer execution/finalization.
#[derive(Debug, Clone)]
pub enum Effect {
    Continue,
    Rewrite(Value),
    Final(Value),
}

/// Request passed into the tool service stack.
#[derive(Debug, Clone)]
pub struct ToolRequest<E = DefaultEnv>
where
    E: crate::env::Env,
{
    pub env: E,
    pub run_id: String,
    pub agent: String,
    pub tool_call_id: String,
    pub tool_name: String,
    pub arguments: Value,
}

/// Response from the tool service stack.
#[derive(Debug, Clone)]
pub struct ToolResponse {
    pub output: Value,
    pub error: Option<String>,
    pub effect: Effect,
}

impl ToolResponse {
    pub fn success(output: Value) -> Self {
        Self {
            output,
            error: None,
            effect: Effect::Continue,
        }
    }

    pub fn error(msg: String) -> Self {
        Self {
            output: Value::Null,
            error: Some(msg),
            effect: Effect::Continue,
        }
    }
}

/// Base tool executor adapting `dyn Tool` to a Tower Service.
#[derive(Clone)]
pub struct BaseToolService {
    tool: Arc<dyn Tool>,
}

impl BaseToolService {
    pub fn new(tool: Arc<dyn Tool>) -> Self {
        Self { tool }
    }
}

impl<E: crate::env::Env> Service<ToolRequest<E>> for BaseToolService {
    type Response = ToolResponse;
    type Error = BoxError;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(
        &mut self,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        std::task::Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: ToolRequest<E>) -> Self::Future {
        let tool = self.tool.clone();
        Box::pin(async move {
            match tool.execute(req.arguments.clone()).await {
                Ok(ToolResult {
                    output,
                    is_final,
                    error,
                }) => {
                    if let Some(err) = error {
                        Ok(ToolResponse {
                            output: Value::Null,
                            error: Some(err),
                            effect: Effect::Continue,
                        })
                    } else if is_final {
                        Ok(ToolResponse {
                            output: output.clone(),
                            error: None,
                            effect: Effect::Final(output),
                        })
                    } else {
                        Ok(ToolResponse::success(output))
                    }
                }
                Err(e) => Ok(ToolResponse::error(e.to_string())),
            }
        })
    }
}

// =============================
// Erased policy layer for boxed tool services
// =============================

/// Boxed service type used by the runner to compose dynamic layers.
pub type ToolBoxService = BoxService<ToolRequest<DefaultEnv>, ToolResponse, BoxError>;

/// An object-safe wrapper intended to layer a boxed tool service. Reserved for future DX.
pub trait ErasedToolLayer: Send + Sync {
    fn layer_boxed(&self, inner: ToolBoxService) -> ToolBoxService;
}

/// Timeout layer wrapper implementing `ErasedToolLayer` via a service_fn.
#[derive(Clone, Copy, Debug)]
pub struct BoxedTimeoutLayer(pub TimeoutLayer);

impl ErasedToolLayer for BoxedTimeoutLayer {
    fn layer_boxed(&self, inner: ToolBoxService) -> ToolBoxService {
        let d = self.0.duration;
        let shared = std::sync::Arc::new(tokio::sync::Mutex::new(inner));
        let svc = service_fn(move |req: ToolRequest<DefaultEnv>| {
            let shared = shared.clone();
            async move {
                let mut inner = shared.lock().await;
                match timeout(d, inner.call(req)).await {
                    Ok(res) => res,
                    Err(_elapsed) => Ok(ToolResponse::error("timeout".to_string())),
                }
            }
        });
        BoxService::new(svc)
    }
}

/// Retry layer wrapper implementing `ErasedToolLayer` via a service_fn.
#[derive(Clone, Copy, Debug)]
pub struct BoxedRetryLayer(pub RetryLayer);

impl ErasedToolLayer for BoxedRetryLayer {
    fn layer_boxed(&self, inner: ToolBoxService) -> ToolBoxService {
        let attempts = self.0.attempts;
        let delay = self.0.delay;
        let shared = std::sync::Arc::new(tokio::sync::Mutex::new(inner));
        let svc = service_fn(move |req: ToolRequest<DefaultEnv>| {
            let shared = shared.clone();
            async move {
                let mut last_resp: Option<ToolResponse> = None;
                let mut last_err: Option<BoxError> = None;
                for i in 0..attempts {
                    let mut inner = shared.lock().await;
                    match inner.call(req.clone()).await {
                        Ok(resp) => {
                            if resp.error.is_none() {
                                return Ok(resp);
                            } else {
                                last_resp = Some(resp);
                            }
                        }
                        Err(e) => {
                            last_err = Some(e);
                        }
                    }
                    drop(inner);
                    if i + 1 < attempts {
                        if let Some(d) = delay {
                            sleep(d).await;
                        }
                    }
                }
                if let Some(resp) = last_resp {
                    Ok(resp)
                } else if let Some(e) = last_err {
                    Err(e)
                } else {
                    Ok(ToolResponse::error("retry exhausted".to_string()))
                }
            }
        });
        BoxService::new(svc)
    }
}

/// Input schema layer wrapper implementing `ErasedToolLayer` via a service_fn.
#[derive(Clone, Debug)]
pub struct BoxedInputSchemaLayer(pub InputSchemaLayer);

impl ErasedToolLayer for BoxedInputSchemaLayer {
    fn layer_boxed(&self, inner: ToolBoxService) -> ToolBoxService {
        let schema = self.0.schema.clone();
        let strict = self.0.strict;
        let shared = std::sync::Arc::new(tokio::sync::Mutex::new(inner));
        let svc = service_fn(move |req: ToolRequest<DefaultEnv>| {
            let shared = shared.clone();
            let schema = schema.clone();
            async move {
                if !schema.is_null() {
                    if let Err(_e) = validate_against_minimal(&schema, &req.arguments) {
                        if strict {
                            return Ok(ToolResponse::error("schema validation failed".to_string()));
                        }
                    }
                }
                let mut inner = shared.lock().await;
                inner.call(req).await
            }
        });
        BoxService::new(svc)
    }
}

/// Approval layer wrapper implementing `ErasedToolLayer` via a predicate.
///
/// This dynamic variant does not require a typed Env. It uses a user-provided
/// predicate to decide approval based on (agent, tool_name, arguments).
type ApprovalPredicate = dyn Fn(&str, &str, &Value) -> bool + Send + Sync;

#[derive(Clone)]
pub struct BoxedApprovalLayer {
    predicate: Arc<ApprovalPredicate>,
}

impl std::fmt::Debug for BoxedApprovalLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BoxedApprovalLayer").finish()
    }
}

impl BoxedApprovalLayer {
    pub fn new<F>(pred: F) -> Self
    where
        F: Fn(&str, &str, &Value) -> bool + Send + Sync + 'static,
    {
        Self {
            predicate: Arc::new(pred),
        }
    }
}

impl ErasedToolLayer for BoxedApprovalLayer {
    fn layer_boxed(&self, inner: ToolBoxService) -> ToolBoxService {
        let pred = self.predicate.clone();
        let shared = std::sync::Arc::new(tokio::sync::Mutex::new(inner));
        let svc = service_fn(move |req: ToolRequest<DefaultEnv>| {
            let pred = pred.clone();
            let shared = shared.clone();
            async move {
                if !(pred)(&req.agent, &req.tool_name, &req.arguments) {
                    return Ok(ToolResponse::error("approval denied".to_string()));
                }
                let mut inner = shared.lock().await;
                inner.call(req).await
            }
        });
        BoxService::new(svc)
    }
}

// Convenience constructors for dynamic boxed layers
pub fn boxed_timeout_secs(secs: u64) -> Arc<dyn ErasedToolLayer> {
    Arc::new(BoxedTimeoutLayer(TimeoutLayer::secs(secs)))
}

pub fn boxed_retry_times(attempts: usize) -> Arc<dyn ErasedToolLayer> {
    Arc::new(BoxedRetryLayer(RetryLayer::times(attempts)))
}

pub fn boxed_input_schema_lenient(schema: Value) -> Arc<dyn ErasedToolLayer> {
    Arc::new(BoxedInputSchemaLayer(InputSchemaLayer::lenient(schema)))
}

pub fn boxed_input_schema_strict(schema: Value) -> Arc<dyn ErasedToolLayer> {
    Arc::new(BoxedInputSchemaLayer(InputSchemaLayer::strict(schema)))
}

pub fn boxed_approval_with<F>(pred: F) -> Arc<dyn ErasedToolLayer>
where
    F: Fn(&str, &str, &Value) -> bool + Send + Sync + 'static,
{
    Arc::new(BoxedApprovalLayer::new(pred))
}

// =============================
// Model provider adapter
// =============================

/// Request to a model provider via Tower service.
#[derive(Clone, Debug)]
pub struct ModelRequest {
    pub messages: Vec<Message>,
    pub tools: Vec<Arc<dyn Tool>>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
}

/// Service adapting a `ModelProvider` to a Tower `Service<ModelRequest>`
#[derive(Clone)]
pub struct ModelService {
    provider: Arc<dyn ModelProvider>,
}

impl ModelService {
    pub fn new(provider: Arc<dyn ModelProvider>) -> Self {
        Self { provider }
    }
}

impl Service<ModelRequest> for ModelService {
    type Response = (crate::items::ModelResponse, Usage);
    type Error = BoxError;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(
        &mut self,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        std::task::Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: ModelRequest) -> Self::Future {
        let provider = self.provider.clone();
        Box::pin(async move {
            let out = provider
                .complete(req.messages, req.tools, req.temperature, req.max_tokens)
                .await?;
            Ok(out)
        })
    }
}

/// Utility to build a boxed service stack for a given tool.
pub fn build_tool_stack<E: crate::env::Env>(
    tool: Arc<dyn Tool>,
) -> BoxService<ToolRequest<E>, ToolResponse, BoxError> {
    // Default lenient schema validation
    let schema = tool.parameters_schema();
    let base = BaseToolService::new(tool);
    let with_schema = InputSchemaLayer::lenient(schema).layer(base);
    BoxService::new(with_schema)
}

/// Input schema validation layer. Scope-agnostic; validates req.arguments.
#[derive(Clone, Debug)]
pub struct InputSchemaLayer {
    schema: Value,
    strict: bool,
}

impl InputSchemaLayer {
    pub fn strict(schema: Value) -> Self {
        Self {
            schema,
            strict: true,
        }
    }
    pub fn lenient(schema: Value) -> Self {
        Self {
            schema,
            strict: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct InputSchemaService<S> {
    inner: S,
    schema: Option<Value>,
    strict: bool,
}

impl<S> Layer<S> for InputSchemaLayer {
    type Service = InputSchemaService<S>;
    fn layer(&self, inner: S) -> Self::Service {
        InputSchemaService {
            inner,
            schema: Some(self.schema.clone()),
            strict: self.strict,
        }
    }
}

impl<S, E> Service<ToolRequest<E>> for InputSchemaService<S>
where
    S: Service<ToolRequest<E>, Response = ToolResponse, Error = BoxError> + Clone + Send + 'static,
    S::Future: Send + 'static,
    E: crate::env::Env,
{
    type Response = ToolResponse;
    type Error = BoxError;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(
        &mut self,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: ToolRequest<E>) -> Self::Future {
        let mut inner = self.inner.clone();
        let schema = self.schema.clone();
        let strict = self.strict;
        Box::pin(async move {
            if let Some(schema) = schema {
                if let Err(_msg) = validate_against_minimal(&schema, &req.arguments) {
                    if strict {
                        return Ok(ToolResponse {
                            output: Value::Null,
                            error: Some("schema validation failed".to_string()),
                            effect: Effect::Continue,
                        });
                    }
                }
            }
            inner.call(req).await
        })
    }
}

fn validate_against_minimal(schema: &Value, args: &Value) -> Result<(), String> {
    if let Some(required) = schema.get("required").and_then(|v| v.as_array()) {
        for field in required {
            if let Some(name) = field.as_str() {
                if args.get(name).is_none() {
                    return Err(format!("missing required field: {}", name));
                }
            }
        }
    }
    if let Some(props) = schema.get("properties").and_then(|v| v.as_object()) {
        if let Some(input) = props.get("input") {
            if input.get("type").and_then(|t| t.as_str()) == Some("string")
                && args.get("input").and_then(|v| v.as_str()).is_none()
            {
                return Err("input must be a string".to_string());
            }
        }
    }
    Ok(())
}

/// Generic timeout layer: applies at any scope and times out tool execution.
#[derive(Clone, Copy, Debug)]
pub struct TimeoutLayer {
    duration: Duration,
}

impl TimeoutLayer {
    pub fn secs(secs: u64) -> Self {
        Self {
            duration: Duration::from_secs(secs),
        }
    }
    pub fn from_duration(duration: Duration) -> Self {
        Self { duration }
    }
}

#[derive(Clone, Debug)]
pub struct TimeoutService<S> {
    inner: S,
    duration: Duration,
}

impl<S> Layer<S> for TimeoutLayer {
    type Service = TimeoutService<S>;
    fn layer(&self, inner: S) -> Self::Service {
        TimeoutService {
            inner,
            duration: self.duration,
        }
    }
}

impl<S, E> Service<ToolRequest<E>> for TimeoutService<S>
where
    S: Service<ToolRequest<E>, Response = ToolResponse, Error = BoxError> + Clone + Send + 'static,
    S::Future: Send + 'static,
    E: crate::env::Env,
{
    type Response = ToolResponse;
    type Error = BoxError;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(
        &mut self,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: ToolRequest<E>) -> Self::Future {
        let mut inner = self.inner.clone();
        let d = self.duration;
        Box::pin(async move {
            match timeout(d, inner.call(req)).await {
                Ok(res) => res,
                Err(_elapsed) => Ok(ToolResponse {
                    output: Value::Null,
                    error: Some("timeout".to_string()),
                    effect: Effect::Continue,
                }),
            }
        })
    }
}

/// Generic retry layer: retries on tool error responses or inner service errors.
#[derive(Clone, Copy, Debug)]
pub struct RetryLayer {
    attempts: usize,
    delay: Option<Duration>,
}

impl RetryLayer {
    pub fn times(attempts: usize) -> Self {
        Self {
            attempts: attempts.max(1),
            delay: None,
        }
    }
    pub fn with_delay(mut self, delay: Duration) -> Self {
        self.delay = Some(delay);
        self
    }
}

#[derive(Clone, Debug)]
pub struct RetryService<S> {
    inner: S,
    attempts: usize,
    delay: Option<Duration>,
}

impl<S> Layer<S> for RetryLayer {
    type Service = RetryService<S>;
    fn layer(&self, inner: S) -> Self::Service {
        RetryService {
            inner,
            attempts: self.attempts,
            delay: self.delay,
        }
    }
}

/// Approval capability trait; environments can implement this to enforce approvals.
pub trait HasApproval {
    fn approve(&self, agent: &str, tool: &str, args: &Value) -> bool;
}

/// Generic approval layer: denies execution if `approve` returns false.
#[derive(Clone, Copy, Debug, Default)]
pub struct ApprovalLayer;

#[derive(Clone, Debug)]
pub struct ApprovalService<S> {
    inner: S,
}

impl<S> Layer<S> for ApprovalLayer {
    type Service = ApprovalService<S>;
    fn layer(&self, inner: S) -> Self::Service {
        ApprovalService { inner }
    }
}

impl<S, E> Service<ToolRequest<E>> for ApprovalService<S>
where
    E: HasApproval + crate::env::Env,
    S: Service<ToolRequest<E>, Response = ToolResponse, Error = BoxError> + Clone + Send + 'static,
    S::Future: Send + 'static,
{
    type Response = ToolResponse;
    type Error = BoxError;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(
        &mut self,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: ToolRequest<E>) -> Self::Future {
        let mut inner = self.inner.clone();
        Box::pin(async move {
            if !req.env.approve(&req.agent, &req.tool_name, &req.arguments) {
                return Ok(ToolResponse::error("not approved".to_string()));
            }
            inner.call(req).await
        })
    }
}

impl<S, E> Service<ToolRequest<E>> for RetryService<S>
where
    S: Service<ToolRequest<E>, Response = ToolResponse, Error = BoxError> + Clone + Send + 'static,
    S::Future: Send + 'static,
    E: crate::env::Env,
{
    type Response = ToolResponse;
    type Error = BoxError;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(
        &mut self,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: ToolRequest<E>) -> Self::Future {
        let mut inner = self.inner.clone();
        let attempts = self.attempts;
        let delay = self.delay;
        let req_clone = req.clone();
        Box::pin(async move {
            let mut last_resp: Option<ToolResponse> = None;
            let mut last_err: Option<BoxError> = None;
            for i in 0..attempts {
                let cur_req = if i == 0 {
                    req.clone()
                } else {
                    req_clone.clone()
                };
                match inner.call(cur_req).await {
                    Ok(resp) => {
                        if resp.error.is_none() {
                            return Ok(resp);
                        } else {
                            last_resp = Some(resp);
                        }
                    }
                    Err(e) => {
                        last_err = Some(e);
                    }
                }
                if i + 1 < attempts {
                    if let Some(d) = delay {
                        sleep(d).await;
                    }
                }
            }
            if let Some(resp) = last_resp {
                return Ok(resp);
            }
            if let Some(e) = last_err {
                return Err(e);
            }
            // Fallback (should not happen)
            Ok(ToolResponse::error("retry exhausted".to_string()))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // use crate::model::MockProvider; // replaced with local mock in tests below
    use crate::tool::FunctionTool;
    use std::sync::Arc;
    use tower::ServiceExt;

    #[tokio::test]
    async fn base_tool_service_executes() {
        let tool = Arc::new(FunctionTool::simple("uppercase", "Upper", |s: String| {
            s.to_uppercase()
        }));
        let stack = build_tool_stack::<DefaultEnv>(tool);
        let req = ToolRequest {
            env: DefaultEnv,
            run_id: "r".into(),
            agent: "A".into(),
            tool_call_id: "t1".into(),
            tool_name: "uppercase".into(),
            arguments: serde_json::json!({"input": "abc"}),
        };

        let resp = stack.oneshot(req).await.unwrap();
        assert!(resp.error.is_none());
        assert_eq!(resp.output, serde_json::json!("ABC"));
        matches!(resp.effect, Effect::Continue);
    }

    #[tokio::test]
    async fn timeout_layer_times_out() {
        // Tool that blocks >50ms
        let tool = Arc::new(FunctionTool::new(
            "block".to_string(),
            "Blocks".to_string(),
            serde_json::json!({"type":"object"}),
            |_args| {
                std::thread::sleep(Duration::from_millis(100));
                Ok(Value::String("done".into()))
            },
        ));
        let base = BaseToolService::new(tool);
        let svc = TimeoutLayer::from_duration(Duration::from_millis(50)).layer(base);
        let req = ToolRequest {
            env: DefaultEnv,
            run_id: "r".into(),
            agent: "A".into(),
            tool_call_id: "t".into(),
            tool_name: "block".into(),
            arguments: Value::Null,
        };
        let resp = svc.oneshot(req).await.unwrap();
        assert!(resp.error.is_some());
        assert_eq!(resp.error.unwrap(), "timeout");
    }

    #[tokio::test]
    async fn input_schema_layer_strict_rejects_invalid() {
        let tool = Arc::new(FunctionTool::new(
            "echo".to_string(),
            "Echo".to_string(),
            serde_json::json!({
                "type": "object",
                "properties": {"input": {"type":"string"}},
                "required": ["input"]
            }),
            |args| Ok(args.get("input").cloned().unwrap_or(Value::Null)),
        ));
        let base = BaseToolService::new(tool);
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"input": {"type":"string"}},
            "required": ["input"]
        });
        let svc = InputSchemaLayer::strict(schema).layer(base);
        let req = ToolRequest {
            env: DefaultEnv,
            run_id: "r".into(),
            agent: "A".into(),
            tool_call_id: "t".into(),
            tool_name: "echo".into(),
            arguments: serde_json::json!({}),
        };
        let resp = svc.oneshot(req).await.unwrap();
        assert!(resp.error.is_some());
    }

    #[tokio::test]
    async fn input_schema_layer_lenient_allows_invalid() {
        let tool = Arc::new(FunctionTool::new(
            "echo".to_string(),
            "Echo".to_string(),
            serde_json::json!({
                "type": "object",
                "properties": {"input": {"type":"string"}},
                "required": ["input"]
            }),
            |_args| Ok(Value::String("ok".into())),
        ));
        let base = BaseToolService::new(tool);
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"input": {"type":"string"}},
            "required": ["input"]
        });
        let svc = InputSchemaLayer::lenient(schema).layer(base);
        let req = ToolRequest {
            env: DefaultEnv,
            run_id: "r".into(),
            agent: "A".into(),
            tool_call_id: "t".into(),
            tool_name: "echo".into(),
            arguments: serde_json::json!({}),
        };
        let resp = svc.oneshot(req).await.unwrap();
        assert!(resp.error.is_none());
        assert_eq!(resp.output, Value::String("ok".into()));
    }

    #[tokio::test]
    async fn retry_layer_succeeds_after_failure() {
        use std::sync::atomic::{AtomicBool, Ordering};
        static FIRST_FAIL: AtomicBool = AtomicBool::new(true);

        let tool = Arc::new(FunctionTool::new(
            "flaky".to_string(),
            "Flaky".to_string(),
            serde_json::json!({"type":"object"}),
            |_args| {
                if FIRST_FAIL.swap(false, Ordering::SeqCst) {
                    Err(crate::error::AgentsError::ToolExecutionError {
                        message: "boom".into(),
                    })
                } else {
                    Ok(Value::String("ok".into()))
                }
            },
        ));
        let base = BaseToolService::new(tool);
        let svc = RetryLayer::times(2).layer(base);
        let req = ToolRequest {
            env: DefaultEnv,
            run_id: "r".into(),
            agent: "A".into(),
            tool_call_id: "t".into(),
            tool_name: "flaky".into(),
            arguments: Value::Null,
        };
        let resp = svc.oneshot(req).await.unwrap();
        assert!(resp.error.is_none());
        assert_eq!(resp.output, Value::String("ok".into()));
    }

    #[tokio::test]
    async fn retry_layer_exhausts_and_errors() {
        let tool = Arc::new(FunctionTool::new(
            "always_fail".to_string(),
            "Always fail".to_string(),
            serde_json::json!({"type":"object"}),
            |_args| {
                Err(crate::error::AgentsError::ToolExecutionError {
                    message: "boom".into(),
                })
            },
        ));
        let base = BaseToolService::new(tool);
        let svc = RetryLayer::times(2).layer(base);
        let req = ToolRequest {
            env: DefaultEnv,
            run_id: "r".into(),
            agent: "A".into(),
            tool_call_id: "t".into(),
            tool_name: "always_fail".into(),
            arguments: Value::Null,
        };
        let resp = svc.oneshot(req).await; // last attempt returns service Ok with error payload or Err; both acceptable as failure
        if let Ok(r) = resp {
            assert!(r.error.is_some());
        }
    }

    #[tokio::test]
    async fn approval_layer_denies_without_approval() {
        #[derive(Clone, Default)]
        struct Deny;
        impl HasApproval for Deny {
            fn approve(&self, _agent: &str, _tool: &str, _args: &Value) -> bool {
                false
            }
        }

        let tool = Arc::new(FunctionTool::simple("echo", "Echo", |s: String| s));
        let base = BaseToolService::new(tool);
        let svc = ApprovalLayer.layer(base);
        let req = ToolRequest {
            env: Deny,
            run_id: "r".into(),
            agent: "A".into(),
            tool_call_id: "t".into(),
            tool_name: "echo".into(),
            arguments: serde_json::json!({"input":"x"}),
        };
        let resp = svc.oneshot(req).await.unwrap();
        assert!(resp.error.is_some());
        assert_eq!(resp.error.unwrap(), "not approved");
    }

    #[tokio::test]
    async fn model_service_pass_through() {
        struct MP;
        #[async_trait::async_trait]
        impl crate::model::ModelProvider for MP {
            async fn complete(
                &self,
                _messages: Vec<crate::items::Message>,
                _tools: Vec<std::sync::Arc<dyn crate::tool::Tool>>,
                _temperature: Option<f32>,
                _max_tokens: Option<u32>,
            ) -> crate::error::Result<(crate::items::ModelResponse, crate::usage::Usage)>
            {
                Ok((
                    crate::items::ModelResponse::new_message("ok"),
                    crate::usage::Usage::new(0, 0),
                ))
            }
            fn model_name(&self) -> &str {
                "m"
            }
        }
        let provider = std::sync::Arc::new(MP);
        let mut svc = super::ModelService::new(provider);
        let req = super::ModelRequest {
            messages: vec![crate::items::Message::user("hi")],
            tools: vec![],
            temperature: None,
            max_tokens: None,
        };
        let (_resp, _usage) = svc.call(req).await.unwrap();
    }
}
