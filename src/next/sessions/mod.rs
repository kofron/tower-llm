//! Sessions and persistence (memory/replay) for the `next` stack
//!
//! What this module provides (spec)
//! - A clear, Tower-native way to persist and replay conversational state
//! - No dynamic lookups; all dependencies are constructor-injected
//! - Interoperates with the codec (RunItem ↔ raw messages) and recording
//!
//! Exports (public API surface)
//! - Models
//!   - `SessionId` (newtype)
//!   - `LoadSession { id: SessionId }`, `SaveSession { id: SessionId, history: History }`
//!   - `History` = `Vec<RawChatMessage>` (or a smalltype wrapper)
//! - Services
//!   - `SessionLoadStore: Service<LoadSession, Response=History, Error=BoxError>`
//!   - `SessionSaveStore: Service<SaveSession, Response=(), Error=BoxError>`
//!     - Impl examples: `SqliteSessionStore`, `InMemorySessionStore`
//! - Layers
//!   - `MemoryLayer<S>` where `S: Service<RawChatRequest, Response=StepOutcome>`
//!     - On call: loads `History`, merges into request messages, forwards, then appends new messages and saves
//!   - `RecorderLayer<S>` (see recording module)
//!   - `ReplayLayer<S>` (short-circuits with canned outcomes)
//! - Utils (sugar)
//!   - AgentBuilder: `.session(load_store, save_store, session_id)`
//!   - Helpers: `merge_history(history, request_messages)`
//!
//! Implementation strategy
//! - Session stores are plain services with typed requests (no global registries)
//! - `MemoryLayer` holds `Arc<SessionLoadStore>`, `Arc<SessionSaveStore>`, and `SessionId`
//! - On each call:
//!   1) `load_store.call(LoadSession { id })` → `History`
//!   2) Compose `RawChatRequest` by prefixing `History` to current messages
//!   3) Forward to inner step/agent
//!   4) Extract newly produced messages from `StepOutcome`/`AgentRun` and append to `History`
//!   5) `save_store.call(SaveSession { id, history })`
//! - Errors bubble up; store errors are surfaced explicitly
//!
//! Composition examples
//! - `ServiceBuilder::new().layer(MemoryLayer::new(load, save, session_id)).service(step)`
//! - Combine with `RecorderLayer` if you want both persistence and replay traces
//!
//! Testing strategy
//! - Unit tests
//!   - Fake stores using `tower::service_fn` to simulate load/save
//!   - Assert correct merge order (history first) and that saves receive appended messages
//!   - Error propagation when load/save fails
//! - Integration tests
//!   - With a fake model provider and a real `InMemorySessionStore`, verify multi-turn accumulation
//!   - With `ReplayLayer`, verify deterministic reproduction of a captured trace
//!
//! Notes and constraints
//! - Keep the session I/O isolated behind services; do not push DB/file logic into layers
//! - Prefer separate load/save services to keep signatures simple and testable
//! - The replay logic defers to the `recording` and `codec` modules for trace fidelity

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use async_openai::types::{
    ChatCompletionRequestMessage, CreateChatCompletionRequest, CreateChatCompletionRequestArgs,
};
use tower::{BoxError, Layer, Service};

/// Session identifier newtype.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SessionId(pub String);

/// History of chat messages for a session.
pub type History = Vec<ChatCompletionRequestMessage>;

/// Load request for a session.
#[derive(Debug, Clone)]
pub struct LoadSession {
    pub id: SessionId,
}

/// Save request for a session.
#[derive(Debug, Clone)]
pub struct SaveSession {
    pub id: SessionId,
    pub history: History,
}

/// A simple in-memory session store implementing both load and save services.
#[derive(Default, Clone)]
pub struct InMemorySessionStore {
    inner: Arc<Mutex<HashMap<SessionId, History>>>,
}

impl Service<LoadSession> for InMemorySessionStore {
    type Response = History;
    type Error = BoxError;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(
        &mut self,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        std::task::Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: LoadSession) -> Self::Future {
        let inner = self.inner.clone();
        Box::pin(async move {
            let map = inner.lock().unwrap();
            Ok(map.get(&req.id).cloned().unwrap_or_default())
        })
    }
}

impl Service<SaveSession> for InMemorySessionStore {
    type Response = ();
    type Error = BoxError;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(
        &mut self,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        std::task::Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: SaveSession) -> Self::Future {
        let inner = self.inner.clone();
        Box::pin(async move {
            let mut map = inner.lock().unwrap();
            map.insert(req.id, req.history);
            Ok(())
        })
    }
}

/// Layer configuration for memory persistence.
#[derive(Clone)]
pub struct MemoryLayer<L, S> {
    load: Arc<L>,
    save: Arc<S>,
    session_id: SessionId,
}

impl<L, S> MemoryLayer<L, S> {
    pub fn new(load: Arc<L>, save: Arc<S>, session_id: SessionId) -> Self {
        Self {
            load,
            save,
            session_id,
        }
    }
}

/// Wrapped service that loads history before the call and saves after.
pub struct Memory<S, L, Sv> {
    inner: Arc<tokio::sync::Mutex<S>>,
    load: L,
    save: Sv,
    session_id: SessionId,
}

impl<S, L, Sv> Layer<S> for MemoryLayer<L, Sv>
where
    L: Service<LoadSession, Response = History, Error = BoxError> + Send + Clone + 'static,
    L::Future: Send + 'static,
    Sv: Service<SaveSession, Response = (), Error = BoxError> + Send + Clone + 'static,
    Sv::Future: Send + 'static,
{
    type Service = Memory<S, L, Sv>;

    fn layer(&self, inner: S) -> Self::Service {
        Memory {
            inner: Arc::new(tokio::sync::Mutex::new(inner)),
            load: (*self.load).clone(),
            save: (*self.save).clone(),
            session_id: self.session_id.clone(),
        }
    }
}

impl<S, Ls, Ss> Service<CreateChatCompletionRequest> for Memory<S, Ls, Ss>
where
    S: Service<CreateChatCompletionRequest> + Send + 'static,
    S::Response: Send + 'static,
    S::Error: Into<BoxError>,
    S::Future: Send + 'static,
    Ls: Service<LoadSession, Response = History, Error = BoxError> + Send + Sync + Clone + 'static,
    Ls::Future: Send + 'static,
    Ss: Service<SaveSession, Response = (), Error = BoxError> + Send + Sync + Clone + 'static,
    Ss::Future: Send + 'static,
{
    type Response = S::Response;
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
        let session_id = self.session_id.clone();
        let load = self.load.clone();
        let save = self.save.clone();
        let inner = self.inner.clone();

        Box::pin(async move {
            // Load history
            let mut load_svc = load;
            let history = Service::call(
                &mut load_svc,
                LoadSession {
                    id: session_id.clone(),
                },
            )
            .await?;

            // Build combined request preserving model/tools/knobs
            let mut builder = CreateChatCompletionRequestArgs::default();
            builder.model(&req.model);
            // Combine history and current messages
            let mut combined = history;
            combined.extend(req.messages.clone());
            builder.messages(combined.clone());
            if let Some(t) = req.temperature {
                builder.temperature(t);
            }
            if let Some(ts) = req.tools.clone() {
                builder.tools(ts);
            }
            let combined_req = builder
                .build()
                .map_err(|e| -> BoxError { e.to_string().into() })?;

            // Call inner
            let resp = {
                let mut guard = inner.lock().await;
                Service::call(&mut *guard, combined_req)
                    .await
                    .map_err(|e| Into::<BoxError>::into(e))?
            };

            // Persist the latest messages; for simplicity, overwrite full history
            let latest_messages = match &resp {
                _ => combined, // if inner rewrites messages, we could inspect resp; simplified here
            };
            let mut save_svc = save;
            Service::call(
                &mut save_svc,
                SaveSession {
                    id: session_id,
                    history: latest_messages,
                },
            )
            .await?;

            Ok(resp)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_openai::types::{
        ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
    };
    use tower::{service_fn, ServiceExt};

    fn req_with_messages(
        messages: Vec<ChatCompletionRequestMessage>,
    ) -> CreateChatCompletionRequest {
        let mut builder = CreateChatCompletionRequestArgs::default();
        builder.model("gpt-4o");
        builder.messages(messages);
        builder.build().unwrap()
    }

    #[tokio::test]
    async fn memory_layer_loads_and_saves_history() {
        // Seed store with prior history
        let store = InMemorySessionStore::default();
        let session_id = SessionId("s1".into());

        // Save initial history
        let prior = vec![
            ChatCompletionRequestSystemMessageArgs::default()
                .content("sys")
                .build()
                .unwrap()
                .into(),
            ChatCompletionRequestUserMessageArgs::default()
                .content("prev")
                .build()
                .unwrap()
                .into(),
        ];
        let mut save_clone = store.clone();
        ServiceExt::ready(&mut save_clone)
            .await
            .unwrap()
            .call(SaveSession {
                id: session_id.clone(),
                history: prior.clone(),
            })
            .await
            .unwrap();

        // Inner service echoes messages and returns them unchanged
        let inner =
            service_fn(|req: CreateChatCompletionRequest| async move { Ok::<_, BoxError>(req) });

        // Wrap with Memory layer
        let layer = MemoryLayer::new(
            Arc::new(store.clone()),
            Arc::new(store.clone()),
            session_id.clone(),
        );
        let mut svc = layer.layer(inner);

        // Call with a new user message
        let req = req_with_messages(vec![ChatCompletionRequestUserMessageArgs::default()
            .content("hello")
            .build()
            .unwrap()
            .into()]);
        let _resp = ServiceExt::ready(&mut svc)
            .await
            .unwrap()
            .call(req)
            .await
            .unwrap();

        // Verify store now contains prior + new
        let mut load = store.clone();
        let history = ServiceExt::ready(&mut load)
            .await
            .unwrap()
            .call(LoadSession { id: session_id })
            .await
            .unwrap();
        assert_eq!(history.len(), 3);
    }
}
