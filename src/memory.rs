//! # Session Memory Management
//!
//! The session management system is responsible for maintaining the state of
//! conversations, allowing agents to retain context across multiple interactions.
//! This is crucial for building conversational agents that can remember previous
//! messages and actions.
//!
//! The core of this system is the [`Session`] trait, which defines a standard
//! interface for session storage. This crate provides two implementations:
//!
//! - [`MemorySession`]: An in-memory session store suitable for testing and simple
//!   use cases where persistence is not required.
//! - [`SqliteSession`]: A persistent session store that uses SQLite to save
//!   conversation history to a file.
//!
//! ## The `Session` Trait
//!
//! The [`Session`] trait provides methods for adding, retrieving, and clearing
//! conversation items (`RunItem`). It ensures that different session storage
//! implementations can be used interchangeably.
//!
//! Implementors of this trait are responsible for storing and retrieving the
//! sequence of `RunItem`s that make up a conversation.
use async_trait::async_trait;
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::{Arc, Mutex};

use crate::error::Result;
use crate::items::{Message, RunItem};

/// Defines the interface for session storage implementations.
#[async_trait]
pub trait Session: Send + Sync + Debug {
    /// Returns the unique identifier for the session.
    fn session_id(&self) -> &str;

    /// Retrieves a list of `RunItem`s from the session.
    ///
    /// # Arguments
    ///
    /// * `limit` - An optional `usize` to limit the number of items returned.
    ///   If `None`, all items are retrieved.
    async fn get_items(&self, limit: Option<usize>) -> Result<Vec<RunItem>>;

    /// Adds a vector of `RunItem`s to the session.
    ///
    /// This method is used to persist new events in the conversation, such as
    /// user messages, agent responses, and tool calls.
    async fn add_items(&self, items: Vec<RunItem>) -> Result<()>;

    /// Removes and returns the most recent `RunItem` from the session.
    ///
    /// This can be useful for implementing "undo" functionality or for
    /// correcting the last action in a conversation.
    async fn pop_item(&self) -> Result<Option<RunItem>>;

    /// Clears all items from the session, effectively resetting the conversation.
    async fn clear_session(&self) -> Result<()>;

    /// Retrieves the conversation history as a vector of `Message`s.
    ///
    /// This method converts the stored `RunItem`s into the `Message` format
    /// that is expected by the LLM for generating the next response.
    async fn get_messages(&self, limit: Option<usize>) -> Result<Vec<Message>> {
        let items = self.get_items(limit).await?;
        Ok(crate::items::ItemHelpers::to_messages(&items))
    }
}

/// An in-memory session storage implementation, useful for testing and simple
/// applications where persistence is not required.
///
/// `MemorySession` stores the entire conversation history in a `Vec<RunItem>`
/// protected by a `Mutex`, ensuring thread-safe access.
#[derive(Debug, Clone)]
pub struct MemorySession {
    session_id: String,
    items: Arc<Mutex<Vec<RunItem>>>,
}

impl MemorySession {
    /// Creates a new `MemorySession` with the given session ID.
    pub fn new(session_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            items: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

#[async_trait]
impl Session for MemorySession {
    fn session_id(&self) -> &str {
        &self.session_id
    }

    async fn get_items(&self, limit: Option<usize>) -> Result<Vec<RunItem>> {
        let items = self.items.lock().unwrap();
        let result = if let Some(limit) = limit {
            let start = items.len().saturating_sub(limit);
            items[start..].to_vec()
        } else {
            items.clone()
        };
        Ok(result)
    }

    async fn add_items(&self, new_items: Vec<RunItem>) -> Result<()> {
        let mut items = self.items.lock().unwrap();
        items.extend(new_items);
        Ok(())
    }

    async fn pop_item(&self) -> Result<Option<RunItem>> {
        let mut items = self.items.lock().unwrap();
        Ok(items.pop())
    }

    async fn clear_session(&self) -> Result<()> {
        let mut items = self.items.lock().unwrap();
        items.clear();
        Ok(())
    }
}

/// Manages multiple, isolated sessions in a thread-safe manner.
///
/// `SessionManager` acts as a registry for different sessions, allowing you to
/// store, retrieve, and manage them by their unique session IDs. This is
/// essential for applications that need to handle multiple concurrent
/// conversations, such as a web server.
///
/// ## Example: Managing Multiple User Sessions
///
/// ```rust
/// use openai_agents_rs::memory::{SessionManager, MemorySession};
/// use std::sync::Arc;
///
/// let manager = SessionManager::new();
///
/// // Create and register sessions for two different users.
/// let user1_session = Arc::new(MemorySession::new("user1"));
/// let user2_session = Arc::new(MemorySession::new("user2"));
///
/// manager.register(user1_session);
/// manager.register(user2_session);
///
/// // Retrieve a specific session.
/// let session = manager.get("user1").unwrap();
/// assert_eq!(session.session_id(), "user1");
///
/// // List all active sessions.
/// let session_ids = manager.list_sessions();
/// assert_eq!(session_ids.len(), 2);
/// assert!(session_ids.contains(&"user1".to_string()));
/// ```
#[derive(Debug)]
pub struct SessionManager {
    sessions: Arc<Mutex<HashMap<String, Arc<dyn Session>>>>,
}

impl SessionManager {
    /// Creates a new, empty `SessionManager`.
    pub fn new() -> Self {
        Self {
            sessions: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Registers a session with the manager.
    ///
    /// The session is stored in a `HashMap` with its session ID as the key.
    pub fn register(&self, session: Arc<dyn Session>) {
        let mut sessions = self.sessions.lock().unwrap();
        sessions.insert(session.session_id().to_string(), session);
    }

    /// Retrieves a session by its ID.
    ///
    /// Returns an `Option<Arc<dyn Session>>`, with `None` if no session with
    /// the given ID is found.
    pub fn get(&self, session_id: &str) -> Option<Arc<dyn Session>> {
        let sessions = self.sessions.lock().unwrap();
        sessions.get(session_id).cloned()
    }

    /// Removes a session from the manager by its ID.
    ///
    /// Returns the removed session if it existed.
    pub fn remove(&self, session_id: &str) -> Option<Arc<dyn Session>> {
        let mut sessions = self.sessions.lock().unwrap();
        sessions.remove(session_id)
    }

    /// Returns a list of all registered session IDs.
    pub fn list_sessions(&self) -> Vec<String> {
        let sessions = self.sessions.lock().unwrap();
        sessions.keys().cloned().collect()
    }
}

impl Default for SessionManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::items::{MessageItem, Role, ToolCallItem};
    use chrono::Utc;

    #[tokio::test]
    async fn test_memory_session() {
        let session = MemorySession::new("test_session");
        assert_eq!(session.session_id(), "test_session");

        // Test adding items
        let items = vec![
            RunItem::Message(MessageItem {
                id: "1".to_string(),
                role: Role::User,
                content: "Hello".to_string(),
                created_at: Utc::now(),
            }),
            RunItem::Message(MessageItem {
                id: "2".to_string(),
                role: Role::Assistant,
                content: "Hi there!".to_string(),
                created_at: Utc::now(),
            }),
        ];

        session.add_items(items.clone()).await.unwrap();

        // Test getting items
        let retrieved = session.get_items(None).await.unwrap();
        assert_eq!(retrieved.len(), 2);

        // Test getting with limit
        let limited = session.get_items(Some(1)).await.unwrap();
        assert_eq!(limited.len(), 1);

        // Test pop
        let popped = session.pop_item().await.unwrap();
        assert!(popped.is_some());

        let remaining = session.get_items(None).await.unwrap();
        assert_eq!(remaining.len(), 1);

        // Test clear
        session.clear_session().await.unwrap();
        let cleared = session.get_items(None).await.unwrap();
        assert!(cleared.is_empty());
    }

    #[tokio::test]
    async fn test_sqlite_session() {
        // Test moved to sqlite_session.rs with real implementation
        // This test now uses the real SqliteSession from sqlite_session module
        use crate::sqlite_session::SqliteSession;

        let session = SqliteSession::new_in_memory("user_123").await.unwrap();
        assert_eq!(session.session_id(), "user_123");

        let items = vec![RunItem::Message(MessageItem {
            id: "1".to_string(),
            role: Role::User,
            content: "Test message".to_string(),
            created_at: Utc::now(),
        })];

        session.add_items(items).await.unwrap();
        let retrieved = session.get_items(None).await.unwrap();
        assert_eq!(retrieved.len(), 1);
    }

    #[tokio::test]
    async fn test_session_get_messages() {
        let session = MemorySession::new("msg_test");

        let items = vec![
            RunItem::Message(MessageItem {
                id: "1".to_string(),
                role: Role::User,
                content: "What's the weather?".to_string(),
                created_at: Utc::now(),
            }),
            RunItem::ToolCall(ToolCallItem {
                id: "2".to_string(),
                tool_name: "get_weather".to_string(),
                arguments: serde_json::json!({"city": "Tokyo"}),
                created_at: Utc::now(),
            }),
        ];

        session.add_items(items).await.unwrap();
        let messages = session.get_messages(None).await.unwrap();

        // Message items and tool calls should be converted
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, Role::User);
        assert_eq!(messages[0].content, "What's the weather?");
        // Tool calls create an assistant message with tool_calls
        assert_eq!(messages[1].role, Role::Assistant);
        assert!(messages[1].tool_calls.is_some());
    }

    #[tokio::test]
    async fn test_session_manager() {
        let manager = SessionManager::new();

        let session1 = Arc::new(MemorySession::new("session1"));
        let session2 = Arc::new(MemorySession::new("session2"));

        manager.register(session1.clone());
        manager.register(session2.clone());

        // Test listing
        let sessions = manager.list_sessions();
        assert_eq!(sessions.len(), 2);
        assert!(sessions.contains(&"session1".to_string()));
        assert!(sessions.contains(&"session2".to_string()));

        // Test getting
        let retrieved = manager.get("session1");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().session_id(), "session1");

        // Test removing
        let removed = manager.remove("session1");
        assert!(removed.is_some());

        let sessions_after = manager.list_sessions();
        assert_eq!(sessions_after.len(), 1);
        assert!(!sessions_after.contains(&"session1".to_string()));
    }

    #[tokio::test]
    async fn test_multiple_sessions_isolation() {
        let session1 = MemorySession::new("user1");
        let session2 = MemorySession::new("user2");

        let items1 = vec![RunItem::Message(MessageItem {
            id: "1".to_string(),
            role: Role::User,
            content: "Session 1 message".to_string(),
            created_at: Utc::now(),
        })];

        let items2 = vec![RunItem::Message(MessageItem {
            id: "2".to_string(),
            role: Role::User,
            content: "Session 2 message".to_string(),
            created_at: Utc::now(),
        })];

        session1.add_items(items1).await.unwrap();
        session2.add_items(items2).await.unwrap();

        let retrieved1 = session1.get_items(None).await.unwrap();
        let retrieved2 = session2.get_items(None).await.unwrap();

        assert_eq!(retrieved1.len(), 1);
        assert_eq!(retrieved2.len(), 1);

        // Verify isolation
        if let RunItem::Message(msg) = &retrieved1[0] {
            assert_eq!(msg.content, "Session 1 message");
        }
        if let RunItem::Message(msg) = &retrieved2[0] {
            assert_eq!(msg.content, "Session 2 message");
        }
    }
}
