//! Session memory management for maintaining conversation history
//! 
//! Sessions allow agents to maintain context across multiple runs.

use async_trait::async_trait;
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::{Arc, Mutex};

use crate::error::Result;
use crate::items::{Message, RunItem};

/// Trait for session storage implementations
#[async_trait]
pub trait Session: Send + Sync + Debug {
    /// Get the session ID
    fn session_id(&self) -> &str;

    /// Retrieve conversation history
    async fn get_items(&self, limit: Option<usize>) -> Result<Vec<RunItem>>;

    /// Add new items to the session
    async fn add_items(&self, items: Vec<RunItem>) -> Result<()>;

    /// Remove and return the most recent item
    async fn pop_item(&self) -> Result<Option<RunItem>>;

    /// Clear all items in the session
    async fn clear_session(&self) -> Result<()>;

    /// Get messages formatted for the conversation
    async fn get_messages(&self, limit: Option<usize>) -> Result<Vec<Message>> {
        let items = self.get_items(limit).await?;
        Ok(crate::items::ItemHelpers::to_messages(&items))
    }
}

/// In-memory session storage (for testing and simple use cases)
#[derive(Debug, Clone)]
pub struct MemorySession {
    session_id: String,
    items: Arc<Mutex<Vec<RunItem>>>,
}

impl MemorySession {
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

/// SQLite-based session storage
/// Note: This is a simplified mock implementation. 
/// In production, you'd use actual SQLite with sqlx or rusqlite.
#[derive(Debug)]
pub struct SqliteSession {
    session_id: String,
    db_path: String,
    // In a real implementation, this would be a connection pool
    storage: Arc<Mutex<HashMap<String, Vec<RunItem>>>>,
}

impl SqliteSession {
    pub fn new(session_id: impl Into<String>, db_path: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            db_path: db_path.into(),
            storage: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn new_default(session_id: impl Into<String>) -> Self {
        Self::new(session_id, "sessions.db")
    }
}

#[async_trait]
impl Session for SqliteSession {
    fn session_id(&self) -> &str {
        &self.session_id
    }

    async fn get_items(&self, limit: Option<usize>) -> Result<Vec<RunItem>> {
        let storage = self.storage.lock().unwrap();
        let items = storage.get(&self.session_id).cloned().unwrap_or_default();
        
        let result = if let Some(limit) = limit {
            let start = items.len().saturating_sub(limit);
            items[start..].to_vec()
        } else {
            items
        };
        Ok(result)
    }

    async fn add_items(&self, new_items: Vec<RunItem>) -> Result<()> {
        let mut storage = self.storage.lock().unwrap();
        let items = storage.entry(self.session_id.clone()).or_insert_with(Vec::new);
        items.extend(new_items);
        Ok(())
    }

    async fn pop_item(&self) -> Result<Option<RunItem>> {
        let mut storage = self.storage.lock().unwrap();
        if let Some(items) = storage.get_mut(&self.session_id) {
            Ok(items.pop())
        } else {
            Ok(None)
        }
    }

    async fn clear_session(&self) -> Result<()> {
        let mut storage = self.storage.lock().unwrap();
        storage.remove(&self.session_id);
        Ok(())
    }
}

/// Session manager for handling multiple sessions
#[derive(Debug)]
pub struct SessionManager {
    sessions: Arc<Mutex<HashMap<String, Arc<dyn Session>>>>,
}

impl SessionManager {
    pub fn new() -> Self {
        Self {
            sessions: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Register a session
    pub fn register(&self, session: Arc<dyn Session>) {
        let mut sessions = self.sessions.lock().unwrap();
        sessions.insert(session.session_id().to_string(), session);
    }

    /// Get a session by ID
    pub fn get(&self, session_id: &str) -> Option<Arc<dyn Session>> {
        let sessions = self.sessions.lock().unwrap();
        sessions.get(session_id).cloned()
    }

    /// Remove a session
    pub fn remove(&self, session_id: &str) -> Option<Arc<dyn Session>> {
        let mut sessions = self.sessions.lock().unwrap();
        sessions.remove(session_id)
    }

    /// List all session IDs
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
        let session = SqliteSession::new_default("user_123");
        assert_eq!(session.session_id(), "user_123");

        let items = vec![
            RunItem::Message(MessageItem {
                id: "1".to_string(),
                role: Role::User,
                content: "Test message".to_string(),
                created_at: Utc::now(),
            }),
        ];

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
        
        // Only Message items should be converted
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, Role::User);
        assert_eq!(messages[0].content, "What's the weather?");
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

        let items1 = vec![
            RunItem::Message(MessageItem {
                id: "1".to_string(),
                role: Role::User,
                content: "Session 1 message".to_string(),
                created_at: Utc::now(),
            }),
        ];

        let items2 = vec![
            RunItem::Message(MessageItem {
                id: "2".to_string(),
                role: Role::User,
                content: "Session 2 message".to_string(),
                created_at: Utc::now(),
            }),
        ];

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
