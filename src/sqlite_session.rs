//! SQLite-based session storage implementation
//!
//! Provides persistent storage for conversation sessions using SQLite.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use sqlx::{Pool, Row, Sqlite, SqlitePool};
use std::path::Path;

use crate::error::{AgentsError, Result};
use crate::items::{
    HandoffItem, Message, MessageItem, Role, RunItem, ToolCallItem, ToolOutputItem,
};
use crate::memory::Session;

/// SQLite-based session storage
pub struct SqliteSession {
    session_id: String,
    pool: Pool<Sqlite>,
}

impl SqliteSession {
    /// Create a new SQLite session with a specific database path
    pub async fn new(session_id: impl Into<String>, db_path: impl AsRef<Path>) -> Result<Self> {
        let session_id = session_id.into();
        let db_url = format!("sqlite:{}", db_path.as_ref().display());

        // Create connection pool
        let pool = SqlitePool::connect(&db_url).await?;

        // Run migrations to create tables
        Self::run_migrations(&pool).await?;

        Ok(Self { session_id, pool })
    }

    /// Create a new SQLite session with the default database
    pub async fn new_default(session_id: impl Into<String>) -> Result<Self> {
        Self::new(session_id, "sessions.db").await
    }

    /// Create an in-memory SQLite session (useful for testing)
    pub async fn new_in_memory(session_id: impl Into<String>) -> Result<Self> {
        let session_id = session_id.into();
        let pool = SqlitePool::connect("sqlite::memory:").await?;

        // Run migrations
        Self::run_migrations(&pool).await?;

        Ok(Self { session_id, pool })
    }

    /// Run database migrations to create necessary tables
    async fn run_migrations(pool: &Pool<Sqlite>) -> Result<()> {
        // Create sessions table
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                item_type TEXT NOT NULL,
                item_data TEXT NOT NULL,
                created_at TEXT NOT NULL,
                sequence_num INTEGER NOT NULL,
                UNIQUE(session_id, sequence_num)
            )
            "#,
        )
        .execute(pool)
        .await?;

        // Create index for efficient queries
        sqlx::query(
            r#"
            CREATE INDEX IF NOT EXISTS idx_session_id 
            ON sessions(session_id, sequence_num)
            "#,
        )
        .execute(pool)
        .await?;

        Ok(())
    }

    /// Serialize a RunItem to JSON
    fn serialize_item(item: &RunItem) -> Result<String> {
        Ok(serde_json::to_string(item)?)
    }

    /// Deserialize a RunItem from JSON
    fn deserialize_item(data: &str) -> Result<RunItem> {
        Ok(serde_json::from_str(data)?)
    }

    /// Get the item type as a string for database storage
    fn get_item_type(item: &RunItem) -> &'static str {
        match item {
            RunItem::Message(_) => "message",
            RunItem::ToolCall(_) => "tool_call",
            RunItem::ToolOutput(_) => "tool_output",
            RunItem::Handoff(_) => "handoff",
        }
    }
}

#[async_trait]
impl Session for SqliteSession {
    fn session_id(&self) -> &str {
        &self.session_id
    }

    async fn get_items(&self, limit: Option<usize>) -> Result<Vec<RunItem>> {
        let query = if let Some(limit) = limit {
            sqlx::query(
                r#"
                SELECT item_data 
                FROM sessions 
                WHERE session_id = ?
                ORDER BY sequence_num DESC
                LIMIT ?
                "#,
            )
            .bind(&self.session_id)
            .bind(limit as i64)
        } else {
            sqlx::query(
                r#"
                SELECT item_data 
                FROM sessions 
                WHERE session_id = ?
                ORDER BY sequence_num ASC
                "#,
            )
            .bind(&self.session_id)
        };

        let rows = query.fetch_all(&self.pool).await?;

        let mut items = Vec::new();
        for row in rows {
            let data: String = row.get("item_data");
            items.push(Self::deserialize_item(&data)?);
        }

        // If we had a limit, we need to reverse since we selected in DESC order
        if limit.is_some() {
            items.reverse();
        }

        Ok(items)
    }

    async fn add_items(&self, items: Vec<RunItem>) -> Result<()> {
        // Get the current max sequence number
        let max_seq: Option<i64> = sqlx::query_scalar(
            r#"
            SELECT MAX(sequence_num) 
            FROM sessions 
            WHERE session_id = ?
            "#,
        )
        .bind(&self.session_id)
        .fetch_one(&self.pool)
        .await?;

        let mut sequence_num = max_seq.unwrap_or(0) + 1;

        // Insert each item
        for item in items {
            let item_type = Self::get_item_type(&item);
            let item_data = Self::serialize_item(&item)?;
            let created_at = Utc::now().to_rfc3339();

            sqlx::query(
                r#"
                INSERT INTO sessions (session_id, item_type, item_data, created_at, sequence_num)
                VALUES (?, ?, ?, ?, ?)
                "#,
            )
            .bind(&self.session_id)
            .bind(item_type)
            .bind(item_data)
            .bind(created_at)
            .bind(sequence_num)
            .execute(&self.pool)
            .await?;

            sequence_num += 1;
        }

        Ok(())
    }

    async fn pop_item(&self) -> Result<Option<RunItem>> {
        // Start a transaction
        let mut tx = self.pool.begin().await?;

        // Get the last item
        let row = sqlx::query(
            r#"
            SELECT id, item_data 
            FROM sessions 
            WHERE session_id = ?
            ORDER BY sequence_num DESC
            LIMIT 1
            "#,
        )
        .bind(&self.session_id)
        .fetch_optional(&mut *tx)
        .await?;

        if let Some(row) = row {
            let id: i64 = row.get("id");
            let data: String = row.get("item_data");

            // Delete the item
            sqlx::query("DELETE FROM sessions WHERE id = ?")
                .bind(id)
                .execute(&mut *tx)
                .await?;

            // Commit the transaction
            tx.commit().await?;

            Ok(Some(Self::deserialize_item(&data)?))
        } else {
            Ok(None)
        }
    }

    async fn clear_session(&self) -> Result<()> {
        sqlx::query("DELETE FROM sessions WHERE session_id = ?")
            .bind(&self.session_id)
            .execute(&self.pool)
            .await?;

        Ok(())
    }
}

impl std::fmt::Debug for SqliteSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SqliteSession")
            .field("session_id", &self.session_id)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::items::ToolCall;

    #[tokio::test]
    async fn test_sqlite_session_basic() {
        let session = SqliteSession::new_in_memory("test_session").await.unwrap();

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

        // Verify content
        if let RunItem::Message(msg) = &retrieved[0] {
            assert_eq!(msg.content, "Hello");
            assert_eq!(msg.role, Role::User);
        } else {
            panic!("Expected Message item");
        }
    }

    #[tokio::test]
    async fn test_sqlite_session_with_limit() {
        let session = SqliteSession::new_in_memory("test_limit").await.unwrap();

        // Add multiple items
        let mut items = vec![];
        for i in 0..5 {
            items.push(RunItem::Message(MessageItem {
                id: format!("{}", i),
                role: Role::User,
                content: format!("Message {}", i),
                created_at: Utc::now(),
            }));
        }

        session.add_items(items).await.unwrap();

        // Get with limit
        let limited = session.get_items(Some(2)).await.unwrap();
        assert_eq!(limited.len(), 2);

        // Should get the last 2 messages
        if let RunItem::Message(msg) = &limited[0] {
            assert_eq!(msg.content, "Message 3");
        }
        if let RunItem::Message(msg) = &limited[1] {
            assert_eq!(msg.content, "Message 4");
        }
    }

    #[tokio::test]
    async fn test_sqlite_session_pop() {
        let session = SqliteSession::new_in_memory("test_pop").await.unwrap();

        let items = vec![
            RunItem::Message(MessageItem {
                id: "1".to_string(),
                role: Role::User,
                content: "First".to_string(),
                created_at: Utc::now(),
            }),
            RunItem::Message(MessageItem {
                id: "2".to_string(),
                role: Role::User,
                content: "Second".to_string(),
                created_at: Utc::now(),
            }),
        ];

        session.add_items(items).await.unwrap();

        // Pop the last item
        let popped = session.pop_item().await.unwrap();
        assert!(popped.is_some());

        if let Some(RunItem::Message(msg)) = popped {
            assert_eq!(msg.content, "Second");
        }

        // Verify only one item remains
        let remaining = session.get_items(None).await.unwrap();
        assert_eq!(remaining.len(), 1);
    }

    #[tokio::test]
    async fn test_sqlite_session_clear() {
        let session = SqliteSession::new_in_memory("test_clear").await.unwrap();

        let items = vec![RunItem::Message(MessageItem {
            id: "1".to_string(),
            role: Role::User,
            content: "Test".to_string(),
            created_at: Utc::now(),
        })];

        session.add_items(items).await.unwrap();

        // Clear the session
        session.clear_session().await.unwrap();

        // Verify it's empty
        let remaining = session.get_items(None).await.unwrap();
        assert!(remaining.is_empty());
    }

    #[tokio::test]
    async fn test_sqlite_session_complex_items() {
        let session = SqliteSession::new_in_memory("test_complex").await.unwrap();

        let items = vec![
            RunItem::Message(MessageItem {
                id: "1".to_string(),
                role: Role::User,
                content: "Calculate something".to_string(),
                created_at: Utc::now(),
            }),
            RunItem::ToolCall(ToolCallItem {
                id: "2".to_string(),
                tool_name: "calculator".to_string(),
                arguments: serde_json::json!({"a": 1, "b": 2}),
                created_at: Utc::now(),
            }),
            RunItem::ToolOutput(ToolOutputItem {
                id: "3".to_string(),
                tool_call_id: "2".to_string(),
                output: serde_json::json!(3),
                error: None,
                created_at: Utc::now(),
            }),
            RunItem::Handoff(HandoffItem {
                id: "4".to_string(),
                from_agent: "Main".to_string(),
                to_agent: "Specialist".to_string(),
                reason: Some("Complex calculation".to_string()),
                created_at: Utc::now(),
            }),
        ];

        session.add_items(items.clone()).await.unwrap();

        let retrieved = session.get_items(None).await.unwrap();
        assert_eq!(retrieved.len(), 4);

        // Verify each item type
        assert!(matches!(retrieved[0], RunItem::Message(_)));
        assert!(matches!(retrieved[1], RunItem::ToolCall(_)));
        assert!(matches!(retrieved[2], RunItem::ToolOutput(_)));
        assert!(matches!(retrieved[3], RunItem::Handoff(_)));
    }

    #[tokio::test]
    async fn test_multiple_sessions() {
        // Use an in-memory database with a shared cache for testing multiple sessions
        // Note: For real file-based multi-session testing, you'd need proper temp directory handling
        // For now, we'll test the concept with separate in-memory databases

        let session1 = SqliteSession::new_in_memory("user1").await.unwrap();
        let session2 = SqliteSession::new_in_memory("user2").await.unwrap();

        // Add different items to each session
        session1
            .add_items(vec![RunItem::Message(MessageItem {
                id: "1".to_string(),
                role: Role::User,
                content: "Session 1 message".to_string(),
                created_at: Utc::now(),
            })])
            .await
            .unwrap();

        session2
            .add_items(vec![RunItem::Message(MessageItem {
                id: "2".to_string(),
                role: Role::User,
                content: "Session 2 message".to_string(),
                created_at: Utc::now(),
            })])
            .await
            .unwrap();

        // Verify isolation
        let items1 = session1.get_items(None).await.unwrap();
        let items2 = session2.get_items(None).await.unwrap();

        assert_eq!(items1.len(), 1);
        assert_eq!(items2.len(), 1);

        if let RunItem::Message(msg) = &items1[0] {
            assert_eq!(msg.content, "Session 1 message");
        }
        if let RunItem::Message(msg) = &items2[0] {
            assert_eq!(msg.content, "Session 2 message");
        }

        // No cleanup needed for in-memory databases
    }
}
