//! Session trait for compatibility with SqliteSession

use async_trait::async_trait;
use std::fmt::Debug;

use crate::error::Result;
use crate::items::{Message, RunItem};

/// Defines the interface for session storage implementations.
#[async_trait]
pub trait Session: Send + Sync + Debug {
    /// Returns the unique identifier for the session.
    fn session_id(&self) -> &str;

    /// Retrieves a list of `RunItem`s from the session.
    async fn get_items(&self, limit: Option<usize>) -> Result<Vec<RunItem>>;

    /// Adds a vector of `RunItem`s to the session.
    async fn add_items(&self, items: Vec<RunItem>) -> Result<()>;

    /// Removes and returns the most recent `RunItem` from the session.
    async fn pop_item(&self) -> Result<Option<RunItem>>;

    /// Clears all items from the session, effectively resetting the conversation.
    async fn clear_session(&self) -> Result<()>;

    /// Retrieves the conversation history as a vector of `Message`s.
    async fn get_messages(&self, limit: Option<usize>) -> Result<Vec<Message>> {
        let items = self.get_items(limit).await?;
        Ok(crate::items::ItemHelpers::to_messages(&items))
    }
}
