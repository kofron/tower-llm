//! # Integration Tests for SQLite Session Storage
//!
//! These tests verify the end-to-end functionality of the `SqliteSession`,
//! ensuring that it can correctly store, retrieve, and manage conversation
//! history in a persistent manner.

use chrono::Utc;
use openai_agents_rs::{
    items::{MessageItem, Role, RunItem},
    memory::Session,
    sqlite_session::SqliteSession,
};
use tempfile::NamedTempFile;

#[tokio::test]
async fn test_sqlite_persistence() {
    // 1. Set up a temporary database file for the test.
    //
    // Using a `NamedTempFile` ensures that the database is created in a
    // temporary directory and is automatically cleaned up when the test finishes.
    let temp_file = NamedTempFile::new().unwrap();
    let db_path = temp_file.path();
    let session_id = "integration_test_user";

    // 2. Create the first session instance and add data to it.
    //
    // This block simulates the first run of the application where the initial
    // conversation takes place.
    {
        let session = SqliteSession::new(session_id, db_path).await.unwrap();
        assert!(
            session.get_items(None).await.unwrap().is_empty(),
            "Session should be initially empty"
        );

        let items = vec![
            RunItem::Message(MessageItem {
                id: "1".to_string(),
                role: Role::User,
                content: "Hello, this is a test".to_string(),
                created_at: Utc::now(),
            }),
            RunItem::Message(MessageItem {
                id: "2".to_string(),
                role: Role::Assistant,
                content: "Hello! I can help you test.".to_string(),
                created_at: Utc::now(),
            }),
        ];

        session.add_items(items).await.unwrap();
        assert_eq!(
            session.get_items(None).await.unwrap().len(),
            2,
            "Should have 2 items after adding"
        );
    }

    // 3. Create a new session instance connected to the same database file.
    //
    // This simulates a subsequent run of the application. The new session
    // instance should be able to load the history from the previous run.
    {
        let session = SqliteSession::new(session_id, db_path).await.unwrap();

        // 4. Verify that the data has been persisted and can be retrieved.
        let retrieved = session.get_items(None).await.unwrap();
        assert_eq!(
            retrieved.len(),
            2,
            "Should retrieve 2 items from the persistent storage"
        );

        if let RunItem::Message(msg) = &retrieved[0] {
            assert_eq!(msg.content, "Hello, this is a test");
            assert_eq!(msg.role, Role::User);
        }

        if let RunItem::Message(msg) = &retrieved[1] {
            assert_eq!(msg.content, "Hello! I can help you test.");
            assert_eq!(msg.role, Role::Assistant);
        }

        // 5. Verify that the retrieved items can be correctly converted to messages.
        let messages = session.get_messages(None).await.unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].content, "Hello, this is a test");
        assert_eq!(messages[1].content, "Hello! I can help you test.");
    }
}

#[tokio::test]
async fn test_sqlite_multiple_sessions_in_same_db() {
    // 1. Set up a single temporary database file for the test.
    //
    // This test will verify that two different sessions can coexist in the
    // same database file without interfering with each other.
    let temp_file = NamedTempFile::new().unwrap();
    let db_path = temp_file.path();

    // 2. Create two different sessions that use the same database file.
    let session1 = SqliteSession::new("alice", db_path).await.unwrap();
    let session2 = SqliteSession::new("bob", db_path).await.unwrap();

    // 3. Add different data to each session.
    session1
        .add_items(vec![RunItem::Message(MessageItem {
            id: "a1".to_string(),
            role: Role::User,
            content: "Alice's message".to_string(),
            created_at: Utc::now(),
        })])
        .await
        .unwrap();

    session2
        .add_items(vec![RunItem::Message(MessageItem {
            id: "b1".to_string(),
            role: Role::User,
            content: "Bob's message".to_string(),
            created_at: Utc::now(),
        })])
        .await
        .unwrap();

    // 4. Verify the isolation between the two sessions.
    //
    // We retrieve the items for each session and assert that each session
    // only contains its own data.
    let alice_items = session1.get_items(None).await.unwrap();
    let bob_items = session2.get_items(None).await.unwrap();

    assert_eq!(alice_items.len(), 1);
    assert_eq!(bob_items.len(), 1);

    if let RunItem::Message(msg) = &alice_items[0] {
        assert_eq!(msg.content, "Alice's message");
    }

    if let RunItem::Message(msg) = &bob_items[0] {
        assert_eq!(msg.content, "Bob's message");
    }
}
