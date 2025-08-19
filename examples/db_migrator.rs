//! # Example: Database Migrator with Tower Layers
//!
//! This example demonstrates a database migration workflow using Tower layers
//! to manage SQLite transactions. Tools propose operations (execute SQL, commit,
//! rollback) which are executed against an in-memory database.
//!
//! To run:
//!
//! ```bash
//! export OPENAI_API_KEY="your-api-key"
//! cargo run --example db_migrator
//! ```

use openai_agents_rs::{error::AgentsError, runner::RunConfig, Agent, FunctionTool, Runner};
use rusqlite::Connection;
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};

/// Shared database connection wrapped in Arc<Mutex> for thread safety
struct DbState {
    conn: Arc<Mutex<Connection>>,
}

impl DbState {
    fn new() -> Self {
        let conn = Connection::open_in_memory().expect("Failed to create in-memory database");
        Self {
            conn: Arc::new(Mutex::new(conn)),
        }
    }

    fn execute_sql(&self, sql: &str) -> Result<Value, AgentsError> {
        let conn = self.conn.lock().unwrap();
        match conn.execute(sql, []) {
            Ok(rows_affected) => Ok(json!({
                "status": "success",
                "rows_affected": rows_affected,
                "message": format!("Executed: {}", sql)
            })),
            Err(e) => Err(AgentsError::Other(format!("SQL error: {}", e))),
        }
    }

    fn query_sql(&self, sql: &str) -> Result<Value, AgentsError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare(sql)
            .map_err(|e| AgentsError::Other(format!("Prepare error: {}", e)))?;

        let column_count = stmt.column_count();
        let column_names: Vec<String> = (0..column_count)
            .map(|i| stmt.column_name(i).unwrap_or("unknown").to_string())
            .collect();

        let rows = stmt
            .query_map([], |row| {
                let mut row_data = json!({});
                for (i, name) in column_names.iter().enumerate() {
                    let value: String = row.get(i).unwrap_or_default();
                    row_data[name] = json!(value);
                }
                Ok(row_data)
            })
            .map_err(|e| AgentsError::Other(format!("Query error: {}", e)))?;

        let results: Result<Vec<Value>, _> = rows.collect();
        let results = results.map_err(|e| AgentsError::Other(format!("Row error: {}", e)))?;

        Ok(json!({
            "status": "success",
            "rows": results,
            "count": results.len()
        }))
    }
}

fn create_db_tools(db_state: Arc<DbState>) -> Vec<Arc<dyn openai_agents_rs::Tool>> {
    let db_state_execute = db_state.clone();
    let execute_tool = FunctionTool::new(
        "execute_sql".to_string(),
        "Execute SQL statement (CREATE, INSERT, UPDATE, DELETE)".to_string(),
        json!({
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "SQL statement to execute"
                }
            },
            "required": ["sql"]
        }),
        move |args: Value| {
            let sql = args["sql"].as_str().unwrap_or("");
            db_state_execute.execute_sql(sql)
        },
    );

    let db_state_query = db_state.clone();
    let query_tool = FunctionTool::new(
        "query_sql".to_string(),
        "Query the database (SELECT statements)".to_string(),
        json!({
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "SQL SELECT statement"
                }
            },
            "required": ["sql"]
        }),
        move |args: Value| {
            let sql = args["sql"].as_str().unwrap_or("");
            db_state_query.query_sql(sql)
        },
    );

    let db_state_schema = db_state.clone();
    let schema_tool = FunctionTool::new(
        "get_schema".to_string(),
        "Get the current database schema".to_string(),
        json!({
            "type": "object",
            "properties": {}
        }),
        move |_args: Value| {
            let sql = "SELECT name, sql FROM sqlite_master WHERE type='table'";
            db_state_schema.query_sql(sql)
        },
    );

    vec![
        Arc::new(execute_tool),
        Arc::new(query_tool),
        Arc::new(schema_tool),
    ]
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create database state
    let db_state = Arc::new(DbState::new());

    // Create tools with database access
    let tools = create_db_tools(db_state.clone());

    // Create agent with database tools
    let mut agent = Agent::simple(
        "DatabaseMigrator",
        "You are a database migration assistant. Your task is to:
1. Create a 'users' table with columns: id (INTEGER PRIMARY KEY), name (TEXT), email (TEXT)
2. Insert at least 3 sample users
3. Query all users to verify the data

Use the provided SQL tools to accomplish this task. Start by checking the schema, then create the table, insert data, and finally query to verify."
    );

    for tool in tools {
        agent = agent.with_tool(tool);
    }

    // Run the agent
    println!("Starting database migration...\n");

    let config = RunConfig::default();
    let result = Runner::run(
        agent,
        "Please set up the users table with sample data as described in your instructions.",
        config,
    )
    .await?;

    if result.is_success() {
        println!("\n✅ Migration completed successfully!");
        println!("Final output: {}", result.final_output);

        // Verify the final state
        println!("\n📊 Final database state:");
        match db_state.query_sql("SELECT * FROM users") {
            Ok(result) => {
                println!("{}", serde_json::to_string_pretty(&result)?);
            }
            Err(e) => {
                println!("Could not query users table: {:?}", e);
            }
        }
    } else {
        println!("\n❌ Migration failed");
        if let Some(error) = result.error {
            println!("Error: {}", error);
        }
    }

    Ok(())
}
