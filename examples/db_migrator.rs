//! # Example: Transactional Database Migrator with Context Handler
//!
//! This example demonstrates a side-effecting workflow where the context handler
//! manages a live SQLite transaction on a dedicated thread. The tool proposes
//! operations (execute SQL, commit, rollback), while the handler applies them to
//! an in-memory database and rewrites outputs with structured status.
//!
//! To run:
//!
//! ```bash
//! export OPENAI_API_KEY="your-api-key"
//! cargo run --example db_migrator
//! ```
//!
//! Expected: The agent should call tools to create a table and add rows.  
//! It might be the case that it does this multiple times - but at the end of the day, it must do at least that.  It is expected
//! that the agent will succeed and be able to select all of the users during its run.

use openai_agents_rs::{
    runner::RunConfig, Agent, ContextStep, ContextualAgent, FunctionTool, RunResultWithContext,
    Runner, ToolContext,
};
use rusqlite::Connection;
use serde_json::Value;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::thread;

#[derive(Debug)]
enum DbCommand {
    Exec(String),
    Commit,
    Rollback,
}

#[derive(Debug)]
enum DbReply {
    Ok { changes: i64 },
    Error { message: String },
    Done,
}

fn start_db_worker() -> (Sender<DbCommand>, Receiver<DbReply>) {
    let (tx_cmd, rx_cmd) = mpsc::channel::<DbCommand>();
    let (tx_rep, rx_rep) = mpsc::channel::<DbReply>();

    thread::spawn(move || {
        // Single-threaded in-memory DB; one transaction per run
        let conn = Connection::open_in_memory();
        let conn = match conn {
            Ok(c) => c,
            Err(e) => {
                let _ = tx_rep.send(DbReply::Error {
                    message: format!("open error: {}", e),
                });
                return;
            }
        };

        if let Err(e) = conn.execute_batch("BEGIN IMMEDIATE;") {
            let _ = tx_rep.send(DbReply::Error {
                message: format!("begin error: {}", e),
            });
            return;
        }

        for cmd in rx_cmd.iter() {
            match cmd {
                DbCommand::Exec(sql) => match conn.execute_batch(&sql) {
                    Ok(_) => {
                        let changes = conn.changes() as i64;
                        let _ = tx_rep.send(DbReply::Ok { changes });
                    }
                    Err(e) => {
                        let _ = tx_rep.send(DbReply::Error {
                            message: e.to_string(),
                        });
                    }
                },
                DbCommand::Commit => {
                    let res = conn.execute_batch("COMMIT;");
                    let _ = tx_rep.send(match res {
                        Ok(_) => DbReply::Done,
                        Err(e) => DbReply::Error {
                            message: e.to_string(),
                        },
                    });
                    break;
                }
                DbCommand::Rollback => {
                    let res = conn.execute_batch("ROLLBACK;");
                    let _ = tx_rep.send(match res {
                        Ok(_) => DbReply::Done,
                        Err(e) => DbReply::Error {
                            message: e.to_string(),
                        },
                    });
                    break;
                }
            }
        }
    });

    (tx_cmd, rx_rep)
}

#[derive(Clone)]
struct DbCtx {
    tx: Sender<DbCommand>,
    rx: Arc<std::sync::Mutex<Receiver<DbReply>>>,
    applied: Vec<String>,
    errors: Vec<String>,
}

struct MigratorHandler;

impl ToolContext<DbCtx> for MigratorHandler {
    fn on_tool_output(
        &self,
        mut ctx: DbCtx,
        _tool_name: &str,
        _arguments: &Value,
        result: Result<Value, String>,
    ) -> openai_agents_rs::Result<ContextStep<DbCtx>> {
        let mut finalize = None;
        if let Ok(Value::Object(map)) = result {
            let op = map.get("op").and_then(|v| v.as_str()).unwrap_or("");
            match op {
                "exec" => {
                    if let Some(sql) = map.get("sql").and_then(|v| v.as_str()) {
                        let _ = ctx.tx.send(DbCommand::Exec(sql.to_string()));
                        let rep = ctx.rx.lock().unwrap().recv().unwrap();
                        match rep {
                            DbReply::Ok { changes } => {
                                ctx.applied.push(sql.to_string());
                                let rewritten = serde_json::json!({
                                    "status": "ok",
                                    "changes": changes,
                                });
                                return Ok(ContextStep::rewrite(ctx, rewritten));
                            }
                            DbReply::Error { message } => {
                                ctx.errors.push(message.clone());
                                let rewritten = serde_json::json!({
                                    "status": "error",
                                    "message": message,
                                });
                                return Ok(ContextStep::rewrite(ctx, rewritten));
                            }
                            DbReply::Done => {}
                        }
                    }
                }
                "commit" => finalize = Some(DbCommand::Commit),
                "rollback" => finalize = Some(DbCommand::Rollback),
                _ => {}
            }
        }

        if let Some(cmd) = finalize {
            let _ = ctx.tx.send(cmd);
            let rep = ctx.rx.lock().unwrap().recv().unwrap();
            let (status, message) = match rep {
                DbReply::Done => ("committed_or_rolled_back", None),
                DbReply::Error { message } => ("error", Some(message)),
                DbReply::Ok { .. } => ("unexpected", None),
            };
            let summary = serde_json::json!({
                "status": status,
                "message": message,
                "applied": ctx.applied,
                "errors": ctx.errors,
            });
            return Ok(ContextStep::final_output(ctx, summary));
        }

        // No-op
        Ok(ContextStep::rewrite(
            ctx,
            serde_json::json!({"status": "noop"}),
        ))
    }
}

fn migrator_tool() -> Arc<FunctionTool> {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "op": { "type": "string", "enum": ["exec", "commit", "rollback"], "description": "Operation type" },
            "sql": { "type": ["string", "null"], "description": "SQL to execute when op == 'exec'" }
        },
        "required": ["op"]
    });

    let func = |args: Value| -> openai_agents_rs::Result<Value> { Ok(args) };
    Arc::new(FunctionTool::new(
        "migrate".to_string(),
        "Execute SQL statements, or commit/rollback the transaction.".to_string(),
        schema,
        func,
    ))
}

fn build_agent() -> ContextualAgent<DbCtx> {
    let instructions = r#"
You are a cautious database migrator. Use the migrate tool to:
- Execute SQL via {"op":"exec","sql":"..."}
- Commit via {"op":"commit"} or rollback via {"op":"rollback"}
Validate changes (e.g., create tables before inserts). When done, commit.  When you have finished,
select all of the users from the table and return the results to validate that the table was correctly created 
and populated.
"#;

    let factory = || {
        let (tx, rx) = start_db_worker();
        DbCtx {
            tx,
            rx: Arc::new(std::sync::Mutex::new(rx)),
            applied: vec![],
            errors: vec![],
        }
    };

    Agent::simple("DbMigrator", instructions)
        .with_tool(migrator_tool())
        .with_context_factory_typed(factory, MigratorHandler)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Running DB migrator with contextual handler...\n");

    let agent = build_agent();
    let RunResultWithContext { result, context } = Runner::run_with_context(
        agent,
        "Create a 'users' table and insert two rows, then commit.",
        RunConfig::default(),
    )
    .await?;

    if result.is_success() {
        println!("Final Response:\n{}\n", result.final_output);
        println!("Applied Statements: {:?}", context.applied);
        println!("Errors: {:?}", context.errors);
    } else {
        println!("Error: {:?}", result.error());
    }

    Ok(())
}
