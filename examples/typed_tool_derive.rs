use openai_agents_rs::{tool_args, tool_output, TypedFunctionTool, Tool};

#[tool_args]
struct AddArgs {
    x: i32,
    y: i32,
}

#[tool_output]
struct Sum {
    sum: i32,
}

#[tokio::main]
async fn main() {
    let tool: TypedFunctionTool<AddArgs, Sum, _> =
        TypedFunctionTool::new_inferred("add", "Adds two integers", |a: AddArgs| {
            Ok(Sum { sum: a.x + a.y })
        });
    let res = tool
        .execute(serde_json::json!({"x":2, "y":5}))
        .await
        .expect("tool executes");
    println!("{}", res.output);
}


