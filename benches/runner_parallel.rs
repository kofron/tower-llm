use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use openai_agents_rs::{
    model::ModelProvider, runner::RunConfig, Agent, FunctionTool, Runner, Tool,
};
use serde_json::Value;
use std::sync::{Arc, Mutex};
use tokio::runtime::Runtime;

// Simple bench provider that returns N tool calls on the first call, then a final message.
struct BenchProvider {
    name: String,
    remaining: Mutex<i32>,
    tool_calls: usize,
}

#[async_trait::async_trait]
impl ModelProvider for BenchProvider {
    async fn complete(
        &self,
        _messages: Vec<openai_agents_rs::items::Message>,
        _tools: Vec<Arc<dyn Tool>>,
        _temperature: Option<f32>,
        _max_tokens: Option<u32>,
    ) -> openai_agents_rs::Result<(
        openai_agents_rs::items::ModelResponse,
        openai_agents_rs::usage::Usage,
    )> {
        let mut rem = self.remaining.lock().unwrap();
        if *rem > 0 {
            *rem -= 1;
            // emit tool calls
            let calls = (0..self.tool_calls)
                .map(|i| openai_agents_rs::items::ToolCall {
                    id: format!("call_{}", i),
                    name: "slow".to_string(),
                    arguments: serde_json::json!({"input": format!("{}", i)}),
                })
                .collect::<Vec<_>>();
            Ok((
                openai_agents_rs::items::ModelResponse::new_tool_calls(calls),
                openai_agents_rs::usage::Usage::empty(),
            ))
        } else {
            Ok((
                openai_agents_rs::items::ModelResponse::new_message("done"),
                openai_agents_rs::usage::Usage::empty(),
            ))
        }
    }

    fn model_name(&self) -> &str {
        &self.name
    }
}

fn slow_tool() -> Arc<FunctionTool> {
    Arc::new(FunctionTool::new(
        "slow".to_string(),
        "Sleeps briefly".to_string(),
        serde_json::json!({"type":"object","properties":{"input":{"type":"string"}},"required":["input"]}),
        |_args| {
            std::thread::sleep(std::time::Duration::from_millis(10));
            Ok(Value::String("ok".into()))
        },
    ))
}

fn bench_runner(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let agent = Agent::simple("Bench", "Use tools").with_tool(slow_tool());

    let mk_provider = |turns: i32, tc: usize| -> Arc<dyn ModelProvider> {
        Arc::new(BenchProvider {
            name: "bench".into(),
            remaining: Mutex::new(turns),
            tool_calls: tc,
        })
    };

    // Sequential
    c.bench_function("runner_sequential_8tools", |b| {
        b.to_async(&rt).iter_batched(
            || {
                let provider = mk_provider(1, 8);
                RunConfig::default()
                    .with_model_provider(Some(provider))
                    .with_parallel_tools(false)
            },
            |cfg| async {
                let _ = Runner::run(agent.clone(), "run".to_string(), cfg)
                    .await
                    .unwrap();
            },
            BatchSize::SmallInput,
        )
    });

    // Parallel
    c.bench_function("runner_parallel_8tools", |b| {
        b.to_async(&rt).iter_batched(
            || {
                let provider = mk_provider(1, 8);
                RunConfig::default().with_model_provider(Some(provider))
            },
            |cfg| async {
                let _ = Runner::run(agent.clone(), "run".to_string(), cfg)
                    .await
                    .unwrap();
            },
            BatchSize::SmallInput,
        )
    });
}

criterion_group!(benches, bench_runner);
criterion_main!(benches);
