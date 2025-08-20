use async_openai::types::{CreateChatCompletionRequest, CreateChatCompletionRequestArgs};
use tower::{Layer, Service, ServiceExt};

// Core module is now at root level
// use tower_llm directly

// A dummy step service that returns Next once, then Done.
#[derive(Clone, Default)]
struct DummyStep {
    turns: usize,
}

impl Service<CreateChatCompletionRequest> for DummyStep {
    type Response = tower_llm::StepOutcome;
    type Error = tower::BoxError;
    type Future = std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Self::Response, Self::Error>> + Send>,
    >;

    fn poll_ready(
        &mut self,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        std::task::Poll::Ready(Ok(()))
    }

    fn call(&mut self, _req: CreateChatCompletionRequest) -> Self::Future {
        let is_first = self.turns == 0;
        self.turns += 1;
        Box::pin(async move {
            if is_first {
                Ok(tower_llm::StepOutcome::Next {
                    messages: vec![],
                    aux: Default::default(),
                    invoked_tools: vec![],
                })
            } else {
                Ok(tower_llm::StepOutcome::Done {
                    messages: vec![],
                    aux: Default::default(),
                })
            }
        })
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let policy = tower_llm::Policy::new()
        .or_max_steps(3)
        .until_no_tool_calls()
        .build();
    let agent = tower_llm::AgentLoopLayer::new(policy).layer(DummyStep::default());

    let req = CreateChatCompletionRequestArgs::default()
        .model("gpt-4o")
        .messages(vec![])
        .build()?;

    let mut agent = agent;
    let run: tower_llm::AgentRun = ServiceExt::ready(&mut agent).await?.call(req).await?;
    println!("steps={} stop={:?}", run.steps, run.stop);
    Ok(())
}
