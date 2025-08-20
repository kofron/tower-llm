//! Service abstractions for the experimental `next` module
//!
//! What this module provides (spec)
//! - Uniform tool invocation services and a router to dispatch them
//! - The one-step model+tools service (Step)
//! - A thin agent builder that assembles a routed step and wraps it with the loop layer
//!
//! Exports
//! - `ToolDef`: function spec + `Service<ToolInvocation, ToolOutput>` glue for a tool
//! - `ToolRouter`: name→index dispatch over boxed tool services; returns `ToolOutput`
//! - `Step<S>`: `Service<RawChatRequest, StepOutcome>` parametric over a tool service `S`
//! - `Agent`: facade with `AgentBuilder` and a boxed `AgentSvc` type alias
//!
//! Implementation strategy (high-level)
//! - Tools: standardize I/O to `{ToolInvocation -> ToolOutput}`; adapt user-defined handlers into Tower services
//! - Router: either `tower::steer` or a simple name→index map of `BoxService`s; readiness via `ServiceExt::ready`
//! - Step: holds provider client, model config, tool specs, and a tool service; on call, executes model and optional tool calls
//! - AgentBuilder: composes router → step → loop using Tower layering; keeps DI static and explicit
//!
//! Composition examples
//! - `let (router, specs) = ToolRouter::new(vec![tool_a, tool_b]); let step = StepLayer::new(client, model, specs).layer(router);`
//! - `let agent = Agent::builder(client).model("gpt-4o").tool(my_tool).policy(policy).build();`
//!
//! Testing strategy
//! - `ToolDef::from_handler` test (provided): validates JSON args flow and output
//! - Router test (provided): unknown tool → explicit error
//! - Step tests (to add): fake provider returning (content|tool_calls); assert StepOutcome and tool invocation path
//! - Agent builder test: composes the above with a very small policy (max_steps=1)
//!
//! Example ideas
//! - Simple calculator tool; route and call directly via `ToolRouter`
//! - Same tool through `Step` with a model that always asks for that tool
//! - Full agent built with builder sugar stopping on `until_no_tool_calls`

pub use crate::next::{ToolDef, ToolInvocation, ToolOutput, ToolRouter};

#[cfg(test)]
mod tests {
    use super::*;
    use futures::FutureExt;
    use serde_json::json;
    use tower::{Service, ServiceExt};

    #[tokio::test]
    async fn tooldef_from_handler_invokes_handler() {
        let t = ToolDef::from_handler(
            "double",
            "double a number",
            json!({
                "type": "object",
                "properties": { "x": {"type": "number"} },
                "required": ["x"],
            }),
            std::sync::Arc::new(|args: serde_json::Value| {
                async move {
                    let x = args.get("x").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    Ok::<_, tower::BoxError>(json!({"y": x * 2.0}))
                }
                .boxed()
            }),
        );

        let mut router = ToolRouter::new(vec![t]).0;
        let out = router
            .ready()
            .await
            .unwrap()
            .call(ToolInvocation {
                id: "1".into(),
                name: "double".into(),
                arguments: json!({"x": 2.5}),
            })
            .await
            .unwrap();
        assert_eq!(out.result, json!({"y": 5.0}));
    }

    #[tokio::test]
    async fn router_unknown_tool_returns_error() {
        let (mut router, _specs) = ToolRouter::new(vec![]);
        let err = router
            .ready()
            .await
            .unwrap()
            .call(ToolInvocation {
                id: "1".into(),
                name: "missing".into(),
                arguments: json!({}),
            })
            .await
            .unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("unknown tool"));
    }
}
