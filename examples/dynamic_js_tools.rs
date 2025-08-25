use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_openai::types::{
    ChatCompletionRequestMessage, ChatCompletionTool, ChatCompletionToolArgs,
    ChatCompletionToolType, CreateChatCompletionRequest, CreateChatCompletionRequestArgs,
    FunctionObjectArgs,
};
use boa_engine::{Context, Source};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tower::{BoxError, Layer, Service, ServiceExt};

use tower_llm::provider::OpenAIProvider;
use tower_llm::{
    policies, tool_typed, Client, OpenAIConfig, ReasoningEffort, StepLayer, ToolDef,
    ToolInvocation, ToolOutput, ToolRouter,
};

// =============================
// Dynamic registry (example-local)
// =============================

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DynamicTool {
    description: String,
    parameters_schema: Value,
    code: String, // JS function body: receives `args` and must `return <jsonable>`
}

type DynRegistry = Arc<Mutex<HashMap<String, DynamicTool>>>;

// =============================
// Meta-tool: define_js_tool
// =============================

#[derive(Debug, Deserialize, JsonSchema)]
struct DefineJsArgs {
    name: String,
    description: String,
    parameters_schema: Value,
    code: String,
}

fn define_js_tool(registry: DynRegistry) -> ToolDef {
    tool_typed(
        "define_js_tool",
        "Define a new dynamic JavaScript tool with a name, schema, and code",
        move |args: DefineJsArgs| {
            let reg = registry.clone();
            async move {
                let mut map = reg.lock().unwrap();
                if map.contains_key(&args.name) {
                    return Err::<serde_json::Value, BoxError>(
                        format!("tool '{}' already exists", args.name).into(),
                    );
                }
                // Compile-check the provided code with empty args and surface errors to the model
                let args_json = "{}".to_string();
                let wrapped = format!(
                    r#"(function() {{
    try {{
        const args = {args};
        const handle = (args) => {{ {body} }};
        const out = handle(args);
        return JSON.stringify({{ ok: true, result: (out === undefined) ? null : out }});
    }} catch (e) {{
        return JSON.stringify({{ ok: false, error: String(e) }});
    }}
}})();"#,
                    args = args_json,
                    body = args.code,
                );
                let mut ctx = Context::default();
                match ctx.eval(Source::from_bytes(&wrapped)) {
                    Ok(val) => {
                        let s = val
                            .to_string(&mut ctx)
                            .map_err(|e| format!("to_string error: {e}"))?
                            .to_std_string_escaped();
                        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&s) {
                            if v.get("ok").and_then(|b| b.as_bool()) != Some(true) {
                                let msg = v
                                    .get("error")
                                    .and_then(|e| e.as_str())
                                    .unwrap_or("compile error");
                                return Err::<serde_json::Value, BoxError>(
                                    format!("dynamic tool compile failed: {}", msg).into(),
                                );
                            }
                        }
                    }
                    Err(e) => {
                        return Err::<serde_json::Value, BoxError>(
                            format!("dynamic tool compile failed: {}", e).into(),
                        );
                    }
                }
                map.insert(
                    args.name,
                    DynamicTool {
                        description: args.description,
                        parameters_schema: args.parameters_schema,
                        code: args.code,
                    },
                );
                Ok(serde_json::json!({ "ok": true }))
            }
        },
    )
}

// =============================
// Dynamic Tool Router wrapper
// =============================

#[derive(Clone)]
struct DynamicToolService<S> {
    inner: S,
    registry: DynRegistry,
}

impl<S> DynamicToolService<S> {
    fn new(inner: S, registry: DynRegistry) -> Self {
        Self { inner, registry }
    }
}

impl<S> Service<ToolInvocation> for DynamicToolService<S>
where
    S: Service<ToolInvocation, Response = ToolOutput, Error = BoxError> + Clone + Send + 'static,
    S::Future: Send + 'static,
{
    type Response = ToolOutput;
    type Error = BoxError;
    type Future = std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Self::Response, Self::Error>> + Send>,
    >;

    fn poll_ready(
        &mut self,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: ToolInvocation) -> Self::Future {
        let name = req.name.clone();
        let registry = self.registry.clone();
        let mut inner = self.inner.clone();
        Box::pin(async move {
            let dt_opt: Option<DynamicTool> = {
                let map = registry.lock().unwrap();
                map.get(&name).cloned()
            };
            if let Some(dt) = dt_opt {
                // Execute JS with Boa: JSON.stringify a result envelope so it's always JSON
                let args_json = serde_json::to_string(&req.arguments)?;
                let wrapped = format!(
                    r#"(function() {{
    try {{
        const args = {args};
        const handle = (args) => {{ {body} }};
        const out = handle(args);
        return JSON.stringify({{ ok: true, result: (out === undefined) ? null : out }});
    }} catch (e) {{
        return JSON.stringify({{ ok: false, error: String(e) }});
    }}
}})();"#,
                    args = args_json,
                    body = dt.code,
                );
                let mut ctx = Context::default();
                let val = match ctx.eval(Source::from_bytes(&wrapped)) {
                    Ok(v) => v,
                    Err(e) => {
                        // Surface syntax/compile errors as a JSON tool result
                        let out = serde_json::json!({
                            "ok": false,
                            "error": format!("compile: {}", e),
                        });
                        return Ok(ToolOutput {
                            id: req.id,
                            result: out,
                        });
                    }
                };
                let s = val
                    .to_string(&mut ctx)
                    .map_err(|e| format!("to_string error: {e}"))?
                    .to_std_string_escaped();
                let out: Value = match serde_json::from_str(&s) {
                    Ok(v) => v,
                    Err(_e) => serde_json::json!({
                        "ok": false,
                        "error": "non-json result",
                        "raw": s,
                    }),
                };
                Ok(ToolOutput {
                    id: req.id,
                    result: out,
                })
            } else {
                ServiceExt::ready(&mut inner).await?.call(req).await
            }
        })
    }
}

// =============================
// Advertise dynamic tools layer
// =============================

#[derive(Clone)]
struct AdvertiseDynamicToolsLayer {
    registry: DynRegistry,
    baseline: Arc<Vec<ChatCompletionTool>>, // baseline specs (e.g., meta-tools)
}

impl AdvertiseDynamicToolsLayer {
    fn new(registry: DynRegistry, baseline: Arc<Vec<ChatCompletionTool>>) -> Self {
        Self { registry, baseline }
    }
}

#[derive(Clone)]
struct AdvertiseDynamicTools<S> {
    inner: S,
    registry: DynRegistry,
    baseline: Arc<Vec<ChatCompletionTool>>,
}

impl<S> Layer<S> for AdvertiseDynamicToolsLayer {
    type Service = AdvertiseDynamicTools<S>;
    fn layer(&self, inner: S) -> Self::Service {
        AdvertiseDynamicTools {
            inner,
            registry: self.registry.clone(),
            baseline: self.baseline.clone(),
        }
    }
}

impl<S, R> Service<CreateChatCompletionRequest> for AdvertiseDynamicTools<S>
where
    S: Service<CreateChatCompletionRequest, Response = R, Error = BoxError> + Send + 'static,
    S::Future: Send + 'static,
    R: Send + 'static,
{
    type Response = S::Response;
    type Error = BoxError;
    type Future = std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Self::Response, Self::Error>> + Send>,
    >;

    fn poll_ready(
        &mut self,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: CreateChatCompletionRequest) -> Self::Future {
        let mut builder = CreateChatCompletionRequestArgs::default();
        builder.model(&req.model);
        builder.messages(req.messages.clone());
        if let Some(t) = req.temperature {
            builder.temperature(t);
        }
        if let Some(mt) = req.max_completion_tokens {
            builder.max_completion_tokens(mt);
        }
        #[allow(deprecated)]
        if let Some(mt) = req.max_tokens {
            builder.max_tokens(mt);
        }
        if let Some(eff) = req.reasoning_effort.clone() {
            builder.reasoning_effort(eff);
        }

        // Merge existing request tools, baseline specs, and dynamic specs
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut tools: Vec<ChatCompletionTool> = Vec::new();

        if let Some(existing) = req.tools.clone() {
            for t in existing {
                if seen.insert(t.function.name.clone()) {
                    tools.push(t);
                }
            }
        }

        for t in self.baseline.iter() {
            if seen.insert(t.function.name.clone()) {
                tools.push(t.clone());
            }
        }

        let map = self.registry.lock().unwrap();
        for (name, dt) in map.iter() {
            if !seen.insert(name.to_string()) {
                continue;
            }
            let func = FunctionObjectArgs::default()
                .name(name)
                .description(&dt.description)
                .parameters(dt.parameters_schema.clone())
                .build()
                .expect("valid function object");
            let tool = ChatCompletionToolArgs::default()
                .r#type(ChatCompletionToolType::Function)
                .function(func)
                .build()
                .expect("valid chat tool");
            tools.push(tool);
        }
        if !tools.is_empty() {
            builder.tools(tools);
        }
        let req2 = builder.build().expect("rebuilt request");
        let fut = self.inner.call(req2);
        Box::pin(fut)
    }
}

// (removed scripted provider; using real provider)

// =============================
// Example main
// =============================

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_target(false)
        .with_thread_ids(false)
        .with_thread_names(false)
        .with_file(false)
        .with_line_number(false)
        .with_level(true)
        .with_ansi(true)
        .compact()
        .init();

    // Registry and meta-tool
    let registry: DynRegistry = Arc::new(Mutex::new(HashMap::new()));
    let def_tool = define_js_tool(registry.clone());

    // Base tool router with only define_js_tool
    let (router, specs) = ToolRouter::new(vec![def_tool]);
    let dynamic_router = DynamicToolService::new(router, registry.clone());
    let tool_svc = tower::util::BoxCloneService::new(dynamic_router);

    // Step with real OpenAI provider
    let client = Arc::new(Client::<OpenAIConfig>::new());
    let provider = OpenAIProvider::new(client);
    let baseline_specs = Arc::new(specs.clone());
    let step = StepLayer::new(provider, "gpt-5", specs).layer(tool_svc);

    // Advertise dynamic and baseline tools on each request
    let step = AdvertiseDynamicToolsLayer::new(registry.clone(), baseline_specs).layer(step);

    // Agent loop: stop after either a tool is called or max steps
    let policy = tower_llm::CompositePolicy::new(vec![policies::max_steps(25)]);
    let agent = tower_llm::AgentLoopLayer::new(policy).layer(step);
    let mut agent = tower::util::BoxService::new(agent);

    // Seed user request
    let user = async_openai::types::ChatCompletionRequestUserMessageArgs::default()
        .content("Hello!  I've given you a tool to define new tools, and I want you to use that tool to build tools that you need in order to solve the problems that I give you.  Your tools can be written in plain Javascript but you can't use any libraries, any IO, or anything like that.  It's a very stripped down JS interpreter.  Let's try this for starters.  I want you to find the nearest perfect square to 35^9.")
        .build()
        .unwrap();
    let req = CreateChatCompletionRequestArgs::default()
        .model("gpt-5")
        .reasoning_effort(ReasoningEffort::High)
        .messages(vec![ChatCompletionRequestMessage::from(user)])
        .build()
        .unwrap();

    let out = ServiceExt::ready(&mut agent).await?.call(req).await?;
    println!("Steps: {} | Stop: {:?}", out.steps, out.stop);
    for m in out.messages {
        match m {
            ChatCompletionRequestMessage::System(s) => {
                let txt = match s.content {
                    async_openai::types::ChatCompletionRequestSystemMessageContent::Text(t) => t,
                    _ => String::from("[non-text system content]"),
                };
                println!("SYSTEM: {}", txt);
            }
            ChatCompletionRequestMessage::User(u) => {
                let txt = match u.content {
                    async_openai::types::ChatCompletionRequestUserMessageContent::Text(t) => t,
                    _ => String::from("[non-text user content]"),
                };
                println!("USER: {}", txt);
            }
            ChatCompletionRequestMessage::Assistant(a) => {
                let txt = if let Some(content) = a.content {
                    match content {
                        async_openai::types::ChatCompletionRequestAssistantMessageContent::Text(
                            t,
                        ) => t,
                        _ => String::from(""),
                    }
                } else {
                    String::new()
                };
                if !txt.is_empty() {
                    println!("ASSISTANT: {}", txt);
                }
                if let Some(calls) = a.tool_calls {
                    for c in calls {
                        println!(
                            "ASSISTANT requested tool: {} args={} (id={})",
                            c.function.name, c.function.arguments, c.id
                        );
                    }
                }
            }
            ChatCompletionRequestMessage::Tool(t) => {
                let txt = match t.content {
                    async_openai::types::ChatCompletionRequestToolMessageContent::Text(s) => s,
                    _ => String::from("[non-text tool content]"),
                };
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&txt) {
                    println!(
                        "TOOL[{}]: {}",
                        t.tool_call_id,
                        serde_json::to_string_pretty(&v).unwrap()
                    );
                } else {
                    println!("TOOL[{}]: {}", t.tool_call_id, txt);
                }
            }
            _ => {}
        }
    }

    Ok(())
}
