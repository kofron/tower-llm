//! Composite handoff example showing complex multi-policy coordination with real LLM agents.
//!
//! This example demonstrates:
//! - Combining multiple handoff policies (explicit + sequential)
//! - Advanced picker logic based on request analysis
//! - Real-world scenario: Customer support with escalation workflows

use async_openai::{
    config::OpenAIConfig,
    types::{
        ChatCompletionRequestMessage, ChatCompletionRequestUserMessageArgs,
        ChatCompletionRequestSystemMessageArgs,
        CreateChatCompletionRequestArgs,
    },
    Client,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use tower::{Service, ServiceExt};
use tower_llm::{
    groups::{
        GroupBuilder, MultiExplicitHandoffPolicy, SequentialHandoffPolicy, 
        CompositeHandoffPolicy, AnyHandoffPolicy, PickRequest
    },
    policies, Agent, AgentSvc, CompositePolicy,
};

// Customer lookup tool
#[derive(Debug, Deserialize, JsonSchema)]
struct CustomerLookupArgs {
    /// Customer ID or email
    identifier: String,
}

// Refund processing tool
#[derive(Debug, Deserialize, JsonSchema)]
struct RefundArgs {
    /// Customer ID
    customer_id: String,
    /// Amount to refund
    amount: f64,
    /// Reason for refund
    reason: String,
}

// Technical diagnostic tool
#[derive(Debug, Deserialize, JsonSchema)]
struct DiagnosticArgs {
    /// Type of diagnostic (api, connection, performance)
    diagnostic_type: String,
    /// Additional parameters
    parameters: Option<String>,
}

/// Smart picker based on request content and urgency
#[derive(Clone)]
struct SupportPicker;

impl Service<PickRequest> for SupportPicker {
    type Response = String;
    type Error = tower::BoxError;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(
        &mut self,
        _: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        std::task::Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: PickRequest) -> Self::Future {
        Box::pin(async move {
            let content = req
                .messages
                .iter()
                .find_map(|msg| {
                    if let ChatCompletionRequestMessage::User(user_msg) = msg {
                        match &user_msg.content {
                            async_openai::types::ChatCompletionRequestUserMessageContent::Text(
                                text,
                            ) => Some(text.to_lowercase()),
                            _ => None,
                        }
                    } else {
                        None
                    }
                })
                .unwrap_or_default();

            // High-priority escalation keywords go straight to manager
            if content.contains("lawsuit") || content.contains("lawyer") || 
               content.contains("unacceptable") || content.contains("terrible service") {
                println!("🚨 Picker: Critical issue detected → routing to manager_agent");
                return Ok("manager_agent".to_string());
            }

            // Billing issues go to billing specialist
            if content.contains("billing") || content.contains("refund") || 
               content.contains("charge") || content.contains("payment") || content.contains("invoice") {
                println!("💳 Picker: Billing issue → routing to billing_agent");
                return Ok("billing_agent".to_string());
            }

            // Technical issues go to technical specialist
            if content.contains("api") || content.contains("technical") || 
               content.contains("error") || content.contains("integration") || 
               content.contains("bug") || content.contains("not working") {
                println!("🔧 Picker: Technical issue → routing to technical_agent");
                return Ok("technical_agent".to_string());
            }

            // Everything else starts with Tier 1
            println!("📞 Picker: General inquiry → routing to tier1_agent");
            Ok("tier1_agent".to_string())
        })
    }
}

fn create_tier1_agent(client: Arc<Client<OpenAIConfig>>) -> AgentSvc {
    // Customer lookup tool for tier 1
    let customer_lookup = tower_llm::tool_typed(
        "customer_lookup",
        "Look up customer information by ID or email",
        |args: CustomerLookupArgs| async move {
            println!("  🔍 Looking up customer: {}", args.identifier);
            Ok::<_, tower::BoxError>(json!({
                "customer_id": "CUST-12345",
                "name": "John Doe",
                "email": args.identifier,
                "account_status": "active",
                "subscription": "premium",
                "support_tier": "standard"
            }))
        },
    );

    Agent::builder(client)
        .model("gpt-4o-mini")
        .temperature(0.5)
        .tool(customer_lookup)
        .policy(CompositePolicy::new(vec![
            policies::until_no_tool_calls(),
            policies::max_steps(3),
        ]))
        .build()
}

fn create_billing_agent(client: Arc<Client<OpenAIConfig>>) -> AgentSvc {
    // Refund processing tool
    let refund_tool = tower_llm::tool_typed(
        "process_refund",
        "Process a refund for a customer",
        |args: RefundArgs| async move {
            println!("  💰 Processing refund: ${} for customer {} (reason: {})", 
                args.amount, args.customer_id, args.reason);
            Ok::<_, tower::BoxError>(json!({
                "refund_id": "REF-789012",
                "customer_id": args.customer_id,
                "amount": args.amount,
                "status": "processed",
                "estimated_time": "3-5 business days"
            }))
        },
    );

    Agent::builder(client)
        .model("gpt-4o-mini")
        .temperature(0.3) // Lower temperature for financial accuracy
        .tool(refund_tool)
        .policy(CompositePolicy::new(vec![
            policies::until_no_tool_calls(),
            policies::max_steps(3),
        ]))
        .build()
}

fn create_technical_agent(client: Arc<Client<OpenAIConfig>>) -> AgentSvc {
    // Diagnostic tool
    let diagnostic_tool = tower_llm::tool_typed(
        "run_diagnostic",
        "Run technical diagnostics",
        |args: DiagnosticArgs| async move {
            println!("  🔧 Running diagnostic: {} ({})", 
                args.diagnostic_type, 
                args.parameters.as_deref().unwrap_or("default"));
            
            let result = match args.diagnostic_type.as_str() {
                "api" => json!({
                    "status": "operational",
                    "latency_ms": 45,
                    "error_rate": 0.001,
                    "recommendation": "API is functioning normally"
                }),
                "connection" => json!({
                    "status": "stable",
                    "packet_loss": 0.0,
                    "bandwidth": "100 Mbps",
                    "recommendation": "Connection is stable"
                }),
                "performance" => json!({
                    "cpu_usage": "15%",
                    "memory_usage": "45%",
                    "disk_io": "normal",
                    "recommendation": "System performance is optimal"
                }),
                _ => json!({"error": "Unknown diagnostic type"})
            };
            
            Ok::<_, tower::BoxError>(result)
        },
    );

    Agent::builder(client)
        .model("gpt-4o-mini")
        .temperature(0.2) // Low temperature for technical precision
        .tool(diagnostic_tool)
        .policy(CompositePolicy::new(vec![
            policies::until_no_tool_calls(),
            policies::max_steps(3),
        ]))
        .build()
}

fn create_manager_agent(client: Arc<Client<OpenAIConfig>>) -> AgentSvc {
    // Manager has authority but typically doesn't need tools
    Agent::builder(client)
        .model("gpt-4o-mini")
        .temperature(0.6) // Balanced for empathy and problem-solving
        .policy(CompositePolicy::new(vec![
            policies::until_no_tool_calls(),
            policies::max_steps(2),
        ]))
        .build()
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Initialize tracing with human-readable output
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .with_thread_ids(false)
        .with_thread_names(false)
        .with_file(false)
        .with_line_number(false)
        .with_level(true)
        .with_ansi(true)
        .compact()
        .init();

    println!("=== Composite Handoff Example with Real LLM Agents ===\n");
    
    println!("This example demonstrates advanced multi-policy coordination:");
    println!("• Smart picker routes based on content analysis");
    println!("• Explicit policy handles specialist escalations");
    println!("• Sequential policy manages escalation workflow");
    println!("• Composite policy combines both seamlessly\n");

    // Create OpenAI client
    let client = Arc::new(Client::<OpenAIConfig>::new());

    // Create specialized agents
    let tier1_agent = create_tier1_agent(client.clone());
    let billing_agent = create_billing_agent(client.clone());
    let technical_agent = create_technical_agent(client.clone());
    let manager_agent = create_manager_agent(client.clone());

    // Create explicit handoff policy for specialist escalations
    let mut explicit_handoffs = HashMap::new();
    explicit_handoffs.insert("escalate_to_billing".to_string(), "billing_agent".to_string());
    explicit_handoffs.insert("escalate_to_technical".to_string(), "technical_agent".to_string());
    explicit_handoffs.insert("escalate_to_manager".to_string(), "manager_agent".to_string());
    let explicit_policy = MultiExplicitHandoffPolicy::new(explicit_handoffs);

    // Create sequential policy for escalation workflow
    // This creates an automatic escalation path when agents can't resolve issues
    let escalation_sequence = vec![
        "tier1_agent".to_string(),
        "manager_agent".to_string(),
    ];
    let sequential_policy = SequentialHandoffPolicy::new(escalation_sequence);

    // Combine policies with CompositeHandoffPolicy
    let composite_policy = CompositeHandoffPolicy::new(vec![
        AnyHandoffPolicy::from(explicit_policy),
        AnyHandoffPolicy::from(sequential_policy),
    ]);

    // Create smart picker
    let picker = SupportPicker;

    // Build coordinator
    let mut coordinator = GroupBuilder::new()
        .agent("tier1_agent", tier1_agent)
        .agent("billing_agent", billing_agent)
        .agent("technical_agent", technical_agent)
        .agent("manager_agent", manager_agent)
        .picker(picker)
        .handoff_policy(composite_policy)
        .build();

    // Test scenarios
    println!("--- Scenario 1: Basic Support Question ---");
    println!("User: 'How do I reset my password?'\n");
    
    let system_msg = ChatCompletionRequestSystemMessageArgs::default()
        .content("You are a customer support agent. Help customers with their inquiries. For specialized issues, you can escalate to billing (escalate_to_billing), technical (escalate_to_technical), or manager (escalate_to_manager) using the appropriate tools.")
        .build()?;
    
    let user_message1 = ChatCompletionRequestUserMessageArgs::default()
        .content("How do I reset my password?")
        .build()?;
    
    let req1 = CreateChatCompletionRequestArgs::default()
        .model("gpt-4o-mini")
        .messages(vec![
            ChatCompletionRequestMessage::System(system_msg.clone()),
            ChatCompletionRequestMessage::User(user_message1)
        ])
        .build()?;

    match coordinator.ready().await?.call(req1).await {
        Ok(result) => {
            println!("✅ Handled by Tier 1 support");
            println!("   Messages: {}, Steps: {}\n", result.messages.len(), result.steps);
        }
        Err(e) => println!("❌ Error: {}\n", e),
    }

    println!("--- Scenario 2: Billing Issue (Direct Route) ---");
    println!("User: 'I need a refund for the duplicate charge on my account'\n");
    
    let user_message2 = ChatCompletionRequestUserMessageArgs::default()
        .content("I need a refund for the duplicate charge of $49.99 on my account")
        .build()?;
    
    let req2 = CreateChatCompletionRequestArgs::default()
        .model("gpt-4o-mini")
        .messages(vec![
            ChatCompletionRequestMessage::System(system_msg.clone()),
            ChatCompletionRequestMessage::User(user_message2)
        ])
        .build()?;

    match coordinator.ready().await?.call(req2).await {
        Ok(result) => {
            println!("✅ Routed directly to billing specialist");
            println!("   Messages: {}, Steps: {}", result.messages.len(), result.steps);
            
            // Show if refund was processed
            for msg in &result.messages {
                if let ChatCompletionRequestMessage::Assistant(assistant_msg) = msg {
                    if let Some(content) = &assistant_msg.content {
                        let text = match content {
                            async_openai::types::ChatCompletionRequestAssistantMessageContent::Text(t) => t.as_str(),
                            _ => "",
                        };
                        if text.contains("refund") || text.contains("REF-") {
                            println!("   💰 Refund processed successfully\n");
                            break;
                        }
                    }
                }
            }
        }
        Err(e) => println!("❌ Error: {}\n", e),
    }

    println!("--- Scenario 3: Technical Issue ---");
    println!("User: 'The API is returning 500 errors intermittently'\n");
    
    let user_message3 = ChatCompletionRequestUserMessageArgs::default()
        .content("The API is returning 500 errors intermittently. Can you help diagnose the issue?")
        .build()?;
    
    let req3 = CreateChatCompletionRequestArgs::default()
        .model("gpt-4o-mini")
        .messages(vec![
            ChatCompletionRequestMessage::System(system_msg.clone()),
            ChatCompletionRequestMessage::User(user_message3)
        ])
        .build()?;

    match coordinator.ready().await?.call(req3).await {
        Ok(result) => {
            println!("✅ Routed to technical specialist");
            println!("   Messages: {}, Steps: {}\n", result.messages.len(), result.steps);
        }
        Err(e) => println!("❌ Error: {}\n", e),
    }

    println!("--- Scenario 4: Escalation to Manager ---");
    println!("User: 'This is unacceptable! I've been charged three times!'\n");
    
    let user_message4 = ChatCompletionRequestUserMessageArgs::default()
        .content("This is unacceptable! I've been charged three times for the same service and no one is helping me!")
        .build()?;
    
    let req4 = CreateChatCompletionRequestArgs::default()
        .model("gpt-4o-mini")
        .messages(vec![
            ChatCompletionRequestMessage::System(system_msg),
            ChatCompletionRequestMessage::User(user_message4)
        ])
        .build()?;

    match coordinator.ready().await?.call(req4).await {
        Ok(result) => {
            println!("✅ Escalated directly to manager");
            println!("   Messages: {}, Steps: {}", result.messages.len(), result.steps);
            
            if let Some(ChatCompletionRequestMessage::Assistant(msg)) = result.messages.last() {
                if let Some(content) = &msg.content {
                    let text = match content {
                        async_openai::types::ChatCompletionRequestAssistantMessageContent::Text(t) => t,
                        _ => "(non-text content)",
                    };
                    let preview = if text.len() > 200 {
                        format!("{}...", &text[..200])
                    } else {
                        text.to_string()
                    };
                    println!("   Manager response: {}\n", preview);
                }
            }
        }
        Err(e) => println!("❌ Error: {}\n", e),
    }

    println!("=== Composite Policy Architecture ===");
    println!("1. **Smart Picker Logic**:");
    println!("   - Critical issues → manager_agent (immediate escalation)");
    println!("   - Billing keywords → billing_agent");
    println!("   - Technical keywords → technical_agent");
    println!("   - Everything else → tier1_agent");
    println!();
    println!("2. **Explicit Handoff Policy** (handles specialist escalations):");
    println!("   - escalate_to_billing → billing_agent");
    println!("   - escalate_to_technical → technical_agent");
    println!("   - escalate_to_manager → manager_agent");
    println!();
    println!("3. **Sequential Handoff Policy** (handles escalation workflow):");
    println!("   - tier1_agent → manager_agent (for unresolved issues)");
    println!();
    println!("4. **Composite Benefits**:");
    println!("   - Multiple routing strategies in one system");
    println!("   - Flexible escalation paths");
    println!("   - Specialist expertise routing");
    println!("   - Consistent workflow enforcement");
    println!("   - Real LLM agents with specialized tools\n");

    Ok(())
}