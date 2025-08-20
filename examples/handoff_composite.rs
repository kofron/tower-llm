//! Composite handoff example showing complex multi-policy coordination.
//!
//! This example demonstrates:
//! - Combining multiple handoff policies (explicit + sequential)
//! - Advanced picker logic based on request analysis
//! - Real-world scenario: Customer support with escalation workflows

use std::collections::HashMap;
use std::sync::Arc;
use async_openai::types::{ChatCompletionRequestMessage, CreateChatCompletionRequest};
use serde_json::json;
use tower::{Service, ServiceExt};
use tower_llm::groups::{
    AgentName, AgentPicker, HandoffPolicy, HandoffRequest, CompositeHandoffPolicy,
    ExplicitHandoffPolicy, SequentialHandoffPolicy, HandoffCoordinator
};

// Tier 1 support - handles basic questions
#[derive(Clone)]
struct Tier1Agent;

impl Service<CreateChatCompletionRequest> for Tier1Agent {
    type Response = tower_llm::StepOutcome;
    type Error = tower::BoxError;
    type Future = std::pin::Pin<Box<dyn std::future::Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, _: &mut std::task::Context<'_>) -> std::task::Poll<Result<(), Self::Error>> {
        std::task::Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: CreateChatCompletionRequest) -> Self::Future {
        Box::pin(async move {
            let last_message = req.messages.last()
                .and_then(|m| m.content.as_ref())
                .unwrap_or("");

            // Check if this needs escalation
            if last_message.contains("refund") || last_message.contains("cancel") || 
               last_message.contains("billing") || last_message.contains("angry") {
                return Ok(tower_llm::StepOutcome::Next {
                    messages: vec![ChatCompletionRequestMessage {
                        role: async_openai::types::Role::Assistant,
                        content: Some("I understand this is a billing concern. Let me escalate this to our billing specialist who can better assist you.".to_string()),
                        name: None,
                        tool_calls: Some(vec![async_openai::types::ChatCompletionMessageToolCall {
                            id: "escalate_billing".to_string(),
                            r#type: async_openai::types::ChatCompletionMessageToolCallType::Function,
                            function: async_openai::types::FunctionCall {
                                name: "escalate_to_billing".to_string(),
                                arguments: json!({"reason": "Billing inquiry requiring specialist"}).to_string(),
                            },
                        }]),
                        tool_call_id: None,
                    }],
                    aux: tower_llm::StepAux {
                        prompt_tokens: 40,
                        completion_tokens: 25,
                        tool_invocations: 1,
                    },
                    invoked_tools: vec![],
                });
            }

            if last_message.contains("technical") || last_message.contains("api") || 
               last_message.contains("integration") || last_message.contains("error") {
                return Ok(tower_llm::StepOutcome::Next {
                    messages: vec![ChatCompletionRequestMessage {
                        role: async_openai::types::Role::Assistant,
                        content: Some("This appears to be a technical issue. Let me connect you with our technical support team.".to_string()),
                        name: None,
                        tool_calls: Some(vec![async_openai::types::ChatCompletionMessageToolCall {
                            id: "escalate_technical".to_string(),
                            r#type: async_openai::types::ChatCompletionMessageToolCallType::Function,
                            function: async_openai::types::FunctionCall {
                                name: "escalate_to_technical".to_string(),
                                arguments: json!({"reason": "Technical issue requiring specialist"}).to_string(),
                            },
                        }]),
                        tool_call_id: None,
                    }],
                    aux: tower_llm::StepAux {
                        prompt_tokens: 40,
                        completion_tokens: 20,
                        tool_invocations: 1,
                    },
                    invoked_tools: vec![],
                });
            }

            // Handle basic questions
            Ok(tower_llm::StepOutcome::Done {
                messages: vec![ChatCompletionRequestMessage {
                    role: async_openai::types::Role::Assistant,
                    content: Some("I can help with that! For basic account questions, you can find information in your dashboard under Settings > Account. Is there anything specific you'd like to know?".to_string()),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                }],
                aux: tower_llm::StepAux {
                    prompt_tokens: 30,
                    completion_tokens: 25,
                    tool_invocations: 0,
                },
            })
        })
    }
}

// Billing specialist
#[derive(Clone)]
struct BillingAgent;

impl Service<CreateChatCompletionRequest> for BillingAgent {
    type Response = tower_llm::StepOutcome;
    type Error = tower::BoxError;
    type Future = std::pin::Pin<Box<dyn std::future::Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, _: &mut std::task::Context<'_>) -> std::task::Poll<Result<(), Self::Error>> {
        std::task::Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: CreateChatCompletionRequest) -> Self::Future {
        Box::pin(async move {
            let last_message = req.messages.last()
                .and_then(|m| m.content.as_ref())
                .unwrap_or("");

            // Check if this needs manager escalation
            if last_message.contains("lawsuit") || last_message.contains("lawyer") || 
               last_message.contains("unacceptable") || last_message.contains("furious") {
                return Ok(tower_llm::StepOutcome::Next {
                    messages: vec![ChatCompletionRequestMessage {
                        role: async_openai::types::Role::Assistant,
                        content: Some("I understand your frustration. Given the severity of this issue, I'm going to escalate this to our support manager immediately.".to_string()),
                        name: None,
                        tool_calls: Some(vec![async_openai::types::ChatCompletionMessageToolCall {
                            id: "escalate_manager".to_string(),
                            r#type: async_openai::types::ChatCompletionMessageToolCallType::Function,
                            function: async_openai::types::FunctionCall {
                                name: "escalation_workflow".to_string(),
                                arguments: json!({"progress": "manager_escalation"}).to_string(),
                            },
                        }]),
                        tool_call_id: None,
                    }],
                    aux: tower_llm::StepAux {
                        prompt_tokens: 50,
                        completion_tokens: 30,
                        tool_invocations: 1,
                    },
                    invoked_tools: vec![],
                });
            }

            // Handle billing issues
            Ok(tower_llm::StepOutcome::Done {
                messages: vec![ChatCompletionRequestMessage {
                    role: async_openai::types::Role::Assistant,
                    content: Some("I've reviewed your billing history. I can process a refund for the disputed charge of $49.99. The refund will appear on your statement within 3-5 business days. I've also added a credit to your account for the inconvenience.".to_string()),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                }],
                aux: tower_llm::StepAux {
                    prompt_tokens: 60,
                    completion_tokens: 40,
                    tool_invocations: 0,
                },
            })
        })
    }
}

// Technical support specialist
#[derive(Clone)]
struct TechnicalAgent;

impl Service<CreateChatCompletionRequest> for TechnicalAgent {
    type Response = tower_llm::StepOutcome;
    type Error = tower::BoxError;
    type Future = std::pin::Pin<Box<dyn std::future::Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, _: &mut std::task::Context<'_>) -> std::task::Poll<Result<(), Self::Error>> {
        std::task::Poll::Ready(Ok(()))
    }

    fn call(&mut self, _req: CreateChatCompletionRequest) -> Self::Future {
        Box::pin(async move {
            Ok(tower_llm::StepOutcome::Done {
                messages: vec![ChatCompletionRequestMessage {
                    role: async_openai::types::Role::Assistant,
                    content: Some("I've identified the API integration issue. The problem is with your authentication headers - you're using an old API key format. Here's the solution:\n\n1. Generate a new API key from your dashboard\n2. Update your code to use the new format: `Bearer sk-proj-...`\n3. Clear your application cache\n\nI've also sent you a code example via email. Let me know if you need further assistance!".to_string()),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                }],
                aux: tower_llm::StepAux {
                    prompt_tokens: 80,
                    completion_tokens: 60,
                    tool_invocations: 0,
                },
            })
        })
    }
}

// Support manager - handles escalated cases
#[derive(Clone)]
struct ManagerAgent;

impl Service<CreateChatCompletionRequest> for ManagerAgent {
    type Response = tower_llm::StepOutcome;
    type Error = tower::BoxError;
    type Future = std::pin::Pin<Box<dyn std::future::Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, _: &mut std::task::Context<'_>) -> std::task::Poll<Result<(), Self::Error>> {
        std::task::Poll::Ready(Ok(()))
    }

    fn call(&mut self, _req: CreateChatCompletionRequest) -> Self::Future {
        Box::pin(async move {
            Ok(tower_llm::StepOutcome::Done {
                messages: vec![ChatCompletionRequestMessage {
                    role: async_openai::types::Role::Assistant,
                    content: Some("I'm the support manager and I sincerely apologize for this experience. I've personally reviewed your case and here's what I'm doing to make this right:\n\n1. Full refund of $149.97 processed immediately\n2. 3 months of service credit added to your account\n3. Priority support status for future issues\n4. Direct line to my team: support-priority@company.com\n\nI take these situations very seriously, and we're implementing process improvements to prevent this from happening again. You should see the refund within 24 hours.".to_string()),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                }],
                aux: tower_llm::StepAux {
                    prompt_tokens: 100,
                    completion_tokens: 90,
                    tool_invocations: 0,
                },
            })
        })
    }
}

// Smart picker based on request content and urgency
#[derive(Clone)]
struct SupportPicker;

impl AgentPicker<CreateChatCompletionRequest> for SupportPicker {
    fn pick(&self, req: &CreateChatCompletionRequest) -> Result<AgentName, tower::BoxError> {
        let content = req.messages.last()
            .and_then(|m| m.content.as_ref())
            .unwrap_or("")
            .to_lowercase();

        // High-priority escalation keywords go straight to manager
        if content.contains("lawsuit") || content.contains("lawyer") || 
           content.contains("ceo") || content.contains("unacceptable") {
            return Ok(AgentName("manager_agent".to_string()));
        }

        // Billing issues go to billing specialist
        if content.contains("billing") || content.contains("refund") || 
           content.contains("charge") || content.contains("payment") {
            return Ok(AgentName("billing_agent".to_string()));
        }

        // Technical issues go to technical specialist
        if content.contains("api") || content.contains("technical") || 
           content.contains("error") || content.contains("integration") {
            return Ok(AgentName("technical_agent".to_string()));
        }

        // Everything else starts with Tier 1
        Ok(AgentName("tier1_agent".to_string()))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("=== Composite Handoff Example ===\n");
    
    println!("This example demonstrates advanced multi-policy coordination:");
    println!("• Smart picker routes based on content analysis");
    println!("• Explicit policy handles specialist escalations");
    println!("• Sequential policy manages escalation workflow");
    println!("• Composite policy combines both seamlessly\n");

    // Set up agents
    let mut agents = HashMap::new();
    agents.insert(
        AgentName("tier1_agent".to_string()),
        tower::util::BoxService::new(Tier1Agent)
    );
    agents.insert(
        AgentName("billing_agent".to_string()),
        tower::util::BoxService::new(BillingAgent)
    );
    agents.insert(
        AgentName("technical_agent".to_string()),
        tower::util::BoxService::new(TechnicalAgent)
    );
    agents.insert(
        AgentName("manager_agent".to_string()),
        tower::util::BoxService::new(ManagerAgent)
    );

    // Create explicit handoff policy for specialist escalations
    let mut explicit_handoffs = HashMap::new();
    explicit_handoffs.insert("escalate_to_billing".to_string(), AgentName("billing_agent".to_string()));
    explicit_handoffs.insert("escalate_to_technical".to_string(), AgentName("technical_agent".to_string()));
    let explicit_policy = ExplicitHandoffPolicy::new(explicit_handoffs);

    // Create sequential policy for escalation workflow
    let escalation_sequence = vec![
        AgentName("billing_agent".to_string()),
        AgentName("manager_agent".to_string()),
    ];
    let sequential_policy = SequentialHandoffPolicy::new(escalation_sequence);

    // Combine policies with CompositeHandoffPolicy
    let composite_policy = CompositeHandoffPolicy::new(vec![
        Box::new(explicit_policy),
        Box::new(sequential_policy),
    ]);

    // Create smart picker
    let picker = SupportPicker;

    // Build coordinator
    let coordinator = HandoffCoordinator::new(
        Arc::new(tokio::sync::Mutex::new(agents)),
        picker,
        composite_policy,
    );

    // Test scenarios
    println!("--- Scenario 1: Basic Question ---");
    println!("Input: 'How do I change my password?'");
    
    let req1 = async_openai::types::CreateChatCompletionRequestArgs::default()
        .model("gpt-4o")
        .messages(vec![ChatCompletionRequestMessage {
            role: async_openai::types::Role::User,
            content: Some("How do I change my password?".to_string()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }])
        .build()?;

    let mut coord1 = coordinator.clone();
    match coord1.ready().await?.call(req1).await {
        Ok(_) => println!("✅ Handled by Tier 1 support\n"),
        Err(e) => println!("❌ Error: {}\n", e),
    }

    println!("--- Scenario 2: Billing Issue with Escalation ---");
    println!("Input: 'I want a refund and this is unacceptable service!'");
    
    let req2 = async_openai::types::CreateChatCompletionRequestArgs::default()
        .model("gpt-4o")
        .messages(vec![ChatCompletionRequestMessage {
            role: async_openai::types::Role::User,
            content: Some("I want a refund and this is unacceptable service!".to_string()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }])
        .build()?;

    let mut coord2 = coordinator.clone();
    match coord2.ready().await?.call(req2).await {
        Ok(_) => println!("✅ Routed to billing, escalated to manager via sequential policy\n"),
        Err(e) => println!("❌ Error: {}\n", e),
    }

    println!("--- Scenario 3: Technical Issue ---");
    println!("Input: 'Getting 401 errors with the API integration'");
    
    let req3 = async_openai::types::CreateChatCompletionRequestArgs::default()
        .model("gpt-4o")
        .messages(vec![ChatCompletionRequestMessage {
            role: async_openai::types::Role::User,
            content: Some("Getting 401 errors with the API integration".to_string()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }])
        .build()?;

    let mut coord3 = coordinator;
    match coord3.ready().await?.call(req3).await {
        Ok(_) => println!("✅ Routed directly to technical specialist\n"),
        Err(e) => println!("❌ Error: {}\n", e),
    }

    println!("=== Composite Policy Architecture ===");
    println!("1. **Smart Picker Logic**:");
    println!("   - Lawsuit/Legal → manager_agent (immediate escalation)");
    println!("   - Billing keywords → billing_agent");
    println!("   - Technical keywords → technical_agent");
    println!("   - Everything else → tier1_agent");
    println!();
    println!("2. **Explicit Handoff Policy** (handles specialist escalations):");
    println!("   - escalate_to_billing → billing_agent");
    println!("   - escalate_to_technical → technical_agent");
    println!();
    println!("3. **Sequential Handoff Policy** (handles escalation workflow):");
    println!("   - billing_agent → manager_agent (for serious issues)");
    println!();
    println!("4. **Composite Benefits**:");
    println!("   - Multiple routing strategies in one system");
    println!("   - Flexible escalation paths");
    println!("   - Specialist expertise routing");
    println!("   - Consistent workflow enforcement");

    Ok(())
}