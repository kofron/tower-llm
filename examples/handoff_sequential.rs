//! Sequential handoff example showing workflow orchestration.
//!
//! This example demonstrates:
//! - Sequential workflow where agents pass work in a predefined order
//! - How picker vs policy work together for complex workflows
//! - Research → Analysis → Report generation pipeline

use std::collections::HashMap;
use std::sync::Arc;
use async_openai::types::{ChatCompletionRequestMessage, CreateChatCompletionRequest};
use serde_json::json;
use tower::{Service, ServiceExt};
use tower_llm::groups::{
    AgentName, AgentPicker, HandoffRequest, SequentialHandoffPolicy, 
    HandoffCoordinator
};

// Research agent - gathers information
#[derive(Clone)]
struct ResearchAgent {
    step: Arc<std::sync::Mutex<usize>>,
}

impl ResearchAgent {
    fn new() -> Self {
        Self {
            step: Arc::new(std::sync::Mutex::new(0)),
        }
    }
}

impl Service<CreateChatCompletionRequest> for ResearchAgent {
    type Response = tower_llm::StepOutcome;
    type Error = tower::BoxError;
    type Future = std::pin::Pin<Box<dyn std::future::Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, _: &mut std::task::Context<'_>) -> std::task::Poll<Result<(), Self::Error>> {
        std::task::Poll::Ready(Ok(()))
    }

    fn call(&mut self, _req: CreateChatCompletionRequest) -> Self::Future {
        let step = self.step.clone();
        Box::pin(async move {
            let mut current_step = step.lock().unwrap();
            *current_step += 1;
            
            Ok(tower_llm::StepOutcome::Next {
                messages: vec![ChatCompletionRequestMessage {
                    role: async_openai::types::Role::Assistant,
                    content: Some("Research completed: I've gathered information about the topic. The key findings are:\n1. Current market trends\n2. Technical specifications\n3. Competitive landscape\n\nPassing to analysis team for deeper insights.".to_string()),
                    name: None,
                    tool_calls: Some(vec![async_openai::types::ChatCompletionMessageToolCall {
                        id: "research_complete".to_string(),
                        r#type: async_openai::types::ChatCompletionMessageToolCallType::Function,
                        function: async_openai::types::FunctionCall {
                            name: "sequential_handoff".to_string(),
                            arguments: json!({"progress": "research_complete"}).to_string(),
                        },
                    }]),
                    tool_call_id: None,
                }],
                aux: tower_llm::StepAux {
                    prompt_tokens: 100,
                    completion_tokens: 50,
                    tool_invocations: 1,
                },
                invoked_tools: vec![],
            })
        })
    }
}

// Analysis agent - processes research findings
#[derive(Clone)]
struct AnalysisAgent {
    step: Arc<std::sync::Mutex<usize>>,
}

impl AnalysisAgent {
    fn new() -> Self {
        Self {
            step: Arc::new(std::sync::Mutex::new(0)),
        }
    }
}

impl Service<CreateChatCompletionRequest> for AnalysisAgent {
    type Response = tower_llm::StepOutcome;
    type Error = tower::BoxError;
    type Future = std::pin::Pin<Box<dyn std::future::Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, _: &mut std::task::Context<'_>) -> std::task::Poll<Result<(), Self::Error>> {
        std::task::Poll::Ready(Ok(()))
    }

    fn call(&mut self, _req: CreateChatCompletionRequest) -> Self::Future {
        let step = self.step.clone();
        Box::pin(async move {
            let mut current_step = step.lock().unwrap();
            *current_step += 1;
            
            Ok(tower_llm::StepOutcome::Next {
                messages: vec![ChatCompletionRequestMessage {
                    role: async_openai::types::Role::Assistant,
                    content: Some("Analysis completed: Based on the research findings, I've identified:\n\n**Key Insights:**\n- Market opportunity: $10M potential\n- Technical feasibility: High\n- Risk factors: Medium\n- Timeline: 6-9 months\n\n**Recommendations:**\n- Proceed with Phase 1 implementation\n- Allocate 3 engineers\n- Target Q2 launch\n\nPassing to report generator for final documentation.".to_string()),
                    name: None,
                    tool_calls: Some(vec![async_openai::types::ChatCompletionMessageToolCall {
                        id: "analysis_complete".to_string(),
                        r#type: async_openai::types::ChatCompletionMessageToolCallType::Function,
                        function: async_openai::types::FunctionCall {
                            name: "sequential_handoff".to_string(),
                            arguments: json!({"progress": "analysis_complete"}).to_string(),
                        },
                    }]),
                    tool_call_id: None,
                }],
                aux: tower_llm::StepAux {
                    prompt_tokens: 150,
                    completion_tokens: 80,
                    tool_invocations: 1,
                },
                invoked_tools: vec![],
            })
        })
    }
}

// Report agent - generates final deliverable
#[derive(Clone)]
struct ReportAgent;

impl Service<CreateChatCompletionRequest> for ReportAgent {
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
                    content: Some("# Executive Summary Report\n\n## Project Feasibility Analysis\n\n### Research Findings\n- Market trends identified and documented\n- Technical specifications compiled\n- Competitive analysis completed\n\n### Analysis Results\n- **Market Opportunity**: $10M potential revenue\n- **Technical Feasibility**: High confidence\n- **Risk Assessment**: Medium risk profile\n- **Timeline Estimate**: 6-9 months to market\n\n### Recommendations\n1. **Immediate Action**: Proceed with Phase 1 implementation\n2. **Resource Allocation**: 3 senior engineers + 1 PM\n3. **Target Timeline**: Q2 launch\n4. **Success Metrics**: User adoption >1K, Revenue >$500K\n\n### Next Steps\n- Secure budget approval\n- Begin technical design phase\n- Establish project team\n- Set up monitoring and metrics\n\n*Report generated through collaborative AI workflow: Research → Analysis → Documentation*".to_string()),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                }],
                aux: tower_llm::StepAux {
                    prompt_tokens: 200,
                    completion_tokens: 150,
                    tool_invocations: 0,
                },
            })
        })
    }
}

// Always routes to research agent to start the pipeline
#[derive(Clone)]
struct WorkflowPicker;

impl AgentPicker<CreateChatCompletionRequest> for WorkflowPicker {
    fn pick(&self, _req: &CreateChatCompletionRequest) -> Result<AgentName, tower::BoxError> {
        // Always start with research agent for this workflow
        Ok(AgentName("research_agent".to_string()))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("=== Sequential Handoff Example ===\n");
    
    println!("This example demonstrates sequential workflow orchestration:");
    println!("• Picker: Always routes to research_agent (workflow entry point)");
    println!("• Policy: Enforces Research → Analysis → Report sequence");
    println!("• Each agent contributes specialized expertise to the pipeline\n");

    // Set up agents
    let mut agents = HashMap::new();
    agents.insert(
        AgentName("research_agent".to_string()),
        tower::util::BoxService::new(ResearchAgent::new())
    );
    agents.insert(
        AgentName("analysis_agent".to_string()),
        tower::util::BoxService::new(AnalysisAgent::new())
    );
    agents.insert(
        AgentName("report_agent".to_string()),
        tower::util::BoxService::new(ReportAgent)
    );

    // Create sequential handoff policy
    let sequence = vec![
        AgentName("research_agent".to_string()),
        AgentName("analysis_agent".to_string()),
        AgentName("report_agent".to_string()),
    ];
    let handoff_policy = SequentialHandoffPolicy::new(sequence);

    // Create picker that always starts with research
    let picker = WorkflowPicker;

    // Build coordinator
    let coordinator = HandoffCoordinator::new(
        Arc::new(tokio::sync::Mutex::new(agents)),
        picker,
        handoff_policy,
    );

    println!("--- Running Complete Workflow ---");
    println!("Input: 'Analyze market opportunity for our new product feature'");
    println!("Expected flow: Research → Analysis → Report\n");

    let req = async_openai::types::CreateChatCompletionRequestArgs::default()
        .model("gpt-4o")
        .messages(vec![ChatCompletionRequestMessage {
            role: async_openai::types::Role::User,
            content: Some("Analyze market opportunity for our new product feature".to_string()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }])
        .build()?;

    let mut coordinator_service = coordinator;
    match coordinator_service.ready().await?.call(req).await {
        Ok(result) => {
            println!("✅ Workflow completed successfully!");
            println!("📊 Final result: {:?}\n", result);
        },
        Err(e) => println!("❌ Workflow failed: {}\n", e),
    }

    println!("=== Key Workflow Benefits ===");
    println!("1. **Specialization**: Each agent focuses on their expertise");
    println!("   - Research agent: Information gathering");
    println!("   - Analysis agent: Data processing and insights");
    println!("   - Report agent: Professional documentation");
    println!();
    println!("2. **Quality Gates**: Sequential handoffs ensure complete work");
    println!("   - Research must complete before analysis begins");
    println!("   - Analysis must complete before report generation");
    println!("   - Each step builds on previous work");
    println!();
    println!("3. **Consistency**: Enforced workflow prevents shortcuts");
    println!("   - SequentialHandoffPolicy ensures proper order");
    println!("   - No skipping steps or ad-hoc routing");
    println!("   - Predictable, repeatable process");
    println!();
    println!("4. **Picker vs Policy Roles**:");
    println!("   - **Picker**: Entry point routing (always → research_agent)");
    println!("   - **Policy**: Workflow orchestration (research → analysis → report)");
    println!("   - Clear separation of concerns");

    Ok(())
}