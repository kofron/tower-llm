//! Example demonstrating capability-based approval architecture.
//!
//! This example shows how the Tower-based approval system works with
//! environment capabilities and deny-by-default security.

use openai_agents_rs::{
    env::{Approval, EnvBuilder},
    layers,
    runner::RunConfig,
    Agent, FunctionTool, Runner,
};
use std::sync::Arc;

/// Example approval implementation that approves safe operations
/// and denies dangerous operations.
struct SafetyApproval;

impl Approval for SafetyApproval {
    fn request_approval(&self, operation: &str, details: &str) -> bool {
        println!("🛡️  Approval requested for operation: {}", operation);
        println!("   Details: {}", details);

        // Simple safety policy: approve "safe" operations, deny "danger"
        let approved = operation == "safe";
        println!(
            "   Decision: {}",
            if approved {
                "✅ APPROVED"
            } else {
                "❌ DENIED"
            }
        );
        approved
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Tower-Based Approval Architecture Demo ===");
    println!();

    // Create tools that will be subject to approval
    let safe_tool = Arc::new(FunctionTool::simple(
        "safe",
        "A safe operation that should be approved",
        |s: String| format!("✅ Safe operation completed: {}", s),
    ));

    let danger_tool = Arc::new(FunctionTool::simple(
        "danger",
        "A dangerous operation that should be denied",
        |s: String| format!("💥 Dangerous operation: {}", s),
    ));

    // Create environment with approval capability
    let env = EnvBuilder::new()
        .with_capability(Arc::new(SafetyApproval))
        .build();

    println!("📋 Environment created with SafetyApproval capability");
    println!("   Policy: approve 'safe' operations, deny 'danger' operations");
    println!();

    // Create agent with ApprovalLayer applied
    let agent = Agent::simple("SafetyBot", "I follow safety policies")
        .with_tool(safe_tool)
        .with_tool(danger_tool)
        .layer(layers::ApprovalLayer); // This layer checks approval capability

    println!("🤖 Agent created with ApprovalLayer applied");
    println!("   All tool executions will require approval");
    println!();

    // Note: This is a demonstration of the architecture. The current Runner
    // uses DefaultEnv internally and doesn't support custom environments yet.
    // let result = Runner::run(agent, "Demonstrate safe and danger tools", RunConfig::default()).await?;

    println!("🎯 Architecture Overview:");
    println!("   1. Agent has ApprovalLayer attached");
    println!("   2. Layer calls env.capability::<Approval>() for each tool");
    println!("   3. If capability exists, calls request_approval()");
    println!("   4. If no capability or denied → tool execution blocked");
    println!("   5. Deny-by-default security model");
    println!();

    println!("🚀 Example setup completed!");
    println!();
    println!("📝 To run this agent with OpenAI (requires OPENAI_API_KEY):");
    println!("   export OPENAI_API_KEY=\"your-api-key\"");
    println!("   cargo run --example approval");
    println!();
    println!("🔧 Note: The current Runner uses DefaultEnv internally.");
    println!("   In a production system, you would:");
    println!("   1. Extend Runner to accept custom environments");
    println!("   2. Or create environment-aware tool services directly");
    println!("   3. Example: tool.into_service::<CapabilityEnv>().layer(ApprovalLayer)");

    println!();
    println!("🎉 Demo completed!");
    println!();
    println!("💡 Key benefits of this architecture:");
    println!("   • Type-safe: ApprovalLayer<E> requires E: Env");
    println!("   • Deny-by-default: missing capability = automatic denial");
    println!("   • Composable: layers stack cleanly with other Tower middleware");
    println!("   • Testable: approval logic is pure and isolated in capability");
    println!("   • Flexible: different approval policies per environment");

    Ok(())
}
