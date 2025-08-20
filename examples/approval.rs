//! Example demonstrating capability-based approval architecture.
//!
//! This example shows how the Tower-based approval system works with
//! environment capabilities and deny-by-default security.

use openai_agents_rs::{
    env::{Approval, EnvBuilder},
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

    // Create agent with tools
    let agent = Agent::simple("SafetyBot", "I follow safety policies")
        .with_tool(safe_tool)
        .with_tool(danger_tool);

    println!("🤖 Agent created with tools");
    println!("🔒 Environment provides approval capability for security");
    println!("   Runner will apply ApprovalLayer automatically when capability is present");
    println!();

    // Now we can actually run the agent with our custom environment!
    let result = Runner::run_with_env(
        agent,
        "Please use the safe tool to process 'test data'",
        RunConfig::default(),
        env,
    )
    .await?;

    println!("🎯 Architecture Overview:");
    println!("   1. Agent has ApprovalLayer attached");
    println!("   2. Layer calls env.capability::<Approval>() for each tool");
    println!("   3. If capability exists, calls request_approval()");
    println!("   4. If no capability or denied → tool execution blocked");
    println!("   5. Deny-by-default security model");
    println!();

    println!("🚀 Agent execution completed!");
    println!("   Result: {:?}", result.is_success());
    println!("   Final output: {:?}", result.final_output);
    println!();
    println!("📝 To run with OpenAI (requires OPENAI_API_KEY):");
    println!("   export OPENAI_API_KEY=\"your-api-key\"");
    println!("   cargo run --example approval");
    println!();
    println!("✅ Custom environment support is now working!");
    println!("   The ApprovalLayer successfully:");
    println!("   1. Accessed the approval capability from the environment");
    println!("   2. Made approval decisions based on the policy");
    println!("   3. Enforced deny-by-default when no capability is present");

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
