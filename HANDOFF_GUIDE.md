# Handoff System Guide

This guide explains the handoff system for multi-agent coordination in tower-llm. The handoff system enables agents to collaborate by transferring control, sharing context, and coordinating workflows.

## Architecture Overview

The handoff system is built on two key concepts that work together:

1. **AgentPicker**: WHO starts the conversation (initial routing)
2. **HandoffPolicy**: HOW agents collaborate during conversation (runtime handoffs)

This separation provides flexibility and clear responsibilities:

- **Picker** handles initial routing based on request analysis
- **Policy** manages ongoing collaboration and workflow orchestration

## Core Components

### AgentPicker Trait

Controls initial agent selection based on incoming requests:

```rust
pub trait AgentPicker<Request>: Clone + Send + Sync + 'static {
    fn pick(&self, req: &Request) -> Result<AgentName, BoxError>;
}
```

**Key characteristics:**

- Runs **once** at the start of each conversation
- Analyzes request content, metadata, or context
- Routes to the most appropriate initial agent
- Cannot change during conversation execution

**Example use cases:**

- Route math questions → MathAgent
- Route billing issues → BillingAgent
- Route technical problems → TechnicalAgent
- Route based on user role or permissions

### HandoffPolicy Trait

Manages runtime agent collaboration and handoffs:

```rust
pub trait HandoffPolicy: Send + Sync + 'static {
    fn handoff_tools(&self) -> Vec<ChatCompletionTool>;
    fn handle_handoff_tool(&self, invocation: &ToolInvocation) -> Result<HandoffRequest, BoxError>;
    fn should_handoff(&self, state: &LoopState, outcome: &StepOutcome) -> Option<HandoffRequest>;
    fn is_handoff_tool(&self, tool_name: &str) -> bool;

    // Transform messages during handoff (new in v0.0.5)
    fn transform_on_handoff(
        &self,
        messages: Vec<ChatCompletionRequestMessage>,
        from_agent: &str,
        to_agent: &str,
        handoff: &HandoffRequest,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<ChatCompletionRequestMessage>, BoxError>> + Send>> {
        Box::pin(async move { Ok(messages) })  // Default: no transformation
    }
}
```

**Key characteristics:**

- Runs **continuously** during conversation
- Provides handoff tools to agents
- Processes handoff tool invocations
- Can trigger automatic handoffs based on state
- Manages workflow orchestration

**Example use cases:**

- Explicit handoffs via tool calls
- Sequential workflows (research → analysis → report)
- Escalation chains (tier1 → specialist → manager)
- Conditional routing based on conversation state

## Message Transformation (New in v0.0.5)

The handoff system now supports async message transformation during handoffs. This powerful feature enables:

### Built-in Transformations

#### CompactingHandoffPolicy

Automatically compacts conversation history during handoffs to manage context length:

```rust
use tower_llm::groups::CompactingHandoffPolicy;
use tower_llm::auto_compaction::{CompactionPolicy, CompactionStrategy};

let compaction_policy = CompactionPolicy {
    compaction_model: "gpt-4o-mini".to_string(),
    compaction_strategy: CompactionStrategy::PreserveSystemAndRecent { recent_count: 5 },
    orphaned_tool_call_strategy: OrphanedToolCallStrategy::DropAndReappend,
    ..Default::default()
};

let provider = Arc::new(tokio::sync::Mutex::new(
    OpenAIProvider::new(client.clone())
));

// Wrap any existing policy with compaction
let compacting_policy = CompactingHandoffPolicy::new(
    base_handoff_policy,
    compaction_policy,
    provider,
);
```

### Custom Transformations

Implement `transform_on_handoff` in your custom policy:

```rust
impl HandoffPolicy for MyCustomPolicy {
    // ... other methods ...

    fn transform_on_handoff(
        &self,
        messages: Vec<ChatCompletionRequestMessage>,
        from_agent: &str,
        to_agent: &str,
        handoff: &HandoffRequest,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<ChatCompletionRequestMessage>, BoxError>> + Send>> {
        Box::pin(async move {
            // Example: Redact sensitive information
            let mut transformed = messages;
            if from_agent == "internal" && to_agent == "external" {
                transformed = redact_sensitive_data(transformed).await?;
            }
            Ok(transformed)
        })
    }
}
```

### Use Cases for Transformation

1. **Context Management**: Compact long conversations to stay within token limits
2. **Privacy**: Redact PII when crossing trust boundaries
3. **Translation**: Translate messages between agents operating in different languages
4. **Enrichment**: Add relevant context from external systems
5. **Filtering**: Remove irrelevant messages for specific agents
6. **Formatting**: Restructure messages for agent-specific requirements

### Transformation Pipeline

Transformations are applied in the HandoffCoordinator after a handoff is confirmed:

1. Handoff is triggered (via tool call or automatic policy)
2. Messages are prepared (original + accumulated)
3. `transform_on_handoff` is called with full context
4. Transformed messages are passed to the next agent
5. If transformation fails, original messages are used (graceful fallback)

## Policy Implementations

### ExplicitHandoffPolicy

Enables agents to explicitly hand off using tool calls:

```rust
let mut handoffs = HashMap::new();
handoffs.insert("handoff_to_specialist".to_string(), AgentName("specialist".to_string()));
let policy = ExplicitHandoffPolicy::new(handoffs);
```

**When to use:**

- Agents need full control over when to hand off
- Dynamic handoff decisions based on conversation content
- Ad-hoc collaboration between specialized agents

**Example scenario:**
Math agent recognizes explanation request and hands off to writing specialist.

### SequentialHandoffPolicy

Enforces predefined workflow sequences:

```rust
let sequence = vec![
    AgentName("research".to_string()),
    AgentName("analysis".to_string()),
    AgentName("report".to_string()),
];
let policy = SequentialHandoffPolicy::new(sequence);
```

**When to use:**

- Well-defined multi-step workflows
- Quality gates between process stages
- Consistent, repeatable processes

**Example scenario:**
Research → Analysis → Report generation pipeline.

### CompositeHandoffPolicy

Combines multiple policies for complex coordination:

```rust
let policy = CompositeHandoffPolicy::new(vec![
    Box::new(explicit_policy),
    Box::new(sequential_policy),
]);
```

**When to use:**

- Multiple coordination patterns in one system
- Flexible escalation and workflow paths
- Complex organizational hierarchies

**Example scenario:**
Support system with specialist escalations + manager escalation workflow.

## Integration with Tower Ecosystem

The handoff system integrates seamlessly with tower-llm's middleware:

### Policy Enforcement

```rust
let policy = tower_llm::CompositePolicy::new(vec![
    tower_llm::policies::until_no_tool_calls(),
    tower_llm::policies::max_steps(10),
    tower_llm::budgets::budget_policy(budget),
]);

let coordinator_with_policies = tower_llm::AgentLoopLayer::new(policy)
    .layer(handoff_coordinator);
```

**Benefits:**

- Budget tracking across entire multi-agent workflow
- Global step limits prevent infinite handoff loops
- Consistent termination conditions

### Resilience Patterns

```rust
let resilient_coordinator = tower_llm::resilience::TimeoutLayer::new(Duration::from_secs(60))
    .layer(tower_llm::resilience::RetryLayer::new(retry_policy, classifier)
        .layer(coordinator_with_policies));
```

**Benefits:**

- Fault-tolerant handoff execution
- Automatic retry for transient failures
- Global timeout protection

### Observability

```rust
let observable_coordinator = tower_llm::observability::TracingLayer::new()
    .layer(tower_llm::observability::MetricsLayer::new(collector)
        .layer(resilient_coordinator));
```

**Benefits:**

- Distributed tracing across agent handoffs
- Performance metrics for multi-agent workflows
- Debugging and monitoring capabilities

## Best Practices

### Picker Design

1. **Keep it simple**: Focus on clear routing rules
2. **Fail gracefully**: Always have a default agent
3. **Analyze content**: Use request content, not just metadata
4. **Consider context**: User role, permissions, history

### Policy Design

1. **Single responsibility**: Each policy handles one coordination pattern
2. **Compose wisely**: Use CompositeHandoffPolicy for complex scenarios
3. **Prevent loops**: Include maximum handoff limits
4. **Clear semantics**: Tool names should be self-explanatory

### Error Handling

1. **Validate handoffs**: Check target agents exist
2. **Handle failures**: Graceful degradation when handoffs fail
3. **Log decisions**: Record handoff reasons for debugging
4. **Monitor performance**: Track handoff success rates

### Testing Strategy

1. **Unit tests**: Test each policy independently
2. **Integration tests**: Test picker + policy combinations
3. **End-to-end tests**: Test complete workflows
4. **Error scenarios**: Test failure modes and edge cases

## Common Patterns

### Specialist Routing

```rust
// Picker routes by topic
impl AgentPicker<Request> for TopicPicker {
    fn pick(&self, req: &Request) -> Result<AgentName, BoxError> {
        if req.content.contains("billing") {
            Ok(AgentName("billing_agent".to_string()))
        } else if req.content.contains("technical") {
            Ok(AgentName("technical_agent".to_string()))
        } else {
            Ok(AgentName("general_agent".to_string()))
        }
    }
}

// Policy allows cross-specialist handoffs
let mut handoffs = HashMap::new();
handoffs.insert("consult_billing".to_string(), AgentName("billing_agent".to_string()));
handoffs.insert("consult_technical".to_string(), AgentName("technical_agent".to_string()));
let policy = ExplicitHandoffPolicy::new(handoffs);
```

### Escalation Chain

```rust
// Picker routes by urgency
impl AgentPicker<Request> for UrgencyPicker {
    fn pick(&self, req: &Request) -> Result<AgentName, BoxError> {
        if req.priority == Priority::Critical {
            Ok(AgentName("manager".to_string()))
        } else {
            Ok(AgentName("tier1".to_string()))
        }
    }
}

// Policy enforces escalation sequence
let escalation = vec![
    AgentName("tier1".to_string()),
    AgentName("tier2".to_string()),
    AgentName("manager".to_string()),
];
let policy = SequentialHandoffPolicy::new(escalation);
```

### Workflow Pipeline

```rust
// Picker always starts with research
impl AgentPicker<Request> for WorkflowPicker {
    fn pick(&self, _req: &Request) -> Result<AgentName, BoxError> {
        Ok(AgentName("research_agent".to_string()))
    }
}

// Policy enforces research → analysis → report
let workflow = vec![
    AgentName("research_agent".to_string()),
    AgentName("analysis_agent".to_string()),
    AgentName("report_agent".to_string()),
];
let policy = SequentialHandoffPolicy::new(workflow);
```

## Examples

The codebase includes comprehensive examples demonstrating different handoff patterns:

- **`handoff_basic.rs`**: Basic picker vs policy distinction with math/writing agents
- **`handoff_sequential.rs`**: Sequential workflow with research → analysis → report
- **`handoff_composite.rs`**: Complex support system with multiple policies
- **`handoff_integration.rs`**: Complete Tower ecosystem integration
- **`handoff_compaction.rs`**: Automatic conversation compaction during handoffs (NEW)

Each example includes detailed explanations and demonstrates real-world usage patterns.

## Migration from Existing Code

If you have existing agent coordination code, here's how to migrate:

### From Simple Routing

```rust
// Old approach
match request.topic {
    Topic::Math => math_agent.call(request).await,
    Topic::Writing => writing_agent.call(request).await,
}

// New approach
let picker = TopicPicker;
let policy = ExplicitHandoffPolicy::new(handoffs);
let coordinator = HandoffCoordinator::new(agents, picker, policy);
coordinator.call(request).await
```

### From Manual Handoffs

```rust
// Old approach
let result1 = research_agent.call(request).await?;
let request2 = build_analysis_request(result1);
let result2 = analysis_agent.call(request2).await?;
let request3 = build_report_request(result2);
let result3 = report_agent.call(request3).await?;

// New approach
let sequence = vec![research, analysis, report];
let policy = SequentialHandoffPolicy::new(sequence);
let coordinator = HandoffCoordinator::new(agents, picker, policy);
coordinator.call(request).await  // Handles entire workflow
```

## Performance Considerations

### Memory Usage

- HandoffCoordinator maintains conversation context
- Consider context trimming for long conversations
- Clone bounds may require Arc<Mutex<>> for shared state

### Latency

- Each handoff adds network round-trip
- Balance specialization vs. latency requirements
- Consider async handoff caching strategies

### Throughput

- Multiple coordinators can run concurrently
- Agent pools can be shared across coordinators
- Monitor resource utilization per agent

## Troubleshooting

### Common Issues

**HandoffRequest not processed**

- Check HandoffPolicy.is_handoff_tool() implementation
- Verify tool names match between policy and agent calls
- Enable tracing to see handoff decisions

**Infinite handoff loops**

- Implement maximum handoff limits in policies
- Use global step limits via tower_llm policies
- Monitor handoff patterns in production

**Clone trait errors**

- BoxService doesn't implement Clone
- Use Arc<Mutex<>> for shared agent state
- Consider implementing custom Clone where needed

### Debugging Tips

1. **Enable tracing**: Use TracingLayer for handoff visibility
2. **Check tool calls**: Verify handoff tools are available to agents
3. **Test in isolation**: Unit test picker and policy separately
4. **Monitor metrics**: Track handoff success/failure rates
5. **Validate configuration**: Ensure all referenced agents exist

## Contributing

When adding new handoff patterns:

1. Follow existing trait patterns
2. Add comprehensive tests
3. Include example usage
4. Update this documentation
5. Consider Tower ecosystem integration

The handoff system is designed to be extensible. New HandoffPolicy implementations should follow the established patterns and integrate cleanly with the Tower middleware stack.
