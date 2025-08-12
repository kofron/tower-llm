//! Usage tracking for token consumption and costs

use serde::{Deserialize, Serialize};
use std::ops::Add;

/// Token usage information
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Usage {
    /// Number of prompt tokens used
    pub prompt_tokens: usize,

    /// Number of completion tokens generated
    pub completion_tokens: usize,

    /// Total tokens (prompt + completion)
    pub total_tokens: usize,

    /// Number of requests made
    pub request_count: usize,
}

impl Usage {
    /// Create a new usage instance
    pub fn new(prompt_tokens: usize, completion_tokens: usize) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            request_count: 1,
        }
    }

    /// Create an empty usage instance
    pub fn empty() -> Self {
        Self::default()
    }

    /// Add another usage to this one  
    pub fn add_usage(&mut self, other: &Usage) {
        self.prompt_tokens += other.prompt_tokens;
        self.completion_tokens += other.completion_tokens;
        self.total_tokens += other.total_tokens;
        self.request_count += other.request_count;
    }

    /// Estimate cost based on model pricing (in USD)
    /// Note: These are example prices and should be updated based on actual pricing
    pub fn estimate_cost(&self, model: &str) -> f64 {
        let (prompt_price, completion_price) = match model {
            "gpt-4" | "gpt-4-0613" => (0.03, 0.06), // per 1K tokens
            "gpt-4-32k" => (0.06, 0.12),
            "gpt-3.5-turbo" | "gpt-3.5-turbo-0613" => (0.0015, 0.002),
            "gpt-3.5-turbo-16k" => (0.003, 0.004),
            _ => (0.002, 0.002), // Default pricing
        };

        let prompt_cost = (self.prompt_tokens as f64 / 1000.0) * prompt_price;
        let completion_cost = (self.completion_tokens as f64 / 1000.0) * completion_price;

        prompt_cost + completion_cost
    }

    /// Check if usage exceeds limits
    pub fn exceeds_limits(&self, max_tokens: Option<usize>, max_requests: Option<usize>) -> bool {
        if let Some(max) = max_tokens {
            if self.total_tokens > max {
                return true;
            }
        }
        if let Some(max) = max_requests {
            if self.request_count > max {
                return true;
            }
        }
        false
    }
}

impl Add for Usage {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            prompt_tokens: self.prompt_tokens + other.prompt_tokens,
            completion_tokens: self.completion_tokens + other.completion_tokens,
            total_tokens: self.total_tokens + other.total_tokens,
            request_count: self.request_count + other.request_count,
        }
    }
}

/// Aggregated usage statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UsageStats {
    /// Total usage across all runs
    pub total: Usage,

    /// Usage by model
    pub by_model: std::collections::HashMap<String, Usage>,

    /// Usage by agent
    pub by_agent: std::collections::HashMap<String, Usage>,
}

impl UsageStats {
    /// Create new empty stats
    pub fn new() -> Self {
        Self::default()
    }

    /// Record usage for a specific model and agent
    pub fn record(&mut self, model: &str, agent: &str, usage: Usage) {
        // Update total
        self.total.add_usage(&usage);

        // Update by model
        self.by_model
            .entry(model.to_string())
            .and_modify(|u| u.add_usage(&usage))
            .or_insert(usage.clone());

        // Update by agent
        self.by_agent
            .entry(agent.to_string())
            .and_modify(|u| u.add_usage(&usage))
            .or_insert(usage);
    }

    /// Get total estimated cost
    pub fn total_cost(&self) -> f64 {
        self.by_model
            .iter()
            .map(|(model, usage)| usage.estimate_cost(model))
            .sum()
    }

    /// Get a summary report
    pub fn summary(&self) -> String {
        let mut report = format!(
            "Usage Summary:\n\
             Total Tokens: {}\n\
             Total Requests: {}\n\
             Estimated Cost: ${:.4}\n",
            self.total.total_tokens,
            self.total.request_count,
            self.total_cost()
        );

        if !self.by_model.is_empty() {
            report.push_str("\nBy Model:\n");
            for (model, usage) in &self.by_model {
                report.push_str(&format!(
                    "  {}: {} tokens, ${:.4}\n",
                    model,
                    usage.total_tokens,
                    usage.estimate_cost(model)
                ));
            }
        }

        if !self.by_agent.is_empty() {
            report.push_str("\nBy Agent:\n");
            for (agent, usage) in &self.by_agent {
                report.push_str(&format!(
                    "  {}: {} tokens, {} requests\n",
                    agent, usage.total_tokens, usage.request_count
                ));
            }
        }

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_usage_creation() {
        let usage = Usage::new(100, 50);
        assert_eq!(usage.prompt_tokens, 100);
        assert_eq!(usage.completion_tokens, 50);
        assert_eq!(usage.total_tokens, 150);
        assert_eq!(usage.request_count, 1);
    }

    #[test]
    fn test_usage_add() {
        let mut usage1 = Usage::new(100, 50);
        let usage2 = Usage::new(200, 100);

        usage1.add_usage(&usage2);

        assert_eq!(usage1.prompt_tokens, 300);
        assert_eq!(usage1.completion_tokens, 150);
        assert_eq!(usage1.total_tokens, 450);
        assert_eq!(usage1.request_count, 2);
    }

    #[test]
    fn test_usage_add_operator() {
        let usage1 = Usage::new(100, 50);
        let usage2 = Usage::new(200, 100);

        let combined = usage1 + usage2;

        assert_eq!(combined.prompt_tokens, 300);
        assert_eq!(combined.completion_tokens, 150);
        assert_eq!(combined.total_tokens, 450);
        assert_eq!(combined.request_count, 2);
    }

    #[test]
    fn test_cost_estimation() {
        let usage = Usage::new(1000, 500); // 1K prompt, 0.5K completion

        // GPT-4: $0.03/1K prompt, $0.06/1K completion
        let gpt4_cost = usage.estimate_cost("gpt-4");
        assert!((gpt4_cost - 0.06).abs() < 0.001); // 0.03 + 0.03

        // GPT-3.5: $0.0015/1K prompt, $0.002/1K completion
        let gpt35_cost = usage.estimate_cost("gpt-3.5-turbo");
        assert!((gpt35_cost - 0.0025).abs() < 0.0001); // 0.0015 + 0.001
    }

    #[test]
    fn test_exceeds_limits() {
        let usage = Usage::new(1000, 500);

        // Should not exceed
        assert!(!usage.exceeds_limits(Some(2000), Some(2)));

        // Should exceed token limit
        assert!(usage.exceeds_limits(Some(1000), None));

        // Should exceed request limit
        assert!(usage.exceeds_limits(None, Some(0)));
    }

    #[test]
    fn test_usage_stats() {
        let mut stats = UsageStats::new();

        stats.record("gpt-4", "Agent1", Usage::new(100, 50));
        stats.record("gpt-4", "Agent2", Usage::new(200, 100));
        stats.record("gpt-3.5-turbo", "Agent1", Usage::new(300, 150));

        assert_eq!(stats.total.prompt_tokens, 600);
        assert_eq!(stats.total.completion_tokens, 300);
        assert_eq!(stats.total.total_tokens, 900);
        assert_eq!(stats.total.request_count, 3);

        assert_eq!(stats.by_model.len(), 2);
        assert_eq!(stats.by_agent.len(), 2);

        let gpt4_usage = stats.by_model.get("gpt-4").unwrap();
        assert_eq!(gpt4_usage.total_tokens, 450);

        let agent1_usage = stats.by_agent.get("Agent1").unwrap();
        assert_eq!(agent1_usage.total_tokens, 600);
    }

    #[test]
    fn test_usage_stats_summary() {
        let mut stats = UsageStats::new();
        stats.record("gpt-4", "TestAgent", Usage::new(1000, 500));

        let summary = stats.summary();
        assert!(summary.contains("Total Tokens: 1500"));
        assert!(summary.contains("Total Requests: 1"));
        assert!(summary.contains("By Model:"));
        assert!(summary.contains("gpt-4"));
        assert!(summary.contains("By Agent:"));
        assert!(summary.contains("TestAgent"));
    }

    #[test]
    fn test_usage_serialization() {
        let usage = Usage::new(100, 50);
        let serialized = serde_json::to_string(&usage).unwrap();
        let deserialized: Usage = serde_json::from_str(&serialized).unwrap();

        assert_eq!(usage, deserialized);
    }

    #[test]
    fn test_empty_usage() {
        let usage = Usage::empty();
        assert_eq!(usage.prompt_tokens, 0);
        assert_eq!(usage.completion_tokens, 0);
        assert_eq!(usage.total_tokens, 0);
        assert_eq!(usage.request_count, 0);
    }
}
