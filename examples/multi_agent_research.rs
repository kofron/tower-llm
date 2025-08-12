//! Multi-agent research assistant example
//!
//! This example demonstrates how multiple specialized agents can work together
//! through handoffs to complete complex research tasks.
//!
//! Run with: cargo run --example multi_agent_research

use openai_agents_rs::{
    handoff::Handoff, runner::RunConfig, Agent, FunctionTool, Runner,
};
use serde_json::Value;
use std::sync::{Arc, Mutex};

/// Simulated database of information
#[derive(Clone)]
struct KnowledgeBase {
    data: Arc<Mutex<Vec<(String, String)>>>,
}

impl KnowledgeBase {
    fn new() -> Self {
        let data = vec![
            // Technology facts
            ("Rust programming language".to_string(), 
             "Rust is a systems programming language focused on safety, speed, and concurrency. It was first released in 2010 and is known for its memory safety guarantees without garbage collection.".to_string()),
            ("Python programming language".to_string(),
             "Python is a high-level, interpreted programming language known for its simplicity and readability. Created by Guido van Rossum and first released in 1991.".to_string()),
            ("Machine Learning".to_string(),
             "Machine Learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. Key algorithms include neural networks, decision trees, and support vector machines.".to_string()),
            
            // Historical facts
            ("World War II".to_string(),
             "World War II (1939-1945) was a global conflict involving most of the world's nations. It ended with the Allied victory and reshaped the global political landscape.".to_string()),
            ("Moon Landing".to_string(),
             "The Apollo 11 mission successfully landed humans on the Moon on July 20, 1969. Neil Armstrong and Buzz Aldrin were the first humans to walk on the lunar surface.".to_string()),
            ("Industrial Revolution".to_string(),
             "The Industrial Revolution (1760-1840) was a period of major industrialization that transformed largely rural, agrarian societies into industrialized, urban ones.".to_string()),
            
            // Science facts
            ("DNA Structure".to_string(),
             "DNA (Deoxyribonucleic acid) has a double helix structure discovered by Watson and Crick in 1953. It carries genetic instructions for all known living organisms.".to_string()),
            ("Theory of Relativity".to_string(),
             "Einstein's Theory of Relativity consists of special relativity (1905) and general relativity (1915), revolutionizing our understanding of space, time, and gravity.".to_string()),
            ("Quantum Mechanics".to_string(),
             "Quantum mechanics describes nature at the smallest scales of energy levels of atoms and subatomic particles. Key principles include wave-particle duality and uncertainty.".to_string()),
        ];
        
        Self {
            data: Arc::new(Mutex::new(data)),
        }
    }

    fn search(&self, query: &str) -> Vec<(String, String)> {
        let data = self.data.lock().unwrap();
        let query_lower = query.to_lowercase();
        
        data.iter()
            .filter(|(topic, content)| {
                topic.to_lowercase().contains(&query_lower) ||
                content.to_lowercase().contains(&query_lower)
            })
            .cloned()
            .collect()
    }

    fn add_fact(&self, topic: String, content: String) {
        let mut data = self.data.lock().unwrap();
        println!("  [KnowledgeBase] Added fact about: {}", topic);
        data.push((topic, content));
    }
}

/// Create a search tool for the knowledge base
fn create_search_tool(kb: KnowledgeBase) -> Arc<FunctionTool> {
    Arc::new(FunctionTool::new(
        "search_knowledge".to_string(),
        "Search the knowledge base for information on a topic".to_string(),
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                }
            },
            "required": ["query"]
        }),
        move |args| {
            let query = args.get("query")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            
            println!("  [Search Tool] Searching for: {}", query);
            
            let results = kb.search(query);
            
            if results.is_empty() {
                Ok(Value::String(format!("No information found for '{}'", query)))
            } else {
                let formatted: Vec<String> = results.iter()
                    .map(|(topic, content)| format!("**{}**: {}", topic, content))
                    .collect();
                Ok(Value::String(formatted.join("\n\n")))
            }
        },
    ))
}

/// Create a fact storage tool
fn create_store_tool(kb: KnowledgeBase) -> Arc<FunctionTool> {
    Arc::new(FunctionTool::new(
        "store_fact".to_string(),
        "Store a new fact in the knowledge base".to_string(),
        serde_json::json!({
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Topic or title of the fact"
                },
                "content": {
                    "type": "string",
                    "description": "Detailed content of the fact"
                }
            },
            "required": ["topic", "content"]
        }),
        move |args| {
            let topic = args.get("topic")
                .and_then(|v| v.as_str())
                .unwrap_or("Unknown")
                .to_string();
            let content = args.get("content")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            
            kb.add_fact(topic.clone(), content);
            Ok(Value::String(format!("Successfully stored fact about '{}'", topic)))
        },
    ))
}

/// Create a summarization tool
fn create_summarize_tool() -> Arc<FunctionTool> {
    Arc::new(FunctionTool::simple(
        "summarize",
        "Create a brief summary of the provided text",
        |text: String| {
            println!("  [Summarize Tool] Creating summary...");
            // In a real implementation, this might use an NLP model
            let sentences: Vec<&str> = text.split(". ").collect();
            if sentences.len() <= 2 {
                text
            } else {
                format!("Summary: {}", sentences[0])
            }
        },
    ))
}

fn create_research_agent(kb: KnowledgeBase) -> Agent {
    Agent::simple(
        "ResearchAgent",
        "You are a research specialist. Your job is to search for information in the knowledge base 
         and provide comprehensive answers based on available data. Use the search_knowledge tool 
         to find relevant information. If you find multiple relevant facts, compile them into a 
         coherent response."
    )
    .with_tool(create_search_tool(kb))
}

fn create_analyst_agent() -> Agent {
    Agent::simple(
        "AnalystAgent",
        "You are an analytical specialist. Your job is to analyze information, identify patterns, 
         draw conclusions, and provide insights. You can summarize complex information and 
         synthesize multiple sources into coherent analysis."
    )
    .with_tool(create_summarize_tool())
}

fn create_archivist_agent(kb: KnowledgeBase) -> Agent {
    Agent::simple(
        "ArchivistAgent",
        "You are an information archivist. Your job is to store new facts and information 
         in the knowledge base for future reference. Use the store_fact tool to save 
         important information."
    )
    .with_tool(create_store_tool(kb))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Multi-Agent Research System ===\n");

    // Create shared knowledge base
    let kb = KnowledgeBase::new();

    // Create specialized agents
    let research_agent = create_research_agent(kb.clone());
    let analyst_agent = create_analyst_agent();
    let archivist_agent = create_archivist_agent(kb.clone());

    // Create handoffs
    let research_handoff = Handoff::new(
        research_agent,
        "Searches the knowledge base for specific information"
    );
    
    let analyst_handoff = Handoff::new(
        analyst_agent,
        "Analyzes and synthesizes information to provide insights"
    );
    
    let archivist_handoff = Handoff::new(
        archivist_agent,
        "Stores new information in the knowledge base"
    );

    // Create the coordinator agent that delegates to specialists
    let coordinator = Agent::simple(
        "CoordinatorAgent",
        "You are a research coordinator. You delegate tasks to specialized agents:
         - Use ResearchAgent when you need to search for existing information
         - Use AnalystAgent when you need to analyze or summarize information
         - Use ArchivistAgent when you need to store new information
         
         Always delegate to the appropriate specialist rather than trying to answer directly.
         After receiving results from specialists, provide a comprehensive response to the user."
    )
    .with_handoffs(vec![research_handoff, analyst_handoff, archivist_handoff])
    .with_max_turns(10);

    // Test queries that require different agents
    let queries = vec![
        "What can you tell me about Rust programming?",
        "Search for information about machine learning and then analyze its key concepts.",
        "Store this fact: 'Rust 1.0 was released on May 15, 2015' and then search for all Rust-related information.",
        "Find information about Einstein's theories and provide an analysis of their impact.",
    ];

    for query in queries {
        println!("Query: {}", query);
        println!("{}", "-".repeat(60));

        let result = Runner::run(
            coordinator.clone(),
            query,
            RunConfig::default(),
        )
        .await?;

        if result.is_success() {
            println!("\nResponse: {}\n", result.final_output);
            
            // Show which agents were involved
            let handoffs: Vec<_> = result.items.iter()
                .filter_map(|item| {
                    if let openai_agents_rs::items::RunItem::Handoff(h) = item {
                        Some(format!("{} → {}", h.from_agent, h.to_agent))
                    } else {
                        None
                    }
                })
                .collect();
            
            if !handoffs.is_empty() {
                println!("Agent handoffs: {}", handoffs.join(", "));
            }
        } else {
            println!("Error: {:?}", result.error());
        }
        
        println!("\n{}\n", "=".repeat(70));
    }

    // Interactive mode
    println!("Interactive mode - you can:");
    println!("  - Ask questions about topics in the knowledge base");
    println!("  - Request analysis of information");
    println!("  - Store new facts");
    println!("  - Type 'quit' to exit\n");

    use std::io::{self, Write};
    
    loop {
        print!("Your request: ");
        io::stdout().flush()?;
        
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();
        
        if input.eq_ignore_ascii_case("quit") || input.eq_ignore_ascii_case("exit") {
            println!("Goodbye!");
            break;
        }
        
        println!("\nProcessing...\n");
        
        let result = Runner::run(
            coordinator.clone(),
            input,
            RunConfig::default(),
        )
        .await?;
        
        if result.is_success() {
            println!("\n{}\n", result.final_output);
        } else {
            println!("\nError: {:?}\n", result.error());
        }
    }

    Ok(())
}
