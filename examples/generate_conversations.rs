//! Example: generate a few valid conversations using the generator.

use proptest::prelude::*;
use proptest::strategy::ValueTree;
use tower_llm::validation::gen::{valid_conversation, GeneratorConfig};

fn main() {
    let cfg = GeneratorConfig::default();
    let strat = valid_conversation(cfg);
    let mut runner = proptest::test_runner::TestRunner::default();
    for i in 0..3 {
        let value = strat.new_tree(&mut runner).unwrap().current();
        println!("=== Conversation {} ===", i + 1);
        for (idx, msg) in value.iter().enumerate() {
            println!("{:02}: {:?}", idx, msg);
        }
    }
}
