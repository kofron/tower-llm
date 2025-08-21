#![allow(deprecated)]
//! Property tests for conversation validation generators and mutators.

use proptest::prelude::*;
use tower_llm::validation::{gen, validate_conversation, ValidationPolicy};

proptest! {
    #[test]
    fn generated_valid_conversations_have_no_violations(msgs in gen::valid_conversation(gen::GeneratorConfig::default())) {
        let out = validate_conversation(&msgs, &ValidationPolicy::default());
        prop_assert!(out.is_none());
    }
}

// Note: Keep PBT light initially to avoid long CI times; curated tests exist in unit tests.
