#![allow(deprecated)]
//! Property tests: applying a single mutation to a valid conversation should yield the expected violation.

use proptest::prelude::*;
use tower_llm::validation::{gen, mutate, validate_conversation, ValidationPolicy, ViolationCode};

fn has_violation(
    violations: &[tower_llm::validation::Violation],
    predicate: fn(&ViolationCode) -> bool,
) -> bool {
    violations.iter().any(|v| predicate(&v.code))
}

fn is_assistant_before_user(c: &ViolationCode) -> bool {
    matches!(c, ViolationCode::AssistantBeforeUser { .. })
}
fn is_repeated_role_user(c: &ViolationCode) -> bool {
    matches!(c, ViolationCode::RepeatedRole { role, .. } if role == "user")
}
fn is_missing_tool_responses(c: &ViolationCode) -> bool {
    matches!(c, ViolationCode::MissingToolResponses { .. })
}
fn is_unknown_tool_response(c: &ViolationCode) -> bool {
    matches!(c, ViolationCode::UnknownToolResponse { .. })
}
fn is_unknown_or_tool_before(c: &ViolationCode) -> bool {
    is_unknown_tool_response(c) || is_tool_before_assistant(c)
}
fn is_tool_responses_out_of_order(c: &ViolationCode) -> bool {
    matches!(c, ViolationCode::ToolResponsesOutOfOrder { .. })
}
fn is_duplicate_tool_response(c: &ViolationCode) -> bool {
    matches!(c, ViolationCode::DuplicateToolResponse { .. })
}
fn is_system_not_first(c: &ViolationCode) -> bool {
    matches!(c, ViolationCode::SystemNotFirst { .. })
}

fn is_duplicate_tool_call_ids_in_assistant(c: &ViolationCode) -> bool {
    matches!(c, ViolationCode::DuplicateToolCallIdsInAssistant { .. })
}
fn is_empty_tool_call_id_in_assistant(c: &ViolationCode) -> bool {
    matches!(c, ViolationCode::EmptyToolCallIdInAssistant { .. })
}
fn is_empty_tool_message_id(c: &ViolationCode) -> bool {
    matches!(c, ViolationCode::EmptyToolMessageId { .. })
}
fn is_tool_before_assistant(c: &ViolationCode) -> bool {
    matches!(c, ViolationCode::ToolBeforeAssistant { .. })
}
fn is_tool_responses_not_contiguous(c: &ViolationCode) -> bool {
    matches!(c, ViolationCode::ToolResponsesNotContiguous { .. })
}
fn is_no_user_message(c: &ViolationCode) -> bool {
    matches!(c, ViolationCode::NoUserMessage)
}

proptest! {
    #[test]
    fn mutation_assistant_before_user_detected(mut msgs in gen::valid_conversation(gen::GeneratorConfig::default())) {
        let applied = mutate::apply_violation(&mut msgs, mutate::MutationKind::AssistantBeforeUser);
        prop_assume!(applied);
        let out = validate_conversation(&msgs, &ValidationPolicy::default()).expect("violations after mutation");
        prop_assert!(has_violation(&out, is_assistant_before_user));
    }

    #[test]
    fn mutation_repeated_user_detected(mut msgs in gen::valid_conversation(gen::GeneratorConfig::default())) {
        let applied = mutate::apply_violation(&mut msgs, mutate::MutationKind::RepeatedUser);
        prop_assume!(applied);
        let out = validate_conversation(&msgs, &ValidationPolicy::default()).expect("violations after mutation");
        prop_assert!(has_violation(&out, is_repeated_role_user));
    }

    #[test]
    fn mutation_missing_one_tool_response_detected(mut msgs in gen::valid_conversation(gen::GeneratorConfig { must_have_tool_calls: true, min_tool_calls: 1, ..Default::default() })) {
        let applied = mutate::apply_violation(&mut msgs, mutate::MutationKind::MissingOneToolResponse);
        prop_assume!(applied);
        let out = validate_conversation(&msgs, &ValidationPolicy::default()).expect("violations after mutation");
        prop_assert!(has_violation(&out, is_missing_tool_responses));
    }

    #[test]
    fn mutation_unknown_tool_response_detected(mut msgs in gen::valid_conversation(gen::GeneratorConfig::default())) {
        let applied = mutate::apply_violation(&mut msgs, mutate::MutationKind::UnknownToolResponse);
        prop_assume!(applied);
        let out = validate_conversation(&msgs, &ValidationPolicy::default()).expect("violations after mutation");
        prop_assert!(has_violation(&out, is_unknown_or_tool_before));
    }

    #[test]
    fn mutation_reorder_tool_responses_detected(mut msgs in gen::valid_conversation(gen::GeneratorConfig { must_have_tool_calls: true, min_tool_calls: 2, ..Default::default() })) {
        let applied = mutate::apply_violation(&mut msgs, mutate::MutationKind::ReorderToolResponses);
        prop_assume!(applied);
        let out = validate_conversation(&msgs, &ValidationPolicy::default()).expect("violations after mutation");
        prop_assert!(has_violation(&out, is_tool_responses_out_of_order));
    }

    #[test]
    fn mutation_duplicate_tool_response_detected(mut msgs in gen::valid_conversation(gen::GeneratorConfig { must_have_tool_calls: true, min_tool_calls: 1, ..Default::default() })) {
        let applied = mutate::apply_violation(&mut msgs, mutate::MutationKind::DuplicateToolResponse);
        prop_assume!(applied);
        let out = validate_conversation(&msgs, &ValidationPolicy::default()).expect("violations after mutation");
        prop_assert!(has_violation(&out, is_duplicate_tool_response));
    }

    #[test]
    fn mutation_system_not_first_detected(mut msgs in gen::valid_conversation(gen::GeneratorConfig::default())) {
        let applied = mutate::apply_violation(&mut msgs, mutate::MutationKind::SystemNotFirst);
        prop_assume!(applied);
        let out = validate_conversation(&msgs, &ValidationPolicy::default()).expect("violations after mutation");
        prop_assert!(has_violation(&out, is_system_not_first));
    }
}

proptest! {
    #[test]
    fn mutation_duplicate_tool_call_ids_in_assistant_detected(mut msgs in gen::valid_conversation(gen::GeneratorConfig { must_have_tool_calls: true, min_tool_calls: 2, ..Default::default() })) {
        let applied = mutate::apply_violation(&mut msgs, mutate::MutationKind::DuplicateToolCallIdsInAssistant);
        prop_assume!(applied);
        let out = validate_conversation(&msgs, &ValidationPolicy::default()).expect("violations after mutation");
        prop_assert!(has_violation(&out, is_duplicate_tool_call_ids_in_assistant));
    }

    #[test]
    fn mutation_empty_tool_call_id_in_assistant_detected(mut msgs in gen::valid_conversation(gen::GeneratorConfig { must_have_tool_calls: true, min_tool_calls: 1, ..Default::default() })) {
        let applied = mutate::apply_violation(&mut msgs, mutate::MutationKind::EmptyToolCallIdInAssistant);
        prop_assume!(applied);
        let out = validate_conversation(&msgs, &ValidationPolicy::default()).expect("violations after mutation");
        prop_assert!(has_violation(&out, is_empty_tool_call_id_in_assistant));
    }

    #[test]
    fn mutation_empty_tool_message_id_detected(mut msgs in gen::valid_conversation(gen::GeneratorConfig { must_have_tool_calls: true, min_tool_calls: 1, ..Default::default() })) {
        let applied = mutate::apply_violation(&mut msgs, mutate::MutationKind::EmptyToolMessageId);
        prop_assume!(applied);
        let out = validate_conversation(&msgs, &ValidationPolicy::default()).expect("violations after mutation");
        prop_assert!(has_violation(&out, is_empty_tool_message_id));
    }

    #[test]
    fn mutation_tool_before_assistant_detected(mut msgs in gen::valid_conversation(gen::GeneratorConfig::default())) {
        let applied = mutate::apply_violation(&mut msgs, mutate::MutationKind::ToolBeforeAssistant);
        prop_assume!(applied);
        let out = validate_conversation(&msgs, &ValidationPolicy::default()).expect("violations after mutation");
        prop_assert!(has_violation(&out, is_tool_before_assistant));
    }

    #[test]
    fn mutation_tool_responses_not_contiguous_detected(mut msgs in gen::valid_conversation(gen::GeneratorConfig { must_have_tool_calls: true, min_tool_calls: 1, ..Default::default() })) {
        let applied = mutate::apply_violation(&mut msgs, mutate::MutationKind::ToolResponsesNotContiguous);
        prop_assume!(applied);
        let mut policy = ValidationPolicy::default();
        policy.enforce_contiguous_tool_responses = true;
        let out = validate_conversation(&msgs, &policy).expect("violations after mutation");
        prop_assert!(has_violation(&out, is_tool_responses_not_contiguous));
    }

    #[test]
    fn mutation_remove_all_users_detected(mut msgs in gen::valid_conversation(gen::GeneratorConfig::default())) {
        let applied = mutate::apply_violation(&mut msgs, mutate::MutationKind::RemoveAllUsers);
        prop_assume!(applied);
        let mut policy = ValidationPolicy::default();
        policy.require_user_present = true;
        policy.require_user_first = false;
        let out = validate_conversation(&msgs, &policy).expect("violations after mutation");
        prop_assert!(has_violation(&out, is_no_user_message));
    }
}
