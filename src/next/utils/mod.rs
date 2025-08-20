//! Utility helpers (DX sugar) for the experimental `next` module
//!
//! What this module provides (spec)
//! - Helpers that keep everyday usage concise without hiding Tower semantics
//!
//! Exports
//! - `tool_typed<A, R, H>`: build a `ToolDef` from typed args/output using `schemars` and `serde`
//! - `simple_chat_request(system, user)` to quickly construct a raw OpenAI request
//! - `run(&mut AgentSvc, system, user)` to simplify executing an agent
//!
//! Implementation strategy
//! - `tool_typed` generates a JSON schema from `A: JsonSchema`, deserializes input `Value` into `A`, runs the handler, and serializes `R`
//! - Keep these helpers thin and side-effect free; they assemble or adapt existing services/layers
//!
//! Testing strategy
//! - Unit tests for request/message counts and typed tool schema and arg parsing
//! - (Optional) compile-fail tests for schema mismatches (future)
//!
//! Example ideas
//! - Define a typed tool `AddArgs { a, b }` and construct it via `tool_typed`
//! - Build a tiny agent and run a system/user prompt with `run` for a one-liner demo

pub use crate::next::{run, simple_chat_request, tool_typed};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_request_builds_two_messages() {
        let req = simple_chat_request("sys", "user");
        assert_eq!(req.messages.len(), 2);
    }
}
