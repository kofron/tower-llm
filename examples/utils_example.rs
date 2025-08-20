// Core module is now at root level
// use tower_llm directly

fn main() {
    let req = tower_llm::simple_chat_request("system", "hello");
    println!("messages_len={}", req.messages.len());
}
