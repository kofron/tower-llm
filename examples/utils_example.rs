// Core module is now at root level
// use openai_agents_rs directly

fn main() {
    let req = openai_agents_rs::simple_chat_request("system", "hello");
    println!("messages_len={}", req.messages.len());
}


