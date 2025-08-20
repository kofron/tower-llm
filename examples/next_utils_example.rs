#[path = "../src/next/mod.rs"]
mod next;

fn main() {
    let req = next::simple_chat_request("system", "hello");
    println!("messages_len={}", req.messages.len());
}


