//! Example: LLM inference with MNN Rust bindings
//!
//! Usage:
//!   cargo run --example llm_example -- /path/to/model/config.json

use mnn::Llm;
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 2 {
        eprintln!("Usage: {} <path_to_model_config>", args[0]);
        eprintln!("Example: {} /path/to/qwen-1.5b/config.json", args[0]);
        std::process::exit(1);
    }
    
    let config_path = &args[1];
    
    println!("MNN Version: {}", mnn::get_version());
    println!("Creating LLM from config: {}", config_path);
    
    // Create LLM instance
    let mut llm = Llm::create(config_path)?;
    
    println!("Loading model...");
    llm.load()?;
    println!("Model loaded successfully!");
    
    // Test response
    println!("\n=== Testing response ===");
    let response = llm.response("你好")?;
    println!("Response: {}", response);
    
    // Test tokenizer
    println!("\n=== Testing tokenizer ===");
    let text = "Hello, world!";
    let tokens = llm.tokenizer_encode(text)?;
    println!("Encoded '{}': {:?}", text, tokens);
    
    // Decode first few tokens
    for &token_id in tokens.iter().take(5) {
        let decoded = llm.tokenizer_decode(token_id)?;
        println!("  Token {} -> '{}'", token_id, decoded);
    }
    
    // Test generate
    println!("\n=== Testing generate ===");
    let input_ids = llm.tokenizer_encode("Hello")?;
    println!("Input tokens: {:?}", input_ids);
    
    let output_ids = llm.generate(&input_ids)?;
    println!("Output tokens: {:?}", output_ids);
    
    // Get context info
    if let Some(ctx) = llm.get_context() {
        println!("\n=== Context Info ===");
        println!("Prompt length: {}", ctx.prompt_len);
        println!("Generated length: {}", ctx.gen_seq_len);
        println!("Prefill time: {} us", ctx.prefill_us);
        println!("Decode time: {} us", ctx.decode_us);
    }
    
    // Reset and test another query
    println!("\n=== Testing reset and new query ===");
    llm.reset();
    let response2 = llm.response("What is 1 + 1?")?;
    println!("Response: {}", response2);
    
    println!("\nDone!");
    Ok(())
}
