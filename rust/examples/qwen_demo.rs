//! Working Qwen3-0.6B demonstration
//!
//! This example demonstrates all working functionality:
//! - Model loading
//! - Text generation with response()
//! - Tokenizer encode/decode
//! - Context reset
//!
//! Note: generate() function has a known issue and is excluded
//!
//! Usage:
//!   DYLD_LIBRARY_PATH=../build:$DYLD_LIBRARY_PATH cargo run --example qwen_demo

use mnn::Llm;
use std::path::Path;
use std::time::Instant;

const MODEL_PATH: &str = "../models/qwen3-0.6b/config.json";

fn print_separator() {
    println!("\n{}", "=".repeat(60));
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("MNN Rust Binding - Qwen3-0.6B Demo");
    print_separator();

    // Check if model exists
    if !Path::new(MODEL_PATH).exists() {
        eprintln!("\n❌ Error: Model not found at {}", MODEL_PATH);
        eprintln!("\nPlease download the model first:");
        eprintln!("  cd rust && ./download_model.sh");
        std::process::exit(1);
    }

    println!("✓ Model found: {}", MODEL_PATH);
    println!("✓ MNN Version: {}", mnn::get_version());

    // Create and load model
    print_separator();
    println!("Loading model...");
    let start = Instant::now();
    let mut llm = Llm::create(MODEL_PATH)?;
    llm.load()?;
    let load_time = start.elapsed();
    println!("✓ Model loaded in {:?}", load_time);

    // Test 1: Simple greeting in Chinese
    print_separator();
    println!("Test 1: Simple Greeting (Chinese)");
    println!("Query: 你好");
    let start = Instant::now();
    let response = llm.response("你好")?;
    println!("Response: {}", response);
    println!("Time: {:?}", start.elapsed());

    // Test 2: Math question
    print_separator();
    println!("Test 2: Math Question");
    println!("Query: What is 2 + 2?");
    let start = Instant::now();
    let response = llm.response("What is 2 + 2?")?;
    println!("Response: {}", response);
    println!("Time: {:?}", start.elapsed());

    // Test 3: Reset context
    print_separator();
    println!("Test 3: Context Reset");
    llm.reset();
    println!("✓ Context reset");
    println!("Query: 介绍一下你自己");
    let start = Instant::now();
    let response = llm.response("介绍一下你自己")?;
    println!("Response: {}", response);
    println!("Time: {:?}", start.elapsed());

    // Test 4: Tokenizer
    print_separator();
    println!("Test 4: Tokenizer Encode/Decode");
    let test_text = "Hello, world! 你好世界！";
    println!("Text: {}", test_text);

    let tokens = llm.tokenizer_encode(test_text)?;
    println!("Encoded tokens: {:?}", tokens);
    println!("Token count: {}", tokens.len());

    print!("Decoded: ");
    for &token_id in tokens.iter() {
        let decoded = llm.tokenizer_decode(token_id)?;
        print!("{}", decoded);
    }
    println!();

    // Summary
    print_separator();
    println!("✅ All tests completed successfully!");
    println!("\nSummary:");
    println!("  - Model load time: {:?}", load_time);
    println!("  - response() function: ✓ Working");
    println!("  - tokenizer_encode(): ✓ Working");
    println!("  - tokenizer_decode(): ✓ Working");
    println!("  - reset(): ✓ Working");
    println!("\nNote: generate() function has a known issue and is excluded from this demo");
    print_separator();

    Ok(())
}
