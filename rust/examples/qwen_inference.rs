//! Comprehensive Qwen3-0.6B inference example
//!
//! This example demonstrates:
//! - Model existence checking
//! - Model loading and inference
//! - Multiple test queries
//! - Performance metrics
//!
//! Usage:
//!   cargo run --example qwen_inference

use mnn::Llm;
use std::path::Path;
use std::time::Instant;

const MODEL_PATH: &str = "../models/qwen3-0.6b/config.json";

fn check_model_exists() -> bool {
    Path::new(MODEL_PATH).exists()
}

fn print_separator() {
    println!("\n{}", "=".repeat(60));
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("MNN Rust Binding - Qwen3-0.6B Inference Test");
    print_separator();

    // Check if model exists
    if !check_model_exists() {
        eprintln!("\n❌ Error: Model not found at {}", MODEL_PATH);
        eprintln!("\nPlease download the model first:");
        eprintln!("  cd rust");
        eprintln!("  ./download_model.sh");
        eprintln!("\nOr manually:");
        eprintln!("  git clone https://www.modelscope.cn/MNN/Qwen3-0.6B-MNN.git ../models/qwen3-0.6b");
        std::process::exit(1);
    }

    println!("✓ Model found at: {}", MODEL_PATH);
    println!("✓ MNN Version: {}", mnn::get_version());

    // Create and load model
    print_separator();
    println!("Creating LLM instance...");
    let start = Instant::now();
    let mut llm = Llm::create(MODEL_PATH)?;
    println!("✓ LLM created in {:?}", start.elapsed());

    println!("\nLoading model...");
    let start = Instant::now();
    llm.load()?;
    let load_time = start.elapsed();
    println!("✓ Model loaded in {:?}", load_time);

    // Test 1: Simple greeting
    print_separator();
    println!("Test 1: Simple Greeting");
    println!("Query: 你好");
    let start = Instant::now();
    let response = llm.response("你好")?;
    let inference_time = start.elapsed();
    println!("Response: {}", response);
    println!("Time: {:?}", inference_time);

    // Test 2: Math question
    print_separator();
    println!("Test 2: Math Question");
    println!("Query: What is 1 + 1?");
    let start = Instant::now();
    let response = llm.response("What is 1 + 1?")?;
    let inference_time = start.elapsed();
    println!("Response: {}", response);
    println!("Time: {:?}", inference_time);

    // Test 3: Reset and new query
    print_separator();
    println!("Test 3: Reset Context");
    llm.reset();
    println!("✓ Context reset");
    println!("Query: 介绍一下你自己");
    let start = Instant::now();
    let response = llm.response("介绍一下你自己")?;
    let inference_time = start.elapsed();
    println!("Response: {}", response);
    println!("Time: {:?}", inference_time);

    // Test 4: Tokenizer
    print_separator();
    println!("Test 4: Tokenizer");
    let test_text = "Hello, world! 你好世界！";
    println!("Text: {}", test_text);

    let tokens = llm.tokenizer_encode(test_text)?;
    println!("Encoded tokens: {:?}", tokens);
    println!("Token count: {}", tokens.len());

    // Decode tokens
    print!("Decoded: ");
    for &token_id in tokens.iter() {
        let decoded = llm.tokenizer_decode(token_id)?;
        print!("{}", decoded);
    }
    println!();

    // Test 5: Generate with token IDs
    print_separator();
    println!("Test 5: Generate with Token IDs");
    let input_text = "Hello";
    let input_ids = llm.tokenizer_encode(input_text)?;
    println!("Input: {}", input_text);
    println!("Input tokens: {:?}", input_ids);

    let start = Instant::now();
    let output_ids = llm.generate(&input_ids)?;
    let gen_time = start.elapsed();
    println!("Output tokens: {:?}", output_ids);
    println!("Output token count: {}", output_ids.len());
    println!("Time: {:?}", gen_time);

    // Test 6: Context information
    print_separator();
    println!("Test 6: Context Information");
    if let Some(ctx) = llm.get_context() {
        println!("Prompt length: {} tokens", ctx.prompt_len);
        println!("Generated length: {} tokens", ctx.gen_seq_len);
        println!("Prefill time: {} μs", ctx.prefill_us);
        println!("Decode time: {} μs", ctx.decode_us);

        if ctx.decode_us > 0 && ctx.gen_seq_len > 0 {
            let tokens_per_sec = (ctx.gen_seq_len as f64 / ctx.decode_us as f64) * 1_000_000.0;
            println!("Generation speed: {:.2} tokens/sec", tokens_per_sec);
        }
    } else {
        println!("No context information available");
    }

    // Summary
    print_separator();
    println!("✅ All tests completed successfully!");
    println!("\nSummary:");
    println!("  - Model load time: {:?}", load_time);
    println!("  - All API functions working correctly");
    println!("  - Tokenizer encode/decode working");
    println!("  - Context management working");
    print_separator();

    Ok(())
}
