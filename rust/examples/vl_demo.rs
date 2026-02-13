//! Qwen3-VL Vision Language Model Demo
//!
//! Demonstrates image understanding with Qwen3-VL

use mnn::Llm;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║     Qwen3-VL - Vision Language Model Demo                 ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();

    let model_path = "/Users/songjinde/git/MNNX/jan/models/Qwen3-VL-4B-Instruct-MNN/config.json";
    
    if !Path::new(model_path).exists() {
        eprintln!("❌ Model not found at: {}", model_path);
        std::process::exit(1);
    }

    println!("✓ Model found: {}", model_path);
    println!("✓ MNN Version: {}", mnn::get_version());
    println!();

    // Create and load VL model
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("📦 Loading Qwen3-VL Model...");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    let mut llm = Llm::create(model_path)?;
    llm.load()?;
    println!("✓ Model loaded successfully!");
    println!();

    // Test 1: Image description
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("🖼️  Test 1: Describe Image");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Query: 请详细描述这张图片中的内容");
    println!();
    
    let response = llm.response("请详细描述这张图片中的内容")?;
    println!("Response:");
    println!("{}", response);
    println!();

    // Test 2: Image Q&A
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("❓ Test 2: Question about Image");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Query: 图片中有几只猫？它们在做什么？");
    println!();
    
    llm.reset();
    let response = llm.response("图片中有几只猫？它们在做什么？")?;
    println!("Response:");
    println!("{}", response);
    println!();

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("✅ Vision-Language Demo Complete!");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    Ok(())
}
