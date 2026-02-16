//! Qwen3-VL 完整演示 - Vision Language Model
//!
//! 演示如何使用 <img> 标签实现多模态输入
//!
//! # 机制说明
//!
//! MNN 的 VL 模型通过特殊的 XML 标签格式接收图片：
//!
//! ```text
//! <img>/absolute/path/to/image.jpg</img>
//! ```
//!
//! MNN C++ 引擎会：
//! 1. 解析 <img> 标签
//! 2. 读取图片文件
//! 3. 使用 visual encoder 处理图片
//! 4. 将视觉特征注入到 LLM
//!
//! # 使用示例

use mnn::Llm;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║     Qwen3-VL - Vision Language Model 完整演示            ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();

    let model_path = "/Users/songjinde/git/MNNX/jan/models/Qwen3-VL-4B-Instruct-MNN/config.json";
    let image_path = "/Users/songjinde/git/MNNX/MNN/rust/cat.jpg";

    if !Path::new(model_path).exists() {
        eprintln!("❌ 模型未找到: {}", model_path);
        eprintln!("\n请设置正确的模型路径");
        std::process::exit(1);
    }

    if !Path::new(image_path).exists() {
        eprintln!("❌ 图片未找到: {}", image_path);
        eprintln!("\n请设置正确的图片路径");
        std::process::exit(1);
    }

    println!("✓ 模型: {}", model_path);
    println!("✓ 图片: {}", image_path);
    println!("✓ MNN 版本: {}", mnn::get_version());
    println!();

    // 加载模型
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("📦 步骤 1: 加载模型");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    let mut llm = Llm::create(model_path)?;
    llm.load()?;
    println!("✓ 模型加载成功！");
    println!();

    // 测试 1: 基础图片描述
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("🖼️  测试 1: 图片描述");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    let prompt1 = format!(
        "<|im_start|>user\n<img>{}</img>请用中文详细描述这张图片。<|im_end|>\n<|im_start|>assistant\n",
        image_path
    );
    
    println!("Prompt: {}", prompt1);
    println!();
    
    let response1 = llm.response_with_options(&prompt1, false, 512)?;
    println!("📝 响应:");
    println!("{}", response1);
    println!();

    // 测试 2: 图片问答
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("❓ 测试 2: 图片问答");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    llm.reset();
    
    let prompt2 = format!(
        "<|im_start|>user\n<img>{}</img>图片中有几只猫？它们在做什么？<|im_end|>\n<|im_start|>assistant\n",
        image_path
    );
    
    println!("Prompt: {}", prompt2);
    println!();
    
    let response2 = llm.response_with_options(&prompt2, false, 256)?;
    println!("📝 响应:");
    println!("{}", response2);
    println!();

    // 测试 3: 英文描述
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("🌍 测试 3: 英文描述");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    llm.reset();
    
    let prompt3 = format!(
        "<|im_start|>user\n<img>{}</img>Describe this image in English.<|im_end|>\n<|im_start|>assistant\n",
        image_path
    );
    
    println!("Prompt: {}", prompt3);
    println!();
    
    let response3 = llm.response_with_options(&prompt3, false, 512)?;
    println!("📝 响应:");
    println!("{}", response3);
    println!();

    // 总结
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("✅ 所有测试完成！");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();
    println!("🎯 关键要点:");
    println!("  1. 使用 <img>{path}</img> 标签传递图片路径");
    println!("  2. 路径必须是绝对路径");
    println!("  3. 使用 Qwen3-VL 的 chat template 格式");
    println!("  4. 图片会自动被 visual encoder 处理");
    println!();

    Ok(())
}
