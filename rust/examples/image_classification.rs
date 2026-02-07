//! Image Classification Example using MobileNetV2
//!
//! This example demonstrates:
//! - Loading a pre-trained model
//! - Image preprocessing with ImageProcess
//! - Running inference
//! - Getting and interpreting results
//!
//! Usage:
//!   DYLD_LIBRARY_PATH=../build:$DYLD_LIBRARY_PATH \
//!   cargo run --example image_classification -- \
//!     ../benchmark/models/MobileNetV2_224.mnn <image_path>

use mnn::{ImageFormat, ImageProcess, ImageProcessConfig, FilterType, Interpreter, Tensor};
use image::GenericImageView;
use std::env;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <model_path> <image_path>", args[0]);
        eprintln!("\nExample:");
        eprintln!("  {} ../benchmark/models/MobileNetV2_224.mnn test.jpg", args[0]);
        eprintln!("\nNote: MobileNetV2 expects 224x224 RGB input");
        std::process::exit(1);
    }

    let model_path = &args[1];
    let image_path = &args[2];

    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║     MNN Rust Binding - Image Classification Demo          ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();
    println!("MNN Version: {}", mnn::get_version());
    println!("Model: {}", model_path);
    println!("Image: {}", image_path);
    println!();

    // Load model
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("📦 Step 1: Loading Model");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    let start = Instant::now();
    let mut interpreter = Interpreter::create_from_file(model_path)?;
    println!("✓ Model loaded in {:?}", start.elapsed());

    // Create session
    let num_threads = 4;
    let session = interpreter.create_session(num_threads)?;
    println!("✓ Session created with {} threads", num_threads);

    // Get input tensor info
    let input_tensor = interpreter
        .get_session_input(&session, None)
        .expect("Failed to get input tensor");
    let input_shape = input_tensor.shape();
    println!("✓ Input shape: {:?}", input_shape);

    let batch = input_shape[0];
    let channels = input_shape[1];
    let height = input_shape[2] as u32;
    let width = input_shape[3] as u32;
    println!("✓ Model expects: {}x{}x{}x{} (BxCxHxW)", batch, channels, height, width);

    // Load and preprocess image
    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("🖼️  Step 2: Loading and Preprocessing Image");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    let img = image::open(image_path)?;
    let (orig_width, orig_height) = img.dimensions();
    println!("✓ Original image size: {}x{}", orig_width, orig_height);

    // Configure image processor
    let config = ImageProcessConfig {
        source_format: ImageFormat::RGB,
        dest_format: ImageFormat::RGB,
        filter_type: FilterType::Bilinear,
        mean: [127.5, 127.5, 127.5], // MobileNet normalization
        normal: [0.00784, 0.00784, 0.00784],
        wrap: mnn::WrapMode::ClampToEdge,
    };

    let process = ImageProcess::new(&config)?;
    println!("✓ Image processor configured");

    // Convert image to RGB
    let img_rgb = img.to_rgb8();
    let raw_data = img_rgb.as_raw();

    // Get input tensor and resize if needed
    let mut input_tensor = interpreter
        .get_session_input(&session, None)
        .expect("Failed to get input tensor");

    // Calculate affine transform matrix for resizing
    let scale_x = orig_width as f32 / width as f32;
    let scale_y = orig_height as f32 / height as f32;
    let matrix: [f32; 9] = [
        scale_x, 0.0, 0.0,
        0.0, scale_y, 0.0,
        0.0, 0.0, 1.0,
    ];
    process.set_matrix(&matrix);

    // Convert and resize image
    println!("✓ Resizing image to {}x{}...", width, height);
    process.convert(raw_data, orig_width as i32, orig_height as i32, 0, &mut input_tensor);

    // Run inference
    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("⚡ Step 3: Running Inference");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    let start = Instant::now();
    interpreter.run_session(&session)?;
    let duration = start.elapsed();
    println!("✓ Inference completed in {:?}", duration);
    println!("✓ Performance: {:.2} FPS", 1000.0 / duration.as_millis() as f64);

    // Get output
    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("📊 Step 4: Processing Results");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    let output_tensor = interpreter
        .get_session_output(&session, None)
        .expect("Failed to get output tensor");
    let output_shape = output_tensor.shape();
    println!("✓ Output shape: {:?}", output_shape);

    // Copy output to host
    let host_output = Tensor::from_device(&output_tensor, true)
        .expect("Failed to create host output tensor");

    let output_data = unsafe { host_output.data_as_slice::<f32>() };
    println!("✓ Output size: {} values", output_data.len());

    // Find top-K predictions
    let mut results: Vec<(usize, f32)> = output_data
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();

    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("🏆 Top 5 Predictions:");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    for (i, (class_id, score)) in results.iter().take(5).enumerate() {
        println!("  {}. Class {}: {:.4}%", i + 1, class_id, score * 100.0);
    }

    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("✅ Classification completed successfully!");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    Ok(())
}
