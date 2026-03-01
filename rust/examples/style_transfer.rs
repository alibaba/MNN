use image::GenericImageView;
use mnn::{FilterType, ImageFormat, ImageProcess, ImageProcessConfig, Interpreter, Tensor};
use std::env;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        println!(
            "Usage: {} <model_path> <image_path> [output_path] [style_name]",
            args[0]
        );
        println!(
            "Example: {} style_transfer.mnn input.jpg output.jpg",
            args[0]
        );
        return Ok(());
    }

    let model_path = &args[1];
    let image_path = &args[2];
    let output_path = if args.len() > 3 {
        &args[3]
    } else {
        "output.jpg"
    };

    // Load model
    println!("Loading model from {}...", model_path);
    let mut interpreter = Interpreter::create_from_file(model_path)?;

    // Config session
    let num_threads = 4;
    let session = interpreter.create_session(num_threads)?;

    // Get input tensor info
    let input_tensor = interpreter
        .get_session_input(&session, None)
        .expect("Failed to get input tensor");
    let input_shape = input_tensor.shape();
    println!("Model Input Shape: {:?}", input_shape);

    let input_height = input_shape[2];
    let input_width = input_shape[3];

    // Load and resize image
    println!("Loading image from {}...", image_path);
    let img = image::open(image_path)?;
    let (orig_width, orig_height) = img.dimensions();
    println!("Original image size: {}x{}", orig_width, orig_height);

    // Calculate new dimensions preserving aspect ratio but fitting in model input if needed?
    // The JS example resizes to fit inputSize (224) with padding.
    // Here we will just resize to the model input size for simplicity, or we can follow the JS logic.
    // The model input shape for style transfer is usually fixed (e.g. 1x3x224x224).

    let config = ImageProcessConfig {
        source_format: ImageFormat::RGBA,
        dest_format: ImageFormat::RGB,
        filter_type: FilterType::Bilinear,
        mean: [0.0, 0.0, 0.0],
        normal: [1.0, 1.0, 1.0],
        wrap: mnn::WrapMode::ClampToEdge,
    };

    let process = ImageProcess::new(&config)?;

    // Convert to RGBA8 for consistency
    let img_rgba = img.to_rgba8();
    let raw_data = img_rgba.as_raw();

    // Resize tensor to match input image (if model supports dynamic shape) or resize image to match model?
    // Style transfer models usually support dynamic shape but MNN might need fixed input or explicit resize.
    // Let's assume we resize the image to model input size for now.

    // Create a temporary tensor for the resized input if we were doing manual resize.
    // But MnnImageProcess can handle resize/crop/convert from source image to dest tensor.
    // We just need to ensure input tensor is sized correctly.
    // If the model input is fixed, we should probably resize the session to match the image or resize the image.
    // The JS example calculates a new size based on inputSize (224) and resizes the image to that, padding if necessary.

    // Let's try to resize session to the image size (if model supports it)
    // Or simpler: Resize input tensor to 224x224 (default) and use ImageProcess to scale image into it.

    let target_width = if input_width > 0 { input_width } else { 224 };
    let target_height = if input_height > 0 { input_height } else { 224 };

    if input_width <= 0 || input_height <= 0 {
        interpreter.resize_session(&session); // Ensure session is ready
    }

    let mut input_tensor = interpreter
        .get_session_input(&session, None)
        .expect("Failed to get input tensor");

    println!("Processing image to {}x{}...", target_width, target_height);

    // Calculate affine transform matrix for resizing (dst -> src)
    let sx = orig_width as f32 / target_width as f32;
    let sy = orig_height as f32 / target_height as f32;
    let matrix: [f32; 9] = [sx, 0.0, 0.0, 0.0, sy, 0.0, 0.0, 0.0, 1.0];
    process.set_matrix(&matrix);

    // Use ImageProcess to convert/resize raw image data to input tensor
    // Note: ImageProcess::convert takes src stride. For RGBA, stride = width * 4.
    process.convert(
        raw_data,
        orig_width as i32,
        orig_height as i32,
        0,
        &mut input_tensor,
    );

    // Run inference
    println!("Running inference...");
    let start = Instant::now();
    interpreter.run_session(&session)?;
    let duration = start.elapsed();
    println!("Inference took: {:?}", duration);

    // Get output
    let output_tensor = interpreter
        .get_session_output(&session, None)
        .expect("Failed to get output tensor");
    let output_shape = output_tensor.shape();
    println!("Output Shape: {:?}", output_shape);

    // Helper to clamp values
    let clamp = |v: f32| -> u8 {
        if v < 0.0 {
            0
        } else if v > 255.0 {
            255
        } else {
            v as u8
        }
    };

    // Post-process: Convert output tensor (NCHW float) to image (HWC uint8)
    // We can manually loop or use ImageProcess if we had a reverse config (but ImageProcess is usually for input).
    // Let's do manual post-process.

    // Wait for data to be ready (copy to host)
    let host_output =
        Tensor::from_device(&output_tensor, true).expect("Failed to create host output tensor");

    let out_h = output_shape[2] as usize;
    let out_w = output_shape[3] as usize;

    let output_data = unsafe { host_output.data_as_slice::<f32>() };

    let mut out_img = image::RgbImage::new(out_w as u32, out_h as u32);

    for y in 0..out_h {
        for x in 0..out_w {
            // NCHW format: [batch, channel, height, width]
            let r = output_data[y * out_w + x];
            let g = output_data[out_h * out_w + y * out_w + x];
            let b = output_data[2 * out_h * out_w + y * out_w + x];

            out_img.put_pixel(
                x as u32,
                y as u32,
                image::Rgb([clamp(r), clamp(g), clamp(b)]),
            );
        }
    }

    println!("Saving result to {}...", output_path);
    out_img.save(output_path)?;
    println!("Done!");

    Ok(())
}
