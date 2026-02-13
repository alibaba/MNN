use image::{ImageBuffer, Rgb, RgbImage};

fn main() {
    // Create a simple 224x224 test image with some colors
    let mut img: RgbImage = ImageBuffer::new(224, 224);
    
    // Create a simple pattern
    for y in 0..224 {
        for x in 0..224 {
            let r = if x < 112 && y < 112 { 255 } else { 0 };
            let g = if x >= 112 && y < 112 { 255 } else { 0 };
            let b = if y >= 112 { 255 } else { 0 };
            img.put_pixel(x, y, Rgb([r, g, b]));
        }
    }
    
    img.save("test_image.jpg").unwrap();
    println!("Test image saved to test_image.jpg");
}
