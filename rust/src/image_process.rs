//! Safe wrapper for MNN ImageProcess

use crate::ffi;
use crate::tensor::Tensor;

/// Image Format
#[repr(i32)]
#[derive(Debug, Clone, Copy)]
pub enum ImageFormat {
    RGBA = 0,
    RGB = 1,
    BGR = 2,
    GRAY = 3,
    BGRA = 4,
    YuvNv21 = 5,
    YuvNv12 = 6,
}

/// Filter Type
#[repr(i32)]
#[derive(Debug, Clone, Copy)]
pub enum FilterType {
    Nearest = 0,
    Bilinear = 1,
    Bicubic = 2,
}

/// Wrap Mode
#[repr(i32)]
#[derive(Debug, Clone, Copy)]
pub enum WrapMode {
    ClampToEdge = 0,
    Zero = 1,
}

/// Image Process Configuration
#[derive(Debug, Clone)]
pub struct ImageProcessConfig {
    pub mean: [f32; 3],
    pub normal: [f32; 3],
    pub source_format: ImageFormat,
    pub dest_format: ImageFormat,
    pub filter_type: FilterType,
    pub wrap: WrapMode,
}

impl Default for ImageProcessConfig {
    fn default() -> Self {
        Self {
            mean: [0.0, 0.0, 0.0],
            normal: [1.0, 1.0, 1.0],
            source_format: ImageFormat::RGBA,
            dest_format: ImageFormat::RGB, // Default tensor format is usually NCHW float, but input to convert is pixels
            filter_type: FilterType::Nearest,
            wrap: WrapMode::ClampToEdge,
        }
    }
}

/// Image Process wrapper
pub struct ImageProcess {
    ptr: *mut ffi::MnnImageProcess,
}

impl ImageProcess {
    /// Create new ImageProcess with config
    pub fn new(config: &ImageProcessConfig) -> crate::Result<Self> {
        let c_config = ffi::MnnImageProcessConfig {
            mean: config.mean,
            normal: config.normal,
            source_format: config.source_format as i32,
            dest_format: config.dest_format as i32,
            filter_type: config.filter_type as i32,
            wrap: config.wrap as i32,
        };

        unsafe {
            let ptr = ffi::mnn_image_process_create(&c_config);
            if ptr.is_null() {
                Err(crate::error::MnnError::InitError("Failed to create ImageProcess".to_string()))
            } else {
                Ok(ImageProcess { ptr })
            }
        }
    }

    /// Convert image data to tensor
    /// source: raw image data
    /// width, height: dimensions of source image
    /// stride: stride of source image (0 for default w * channels)
    /// dest: destination tensor
    pub fn convert(&self, source: &[u8], width: i32, height: i32, stride: i32, dest: &mut Tensor) {
        unsafe {
            ffi::mnn_image_process_convert(
                self.ptr,
                source.as_ptr(),
                width,
                height,
                stride,
                dest.ptr
            );
        }
    }

    /// Set affine transform matrix (3x3)
    pub fn set_matrix(&self, matrix: &[f32; 9]) {
        unsafe {
            ffi::mnn_image_process_set_matrix(self.ptr, matrix.as_ptr());
        }
    }
}

impl Drop for ImageProcess {
    fn drop(&mut self) {
        unsafe { ffi::mnn_image_process_destroy(self.ptr) };
    }
}
