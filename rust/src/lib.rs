//! MNN Rust Bindings
//!
//! This crate provides Rust bindings for the MNN (Mobile Neural Network) LLM API.
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use mnn::Llm;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create LLM from config path
//!     let mut llm = Llm::create("path/to/model/config.json")?;
//!     
//!     // Load model
//!     llm.load()?;
//!     
//!     // Generate response
//!     let response = llm.response("你好")?;
//!     println!("{}", response);
//!     
//!     Ok(())
//! }
//! ```
//!
//! # Features
//!
//! - **Llm**: High-level LLM inference API
//!   - Create from config file
//!   - Generate text responses
//!   - Token generation
//!   - Tokenizer encode/decode
//!
//! - **Embedding**: Text embedding API
//!   - Create from config file
//!   - Generate text embeddings

pub mod error;
pub mod ffi;
pub mod image_process;
pub mod interpreter;
pub mod llm;
pub mod tensor;

// Re-export main types
pub use error::{MnnError, Result};
pub use image_process::{FilterType, ImageFormat, ImageProcess, ImageProcessConfig, WrapMode};
pub use interpreter::{Interpreter, Session};
pub use llm::{Embedding, Llm, LlmContext};
pub use tensor::Tensor;

use std::ffi::CStr;

/// Get MNN version string
pub fn get_version() -> &'static str {
    unsafe {
        let ptr = ffi::mnn_get_version();
        if ptr.is_null() {
            "unknown"
        } else {
            CStr::from_ptr(ptr).to_str().unwrap_or("unknown")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_version() {
        let version = get_version();
        // Version should not be empty
        assert!(!version.is_empty());
        println!("MNN Version: {}", version);
    }

    #[test]
    fn test_error_display() {
        // Test error messages are properly formatted
        let err = MnnError::NotLoaded;
        assert_eq!(format!("{}", err), "Model not loaded, call load() first");

        let err = MnnError::CreateFailed;
        assert_eq!(
            format!("{}", err),
            "Failed to create LLM: invalid config path or model"
        );

        let err = MnnError::BufferTooSmall {
            needed: 100,
            capacity: 50,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("100"));
        assert!(msg.contains("50"));

        let err2 = MnnError::InvalidConfig("test".to_string());
        let msg2 = format!("{}", err2);
        assert!(msg2.contains("test"));
    }

    #[test]
    fn test_tokenization_error() {
        let err = MnnError::TokenizationError { code: -1 };
        let msg = format!("{}", err);
        assert!(msg.contains("-1"));
    }

    #[test]
    fn test_buffer_overflow_error() {
        let err = MnnError::BufferOverflow {
            actual: 5000,
            capacity: 4096,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("5000"));
        assert!(msg.contains("4096"));
    }

    #[test]
    fn test_llm_context_default() {
        let ctx = LlmContext::default();
        assert_eq!(ctx.prompt_len, 0);
        assert_eq!(ctx.gen_seq_len, 0);
        assert_eq!(ctx.all_seq_len, 0);
    }
}
