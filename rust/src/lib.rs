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
pub mod llm;

// Re-export main types
pub use error::{MnnError, Result};
pub use llm::{Embedding, Llm, LlmContext};

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
    }
}
