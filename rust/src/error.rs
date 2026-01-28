//! Error types for MNN Rust bindings

use thiserror::Error;

/// Error type for MNN operations
#[derive(Error, Debug)]
pub enum MnnError {
    /// Failed to create LLM instance
    #[error("Failed to create LLM: invalid config path or model")]
    CreateFailed,

    /// Failed to load model
    #[error("Failed to load model")]
    LoadFailed,

    /// Model not loaded
    #[error("Model not loaded, call load() first")]
    NotLoaded,

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Buffer too small
    #[error("Buffer too small: need {needed}, have {capacity}")]
    BufferTooSmall { needed: usize, capacity: usize },

    /// Null pointer error
    #[error("Null pointer returned from MNN")]
    NullPointer,

    /// Invalid UTF-8 in string
    #[error("Invalid UTF-8 string: {0}")]
    InvalidUtf8(#[from] std::str::Utf8Error),

    /// Null byte in string
    #[error("Null byte in string: {0}")]
    NullByte(#[from] std::ffi::NulError),

    /// Initialization error
    #[error("Initialization error: {0}")]
    InitError(String),

    /// Runtime error
    #[error("Runtime error: {0}")]
    RuntimeError(String),
}

/// Result type for MNN operations
pub type Result<T> = std::result::Result<T, MnnError>;
