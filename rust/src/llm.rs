//! Safe Rust wrapper for MNN LLM API
//!
//! This module provides a safe, idiomatic Rust API for the MNN LLM inference engine.

use crate::error::{MnnError, Result};
use crate::ffi;
use libc::c_int;
use std::ffi::{CStr, CString};
use std::ptr::NonNull;

/// Context information for LLM generation
#[derive(Debug, Clone, Default)]
pub struct LlmContext {
    /// Length of the prompt in tokens
    pub prompt_len: i32,
    /// Number of generated tokens
    pub gen_seq_len: i32,
    /// Total sequence length
    pub all_seq_len: i32,
    /// Time spent loading (microseconds)
    pub load_us: i64,
    /// Time spent on vision processing (microseconds)
    pub vision_us: i64,
    /// Time spent on audio processing (microseconds)
    pub audio_us: i64,
    /// Time spent on prefill (microseconds)
    pub prefill_us: i64,
    /// Time spent on decode (microseconds)
    pub decode_us: i64,
    /// Time spent on sampling (microseconds)
    pub sample_us: i64,
    /// Megapixels processed
    pub pixels_mp: f32,
    /// Audio input duration in seconds
    pub audio_input_s: f32,
    /// Current token ID
    pub current_token: i32,
}

impl From<ffi::MnnLlmContext> for LlmContext {
    fn from(ctx: ffi::MnnLlmContext) -> Self {
        Self {
            prompt_len: ctx.prompt_len,
            gen_seq_len: ctx.gen_seq_len,
            all_seq_len: ctx.all_seq_len,
            load_us: ctx.load_us,
            vision_us: ctx.vision_us,
            audio_us: ctx.audio_us,
            prefill_us: ctx.prefill_us,
            decode_us: ctx.decode_us,
            sample_us: ctx.sample_us,
            pixels_mp: ctx.pixels_mp,
            audio_input_s: ctx.audio_input_s,
            current_token: ctx.current_token,
        }
    }
}

/// MNN LLM wrapper for language model inference
///
/// # Example
///
/// ```rust,no_run
/// use mnn::Llm;
///
/// let llm = Llm::create("path/to/config.json").unwrap();
/// llm.load().unwrap();
/// let response = llm.response("Hello!").unwrap();
/// println!("{}", response);
/// ```
pub struct Llm {
    inner: NonNull<ffi::MnnLlm>,
    loaded: bool,
}

// Safety: MNN LLM is thread-safe for operations after loading
unsafe impl Send for Llm {}

impl Llm {
    /// Create a new LLM instance from a config file path
    ///
    /// # Arguments
    /// * `config_path` - Path to the model configuration file (config.json)
    ///
    /// # Returns
    /// A new `Llm` instance, or an error if creation failed
    pub fn create(config_path: &str) -> Result<Self> {
        let c_path = CString::new(config_path)?;
        let ptr = unsafe { ffi::mnn_llm_create(c_path.as_ptr()) };
        
        NonNull::new(ptr)
            .map(|inner| Self { inner, loaded: false })
            .ok_or(MnnError::CreateFailed)
    }

    /// Load the model into memory
    ///
    /// This must be called before any inference operations.
    pub fn load(&mut self) -> Result<()> {
        let success = unsafe { ffi::mnn_llm_load(self.inner.as_ptr()) };
        if success {
            self.loaded = true;
            Ok(())
        } else {
            Err(MnnError::LoadFailed)
        }
    }

    /// Check if the model has been loaded
    pub fn is_loaded(&self) -> bool {
        self.loaded
    }

    /// Generate a response for the given query
    ///
    /// # Arguments
    /// * `query` - The input query string
    ///
    /// # Returns
    /// The generated response string
    pub fn response(&self, query: &str) -> Result<String> {
        self.response_with_options(query, false, -1)
    }

    /// Generate a response with streaming output
    ///
    /// The response will be printed to stdout as it's generated.
    ///
    /// # Arguments
    /// * `query` - The input query string
    pub fn response_stream(&self, query: &str) -> Result<()> {
        self.response_with_options(query, true, -1)?;
        Ok(())
    }

    extern "C" fn trampolin_callback<F>(chunk: *const libc::c_char, user_data: *mut libc::c_void)
    where
        F: FnMut(&str),
    {
        let callback = unsafe { &mut *(user_data as *mut F) };
        let chunk_str = unsafe { CStr::from_ptr(chunk).to_str().unwrap_or("") };
        callback(chunk_str);
    }

    /// Generate a response with streaming output via callback
    /// 
    /// # Arguments
    /// * `query` - The input query string
    /// * `callback` - Closure called with each chunk of text
    pub fn response_stream_callback<F>(&self, query: &str, mut callback: F) -> Result<()> 
    where F: FnMut(&str)
    {
         if !self.loaded {
            return Err(MnnError::NotLoaded);
        }
        
        let c_query = CString::new(query)?;
        
        let user_data = &mut callback as *mut F as *mut libc::c_void;
        
        let ptr = unsafe {
            ffi::mnn_llm_response_with_callback(
                self.inner.as_ptr(),
                c_query.as_ptr(),
                Self::trampolin_callback::<F>,
                user_data,
                -1
            )
        };
        
        if ptr.is_null() {
             return Err(MnnError::NullPointer);
        }
        unsafe { ffi::mnn_string_free(ptr) };
        Ok(())
    }

    /// Generate a response with custom options
    ///
    /// # Arguments
    /// * `query` - The input query string
    /// * `stream` - If true, output will be streamed to stdout
    /// * `max_new_tokens` - Maximum number of tokens to generate (-1 for default)
    pub fn response_with_options(
        &self,
        query: &str,
        stream: bool,
        max_new_tokens: i32,
    ) -> Result<String> {
        if !self.loaded {
            return Err(MnnError::NotLoaded);
        }

        let c_query = CString::new(query)?;
        let ptr = unsafe {
            ffi::mnn_llm_response(
                self.inner.as_ptr(),
                c_query.as_ptr(),
                stream,
                max_new_tokens as c_int,
            )
        };

        if ptr.is_null() {
            return Err(MnnError::NullPointer);
        }

        let result = unsafe { CStr::from_ptr(ptr) };
        let s = result.to_str()?.to_owned();
        unsafe { ffi::mnn_string_free(ptr) };
        Ok(s)
    }

    /// Generate output tokens from input tokens
    ///
    /// # Arguments
    /// * `input_ids` - Array of input token IDs
    ///
    /// # Returns
    /// Vector of output token IDs
    pub fn generate(&self, input_ids: &[i32]) -> Result<Vec<i32>> {
        self.generate_with_options(input_ids, -1)
    }

    /// Generate output tokens with maximum token limit
    ///
    /// # Arguments
    /// * `input_ids` - Array of input token IDs
    /// * `max_new_tokens` - Maximum number of new tokens to generate
    ///
    /// # Returns
    /// Vector of output token IDs
    pub fn generate_with_options(&self, input_ids: &[i32], max_new_tokens: i32) -> Result<Vec<i32>> {
        if !self.loaded {
            return Err(MnnError::NotLoaded);
        }

        // Pre-allocate a reasonable buffer size
        let capacity = input_ids.len() + 4096;
        let mut output_ids: Vec<i32> = vec![0; capacity];

        let result_len = unsafe {
            ffi::mnn_llm_generate(
                self.inner.as_ptr(),
                input_ids.as_ptr(),
                input_ids.len() as c_int,
                output_ids.as_mut_ptr(),
                capacity as c_int,
                max_new_tokens as c_int,
            )
        };

        if result_len < 0 {
            return Err(MnnError::NullPointer);
        }

        if result_len as usize > capacity {
            return Err(MnnError::BufferTooSmall {
                needed: result_len as usize,
                capacity,
            });
        }

        output_ids.truncate(result_len as usize);
        Ok(output_ids)
    }

    /// Reset the conversation history
    pub fn reset(&self) {
        unsafe { ffi::mnn_llm_reset(self.inner.as_ptr()) };
    }

    /// Set configuration from a JSON string
    ///
    /// # Arguments
    /// * `config` - JSON configuration string
    pub fn set_config(&self, config: &str) -> Result<()> {
        let c_config = CString::new(config)?;
        let success = unsafe { ffi::mnn_llm_set_config(self.inner.as_ptr(), c_config.as_ptr()) };
        if success {
            Ok(())
        } else {
            Err(MnnError::InvalidConfig(config.to_string()))
        }
    }

    /// Dump the current configuration as a JSON string
    pub fn dump_config(&self) -> Result<String> {
        let ptr = unsafe { ffi::mnn_llm_dump_config(self.inner.as_ptr()) };
        if ptr.is_null() {
            return Err(MnnError::NullPointer);
        }
        let result = unsafe { CStr::from_ptr(ptr) };
        let s = result.to_str()?.to_owned();
        unsafe { ffi::mnn_string_free(ptr) };
        Ok(s)
    }

    /// Apply chat template to a query
    ///
    /// # Arguments
    /// * `query` - The query string
    ///
    /// # Returns
    /// The templated query string
    pub fn apply_chat_template(&self, query: &str) -> Result<String> {
        let c_query = CString::new(query)?;
        let ptr = unsafe { ffi::mnn_llm_apply_chat_template(self.inner.as_ptr(), c_query.as_ptr()) };
        if ptr.is_null() {
            return Err(MnnError::NullPointer);
        }
        let result = unsafe { CStr::from_ptr(ptr) };
        let s = result.to_str()?.to_owned();
        unsafe { ffi::mnn_string_free(ptr) };
        Ok(s)
    }

    /// Encode text to token IDs using the tokenizer
    ///
    /// # Arguments
    /// * `text` - The text to encode
    ///
    /// # Returns
    /// Vector of token IDs
    pub fn tokenizer_encode(&self, text: &str) -> Result<Vec<i32>> {
        let c_text = CString::new(text)?;
        let capacity = text.len() * 4; // Reasonable estimate
        let mut output_ids: Vec<i32> = vec![0; capacity];

        let result_len = unsafe {
            ffi::mnn_llm_tokenizer_encode(
                self.inner.as_ptr(),
                c_text.as_ptr(),
                output_ids.as_mut_ptr(),
                capacity as c_int,
            )
        };

        if result_len < 0 {
            return Err(MnnError::NullPointer);
        }

        output_ids.truncate(result_len as usize);
        Ok(output_ids)
    }

    /// Decode a token ID to a string
    ///
    /// # Arguments
    /// * `token_id` - The token ID to decode
    ///
    /// # Returns
    /// The decoded string
    pub fn tokenizer_decode(&self, token_id: i32) -> Result<String> {
        let ptr = unsafe { ffi::mnn_llm_tokenizer_decode(self.inner.as_ptr(), token_id as c_int) };
        if ptr.is_null() {
            return Err(MnnError::NullPointer);
        }
        let result = unsafe { CStr::from_ptr(ptr) };
        let s = result.to_str()?.to_owned();
        unsafe { ffi::mnn_string_free(ptr) };
        Ok(s)
    }

    /// Check if generation has stopped
    pub fn stopped(&self) -> bool {
        unsafe { ffi::mnn_llm_stopped(self.inner.as_ptr()) }
    }

    /// Get current history length
    pub fn get_current_history(&self) -> usize {
        unsafe { ffi::mnn_llm_get_current_history(self.inner.as_ptr()) as usize }
    }

    /// Erase history in specified range
    ///
    /// # Arguments
    /// * `begin` - Start index of history to erase
    /// * `end` - End index of history to erase
    pub fn erase_history(&self, begin: usize, end: usize) {
        unsafe { ffi::mnn_llm_erase_history(self.inner.as_ptr(), begin, end) };
    }

    /// Get LLM context data
    pub fn get_context(&self) -> Option<LlmContext> {
        let mut ctx = ffi::MnnLlmContext::default();
        let success = unsafe { ffi::mnn_llm_get_context(self.inner.as_ptr(), &mut ctx) };
        if success {
            Some(ctx.into())
        } else {
            None
        }
    }
}

impl Drop for Llm {
    fn drop(&mut self) {
        unsafe { ffi::mnn_llm_destroy(self.inner.as_ptr()) };
    }
}

/// MNN Embedding wrapper for text embedding
pub struct Embedding {
    inner: NonNull<ffi::MnnEmbedding>,
}

unsafe impl Send for Embedding {}

impl Embedding {
    /// Create a new Embedding instance from a config file path
    ///
    /// # Arguments
    /// * `config_path` - Path to the model configuration file
    pub fn create(config_path: &str) -> Result<Self> {
        let c_path = CString::new(config_path)?;
        let ptr = unsafe { ffi::mnn_embedding_create(c_path.as_ptr()) };
        
        NonNull::new(ptr)
            .map(|inner| Self { inner })
            .ok_or(MnnError::CreateFailed)
    }

    /// Get the embedding dimension
    pub fn dim(&self) -> i32 {
        unsafe { ffi::mnn_embedding_dim(self.inner.as_ptr()) }
    }

    /// Compute text embedding
    ///
    /// # Arguments
    /// * `text` - The text to embed
    ///
    /// # Returns
    /// Vector of embedding values
    pub fn txt_embedding(&self, text: &str) -> Result<Vec<f32>> {
        let c_text = CString::new(text)?;
        let dim = self.dim() as usize;
        let mut output: Vec<f32> = vec![0.0; dim];

        let success = unsafe {
            ffi::mnn_embedding_txt(self.inner.as_ptr(), c_text.as_ptr(), output.as_mut_ptr())
        };

        if success {
            Ok(output)
        } else {
            Err(MnnError::NullPointer)
        }
    }
}

impl Drop for Embedding {
    fn drop(&mut self) {
        unsafe { ffi::mnn_embedding_destroy(self.inner.as_ptr()) };
    }
}
