//! Raw FFI bindings to MNN C API
//!
//! This module provides unsafe bindings to the C FFI wrapper.
//! Users should prefer the safe wrappers in the `llm` module.

use libc::{c_char, c_int, size_t, c_void};

/// Opaque handle to MNN LLM instance
#[repr(C)]
pub struct MnnLlm {
    _private: [u8; 0],
}

/// Opaque handle to MNN Embedding instance
#[repr(C)]
pub struct MnnEmbedding {
    _private: [u8; 0],
}

/// LLM Context data structure
#[repr(C)]
#[derive(Debug, Clone, Default)]
pub struct MnnLlmContext {
    pub prompt_len: c_int,
    pub gen_seq_len: c_int,
    pub all_seq_len: c_int,
    pub load_us: i64,
    pub vision_us: i64,
    pub audio_us: i64,
    pub prefill_us: i64,
    pub decode_us: i64,
    pub sample_us: i64,
    pub pixels_mp: f32,
    pub audio_input_s: f32,
    pub current_token: c_int,
}

pub type MnnLlmCallback = extern "C" fn(chunk: *const c_char, user_data: *mut c_void);

#[link(name = "mnn_c", kind = "static")]
extern "C" {
    // LLM Functions
    pub fn mnn_llm_create(config_path: *const c_char) -> *mut MnnLlm;
    pub fn mnn_llm_destroy(llm: *mut MnnLlm);
    pub fn mnn_llm_load(llm: *mut MnnLlm) -> bool;
    pub fn mnn_llm_response(
        llm: *mut MnnLlm,
        query: *const c_char,
        stream: bool,
        max_new_tokens: c_int,
    ) -> *mut c_char;

    pub fn mnn_llm_response_with_callback(
        llm: *mut MnnLlm,
        query: *const c_char,
        callback: MnnLlmCallback,
        user_data: *mut libc::c_void,
        max_new_tokens: c_int,
    ) -> *mut c_char;
    pub fn mnn_llm_generate(
        llm: *mut MnnLlm,
        input_ids: *const c_int,
        input_len: c_int,
        output_ids: *mut c_int,
        output_capacity: c_int,
        max_new_tokens: c_int,
    ) -> c_int;
    pub fn mnn_llm_reset(llm: *mut MnnLlm);
    pub fn mnn_llm_set_config(llm: *mut MnnLlm, config: *const c_char) -> bool;
    pub fn mnn_llm_dump_config(llm: *mut MnnLlm) -> *mut c_char;
    pub fn mnn_llm_apply_chat_template(llm: *mut MnnLlm, query: *const c_char) -> *mut c_char;
    pub fn mnn_llm_tokenizer_encode(
        llm: *mut MnnLlm,
        text: *const c_char,
        output_ids: *mut c_int,
        output_capacity: c_int,
    ) -> c_int;
    pub fn mnn_llm_tokenizer_decode(llm: *mut MnnLlm, token_id: c_int) -> *mut c_char;
    pub fn mnn_llm_stopped(llm: *mut MnnLlm) -> bool;
    pub fn mnn_llm_get_current_history(llm: *mut MnnLlm) -> size_t;
    pub fn mnn_llm_erase_history(llm: *mut MnnLlm, begin: size_t, end: size_t);
    pub fn mnn_llm_get_context(llm: *mut MnnLlm, context: *mut MnnLlmContext) -> bool;

    // Embedding Functions
    pub fn mnn_embedding_create(config_path: *const c_char) -> *mut MnnEmbedding;
    pub fn mnn_embedding_destroy(emb: *mut MnnEmbedding);
    pub fn mnn_embedding_dim(emb: *mut MnnEmbedding) -> c_int;
    pub fn mnn_embedding_txt(
        emb: *mut MnnEmbedding,
        text: *const c_char,
        output: *mut f32,
    ) -> bool;

    // Utility Functions
    pub fn mnn_string_free(str: *mut c_char);
    pub fn mnn_get_version() -> *const c_char;
}
