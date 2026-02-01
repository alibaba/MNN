//! Raw FFI bindings to MNN C API
//!
//! This module provides unsafe bindings to the C FFI wrapper.
//! Users should prefer the safe wrappers in the `llm` module.

use libc::{c_char, c_int, c_void, size_t};

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

/// Opaque handle to MNN Interpreter instance
#[repr(C)]
pub struct MnnInterpreter {
    _private: [u8; 0],
}

/// Opaque handle to MNN Session instance
#[repr(C)]
pub struct MnnSession {
    _private: [u8; 0],
}

/// Opaque handle to MNN Tensor instance
#[repr(C)]
pub struct MnnTensor {
    _private: [u8; 0],
}

/// Opaque handle to MNN ImageProcess instance
#[repr(C)]
pub struct MnnImageProcess {
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

/// Image Process Config
#[repr(C)]
#[derive(Debug, Clone)]
pub struct MnnImageProcessConfig {
    pub mean: [f32; 3],
    pub normal: [f32; 3],
    pub source_format: c_int,
    pub dest_format: c_int,
    pub filter_type: c_int,
    pub wrap: c_int,
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
    pub fn mnn_embedding_txt(emb: *mut MnnEmbedding, text: *const c_char, output: *mut f32)
        -> bool;

    // Interpreter Functions
    pub fn mnn_interpreter_create_from_file(file: *const c_char) -> *mut MnnInterpreter;
    pub fn mnn_interpreter_destroy(interpreter: *mut MnnInterpreter);
    pub fn mnn_interpreter_create_session(
        interpreter: *mut MnnInterpreter,
        num_threads: c_int,
    ) -> *mut MnnSession;
    pub fn mnn_interpreter_get_session_input(
        interpreter: *mut MnnInterpreter,
        session: *mut MnnSession,
        name: *const c_char,
    ) -> *mut MnnTensor;
    pub fn mnn_interpreter_get_session_output(
        interpreter: *mut MnnInterpreter,
        session: *mut MnnSession,
        name: *const c_char,
    ) -> *mut MnnTensor;
    pub fn mnn_interpreter_run_session(
        interpreter: *mut MnnInterpreter,
        session: *mut MnnSession,
    ) -> c_int;
    pub fn mnn_interpreter_resize_session(
        interpreter: *mut MnnInterpreter,
        session: *mut MnnSession,
    );

    // Tensor Functions
    pub fn mnn_tensor_create(
        shape: *const c_int,
        dim_num: c_int,
        type_: c_int,
        data: *mut c_void,
    ) -> *mut MnnTensor;
    pub fn mnn_tensor_create_device(
        shape: *const c_int,
        dim_num: c_int,
        type_: c_int,
    ) -> *mut MnnTensor;
    pub fn mnn_tensor_destroy(tensor: *mut MnnTensor);
    pub fn mnn_tensor_get_shape(
        tensor: *mut MnnTensor,
        shape_ptr: *mut c_int,
        max_dim: c_int,
    ) -> c_int;
    pub fn mnn_tensor_get_data(tensor: *mut MnnTensor) -> *mut c_void;
    pub fn mnn_tensor_get_size(tensor: *mut MnnTensor) -> c_int;
    pub fn mnn_tensor_copy_from_host(tensor: *mut MnnTensor, host_tensor: *const MnnTensor)
        -> bool;
    pub fn mnn_tensor_copy_to_host(tensor: *const MnnTensor, host_tensor: *mut MnnTensor) -> bool;
    pub fn mnn_tensor_create_host_from_device(
        device_tensor: *const MnnTensor,
        copy_data: bool,
    ) -> *mut MnnTensor;

    // ImageProcess Functions
    pub fn mnn_image_process_create(config: *const MnnImageProcessConfig) -> *mut MnnImageProcess;
    pub fn mnn_image_process_destroy(process: *mut MnnImageProcess);
    pub fn mnn_image_process_convert(
        process: *mut MnnImageProcess,
        source: *const u8,
        src_w: c_int,
        src_h: c_int,
        src_stride: c_int,
        dest: *mut MnnTensor,
    );
    pub fn mnn_image_process_set_matrix(process: *mut MnnImageProcess, matrix: *const f32);

    // Utility Functions
    pub fn mnn_string_free(str: *mut c_char);
    pub fn mnn_get_version() -> *const c_char;
}
