// MNN C FFI Header
// This header provides a C-compatible interface to the MNN LLM C++ API
// for use with FFI bindings in languages like Rust.

#ifndef MNN_C_H
#define MNN_C_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Opaque Types
// ============================================================================

// LLM Types
typedef struct MnnLlm MnnLlm;
typedef struct MnnEmbedding MnnEmbedding;

// Inference Types
typedef struct MnnInterpreter MnnInterpreter;
typedef struct MnnSession MnnSession;
typedef struct MnnTensor MnnTensor;
typedef struct MnnImageProcess MnnImageProcess;

// LLM Context data structure for FFI
typedef struct MnnLlmContext {
    int prompt_len;
    int gen_seq_len;
    int all_seq_len;
    int64_t load_us;
    int64_t vision_us;
    int64_t audio_us;
    int64_t prefill_us;
    int64_t decode_us;
    int64_t sample_us;
    float pixels_mp;
    float audio_input_s;
    int current_token;
} MnnLlmContext;

// Image Process Config
typedef struct MnnImageProcessConfig {
    float mean[3];
    float normal[3];
    int source_format; // 0: RGBA, 1: RGB, 2: BGR, 3: GRAY, 4: BGRA, 5: YUV_NV21
    int dest_format;   // 0: RGBA, 1: RGB, 2: BGR, 3: GRAY, 4: BGRA, 5: YUV_NV21, 6: YUV_NV12
    int filter_type;   // 0: NEAREST, 1: BILINEAR, 2: BICUBIC
    int wrap;          // 0: CLAMP_TO_EDGE, 1: ZERO
} MnnImageProcessConfig;

// ============================================================================
// LLM Functions
// ============================================================================

/// Create a new LLM instance from config path
MnnLlm* mnn_llm_create(const char* config_path);

/// Destroy an LLM instance and free its resources
void mnn_llm_destroy(MnnLlm* llm);

/// Load the model into memory
bool mnn_llm_load(MnnLlm* llm);

/// Generate a response for the given query
char* mnn_llm_response(MnnLlm* llm, const char* query, bool stream, int max_new_tokens);

/// Callback function type for streaming output
typedef void (*MnnLlmCallback)(const char* chunk, void* user_data);

/// Generate a response with a callback for streaming output
char* mnn_llm_response_with_callback(MnnLlm* llm, const char* query, MnnLlmCallback callback, void* user_data, int max_new_tokens);

/// Generate output tokens from input tokens
int mnn_llm_generate(MnnLlm* llm, const int* input_ids, int input_len,
                     int* output_ids, int output_capacity, int max_new_tokens);

/// Reset the LLM conversation history
void mnn_llm_reset(MnnLlm* llm);

/// Set LLM configuration from JSON string
bool mnn_llm_set_config(MnnLlm* llm, const char* config);

/// Dump current configuration as JSON string
char* mnn_llm_dump_config(MnnLlm* llm);

/// Apply chat template to a query
char* mnn_llm_apply_chat_template(MnnLlm* llm, const char* query);

/// Encode text to token IDs using tokenizer
int mnn_llm_tokenizer_encode(MnnLlm* llm, const char* text,
                              int* output_ids, int output_capacity);

/// Decode token ID to string
char* mnn_llm_tokenizer_decode(MnnLlm* llm, int token_id);

/// Check if generation has stopped
bool mnn_llm_stopped(MnnLlm* llm);

/// Get current history length
size_t mnn_llm_get_current_history(MnnLlm* llm);

/// Erase history in specified range
void mnn_llm_erase_history(MnnLlm* llm, size_t begin, size_t end);

/// Get LLM context data
bool mnn_llm_get_context(MnnLlm* llm, MnnLlmContext* context);

// ============================================================================
// Embedding Functions
// ============================================================================

/// Create a new Embedding instance from config path
MnnEmbedding* mnn_embedding_create(const char* config_path);

/// Destroy an Embedding instance
void mnn_embedding_destroy(MnnEmbedding* emb);

/// Get embedding dimension
int mnn_embedding_dim(MnnEmbedding* emb);

/// Compute text embedding
bool mnn_embedding_txt(MnnEmbedding* emb, const char* text, float* output);

// ============================================================================
// Interpreter Functions
// ============================================================================

MnnInterpreter* mnn_interpreter_create_from_file(const char* file);
void mnn_interpreter_destroy(MnnInterpreter* interpreter);
MnnSession* mnn_interpreter_create_session(MnnInterpreter* interpreter, int num_threads);
MnnTensor* mnn_interpreter_get_session_input(MnnInterpreter* interpreter, MnnSession* session, const char* name);
MnnTensor* mnn_interpreter_get_session_output(MnnInterpreter* interpreter, MnnSession* session, const char* name);
int mnn_interpreter_run_session(MnnInterpreter* interpreter, MnnSession* session);
void mnn_interpreter_resize_session(MnnInterpreter* interpreter, MnnSession* session);

// ============================================================================
// Tensor Functions
// ============================================================================

MnnTensor* mnn_tensor_create(const int* shape, int dim_num, int type, void* data);
MnnTensor* mnn_tensor_create_device(const int* shape, int dim_num, int type);
void mnn_tensor_destroy(MnnTensor* tensor);
int mnn_tensor_get_shape(MnnTensor* tensor, int* shape_ptr, int max_dim);
void* mnn_tensor_get_data(MnnTensor* tensor);
int mnn_tensor_get_size(MnnTensor* tensor);
bool mnn_tensor_copy_from_host(MnnTensor* tensor, const MnnTensor* host_tensor);
bool mnn_tensor_copy_to_host(const MnnTensor* tensor, MnnTensor* host_tensor);
MnnTensor* mnn_tensor_create_host_from_device(const MnnTensor* device_tensor, bool copy_data);

// ============================================================================
// ImageProcess Functions
// ============================================================================

MnnImageProcess* mnn_image_process_create(const MnnImageProcessConfig* config);
void mnn_image_process_destroy(MnnImageProcess* process);
void mnn_image_process_convert(MnnImageProcess* process, const uint8_t* source, int src_w, int src_h, int src_stride, MnnTensor* dest);

// ============================================================================
// Utility Functions
// ============================================================================

/// Free a string allocated by MNN C API
void mnn_string_free(char* str);

/// Get MNN version string
const char* mnn_get_version(void);

#ifdef __cplusplus
}
#endif

#endif // MNN_C_H
