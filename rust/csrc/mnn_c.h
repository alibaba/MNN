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

// Opaque handle to MNN LLM instance
typedef struct MnnLlm MnnLlm;

// Opaque handle to MNN Embedding instance
typedef struct MnnEmbedding MnnEmbedding;

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

// ============================================================================
// LLM Functions
// ============================================================================

/// Create a new LLM instance from config path
/// @param config_path Path to the model configuration file (config.json)
/// @return Pointer to the LLM instance, or NULL on failure
MnnLlm* mnn_llm_create(const char* config_path);

/// Destroy an LLM instance and free its resources
/// @param llm Pointer to the LLM instance
void mnn_llm_destroy(MnnLlm* llm);

/// Load the model into memory
/// @param llm Pointer to the LLM instance
/// @return true on success, false on failure
bool mnn_llm_load(MnnLlm* llm);

/// Generate a response for the given query
/// @param llm Pointer to the LLM instance
/// @param query The input query string
/// @param stream If true, output will be streamed to stdout
/// @param max_new_tokens Maximum number of tokens to generate (-1 for default)
/// @return Pointer to the response string (caller must free with mnn_string_free)
char* mnn_llm_response(MnnLlm* llm, const char* query, bool stream, int max_new_tokens);

/// Callback function type for streaming output
/// @param chunk The text chunk generated
/// @param user_data User data passed to the callback
typedef void (*MnnLlmCallback)(const char* chunk, void* user_data);

/// Generate a response with a callback for streaming output
/// @param llm Pointer to the LLM instance
/// @param query The input query string
/// @param callback The callback function to be called with each chunk of output
/// @param user_data User data to be passed to the callback
/// @param max_new_tokens Maximum number of tokens to generate (-1 for default)
/// @return Pointer to the response string (caller must free with mnn_string_free)
char* mnn_llm_response_with_callback(MnnLlm* llm, const char* query, MnnLlmCallback callback, void* user_data, int max_new_tokens);

/// Generate output tokens from input tokens
/// @param llm Pointer to the LLM instance
/// @param input_ids Array of input token IDs
/// @param input_len Length of input_ids array
/// @param output_ids Pointer to array for output token IDs (caller allocates)
/// @param output_capacity Capacity of output_ids array
/// @param max_new_tokens Maximum number of new tokens to generate
/// @return Number of tokens generated, or -1 on error
int mnn_llm_generate(MnnLlm* llm, const int* input_ids, int input_len,
                     int* output_ids, int output_capacity, int max_new_tokens);

/// Reset the LLM conversation history
/// @param llm Pointer to the LLM instance
void mnn_llm_reset(MnnLlm* llm);

/// Set LLM configuration from JSON string
/// @param llm Pointer to the LLM instance
/// @param config JSON configuration string
/// @return true on success, false on failure
bool mnn_llm_set_config(MnnLlm* llm, const char* config);

/// Dump current configuration as JSON string
/// @param llm Pointer to the LLM instance
/// @return Pointer to JSON config string (caller must free with mnn_string_free)
char* mnn_llm_dump_config(MnnLlm* llm);

/// Apply chat template to a query
/// @param llm Pointer to the LLM instance
/// @param query The query string
/// @return Pointer to templated string (caller must free with mnn_string_free)
char* mnn_llm_apply_chat_template(MnnLlm* llm, const char* query);

/// Encode text to token IDs using tokenizer
/// @param llm Pointer to the LLM instance
/// @param text The text to encode
/// @param output_ids Pointer to array for output token IDs (caller allocates)
/// @param output_capacity Capacity of output_ids array
/// @return Number of tokens encoded, or -1 on error
int mnn_llm_tokenizer_encode(MnnLlm* llm, const char* text,
                              int* output_ids, int output_capacity);

/// Decode token ID to string
/// @param llm Pointer to the LLM instance
/// @param token_id The token ID to decode
/// @return Pointer to decoded string (caller must free with mnn_string_free)
char* mnn_llm_tokenizer_decode(MnnLlm* llm, int token_id);

/// Check if generation has stopped
/// @param llm Pointer to the LLM instance
/// @return true if stopped, false otherwise
bool mnn_llm_stopped(MnnLlm* llm);

/// Get current history length
/// @param llm Pointer to the LLM instance
/// @return Current history length
size_t mnn_llm_get_current_history(MnnLlm* llm);

/// Erase history in specified range
/// @param llm Pointer to the LLM instance
/// @param begin Start index of history to erase
/// @param end End index of history to erase
void mnn_llm_erase_history(MnnLlm* llm, size_t begin, size_t end);

/// Get LLM context data
/// @param llm Pointer to the LLM instance
/// @param context Pointer to context struct to fill
/// @return true on success, false if context unavailable
bool mnn_llm_get_context(MnnLlm* llm, MnnLlmContext* context);

// ============================================================================
// Embedding Functions
// ============================================================================

/// Create a new Embedding instance from config path
/// @param config_path Path to the model configuration file
/// @return Pointer to the Embedding instance, or NULL on failure
MnnEmbedding* mnn_embedding_create(const char* config_path);

/// Destroy an Embedding instance
/// @param emb Pointer to the Embedding instance
void mnn_embedding_destroy(MnnEmbedding* emb);

/// Get embedding dimension
/// @param emb Pointer to the Embedding instance
/// @return Embedding dimension
int mnn_embedding_dim(MnnEmbedding* emb);

/// Compute text embedding
/// @param emb Pointer to the Embedding instance
/// @param text The text to embed
/// @param output Pointer to output buffer (caller allocates, size = dim)
/// @return true on success, false on failure
bool mnn_embedding_txt(MnnEmbedding* emb, const char* text, float* output);

// ============================================================================
// Utility Functions
// ============================================================================

/// Free a string allocated by MNN C API
/// @param str Pointer to the string to free
void mnn_string_free(char* str);

/// Get MNN version string
/// @return Version string (do not free)
const char* mnn_get_version(void);

#ifdef __cplusplus
}
#endif

#endif // MNN_C_H
