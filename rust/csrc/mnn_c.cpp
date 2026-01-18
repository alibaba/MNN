// MNN C FFI Implementation
// C wrapper around MNN LLM C++ API

#include "mnn_c.h"
#include "llm/llm.hpp"
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <MNN/ImageProcess.hpp>
#include <MNN/MNNDefine.h>
#include <cstring>
#include <sstream>
#include <iostream>

// Internal struct definitions
struct MnnLlm {
    MNN::Transformer::Llm* llm;
};

struct MnnEmbedding {
    MNN::Transformer::Embedding* emb;
};

struct MnnInterpreter {
    MNN::Interpreter* interp;
};

struct MnnSession {
    MNN::Session* sess;
};

struct MnnTensor {
    MNN::Tensor* tensor;
    bool owned; // Whether this wrapper owns the tensor
};

struct MnnImageProcess {
    MNN::CV::ImageProcess* process;
};

// Helper to duplicate a string for FFI
static char* duplicate_string(const std::string& str) {
    char* result = (char*)malloc(str.length() + 1);
    if (result) {
        strcpy(result, str.c_str());
    }
    return result;
}

// ============================================================================
// LLM Functions
// ============================================================================

MnnLlm* mnn_llm_create(const char* config_path) {
    if (!config_path) {
        return nullptr;
    }
    
    auto* llm = MNN::Transformer::Llm::createLLM(config_path);
    if (!llm) {
        return nullptr;
    }
    
    auto* wrapper = new MnnLlm();
    wrapper->llm = llm;
    return wrapper;
}

void mnn_llm_destroy(MnnLlm* llm) {
    if (llm) {
        if (llm->llm) {
            MNN::Transformer::Llm::destroy(llm->llm);
        }
        delete llm;
    }
}

bool mnn_llm_load(MnnLlm* llm) {
    if (!llm || !llm->llm) {
        return false;
    }
    return llm->llm->load();
}

char* mnn_llm_response(MnnLlm* llm, const char* query, bool stream, int max_new_tokens) {
    if (!llm || !llm->llm || !query) {
        return nullptr;
    }
    
    std::ostringstream output_stream;
    std::ostream* os = stream ? &std::cout : &output_stream;
    
    llm->llm->response(query, os, nullptr, max_new_tokens > 0 ? max_new_tokens : -1);
    
    // Clear dangling pointer in context after use
    llm->llm->generate_init(nullptr);

    if (stream) {
        return duplicate_string("");
    }
    return duplicate_string(output_stream.str());
}

// Custom streambuf to redirect output to callback
class CallbackStreamBuf : public std::streambuf {
public:
    CallbackStreamBuf(MnnLlmCallback callback, void* user_data)
        : callback_(callback), user_data_(user_data) {}

protected:
    std::streamsize xsputn(const char* s, std::streamsize n) override {
        std::string chunk(s, n);
        callback_(chunk.c_str(), user_data_);
        return n;
    }
    
    int_type overflow(int_type c) override {
        if (c != EOF) {
            char ch = static_cast<char>(c);
            char chunk[2] = {ch, '\0'};
            callback_(chunk, user_data_);
        }
        return c;
    }

private:
    MnnLlmCallback callback_;
    void* user_data_;
};

char* mnn_llm_response_with_callback(MnnLlm* llm, const char* query, MnnLlmCallback callback, void* user_data, int max_new_tokens) {
    if (!llm || !llm->llm || !query) {
        return nullptr;
    }
    
    if (!callback) {
        return nullptr;
    }
    
    CallbackStreamBuf buf(callback, user_data);
    std::ostream os(&buf);
    
    // We pass our custom stream to capture output
    llm->llm->response(query, &os, nullptr, max_new_tokens > 0 ? max_new_tokens : -1);
    
    // Clear dangling pointer in context after use
    llm->llm->generate_init(nullptr);

    // For streaming, we don't accumulate the full response in this function
    return duplicate_string("");
}

int mnn_llm_generate(MnnLlm* llm, const int* input_ids, int input_len,
                     int* output_ids, int output_capacity, int max_new_tokens) {
    if (!llm || !llm->llm || !input_ids || input_len <= 0 || !output_ids || output_capacity <= 0) {
        return -1;
    }
    
    std::vector<int> input_vec(input_ids, input_ids + input_len);
    std::vector<int> output_vec = llm->llm->generate(input_vec, max_new_tokens > 0 ? max_new_tokens : -1);
    
    int copy_len = std::min((int)output_vec.size(), output_capacity);
    std::copy(output_vec.begin(), output_vec.begin() + copy_len, output_ids);
    
    return (int)output_vec.size();
}

void mnn_llm_reset(MnnLlm* llm) {
    if (llm && llm->llm) {
        llm->llm->reset();
    }
}

bool mnn_llm_set_config(MnnLlm* llm, const char* config) {
    if (!llm || !llm->llm || !config) {
        return false;
    }
    return llm->llm->set_config(config);
}

char* mnn_llm_dump_config(MnnLlm* llm) {
    if (!llm || !llm->llm) {
        return nullptr;
    }
    return duplicate_string(llm->llm->dump_config());
}

char* mnn_llm_apply_chat_template(MnnLlm* llm, const char* query) {
    if (!llm || !llm->llm || !query) {
        return nullptr;
    }
    return duplicate_string(llm->llm->apply_chat_template(query));
}

int mnn_llm_tokenizer_encode(MnnLlm* llm, const char* text,
                              int* output_ids, int output_capacity) {
    if (!llm || !llm->llm || !text || !output_ids || output_capacity <= 0) {
        return -1;
    }
    
    std::vector<int> tokens = llm->llm->tokenizer_encode(text);
    int copy_len = std::min((int)tokens.size(), output_capacity);
    std::copy(tokens.begin(), tokens.begin() + copy_len, output_ids);
    
    return (int)tokens.size();
}

char* mnn_llm_tokenizer_decode(MnnLlm* llm, int token_id) {
    if (!llm || !llm->llm) {
        return nullptr;
    }
    return duplicate_string(llm->llm->tokenizer_decode(token_id));
}

bool mnn_llm_stopped(MnnLlm* llm) {
    if (!llm || !llm->llm) {
        return true;
    }
    return llm->llm->stoped();
}

size_t mnn_llm_get_current_history(MnnLlm* llm) {
    if (!llm || !llm->llm) {
        return 0;
    }
    return llm->llm->getCurrentHistory();
}

void mnn_llm_erase_history(MnnLlm* llm, size_t begin, size_t end) {
    if (llm && llm->llm) {
        llm->llm->eraseHistory(begin, end);
    }
}

bool mnn_llm_get_context(MnnLlm* llm, MnnLlmContext* context) {
    if (!llm || !llm->llm || !context) {
        return false;
    }
    
    const auto* ctx = llm->llm->getContext();
    if (!ctx) {
        return false;
    }
    
    context->prompt_len = ctx->prompt_len;
    context->gen_seq_len = ctx->gen_seq_len;
    context->all_seq_len = ctx->all_seq_len;
    context->load_us = ctx->load_us;
    context->vision_us = ctx->vision_us;
    context->audio_us = ctx->audio_us;
    context->prefill_us = ctx->prefill_us;
    context->decode_us = ctx->decode_us;
    context->sample_us = ctx->sample_us;
    context->pixels_mp = ctx->pixels_mp;
    context->audio_input_s = ctx->audio_input_s;
    context->current_token = ctx->current_token;
    
    return true;
}

// ============================================================================
// Embedding Functions
// ============================================================================

MnnEmbedding* mnn_embedding_create(const char* config_path) {
    if (!config_path) {
        return nullptr;
    }
    
    auto* emb = MNN::Transformer::Embedding::createEmbedding(config_path, true);
    if (!emb) {
        return nullptr;
    }
    
    auto* wrapper = new MnnEmbedding();
    wrapper->emb = emb;
    return wrapper;
}

void mnn_embedding_destroy(MnnEmbedding* emb) {
    if (emb) {
        if (emb->emb) {
            MNN::Transformer::Llm::destroy(emb->emb);
        }
        delete emb;
    }
}

int mnn_embedding_dim(MnnEmbedding* emb) {
    if (!emb || !emb->emb) {
        return 0;
    }
    return emb->emb->dim();
}

bool mnn_embedding_txt(MnnEmbedding* emb, const char* text, float* output) {
    if (!emb || !emb->emb || !text || !output) {
        return false;
    }
    
    auto result = emb->emb->txt_embedding(text);
    if (!result.get()) {
        return false;
    }
    
    auto info = result->getInfo();
    if (!info || info->size <= 0) {
        return false;
    }
    
    auto ptr = result->readMap<float>();
    if (!ptr) {
        return false;
    }
    
    std::copy(ptr, ptr + info->size, output);
    return true;
}

// ============================================================================
// Interpreter Functions
// ============================================================================

MnnInterpreter* mnn_interpreter_create_from_file(const char* file) {
    auto interp = MNN::Interpreter::createFromFile(file);
    if (!interp) return nullptr;

    MnnInterpreter* wrapper = new MnnInterpreter();
    wrapper->interp = interp;
    return wrapper;
}

void mnn_interpreter_destroy(MnnInterpreter* interpreter) {
    if (interpreter) {
        if (interpreter->interp) {
            MNN::Interpreter::destroy(interpreter->interp);
        }
        delete interpreter;
    }
}

MnnSession* mnn_interpreter_create_session(MnnInterpreter* interpreter, int num_threads) {
    if (!interpreter || !interpreter->interp) return nullptr;

    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_CPU;
    config.numThread = num_threads;

    auto sess = interpreter->interp->createSession(config);
    if (!sess) return nullptr;

    MnnSession* wrapper = new MnnSession();
    wrapper->sess = sess;
    return wrapper;
}

MnnTensor* mnn_interpreter_get_session_input(MnnInterpreter* interpreter, MnnSession* session, const char* name) {
    if (!interpreter || !interpreter->interp || !session) return nullptr;

    auto tensor = interpreter->interp->getSessionInput(session->sess, name);
    if (!tensor) return nullptr;

    MnnTensor* wrapper = new MnnTensor();
    wrapper->tensor = tensor;
    wrapper->owned = false; // Owned by Session
    return wrapper;
}

MnnTensor* mnn_interpreter_get_session_output(MnnInterpreter* interpreter, MnnSession* session, const char* name) {
    if (!interpreter || !interpreter->interp || !session) return nullptr;

    auto tensor = interpreter->interp->getSessionOutput(session->sess, name);
    if (!tensor) return nullptr;

    MnnTensor* wrapper = new MnnTensor();
    wrapper->tensor = tensor;
    wrapper->owned = false; // Owned by Session
    return wrapper;
}

int mnn_interpreter_run_session(MnnInterpreter* interpreter, MnnSession* session) {
    if (!interpreter || !interpreter->interp || !session) return -1;
    return interpreter->interp->runSession(session->sess);
}

void mnn_interpreter_resize_session(MnnInterpreter* interpreter, MnnSession* session) {
    if (interpreter && interpreter->interp && session) {
        interpreter->interp->resizeSession(session->sess);
    }
}

// ============================================================================
// Tensor Functions
// ============================================================================

MnnTensor* mnn_tensor_create(const int* shape, int dim_num, int type, void* data) {
    std::vector<int> dims;
    if (shape && dim_num > 0) {
        dims.assign(shape, shape + dim_num);
    }

    // Convert int type to halide_type_t (simplistic mapping, need robust solution)
    // 0: float, 1: int, 2: uint8 (approx)
    halide_type_t htype = halide_type_of<float>();
    if (type == 1) htype = halide_type_of<int>();
    if (type == 2) htype = halide_type_of<uint8_t>();

    MNN::Tensor* t = MNN::Tensor::create(dims, htype, data, MNN::Tensor::CAFFE);
    if (!t) return nullptr;

    MnnTensor* wrapper = new MnnTensor();
    wrapper->tensor = t;
    wrapper->owned = true;
    return wrapper;
}

MnnTensor* mnn_tensor_create_device(const int* shape, int dim_num, int type) {
    std::vector<int> dims;
    if (shape && dim_num > 0) {
        dims.assign(shape, shape + dim_num);
    }

    halide_type_t htype = halide_type_of<float>();
    if (type == 1) htype = halide_type_of<int>();
    if (type == 2) htype = halide_type_of<uint8_t>();

    MNN::Tensor* t = MNN::Tensor::createDevice(dims, htype, MNN::Tensor::CAFFE);
    if (!t) return nullptr;

    MnnTensor* wrapper = new MnnTensor();
    wrapper->tensor = t;
    wrapper->owned = true;
    return wrapper;
}

void mnn_tensor_destroy(MnnTensor* tensor) {
    if (tensor) {
        if (tensor->owned && tensor->tensor) {
            MNN::Tensor::destroy(tensor->tensor);
        }
        delete tensor;
    }
}

int mnn_tensor_get_shape(MnnTensor* tensor, int* shape_ptr, int max_dim) {
    if (!tensor || !tensor->tensor) return 0;
    auto shape = tensor->tensor->shape();
    int dims = std::min((int)shape.size(), max_dim);
    if (shape_ptr) {
        for (int i = 0; i < dims; ++i) {
            shape_ptr[i] = shape[i];
        }
    }
    return (int)shape.size();
}

void* mnn_tensor_get_data(MnnTensor* tensor) {
    if (!tensor || !tensor->tensor) return nullptr;
    return tensor->tensor->host<void>();
}

int mnn_tensor_get_size(MnnTensor* tensor) {
    if (!tensor || !tensor->tensor) return 0;
    return tensor->tensor->size();
}

bool mnn_tensor_copy_from_host(MnnTensor* tensor, const MnnTensor* host_tensor) {
    if (!tensor || !tensor->tensor || !host_tensor || !host_tensor->tensor) return false;
    return tensor->tensor->copyFromHostTensor(host_tensor->tensor);
}

bool mnn_tensor_copy_to_host(const MnnTensor* tensor, MnnTensor* host_tensor) {
    if (!tensor || !tensor->tensor || !host_tensor || !host_tensor->tensor) return false;
    return tensor->tensor->copyToHostTensor(host_tensor->tensor);
}

MnnTensor* mnn_tensor_create_host_from_device(const MnnTensor* device_tensor, bool copy_data) {
    if (!device_tensor || !device_tensor->tensor) return nullptr;

    auto t = MNN::Tensor::createHostTensorFromDevice(device_tensor->tensor, copy_data);
    if (!t) return nullptr;

    MnnTensor* wrapper = new MnnTensor();
    wrapper->tensor = t;
    wrapper->owned = true;
    return wrapper;
}

// ============================================================================
// ImageProcess Functions
// ============================================================================

MnnImageProcess* mnn_image_process_create(const MnnImageProcessConfig* config) {
    if (!config) return nullptr;

    MNN::CV::ImageProcess::Config cv_config;
    std::memcpy(cv_config.mean, config->mean, 3 * sizeof(float));
    std::memcpy(cv_config.normal, config->normal, 3 * sizeof(float));
    cv_config.sourceFormat = (MNN::CV::ImageFormat)config->source_format;
    cv_config.destFormat = (MNN::CV::ImageFormat)config->dest_format;
    cv_config.filterType = (MNN::CV::Filter)config->filter_type;
    cv_config.wrap = (MNN::CV::Wrap)config->wrap;

    auto process = MNN::CV::ImageProcess::create(cv_config);
    if (!process) return nullptr;

    MnnImageProcess* wrapper = new MnnImageProcess();
    wrapper->process = process;
    return wrapper;
}

void mnn_image_process_destroy(MnnImageProcess* process) {
    if (process) {
        if (process->process) {
            delete process->process;
        }
        delete process;
    }
}

void mnn_image_process_convert(MnnImageProcess* process, const uint8_t* source, int src_w, int src_h, int src_stride, MnnTensor* dest) {
    if (!process || !process->process || !source || !dest || !dest->tensor) return;
    process->process->convert(source, src_w, src_h, src_stride, dest->tensor);
}

// ============================================================================
// Utility Functions
// ============================================================================

void mnn_string_free(char* str) {
    free(str);
}

const char* mnn_get_version(void) {
    return MNN_VERSION;
}
