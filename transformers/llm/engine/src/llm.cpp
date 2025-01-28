//
//  llm.cpp
//
//  Created by MNN on 2023/08/25.
//  ZhaodeWang
//
// #define MNN_OPEN_TIME_TRACE 1

#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <unordered_set>

#include <MNN/AutoTime.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include "cpp/ExprDebug.hpp"
#include "llm/llm.hpp"
#include "llmconfig.hpp"
#include "tokenizer.hpp"
// 0: no debug, 1: test op time, 2: print tensor info, 3: print tensor in output
#define DEBUG_MODE 0
//#define DEBUG_IMAGE

#include "httplib.h"
#ifdef LLM_SUPPORT_VISION
#include <cv/cv.hpp>
#endif
#ifdef LLM_SUPPORT_AUDIO
#include <audio/audio.hpp>
#endif

using namespace MNN::Express;
namespace MNN {
namespace Transformer {
struct KVMeta {
    size_t block = 4096;
    size_t previous = 0;
    size_t remove = 0;
    int* reserve = nullptr;
    int n_reserve = 0;
    size_t add = 0;
    std::vector<int> reserveHost;
    void sync() {
        int revertNumber = 0;
        for (int i=0; i<n_reserve; ++i) {
            revertNumber += reserve[2*i+1];
        }
        previous = previous - remove + add + revertNumber;
        n_reserve = 0;
        reserve = nullptr;
        remove = 0;
        add = 0;
    }
};
typedef void (*DequantFunction)(const uint8_t*, float*, float, float, int);

static void q41_dequant_ref(const uint8_t* src, float* dst, float scale, float zero, int size) {
    for (int i = 0; i < size / 2; i++) {
        int x          = src[i];
        int x1         = x / 16 - 8;
        int x2         = x % 16 - 8;
        float w1       = x1 * scale + zero;
        float w2       = x2 * scale + zero;
        dst[2 * i]     = w1;
        dst[2 * i + 1] = w2;
    }
}

static void q81_dequant_ref(const uint8_t* src, float* dst, float scale, float zero, int size) {
    for (int i = 0; i < size; i++) {
        dst[i] = (src[i] - 128) * scale + zero;
    }
}

class DiskEmbedding {
public:
    explicit DiskEmbedding(const std::shared_ptr<LlmConfig>& config);
    ~DiskEmbedding() {
    }
    void embedding(const std::vector<int>& input_ids, float* ptr);

private:
    void seek_read(uint8_t* dst, size_t size, size_t offset);
    std::unique_ptr<uint8_t[]> alpha_  = nullptr;
    std::unique_ptr<uint8_t[]> weight_ = nullptr;
    std::unique_ptr<FILE, decltype(&fclose)> fp_;
    DequantFunction dequant_;
    int hidden_size_, weight_token_size_;
    int64_t w_offset_, block_num_, quant_block_, quant_bit_;
};

void DiskEmbedding::seek_read(uint8_t* dst, size_t size, size_t offset) {
    fseek(fp_.get(), offset, SEEK_SET);
    size_t bytes_read = fread(dst, 1, size, fp_.get());
    (void)bytes_read;
}

DiskEmbedding::DiskEmbedding(const std::shared_ptr<LlmConfig>& config) : fp_(nullptr, &fclose) {
    auto tie_embeddings = config->tie_embeddings();
    hidden_size_        = config->hidden_size();
    if (tie_embeddings.size() == 5) {
        w_offset_          = tie_embeddings[0];
        quant_bit_         = tie_embeddings[3];
        quant_block_       = tie_embeddings[4];
        block_num_         = hidden_size_ / quant_block_;
        weight_token_size_ = hidden_size_ * quant_bit_ / 8;
        fp_.reset(fopen(config->llm_weight().c_str(), "rb"));
        // TODO: optimize dequant function
        dequant_        = quant_bit_ == 8 ? q81_dequant_ref : q41_dequant_ref;
        auto a_offset   = tie_embeddings[1];
        auto alpha_size = tie_embeddings[2];
        alpha_.reset(new uint8_t[alpha_size]);
        seek_read(alpha_.get(), alpha_size, a_offset);
    } else {
        weight_token_size_ = hidden_size_ * sizeof(int16_t);
        fp_.reset(fopen(config->embedding_file().c_str(), "rb"));
    }
    weight_.reset(new uint8_t[weight_token_size_]);
}

void DiskEmbedding::embedding(const std::vector<int>& input_ids, float* dst) {
    if (alpha_.get()) {
        // quant
        for (size_t i = 0; i < input_ids.size(); i++) {
            int token = input_ids[i];
            seek_read(weight_.get(), weight_token_size_, w_offset_ + token * weight_token_size_);
            auto dptr      = dst + i * hidden_size_;
            auto alpha_ptr = reinterpret_cast<float*>(alpha_.get()) + token * block_num_ * 2;
            for (int n = 0; n < block_num_; n++) {
                auto dst_ptr     = dptr + n * quant_block_;
                uint8_t* src_ptr = weight_.get() + n * (quant_block_ * quant_bit_ / 8);
                float zero       = (alpha_ptr + n * 2)[0];
                float scale      = (alpha_ptr + n * 2)[1];
                dequant_(src_ptr, dst_ptr, scale, zero, quant_block_);
            }
        }
    } else {
        // bf16
        for (size_t i = 0; i < input_ids.size(); i++) {
            seek_read(weight_.get(), weight_token_size_, input_ids[i] * weight_token_size_);
            int16_t* dst_ptr = reinterpret_cast<int16_t*>(dst + i * hidden_size_);
            for (int j = 0; j < hidden_size_; j++) {
                dst_ptr[j * 2]     = 0;
                dst_ptr[j * 2 + 1] = reinterpret_cast<int16_t*>(weight_.get())[j];
            }
        }
    }
}

class Mllm : public Llm {
public:
    Mllm(std::shared_ptr<LlmConfig> config) : Llm(config) {
        if (config->is_visual()) {
            image_height_  = config->llm_config_.value("image_size", image_height_);
            image_width_   = image_height_;
            img_pad_       = config->llm_config_.value("image_pad", img_pad_);
            vision_start_  = config->llm_config_.value("vision_start", vision_start_);
            vision_end_    = config->llm_config_.value("vision_end", vision_end_);
            image_mean_    = config->llm_config_.value("image_mean", image_mean_);
            image_norm_    = config->llm_config_.value("image_norm", image_norm_);
        }
        if (config->is_audio()) {
        }
    }
    ~Mllm() {
        mul_module_.reset();
    }
    virtual void load() override;
    virtual std::vector<int> tokenizer_encode(const std::string& query, bool use_template = true) override;
    virtual MNN::Express::VARP embedding(const std::vector<int>& input_ids) override;

private:
    // vision config
    int image_height_ = 448, image_width_ = 448, vision_start_ = 151857, vision_end_ = 151858, img_pad_ = 151859;
    std::vector<float> image_mean_{122.7709383, 116.7460125, 104.09373615};
    std::vector<float> image_norm_{0.01459843, 0.01500777, 0.01422007};
    // audio config
    int audio_pad_ = 151646;
    std::vector<int> multimode_process(const std::string& mode, std::string info);
    std::vector<int> vision_process(const std::string& file);
    std::vector<int> audio_process(const std::string& file);
    std::shared_ptr<Module> mul_module_;
    std::vector<VARP> mul_embeddings_;
};

// Llm start
Llm* Llm::createLLM(const std::string& config_path) {
    std::shared_ptr<LlmConfig> config(new LlmConfig(config_path));
    Llm* llm = nullptr;
    if (config->is_visual() || config->is_audio()) {
        llm = new Mllm(config);
    } else {
        llm = new Llm(config);
    }
    return llm;
}

static MNNForwardType backend_type_convert(const std::string& type_str) {
    if (type_str == "cpu")
        return MNN_FORWARD_CPU;
    if (type_str == "metal")
        return MNN_FORWARD_METAL;
    if (type_str == "cuda")
        return MNN_FORWARD_CUDA;
    if (type_str == "opencl")
        return MNN_FORWARD_OPENCL;
    if (type_str == "opengl")
        return MNN_FORWARD_OPENGL;
    if (type_str == "vulkan")
        return MNN_FORWARD_VULKAN;
    if (type_str == "npu")
        return MNN_FORWARD_NN;
    return MNN_FORWARD_AUTO;
}

std::string Llm::dump_config() {
    return config_->config_.dump();
}

bool Llm::set_config(const std::string& content) {
    return config_->config_.merge(content.c_str());
}

int file_size_m(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Could not open the file!" << std::endl;
        return -1;
    }
    long long fileSize = file.tellg();
    file.close();
    return fileSize / (1024 * 1024);
}

void Llm::init_runtime() {
    ScheduleConfig config;
    BackendConfig cpuBackendConfig;
    config.type      = backend_type_convert(config_->backend_type());
    config.numThread = config_->thread_num();
    ExecutorScope::Current()->setGlobalExecutorConfig(config.type, cpuBackendConfig, config.numThread);
    if (config_->power() == "high") {
        cpuBackendConfig.power = BackendConfig::Power_High;
    } else if (config_->power() == "low") {
        cpuBackendConfig.power = BackendConfig::Power_Low;
    }
    if (config_->memory() == "high") {
        cpuBackendConfig.memory = BackendConfig::Memory_High;
    } else if (config_->memory() == "low") {
        cpuBackendConfig.memory = BackendConfig::Memory_Low;
    }
    if (config_->precision() == "high") {
        cpuBackendConfig.precision = BackendConfig::Precision_High;
    } else if (config_->precision() == "low") {
        cpuBackendConfig.precision = BackendConfig::Precision_Low;
    }
    config.backendConfig = &cpuBackendConfig;

    runtime_manager_.reset(Executor::RuntimeManager::createRuntimeManager(config));
    runtime_manager_->setHint(MNN::Interpreter::MEM_ALLOCATOR_TYPE, 0);
    runtime_manager_->setHint(MNN::Interpreter::QKV_QUANT_OPTIONS, config_->quant_qkv());
    runtime_manager_->setHint(MNN::Interpreter::KVCACHE_SIZE_LIMIT, config_->kvcache_limit());
    if (config_->use_cached_mmap()) {
        runtime_manager_->setHint(MNN::Interpreter::USE_CACHED_MMAP, 1);
    }
    std::string tmpPath = config_->tmp_path();
    if (config_->kvcache_mmap()) {
        runtime_manager_->setExternalPath(tmpPath, MNN::Interpreter::EXTERNAL_PATH_KVCACHE_DIR);
    }
    if (config_->use_mmap()) {
        runtime_manager_->setExternalPath(tmpPath, MNN::Interpreter::EXTERNAL_WEIGHT_DIR);
    }
    runtime_manager_->setHintPtr(Interpreter::KVCACHE_INFO, mMeta.get());

#if DEBUG_MODE == 1
    runtime_manager_->setMode(MNN::Interpreter::Session_Debug);
    _initTimeTrace();
#endif
#if DEBUG_MODE == 2
    runtime_manager_->setMode(MNN::Interpreter::Session_Debug);
    _initTensorStatic();
#endif
#if DEBUG_MODE == 3
    runtime_manager_->setMode(MNN::Interpreter::Session_Debug);
    _initDebug();
#endif
    {
        std::string cacheFilePath = tmpPath.length() != 0 ? tmpPath : ".";
        runtime_manager_->setCache(cacheFilePath + "/mnn_cachefile.bin");
    }
}

void Llm::load() {
    init_runtime();
    // init module status
    // 1. load vocab
    MNN_PRINT("load tokenizer\n");
    tokenizer_.reset(Tokenizer::createTokenizer(config_->tokenizer_file()));
    MNN_PRINT("load tokenizer Done\n");
    disk_embedding_.reset(new DiskEmbedding(config_));
    // 3. load model
    Module::Config module_config;
    if (config_->backend_type() == "opencl" || config_->backend_type() == "vulkan") {
        module_config.shapeMutable = false;
    } else {
        module_config.shapeMutable = true;
    }
    module_config.rearrange    = true;
    // using base module for lora module
    if (base_module_ != nullptr) {
        module_config.base = base_module_;
    }
    int layer_nums = config_->layer_nums();
    // load single model
    modules_.resize(1);
    std::string model_path = config_->llm_model();
    MNN_PRINT("load %s ... ", model_path.c_str());
    runtime_manager_->setExternalFile(config_->llm_weight());
    modules_[0].reset(Module::load(
                                       {"input_ids", "attention_mask", "position_ids"},
                                       {"logits"}, model_path.c_str(), runtime_manager_, &module_config));
    MNN_PRINT("Load Module Done!\n");
    decode_modules_.resize(modules_.size());
    for (int v = 0; v < modules_.size(); ++v) {
        decode_modules_[v].reset(Module::clone(modules_[v].get()));
    }
    MNN_PRINT("Clone Decode Module Done!\n");

    prefill_modules_ = modules_;
}

size_t Llm::apply_lora(const std::string& lora_path) {
    std::string model_path = config_->base_dir_ + "/" + lora_path;
    Module::Config module_config;
    module_config.shapeMutable = true;
    module_config.rearrange    = true;
    module_config.base         = modules_.begin()->get();
    size_t lora_index          = modules_.size();
    runtime_manager_->setHint(MNN::Interpreter::USE_CACHED_MMAP, 0);
    modules_.emplace_back(Module::load({"input_ids", "attention_mask", "position_ids"}, {"logits"},
                                        model_path.c_str(), runtime_manager_, &module_config));
    select_module(lora_index);
    return lora_index;
}

Llm* Llm::create_lora(const std::string& lora_path) {
    auto llm = new Llm(std::make_shared<LlmConfig>(*config_));
    llm->set_config("{\"llm_model\": \"" + lora_path + "\", \"use_mmap\": false, \"use_cached_mmap\": false}");
    llm->base_module_ = modules_.begin()->get();
    llm->load();
    return llm;
}

bool Llm::release_module(size_t index) {
    if (index >= modules_.size()) {
        return false;
    }
    if (prefill_modules_[0] == modules_[index]) {
        select_module(0);
    }
    modules_[index].reset();
    return true;
}

bool Llm::select_module(size_t index) {
    if (index >= modules_.size()) {
        return false;
    }
    if (modules_[index] == nullptr) {
        return false;
    }
    if (decode_modules_.empty()) {
        decode_modules_.resize(modules_.size());
        prefill_modules_.resize(modules_.size());
    }
    decode_modules_[0].reset(Module::clone(modules_[index].get()));
    prefill_modules_[0] = modules_[index];
    return true;
}

void Llm::trace(bool start) {
    auto status = MNN::Interpreter::Session_Resize_Check;
    if (start) {
        status = MNN::Interpreter::Session_Resize_Check;
    } else {
        status = MNN::Interpreter::Session_Resize_Fix;
    }
    for (auto& m : decode_modules_) {
        m->traceOrOptimize(status);
    }

    runtime_manager_->updateCache();
    mTracing = start;
}

void Llm::tuning(TuneType type, std::vector<int> candidates) {
    if (type != OP_ENCODER_NUMBER) {
        MNN_ERROR("tuning type not supported\n");
        return;
    }
    if (config_->backend_type() != "metal") {
        return;
    }

    current_modules_     = decode_modules_;
    int64_t min_time     = INT64_MAX;
    int prefer_candidate = 10;
    for (auto& candidate : candidates) {
        runtime_manager_->setHint(MNN::Interpreter::OP_ENCODER_NUMBER_FOR_COMMIT, candidate);
        mMeta->add = 1;
        auto st     = std::chrono::system_clock::now();
        auto logits = forward({0});
        if (nullptr == logits.get()) {
            return;
        }
        if (logits->getInfo()->size == 0) {
            return;
        }
        auto token   = sample(logits, {});
        auto et      = std::chrono::system_clock::now();
        int64_t time = std::chrono::duration_cast<std::chrono::microseconds>(et - st).count();
        if (time < min_time) {
            prefer_candidate = candidate;
            min_time         = time;
            // MNN_PRINT("op encode number:%d, decode time: %lld us\n", candidate, time);
        }
    }
    runtime_manager_->setHint(MNN::Interpreter::OP_ENCODER_NUMBER_FOR_COMMIT, prefer_candidate);
    // clear dirty tuning kv history
    setKVCacheInfo(0, getCurrentHistory());
    reset();
}
void Llm::switchMode(Llm::Stage stage) {
    switch (stage) {
        case Prefill:
            current_modules_ = prefill_modules_;
            break;
        case Decode:
            current_modules_ = decode_modules_;
            break;
        default:
            break;
    }
}

void Llm::setKVCacheInfo(size_t add, size_t remove, int* reserve, int n_reserve) {
    if (remove > mMeta->previous) {
        remove = mMeta->previous;
    }
    mMeta->remove = remove;
    mMeta->reserve = reserve;
    mMeta->n_reserve = n_reserve;
    mMeta->add = add;
}

MNN::Express::VARP Llm::forwardRaw(MNN::Express::VARP hiddenState, MNN::Express::VARP mask, MNN::Express::VARP inputPos) {
    VARP logits;
    std::vector<MNN::Express::VARP> outputs;
    outputs = current_modules_.back()->onForward({hiddenState, mask, inputPos});
    if (outputs.empty()) {
        return nullptr;
    }
    logits = outputs[0];
    mMeta->sync();
    return logits;
}

VARP Llm::forward(const std::vector<int>& input_ids) {
    int seq_len         = input_ids.size();
    auto attention_mask = gen_attention_mask(seq_len);
    auto position_ids = gen_position_ids(seq_len);
    auto hidden_states = embedding(input_ids);
    auto logits = forwardRaw(hidden_states, attention_mask, position_ids);
    mState.all_seq_len_ += seq_len;
    mState.gen_seq_len_++;
    return logits;
}

int Llm::sample(VARP logits, const std::vector<int>& pre_ids, int offset, int size) {
    std::unordered_set<int> ids_set(pre_ids.begin(), pre_ids.end());
    auto scores = (float*)(logits->readMap<float>()) + offset;
    if (0 == size) {
        size = logits->getInfo()->size;
    }
    // repetition penalty
    const float repetition_penalty = 1.1;
    for (auto id : ids_set) {
        float score = scores[id];
        scores[id]  = score < 0 ? score * repetition_penalty : score / repetition_penalty;
    }
    // argmax
    float max_score = scores[0];
    int token_id = 0;
    for (int i = 1; i < size; i++) {
        float score = scores[i];
        if (score > max_score) {
            max_score = score;
            token_id  = i;
        }
    }
    mState.output_ids_.push_back(token_id);
    return token_id;
}

static std::string apply_template(std::string prompt_template, const std::string& content,
                                  const std::string& role = "") {
    if (prompt_template.empty())
        return content;
    if (!role.empty()) {
        const std::string placeholder = "%r";
        size_t start_pos              = prompt_template.find(placeholder);
        if (start_pos == std::string::npos)
            return content;
        prompt_template.replace(start_pos, placeholder.length(), role);
    }
    const std::string placeholder = "%s";
    size_t start_pos              = prompt_template.find(placeholder);
    if (start_pos == std::string::npos)
        return content;
    prompt_template.replace(start_pos, placeholder.length(), content);
    return prompt_template;
}

std::string Llm::apply_prompt_template(const std::string& user_content) const {
    auto chat_prompt = config_->prompt_template();
    return apply_template(chat_prompt, user_content);
}

std::string Llm::apply_chat_template(const std::vector<PromptItem>& chat_prompts) const {
    auto chat_template = config_->chat_template();
    std::string prompt_result;
    auto iter = chat_prompts.begin();
    if (!config_->use_template()) {
        for (; iter != chat_prompts.end(); ++iter) {
            prompt_result += iter->second;
        }
        return prompt_result;
    }
    for (; iter != chat_prompts.end() - 1; ++iter) {
        prompt_result += apply_template(chat_template, iter->second, iter->first);
    }
    if (iter->first == "user") {
        prompt_result += apply_prompt_template(iter->second);
    } else {
        prompt_result += apply_template(chat_template, iter->second, iter->first);
    }
    return prompt_result;
}

void Llm::chat() {
    std::vector<PromptItem> history;
    history.push_back(std::make_pair("system", "You are a helpful assistant."));
    while (true) {
        std::cout << "\nUser: ";
        std::string user_str;
        std::getline(std::cin, user_str);
        if (user_str == "/exit") {
            break;
        }
        if (user_str == "/reset") {
            history.resize(1);
            std::cout << "\nA: reset done." << std::endl;
            continue;
        }
        std::cout << "\nAssistant: " << std::flush;
        if (config_->reuse_kv()) {
            response(user_str);
        } else {
            history.emplace_back(std::make_pair("user", user_str));
            std::ostringstream lineOs;
            response(history, &lineOs, nullptr, 1);
            auto line = lineOs.str();
            while (!stoped() && mState.gen_seq_len_ < config_->max_new_tokens()) {
                std::cout << line << std::flush;
                lineOs.str("");
                generate(1);
                line = lineOs.str();
            }
            history.emplace_back(std::make_pair("assistant", line));
        }
    }
}

void Llm::reset() {
    mState.history_ids_.clear();
    mState.all_seq_len_ = 0;
}

void Llm::generate_init(std::ostream* os, const char* end_with) {
    // init status
    mState.os_ = os;
    if (nullptr != end_with) {
        mState.end_with_ = end_with;
    }
    mState.gen_seq_len_ = 0;
    mState.vision_us_   = 0;
    mState.audio_us_    = 0;
    mState.prefill_us_  = 0;
    mState.decode_us_   = 0;
    mState.current_token_ = 0;
    if (!config_->reuse_kv()) {
        mState.all_seq_len_ = 0;
        mState.history_ids_.clear();
        mMeta->remove = mMeta->previous;
    }
    mState.output_ids_.clear();
    current_modules_ = prefill_modules_;
}
size_t Llm::getCurrentHistory() const {
    return mMeta->previous;
}
void Llm::eraseHistory(size_t begin, size_t end) {
    if (0 == end) {
        end = mMeta->previous;
    }
    if (end > mMeta->previous || begin >= end) {
        MNN_ERROR("Invalid erase range history larger than current\n");
        return;
    }
    if (mMeta->remove != 0) {
        MNN_ERROR("MNN-LLM: erase history hasn't been executed by response, override erase info\n");
    }
    mMeta->remove = mMeta->previous - begin;
    if (end != mMeta->previous) {
        mMeta->reserveHost.resize(2);
        mMeta->reserve = mMeta->reserveHost.data();
        mMeta->n_reserve = 1;
        mMeta->reserve[0] = end - begin;
        mMeta->reserve[1] = mMeta->previous - end;
    }
}

bool Llm::stoped() {
    return is_stop(mState.current_token_);
}

void Llm::generate(int max_token) {
    int len = 0;
    while (len < max_token) {
        auto st = std::chrono::system_clock::now();
        if (nullptr != mState.os_) {
            *mState.os_ << tokenizer_decode(mState.current_token_);
            *mState.os_ << std::flush;
        }
        mState.history_ids_.push_back(mState.current_token_);
        mMeta->add = 1;
        mMeta->remove = 0;
        auto logits = forward({mState.current_token_});
        len++;
        if (nullptr == logits.get()) {
            break;
        }
        if (logits->getInfo()->size == 0) {
            break;
        }
        mState.current_token_ = sample(logits, mState.history_ids_);
        auto et = std::chrono::system_clock::now();
        mState.decode_us_ += std::chrono::duration_cast<std::chrono::microseconds>(et - st).count();
        if (is_stop(mState.current_token_) && nullptr != mState.os_) {
            *mState.os_ << mState.end_with_ << std::flush;
            break;
        }
    }
}

std::vector<int> Llm::generate(const std::vector<int>& input_ids, int max_tokens) {
    if (max_tokens < 0) {
        max_tokens = config_->max_new_tokens();
    }
    mMeta->add = input_ids.size();
    mState.prompt_len_ = static_cast<int>(input_ids.size());
    mState.history_ids_.insert(mState.history_ids_.end(), input_ids.begin(), input_ids.end()); // push to history_ids_
    auto st          = std::chrono::system_clock::now();
    current_modules_ = prefill_modules_;
    auto logits      = forward(input_ids);
    if (nullptr == logits.get()) {
        return {};
    }
    mState.current_token_ = sample(logits, mState.history_ids_);
    logits = nullptr;
    auto et = std::chrono::system_clock::now();
    current_modules_ = decode_modules_;
    mState.prefill_us_ = std::chrono::duration_cast<std::chrono::microseconds>(et - st).count();
    generate(max_tokens);

#ifdef DUMP_PROFILE_INFO
    print_speed();
#endif
    return mState.output_ids_;
}

std::vector<int> Llm::tokenizer_encode(const std::string& user_content, bool use_template) {
    auto prompt = user_content;
    if (config_->use_template() && use_template) {
        prompt = apply_prompt_template(user_content);
    }
    auto input_ids = tokenizer_->encode(prompt);
    return input_ids;
}

void Llm::response(const std::string& user_content, std::ostream* os, const char* end_with, int max_new_tokens) {
    if (!end_with) {
        end_with = "\n";
    }
    generate_init(os, end_with);
    std::vector<int> input_ids;
    input_ids = tokenizer_encode(user_content);
    generate(input_ids, max_new_tokens);
}

void Llm::response(const std::vector<PromptItem>& chat_prompts, std::ostream* os, const char* end_with, int max_new_tokens) {
    if (chat_prompts.empty()) {
        return;
    }
    auto prompt = apply_chat_template(chat_prompts);
    response(prompt, os, end_with, max_new_tokens);
}

Llm::Llm(std::shared_ptr<LlmConfig> config) : config_(config) {
    mMeta.reset(new KVMeta);
}

Llm::~Llm() {
#if DEBUG_MODE == 1
    if (nullptr != gTimeTraceInfo) {
        float opSummer       = 0.0f;
        float opFlopsSummber = 0.0f;
        for (auto& iter : gTimeTraceInfo->mTypes) {
            float summer      = 0.0f;
            float summerflops = 0.0f;
            for (auto& t : iter.second) {
                for (auto& t0 : t.second) {
                    summer += t0.first;
                    summerflops += t0.second;
                }
            }
            summer      = summer;
            summerflops = summerflops;
            MNN_PRINT("%s : %.7f, FLOP: %.7f, Speed: %.7f GFlops\n", iter.first.c_str(), summer, summerflops,
                      summerflops / summer);
            opSummer += summer;
            opFlopsSummber += summerflops;
        }
        MNN_PRINT("OP Summer: %.7f, Flops: %.7f, Speed: %.7f GFlops\n", opSummer, opFlopsSummber,
                  opFlopsSummber / opSummer);
    }
#endif
    current_modules_.clear();
    decode_modules_.clear();
    prefill_modules_.clear();
    modules_.clear();
    runtime_manager_.reset();
}

void Llm::print_speed() {
    auto vision_s   = mState.vision_us_ * 1e-6;
    auto audio_s   = mState.audio_us_ * 1e-6;
    auto prefill_s = mState.prefill_us_ * 1e-6;
    auto decode_s  = mState.decode_us_ * 1e-6;
    auto total_s   = vision_s + audio_s + prefill_s + decode_s;
    printf("\n#################################\n");
    printf(" total tokens num  = %d\n", mState.prompt_len_ + mState.gen_seq_len_);
    printf("prompt tokens num  = %d\n", mState.prompt_len_);
    printf("output tokens num  = %d\n", mState.gen_seq_len_);
    printf("  total time = %.2f s\n", total_s);
    if (1 || vision_s) {
    printf(" vision time = %.2f s\n", audio_s);
    }
    if (1 || audio_s) {
    printf("  audio time = %.2f s\n", audio_s);
    }
    printf("prefill time = %.2f s\n", prefill_s);
    printf(" decode time = %.2f s\n", decode_s);
    printf("  total speed = %.2f tok/s\n", (mState.prompt_len_ + mState.gen_seq_len_) / total_s);
    printf("prefill speed = %.2f tok/s\n", mState.prompt_len_ / prefill_s);
    printf(" decode speed = %.2f tok/s\n", mState.gen_seq_len_ / decode_s);
    printf("   chat speed = %.2f tok/s\n", mState.gen_seq_len_ / total_s);
    printf("##################################\n");
}

static inline bool needNewVar(VARP var, int axis, int seq_len) {
    if (var == nullptr) {
        return true;
    }
    if (var->getInfo()->dim[axis] != seq_len) {
        return true;
    }
    return false;
}

VARP Llm::embedding(const std::vector<int>& input_ids) {
    AUTOTIME;
    int hidden_size = config_->hidden_size();
    int seq_len = static_cast<int>(input_ids.size());
    VARP res = _Input({seq_len, 1, hidden_size}, NCHW);
    // disk embedding to save memory
    disk_embedding_->embedding(input_ids, res->writeMap<float>());
    return res;
}

std::string Llm::tokenizer_decode(int id) {
    std::string word = tokenizer_->decode(id);
    // Fix utf-8 garbled characters
    if (word.length() == 6 && word[0] == '<' && word[word.length() - 1] == '>' && word[1] == '0' && word[2] == 'x') {
        int num = std::stoi(word.substr(3, 2), nullptr, 16);
        word    = static_cast<char>(num);
    }
    return word;
}

VARP Llm::gen_attention_mask(int seq_len) {
    int kv_seq_len = mState.all_seq_len_ + seq_len;
    if (seq_len == 1) {
        kv_seq_len = seq_len;
    }
    if (config_->attention_mask() == "float") {
        if (needNewVar(attention_mask_, 2, seq_len)) {
            attention_mask_ = _Input({1, 1, seq_len, kv_seq_len}, NCHW, halide_type_of<float>());
        } else {
            return attention_mask_;
        }
        auto ptr = attention_mask_->writeMap<float>();
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < kv_seq_len; j++) {
                int row = i + mState.all_seq_len_;
                ptr[kv_seq_len * i + j] = (j > row) * std::numeric_limits<float>::lowest();
            }
        }
        return attention_mask_;
    } else {
        if (needNewVar(attention_mask_, 2, seq_len)) {
            attention_mask_ = _Input({1, 1, seq_len, kv_seq_len}, NCHW, halide_type_of<int>());
        } else {
            return attention_mask_;
        }
        auto ptr = attention_mask_->writeMap<int>();
        if (config_->attention_mask() == "glm") {
            // chatglm
            for (int i = 0; i < seq_len * kv_seq_len; i++) {
                ptr[i] = 0;
            }
            if (seq_len > 1) {
                for (int i = 1; i < seq_len; i++) {
                    ptr[seq_len * i - 1] = 1;
                }
            }
        } else {
            bool is_glm2 = config_->attention_mask() == "glm2";
            for (int i = 0; i < seq_len; i++) {
                for (int j = 0; j < kv_seq_len; j++) {
                    int row              = i + mState.all_seq_len_;
                    ptr[seq_len * i + j] = is_glm2 ? j > row : j <= row;
                }
            }
        }
        return attention_mask_;
    }
}

VARP Llm::gen_position_ids(int seq_len) {
    if (config_->attention_mask() == "glm") {
        // chatglm
        if (needNewVar(position_ids_, 2, seq_len)) {
            position_ids_ = _Input({1, 2, seq_len}, NCHW, halide_type_of<int>());
        }
        auto ptr = position_ids_->writeMap<int>();
        if (seq_len == 1) {
            ptr[0] = mState.all_seq_len_ - mState.gen_seq_len_ - 2;
            ptr[1] = mState.gen_seq_len_ + 1;
        } else {
            for (int i = 0; i < seq_len - 1; i++) {
                ptr[i]           = i;
                ptr[seq_len + i] = 0;
            }
            ptr[seq_len - 1]     = seq_len - 2;
            ptr[2 * seq_len - 1] = 1;
        }
        return position_ids_;
    } else {
        bool is_glm2 = config_->attention_mask() == "glm2";
        if (needNewVar(position_ids_, 0, seq_len)) {
            position_ids_ = _Input({seq_len}, NCHW, halide_type_of<int>());
        }
        auto ptr = position_ids_->writeMap<int>();
        if (seq_len == 1) {
            ptr[0] = is_glm2 ? mState.gen_seq_len_ : mState.all_seq_len_;
        } else {
            for (int i = 0; i < seq_len; i++) {
                ptr[i] = i + mState.all_seq_len_;
            }
        }
        return position_ids_;
    }
}

bool Llm::is_stop(int token_id) {
    return tokenizer_->is_stop(token_id);
}

void Mllm::load() {
    Llm::load();
    if (config_->mllm_config_.empty()) {
        mllm_runtime_manager_ = runtime_manager_;
    } else {
        ScheduleConfig config;
        BackendConfig cpuBackendConfig;
        config.type      = backend_type_convert(config_->backend_type(true));;
        config.numThread = config_->thread_num(true);
        if (config_->power(true) == "high") {
            cpuBackendConfig.power = BackendConfig::Power_High;
        } else if (config_->power(true) == "low") {
            cpuBackendConfig.power = BackendConfig::Power_Low;
        }
        if (config_->memory(true) == "high") {
            cpuBackendConfig.memory = BackendConfig::Memory_High;
        } else if (config_->memory(true) == "low") {
            cpuBackendConfig.memory = BackendConfig::Memory_Low;
        }
        if (config_->precision(true) == "high") {
            cpuBackendConfig.precision = BackendConfig::Precision_High;
        } else if (config_->precision(true) == "low") {
            cpuBackendConfig.precision = BackendConfig::Precision_Low;
        }
        config.backendConfig = &cpuBackendConfig;
        mllm_runtime_manager_.reset(Executor::RuntimeManager::createRuntimeManager(config));
        mllm_runtime_manager_->setHint(MNN::Interpreter::MEM_ALLOCATOR_TYPE, 0);
        mllm_runtime_manager_->setHint(MNN::Interpreter::QKV_QUANT_OPTIONS, config_->quant_qkv());
        mllm_runtime_manager_->setHint(MNN::Interpreter::KVCACHE_SIZE_LIMIT, config_->kvcache_limit());
        std::string tmpPath = config_->tmp_path();
        if (config_->kvcache_mmap()) {
            mllm_runtime_manager_->setExternalPath(tmpPath, MNN::Interpreter::EXTERNAL_PATH_KVCACHE_DIR);
        }
        if (config_->use_mmap()) {
            mllm_runtime_manager_->setExternalPath(tmpPath, MNN::Interpreter::EXTERNAL_WEIGHT_DIR);
        }
    }
    Module::Config module_config;
    module_config.shapeMutable = true;
    module_config.rearrange    = true;
    if (config_->is_visual()) {
        mllm_runtime_manager_->setExternalFile(config_->visual_model() + ".weight");
        mul_module_.reset(Module::load({}, {}, config_->visual_model().c_str(), mllm_runtime_manager_, &module_config));
    }
    if (config_->is_audio()) {
        mllm_runtime_manager_->setExternalFile(config_->audio_model() + ".weight");
        mul_module_.reset(Module::load({}, {}, config_->audio_model().c_str(), mllm_runtime_manager_, &module_config));
    }
}

static void dump_impl(const float *signal, size_t size, int row = 0) {
if (row) {
int col = size / row;
printf("# %d, %d: [\n", row, col);
for (int i = 0; i < 3; i++) {
for (int j = 0; j < 3; j++) {
printf("%f, ", signal[i * col + j]);
}
printf("..., ");
for (int j = col - 3; j < col; j++) {
printf("%f, ", signal[i * col + j]);
}
printf("\n");
}
printf("..., \n");
for (int i = row - 3; i < row; i++) {
for (int j = 0; j < 3; j++) {
printf("%f, ", signal[i * col + j]);
}
printf("..., ");
for (int j = col - 3; j < col; j++) {
printf("%f, ", signal[i * col + j]);
}
printf("\n");
}
printf("]\n");
} else {
printf("# %lu: [", size);
for (int i = 0; i < 3; i++) {
printf("%f, ", signal[i]);
}
printf("..., ");
for (int i = size - 3; i < size; i++) {
printf("%f, ", signal[i]);
}
printf("]\n");
}
}

void dump_var(VARP var) {
auto dims    = var->getInfo()->dim;
bool isfloat = true;
printf("{\ndtype = ");
if (var->getInfo()->type == halide_type_of<float>()) {
printf("float");
isfloat = true;
} else if (var->getInfo()->type == halide_type_of<int>()) {
printf("int");
isfloat = false;
}
printf("\nformat = %d\n", var->getInfo()->order);
printf("\ndims = [");
for (int i = 0; i < dims.size(); i++) {
printf("%d ", dims[i]);
}
printf("]\n");

if (isfloat) {
if ((dims.size() > 2 && dims[1] > 1 && dims[2] > 1) || (dims.size() == 2 && dims[0] > 1 && dims[1] > 1)) {
int row = dims[dims.size() - 2];
dump_impl(var->readMap<float>(), var->getInfo()->size, row);
} else {
printf("data = [");
auto total = var->getInfo()->size;
if (total > 32) {
for (int i = 0; i < 5; i++) {
printf("%f ", var->readMap<float>()[i]);
}
printf("..., ");
for (int i = total - 5; i < total; i++) {
printf("%f ", var->readMap<float>()[i]);
}
} else {
for (int i = 0; i < total; i++) {
printf("%f ", var->readMap<float>()[i]);
}
}
printf("]\n}\n");
}
} else {
printf("data = [");
int size = var->getInfo()->size > 10 ? 10 : var->getInfo()->size;
for (int i = 0; i < size; i++) {
printf("%d ", var->readMap<int>()[i]);
}
printf("]\n}\n");
}
}

std::vector<int> Mllm::vision_process(const std::string& file) {
#ifdef LLM_SUPPORT_VISION
    VARP image = MNN::CV::imread(file);
    if (image == nullptr) {
        MNN_PRINT("Mllm Can't open image: %s\n", file.c_str());
        return std::vector<int>(0);
    }
    auto st    = std::chrono::system_clock::now();
    VARP image_embedding;

    if (mul_module_->getInfo()->inputNames[0] == "patches") {
        // Qwen2-VL
        image_height_ = round(image_height_ / 28.0) * 28;
        image_width_ = round(image_width_ / 28.0) * 28;
        image        = MNN::CV::resize(image, {image_height_, image_width_}, 0, 0,
                                     MNN::CV::INTER_LINEAR, MNN::CV::COLOR_BGR2RGB,
                                     image_mean_, image_norm_);
        image        = MNN::Express::_Unsqueeze(image, {0});
        image        = MNN::Express::_Convert(image, NCHW);
        auto patches = MNN::Express::_Concat({image, image}, 0);
        auto patches_dim = patches->getInfo()->dim;
        int temporal = patches_dim[0];
        int channel  = patches_dim[1];
        int height   = patches_dim[2];
        int width    = patches_dim[3];
        constexpr int temporal_patch_size = 2;
        constexpr int patch_size = 14;
        constexpr int merge_size = 2;
        int grid_t = temporal / temporal_patch_size;
        int grid_h = height / patch_size;
        int grid_w = width / patch_size;
        // build patches
        patches = MNN::Express::_Reshape(patches, {
            grid_t, temporal_patch_size,
            channel,
            grid_h / merge_size, merge_size, patch_size,
            grid_w / merge_size, merge_size, patch_size,
        });
        patches = MNN::Express::_Permute(patches, {0, 3, 6, 4, 7, 2, 1, 5, 8});
        patches = MNN::Express::_Reshape(patches, {
            grid_t * grid_h * grid_w,
            channel * temporal_patch_size * patch_size * patch_size
        });
        const int seq_len = grid_t * grid_h * grid_w;
        // build position_ids
        const int wblock_size = merge_size * merge_size;
        const int hblock_size = wblock_size * grid_w / merge_size;
        VARP position_ids = MNN::Express::_Input({2, seq_len}, NCHW, halide_type_of<int>());
        auto hpos_ptr = position_ids->writeMap<int>();
        auto wpos_ptr = hpos_ptr + seq_len;
        for (int i = 0; i < grid_h; i++) {
            int h_idx = i / merge_size, h_off = i % merge_size;
            for (int j = 0; j < grid_w; j++) {
                int w_idx = j / merge_size, w_off = j % merge_size;
                int index = h_idx * hblock_size + w_idx * wblock_size + h_off * 2 + w_off;
                hpos_ptr[index] = i;
                wpos_ptr[index] = j;
            }
        }
        // build attention_mask
        VARP attention_mask = MNN::Express::_Input({1, seq_len, seq_len}, NCHW);
        ::memset(attention_mask->writeMap<float>(), 0, seq_len * seq_len * sizeof(float));
#ifdef DEBUG_IMAGE
        patches.fix(MNN::Express::VARP::CONSTANT);
        patches->setName("patches");
        position_ids.fix(MNN::Express::VARP::CONSTANT);
        position_ids->setName("position_ids");
        attention_mask.fix(MNN::Express::VARP::CONSTANT);
        attention_mask->setName("attention_mask");
        MNN::Express::Variable::save({patches, position_ids, attention_mask}, "input.mnn");
#endif
        image_embedding = mul_module_->onForward({patches, position_ids, attention_mask})[0];
#ifdef DEBUG_IMAGE
        image_embedding->setName("image_embeds");
        MNN::Express::Variable::save({image_embedding}, "output.mnn");
#endif
    } else {
        image           = MNN::CV::resize(image, {image_height_, image_width_}, 0, 0,
                                          MNN::CV::INTER_LINEAR, MNN::CV::COLOR_BGR2RGB,
                                          image_mean_, image_norm_);
        image           = MNN::Express::_Unsqueeze(image, {0});
        image           = MNN::Express::_Convert(image, NC4HW4);
        image_embedding = mul_module_->forward(image);
    }
    auto et    = std::chrono::system_clock::now();
    mState.vision_us_ = std::chrono::duration_cast<std::chrono::microseconds>(et - st).count();
    mul_embeddings_.push_back(image_embedding);
    int visual_len = image_embedding->getInfo()->dim[0];
    std::vector<int> img_ids(visual_len, img_pad_);
    img_ids.insert(img_ids.begin(), vision_start_);
    img_ids.push_back(vision_end_);
    return img_ids;
#else
    return std::vector<int>(0);
#endif
}

template <typename T>
static inline VARP _var(std::vector<T> vec, const std::vector<int> &dims) {
    return _Const(vec.data(), dims, NHWC, halide_type_of<T>());
}

std::vector<int> Mllm::audio_process(const std::string& file) {
#ifdef LLM_SUPPORT_AUDIO
    constexpr int sample_rate = 16000;
    auto load_res        = MNN::AUDIO::load(file, sample_rate);
    VARP waveform        = load_res.first;
    if (waveform == nullptr) {
        MNN_PRINT("Mllm Can't open audio: %s\n", file.c_str());
        return std::vector<int>(0);
    }
    // int sample_rate      = load_res.second;
    int wav_len          = waveform->getInfo()->dim[0];
    int hop_length       = 160;
    auto st              = std::chrono::system_clock::now();
    auto input_features  = MNN::AUDIO::whisper_fbank(waveform);
    auto audio_embedding = mul_module_->forward(input_features);
    audio_embedding = _Permute(audio_embedding, {1, 0, 2});
    auto et         = std::chrono::system_clock::now();
    mState.audio_us_       = std::chrono::duration_cast<std::chrono::microseconds>(et - st).count();
    mul_embeddings_.push_back(audio_embedding);
    int embed_len = audio_embedding->getInfo()->dim[0];
    std::vector<int> audio_ids(embed_len, audio_pad_);
    return audio_ids;
#else
    return std::vector<int>(0);
#endif
}

std::vector<int> Mllm::multimode_process(const std::string& mode, std::string info) {
    auto file_info = info;
    if (mode == "img") {
        std::regex hw_regex(R"(<hw>(.*?)</hw>)");
        std::sregex_iterator iter(info.begin(), info.end(), hw_regex);
        std::sregex_iterator end;
        file_info = "";

        size_t currentPosition = 0;
        if (iter != end) {
            std::smatch match = *iter;
            size_t matchPosition = match.position();
            if (matchPosition > currentPosition) {
                file_info.append(info.substr(currentPosition, matchPosition - currentPosition));
            }

            std::stringstream hw_ss(match.str(1));
            char comma;
            hw_ss >> image_height_ >> comma >> image_width_;
            currentPosition = matchPosition + match.length();
        }
        if (currentPosition < info.length()) {
            file_info.append(info.substr(currentPosition));
        }
        // std::cout << "hw: " << image_height_ << ", " << image_width_ << std::endl;
        // std::cout << "file: " << file_info << std::endl;
    }
    if (file_info.substr(0, 4) == "http") {
        std::regex url_regex(R"(^https?://([^/]+)(/.*))");
        std::smatch url_match_result;
        std::string host, path;
        if (std::regex_search(file_info, url_match_result, url_regex) && url_match_result.size() == 3) {
            host = url_match_result[1].str();
            path = url_match_result[2].str();
        }
        // std::cout << host << "#" << path << std::endl;
        httplib::Client cli(host);
        auto res  = cli.Get(path);
        file_info = "downloaded_file";
        if (res && res->status == 200) {
            std::ofstream file(file_info, std::ios::binary);
            if (file.is_open()) {
                file.write(res->body.c_str(), res->body.size());
                std::cout << "File has been downloaded successfully." << std::endl;
                file.close();
            } else {
                std::cerr << "Unable to open file to write." << std::endl;
            }
        } else {
            std::cerr << "Failed to download file. Status code: " << (res ? res->status : 0) << std::endl;
        }
    }
    if (mode == "img" && config_->is_visual()) {
        return vision_process(file_info);
    }
    if (mode == "audio" && config_->is_audio()) {
        return audio_process(file_info);
    }
    return std::vector<int>(0);
}

std::vector<int> Mllm::tokenizer_encode(const std::string& query, bool use_template) {
    auto prompt = apply_prompt_template(query);
    // split query
    std::regex multimode_regex("<(img|audio)>(.*?)</\\1>");
    std::string::const_iterator searchStart(prompt.cbegin());
    std::smatch match;
    std::vector<std::string> img_infos;
    std::vector<int> ids{};

    while (std::regex_search(searchStart, prompt.cend(), match, multimode_regex)) {
        // std::cout << "img match: " << match[1].str() << std::endl;
        auto txt_ids = tokenizer_->encode(match.prefix().str());
        ids.insert(ids.end(), txt_ids.begin(), txt_ids.end());
        auto mul_ids = multimode_process(match[1].str(), match[2].str());
        ids.insert(ids.end(), mul_ids.begin(), mul_ids.end());
        searchStart = match.suffix().first;
    }
    if (searchStart != prompt.cend()) {
        auto txt_ids = tokenizer_->encode(std::string(searchStart, prompt.cend()));
        ids.insert(ids.end(), txt_ids.begin(), txt_ids.end());
    }
    // printf("ids = ["); for (auto id : ids) printf("%d, ", id); printf("]\n");
    return ids;
}

VARP Mllm::embedding(const std::vector<int>& input_ids) {
    if (input_ids.size() == 1) {
        return Llm::embedding(input_ids);
    }
    std::vector<VARP> embeddings;
    int mul_idx = 0;
    std::vector<int> cur_txt_ids;
    bool in_audio = false;
    for (int i = 0; i < input_ids.size(); i++) {
        int id = input_ids[i];
        // audio
        if (in_audio) {
            if (id == audio_pad_) {
                continue;
            } else {
                cur_txt_ids.clear();
                in_audio = false;
            }
        } else if (id == audio_pad_) {
            auto txt_embedding = Llm::embedding(cur_txt_ids);
            auto mul_embedding = mul_embeddings_[mul_idx++];
            embeddings.push_back(txt_embedding);
            embeddings.push_back(mul_embedding);
            in_audio = true;
        }
        // vision
        if (id == img_pad_) {
            continue;
        }
        cur_txt_ids.push_back(id);
        if (id == vision_start_) {
            auto txt_embedding = Llm::embedding(cur_txt_ids);
            auto mul_embedding = mul_embeddings_[mul_idx++];
            embeddings.push_back(txt_embedding);
            embeddings.push_back(mul_embedding);
        } else if (id == vision_end_) {
            cur_txt_ids.clear();
            cur_txt_ids.push_back(id);
        }
    }
    mul_embeddings_.clear();
    if (!cur_txt_ids.empty()) {
        auto txt_embedding = Llm::embedding(cur_txt_ids);
        embeddings.push_back(txt_embedding);
    }
    auto embedding = MNN::Express::_Concat(embeddings, 0);
    return embedding;
}
// Llm end

// Embedding start
float Embedding::dist(VARP var0, VARP var1) {
    auto distVar = _Sqrt(_ReduceSum(_Square(var0 - var1)));
    auto dist    = distVar->readMap<float>()[0];
    return dist;
}

Embedding* Embedding::createEmbedding(const std::string& config_path, bool load) {
    std::shared_ptr<LlmConfig> config(new LlmConfig(config_path));
    Embedding* embedding = new Embedding(config);
    if (load) {
        embedding->load();
    }
    return embedding;
}

Embedding::Embedding(std::shared_ptr<LlmConfig> config) : Llm(config) {
}

int Embedding::dim() const {
    return config_->hidden_size();
}

void Embedding::load() {
    init_runtime();
    printf("load tokenizer\n");
    std::cout << config_->tokenizer_file() << std::endl;
    // 1. load vocab
    tokenizer_.reset(Tokenizer::createTokenizer(config_->tokenizer_file()));
    printf("load tokenizer Done\n");
    disk_embedding_.reset(new DiskEmbedding(config_));
    // 2. load model
    Module::Config module_config;
    module_config.shapeMutable = true;
    module_config.rearrange    = true;
    auto model_path            = config_->llm_model();
    MNN_PRINT("load %s ... ", model_path.c_str());
    modules_.resize(1);
    modules_[0].reset(Module::load({"input_ids", "attention_mask", "position_ids"}, {"sentence_embeddings"},
                                   model_path.c_str(), runtime_manager_, &module_config));
    MNN_PRINT("Done!\n");
}

VARP Embedding::ids_embedding(const std::vector<int>& ids) {
    int prompt_len           = ids.size();
    auto inputs_ids          = embedding(ids);
    auto attention_mask      = gen_attention_mask(prompt_len);
    auto position_ids        = gen_position_ids(prompt_len);
    auto outputs             = modules_[0]->onForward({inputs_ids, attention_mask, position_ids});
    auto sentence_embeddings = outputs[0];
    return sentence_embeddings;
}

VARP Embedding::txt_embedding(const std::string& txt) {
    return ids_embedding(tokenizer_encode(txt));
}

VARP Embedding::gen_attention_mask(int seq_len) {
    auto attention_mask = _Input({1, 1, 1, seq_len}, NCHW, halide_type_of<int>());
    auto ptr            = attention_mask->writeMap<int>();
    for (int i = 0; i < seq_len; i++) {
        ptr[i] = 1;
    }
    return attention_mask;
}

VARP Embedding::gen_position_ids(int seq_len) {
    auto position_ids = _Input({1, seq_len}, NCHW, halide_type_of<int>());
    auto ptr          = position_ids->writeMap<int>();
    for (int i = 0; i < seq_len; i++) {
        ptr[i] = i;
    }
    return position_ids;
}
// Embedding end
} // namespace Transformer
} // namespace MNN
