//
//  llm.cpp
//
//  Created by MNN on 2023/08/25.
//  ZhaodeWang
//
// #define MNN_OPEN_TIME_TRACE 1

#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_set>
#include <regex>

#include <MNN/expr/ExecutorScope.hpp>
#include <MNN/AutoTime.hpp>
#include "cpp/ExprDebug.hpp"
#include "llm/llm.hpp"
#include "tokenizer.hpp"
#include "llmconfig.hpp"
// 0: no debug, 1: test op time, 2: print tensor info
#define DEBUG_MODE 0

#ifdef LLM_SUPPORT_VISION
#include "httplib.h"
#include <cv/cv.hpp>
#endif

using namespace MNN::Express;
namespace MNN {
namespace Transformer {

typedef void (*DequantFunction)(const uint8_t*, float*, float, float, int);

static void q41_dequant_ref(const uint8_t* src, float* dst, float scale, float zero, int size) {
    for (int i = 0; i < size / 2; i++) {
        int x = src[i];
        int x1 = x / 16 - 8;
        int x2= x % 16 - 8;
        float w1 = x1 * scale + zero;
        float w2 = x2 * scale + zero;
        dst[2 * i] = w1;
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
    ~DiskEmbedding() {}
    void embedding(const std::vector<int>& input_ids, float* ptr);
private:
    void seek_read(uint8_t* dst, size_t size, size_t offset);
    std::unique_ptr<uint8_t[]> alpha_ = nullptr;
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
    hidden_size_ = config->hidden_size();
    if (tie_embeddings.size() == 5) {
        w_offset_    = tie_embeddings[0];
        quant_bit_   = tie_embeddings[3];
        quant_block_ = tie_embeddings[4];
        block_num_ = hidden_size_ / quant_block_;
        weight_token_size_ = hidden_size_ * quant_bit_ / 8;
        fp_.reset(fopen(config->llm_weight().c_str(), "rb"));
        // TODO: optimize dequant function
        dequant_ = quant_bit_ == 8 ? q81_dequant_ref : q41_dequant_ref;
        auto a_offset    = tie_embeddings[1];
        auto alpha_size  = tie_embeddings[2];
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
            auto dptr = dst + i * hidden_size_;
            auto alpha_ptr = reinterpret_cast<float*>(alpha_.get()) + token * block_num_ * 2;
            for (int n = 0; n < block_num_; n++) {
                auto dst_ptr = dptr + n * quant_block_;
                uint8_t* src_ptr = weight_.get() + n * (quant_block_ * quant_bit_ / 8);
                float zero = (alpha_ptr + n * 2)[0];
                float scale = (alpha_ptr + n * 2)[1];
                dequant_(src_ptr, dst_ptr, scale, zero, quant_block_);
            }
        }
    } else {
        // bf16
        for (size_t i = 0; i < input_ids.size(); i++) {
            seek_read(weight_.get(), weight_token_size_, input_ids[i] * weight_token_size_);
            int16_t* dst_ptr = reinterpret_cast<int16_t*>(dst + i * hidden_size_);
            for (int j = 0; j < hidden_size_; j++) {
                dst_ptr[j * 2] = 0;
                dst_ptr[j * 2 + 1] = reinterpret_cast<int16_t*>(weight_.get())[j];
            }
        }
    }
}

class Lvlm : public Llm {
public:
    Lvlm(std::shared_ptr<LlmConfig> config) : Llm(config) {
        image_size_ = config->llm_config_.value("image_size", image_size_);
        image_pad_ = config->llm_config_.value("image_pad", image_pad_);
        vision_start_ = config->llm_config_.value("vision_start", vision_start_);
        vision_end_ = config->llm_config_.value("vision_end", vision_end_);
        image_mean_ = config->llm_config_.value("image_mean", image_mean_);
        image_norm_ = config->llm_config_.value("image_norm", image_norm_);
    }
    ~Lvlm() { visual_module_.reset(); }
    virtual void load() override;

    virtual std::vector<int> tokenizer_encode(const std::string& query, bool use_template = true) override;
    virtual MNN::Express::VARP embedding(const std::vector<int>& input_ids) override;
private:
    int image_size_ = 448, vision_start_ = 151857, vision_end_ = 151858, image_pad_ = 151859;
    std::vector<float> image_mean_ {122.7709383 , 116.7460125 , 104.09373615};
    std::vector<float> image_norm_ {0.01459843, 0.01500777, 0.01422007};
    std::vector<int> image_process(const std::string& img_info);
    std::shared_ptr<Module> visual_module_;
    std::vector<VARP> image_embeddings_;
};

// Llm start
Llm* Llm::createLLM(const std::string& config_path) {
    std::shared_ptr<LlmConfig> config(new LlmConfig(config_path));
    Llm* llm = nullptr;
    if (config->is_visual()) {
        llm = new Lvlm(config);
    } else {
        llm = new Llm(config);
    }
    return llm;
}

static MNNForwardType backend_type_convert(const std::string& type_str) {
    if (type_str == "cpu") return MNN_FORWARD_CPU;
    if (type_str == "metal") return MNN_FORWARD_METAL;
    if (type_str == "cuda") return MNN_FORWARD_CUDA;
    if (type_str == "opencl") return MNN_FORWARD_OPENCL;
    if (type_str == "opengl") return MNN_FORWARD_OPENGL;
    if (type_str == "vulkan") return MNN_FORWARD_VULKAN;
    if (type_str == "npu") return MNN_FORWARD_NN;
    return MNN_FORWARD_AUTO;
}

std::string Llm::dump_config() {
    return config_->config_.dump();
}

bool Llm::set_config(const std::string& content) {
    return config_->config_.merge(content.c_str());
}

void Llm::init_runtime() {
    ScheduleConfig config;
    BackendConfig cpuBackendConfig;
    config.type          = backend_type_convert(config_->backend_type());
    config.numThread     = config_->thread_num();
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
    ExecutorScope::Current()->setGlobalExecutorConfig(config.type, cpuBackendConfig, config.numThread);

    runtime_manager_.reset(Executor::RuntimeManager::createRuntimeManager(config));
    runtime_manager_->setHint(MNN::Interpreter::MEM_ALLOCATOR_TYPE, 0);
    runtime_manager_->setHint(MNN::Interpreter::DYNAMIC_QUANT_OPTIONS, 1); // 1: per batch quant, 2: per tensor quant
    runtime_manager_->setHint(MNN::Interpreter::QKV_QUANT_OPTIONS, config_->quant_qkv());
    runtime_manager_->setHint(MNN::Interpreter::KVCACHE_SIZE_LIMIT, config_->kvcache_limit());
    std::string tmpPath = config_->tmp_path();
    if (config_->kvcache_mmap()) {
        runtime_manager_->setExternalPath(tmpPath, MNN::Interpreter::EXTERNAL_PATH_KVCACHE_DIR);
    }
    if (config_->use_mmap()) {
        runtime_manager_->setExternalPath(tmpPath, MNN::Interpreter::EXTERNAL_WEIGHT_DIR);
    }

#if DEBUG_MODE==1
    runtime_manager_->setMode(MNN::Interpreter::Session_Debug);
    _initTimeTrace();
#endif
#if DEBUG_MODE==2
    runtime_manager_->setMode(MNN::Interpreter::Session_Debug);
    _initTensorStatic();
#endif
    {
        runtime_manager_->setCache(".tempcache");
    }
}

void Llm::load() {
    init_runtime();
    // init module status
    key_value_shape_ = config_->key_value_shape();
    is_single_ = config_->is_single();
    attention_fused_ = config_->attention_fused();
    MNN_PRINT("### is_single_ = %d\n", is_single_);
    // 1. load vocab
    MNN_PRINT("load tokenizer\n");
    tokenizer_.reset(Tokenizer::createTokenizer(config_->tokenizer_file()));
    MNN_PRINT("load tokenizer Done\n");
    disk_embedding_.reset(new DiskEmbedding(config_));
    // 3. load model
    Module::Config module_config;
    module_config.shapeMutable = true;
    module_config.rearrange = true;
    // using base module for lora module
    if (base_module_ != nullptr) {
        module_config.base = base_module_;
    }
    int layer_nums = config_->layer_nums();
    if (is_single_) {
        // load single model
        key_value_shape_.insert(key_value_shape_.begin(), layer_nums);
        modules_.resize(1);
        std::string model_path = config_->llm_model();
        MNN_PRINT("load %s ... ", model_path.c_str());
        runtime_manager_->setExternalFile(config_->llm_weight());
        if (attention_fused_) {
            modules_[0].reset(Module::load(
                                           {"input_ids", "attention_mask", "position_ids"},
                                           {"logits"}, model_path.c_str(), runtime_manager_, &module_config));
        } else {
            modules_[0].reset(Module::load(
                                           {"input_ids", "attention_mask", "position_ids", "past_key_values"},
                                           {"logits", "presents"}, model_path.c_str(), runtime_manager_, &module_config));
        }
        MNN_PRINT("Load Module Done!\n");
    } else {
        MNN_ERROR("Split version is depercerate\n");
    }
    decode_modules_.resize(modules_.size());
    for (int v=0; v<modules_.size(); ++v) {
        decode_modules_[v].reset(Module::clone(modules_[v].get()));
    }
    MNN_PRINT("Clone Decode Module Done!\n");

    prefill_modules_ = modules_;
}

size_t Llm::apply_lora(const std::string& lora_path) {
    std::string model_path = config_->base_dir_ + "/" + lora_path;
    Module::Config module_config;
    module_config.shapeMutable = true;
    module_config.rearrange = true;
    module_config.base = modules_.begin()->get();
    size_t lora_index = modules_.size();
    if (attention_fused_) {
        modules_.emplace_back(Module::load({"input_ids", "attention_mask", "position_ids"},
                                           {"logits"}, model_path.c_str(), runtime_manager_, &module_config));
    } else {
        modules_.emplace_back(Module::load({"input_ids", "attention_mask", "position_ids", "past_key_values"},
                                           {"logits", "presents"}, model_path.c_str(), runtime_manager_, &module_config));
    }
    select_module(lora_index);
    return lora_index;
}

Llm* Llm::create_lora(const std::string& lora_path) {
    auto llm = new Llm(config_);
    llm->set_config("{\"llm_model\": \"" + lora_path + "\"}");
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
    if(type != OP_ENCODER_NUMBER) {
        MNN_ERROR("tuning type not supported\n");
        return;
    }
    if(config_->backend_type() != "metal") {
        return;
    }

    current_modules_ = decode_modules_;
    int64_t min_time = INT64_MAX;
    int prefer_candidate = 10;
    for(auto& candidate : candidates) {
        runtime_manager_->setHint(MNN::Interpreter::OP_ENCODER_NUMBER_FOR_COMMIT, candidate);

        auto st = std::chrono::system_clock::now();
        auto logits = forward({0});
        if (nullptr == logits.get()) {
            return;
        }
        if (logits->getInfo()->size == 0) {
            return;
        }
        auto token = sample(logits, {});
        auto et = std::chrono::system_clock::now();
        int64_t time = std::chrono::duration_cast<std::chrono::microseconds>(et - st).count();
        if(time < min_time) {
            prefer_candidate = candidate;
            min_time = time;
            //MNN_PRINT("op encode number:%d, decode time: %lld us\n", candidate, time);
        }
    }
    runtime_manager_->setHint(MNN::Interpreter::OP_ENCODER_NUMBER_FOR_COMMIT, prefer_candidate);
}

VARP Llm::forward(const std::vector<int>& input_ids) {
    int seq_len = input_ids.size();
    auto attention_mask = gen_attention_mask(seq_len);
    auto position_ids = gen_position_ids(seq_len);
    VARP logits;
    if (is_single_) {
        std::vector<MNN::Express::VARP> outputs;
        auto hidden_states = embedding(input_ids);
        if (attention_fused_) {
            outputs = current_modules_.back()->onForward({hidden_states, attention_mask, position_ids});
        } else {
            outputs = current_modules_.back()->onForward({hidden_states, attention_mask, position_ids, past_key_values_[0]});
        }
        if (outputs.empty()) {
            return nullptr;
        }
        logits = outputs[0];
        if (!attention_fused_) {
            past_key_values_[0] = outputs[1];
        }
    } else {
        MNN_ERROR("Split models is depercarate\n");
        return nullptr;
    }
    all_seq_len_ += seq_len;
    gen_seq_len_++;
    return logits;
}

int Llm::sample(VARP logits, const std::vector<int>& pre_ids) {
    std::unordered_set<int> ids_set(pre_ids.begin(), pre_ids.end());
    auto scores = (float*)(logits->readMap<float>());
    auto size = logits->getInfo()->size;
    // repetition penalty
    const float repetition_penalty = 1.1;
    for (auto id : ids_set) {
        float score = scores[id];
        scores[id] = score < 0 ? score * repetition_penalty : score / repetition_penalty;
    }
    // argmax
    float max_score = 0;
    int token_id = 0;
    for (int i = 0; i < size; i++) {
        float score = scores[i];
        if (score > max_score) {
            max_score = score;
            token_id = i;
        }
    }
    return token_id;
}

static std::string apply_template(std::string prompt_template, const std::string& content, const std::string& role = "") {
    if (prompt_template.empty()) return content;
    if (!role.empty()) {
        const std::string placeholder = "%r";
        size_t start_pos = prompt_template.find(placeholder);
        if (start_pos == std::string::npos) return content;
        prompt_template.replace(start_pos, placeholder.length(), role);
    }
    const std::string placeholder = "%s";
    size_t start_pos = prompt_template.find(placeholder);
    if (start_pos == std::string::npos) return content;
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
        std::cout << "\nQ: ";
        std::string user_str;
        std::cin >> user_str;
        if (user_str == "/exit") {
            break;
        }
        if (user_str == "/reset") {
            history.resize(1);
            std::cout << "\nA: reset done." << std::endl;
            continue;
        }
        std::cout << "\nA: " << std::flush;
        if (config_->reuse_kv()) {
            response(user_str);
        } else {
            history.emplace_back(std::make_pair("user", user_str));
            auto assistant_str = response(history);
            history.emplace_back(std::make_pair("assistant", assistant_str));
        }
        std::cout << std::endl;
    }
}

void Llm::reset() {
    history_ids_.clear();
    all_seq_len_ = 0;
}

void Llm::generate_init() {
    // init status
    gen_seq_len_ = 0;
    prefill_us_ = 0;
    decode_us_ = 0;
    past_key_values_.clear();
    if (is_single_) {
        past_key_values_.push_back(_Input(key_value_shape_, NCHW));
    } else {
        for (int i = 0; i < config_->layer_nums(); i++) {
            past_key_values_.push_back(_Input(key_value_shape_, NCHW));
        }
    }
    if (!config_->reuse_kv()) {
        all_seq_len_ = 0;
        history_ids_.clear();
    }
    current_modules_ = prefill_modules_;
}

std::vector<int> Llm::generate(const std::vector<int>& input_ids, int max_new_tokens) {
    generate_init();
    std::vector<int> output_ids, all_ids = input_ids;
    prompt_len_ = static_cast<int>(input_ids.size());
    if (max_new_tokens < 0) { max_new_tokens = config_->max_new_tokens(); }
    // prefill
    current_modules_ = prefill_modules_;
    auto logits = forward(input_ids);
    if (logits.get() == nullptr) {
        return {};
    }
    int token = sample(logits, all_ids);
    output_ids.push_back(token);
    all_ids.push_back(token);
    // decode
    current_modules_ = decode_modules_;
    while (gen_seq_len_ < max_new_tokens) {
        logits = nullptr;
        logits = forward({token});
        if (logits.get() == nullptr) {
            return {};
        }
        token = sample(logits, all_ids);
        if (is_stop(token)) { break; }
        output_ids.push_back(token);
        all_ids.push_back(token);
    }
    return output_ids;
}

std::string Llm::generate(const std::vector<int>& input_ids, std::ostream* os, const char* end_with) {
    if (mTracing) {
        // Skip real forward
        current_modules_ = prefill_modules_;
        forward(input_ids);
        current_modules_ = decode_modules_;
        forward({input_ids[0]});
        forward({input_ids[0]});
        return "Test";
    }
    prompt_len_ = static_cast<int>(input_ids.size());
    history_ids_.insert(history_ids_.end(), input_ids.begin(), input_ids.end()); // push to history_ids_
    auto st = std::chrono::system_clock::now();
    current_modules_ = prefill_modules_;
    auto logits = forward(input_ids);
    if (nullptr == logits.get()) {
        return "";
    }
    int token = sample(logits, history_ids_);
    auto et = std::chrono::system_clock::now();
    current_modules_ = decode_modules_;
    std::string output_str = tokenizer_decode(token);
    prefill_us_ = std::chrono::duration_cast<std::chrono::microseconds>(et - st).count();
    *os << output_str << std::flush;
    while (gen_seq_len_ < config_->max_new_tokens()) {
        st = std::chrono::system_clock::now();
        history_ids_.push_back(token);
        logits = nullptr;
        logits = forward({token});
        if (nullptr == logits.get()) {
            return "";
        }
        if (logits->getInfo()->size == 0) {
            return "";
        }
        token = sample(logits, history_ids_);
        et = std::chrono::system_clock::now();
        decode_us_ += std::chrono::duration_cast<std::chrono::microseconds>(et - st).count();
        if (is_stop(token)) {
            *os << end_with << std::flush;
            break;
        }
        auto word = tokenizer_decode(token);
        *os << word << std::flush;
        output_str += word;
    }
    ExecutorScope::Current()->gc(Executor::FULL);
#ifdef DUMP_PROFILE_INFO
    print_speed();
#endif
    return output_str;
}

std::vector<int> Llm::tokenizer_encode(const std::string& user_content, bool use_template) {
    if (!use_template) {
        return tokenizer_->encode(user_content);
    }
    auto prompt = apply_prompt_template(user_content);
    auto input_ids = tokenizer_->encode(prompt);
    return input_ids;
}

std::string Llm::response(const std::string& user_content, std::ostream* os, const char* end_with) {
    generate_init();
    if (!end_with) { end_with = "\n"; }
    std::vector<int> input_ids;
    if (config_->reuse_kv()) {
        auto prompt = apply_prompt_template(user_content);
        if (all_seq_len_ > 0) {
            prompt = "<|im_end|>\n" + prompt;
        }
        input_ids = tokenizer_->encode(prompt);
    } else {
        input_ids = tokenizer_encode(user_content);
    }
    return generate(input_ids, os, end_with);
}

std::string Llm::response(const std::vector<PromptItem>& chat_prompts, std::ostream* os, const char* end_with) {
    if (chat_prompts.empty()) { return ""; }
    generate_init();
    if (!end_with) { end_with = "\n"; }
    auto prompt = apply_chat_template(chat_prompts);
    if (config_->reuse_kv() && all_seq_len_ > 0) {
        prompt = "<|im_end|>\n" + prompt;
    }
    // std::cout << "# prompt : " << prompt << std::endl;
    auto input_ids = tokenizer_->encode(prompt);
    // printf("input_ids (%lu): ", input_ids.size()); for (auto id : input_ids) printf("%d, ", id); printf("\n");
    return generate(input_ids, os, end_with);
}

Llm::~Llm() {
#if DEBUG_MODE==1
    if (nullptr != gTimeTraceInfo) {
        float opSummer = 0.0f;
        float opFlopsSummber = 0.0f;
        for (auto& iter : gTimeTraceInfo->mTypes) {
            float summer = 0.0f;
            float summerflops = 0.0f;
            for (auto& t : iter.second) {
                for (auto& t0 : t.second) {
                    summer += t0.first;
                    summerflops += t0.second;
                }
            }
            summer = summer;
            summerflops = summerflops;
            MNN_PRINT("%s : %.7f, FLOP: %.7f, Speed: %.7f GFlops\n", iter.first.c_str(), summer, summerflops, summerflops / summer);
            opSummer += summer;
            opFlopsSummber+= summerflops;
        }
        MNN_PRINT("OP Summer: %.7f, Flops: %.7f, Speed: %.7f GFlops\n", opSummer, opFlopsSummber, opFlopsSummber/opSummer);
    }
#endif
    current_modules_.clear();
    decode_modules_.clear();
    prefill_modules_.clear();
    modules_.clear();
    runtime_manager_.reset();
}

void Llm::print_speed() {
    auto prefill_s = prefill_us_ * 1e-6;
    auto decode_s = decode_us_ * 1e-6;
    auto total_s = prefill_s + decode_s;
    printf("\n#################################\n");
    printf(" total tokens num  = %d\n", prompt_len_ + gen_seq_len_);
    printf("prompt tokens num  = %d\n", prompt_len_);
    printf("output tokens num  = %d\n", gen_seq_len_);
    printf("  total time = %.2f s\n", total_s);
    printf("prefill time = %.2f s\n", prefill_s);
    printf(" decode time = %.2f s\n", decode_s);
    printf("  total speed = %.2f tok/s\n", (prompt_len_ + gen_seq_len_) / total_s);
    printf("prefill speed = %.2f tok/s\n", prompt_len_ / prefill_s);
    printf(" decode speed = %.2f tok/s\n", gen_seq_len_ / decode_s);
    printf("   chat speed = %.2f tok/s\n", gen_seq_len_ / total_s);
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
    if (needNewVar(inputs_embeds_, 0, seq_len)) {
        inputs_embeds_ = _Input({seq_len, 1, hidden_size}, NCHW);
    }
    // disk embedding to save memory
    disk_embedding_->embedding(input_ids, inputs_embeds_->writeMap<float>());
    return inputs_embeds_;
}

std::string Llm::tokenizer_decode(int id) {
    std::string word = tokenizer_->decode(id);
    // Fix utf-8 garbled characters
    if (word.length() == 6 && word[0] == '<' && word[word.length()-1] == '>' && word[1] == '0' && word[2] == 'x') {
        int num = std::stoi(word.substr(3, 2), nullptr, 16);
        word = static_cast<char>(num);
    }
    return word;
}

VARP Llm::gen_attention_mask(int seq_len) {
    int kv_seq_len = all_seq_len_ + seq_len;
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
                int row = i + all_seq_len_;
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
                    int row = i + all_seq_len_;
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
            ptr[0] = all_seq_len_ - gen_seq_len_ - 2;
            ptr[1] = gen_seq_len_ + 1;
        } else {
            for (int i = 0; i < seq_len - 1; i++) {
                ptr[i] = i;
                ptr[seq_len + i] = 0;
            }
            ptr[seq_len - 1] = seq_len - 2;
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
            ptr[0] = is_glm2 ? gen_seq_len_ : all_seq_len_;
        } else {
            for (int i = 0; i < seq_len; i++) {
                ptr[i] = i + all_seq_len_;
            }
        }
        return position_ids_;
    }
}

bool Llm::is_stop(int token_id) {
    return tokenizer_->is_stop(token_id);
}

void Lvlm::load() {
    Llm::load();
    Module::Config module_config;
    module_config.shapeMutable = true;
    module_config.rearrange = false;
    runtime_manager_->setExternalFile(config_->visual_model() + ".weight");
    visual_module_.reset(Module::load({}, {}, config_->visual_model().c_str(), runtime_manager_, &module_config));
}

std::vector<int> Lvlm::image_process(const std::string& image_info) {
#ifdef LLM_SUPPORT_VISION
    VARP image = nullptr;
    if (image_info.substr(0, 4) == "http") {
        std::regex url_regex(R"(^https?://([^/]+)(/.*))");
        std::smatch url_match_result;
        std::string host, path;
        if (std::regex_search(image_info, url_match_result, url_regex) && url_match_result.size() == 3) {
            host = url_match_result[1].str();
            path = url_match_result[2].str();
        }
        // std::cout << host << "#" << path << std::endl;
        httplib::Client cli(host);
        auto res = cli.Get(path);
        std::string img_file = "downloaded_image.jpg";
        if (res && res->status == 200) {
            std::ofstream file(img_file, std::ios::binary);
            if (file.is_open()) {
                file.write(res->body.c_str(), res->body.size());
                std::cout << "Image has been downloaded successfully." << std::endl;
                file.close();
            } else {
                std::cerr << "Unable to open file to write image." << std::endl;
                exit(0);
            }
        } else {
            std::cerr << "Failed to download image. Status code: " << (res ? res->status : 0) << std::endl;
            exit(0);
        }
        image = MNN::CV::imread(img_file);
    } else {
        image = MNN::CV::imread(image_info);
    }
    image = MNN::CV::resize(image, {image_size_, image_size_}, 0, 0, MNN::CV::INTER_LINEAR, MNN::CV::COLOR_BGR2RGB, image_mean_, image_norm_);
    image = MNN::Express::_Unsqueeze(image, {0});
    image = MNN::Express::_Convert(image, NC4HW4);
    auto image_embedding = visual_module_->forward(image);
    image_embeddings_.push_back(image_embedding);
    int visual_len = image_embedding->getInfo()->dim[0];
    std::vector<int> img_ids(visual_len, image_pad_);
    img_ids.insert(img_ids.begin(), vision_start_);
    img_ids.push_back(vision_end_);
    return img_ids;
#else
    return std::vector<int>(0);
#endif
}

std::vector<int> Lvlm::tokenizer_encode(const std::string& query, bool use_template) {
    auto prompt = apply_prompt_template(query);
    // split query
    std::regex img_regex("<img>(.*?)</img>");
    std::string::const_iterator searchStart(prompt.cbegin());
    std::smatch match;
    std::vector<std::string> img_infos;
    std::vector<int> ids {};

    while (std::regex_search(searchStart, prompt.cend(), match, img_regex)) {
        // std::cout << "img match: " << match[1].str() << std::endl;
        auto txt_ids = tokenizer_->encode(match.prefix().str());
        ids.insert(ids.end(), txt_ids.begin(), txt_ids.end());
        auto img_ids = image_process(match[1].str());
        ids.insert(ids.end(), img_ids.begin(), img_ids.end());
        searchStart = match.suffix().first;
    }
    if (searchStart != prompt.cend()) {
        auto txt_ids = tokenizer_->encode(std::string(searchStart, prompt.cend()));
        ids.insert(ids.end(), txt_ids.begin(), txt_ids.end());
    }
    // printf("ids = ["); for (auto id : ids) printf("%d, ", id); printf("]\n");
    return ids;
}

VARP Lvlm::embedding(const std::vector<int>& input_ids) {
    if (input_ids.size() == 1) {
        return Llm::embedding(input_ids);
    }
    std::vector<VARP> embeddings;
    int img_idx = 0;
    std::vector<int> cur_txt_ids;
    for (int i = 0; i < input_ids.size(); i++) {
        int id = input_ids[i];
        if (id == image_pad_) {
            continue;
        }
        cur_txt_ids.push_back(id);
        if (id == vision_start_) {
            auto txt_embedding = Llm::embedding(cur_txt_ids);
            auto img_embedding = image_embeddings_[img_idx++];
            embeddings.push_back(txt_embedding);
            embeddings.push_back(img_embedding);
        } else if (id == vision_end_) {
            cur_txt_ids.clear();
            cur_txt_ids.push_back(id);
        }
    }
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
    auto dist = distVar->readMap<float>()[0];
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

Embedding::Embedding(std::shared_ptr<LlmConfig> config) : Llm(config) {}

int Embedding::dim() const { return config_->hidden_size(); }

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
    module_config.rearrange = true;
    auto model_path = config_->llm_model();
    MNN_PRINT("load %s ... ", model_path.c_str());
    modules_.resize(1);
    modules_[0].reset(Module::load(
                                   {"input_ids", "attention_mask", "position_ids"},
                                   {"sentence_embeddings"}, model_path.c_str(), runtime_manager_, &module_config));
    MNN_PRINT("Done!\n");
}

VARP Embedding::ids_embedding(const std::vector<int>& ids) {
    int prompt_len = ids.size();
    auto inputs_ids = embedding(ids);
    auto attention_mask = gen_attention_mask(prompt_len);
    auto position_ids = gen_position_ids(prompt_len);
    auto outputs = modules_[0]->onForward({inputs_ids, attention_mask, position_ids});
    auto sentence_embeddings = outputs[0];
    return sentence_embeddings;
}

VARP Embedding::txt_embedding(const std::string& txt) {
    return ids_embedding(tokenizer_encode(txt));
}

VARP Embedding::gen_attention_mask(int seq_len) {
    auto attention_mask = _Input({1, 1, 1, seq_len}, NCHW, halide_type_of<int>());
    auto ptr = attention_mask->writeMap<int>();
    for (int i = 0; i < seq_len; i++) {
        ptr[i] = 1;
    }
    return attention_mask;
}

VARP Embedding::gen_position_ids(int seq_len) {
    auto position_ids = _Input({1, seq_len}, NCHW, halide_type_of<int>());
    auto ptr = position_ids->writeMap<int>();
    for (int i = 0; i < seq_len; i++) {
        ptr[i] = i;
    }
    return position_ids;
}
// Embedding end
}
}
