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

#ifdef USING_VISUAL_MODEL
#include "httplib.h"
#include <cv/cv.hpp>
#endif
using namespace MNN::Express;
namespace MNN {
namespace Transformer {

class Lvlm : public Llm {
public:
    Lvlm(std::shared_ptr<LlmConfig> config) : Llm(config) {
        img_size_ = config->llm_config_.value("img_size", img_size_);
        imgpad_len_ = config->llm_config_.value("imgpad_len", imgpad_len_);
        img_start_ = config->llm_config_.value("img_start", img_start_);
        img_end_ = config->llm_config_.value("img_end", img_end_);
        img_pad_ = config->llm_config_.value("img_pad", img_pad_);
    }
    ~Lvlm() { visual_module_.reset(); }
    virtual void load() override;
private:
    int img_size_ = 448, imgpad_len_ = 256, img_start_ = 151857, img_end_ = 151858, img_pad_ = 151859;
    std::shared_ptr<Module> visual_module_;
    MNN::Express::VARP visual_embedding(const std::vector<int>& input_ids);
    std::vector<int> url_encode(const std::string& url);
    virtual std::vector<int> tokenizer(const std::string& query) override;
    virtual MNN::Express::VARP embedding(const std::vector<int>& input_ids) override;
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
    if (config_->memory() == "low") {
        cpuBackendConfig.memory = BackendConfig::Memory_Low;
    }
    if (config_->precision() == "low") {
        cpuBackendConfig.precision = BackendConfig::Precision_Low;
    }
    config.backendConfig = &cpuBackendConfig;
    ExecutorScope::Current()->setGlobalExecutorConfig(config.type, cpuBackendConfig, config.numThread);

    runtime_manager_.reset(Executor::RuntimeManager::createRuntimeManager(config));
    runtime_manager_->setHint(MNN::Interpreter::MEM_ALLOCATOR_TYPE, 0);
    runtime_manager_->setHint(MNN::Interpreter::DYNAMIC_QUANT_OPTIONS, 1); // 1: per batch quant, 2: per tensor quant
    runtime_manager_->setHint(MNN::Interpreter::KVCACHE_QUANT_OPTIONS, config_->quant_kv()); // 0: no quant, 1: quant key, 2: quant value, 3: quant kv

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
    {
        std::ifstream embedding_bin(config_->embedding_file());
        embedding_bin.close();
    }
    MNN_PRINT("### is_single_ = %d\n", is_single_);
    // 1. load vocab
    MNN_PRINT("load tokenizer\n");
    tokenizer_.reset(Tokenizer::createTokenizer(config_->tokenizer_file()));
    MNN_PRINT("load tokenizer Done\n");
    // 3. load model
    Module::Config module_config;
    module_config.shapeMutable = true;
    module_config.rearrange = true;
    int layer_nums = config_->layer_nums();
    if (is_single_) {
        // load single model
        key_value_shape_.insert(key_value_shape_.begin(), layer_nums);
        modules_.resize(1);
        std::string model_path = config_->llm_model();
        MNN_PRINT("load %s ... ", model_path.c_str());
        runtime_manager_->setExternalFile(config_->llm_weight());
        modules_[0].reset(Module::load(
                                       {"input_ids", "attention_mask", "position_ids", "past_key_values"},
                                       {"logits", "presents"}, model_path.c_str(), runtime_manager_, &module_config));
        MNN_PRINT("Done!\n");
    } else {
        // load split models
        modules_.resize(layer_nums + 2);
        // load lm model
        modules_[layer_nums].reset(Module::load({}, {}, config_->lm_model().c_str(), runtime_manager_, &module_config));
        // load block models
        for (int i = 0; i < layer_nums; i++) {
            std::string model_path = config_->block_model(i);
            MNN_PRINT("load %s ... ", model_path.c_str());
            modules_[i].reset(Module::load(
                                           {"inputs_embeds", "attention_mask", "position_ids", "past_key_values"},
                                           {"hidden_states", "presents"}, model_path.c_str(), runtime_manager_, &module_config));
            MNN_PRINT("Done!\n");
        }
    }
    decode_modules_.resize(modules_.size());
    for (int v=0; v<modules_.size(); ++v) {
        decode_modules_[v].reset(Module::clone(modules_[v].get()));
    }
    prefill_modules_ = modules_;
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
}

VARP Llm::forward(const std::vector<int>& input_ids) {
    int seq_len = input_ids.size();
    auto attention_mask = gen_attention_mask(seq_len);
    auto position_ids = gen_position_ids(seq_len);
    VARP logits;
    if (is_single_) {
        // single model
        auto hidden_states = embedding(input_ids);
        auto outputs = modules_.back()->onForward({hidden_states, attention_mask, position_ids, past_key_values_[0]});
        if (outputs.empty()) {
            return nullptr;
        }
        ExecutorScope::Current()->gc(Executor::FULL);
        logits = outputs[0];
        past_key_values_[0] = outputs[1];
    } else {
        // split block models
        int layer_nums = config_->layer_nums();
        auto hidden_states = embedding(input_ids);
        ExecutorScope::Current()->gc(Executor::FULL);
        for (int i = 0; i < layer_nums; i++) {
            AUTOTIME;
            auto outputs = modules_[i]->onForward({hidden_states, attention_mask, position_ids, past_key_values_[i]});
            hidden_states = outputs[0];
            past_key_values_[i] = outputs[1];
        }
        ExecutorScope::Current()->gc(Executor::FULL);
        {
            AUTOTIME;
            auto outputs = modules_[layer_nums]->onForward({hidden_states});
            logits = outputs[0];
        }
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
        history.emplace_back(std::make_pair("user", user_str));
        auto assistant_str = response(history);
        history.emplace_back(std::make_pair("assistant", assistant_str));
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
}

std::vector<int> Llm::generate(const std::vector<int>& input_ids, int max_new_tokens) {
    generate_init();
    std::vector<int> output_ids, all_ids = input_ids;
    prompt_len_ = static_cast<int>(input_ids.size());
    if (max_new_tokens < 0) { max_new_tokens = config_->max_new_tokens(); }
    // prefill
    auto logits = forward(input_ids);
    if (logits.get() == nullptr) {
        return {};
    }
    int token = sample(logits, all_ids);
    output_ids.push_back(token);
    all_ids.push_back(token);
    // decode
    while (gen_seq_len_ < max_new_tokens) {
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
    prompt_len_ = static_cast<int>(input_ids.size());
    history_ids_.insert(history_ids_.end(), input_ids.begin(), input_ids.end()); // push to history_ids_
    auto st = std::chrono::system_clock::now();
    modules_ = prefill_modules_;
    auto logits = forward(input_ids);
    if (nullptr == logits.get()) {
        return "";
    }
    int token = sample(logits, history_ids_);
    auto et = std::chrono::system_clock::now();
    modules_ = decode_modules_;
    std::string output_str = decode(token);
    prefill_us_ = std::chrono::duration_cast<std::chrono::microseconds>(et - st).count();
    *os << output_str << std::flush;
    while (gen_seq_len_ < config_->max_new_tokens()) {
        st = std::chrono::system_clock::now();
        history_ids_.push_back(token);
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
        auto word = decode(token);
        *os << word << std::flush;
        output_str += word;
    }
#ifdef DUMP_PROFILE_INFO
    print_speed();
#endif
    return output_str;
}

std::vector<int> Llm::tokenizer(const std::string& user_content) {
    auto prompt = apply_prompt_template(user_content);
    auto input_ids = tokenizer_->encode(prompt);
    return input_ids;
}

std::string Llm::response(const std::string& user_content, std::ostream* os, const char* end_with) {
    generate_init();
    if (!end_with) { end_with = "\n"; }
    auto prompt = apply_prompt_template(user_content);
    if (config_->reuse_kv() && all_seq_len_ > 0) {
        prompt = "<|im_end|>\n" + prompt;
    }
    auto input_ids = tokenizer_->encode(prompt);
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
    std::cout << "# prompt : " << prompt << std::endl;
    auto input_ids = tokenizer_->encode(prompt);
    printf("input_ids (%lu): ", input_ids.size()); for (auto id : input_ids) printf("%d, ", id); printf("\n");
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
    // disk embedding to save memory
    int hidden_size = config_->hidden_size();
    int seq_len = static_cast<int>(input_ids.size());
    if (needNewVar(inputs_embeds_, 0, seq_len)) {
        inputs_embeds_ = _Input({seq_len, 1, hidden_size}, NCHW);
    }

    size_t size = hidden_size * sizeof(int16_t);
    FILE* file = fopen(config_->embedding_file().c_str(), "rb");
    std::unique_ptr<int16_t[]> buffer(new int16_t[hidden_size]);
    for (size_t i = 0; i < seq_len; i++) {
        fseek(file, input_ids[i] * size, SEEK_SET);
        fread(buffer.get(), 1, size, file);
        auto ptr = inputs_embeds_->writeMap<int16_t>() + i * hidden_size * 2;
        for (int j = 0; j < hidden_size; j++) {
            ptr[j * 2] = 0;
            ptr[j * 2 + 1] = buffer[j];
        }
    }
    fclose(file);
    return inputs_embeds_;
}

std::string Llm::decode(int id) {
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
    visual_module_.reset(Module::load({}, {}, config_->visual_model().c_str(), runtime_manager_, &module_config));
}

std::vector<int> Lvlm::url_encode(const std::string& url) {
    std::vector<int> ascii_values(imgpad_len_ + 2, img_pad_);
    ascii_values[0] = img_start_;
    ascii_values[imgpad_len_ + 1] = img_end_;
    for (int i = 0; i < url.size(); i++) {
        ascii_values[i + 1] = static_cast<int>(url[i]);
    }
    return ascii_values;
}

std::vector<int> Lvlm::tokenizer(const std::string& query) {
    auto prompt = apply_prompt_template(query);
    // split query
    std::regex img_regex("<img>(.*?)</img>");
    std::string::const_iterator searchStart(prompt.cbegin());
    std::smatch match;
    std::vector<std::string> img_info, txt_info;
    std::vector<int> ids {};
    while (std::regex_search(searchStart, prompt.cend(), match, img_regex)) {
        std::cout << match[1].str() << std::endl;
        auto txt_ids = tokenizer_->encode(match.prefix().str());
        ids.insert(ids.end(), txt_ids.begin(), txt_ids.end());
        auto img_ids = url_encode(match[1].str());
        ids.insert(ids.end(), img_ids.begin(), img_ids.end());
        searchStart = match.suffix().first;
    }
    if (searchStart != prompt.cend()) {
        auto txt_ids = tokenizer_->encode(std::string(searchStart, prompt.cend()));
        ids.insert(ids.end(), txt_ids.begin(), txt_ids.end());
    }
    return ids;
}

VARP Lvlm::embedding(const std::vector<int>& input_ids) {
#ifdef USING_VISUAL_MODEL
    int start_pos = 0, pad_pos = 0, end_pos = 0;
    for (int i = 0; i < input_ids.size(); i++) {
        int id = input_ids[i];
        if (id == img_start_ && !start_pos) {
            start_pos = i;
        }
        if (id == img_pad_ && !pad_pos) {
            pad_pos = i;
        }
        if (id == img_end_ && !end_pos) {
            end_pos = i;
        }
    }
    if (!start_pos) {
        return Llm::embedding(input_ids);
    }
    std::vector<int> prefix(input_ids.begin(), input_ids.begin() + start_pos + 1);
    std::vector<int> img_ascii(input_ids.begin() + start_pos + 1, input_ids.begin() + pad_pos);
    std::vector<int> suffix(input_ids.begin() + end_pos, input_ids.end());
    std::string img_path;
    for (auto ascii_val : img_ascii) {
        img_path += static_cast<char>(ascii_val);
    }
    VARP image = nullptr;
    if (img_path.substr(0, 4) == "http") {
        std::regex url_regex(R"(^https?://([^/]+)(/.*))");
        std::smatch url_match_result;
        std::string host, path;
        if (std::regex_search(img_path, url_match_result, url_regex) && url_match_result.size() == 3) {
            host = url_match_result[1].str();
            path = url_match_result[2].str();
        }
        std::cout << host << "#" << path << std::endl;
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
        image = MNN::CV::imread(img_path);
    }
    image = MNN::CV::resize(image, {img_size_, img_size_}, 0, 0, MNN::CV::INTER_LINEAR, MNN::CV::COLOR_BGR2RGB,
                            {123.25239296, 117.20384, 104.50194688}, {0.0145414 , 0.01494914, 0.01416452});
    image = MNN::Express::_Unsqueeze(image, {0});
    image = MNN::Express::_Convert(image, NC4HW4);
    auto image_embedding = visual_module_->forward(image);
    image_embedding = MNN::Express::_Permute(image_embedding, {1, 0, 2});
    auto prefix_embedding = Llm::embedding(prefix);
    auto suffix_embedding = Llm::embedding(suffix);
    auto embeddings = MNN::Express::_Concat({prefix_embedding, image_embedding, suffix_embedding}, 0);
#else
    auto embeddings = Llm::embedding(input_ids);
#endif
    return embeddings;
}
// Llm end

// Embedding start
float Embedding::dist(VARP var0, VARP var1) {
    auto distVar = _Sqrt(_ReduceSum(_Square(var0 - var1)));
    auto dist = distVar->readMap<float>()[0];
    return dist;
}

Embedding* Embedding::createEmbedding(const std::string& config_path) {
    std::shared_ptr<LlmConfig> config(new LlmConfig(config_path));
    Embedding* embedding = new Embedding(config);
    embedding->load();
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

VARP Embedding::embedding(const std::string& txt) {
    auto ids = tokenizer(txt);
    int prompt_len = ids.size();
    auto inputs_ids = _Const(ids.data(), {prompt_len}, NCHW, halide_type_of<int>());
    auto attention_mask = gen_attention_mask(prompt_len);
    auto position_ids = gen_position_ids(prompt_len);
    auto outputs = modules_[0]->onForward({inputs_ids, attention_mask, position_ids});
    auto sentence_embeddings = outputs[0];
    return sentence_embeddings;
}

std::vector<int> Embedding::tokenizer(const std::string& query) {
    auto prompt = query;
    if (query.size() <= 256) {
        prompt = "为这个句子生成表示以用于检索相关文章：" + query;
    }
    prompt = apply_prompt_template(prompt);
    auto ids = tokenizer_->encode(prompt);
    return ids;
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
