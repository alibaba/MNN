//
//  llmconfig.hpp
//
//  Created by MNN on 2024/07/19.
//  ZhaodeWang
//

#ifndef LLMCONFIG_Hpp
#define LLMCONFIG_Hpp

#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>
#include <MNN/MNNDefine.h>


namespace MNN {
namespace Transformer {

static inline bool has_suffix(const std::string& str, const std::string& suffix) {
    return str.size() >= suffix.size() &&
    str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static inline std::string base_dir(const std::string& path) {
    size_t pos = path.find_last_of("/\\");
    if (pos == std::string::npos) {
        return "./";
    } else {
        return path.substr(0, pos + 1);
    }
}

static inline std::string file_name(const std::string& path) {
    size_t pos = path.find_last_of("/\\");
    if (pos == std::string::npos) {
        return path;
    } else {
        return path.substr(pos + 1);
    }
}

bool merge_json(rapidjson::Value& destination, const rapidjson::Value& source,
                rapidjson::Document::AllocatorType& allocator);

class rapid_json_wrapper {
public:
    rapidjson::Document document;
    rapid_json_wrapper() {}
    rapid_json_wrapper(rapidjson::Document doc) : document(std::move(doc)) {}
    rapid_json_wrapper(const rapid_json_wrapper &other) {
        document.CopyFrom(other.document, document.GetAllocator());
    }
    rapid_json_wrapper& operator=(const rapid_json_wrapper& other) {
        if (this != &other) {
            document.SetObject();
            document.CopyFrom(other.document, document.GetAllocator());
        }
        return *this;
    }
    rapid_json_wrapper(rapid_json_wrapper&& other) noexcept : document(std::move(other.document)) {}
    rapid_json_wrapper& operator=(rapid_json_wrapper&& other) noexcept {
        if (this != &other) {
            document.SetObject();
            document.GetAllocator().Clear();
            document = std::move(other.document);
        }
        return *this;
    }
    static rapid_json_wrapper parse(const std::ifstream& ifile) {
        std::ostringstream ostr;
        ostr << ifile.rdbuf();
        rapidjson::Document document;
        document.Parse(ostr.str().c_str());
        rapid_json_wrapper json_wrapper(std::move(document));
        return json_wrapper;
    }
    static rapid_json_wrapper parse(const char* str) {
        rapidjson::Document document;
        document.Parse(str);
        rapid_json_wrapper json_wrapper(std::move(document));
        return json_wrapper;
    }
    bool empty() { return document.IsNull(); }
    bool merge(const char* str) {
        rapidjson::Document input_doc;
        input_doc.Parse(str);
        if (input_doc.HasParseError()) {
            MNN_PRINT("Config Parse Error: %d\n", input_doc.GetParseError());
            return false;
        }
        // merge
        rapidjson::Document::AllocatorType& allocator = document.GetAllocator();
        return merge_json(document, input_doc, allocator);
    }
    bool merge_and_clear(rapid_json_wrapper& source_) {
        // rapid_json_wrapper has document object
        rapidjson::Value& source = source_.document;
        rapidjson::Value& destination = this->document;
        rapidjson::Document::AllocatorType& allocator = document.GetAllocator();

        for (auto it = source.MemberBegin(); it != source.MemberEnd(); ++it) {
            const char* key = it->name.GetString();
            rapidjson::Value newKey(key, allocator);
            rapidjson::Value newValue;
            newValue.CopyFrom(it->value, allocator);
            destination.AddMember(newKey, newValue, allocator);
        }

        // clear source content
        source.SetNull();
        return true;
    }
    std::string dump() {
        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        document.Accept(writer);
        return buffer.GetString();
    }
    // read value
    rapid_json_wrapper value(const char* key) const {
        if (document.HasMember(key)  && document[key].IsObject()) {
            rapidjson::Document subDoc;
            subDoc.CopyFrom(document[key], subDoc.GetAllocator());
            return rapid_json_wrapper(std::move(subDoc));
        }
        return rapid_json_wrapper();
    }
    float value(const char* key, const float& default_value) const {
        if (document.HasMember(key)) {
            const auto& value = document[key];
            if (value.IsFloat()) return value.GetFloat();
        }
        return default_value;
    }
    int value(const char* key, const int& default_value) const {
        if (document.HasMember(key)) {
            const auto& value = document[key];
            if (value.IsInt()) return value.GetInt();
        }
        return default_value;
    }
    bool value(const char* key, const bool& default_value) const {
        if (document.HasMember(key)) {
            const auto& value = document[key];
            if (value.IsBool()) return value.GetBool();
        }
        return default_value;
    }
    std::string value(const char* key, const std::string& default_value) const {
        if (document.HasMember(key)) {
            const auto& value = document[key];
            if (value.IsString()) return value.GetString();
        }
        return default_value;
    }
    std::vector<int64_t> value(const char* key, const std::vector<int64_t>& default_value) const {
        if (document.HasMember(key)) {
            const auto& value = document[key];
            if (value.IsArray()) {
                std::vector<int64_t> result;
                for (auto& v : value.GetArray()) {
                    result.push_back(v.GetInt64());
                }
                return result;
            }
        }
        return default_value;
    }
    std::vector<int> value(const char* key, const std::vector<int>& default_value) const {
        if (document.HasMember(key)) {
            const auto& value = document[key];
            if (value.IsArray()) {
                std::vector<int> result;
                for (auto& v : value.GetArray()) {
                    if (v.IsInt()) {
                        result.push_back(v.GetInt());
                    }
                }
                return result;
            }
        }
        return default_value;
    }
    std::vector<float> value(const char* key, const std::vector<float>& default_value) const {
        if (document.HasMember(key)) {
            const auto& value = document[key];
            if (value.IsArray()) {
                std::vector<float> result;
                for (auto& v : value.GetArray()) {
                    if (v.IsFloat()) {
                        result.push_back(v.GetFloat());
                    }
                }
                return result;
            }
        }
        return default_value;
    }
    std::vector<std::string> value(const char* key, const std::vector<std::string>& default_value) const {
        if (document.HasMember(key)) {
            const auto& value = document[key];
            if (value.IsArray()) {
                std::vector<std::string> result;
                for (auto& v : value.GetArray()) {
                    if (v.IsString()) {
                        result.push_back(v.GetString());
                    }
                }
                return result;
            }
        }
        return default_value;
    }
    std::string value(const char key[], const char default_value[]) const {
        return value(key, std::string(default_value));
    }
};

class LlmConfig {
public:
    std::string base_dir_;
    rapid_json_wrapper config_, mllm_config_, cur_config_;
    LlmConfig() {}
    LlmConfig(const LlmConfig& other)
        : base_dir_(other.base_dir_),
          config_(other.config_),
          mllm_config_(other.mllm_config_),
          cur_config_(other.cur_config_) {}
    LlmConfig(const std::string& path) {
        // load config
        if (has_suffix(path, ".json")) {
            std::ifstream config_file(path);
            if (config_file.is_open()) {
                config_ = rapid_json_wrapper::parse(config_file);
            } else {
                std::cerr << "Unable to open config file: " << path << std::endl;
                std::cerr << "Error: " << std::strerror(errno) << " (errno: " << errno << ")" << std::endl;
            }
            base_dir_ = base_dir(path);
        } else {
            // compatibility with the original usage
            if (has_suffix(path, ".mnn")) {
                auto model_name = file_name(path);
                std::string json_str = R"({
                    "llm_model": ")" + model_name + R"(",
                    "llm_weight": ")" + model_name + R"(.weight"
                })";
                config_ = rapid_json_wrapper::parse(json_str.c_str());
                base_dir_ = base_dir(path);
            } else {
                const char* json_cstr = "{}";
                config_ = rapid_json_wrapper::parse(json_cstr);
                base_dir_ = path;
            }
        }
        // using config's base_dir
        base_dir_ = config_.value("base_dir", base_dir_);
        // load llm_config for model info
        std::ifstream llm_config_file(llm_config());
        if (llm_config_file.is_open()) {
            auto llm_config_ = rapid_json_wrapper::parse(llm_config_file);
            config_.merge_and_clear(llm_config_);
        } else {
            std::cerr << "Unable to open llm_config file: " << llm_config() << std::endl;
        }
        mllm_config_ = config_.value("mllm");
    }

    // < model file config start
    std::string llm_config() const {
        return base_dir_ + config_.value("llm_config", "llm_config.json");
    }

    std::string llm_model() const {
        return base_dir_ + config_.value("llm_model", "llm.mnn");
    }

    std::string llm_weight() const {
        return base_dir_ + config_.value("llm_weight", "llm.mnn.weight");
    }

    std::string block_model(int index) const {
        return base_dir_ + config_.value("block_model", "block_") + std::to_string(index) + ".mnn";
    }

    std::string lm_model() const {
        return base_dir_ + config_.value("lm_model", "lm.mnn");
    }

    std::string embedding_model() const {
        return base_dir_ + config_.value("embedding_model", "embedding.mnn");
    }

    std::string embedding_file() const {
        return base_dir_ + config_.value("embedding_file", "embeddings_bf16.bin");
    }

    std::string tokenizer_file() const {
        return base_dir_ + config_.value("tokenizer_file", "tokenizer.txt");
    }

    std::string visual_model() const {
        return base_dir_ + config_.value("visual_model", "visual.mnn");
    }

    std::string npu_model_dir() const {
        return base_dir_ + config_.value("npu_model_dir", "");
    }

    std::string audio_model() const {
        return base_dir_ + config_.value("audio_model", "audio.mnn");
    }

    std::string context_file() const {
        return base_dir_ + config_.value("context_file", "context.json");
    }
    // model file config end >

    // < generate config start
    int max_all_tokens() const {
        return config_.value("max_all_tokens", 2048);
    }

    int max_new_tokens() const {
        return config_.value("max_new_tokens", 512);
    }

    bool reuse_kv() const {
        return config_.value("reuse_kv", false);
    }

    bool all_logits() const {
        return config_.value("all_logits", false);
    }
    // generate config end >

    // < backend config start
    std::string backend_type(bool mllm = false) const {
        if (mllm) return mllm_config_.value("backend_type", "cpu");
        return config_.value("backend_type", "cpu");
    }

    int thread_num(bool mllm = false) const {
        if (mllm) return mllm_config_.value("thread_num", 4);
        return config_.value("thread_num", 4);
    }

    std::string precision(bool mllm = false) const {
        if (mllm) return mllm_config_.value("precision", "low");
        return config_.value("precision", "low");
    }
    std::string power(bool mllm = false) const {
        if (mllm) return mllm_config_.value("power", "normal");
        return config_.value("power", "normal");
    }

    std::string memory(bool mllm = false) const {
        if (mllm) return mllm_config_.value("memory", "low");
        return config_.value("memory", "low");
    }
    // backend config end >

    // talker config start
    std::string talker_model() const {
        return base_dir_ + config_.value("talker_model", "talker.mnn");
    }

    std::string talker_weight() const {
        return base_dir_ + config_.value("talker_weight", "talker.mnn.weight");
    }

    std::string talker_embedding_file() const {
        return base_dir_ + config_.value("talker_embedding_file", "talker_embeddings_bf16.bin");
    }

    std::string predit_model() const {
        return base_dir_ + config_.value("predit_model", "predit.mnn");
    }

    std::string dit_model() const {
        return base_dir_ + config_.value("dit_model", "dit.mnn");
    }

    std::string bigvgan_model() const {
        return base_dir_ + config_.value("bigvgan_model", "bigvgan.mnn");
    }

    std::string spk_dict() const {
        return base_dir_ + config_.value("spk_dict", "spk_dict.mnn");
    }

    int talker_max_new_tokens() const {
        return config_.value("talker_max_new_tokens", 2048);
    }

    std::string talker_speaker() const {
        // Chelsie or Ethan
        return config_.value("talker_speaker", "Chelsie");
    }

    int dit_steps() const {
        return config_.value("dit_steps", 5);
    }

    int dit_solver() const {
        // 1: OED, 4: RungeKutta4ODE
        return config_.value("dit_solver", 1);
    }
    // talker config end

    // < llm model config start
    bool is_single() const {
        return config_.value("is_single", true);
    }

    bool is_visual() const {
        return config_.value("is_visual", false);
    }

    bool is_audio() const {
        return config_.value("is_audio", false);
    }

    bool is_mrope() const {
        return config_.value("is_mrope", false);
    }

    bool has_talker() const {
        return config_.value("has_talker", false);
    }

    bool has_deepstack() const {
        return config_.value("has_deepstack", false);
    }

    bool use_template() const {
        return config_.value("use_template", true);
    }

    bool use_mmap() const {
        return config_.value("use_mmap", false);
    }
    bool use_cached_mmap() const {
        return config_.value("use_cached_mmap", true);
    }
    int dynamic_option() const {
        return config_.value("dynamic_option", 0);
    }
    bool kvcache_mmap() const {
        return config_.value("kvcache_mmap", false);
    }
    std::string tmp_path() const {
        return config_.value("tmp_path", "");
    }

    std::string prefix_cache_path() const {
        return config_.value("prefix_cache_path", "prefixcache");
    }

    std::string system_prompt() const {
        return config_.value("system_prompt", "");
    }

    int hidden_size() const {
        return config_.value("hidden_size", 4096);
    }

    int layer_nums() const {
        return config_.value("layer_nums", 32);
    }

    std::vector<int> key_value_shape() const {
        return config_.value("key_value_shape", std::vector<int>{});
    }

    std::string attention_mask() const {
        return config_.value("attention_mask", "int");
    }

    std::string attention_type() const {
        return config_.value("attention_type", "full");
    }

    int sliding_window() const {
        return config_.value("sliding_window", 0);
    }

    bool attention_fused() const {
        return config_.value("attention_fused", true);
    }

    std::string bos() const {
        return config_.value("bos", "");
    }
    std::string system_prompt_template() const {
        return config_.value("system_prompt_template", "<|im_start|>system\n%s<|im_end|>\n");
    }
    std::string user_prompt_template() const {
        return config_.value("user_prompt_template", "<|im_start|>user\n%s<|im_end|>\n");
    }
    std::string assistant_prompt_template() const {
        return config_.value("assistant_prompt_template", "<|im_start|>assistant\n%s<|im_end|>\n");
    }

    // for compatibility with the original usage
    std::string chat_template() const {
        return config_.value("chat_template", "");
    }

    std::string prompt_template() const {
        return config_.value("prompt_template", "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n");
    }

    std::vector<int64_t> tie_embeddings() const {
        return config_.value("tie_embeddings", std::vector<int64_t>{});
    }
    // llm model config end >

    // < sampler config start
    std::string sampler_type() const {
        return config_.value("sampler_type", "greedy");
    }

    std::vector<std::string> mixed_samplers() const {
        return config_.value("mixed_samplers", std::vector<std::string>({"topK", "tfs", "typical", "topP", "min_p", "temperature"}));
    }

    float temperature() const {
        return config_.value("temperature", 1.0f);
    }

    int topK() const {
        return config_.value("topK", 40);
    }

    float topP() const {
        return config_.value("topP", 0.9f);
    }

    float minP() const {
        return config_.value("minP", 0.1f);
    }

    float tfsZ() const {
        return config_.value("tfsZ", 1.0f);
    }

    float typical() const {
        return config_.value("typical", 1.0f);
    }

    float penalty() const {
        return config_.value("penalty", 0.0f);
    }

    int ngram() const {
        return config_.value("n_gram", 8);
    }

    float ngram_factor() const {
        return config_.value("ngram_factor", 1.0f);
    }

    std::string penalty_sampler() const {
        return config_.value("penalty_sampler", "greedy");
    }
    // sampler config end >

    // < speculative decoding config start

    /**
     speculative decoding algrithm.
     optional: "lookahead"、 ”mtp“、 "draftmodel", "eagle"
     */
    std::string speculative_type() const {
        return config_.value("speculative_type", "");
    }

    // speculative draft length
    int draft_predict_length() const {
        return config_.value("draft_predict_length", 3);
    }
    /**
     if speculative_type is set "lookahead",
     purpose: :draft filter and adopt strictness,
     optional: "low" "medium" "high"
     */
    // ========= lookahead config start ===============
    std::string draft_match_strictness() const {
        return config_.value("draft_match_strictness", "low");
    }
    /**
     if speculative_type is set "lookahead",
     purpose: deal if have several draft matchs, how to select one?
     optional 0: "freqxlen" -> draft frequency multiply draft length as metrics, the higher the better
     optional 1: "fcfs" -> first come fiirst serve,  just select the first match draft
     */
    std::string draft_selection_rule() const {
        return config_.value("draft_selection_rule", "freqxlen");
    }
    /**
     if speculative_type is set "lookahead",
     purpose:  lookup prompt, how long history token should match
     */
    int ngram_match_maxlen() const {
        return config_.value("ngram_match_maxlen", 4);
    }
    /**
     if speculative_type is set "lookahead",
     if user have prior knowledge base file, please set path
     */
    std::string lookup_file() const {
        return base_dir_ + config_.value("lookup_file", "lookup_file.txt");
    }
    /**
     if speculative_type is set "lookahead",
     whether should  add decode token to ngram
     */
    bool ngram_update() const {
        return config_.value("ngram_update", false);
    }
    // ========= lookahead config start ===============

    /**
     if speculative_type is set "draftmodel", please set draft model path
     */
    std::string draft_model() const {
        return base_dir_ + config_.value("draft_model", "");
    }
    std::string mtp_model() const {
        return base_dir_ + config_.value("mtp_model", "mtp.mnn");
    }
    std::string eagle_model() const {
        return base_dir_ + config_.value("eagle_model", "eagle.mnn");
    }
    std::string eagle_fc() const {
        return base_dir_ + config_.value("eagle_fc", "eagle_fc.mnn");
    }
    std::string eagle_d2t() const {
        return base_dir_ + config_.value("eagle_d2t", "eagle_d2t.mnn");
    }
    int eagle_depth() const {
        return config_.value("eagle_depth", 3);
    }
    int eagle_topk() const {
        return config_.value("eagle_topk", 1);
    }
    // speculative decoding config end >
};
} // Transformer
} // MNN

#endif
