//
//  llmconfig.hpp
//
//  Created by MNN on 2024/07/19.
//  ZhaodeWang
//

#include "rapidjson/document.h"
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>

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
                rapidjson::Document::AllocatorType& allocator) {
    if (!source.IsObject() || !destination.IsObject()) {
        return false;
    }

    for (auto it = source.MemberBegin(); it != source.MemberEnd(); ++it) {
        const char* key = it->name.GetString();
        if (destination.HasMember(key)) {
            if (destination[key].IsObject() && it->value.IsObject()) {
                // Recursively merge the two JSON objects
                merge_json(destination[key], it->value, allocator);
            } else {
                // Overwrite the value in the destination
                destination[key].CopyFrom(it->value, allocator);
            }
        } else {
            // Add the value to the destination
            rapidjson::Value newKey(key, allocator);
            rapidjson::Value newValue;
            newValue.CopyFrom(it->value, allocator);
            destination.AddMember(newKey, newValue, allocator);
        }
    }
    return true;
}

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
            return false;
        }
        // merge
        rapidjson::Document::AllocatorType& allocator = document.GetAllocator();
        return merge_json(document, input_doc, allocator);
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
    std::string value(const char key[], const char default_value[]) const {
        return value(key, std::string(default_value));
    }
};

class LlmConfig {
public:
    std::string base_dir_;
    rapid_json_wrapper config_, llm_config_, mllm_config_, cur_config_;
    LlmConfig() {}
    LlmConfig(const LlmConfig& other)
        : base_dir_(other.base_dir_),
          config_(other.config_),
          llm_config_(other.llm_config_),
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
            llm_config_ = rapid_json_wrapper::parse(llm_config_file);
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

    std::string audio_model() const {
        return base_dir_ + config_.value("audio_model", "audio.mnn");
    }
    // model file config end >

    // < generate config start
    int max_new_tokens() const {
        return config_.value("max_new_tokens", 512);
    }

    bool reuse_kv() const {
        return config_.value("reuse_kv", false);
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

    int quant_qkv() const {
        return config_.value("quant_qkv", 0);
    }

    int kvcache_limit() const {
        return config_.value("kvcache_limit", -1);
    }
    // backend config end >

    // < llm model config start
    bool is_single() const {
        return llm_config_.value("is_single", true);
    }

    bool is_visual() const {
        return llm_config_.value("is_visual", false);
    }

    bool is_audio() const {
        return llm_config_.value("is_audio", false);
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
    bool kvcache_mmap() const {
        return config_.value("kvcache_mmap", false);
    }
    std::string tmp_path() const {
        return config_.value("tmp_path", "");
    }

    int hidden_size() const {
        return llm_config_.value("hidden_size", 4096);
    }

    int layer_nums() const {
        return llm_config_.value("layer_nums", 32);
    }

    std::vector<int> key_value_shape() const {
        return llm_config_.value("key_value_shape", std::vector<int>{});
    }

    std::string attention_mask() const {
        return llm_config_.value("attention_mask", "int");
    }
    bool attention_fused() const {
        return llm_config_.value("attention_fused", true);
    }

    std::string chat_template() const {
        return llm_config_.value("chat_template", "");
    }

    std::string prompt_template() const {
        return llm_config_.value("prompt_template", "");
    }

    std::vector<int64_t> tie_embeddings() const {
        return llm_config_.value("tie_embeddings", std::vector<int64_t>{});
    }
    // llm model config end >
};
} // Transformer
} // MNN
