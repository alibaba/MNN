//
//  llm.hpp
//
//  Created by MNN on 2023/08/25.
//  ZhaodeWang
//

#ifndef LLM_hpp
#define LLM_hpp

#include <vector>
#include <memory>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <streambuf>
#include <functional>
#include <unordered_map>

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/MathOp.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>
#include "tokenizer.hpp"
#include "rapidjson/document.h"

using namespace MNN;
using namespace Express;
using namespace rapidjson;
class Tokenizer;
class Pipeline;

// Llm start
// llm stream buffer with callback
class LlmStreamBuffer : public std::streambuf {
public:
    using CallBack = std::function<void(const char* str, size_t len)>;;
    LlmStreamBuffer(CallBack callback) : callback_(callback) {}

protected:
    virtual std::streamsize xsputn(const char* s, std::streamsize n) override {
        if (callback_) {
            callback_(s, n);
        }
        return n;
    }

private:
    CallBack callback_ = nullptr;
};

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

class rapid_json_wrapper {
public:
    Document document;
    rapid_json_wrapper() {}
    rapid_json_wrapper(Document doc) : document(std::move(doc)) {}
    static rapid_json_wrapper parse(const std::ifstream& ifile) {
        std::ostringstream ostr;
        ostr << ifile.rdbuf();
        Document document;
        document.Parse(ostr.str().c_str());
        rapid_json_wrapper json_wrapper(std::move(document));
        return json_wrapper;
    }
    static rapid_json_wrapper parse(const char* str) {
        Document document;
        document.Parse(str);
        rapid_json_wrapper json_wrapper(std::move(document));
        return json_wrapper;
    }

    template <typename T>
    T value(const char* key, const T& defualt_value) const {
        if (document.HasMember(key)) {
            const auto& value = document[key];
            if constexpr (std::is_same<T, int>::value) {
                if (value.IsInt()) return value.GetInt();
            } else if constexpr (std::is_same<T, std::string>::value || std::is_same<T, const char*>::value) {
                if (value.IsString()) return value.GetString();
            } else if constexpr (std::is_same<T, bool>::value) {
                if (value.IsBool()) return value.GetBool();
            } else if constexpr (std::is_same<T, std::vector<int>>::value) {
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
        }
        return defualt_value;
    }
    std::string value(const char key[], const char defualt_value[]) const {
        return value(key, std::string(defualt_value));
    }
};

class LlmConfig {
public:
    std::string base_dir_;
    rapid_json_wrapper config_, llm_config_;
    LlmConfig() {}
    LlmConfig(const std::string& path) {
        // load config
        if (has_suffix(path, ".json")) {
            std::ifstream config_file(path);
            if (config_file.is_open()) {
                config_ = rapid_json_wrapper::parse(config_file);
            } else {
                std::cerr << "Unable to open config file: " << path << std::endl;
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
    // model file config end >

    // < generate config start
    int max_new_tokens() const {
        return config_.value("max_new_tokens", 512);
    }
    // generate config end >

    // < backend config start
    std::string backend_type() const {
        return config_.value("backend_type", "cpu");
    }

    int thread_num() const {
        return config_.value("thread_num", 4);
    }

    std::string precision() const {
        return config_.value("precision", "low");
    }

    std::string memory() const {
        return config_.value("memory", "low");
    }
    // backend config end >

    // < llm model config start
    bool is_single() const {
        return llm_config_.value("is_single", true);
    }

    bool is_visual() const {
        return llm_config_.value("is_visual", false);
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

    std::string prompt_template() const {
        return llm_config_.value("prompt_template", "");
    }
    // llm model config end >
};

class MNN_PUBLIC Llm {
public:
    Llm(std::shared_ptr<LlmConfig> config) : config_(config) {}
    virtual ~Llm();
    static Llm* createLLM(const std::string& config_path);
    void chat();
    void trace(bool start);
    virtual void load();
    VARP forward(const std::vector<int>& input_ids);
    int sample(VARP logits, const std::vector<int>& pre_ids);
    std::string apply_chat_template(const std::string& input_str) const;
    std::string response(const std::string& input_str, std::ostream* os = &std::cout, const char* end_with = nullptr);
    void generate_init();
    std::string generate(const std::vector<int>& input_ids, std::ostream* os, const char* end_with);
    std::vector<int> generate(const std::vector<int>& input_ids, int max_new_tokens = -1);
    void print_speed();
    friend class Pipeline;
public:
    // forward info
    int prompt_len_ = 0;
    int gen_seq_len_ = 0;
    int all_seq_len_ = 0;
    // time
    int64_t prefill_us_ = 0;
    int64_t decode_us_ = 0;
    bool is_single_ = true;
    std::shared_ptr<LlmConfig> config_;
    std::unique_ptr<Tokenizer> tokenizer_;
protected:
    std::vector<int> key_value_shape_ = {};
    std::vector<VARP> past_key_values_;
    VARP inputs_embeds_, attention_mask_, position_ids_;
    std::shared_ptr<Executor::RuntimeManager> runtime_manager_;
    std::vector<std::shared_ptr<Module>> modules_;
    std::vector<std::shared_ptr<Module>> decode_modules_;
    std::vector<std::shared_ptr<Module>> prefill_modules_;
    void init_runtime();
    std::string decode(int id);
    bool is_stop(int token_id);
    virtual std::vector<int> tokenizer(const std::string& query);
    virtual VARP embedding(const std::vector<int>& input_ids);
    virtual VARP gen_attention_mask(int seq_len);
    virtual VARP gen_position_ids(int seq_len);
};

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
    VARP visual_embedding(const std::vector<int>& input_ids);
    std::vector<int> url_encode(const std::string& url);
    virtual std::vector<int> tokenizer(const std::string& query) override;
    virtual VARP embedding(const std::vector<int>& input_ids) override;
};
// Llm end

// Embedding start
class Embedding : public Llm {
public:
    Embedding(std::shared_ptr<LlmConfig> config) : Llm(config) {}
    static Embedding* createEmbedding(const std::string& config_path);
    static float dist(VARP var0, VARP var1);
    virtual void load() override;
    VARP embedding(const std::string& txt);
    int dim() { return config_->hidden_size(); }
private:
    virtual std::vector<int> tokenizer(const std::string& query) override;
    virtual VARP gen_attention_mask(int seq_len) override;
    virtual VARP gen_position_ids(int seq_len) override;
};
// Embedding end

#endif // LLM_hpp
