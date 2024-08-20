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

namespace MNN {
namespace Transformer {
class Tokenizer;
class Pipeline;
class LlmConfig;

// Llm start
// llm stream buffer with callback
class MNN_PUBLIC LlmStreamBuffer : public std::streambuf {
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
class MNN_PUBLIC Llm {
    using PromptItem = std::pair<std::string, std::string>; // <role, content>
public:
    Llm(std::shared_ptr<LlmConfig> config) : config_(config) {}
    virtual ~Llm();
    static Llm* createLLM(const std::string& config_path);
    void chat();
    void reset();
    void trace(bool start);
    virtual void load();
    MNN::Express::VARP forward(const std::vector<int>& input_ids);
    int sample(MNN::Express::VARP logits, const std::vector<int>& pre_ids);
    std::string apply_prompt_template(const std::string& user_content) const;
    std::string apply_chat_template(const std::vector<PromptItem>& chat_prompts) const;
    std::string response(const std::string& user_content, std::ostream* os = &std::cout, const char* end_with = nullptr);
    std::string response(const std::vector<PromptItem>& chat_prompts, std::ostream* os = &std::cout, const char* end_with = nullptr);
    void generate_init();
    std::string generate(const std::vector<int>& input_ids, std::ostream* os, const char* end_with);
    std::vector<int> generate(const std::vector<int>& input_ids, int max_new_tokens = -1);
    void print_speed();
    // config function
    std::string dump_config();
    bool set_config(const std::string& content);
    friend class Pipeline;
public:
    // forward info
    int prompt_len_ = 0;
    int gen_seq_len_ = 0;
    int all_seq_len_ = 0;
    std::vector<int> history_ids_;
    // time
    int64_t prefill_us_ = 0;
    int64_t decode_us_ = 0;
    bool is_single_ = true;
protected:
    std::shared_ptr<LlmConfig> config_;
    std::shared_ptr<Tokenizer> tokenizer_;
    std::vector<int> key_value_shape_ = {};
    std::vector<MNN::Express::VARP> past_key_values_;
    MNN::Express::VARP inputs_embeds_, attention_mask_, position_ids_;
    std::shared_ptr<MNN::Express::Executor::RuntimeManager> runtime_manager_;
    std::vector<std::shared_ptr<MNN::Express::Module>> modules_;
    std::vector<std::shared_ptr<MNN::Express::Module>> decode_modules_;
    std::vector<std::shared_ptr<MNN::Express::Module>> prefill_modules_;
    void init_runtime();
    std::string decode(int id);
    bool is_stop(int token_id);
    virtual std::vector<int> tokenizer(const std::string& query);
    virtual MNN::Express::VARP embedding(const std::vector<int>& input_ids);
    virtual MNN::Express::VARP gen_attention_mask(int seq_len);
    virtual MNN::Express::VARP gen_position_ids(int seq_len);
};

// Embedding start
class Embedding : public Llm {
public:
    Embedding(std::shared_ptr<LlmConfig> config);
    static Embedding* createEmbedding(const std::string& config_path);
    static float dist(MNN::Express::VARP var0, MNN::Express::VARP var1);
    virtual void load() override;
    MNN::Express::VARP embedding(const std::string& txt);
    int dim() const;
private:
    virtual std::vector<int> tokenizer(const std::string& query) override;
    virtual MNN::Express::VARP gen_attention_mask(int seq_len) override;
    virtual MNN::Express::VARP gen_position_ids(int seq_len) override;
};
// Embedding end
}
}

#endif // LLM_hpp
