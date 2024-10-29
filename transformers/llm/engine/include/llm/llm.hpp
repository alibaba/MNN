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

class MNN_PUBLIC Llm {
    using PromptItem = std::pair<std::string, std::string>; // <role, content>
public:
    Llm(std::shared_ptr<LlmConfig> config) : config_(config) {}
    virtual ~Llm();
    static Llm* createLLM(const std::string& config_path);
    void reset();
    void trace(bool start);
    virtual void load();
    MNN::Express::VARP forward(const std::vector<int>& input_ids, int kv_seq_len_, int gen_seq_len_, bool is_prefill);
    void generate_init();
    std::string generateTrace(const std::vector<int>& input_ids, std::ostream* os, const char* end_with);
    void print_speed();
    // config function
    std::string dump_config();
    bool set_config(const std::string& content);
    // lora function
    size_t apply_lora(const std::string& lora_path);
    Llm* create_lora(const std::string& lora_path);
    bool release_module(size_t index);
    bool select_module(size_t index);
    friend class Pipeline;
public:
    // time
    int64_t prefill_us_ = 0;
    int64_t decode_us_ = 0;
    bool is_single_ = true;
    bool attention_fused_ = true;
    virtual std::vector<int> tokenizer(const std::string& query);
    std::string decode(int id);
    bool is_stop(int token_id);
    bool reuse_kv() const;
protected:
    std::shared_ptr<LlmConfig> config_;
    std::shared_ptr<Tokenizer> tokenizer_;
    std::vector<int> key_value_shape_ = {};
    std::vector<MNN::Express::VARP> past_key_values_;
    MNN::Express::VARP inputs_embeds_, attention_mask_, position_ids_;
    std::shared_ptr<MNN::Express::Executor::RuntimeManager> runtime_manager_;
    std::vector<std::shared_ptr<MNN::Express::Module>> modules_;
    std::vector<std::shared_ptr<MNN::Express::Module>> prefill_modules_, decode_modules_, current_modules_;
    const MNN::Express::Module* base_module_ = nullptr;
    void init_runtime();
    virtual MNN::Express::VARP embedding(const std::vector<int>& input_ids);
    virtual MNN::Express::VARP gen_attention_mask(int seq_len, int kv_seq_len_, int gen_seq_len_);
    virtual MNN::Express::VARP gen_position_ids(int seq_len, int kv_seq_len_, int gen_seq_len);
    bool mTracing = false;
};

// Embedding start
class Embedding : public Llm {
public:
    Embedding(std::shared_ptr<LlmConfig> config);
    static Embedding* createEmbedding(const std::string& config_path, bool load = true);
    static float dist(MNN::Express::VARP var0, MNN::Express::VARP var1);
    virtual void load() override;
    MNN::Express::VARP ids_embedding(const std::vector<int>& ids);
    MNN::Express::VARP txt_embedding(const std::string& txt);
    int dim() const;
private:
    virtual std::vector<int> tokenizer(const std::string& query) override;
    virtual MNN::Express::VARP gen_attention_mask(int seq_len);
    virtual MNN::Express::VARP gen_position_ids(int seq_len);
};
// Embedding end
}
}

#endif // LLM_hpp
