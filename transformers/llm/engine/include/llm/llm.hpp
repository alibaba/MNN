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

#include "evaluation/evaluation.hpp"
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/MathOp.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>

namespace MNN {
namespace Transformer {
class Tokenizer;
class Pipeline;
class LlmConfig;
class Sampler;
class PromptLib;
struct TimePerformance;


// <role, content>
#define PromptItem std::pair<std::string, std::string>

class MNN_PUBLIC LlmSessionInfo {
public:
    // Sampler needs
    int all_seq_len_=0, gen_seq_len_=0;
    std::vector<int> tokens;
    // PromptLib needs
    std::vector<PromptItem> mHistory;
    std::vector<PromptItem> mInputs;
    // Performance needs
    struct TimePerformance mTimePerformance;
public:
    LlmSessionInfo():all_seq_len_(0),gen_seq_len_(0){}
    void resetSamplerFields();
    void resetPromptFields();
    void resetPerformanceFields();
    void print_speed(std::ostream* os);
    float average_total_speed();
    float average_prefill_speed();
    float average_decode_speed();
    float getTotalPrefillTime();
    float getTotalDecodeTime();
    int getTotalPromptLen();
    int getTotalDecodeLen();
};


class DiskEmbedding;

enum TuneType {
    // op encoder number for commit
    OP_ENCODER_NUMBER = 0,
};

class MNN_PUBLIC Llm {
public:
    std::shared_ptr<Sampler> mSampler;
    std::shared_ptr<PromptLib> mPromptLib;
    std::vector<LlmSessionInfo> mLlmSessionInfos; // Llm conversation session information. Currently, only mLlmSessionInfos[0] is allowed!
public:
    Llm(std::shared_ptr<LlmConfig> config) : config_(config) {}
    virtual ~Llm();
    static Llm* createLLM(const std::string& config_path);
    void chat(bool session_by_line = false, bool from_file = false, 
              std::istream* is = &std::cin, std::ostream* os = &std::cout, 
              const char* end_with = "\n", std::string exit_prompt = "/exit", std::string reset_token = "/reset");
    void reset();
    void trace(bool start);
    void tuning(TuneType type, std::vector<int> candidates);
    virtual void load();
    MNN::Express::VARP forward(const std::vector<int>& input_ids, int kv_seq_len_, int gen_seq_len_, bool is_prefill);
    std::string response(const std::string& user_content, std::ostream* os = &std::cout, const char* end_with = nullptr);
    std::string generate(const std::string& prompt, std::ostream* os = &std::cout, const char* end_with = "\n");
    std::string generate(const std::vector<int>& input_ids, std::ostream* os = &std::cout, const char* end_with = "\n");
    void generate_init();
    std::string generateTrace(const std::vector<int>& input_ids, std::ostream* os, const char* end_with);
    void print_speed();
    void print_speed(std::ostream* os);
    std::vector<float> perplexity(std::string prompt_file, std::ostream* statsOS = nullptr);
    // config function
    std::string dump_config();
    bool set_config(const std::string& content);
    // lora function
    size_t apply_lora(const std::string& lora_path);
    Llm* create_lora(const std::string& lora_path);
    bool release_module(size_t index);
    bool select_module(size_t index);
    // tokenier function
    bool is_stop(int token_id);
    std::string tokenizer_decode(int id);
    virtual std::vector<int> tokenizer_encode(const std::string& query, bool use_template = true);
    friend class Pipeline;
public:
    bool is_single_ = true;
    bool attention_fused_ = true;
    bool is_stop(int token_id);
    bool reuse_kv() const;
public:
    float average_total_speed();
    float average_prefill_speed();
    float average_decode_speed();
    float getTotalPrefillTime();
    float getTotalDecodeTime();
    int getTotalPromptLen();
    int getTotalDecodeLen();
protected:
    std::shared_ptr<LlmConfig> config_;
    std::shared_ptr<Tokenizer> tokenizer_;
    std::shared_ptr<DiskEmbedding> disk_embedding_;
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
protected:
    bool getUserPrompt(bool from_file, std::istream* is, std::string& user_str);
    void chat_init();
    void chat_reset();
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
    virtual MNN::Express::VARP gen_attention_mask(int seq_len);
    virtual MNN::Express::VARP gen_position_ids(int seq_len);
};
// Embedding end
}
}

#endif // LLM_hpp
