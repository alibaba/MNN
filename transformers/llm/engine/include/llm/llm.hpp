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
class DiskEmbedding;

enum TuneType {
    // op encoder number for commit
    OP_ENCODER_NUMBER = 0,
};
struct KVMeta;
class MNN_PUBLIC Llm {
    using PromptItem = std::pair<std::string, std::string>; // <role, content>
public:
    enum Stage {
        Prefill,
        Decode
    };
    struct GenerateState {
        // forward info
        int prompt_len_ = 0;
        int gen_seq_len_ = 0;
        int all_seq_len_ = 0;
        std::vector<int> history_ids_;
        // time
        int64_t vision_us_ = 0;
        int64_t audio_us_ = 0;
        int64_t prefill_us_ = 0;
        int64_t decode_us_ = 0;
        int current_token_ = 0;
        std::vector<int> output_ids_;
        std::ostream* os_ = nullptr;
        std::string end_with_;
    };
    Llm(std::shared_ptr<LlmConfig> config);
    virtual ~Llm();
    static Llm* createLLM(const std::string& config_path);
    void chat();
    void reset();
    void trace(bool start);
    void tuning(TuneType type, std::vector<int> candidates);
    virtual void load();
    void switchMode(Stage stage);
    void setKVCacheInfo(size_t add, size_t remove, int* reserve = nullptr, int n_reserve = 0);
    MNN::Express::VARP forwardRaw(MNN::Express::VARP hiddenState, MNN::Express::VARP mask, MNN::Express::VARP inputPos);
    virtual MNN::Express::VARP gen_attention_mask(int seq_len);
    virtual MNN::Express::VARP embedding(const std::vector<int>& input_ids);

    MNN::Express::VARP forward(const std::vector<int>& input_ids);
    int sample(MNN::Express::VARP logits, const std::vector<int>& pre_ids, int offset = 0, int size = 0);
    std::string apply_prompt_template(const std::string& user_content) const;
    std::string apply_chat_template(const std::vector<PromptItem>& chat_prompts) const;
    size_t getCurrentHistory() const;
    void eraseHistory(size_t begin, size_t end);
    void response(const std::string& user_content, std::ostream* os = &std::cout, const char* end_with = nullptr, int max_new_tokens = -1);
    void response(const std::vector<PromptItem>& chat_prompts, std::ostream* os = &std::cout, const char* end_with = nullptr, int max_new_tokens = -1);
    void generate_init(std::ostream* os = nullptr, const char* end_with = nullptr);
    void generate(int max_token);
    std::vector<int> generate(const std::vector<int>& input_ids, int max_new_tokens = -1);
    bool stoped();
    void print_speed();
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
    const GenerateState& getState() const {
        return mState;
    }
protected:
    std::shared_ptr<KVMeta> mMeta;
    std::shared_ptr<LlmConfig> config_;
    std::shared_ptr<Tokenizer> tokenizer_;
    std::shared_ptr<DiskEmbedding> disk_embedding_;
    MNN::Express::VARP inputs_embeds_, attention_mask_, position_ids_;
    std::shared_ptr<MNN::Express::Executor::RuntimeManager> runtime_manager_;
    std::shared_ptr<MNN::Express::Executor::RuntimeManager> mllm_runtime_manager_;
    std::vector<std::shared_ptr<MNN::Express::Module>> modules_;
    std::vector<std::shared_ptr<MNN::Express::Module>> prefill_modules_, decode_modules_, current_modules_;
    const MNN::Express::Module* base_module_ = nullptr;
    void init_runtime();
    virtual MNN::Express::VARP gen_position_ids(int seq_len);
    bool mTracing = false;
    GenerateState mState;
};

// Embedding start
class MNN_PUBLIC Embedding : public Llm {
public:
    Embedding(std::shared_ptr<LlmConfig> config);
    static Embedding* createEmbedding(const std::string& config_path, bool load = true);
    static float dist(MNN::Express::VARP var0, MNN::Express::VARP var1);
    virtual void load() override;
    MNN::Express::VARP ids_embedding(const std::vector<int>& ids);
    MNN::Express::VARP txt_embedding(const std::string& txt);
    int dim() const;
private:
    virtual MNN::Express::VARP gen_attention_mask(int seq_len) override;
    virtual MNN::Express::VARP gen_position_ids(int seq_len) override;
};
// Embedding end
}
}

#endif // LLM_hpp
