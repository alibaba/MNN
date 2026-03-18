#ifndef SAMPLER_hpp
#define SAMPLER_hpp

#include <vector>
#include <memory>
#include <string>
#include <functional>
#include <unordered_map>
#include <random>

#include <MNN/expr/Expr.hpp>

#include "llmconfig.hpp"
#include "llm/llm.hpp"

namespace MNN {
namespace Transformer {

struct SamplerState {
    std::vector<int> indices;       // token indices (empty = full vocab)
    std::vector<float> logits;      // logit values
    std::vector<float> probs;       // cached softmax probabilities (empty = not computed)
    int vocab_size = 0;
    bool is_subset = false;
    int selected_token = -1;

    void ensureProbs(float temperature);
    void invalidateProbs();
};

class Sampler {
public:
    class SamplerConfig {
    public:
        int max_new_tokens = 512;
        int max_all_tokens = 2048;
        std::string type = "temperature";
        std::string select_type = "temperature";
        float temperature = 0.8;
        int topK = 40;
        float topP = 0.9;
        float minP = 0.05;
        float tfsZ = 1.0;
        float typical = 0.95;
        // penalty
        float repetition_penalty = 1.0;
        float presence_penalty = 0.0;
        float frequency_penalty = 0.0;
        int penalty_window = 0;
        int ngram = 8;
        float ngram_factor = 1.02;
        float max_penalty = 10.;
        std::string sampler = "temperature";
        std::vector<std::string> mixedSamplers = {"topK", "tfs", "typical", "topP", "min_p", "temperature"};
        // logit bias and banned tokens
        std::unordered_map<int, float> logit_bias;
        std::vector<int> banned_tokens;
        void configSampler(std::string sampler_type, std::shared_ptr<LlmConfig> llmConfig);
        void configGreedy(std::shared_ptr<LlmConfig> llmConfig);
        void configTemperature(std::shared_ptr<LlmConfig> llmConfig);
        void configTopK(std::shared_ptr<LlmConfig> llmConfig);
        void configTopP(std::shared_ptr<LlmConfig> llmConfig);
        void configMinP(std::shared_ptr<LlmConfig> llmConfig);
        void configTFS(std::shared_ptr<LlmConfig> llmConfig);
        void configTypical(std::shared_ptr<LlmConfig> llmConfig);
        void configPenalty(std::shared_ptr<LlmConfig> llmConfig);
        void configMixed(std::shared_ptr<LlmConfig> llmConfig);
    };
public:
    static Sampler* createSampler(std::shared_ptr<LlmContext> context, std::shared_ptr<LlmConfig> config);
    Sampler(std::shared_ptr<LlmContext> context, std::shared_ptr<LlmConfig> config);
    int sample(MNN::Express::VARP logits);
private:
    std::shared_ptr<LlmContext> mContext;
    SamplerConfig mConfig;
    std::mt19937 mRng;
    // Pipeline
    using SamplerStep = std::function<void(SamplerState&)>;
    std::vector<SamplerStep> mPipeline;
    void buildPipeline();
    SamplerState createState(MNN::Express::VARP logits);
    // Step implementations
    void stepPenalty(SamplerState& state);
    void stepTopK(SamplerState& state);
    void stepTopP(SamplerState& state);
    void stepMinP(SamplerState& state);
    void stepTfs(SamplerState& state);
    void stepTypical(SamplerState& state);
    void stepLogitBias(SamplerState& state);
    void stepBannedTokens(SamplerState& state);
    void stepSelect(SamplerState& state);
};

} // Transformer
} // MNN

#endif // SAMPLER_hpp
