#ifndef SAMPLER_hpp
#define SAMPLER_hpp

#include <vector>
#include <memory>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <streambuf>
#include <functional>
#include <unordered_map>
#include <utility>

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/MathOp.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>

#include "llmconfig.hpp"
#include "llm/llm.hpp"


namespace MNN {
namespace Transformer {

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
        float penalty = 1.05;
        int ngram = 8;
        float ngram_factor = 1.02; // panalize repeated ngram with a multiplied ngram_factor.
        float max_penalty = 10.;
        std::string sampler = "temperature"; // "greedy", "temperature".
        std::vector<std::string> mixedSamplers= {"topK", "tfs", "typical", "topP", "min_p", "temperature"};
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
    struct SubsetLogits penalty(struct SubsetLogits superset);
    struct SubsetLogits topK(struct SubsetLogits superset);
    struct SubsetLogits topP(struct SubsetLogits superset);
    struct SubsetLogits minP(struct SubsetLogits superset);
    struct SubsetLogits tfs(struct SubsetLogits superset);
    struct SubsetLogits typical(struct SubsetLogits superset);
    struct SubsetLogits mixed(struct SubsetLogits subset);
    struct SubsetLogits subsetSampler(std::string sampler_type, struct SubsetLogits subset);
    int handleSelect(struct SubsetLogits subset);
};


} // Transformer
} // MNN


#endif // SAMPLER_hpp