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

#include "evaluation/evaluation.hpp"
#include "llmconfig.hpp"


namespace MNN {
namespace Transformer {

class Llm;

// a index and its corresponding score
struct IndexScore {
    int index;
    float score;
};
struct IndexScoreCmpLess{
    bool operator()(IndexScore a, IndexScore b) {
        return a.score < b.score;
    }
};
struct IndexScoreCmpGreater{
    bool operator()(IndexScore a, IndexScore b) {
        return a.score > b.score;
    }
};
// a series of index and their corresponding logits
struct SubsetLogits{
    std::vector<int> index;
    MNN::Express::VARP logits;
    bool is_subset;
};
// sample candidate
struct SampleCandidate {
    // StateCacheReference
    int all_seq_len_=0, gen_seq_len_=0;
    std::vector<int> tokens;
};

class MNN_PUBLIC Sampler {
public:
    class LlmSamplerConfig {
    public:
        int max_new_tokens = 512;
        int max_all_tokens = 2048;
    };
protected:
    Llm* mLlm;
    std::vector<struct SampleCandidate> mCandidates;
    int select(struct SubsetLogits& subset, int id);
    int randomSelect(float* probs, size_t size);
    int randomSelect(MNN::Express::VARP probs);
    int reSoftmaxSelect(struct SubsetLogits subset, float temperature=1.0);
    SubsetLogits createSubsetLogits(MNN::Express::VARP logits);
    SubsetLogits createSubsetLogits(MNN::Express::VARP logits, const std::vector<int>& index);
    SubsetLogits createSubsetLogits(int size);
    SubsetLogits createSubsetLogits(const std::vector<float>& scores, const std::vector<int>& index);
    void transformIndex(struct SubsetLogits& superset, struct SubsetLogits& subset);
public:
    static Sampler* createSampler(Llm* llm, const std::string& config_path);
    virtual std::string sample(const std::vector<int>& input_ids, std::ostream* os = &std::cout, const char* end_with = nullptr, struct TimePerformance* time_perf = nullptr) = 0;
    // prepare for another round of sampling
    // in the future, only reset its own.
    virtual void reset(Llm* llm) {}
    virtual void reset() {}
};


class MNN_PUBLIC LocalSampler: public Sampler {
public:
    class LocalSamplerConfig : public LlmSamplerConfig {
    public:
        struct SamplerPenaltyConfig {
            float penalty = 1.05;
            int ngram = 8;
            float ngram_factor = 1.02; // panalize repeated ngram with a multiplied ngram_factor.
            float max_penalty = 10.;
            std::string sampler = "temperature"; // "greedy", "temperature". 
        };
        std::string type = "temperature";
        std::string select_type = "temperature";
        float temperature = 0.8;
        int topK = 40;
        float topP = 0.9;
        float minP = 0.05;
        float tfsZ = 1.0;
        float typical = 0.95;
        struct SamplerPenaltyConfig penaltyConfig;
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
protected:
    LocalSamplerConfig mConfig;
    LocalSamplerConfig getSamplerConfig(std::shared_ptr<LlmConfig> llmConfig);
    int argmaxSelect(struct SubsetLogits superset);
    int packSoftmax(MNN::Express::VARP logits, std::vector<IndexScore>& index_scores, float temperature = 1.0);
    struct SubsetLogits penalty(struct SubsetLogits superset);
    struct SubsetLogits topK(struct SubsetLogits superset);
    struct SubsetLogits topP(struct SubsetLogits superset);
    struct SubsetLogits minP(struct SubsetLogits superset);
    struct SubsetLogits tfs(struct SubsetLogits superset);
    struct SubsetLogits typical(struct SubsetLogits superset);
    struct SubsetLogits mixed(struct SubsetLogits subset);
    struct SubsetLogits subsetSampler(std::string sampler_type, struct SubsetLogits subset);
    int handleSelect(struct SubsetLogits subset);
    std::string handleToken(int token, std::ostream* os = &std::cout, const char* end_with = nullptr);
public:
    LocalSampler(Llm* llm, const std::string& config_path);
    int algorithm(MNN::Express::VARP logits);
    virtual std::string sample(const std::vector<int>& input_ids, std::ostream* os = &std::cout, const char* end_with = nullptr, struct TimePerformance* time_perf = nullptr) override;
    virtual void reset(Llm* llm) override;
    virtual void reset() override;
};


} // Transformer
} // MNN


#endif // SAMPLER_hpp