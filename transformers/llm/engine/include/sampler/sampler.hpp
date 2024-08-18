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
#include <MNN/StateCacheManager.hpp>
#include "evaluation/evaluation.hpp"

namespace MNN {
namespace Transformer {
class Llm;

class MNN_PUBLIC Sampler {
protected:
    Llm* mLlm;
    StateCacheManager*   mStateCacheManager; 
    std::vector<std::pair<std::vector<int>, std::shared_ptr<StateCacheReference>>> mCandidates;
    std::vector<int> mCommonPrefix;
    int mMaxNewTokens;
    int getGenLength(int candidate, int output_len) const {
        return mCandidates[candidate].first.size() - (mCommonPrefix.size() - output_len);
    }
public:
    virtual std::string sample(const std::vector<int>& input_ids, std::ostream* os = &std::cout, const char* end_with = nullptr, struct timePerformance* perf = nullptr) = 0;
    // prepare for another round of sampling
    // in the future, only reset its own.
    virtual void reset() {mStateCacheManager->clear();}
};

class MNN_PUBLIC LocalSampler : public Sampler {
public:
    struct LocalSamplerConfig {
        std::string type = "greedy";
        float temperature = 1.0;
        int topK = 40;
        float topP = 0.9;
        float minP = 0.1;
    };
private:
    struct LocalSamplerConfig mConfig;
    int randomSelect(float* probs, size_t size);
    int argmax(MNN::Express::VARP logits);
    int temperature(MNN::Express::VARP logits, float temperature = 1.0);
    struct IndexProb {
        int index;
        float prob;
    };
    struct IndexProbCmpLess{
        bool operator()(IndexProb a, IndexProb b) {
            return a.prob < b.prob;
        }
    };
    struct IndexProbCmpGreater{
        bool operator()(IndexProb a, IndexProb b) {
            return a.prob > b.prob;
        }
    };
    int reSoftmaxSelect(std::vector<int> index, std::vector<float> scores, float temperature);
    void topK(MNN::Express::VARP logits, int K, std::vector<int>& topKindex, std::vector<float>& topKprob);
    int topK(MNN::Express::VARP logits, int K = 40, float temperature = 1.0);
    void topP(MNN::Express::VARP logits, float p, float temperature, std::vector<int>& topPindex, std::vector<float>& topPprob);
    int topP(MNN::Express::VARP logits, float p = 0.9, float temperature = 1.0);
    void minP(MNN::Express::VARP logits, float p, float temperature, std::vector<int>& minPindex, std::vector<float>& minPprob);
    int minP(MNN::Express::VARP logits, float p = 0.1, float temperature = 1.0);
    std::string handleToken(int token, std::ostream* os = &std::cout, const char* end_with = nullptr);
public:
    LocalSampler(Llm* llm, StateCacheManager* manager, int max_new_tokens, struct LocalSamplerConfig config);
    int algorithm(MNN::Express::VARP logits);
    virtual std::string sample(const std::vector<int>& input_ids, std::ostream* os = &std::cout, const char* end_with = nullptr, struct timePerformance* perf = nullptr) override;
    virtual void reset() override;
};


} // Transformer
} // MNN


#endif // SAMPLER_hpp