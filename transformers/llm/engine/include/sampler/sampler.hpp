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
    std::shared_ptr<StateCacheManager> mStateCacheManager; 
    std::vector<std::pair<std::vector<int>, std::shared_ptr<StateCacheReference>>> mCandidates;
    std::vector<int> mCommonPrefix;
    int mMaxNewTokens;
    int getGenLength(int candidate, int output_len) const {
        return mCandidates[candidate].first.size() - (mCommonPrefix.size() - output_len);
    }
public:
    virtual std::string sample(const std::vector<int>& input_ids, std::ostream* os = &std::cout, const char* end_with = nullptr, struct timePerformance* perf = nullptr) = 0;
};

class MNN_PUBLIC LocalSampler : public Sampler {
private:
    std::string mType;
    int argmax(MNN::Express::VARP logits);
    std::string handleToken(int token, std::ostream* os = &std::cout, const char* end_with = nullptr);
public:
    LocalSampler(Llm* llm, StateCacheManager* manager, int max_new_tokens, std::string type="greedy");
    int algorithm(MNN::Express::VARP logits);
    virtual std::string sample(const std::vector<int>& input_ids, std::ostream* os = &std::cout, const char* end_with = nullptr, struct timePerformance* perf = nullptr) override;
};


} // Transformer
} // MNN


#endif // SAMPLER_hpp