#ifndef PERPLEXITY_hpp
#define PERPLEXITY_hpp

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

namespace MNN {
namespace Transformer {
class Llm;

class MNN_PUBLIC PPLMeasurer {
protected:
    Llm* mLlm;
    StateCacheManager*   mStateCacheManager;
    std::vector<std::vector<int>> mPrompts;
    std::shared_ptr<StateCacheReference> mCandidate;
    int mStride, mMaxLen;
    void init(Llm* llm, StateCacheManager* manager, std::vector<std::vector<int>> prompts, int max_len, int stride);
public:
    PPLMeasurer(Llm* llm, StateCacheManager* manager, std::vector<std::vector<int>> prompts, int max_len=2048, int stride=0);
    PPLMeasurer(Llm* llm, StateCacheManager* manager, std::vector<std::string> prompts, int max_len=2048, int stride=0);
    float perplexity_one(const std::vector<int>& prompt);
    std::vector<float> perplexity();
    // prepare for another round of sampling
    // in the future, only reset its own.
    void reset();
    void reset(int max_len, int stride);
};



} // Transformer
} // MNN


#endif // SAMPLER_hpp