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

#include "sampler.hpp"
#include "evaluation/dataset.hpp"

namespace MNN {
namespace Transformer {
class Llm;

class MNN_PUBLIC TextPPLMeasurer : public Sampler {
protected:
    Llm* mLlm;
    int mStride;
    std::string mDatasetType;
    LlmSamplerConfig mConfig;
public:
    TextPPLMeasurer(Llm* llm, std::shared_ptr<LlmConfig> config);
    float perplexity_one(const std::vector<int>& prompt);
    std::vector<float> perplexity(std::vector<std::vector<int>> prompts);
    std::vector<float> perplexity(std::vector<std::string> prompts);
    virtual std::string sample(const std::vector<int>& input_ids, std::ostream* os = &std::cout, const char* end_with = nullptr, struct TimePerformance* time_perf = nullptr) override { return "perplexity evaluation!\n"; }
    virtual std::vector<float> perplexity(std::string prompt_file, std::ostream* perfOS = nullptr) override;
};

class MNN_PUBLIC ChatPPLMeasurer : public Sampler {
protected:
    Llm* mLlm;
    std::string mDatasetType;
    int mDatasetSampleSize;
    LlmSamplerConfig mConfig;
    void handleToken(int token);
    std::vector<float> sample(const std::vector<int>& input_ids, const std::vector<int>& prompt, struct TimePerformance* time_perf);
public:
    ChatPPLMeasurer(Llm* llm, std::shared_ptr<LlmConfig> config);
    void getStats(const std::vector<std::vector<std::vector<PromptItem>>>& prompts);
    float perplexity_one(const std::vector<std::vector<PromptItem>>& prompt, std::ostream* perfOS);
    std::vector<float> perplexity(const std::vector<std::vector<std::vector<PromptItem>>>& prompts, std::ostream* perfOS);
    virtual std::string sample(const std::vector<int>& input_ids, std::ostream* os = &std::cout, const char* end_with = nullptr, struct TimePerformance* time_perf = nullptr) override { return "perplexity evaluation!\n"; }
    virtual std::vector<float> perplexity(std::string prompt_file, std::ostream* perfOS = nullptr) override;
};



} // Transformer
} // MNN


#endif // SAMPLER_hpp