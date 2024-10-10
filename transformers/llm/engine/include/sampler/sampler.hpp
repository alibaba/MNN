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

namespace MNN {
namespace Transformer {

#define MICRO_TO_MILLI 1e-3f
#define MILLI_TO_MICRO 1000
#define MICRO_TO_SEC 1e-6f
#define SEC_TO_MICRO 1000000

#define MEGA_TO_GIGA (1/1024.f)
#define GIGA_TO_MEGA 1024.f
#define KILLO_TO_GIGA (1/1024.f/1024.f)
#define GIGA_TO_KILLO (1024.f*1024.f)
#define KILLO_TO_MEGA (1/1024.f)
#define MEGA_TO_KILLO 1024.f

struct PrefillTimePerformance {
    size_t prefill_prev_token_ = 0;
    size_t prefill_token_ = 0;
    size_t prefill_us_ = 0;
};

struct DecodeTimePerformance {
    size_t decode_prev_token_ = 0;
    size_t decode_us_ = 0;
};

struct TimePerformance {
    std::vector<PrefillTimePerformance> prefill_record_;
    std::vector<DecodeTimePerformance> decode_record_;
};

void mergePerformance(struct TimePerformance* dst, struct TimePerformance* src);
void clearPerformance(struct TimePerformance* perf);

class Llm;

class MNN_PUBLIC Sampler {
protected:
    Llm* mLlm;
    std::vector<std::vector<int>> mCandidates;
    std::vector<int> mCommonPrefix;
    int mMaxNewTokens;
    int getGenLength(int candidate, int output_len) const {
        return mCandidates[candidate].size() - (mCommonPrefix.size() - output_len);
    }
public:
    virtual std::string sample(const std::vector<int>& input_ids, std::ostream* os = &std::cout, const char* end_with = nullptr, struct TimePerformance* time_perf = nullptr) = 0;
    // prepare for another round of sampling
    // in the future, only reset its own.
    virtual void reset() {}
};

class MNN_PUBLIC LocalSampler : public Sampler {
public:
    struct LocalSamplerConfig {
        std::string type = "temperature";
        float temperature = 0.8;
        int topK = 40;
        float topP = 0.9;
        float minP = 0.05;
        float tfsZ = 1.0;
        float typical = 0.95;
        float penalty = 1.1;
        int ngram = 8;
        float ngram_factor = 1.0; // panalize repeated ngram with a multiplied ngram_factor.
        float max_penalty = 10.;
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
    void tfs(MNN::Express::VARP logits, float z, float temperature, std::vector<int>& index, std::vector<float>& tfsprob);
    int tfs(MNN::Express::VARP logits, float z = 1.0, float temperature = 1.0);
    void typical(MNN::Express::VARP logits, float p, float temperature, std::vector<int>& index, std::vector<float>& minPprob);
    int typical(MNN::Express::VARP logits, float p = 1.0, float temperature = 1.0);
    void penalty(MNN::Express::VARP logits, float penalty = 1.0, bool penalizeNgram = false, int ngram = 8, float ngram_factor = 1.0);
    int penalty(MNN::Express::VARP logits, float penalty = 1.0, int ngram = 8, float ngram_factor = 1.0, float temperature = 1.0);
    // int mixed(MNN::Express::VARP logits);
    std::string handleToken(int token, std::ostream* os = &std::cout, const char* end_with = nullptr);
public:
    LocalSampler(Llm* llm, int max_new_tokens, struct LocalSamplerConfig config);
    int algorithm(MNN::Express::VARP logits);
    virtual std::string sample(const std::vector<int>& input_ids, std::ostream* os = &std::cout, const char* end_with = nullptr, struct TimePerformance* time_perf = nullptr) override;
    virtual void reset() override;
};


} // Transformer
} // MNN


#endif // SAMPLER_hpp