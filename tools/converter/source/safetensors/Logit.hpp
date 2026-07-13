#ifndef Logit_hpp
#define Logit_hpp

#include <string>
#include "SafetensorConverter.hpp"

namespace MNN {
namespace SafeTensors {

struct LogitConfig {
    // Default to Qwen/GPT2 word embedding matrix
    std::string wteWeightName = "module.gpt2.transformer.wte.weight";

    // Optional: if provided, will be used to disambiguate weight layout
    int hiddenSize = 0;

    std::string inputName = "hidden_state";
    std::string outputName = "output";
};

void LogitConvert(const Converter* converter, MNN::NetT* dst, const LogitConfig& config);
void MakeTieEmbedding(const Converter* converter, const MNN::NetT* src, MNN::NetT* dst);
void MakeTopKV(const Converter* converter, const MNN::NetT* logit, MNN::NetT* dst, int K);
void MakeSoftmax(const Converter* converter, const MNN::NetT* logit, MNN::NetT* dst);
void MakeBeamTopKV(const Converter* converter, const MNN::NetT* logit, MNN::NetT* dst, int K);

} // namespace SafeTensors
} // namespace MNN

#endif
