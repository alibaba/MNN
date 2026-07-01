#ifndef HuggingFaceQwen3_hpp
#define HuggingFaceQwen3_hpp

#include "SafetensorConverter.hpp"

namespace MNN {
namespace SafeTensors {

struct HuggingFaceQwen3Config {
    int hiddenSize = 0;
    int headDim = 0;
    int numHead = 0;
    int kvNumHead = 0;
    int blockNumber = 0;
    int maxPositionEmbeddings = 0;
    float ropeTheta = 0.0f;
    int ropeCutHeadDim = 0;
    bool outputLastHiddenState = true;
};

void HuggingFaceQwen3Convert(const Converter* converter, MNN::NetT* dst, const HuggingFaceQwen3Config& config);

} // namespace SafeTensors
} // namespace MNN

#endif
