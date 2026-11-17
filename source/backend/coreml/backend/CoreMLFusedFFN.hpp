// Copyright © Alibaba Group Holding Limited. All rights reserved.
#ifndef MNN_COREML_FUSED_FFN_HPP
#define MNN_COREML_FUSED_FFN_HPP
#include "CoreMLExecutorWrapper.h"

namespace MNN {
// Experimental fixed-weight SwiGLU subgraph. Inputs and outputs are dense FP16
// [tokens, hidden]. The caller supplies an immutable Core ML weight blob.
class MNN_PUBLIC CoreMLFusedFFN {
public:
    bool prepare(int tokens, int hidden, int intermediate, const std::string& weightsPath,
                 const std::vector<uint64_t>& offsets, bool fp32Gate = true);
    bool run(const Tensor* input, const Tensor* output);
    bool compiledFromSource() const;

private:
    int mTokens = 0, mHidden = 0;
    std::shared_ptr<CoreMLExecutorWrapper> mProgram;
};
} // namespace MNN
#endif
