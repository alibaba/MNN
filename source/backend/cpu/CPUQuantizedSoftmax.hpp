//
//  CPUQuantizedSoftmax.hpp
//  MNN
//
//  Created by MNN on 2018/09/29.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUQuantizedSoftmax_hpp
#define CPUQuantizedSoftmax_hpp

#include "core/Execution.hpp"

namespace MNN {

template <typename T>
class CPUQuantizedSoftmax : public Execution {
public:
    CPUQuantizedSoftmax(Backend *backend, const Op *op);
    virtual ~CPUQuantizedSoftmax() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

    void QuantizedSoftmax(const uint8_t *inputData, const std::vector<int> &inputDims, int32_t inputBetaMultiplier,
                          int32_t inputBetaLeftShift, uint8_t *output_data, const std::vector<int> &outputDims);

private:
    int32_t mInputMultiplier;
    int mInputLeftShift;
    int mDiffMin;
    float mBeta;
    float mInputScale;
    std::vector<int> mInputDims;
    std::vector<int> mOutputDims;
};

} // namespace MNN

#endif /* CPUQuantizedSoftmax_hpp */
