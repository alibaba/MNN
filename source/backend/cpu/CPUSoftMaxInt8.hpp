//
//  CPUSoftMaxInt8.hpp
//  MNNCPU
//
//  Created by MNN on 2023/4/22.
//

#ifndef CPUSoftMaxInt8_hpp
#define CPUSoftMaxInt8_hpp
#include "core/Execution.hpp"
#include <math.h>
namespace MNN {

class CPUSoftmaxInt8 : public Execution {
public:
    CPUSoftmaxInt8(Backend *backend, int axis);
    virtual ~CPUSoftmaxInt8() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    static Execution* create(const MNN::Op *op, Backend *backend);

    void QuantizedSoftmax(const uint8_t *inputData, int outerSize, int targetAxis, int32_t inputBetaMultiplier,
                          int32_t inputBetaLeftShift, uint8_t *output_data, int threadNum);

private:
    int32_t mInputMultiplier;
    int mInputLeftShift;
    int mDiffMin;
    int mAxis;
    int mInside;
    int mOutside;
    int mTargetAxis;
    Tensor mStorage;
    Tensor mTempOutput;
    bool mNeedUnpackC4;
};

}
#endif /* CPUSoftMaxInt8_hpp */
