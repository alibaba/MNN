//
//  CPUTFQuantizedConv2D.hpp
//  MNN
//
//  Created by MNN on 2018/08/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUTFQuantizedConv2D_hpp
#define CPUTFQuantizedConv2D_hpp

#include "core/AutoStorage.h"
#include "backend/cpu/CPUConvolution.hpp"
#include "core/Execution.hpp"
#include "TFQuantizeOp_generated.h"

namespace MNN {
class CPUTFQuantizedConv2D : public Execution {
public:
    CPUTFQuantizedConv2D(Backend *backend, const Op *TfQuantizedConv2DOp);
    virtual ~CPUTFQuantizedConv2D();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

    struct QuanParameter {
        int32_t mOutputMultiplier;
        int32_t mOutputShiftBefore;
        int32_t mOutputShiftAfter;
        int32_t mOutputActivationMin;
        int32_t mOutputActivationMax;
        int32_t mOutputOffset;
        int32_t mFilterOffset;
        int32_t mInputOffset;
        int32_t mOffsetAdd;
    };

private:
    const TfQuantizedConv2D *mTfQuantizedConv2D_param;

    Tensor mTempBuffer;
    Tensor mTempDstBuffer;
    Tensor mTempInputSum;
    int mThreadNumber;
    // Reorder as N/4 (HW(C/4))/2 N4 C8
    std::shared_ptr<Tensor> mWeight;

    AutoStorage<int32_t> mBias;
    FusedActivation mFusedActivationFunction;

    QuanParameter *mQuanParameter;
    ConvolutionCommon::Im2ColParameter *mIm2ColParamter;
};
} // namespace MNN
#endif
