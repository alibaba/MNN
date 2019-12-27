//
//  CPUQuanConvolutionDepthwise.hpp
//  MNN
//
//  Created by MNN on 2018/10/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUDepthwise_hpp
#define CPUDepthwise_hpp

#include "core/AutoStorage.h"
#include "core/Execution.hpp"
#include "TFQuantizeOp_generated.h"

namespace MNN {
struct ConstConvolutionParameter;
class CPUQuanConvolutionDepthwise : public Execution {
public:
    CPUQuanConvolutionDepthwise(Backend *backend, const Op *depthwiseOp);
    virtual ~CPUQuanConvolutionDepthwise();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    int mStrideH;
    int mStrideW;
    int mDilateX;
    int mDilateY;
    int mDepthMultiplier;
    int mPaddingHeight;
    int mPaddingWidth;
    int ml, mt, mr, mb;
    int mDstYStep, mSrcYStep, mWeightZStep;
    int32_t mZeroPoint;
    PadMode mPadMode;
    FusedActivation mFusedActivationFunction;
    const TfQuantizedConv2D *mLayerParam;
    AutoStorage<int16_t> mWeight;
    AutoStorage<int32_t> mBias;
    Tensor mTempBuffer;
    ConstConvolutionParameter *mConstParameter = nullptr;
};
} // namespace MNN
#endif /* CPUDepthwise_hpp */
