//
//  MetalTFQuantizedConv2D.hpp
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalTFQuantizedConv2D_hpp
#define MetalTFQuantizedConv2D_hpp

#import "Execution.hpp"
#import "MNN_generated.h"
#import "MetalDefine.h"

#if MNN_METAL_ENABLED
namespace MNN {

class MetalTFQuantizedConv2D : public Execution {
public:
    MetalTFQuantizedConv2D(Backend *backend, const MNN::Op *op);
    virtual ~MetalTFQuantizedConv2D() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    BOOL mDepthwise  = false;
    PadMode mPadMode = PadMode_CAFFE;
    int mGroups      = 0;
    int mKernelX     = 0;
    int mKernelY     = 0;
    int mStrideX     = 0;
    int mStrideY     = 0;
    int mDilateX     = 0;
    int mDilateY     = 0;

    int mInputZeroPoint      = 0;
    int mOutputZeroPoint     = 0;
    int mOutputShiftBefore   = 0;
    int mOutputMultiplier    = 0;
    int mOutputShiftAfter    = 0;
    int mOutputActivationMin = 0;
    int mOutputActivationMax = 0;

    id<MTLBuffer> mWeight      = nil;
    id<MTLBuffer> mBias        = nil;
    id<MTLBuffer> mConstBuffer = nil;

private:
    ErrorCode onConv(const Tensor *input, const Tensor *output);
    ErrorCode onDepthwise(const Tensor *input, const Tensor *output);
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
#endif /* MetalTFQuantizedConv2D_hpp */
