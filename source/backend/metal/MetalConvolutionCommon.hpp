//
//  MetalConvolutionCommon.hpp
//  MNN
//
//  Created by MNN on 2019/02/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalConvolutionCommon_hpp
#define MetalConvolutionCommon_hpp

#import "core/ConvolutionCommon.hpp"
#import "MetalBackend.hpp"
#import "MNNMetalContext.h"
#if MNN_METAL_ENABLED
namespace MNN {

class MetalConvolutionCommon : public Execution {
public:
    MetalConvolutionCommon(Backend *backend, const MNN::Op *op);
    virtual ~MetalConvolutionCommon() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

protected:
    void loadWeight(const MNN::Convolution2D *conv);

    virtual ErrorCode onFloat(const Tensor *input, const Tensor *output)     = 0;
    virtual id<MTLBuffer> weightForFloat(int group, int oc, int ic, int kh, int kw, const float *src);

private:
    id<MTLBuffer> weightForConv(const Convolution2D *, ConvolutionCommon::Int8Common *, bool);

protected:
    bool mDepthwise     = false;
    int mGroups         = 0;
    int mKernelX        = 0;
    int mKernelY        = 0;
    PadMode mPadMode    = PadMode_CAFFE;
    int mPadX           = 0;
    int mPadY           = 0;
    int mStrideX        = 0;
    int mStrideY        = 0;
    int mDilateX        = 0;
    int mDilateY        = 0;
    int mActivationType = 0;
    const MNN::Op *mOp  = nullptr;

    id<MTLBuffer> mWeight      = nil;
    id<MTLBuffer> mBias        = nil;
    id<MTLBuffer> mConstBuffer = nil;
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */

#endif /* MetalConvolutionCommon_hpp */
