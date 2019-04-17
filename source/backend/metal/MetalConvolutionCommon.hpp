//
//  MetalConvolutionCommon.hpp
//  MNN
//
//  Created by MNN on 2019/02/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalConvolutionCommon_hpp
#define MetalConvolutionCommon_hpp

#import "ConvolutionIntFactory.hpp"
#import "Execution.hpp"
#import "MNNMetalContext.h"
#import "MNN_generated.h"
#import "MetalDefine.h"

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

    virtual ErrorCode onQuantized(const Tensor *input, const Tensor *output) = 0;
    virtual ErrorCode onFloat(const Tensor *input, const Tensor *output)     = 0;
    virtual id<MTLBuffer> weightForQuantized(int group, int oc, int ic, int kh, int kw, const int8_t *src);
    virtual id<MTLBuffer> weightForFloat(int group, int oc, int ic, int kh, int kw, const float *src);

private:
    id<MTLBuffer> weightForConv(const Convolution2D *, ConvolutionIntFactory::Int8Common *, bool);

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

    bool mQnt            = false;
    int32_t mQntRange[2] = {0, 0};
    float mQntScale      = 0;
    std::shared_ptr<Tensor> mQntInput;

    id<MTLBuffer> mWeight      = nil;
    id<MTLBuffer> mBias        = nil;
    id<MTLBuffer> mAlpha       = nil;
    id<MTLBuffer> mConstBuffer = nil;
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */

#endif /* MetalConvolutionCommon_hpp */
