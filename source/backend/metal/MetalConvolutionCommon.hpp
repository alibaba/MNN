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
#import "MetalExecution.hpp"
#import "MNNMetalContext.h"
#if MNN_METAL_ENABLED
namespace MNN {

class MetalConvolutionCommon : public MetalExecution {
public:
    MetalConvolutionCommon(Backend *backend, const MNN::Op *op, std::shared_ptr<MNN::Tensor> bias);
    virtual ~MetalConvolutionCommon() = default;
    virtual void onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) override;

protected:
    void loadWeight(const MNN::Convolution2D *conv);

    virtual void onFloat(const Tensor *input, const Tensor *output, id<MTLComputeCommandEncoder> encoder)     = 0;
    virtual std::shared_ptr<MNN::Tensor> weightForFloat(int group, int oc, int ic, int kh, int kw, const float *src);

private:

protected:
    int mKernelX        = 0;
    int mKernelY        = 0;
    int mStrideX        = 0;
    int mStrideY        = 0;
    int mDilateX        = 0;
    int mDilateY        = 0;
    int mActivationType = 0;
    const MNN::Op *mOp  = nullptr;

    std::shared_ptr<MNN::Tensor> mWeight;
    std::shared_ptr<MNN::Tensor> mBias;
    id<MTLBuffer> mConstBuffer = nil;
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */

#endif /* MetalConvolutionCommon_hpp */
