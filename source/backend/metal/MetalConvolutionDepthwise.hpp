//
//  MetalConvolutionDepthwise.hpp
//  MNN
//
//  Created by MNN on 2019/02/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalConvolutionDepthwise_hpp
#define MetalConvolutionDepthwise_hpp

#import "MetalConvolutionCommon.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

class MetalConvolutionDepthwise : public MetalConvolutionCommon {
public:
    MetalConvolutionDepthwise(Backend *backend, const MNN::Op *op);
    virtual ~MetalConvolutionDepthwise() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

protected:
    virtual void onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) override;
    virtual std::shared_ptr<MNN::Tensor> weightTransform(int group, int oc, int ic, int kh, int kw, const float *src, bool int8Weight=false, bool int4Weight=false) override;
private:
    id<MTLComputePipelineState> mPipeline;
    std::pair<MTLSize, MTLSize> mThreads;
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */

#endif /* MetalConvolutionDepthwise_hpp */
