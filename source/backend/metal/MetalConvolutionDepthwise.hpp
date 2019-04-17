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
    virtual ErrorCode onQuantized(const Tensor *input, const Tensor *output) override;
    virtual ErrorCode onFloat(const Tensor *input, const Tensor *output) override;
    virtual id<MTLBuffer> weightForQuantized(int group, int oc, int ic, int kh, int kw, const int8_t *src) override;
    virtual id<MTLBuffer> weightForFloat(int group, int oc, int ic, int kh, int kw, const float *src) override;
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */

#endif /* MetalConvolutionDepthwise_hpp */
