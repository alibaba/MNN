//
//  MetalConvolution1x1.hpp
//  MNN
//
//  Created by MNN on 2019/02/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalConvolution1x1_hpp
#define MetalConvolution1x1_hpp

#import "MetalConvolutionCommon.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

class MetalConvolution1x1 : public MetalConvolutionCommon {
public:
    static bool isValid(const Convolution2D *conv, const Tensor *input);
    MetalConvolution1x1(Backend *backend, const MNN::Op *op);
    virtual ~MetalConvolution1x1() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

protected:
    virtual ErrorCode onQuantized(const Tensor *input, const Tensor *output) override;
    virtual ErrorCode onFloat(const Tensor *input, const Tensor *output) override;
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */

#endif /* MetalConvolution1x1_hpp */
