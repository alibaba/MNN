//
//  MetalConvolution.hpp
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalConvolution_hpp
#define MetalConvolution_hpp

#import "MetalConvolutionCommon.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

class MetalConvolution : public MetalConvolutionCommon {
public:
    MetalConvolution(Backend *backend, const MNN::Op *op);
    virtual ~MetalConvolution() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

protected:
    virtual ErrorCode onFloat(const Tensor *input, const Tensor *output) override;

private:
    int mThreadgroupMemory = 0;
    id<MTLComputePipelineState> mPipeline;
    std::pair<MTLSize, MTLSize> mThreads;
    MTLSize local;
    std::pair<MTLSize, MetalBandwidth> _mThreads;

};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
#endif /* MetalConvolution_hpp */
