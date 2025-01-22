//
//  MetalPReLU.hpp
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalPReLU_hpp
#define MetalPReLU_hpp

#import "MetalExecution.hpp"
#if MNN_METAL_ENABLED
#include "core/BufferAllocator.hpp"
namespace MNN {

class MetalPReLU : public MetalExecution {
public:
    MetalPReLU(Backend *backend, const float *slope, int count);
    virtual ~MetalPReLU();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual void onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) override;

private:
    MemChunk mSlope;
    id<MTLBuffer> mShape;
    id<MTLComputePipelineState> mPipeline;
    std::pair<MTLSize, MTLSize> mThreads;
    bool mShareChannel = false;
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
#endif /* MetalPReLU_hpp */
