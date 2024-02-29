//
//  MetalUnary.hpp
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalUnary_hpp
#define MetalUnary_hpp

#import "MetalExecution.hpp"
#import "MNN_generated.h"

#if MNN_METAL_ENABLED
namespace MNN {

class MetalUnary : public MetalExecution {
public:
    MetalUnary(Backend *backend, id<MTLComputePipelineState> pipeline);
    virtual ~MetalUnary() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual void onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) override;

private:
    id<MTLBuffer> mConstBuffer;
    id<MTLComputePipelineState> mPipeline;
    std::pair<MTLSize, MTLSize> mThreads;
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
#endif /* MetalUnary_hpp */
