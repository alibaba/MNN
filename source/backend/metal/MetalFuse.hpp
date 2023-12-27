//
//  MetalFuse.hpp
//  MNN
//
//  Created by MNN on 2022/11/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalFuse_hpp
#define MetalFuse_hpp

#import "MetalExecution.hpp"
#import "MNN_generated.h"

#if MNN_METAL_ENABLED
namespace MNN {

class MetalFuse : public MetalExecution {
public:
    MetalFuse(Backend *backend, const Op* op);
    virtual ~MetalFuse() = default;
    virtual void onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    const Op* mOp;
    id<MTLBuffer> mConstBuffer;
    id<MTLComputePipelineState> mPipeline;
    std::pair<MTLSize, MTLSize> mThreads;
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
#endif /* MetalFuse_hpp */
