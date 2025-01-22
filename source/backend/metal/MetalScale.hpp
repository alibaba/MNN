//
//  MetalScale.hpp
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalScale_hpp
#define MetalScale_hpp

#import "MetalExecution.hpp"
#import "MNN_generated.h"

#if MNN_METAL_ENABLED
#include "core/BufferAllocator.hpp"
namespace MNN {

class MetalScale : public MetalExecution {
public:
    MetalScale(Backend *backend, const Scale *scale);
    virtual ~MetalScale();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual void onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) override;

private:
    MemChunk mScaleBias;
    size_t mBiasOffset = 0;
    id<MTLBuffer> mConst;
    id<MTLComputePipelineState> mPipeline;
    std::pair<MTLSize, MTLSize> mThreads;
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
#endif /* MetalScale_hpp */
