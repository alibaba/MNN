//
//  MetalInterp.hpp
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalInterp_hpp
#define MetalInterp_hpp

#include "MetalExecution.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

class MetalInterp : public MetalExecution {
public:
    MetalInterp(Backend *backend, const Op* op);
    virtual ~MetalInterp() = default;
    virtual void onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    int32_t mReiszeType;
    id<MTLBuffer> mCordTransform;
    id<MTLBuffer> mShape;
    id<MTLComputePipelineState> mPipeline;
    std::pair<MTLSize, MTLSize> mThreads;
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
#endif /* MetalInterp_hpp */
