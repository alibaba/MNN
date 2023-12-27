//
//  MetalPooling.hpp
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalPooling_hpp
#define MetalPooling_hpp

#import "MetalExecution.hpp"
#import "MNN_generated.h"

#if MNN_METAL_ENABLED
namespace MNN {

class MetalPooling : public MetalExecution {
public:
    MetalPooling(Backend *backend, const Pool *pooling);
    virtual ~MetalPooling() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual void onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) override;

private:
    bool mGlobal;
    PoolType mPoolType;
    int mKernelX;
    int mKernelY;
    int mStrideX;
    int mStrideY;
    int mPadX;
    int mPadY;
    id<MTLBuffer> mConstBuffer;
    MTLSize mGroup;
    MTLSize mLocal;
    id<MTLComputePipelineState> mPipeline;
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
#endif /* MetalPooling_hpp */
