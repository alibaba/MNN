//
//  MetalLayerNorm.hpp
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalLayerNorm_hpp
#define MetalLayerNorm_hpp

#import "MetalExecution.hpp"
#import "MNN_generated.h"

#if MNN_METAL_ENABLED
namespace MNN {

class MetalLayerNorm : public MetalExecution {
public:
    MetalLayerNorm(Backend *backend, const LayerNorm *layernorm);
    virtual ~MetalLayerNorm() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual void onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) override;

private:
    int mOutside;
    int mAxisSize;
    int mInside;
    int mGroup = 1;
    float mEps;
    
    bool has_gamma_beta_ = false;
    bool RMSNorm = false;
    id<MTLBuffer> mGammaBuffer = nil;
    id<MTLBuffer> mBetaBuffer = nil;
    id<MTLBuffer> mShapeBuffer;
    id<MTLComputePipelineState> mPipeline;
    std::pair<MTLSize, MTLSize> mThreads;
    
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
#endif /* MetalLayerNorm_hpp */
