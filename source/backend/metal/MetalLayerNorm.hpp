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
    struct Resource {
        int mGroup = 1;
        float mEps;
        int mAxisSize;

        bool mHasGammaBeta = false;
        bool mRMSNorm = false;
        std::shared_ptr<Tensor> mGammaBuffer;
        std::shared_ptr<Tensor> mBetaBuffer;
    };
    MetalLayerNorm(Backend *backend, std::shared_ptr<Resource> res);
    virtual ~MetalLayerNorm() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual void onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) override;
    static std::shared_ptr<Resource> makeResource(Backend *backend, const LayerNorm *layernorm);
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;

private:
    int mOutside;
    int mInside;
    std::shared_ptr<Resource> mResource;
    id<MTLBuffer> mShapeBuffer;
    id<MTLComputePipelineState> mPipeline;
    std::pair<MTLSize, MTLSize> mThreads;
    
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
#endif /* MetalLayerNorm_hpp */
