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
        int mGammaSize = 0;
        std::shared_ptr<Tensor> mGammaBuffer;
        std::shared_ptr<Tensor> mBetaBuffer;
    };
    MetalLayerNorm(Backend *backend, std::shared_ptr<Resource> res);
    virtual ~MetalLayerNorm() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual void onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) override;
    static std::shared_ptr<Resource> makeResource(Backend *backend, const LayerNorm *layernorm);
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;

    // Used by an owning fused-projection op that folds this LayerNorm into its
    // own GEMV dispatch: it needs gamma/eps to bind, and marks the LayerNorm
    // fused so it skips dispatching.
    std::shared_ptr<Tensor> getGamma() const { return mResource->mGammaBuffer; }
    float getEps() const { return mResource->mEps; }
    bool isRMSNormWithGammaBeta() const { return mResource->mRMSNorm && mResource->mHasGammaBeta; }
    bool isNC4HW4() const { return mIsNC4HW4; }
    void setFused() { mIsFused = true; }

private:
    int mOutside;
    int mInside;
    bool mIsNC4HW4 = false;
    bool mIsBinaryNCHW = false;
    int mChannelUnit;
    bool mIsFused = false;  // set by the owning fused-proj op or the gated-norm fold
    std::shared_ptr<Resource> mResource;
    id<MTLBuffer> mShapeBuffer;
    id<MTLComputePipelineState> mPipeline;
    std::pair<MTLSize, MTLSize> mThreads;
    
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
#endif /* MetalLayerNorm_hpp */
