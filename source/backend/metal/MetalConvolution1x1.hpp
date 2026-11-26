//
//  MetalConvolution1x1.hpp
//  MNN
//
//  Created by MNN on 2019/02/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//
//  Float-weight conv1x1 executor. Weight-quantized residency lives in MetalConv1x1WQuant.
//

#ifndef MetalConvolution1x1_hpp
#define MetalConvolution1x1_hpp

#import "backend/metal/MetalConvolutionCommon.hpp"

#if MNN_METAL_ENABLED

namespace MNN {

class MetalConvolution1x1 : public MetalConvolutionCommon {
public:
    MetalConvolution1x1(Backend* backend, const MNN::Op* op);
    virtual ~MetalConvolution1x1() = default;
    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    void requestRowScale(const Tensor* scale) override { mRowScaleOffer = scale; }
    bool rowScaleFused() const override { return mPrepared && mFusedRowScale != nullptr; }
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;
    virtual void onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                          id<MTLComputeCommandEncoder> encoder) override;
    virtual bool onReplayUpdate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    // Encode call-list signature; public so tests can pin the guard without an encoder.
    uint64_t replayStructure(const std::vector<Tensor*>& inputs) const;

#if MNN_METAL_OP_PROFILE
    const std::string& profileTag() const { return mProfileTag; }
    id<MTLComputePipelineState> selectedPipeline() const { return mPipeline; }
    std::pair<MTLSize, MTLSize> dispatchThreads() const { return mThreads; }
#endif

private:
    ErrorCode resizeImpl(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs);
    ErrorCode prepareFloatConv(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs);
    MetalConvolution1x1(Backend* backend, const MNN::Op* op, std::shared_ptr<MNN::Tensor> weight,
                        std::shared_ptr<MNN::Tensor> bias);
    bool mPrepared = false;
    const Tensor* mRowScaleOffer = nullptr;
    const Tensor* mFusedRowScale = nullptr;
    id<MTLComputePipelineState> mPipeline = nil;
    std::pair<MTLSize, MTLSize> mThreads = {};
    uint64_t mRecordedStructure = 0;

#if MNN_METAL_OP_PROFILE
    std::string mProfileTag;
#endif
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */

#endif /* MetalConvolution1x1_hpp */