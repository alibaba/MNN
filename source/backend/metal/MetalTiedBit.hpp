// Copyright (c) Alibaba, Inc. and its affiliates.
#ifndef MetalTiedBit_hpp
#define MetalTiedBit_hpp

#include "MetalConvolutionCommon.hpp"
#if MNN_METAL_ENABLED
namespace MNN {

// Decode-only approximate output projection. Immutable resources are shared by
// clones; selector scratch belongs to each execution. Lossless Q1 + Q234
// storage supports Q1 screening followed by original-Q4 candidate refinement.
// Both stages reuse the original scale/center, also shared with embedding.
class MetalTiedBit : public MetalConvolutionCommon {
public:
    // Return nullptr to continue standard convolution selection.
    static Execution* create(MetalBackend* backend, const Op* op);
    virtual ~MetalTiedBit() = default;
    virtual bool onClone(Backend* backend, const Op* op, Execution** dst) override;
    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    virtual void onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                          id<MTLComputeCommandEncoder> encoder) override;

protected:
    // The common encode recorder does not support indirect dispatch.
    virtual bool canRecordEncode() const override { return false; }

private:
    MetalTiedBit(Backend* backend, const Op* op, std::shared_ptr<Tensor> bias = nullptr);
    enum Stage { Prepare, Scan, High, SelectHigh, Low, SelectLow, Compact, Exact, StageCount };
    struct ThreadConfig {
        int scanSimdGroups = 2;
        int rerankSimdGroups = 8;
        int histogramChunk = 4096;
    };
    ThreadConfig mThreads;
    int mInputChannels = 0, mVocab = 0, mBlockQuads = 0;
    id<MTLComputePipelineState> mPipelines[StageCount] = {};
    id<MTLBuffer> mSums = nil, mIndices = nil, mCount = nil, mArgs = nil;
    id<MTLBuffer> mHistogram = nil, mThreshold = nil, mParam = nil;
};

} // namespace MNN
#endif
#endif
