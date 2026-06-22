//
//  MetalAttention.mm
//  MNN
//
//  Created by MNN on b'2024/04/29'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalAttention_hpp
#define MetalAttention_hpp

#import "core/Macro.h"
#import "MetalBackend.hpp"
#include "MNN_generated.h"
#include "core/OpCommonUtils.hpp"
#include "MetalKVCacheManager.hpp"

#if MNN_METAL_ENABLED
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

namespace MNN {
class AttentionBufExecution : public MetalExecution {
public:
    AttentionBufExecution(Backend* backend, bool kvCache, bool outputC4, float attnScale,
                          std::shared_ptr<KVQuantParameter> kvQuantParam);
    virtual ~AttentionBufExecution() = default;
    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;

    virtual void onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                          id<MTLComputeCommandEncoder> encoder) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override {
        if (nullptr == dst) {
            return true;
        }
        auto exe = new AttentionBufExecution(bn, mKVCache, mOutputC4, mAttnScale, mKVQuantParameter);
        if (mKVCache && bn->getMetaPtr() == mMeta && mMeta != nullptr) {
            exe->mKVCacheManager = mKVCacheManager;
        }
        *dst = exe;
        return true;
    }

private:
    void _init();
    void compilerShader(const std::vector<Tensor*>& inputs);
    void handleKVAllocMemory();
    bool mKVCache = true;
    std::shared_ptr<MetalKVCacheManager> mKVCacheManager = nullptr;
    float mAttnScale = 0.0f;
    float mScale;
    bool mOutputC4 = false;
    bool mShortSeq = false;
    std::shared_ptr<Tensor> mTempQK, mTempSoftMax;
    int mNumHead = 0, mHeadDim = 0, mValueH = 0, mKvNumHead = 0;
    int mSeqLen;
    // for simd/tensor maxtrix load alignment
    int mKvAlignNum = 32;
    id<MTLComputePipelineState> mKernel_softmax = nil;

    id<MTLComputePipelineState> mKernel_qk = nil;
    id<MTLComputePipelineState> mKernel_qkv = nil;
    id<MTLComputePipelineState> mKernel_copy = nil;
    id<MTLComputePipelineState> mKernel_qk_softmax = nil;
    id<MTLComputePipelineState> mKernelPrefill_qk = nil;
    id<MTLComputePipelineState> mKernelPrefill_qkv = nil;
    id<MTLBuffer> mParamQKV;
    id<MTLBuffer> mParamSoftmax;
    id<MTLBuffer> mParamCopy;

private:
    KVMeta* mMeta;
    bool mQkSimdReduce = false;
    bool mQkSimdMatrix = false;
    bool mQkTensorMatrix = false;
    bool mSftmSimdReduce = false;
    bool mQkvSimdReduce = false;
    bool mQkvSimdMatrix = false;
    bool mDecodeQkSoftmax = false;
    bool mCopySimdReduce = false;

private:
    bool mHasMask = false;
    bool mIsAddMask = false;
    int mBatch, mKvSeqLen, mKvMaxLen, mCurrentKvLen = 0;
    int mQseqSplitNum = 1;
    std::shared_ptr<Tensor> mTempK, mTempV;
    bool mKvInDisk;

    // KV static quantization (only V is quantized on Metal)
    std::shared_ptr<KVQuantParameter> mKVQuantParameter = nullptr;
    bool mQuantValue = false; // whether V cache is stored as int8
    bool mQuantKey = false;   // whether K cache is stored as int8
};

} // namespace MNN
#endif /* MNN_SUPPORT_TRANSFORMER_FUSE */
#endif /* MNN_METAL_ENABLED */
#endif /* MetalAttention_hpp */
