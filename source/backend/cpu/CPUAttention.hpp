//
//  CPUAttention.hpp
//  MNN
//
//  Created by MNN on 2024/03/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#ifndef CPUATTENTION_HPP
#define CPUATTENTION_HPP

#include <functional>
#include "core/Execution.hpp"
#include "core/OpCommonUtils.hpp"
#include "CPUKVCacheManager.hpp"
#include "MNN/ErrorCode.hpp"

namespace MNN {

class CPUAttention : public Execution {
public:
    CPUAttention(Backend *backend, bool kv_cache, int layerIndex = -1, int kvSharedLayerIndex = -1);
    virtual ~CPUAttention();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;
    CPUKVCacheManager* getKVCacheManager() const { return mKVCacheManager.get(); }
    int layerIndex() const { return mLayerIndex; }
private:
    bool mKVCache        = true;
    bool mIsKVShared     = false;
    int mLayerIndex      = -1;
    int mKVSharedLayerIndex = -1;
    CPUKVCacheManager* mSharedKVCache = nullptr;
    int mBytes = 4;
    int mThreadNum = 1;
    int mBlockKV = 512;
    int eP, lP, hP, mPack; // float matmul packing
    int eP8, lP8, hP8;    // GemmInt8 packing
    int mNumHead, mKvNumHead, mHeadDim;
    KVMeta* mMeta;

    // common
    std::shared_ptr<Tensor> mPackQ, mPackQKV, mRunningMax, mRunningSum, mTempQKBlock, mTempOut, mExpfDiffMax;
    std::shared_ptr<CPUKVCacheManager> mKVCacheManager = nullptr;
    bool mUseFlashAttention = true;

    // KV cache quantization mode
    KVQuantMode mKeyQuantMode = KVQuantMode::None;
    KVQuantMode mValueQuantMode = KVQuantMode::None;
    std::shared_ptr<Tensor> mTQ3DequantBuf; // shared by TQ3 and TQ4
    int  mBlockNum   = 1;
    MemChunk mSumQ;
    MemChunk mQueryScale, mQueryZeroPoint, mQueryQuantScale, mQueryQuantZero;
    MemChunk mQuantQuery, mAccumBuffer;

    MemChunk mQuantQK, mQKScale, mQKBias, mSumQK, mArray;
    AutoStorage<int8_t> mGemmBias, mGemmRelu;

    std::function<void(const float*, int8_t*, size_t, const float*, ssize_t, ssize_t, const float*, ssize_t)> mQuantFunc;
    decltype(CoreInt8Functions::Int8GemmKernel) mInt8GemmKernel;
};

} // namespace MNN

#endif // CPUATTENTION_HPP

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
