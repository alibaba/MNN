//
//  AttentionBufExecution.hpp
//  MNN
//
//  Created by MNN on 2024/04/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#ifndef AttentionBufExecution_hpp
#define AttentionBufExecution_hpp

#include "backend/opencl/execution/image/CommonExecution.hpp"
#include "core/OpCommonUtils.hpp"

namespace MNN {
namespace OpenCL {

class KVCacheCLManager {
public:
    KVCacheCLManager(Backend* backend, bool kv_cache);

    ~KVCacheCLManager() = default;
    void allocKVCache(const KVMeta* meta, int seqlen);
    bool reallocKVCache(const KVMeta* meta, int seqlen, bool isExecute = true);
    void setArgs(int numHead, int kvNumHead, int headDim) {
        mNumHead = numHead;
        mKvNumHead = kvNumHead;
        mHeadDim = headDim;
    }
    int pastKvLength() { return mPastLength; }
    void addKvLength(int seq_len) { mPastLength += seq_len; }
    int maxLength() { return mMaxLength; }
    int numHead() { return mNumHead; }
    const cl::Buffer* key() { return mPastKey.get(); }
    const cl::Buffer* value() { return mPastValue.get(); }

    // Called after allocKVCache completes reallocKVCache in resize phase.
    // onExecute checks this to avoid double-executing realloc/Remove.
    bool isReallocDone() const { return mReallocDone; }
    void clearReallocDone() { mReallocDone = false; }

    // Prefix kvcache (share prompt kvcache on disk). Set by AttentionBufExecution
    // from the backend runtime hint. Empty means the feature is disabled.
    void setPrefixCacheDir(const std::string& dir) { mPrefixCacheDir = dir; }
    // True after allocKVCache detected PendingWrite for this layer: onExecute must
    // dump the prefill kvcache to disk once the kernels have run.
    bool savingPrefix() const { return mSaveShareKvPrefix; }
    // Load the per-layer prefix kvcache files into a freshly allocated cache buffer.
    // Returns false (and leaves the cache unallocated) when the files are missing or
    // inconsistent with the current precision, so the caller can fall back to prefill.
    bool loadPrefixKVCache(const KVMeta* meta, int seqlen);
    // Dump the valid [0, mPastLength) kvcache region to the per-layer prefix files.
    void savePrefixKVCache();

private:
    bool mKVCache;
    bool mReallocDone = false;
    const int mExpandChunk = 64;
    std::shared_ptr<cl::Buffer> mPastKey, mPastValue;
    int mPastLength = 0, mMaxLength = 0, mNumHead = 0, mKvNumHead = 0, mHeadDim = 0;
    OpenCLBackend* mOpenCLBackend;
    int mByte = 4;

    // Prefix kvcache state
    std::string mPrefixCacheDir;     // Directory holding <name>_<layer>.k/.v files
    bool mSaveShareKvPrefix = false; // This layer is in PendingWrite mode
    std::string mBasePrefixFileName; // <dir>/<name>_<layer> for this layer (no suffix)
};

class AttentionBufExecution : public CommonExecution {
public:
    AttentionBufExecution(const MNN::Op* op, Backend* backend, bool outputC4);
    AttentionBufExecution(std::shared_ptr<KVCacheCLManager> manager, const MNN::Op* op, Backend* backend);
    ErrorCode longPrefillResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs);
    ErrorCode flashPrefillResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs);
    // Flash-decoding: split the kv axis across workgroups so the P*V matmul stops running
    // on ceil(headDim/8) * headNum work-items. The only decode path.
    ErrorCode flashDecodeResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs);

    ErrorCode UpdateArgs(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs);
    ErrorCode init();
    int getExecuteTime();
    int measureExecuteTime();
    virtual ~AttentionBufExecution() = default;
    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;
    virtual void prebuildOpenCLPrograms(const std::vector<Tensor*>& inputs,
                                        const std::vector<Tensor*>& outputs) override;

private:
    bool mOutputC4 = false;
    float mAttnScale = 0.0f;
    KVMeta* mMeta;
    int getLocalSize(int size, int maxGroupSize);
    bool mIsDecode = false;
    void handleKVCache(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs);
    int mPastKvSeqlen = 0;
    int mKvSeqlen = 0;
    int mKeyValueMaxlen = 0;
    int mDecodeTmpMaxlen = 0;

    uint32_t mMaxWorkGroupSize;
    OpenCLBackend* mOpenCLBackend;
    RecordUpdateInfo mRgUpdateInfo;
    RecordUpdateInfo mRgVUpdateInfo;
    std::vector<RecordUpdateInfo*> mOpRecordUpdateInfo;
    std::shared_ptr<KVCacheCLManager> mKVCacheCLManager;
    std::shared_ptr<Tensor> mTempQK, mTempSoftMax;

private:
    // Fused flash-attention prefill: one kernel for qk + mask + softmax + qkv.
    bool flashPrefillEligible(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs);
    // Shared by the real build and the work-group probe that runs before anything is recorded,
    // so the two cannot drift apart and select different program variants.
    std::set<std::string> flashPrefillBuildOptions(int headDim, int groupSize) const;
    bool mFlashPrefill = false;
    int mFaTileQ = 0, mFaTileKV = 0, mFaWgSize = 0;
    // Every legal tiling for this shape, preferred one first and the rest by shrinking work
    // group. flashPrefillEligible sizes them against the device limit; flashPrefillResize walks
    // the list until one clears the kernel's own CL_KERNEL_WORK_GROUP_SIZE.
    struct FaTiling {
        int tileQ;
        int wgSize;
        int tileKV;
    };
    std::vector<FaTiling> mFaTilings;
    std::shared_ptr<KernelWrap> mKernel_fa;
    std::vector<uint32_t> mGlobalWorkSizeFa;
    std::vector<uint32_t> mLocalWorkSizeFa;
    RecordUpdateInfo mFaUpdateInfo;

    // kv entries per workgroup in the flash-decode partial pass. A compile-time macro on the
    // kernel side, so it cannot be re-chosen per step -- only num_chunk and the global size get
    // patched as kv grows.
    static constexpr int kFdChunk = 64;
    int mFdWgSize = 0, mFdChunk = 0, mFdNumChunk = 0, mFdMaxChunk = 0;
    std::shared_ptr<KernelWrap> mKernel_fdPartial, mKernel_fdReduce;
    std::vector<uint32_t> mGwsFdPartial, mLwsFdPartial, mGwsFdReduce, mLwsFdReduce;
    RecordUpdateInfo mFdPartialUpdateInfo, mFdReduceUpdateInfo;
    size_t mFdPartialGlobal_size[2];
    std::shared_ptr<Tensor> mTempPartialO, mTempPartialML;

private:
    int mAlignQ, mAlignKV, mAlignHDK, mAlignHDN;
    bool mLongPrefill = false;
    int mQseqSplitNum = 1;
    std::shared_ptr<Tensor> mTempQ, mTempK, mTempV, mTempMask, mTempQKV;
    bool mIsAddMask = false;
    bool mNeedKvCache = true;
    bool mHasMask = false;

private:
    std::vector<std::shared_ptr<KernelWrap>> mKernel_rearrange_vec;
    std::vector<std::shared_ptr<KernelWrap>> mKernel_mask_vec;
    std::vector<std::shared_ptr<KernelWrap>> mKernel_trans_vec;
    std::vector<std::shared_ptr<KernelWrap>> mKernel_clip_vec;
    std::vector<std::shared_ptr<KernelWrap>> mKernel_qk_vec;
    std::vector<std::shared_ptr<KernelWrap>> mKernel_softmax_vec;
    std::vector<std::shared_ptr<KernelWrap>> mKernel_qkv_vec;

    std::vector<std::vector<uint32_t>> mGwsQkVec;
    std::vector<std::vector<uint32_t>> mLwsQkVec;
    std::vector<std::vector<uint32_t>> mGwsSoftMaxVec;
    std::vector<std::vector<uint32_t>> mLwsSoftMaxVec;
    std::vector<std::vector<uint32_t>> mGwsQkvVec;
    std::vector<std::vector<uint32_t>> mLwsQkvVec;
    std::vector<std::vector<uint32_t>> mGwsRearrgVec;
    std::vector<std::vector<uint32_t>> mLwsRearrgVec;
    std::vector<std::vector<uint32_t>> mGwsMaskVec;
    std::vector<std::vector<uint32_t>> mLwsMaskVec;
    std::vector<std::vector<uint32_t>> mGwsTransVec;
    std::vector<std::vector<uint32_t>> mLwsTransVec;
    std::vector<std::vector<uint32_t>> mGwsClipVec;
    std::vector<std::vector<uint32_t>> mLwsClipVec;

private:
    // rearrange_k / rearrange_v, shared by flash prefill and flash decode.
    std::shared_ptr<KernelWrap> mKernel_rearrange;
    std::shared_ptr<KernelWrap> mKernel_rearrangeV;

    std::vector<uint32_t> mGlobalWorkSizeRearrg;
    std::vector<uint32_t> mLocalWorkSizeRearrg;
    std::vector<uint32_t> mGlobalWorkSizeRearrgV;
    std::vector<uint32_t> mLocalWorkSizeRearrgV;
};
} // namespace OpenCL
} // namespace MNN
#endif /* AttentionBufExecution_hpp */
#endif /* MNN_SUPPORT_TRANSFORMER_FUSE */
