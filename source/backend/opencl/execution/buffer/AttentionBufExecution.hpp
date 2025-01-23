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

namespace MNN {
namespace OpenCL {

class KVCacheCLManager {
public:
    KVCacheCLManager(Backend *backend, bool kv_cache);

    ~KVCacheCLManager() = default;
    void allocKVCache();
    bool reallocKVCache();
    void setArgs(int pastLength, int numHead, int kvNumHead, int headDim){
        mPastLength = pastLength;
        mNumHead = numHead;
        mKvNumHead = kvNumHead;
        mHeadDim = headDim;
    }
    int kvLength() {
        return mPastLength;
    }
    void addKvLength(){
        mPastLength += 1;
    }
    int maxLength() {
        return mMaxLength;
    }
    int numHead() {
        return mNumHead;
    }
    const cl::Buffer * key() {
        return mPastKey.get();
    }
    const cl::Buffer * value() {
        return mPastValue.get();
    }

private:
    bool mKVCache;
    const int mExpandChunk = 2048;
    std::shared_ptr<cl::Buffer> mPastKey, mPastValue;
    int mPastLength = 0, mMaxLength = 0, mNumHead = 0, mKvNumHead = 0, mHeadDim = 0;
    OpenCLBackend *mOpenCLBackend;
    int mByte = 4;
};

class AttentionBufExecution : public CommonExecution {
public:
    AttentionBufExecution(const MNN::Op *op, Backend *backend, bool kv_cache);
    AttentionBufExecution(std::shared_ptr<KVCacheCLManager> manager, const MNN::Op *op, Backend *backend);
    ErrorCode longPrefillResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode UpdateArgs(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);

    virtual ~AttentionBufExecution() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;

private:
    
    int getLocalSize(int size, int maxGroupSize);
    bool mIsDecode = false;
    bool mIsFirstPrefill = true;
    int mKv_seq_len = 0;
    int mKeyValueMaxlen = 0;
    int mDecodeTmpMaxlen = 0;
    std::shared_ptr<KernelWrap> mKernel_qk;
    std::shared_ptr<KernelWrap> mKernel_softmax;
    std::shared_ptr<KernelWrap> mKernel_qkv;
    std::vector<uint32_t> mGlobalWorkSizeQk{1, 1, 1};
    std::vector<uint32_t> mLocalWorkSizeQk{1, 1, 1, 1};
    std::vector<uint32_t> mGlobalWorkSizeSoftMax{1, 1, 1};
    std::vector<uint32_t> mLocalWorkSizeSoftMax{1, 1, 1, 1};
    std::vector<uint32_t> mGlobalWorkSizeQkv{1, 1, 1};
    std::vector<uint32_t> mLocalWorkSizeQkv{1, 1, 1, 1};
    uint32_t mMaxWorkGroupSize;
    OpenCLBackend *mOpenCLBackend;
    RecordUpdateInfo mRgUpdateInfo;
    RecordUpdateInfo mQkUpdateInfo;
    RecordUpdateInfo mSoftMaxUpdateInfo;
    RecordUpdateInfo mRgVUpdateInfo;
    RecordUpdateInfo mQkvUpdateInfo;
    int mGlobalWorkSizeQk0 = 0;
    size_t mQkGlobal_size[2];
    std::vector<RecordUpdateInfo*> mOpRecordUpdateInfo;
    std::shared_ptr<KVCacheCLManager> mKVCacheCLManager;
    std::shared_ptr<Tensor> mTempQK, mTempSoftMax;
private:
    int mAlignQ, mAlignKV, mAlignHDK, mAlignHDN;
    bool mLongPrefill = false;
    std::shared_ptr<KernelWrap> mKernel_rearrangeQ;
    std::vector<uint32_t> mGlobalWorkSizeRearrgQ{1, 1, 1};
    std::vector<uint32_t> mLocalWorkSizeRearrgQ{1, 1, 1, 1};
    std::shared_ptr<KernelWrap> mKernel_rearrangeV;
    std::vector<uint32_t> mGlobalWorkSizeRearrgV{1, 1, 1};
    std::vector<uint32_t> mLocalWorkSizeRearrgV{1, 1, 1, 1};
    std::shared_ptr<KernelWrap> mKernel_rearrange;
    std::vector<uint32_t> mGlobalWorkSizeRearrg{1, 1, 1};
    std::vector<uint32_t> mLocalWorkSizeRearrg{1, 1, 1, 1};
    std::shared_ptr<KernelWrap> mKernel_mask;
    std::vector<uint32_t> mGlobalWorkSizeMask{1, 1, 1};
    std::vector<uint32_t> mLocalWorkSizeMask{1, 1, 1, 1};
    std::shared_ptr<KernelWrap> mKernel_trans;
    std::vector<uint32_t> mGlobalWorkSizeTrans{1, 1, 1};
    std::vector<uint32_t> mLocalWorkSizeTrans{1, 1, 1, 1};
    std::shared_ptr<KernelWrap> mKernel_clip;
    std::vector<uint32_t> mGlobalWorkSizeClip{1, 1, 1};
    std::vector<uint32_t> mLocalWorkSizeClip{1, 1, 1, 1};
    std::shared_ptr<Tensor> mTempQ, mTempK, mTempV, mTempMask, mTempQKV;
    bool mIsAddMask = false;
};
} // namespace OpenCL
} // namespace MNN
#endif /* AttentionBufExecution_hpp */
#endif/* MNN_SUPPORT_TRANSFORMER_FUSE */
