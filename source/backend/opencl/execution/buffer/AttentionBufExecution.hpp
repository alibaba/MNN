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

class AttentionBufImpl {
public:
    AttentionBufImpl(const MNN::Op *op, Backend *backend, bool kv_cache);

    ~AttentionBufImpl() {
        if(mRecording != NULL){
#ifdef MNN_USE_LIB_WRAPPER
            clReleaseRecordingQCOM(mRecording);
#endif
        }
    }
    ErrorCode onResize(Backend *backend, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onExecute(Backend *backend, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);

private:
    int getLocalSize(int size, int maxGroupSize);
    void allocKVCache();
    void reallocKVCache();
    bool mKVCache;
    float mScale;
    const int mExpandChunk = 2048;
    bool mIsDecode = false;
    bool mIsFirstDecode = true;
    int mPastLength = 0, mMaxLength = 0, mKv_seq_len = 0, mSoftMaxRemainChannels = 0;
    std::shared_ptr<cl::Buffer> mPastKey, mPastValue;
    std::shared_ptr<Tensor> mTempQK, mTempSoftMax;
    int mNumHead = 0, mKvNumHead = 0, mHeadDim = 0, mValueH = 0;
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
    RecordUpdateInfo mQkUpdateInfo;
    RecordUpdateInfo mSoftMaxUpdateInfo;
    RecordUpdateInfo mQkvUpdateInfo;
    int mGlobalWorkSizeQk2 = 0;
    size_t mQkGlobal_size[3];
    int mSoftmaxShape[4];
    cl_recording_qcom mRecording{NULL};
    std::vector<RecordUpdateInfo*> mOpRecordUpdateInfo;
    int mByte = 4;
};

class AttentionBufExecution : public CommonExecution {
public:
    AttentionBufExecution(const MNN::Op *op, Backend *backend, bool kv_cache);
    AttentionBufExecution(std::shared_ptr<AttentionBufImpl> impl, const MNN::Op *op, Backend *backend);

    virtual ~AttentionBufExecution() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;

private:
    std::shared_ptr<AttentionBufImpl> mImpl;
};
} // namespace OpenCL
} // namespace MNN
#endif /* AttentionBufExecution_hpp */
#endif/* MNN_SUPPORT_TRANSFORMER_FUSE */
