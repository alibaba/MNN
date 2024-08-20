//
//  FmhaV2Execution.hpp
//  MNN
//
//  Created by MNN on 2024/06/03.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#ifndef SelfAttentionBufExecution_hpp
#define SelfAttentionBufExecution_hpp

#include "backend/opencl/execution/image/CommonExecution.hpp"

namespace MNN {
namespace OpenCL {

class SelfAttentionBufImpl {
public:
    SelfAttentionBufImpl(const MNN::Op *op, Backend *backend);

    ~SelfAttentionBufImpl() = default;
    ErrorCode onResize(Backend *backend, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onExecute(Backend *backend, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);

private:
    int getLocalSize(int size, int maxGroupSize);
    float mScale;
    int mQseqSplitNum = 1;
    std::shared_ptr<Tensor> mTempQ, mTempK, mTempQK, mTempSoftMax, mTempV, mTempQKV;
    std::shared_ptr<Tensor> mTempTrans;
    int mNumHead = 0, mHeadDim = 0;
    std::vector<std::shared_ptr<KernelWrap>> mKernel_split;
    std::vector<std::shared_ptr<KernelWrap>> mKernel_qk;
    std::vector<std::shared_ptr<KernelWrap>> mKernel_softmax;
    std::vector<std::shared_ptr<KernelWrap>> mKernel_qkv;
    std::vector<std::shared_ptr<KernelWrap>> mKernel_clip;
    std::vector<std::shared_ptr<KernelWrap>> mKernel_trans;
    std::vector<std::vector<uint32_t>> mGlobalWorkSizeSplit;
    std::vector<std::vector<uint32_t>> mLocalWorkSizeSplit;
    std::vector<std::vector<uint32_t>> mGlobalWorkSizeClip;
    std::vector<std::vector<uint32_t>> mLocalWorkSizeClip;
    std::vector<std::vector<uint32_t>> mGlobalWorkSizeQk;
    std::vector<std::vector<uint32_t>> mLocalWorkSizeQk;
    std::vector<std::vector<uint32_t>> mGlobalWorkSizeSoftMax;
    std::vector<std::vector<uint32_t>> mLocalWorkSizeSoftMax;
    std::vector<std::vector<uint32_t>> mGlobalWorkSizeQkv;
    std::vector<std::vector<uint32_t>> mLocalWorkSizeQkv;
    std::vector<std::vector<uint32_t>> mGlobalWorkSizeTrans;
    std::vector<std::vector<uint32_t>> mLocalWorkSizeTrans;
    uint32_t mMaxWorkGroupSize;
    OpenCLBackend *mOpenCLBackend;
    size_t mQkGlobal_size[3];
    int mSoftmaxShape[4];
    int mByte = 4;
    cl_recording_qcom mRecording{NULL};

};

class SelfAttentionBufExecution : public CommonExecution {
public:
    SelfAttentionBufExecution(const MNN::Op *op, Backend *backend);
    SelfAttentionBufExecution(std::shared_ptr<SelfAttentionBufImpl> impl, const MNN::Op *op, Backend *backend);

    virtual ~SelfAttentionBufExecution() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;

private:
    std::shared_ptr<SelfAttentionBufImpl> mImpl;
};
} // namespace OpenCL
} // namespace MNN
#endif /* SelfAttentionBufExecution_hpp */
#endif/* MNN_SUPPORT_TRANSFORMER_FUSE */
