//
//  LoopBufExecution.hpp
//  MNN
//
//  Created by MNN on 2023/04/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED

#ifndef LoopBufExecution_hpp
#define LoopBufExecution_hpp

#include "backend/opencl/execution/image/CommonExecution.hpp"

namespace MNN {
namespace OpenCL {

class LoopGatherBufExecution : public CommonExecution {
public:
    LoopGatherBufExecution(const LoopParam *loop, const MNN::Op *op, Backend *bn);
    virtual ~LoopGatherBufExecution() = default;
    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    const LoopParam *mLoop;
    std::vector<Tensor *> mTensors;
    std::vector<std::shared_ptr<Tensor>> mTmpTensors;
    std::vector<std::shared_ptr<Tensor>> mOffsetTensors;
    int mStride_src[4];
    int mStride_dst[4];
    int mStep[2];
    int mIter[2];
    std::set<std::string> mBuildOptions;
};

class LoopBatchMatMulBufExecution : public CommonExecution {
public:
    LoopBatchMatMulBufExecution(const LoopParam *loop, const MNN::Op *op, Backend *bn);
    virtual ~LoopBatchMatMulBufExecution() = default;
    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;


private:
    const LoopParam *mLoop;
    std::vector<Tensor *> mTensors;
    std::vector<std::shared_ptr<Tensor>> mTmpTensors;
    std::vector<std::shared_ptr<Tensor>> mOffsetTensors;
    int mOffset[4];
    int mStep[4];
    int mIter[4];
    bool mHasBias = false;
    bool mTransposeA = false;
    bool mTransposeB = false;
    std::set<std::string> mBuildOptions;
    bool mBatchGemmOpt = false;
    int mBatch, mM, mN, mK;
};


class LoopBinaryBufExecution : public CommonExecution {
public:
    LoopBinaryBufExecution(const LoopParam *loop, const std::string &compute, const MNN::Op *op, Backend *bn);
    virtual ~LoopBinaryBufExecution() = default;
    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    const LoopParam *mLoop;
    std::vector<Tensor *> mTensors;
    std::set<std::string> mBuildOptions;
};

} // namespace OpenCL
} // namespace MNN
#endif /* LoopBufExecution_hpp */
#endif /* MNN_OPENCL_BUFFER_CLOSED */
