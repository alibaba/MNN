//
//  LoopExecution.hpp
//  MNN
//
//  Created by MNN on 2023/05/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//


#ifndef LoopExecution_hpp
#define LoopExecution_hpp

#include "CommonExecution.hpp"

namespace MNN {
namespace OpenCL {

class LoopGatherExecution : public CommonExecution {
public:
    LoopGatherExecution(const LoopParam *loop, const MNN::Op *op, Backend *bn);
    virtual ~LoopGatherExecution() = default;
    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    ErrorCode InitCommandOnEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);

private:
    const LoopParam *mLoop;
    std::vector<Tensor *> mTensors;
    std::vector<cl::Buffer *> mTmpInitBuffers;
    std::vector<cl::Buffer *> mTmpBuffers;
    std::vector<cl::Buffer *> mOffsetBuffers;
    int mStride_src[4];
    int mStride_dst[4];
    int mStep[2];
    int mIter[2];
    std::set<std::string> mBuildOptions;
};

class LoopBatchMatMulExecution : public CommonExecution {
public:
    LoopBatchMatMulExecution(const LoopParam *loop, const MNN::Op *op, Backend *bn);
    virtual ~LoopBatchMatMulExecution() = default;
    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    const LoopParam *mLoop;
    std::vector<Tensor *> mTensors;
    std::vector<cl::Buffer*> mTmpBuffers;
    std::vector<cl::Buffer*> mOffsetBuffers;
    int mOffset[4];
    int mStep[4];
    int mIter[4];
    bool mHasBias = false;
    bool mTransposeA = false;
    bool mTransposeB = false;
    std::set<std::string> mBuildOptions;
};

class LoopBinaryExecution : public CommonExecution {
public:
    LoopBinaryExecution(const LoopParam *loop, const std::string &compute, const MNN::Op *op, Backend *bn);
    virtual ~LoopBinaryExecution() = default;
    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    ErrorCode cumSumOnEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);

private:
    int mOffset[4];
    int mStep[4];
    int mStride_src0[3];
    int mStride_src1[3];
    int mStride_dst[3];
    const LoopParam *mLoop;
    std::vector<Tensor *> mTensors;
    std::set<std::string> mBuildOptions;
    std::vector<cl::Buffer *> mTmpBuffers;
};

} // namespace OpenCL
} // namespace MNN
#endif /* LoopExecution_hpp */
