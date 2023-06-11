﻿//
//  LoopBufExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED

#include "backend/opencl/execution/buffer/LoopBufExecution.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {
namespace OpenCL {

static void _TileOrPackTensor(Tensor *input, Tensor *output, cl::Kernel& kernel, cl::NDRange &globalWorkSize,
                        cl::NDRange &localWorkSize, const int Width, const int Height, const int Channel,
                        const int Batch, OpenCLRuntime *runTime, const std::string &KernelName, const std::set<std::string> &buildOptions) {
    kernel = runTime->buildKernel("loop_buf", KernelName, buildOptions);
    uint32_t mMaxWorkGroupSize  = static_cast<uint32_t>(runTime->getMaxWorkGroupSize(kernel));
    std::vector<uint32_t> mGlobalWorkSize = {(uint32_t)(Width * Height), (uint32_t)(UP_DIV(Channel, 4)), (uint32_t)(Batch)};

    uint32_t index = 0;
    kernel.setArg(index++, mGlobalWorkSize[0]);
    kernel.setArg(index++, mGlobalWorkSize[1]);
    kernel.setArg(index++, mGlobalWorkSize[2]);
    kernel.setArg(index++, openCLBuffer(input));
    kernel.setArg(index++, openCLBuffer(output));
    kernel.setArg(index++, Width);
    kernel.setArg(index++, Height);
    kernel.setArg(index++, Channel);

    std::vector<uint32_t> mLocalWorkSize = localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, runTime, KernelName, kernel).first;

    globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1], mGlobalWorkSize[2]};
    localWorkSize  = {mLocalWorkSize[0], mLocalWorkSize[1], mLocalWorkSize[2]};
}

static void _setTensorStack(std::vector<Tensor *> &result, const std::vector<Tensor *> &inputs,
                            const std::vector<Tensor *> &outputs, const LoopParam *loop) {
    if (loop->inputIndexes() != nullptr) {
        for (int i = 0; i < loop->inputIndexes()->size(); ++i) {
            result[loop->inputIndexes()->data()[i]] = inputs[i];
        }
    }
    for (int i = 0; i < loop->outputIndexes()->size(); ++i) {
        result[loop->outputIndexes()->data()[i]] = outputs[i];
    }
}


 LoopGatherBufExecution::LoopGatherBufExecution(const LoopParam *loop, const MNN::Op *op, Backend *bn)
     : CommonExecution(bn, op) {
     mLoop = loop;
     mTensors.resize(mLoop->tensorNumber());
     auto cmd = loop->commands()->GetAs<RegionCommand>(0);
 }
 ErrorCode LoopGatherBufExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
     auto cmd                      = mLoop->commands()->GetAs<RegionCommand>(0);
     OpenCLBackend *mOpenCLBackend = (OpenCLBackend *)backend();
     auto runTime                  = mOpenCLBackend->getOpenCLRuntime();
     _setTensorStack(mTensors, inputs, outputs, mLoop);
     mUnits.clear();
     mOffsetTensors.clear();
     mTmpTensors.resize(2);
     int x = cmd->size()->data()[0];
     int y = cmd->size()->data()[1];
     int z = cmd->size()->data()[2];
     int n = mLoop->loopNumber();

     auto srcStride = cmd->view()->GetAs<View>(1)->stride()->data();
     auto dstStride = cmd->view()->GetAs<View>(0)->stride()->data();
     for (int i = 0; i < 3; ++i) {
        mStride_src[i] = srcStride[i];
        mStride_dst[i] = dstStride[i];
     }

     mStride_src[3] = cmd->view()->GetAs<View>(1)->offset();
     mStride_dst[3] = cmd->view()->GetAs<View>(0)->offset();
     ::memcpy(mStep, cmd->steps()->data(), cmd->steps()->size() * sizeof(int));
     ::memcpy(mIter, cmd->iterIndexes()->data(), cmd->iterIndexes()->size() * sizeof(int));

     // tile input
     {
        auto input = mTensors[cmd->indexes()->data()[1]];
        std::vector<int> Shape = tensorShapeFormat(input);
        const int Channel = Shape.at(3);
        const int Width = Shape.at(2);
        const int Height = Shape.at(1);
        const int Batch = Shape.at(0);
        mTmpTensors[1] = std::make_shared<Tensor>(Tensor::createDevice<float>(std::vector<int>{Batch, Channel, Height, Width}));
        mOpenCLBackend->onAcquireBuffer(mTmpTensors[1].get(), Backend::DYNAMIC);

        Unit unit;
        _TileOrPackTensor(mTensors[cmd->indexes()->data()[1]], mTmpTensors[1].get(), unit.kernel, unit.globalWorkSize, unit.localWorkSize, Width, Height,Channel, Batch, runTime, "tile_buf", mBuildOptions);
        mUnits.emplace_back(unit);
     }

     for(int i = 0; i < cmd->iterIndexes()->size(); ++i){
        if (mIter[i] >= 0) {
            auto input = mTensors[cmd->iterIndexes()->data()[i]];
            std::vector<int> Shape = tensorShapeFormat(input);
            const int Channel = Shape.at(3);
            const int Width = Shape.at(2);
            const int Height = Shape.at(1);
            const int Batch = Shape.at(0);
            mOffsetTensors.emplace_back(std::make_shared<Tensor>(Tensor::createDevice<float>(std::vector<int>{Batch, Channel, Height, Width})));
            mOpenCLBackend->onAcquireBuffer(mOffsetTensors.back().get(), Backend::DYNAMIC);

            Unit unit;
            _TileOrPackTensor(input, mOffsetTensors.back().get(), unit.kernel, unit.globalWorkSize, unit.localWorkSize, Width, Height, Channel, Batch, runTime, "tile_buf", mBuildOptions);
            mUnits.emplace_back(unit);
        }
     }
     
     // gather
     {
        mTmpTensors[0] = std::make_shared<Tensor>(Tensor::createDevice<float>(std::vector<int>{n, z, y, x}));
        mOpenCLBackend->onAcquireBuffer(mTmpTensors[0].get(), Backend::DYNAMIC);
        int offset_index = 0;

        Unit unit;
        std::string KernelName = "batch_gather_buf";
        unit.kernel = runTime->buildKernel("loop_buf", KernelName, mBuildOptions);
        uint32_t mMaxWorkGroupSize = static_cast<uint32_t>(runTime->getMaxWorkGroupSize(unit.kernel));
        std::vector<uint32_t> mGlobalWorkSize = {(uint32_t)(x * y), (uint32_t)(z), (uint32_t)(n)};

        uint32_t index = 0;
        unit.kernel.setArg(index++, mGlobalWorkSize[0]);
        unit.kernel.setArg(index++, mGlobalWorkSize[1]);
        unit.kernel.setArg(index++, mGlobalWorkSize[2]);
        unit.kernel.setArg(index++, openCLBuffer(mTmpTensors[0].get()));
        unit.kernel.setArg(index++, openCLBuffer(mTmpTensors[1].get()));
        for (int i = 0; i < cmd->iterIndexes()->size(); ++i) {
            if (mIter[i] >= 0) {
                unit.kernel.setArg(index++, openCLBuffer(mOffsetTensors[offset_index++].get()));
            } else {
                unit.kernel.setArg(index++, openCLBuffer(mTensors[cmd->indexes()->data()[1]]));
            }
        }
        unit.kernel.setArg(index++, x);
        unit.kernel.setArg(index++, sizeof(mStride_src), mStride_src);
        unit.kernel.setArg(index++, sizeof(mStride_dst), mStride_dst);
        unit.kernel.setArg(index++, sizeof(mStep), mStep);
        unit.kernel.setArg(index++, sizeof(mIter), mIter);

        std::vector<uint32_t> mLocalWorkSize = localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, runTime, KernelName, unit.kernel).first;

        unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1], mGlobalWorkSize[2]};
        unit.localWorkSize  = {mLocalWorkSize[0], mLocalWorkSize[1], mLocalWorkSize[2]};
        mUnits.emplace_back(unit);
     }

     //pack output
     {
        auto output = mTensors[cmd->indexes()->data()[0]];
        std::vector<int> Shape = tensorShapeFormat(output);
        const int Channel = Shape.at(3);
        const int Width = Shape.at(2);
        const int Height = Shape.at(1);
        const int Batch = Shape.at(0);
        Unit unit;
        _TileOrPackTensor(mTmpTensors[0].get(), mTensors[cmd->indexes()->data()[0]], unit.kernel, unit.globalWorkSize, unit.localWorkSize, Width, Height, Channel, Batch, runTime, "pack_buf", mBuildOptions);
        mUnits.emplace_back(unit);
     }

     for (int i = 0; i < mTmpTensors.size(); ++i) {
        mOpenCLBackend->onReleaseBuffer(mTmpTensors[i].get(), Backend::DYNAMIC);
     }
     for (int i = 0; i < mOffsetTensors.size(); ++i) {
        mOpenCLBackend->onReleaseBuffer(mOffsetTensors[i].get(), Backend::DYNAMIC);
     }

     return NO_ERROR;
 }


LoopBatchMatMulBufExecution::LoopBatchMatMulBufExecution(const LoopParam *loop, const MNN::Op *op, Backend *bn)
     : CommonExecution(bn, op) {
     mLoop = loop;
     mTensors.resize(mLoop->tensorNumber());
     auto cmd = loop->commands()->GetAs<RegionCommand>(0);
     mHasBias = cmd->indexes()->size() > 3;
     mTransposeA = cmd->op()->main_as_MatMul()->transposeA();
     mTransposeB = cmd->op()->main_as_MatMul()->transposeB();
}
ErrorCode LoopBatchMatMulBufExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
     auto cmd     = mLoop->commands()->GetAs<RegionCommand>(0);
     OpenCLBackend *mOpenCLBackend = (OpenCLBackend *)backend();
     auto runTime = mOpenCLBackend->getOpenCLRuntime();
     _setTensorStack(mTensors, inputs, outputs, mLoop);

     mOffset[0] = cmd->view()->GetAs<View>(0)->offset();
     mOffset[1] = cmd->view()->GetAs<View>(1)->offset();
     mOffset[2] = cmd->view()->GetAs<View>(2)->offset();
     mUnits.clear();
     mOffsetTensors.clear();
     mTmpTensors.resize(3);
     if (mHasBias) {
        mTmpTensors.resize(4);
        mOffset[3] = cmd->view()->GetAs<View>(3)->offset();
     }

     ::memcpy(mStep, cmd->steps()->data(), cmd->steps()->size() * sizeof(int));
     ::memcpy(mIter, cmd->iterIndexes()->data(), cmd->iterIndexes()->size() * sizeof(int));
     int e = cmd->size()->data()[0];
     int l = cmd->size()->data()[1];
     int h = cmd->size()->data()[2];
     int n = mLoop->loopNumber();

     // tile input     
     for (int i = 1; i < cmd->indexes()->size(); ++i) {
        auto input = mTensors[cmd->indexes()->data()[i]];
        std::vector<int> Shape = tensorShapeFormat(input);
        const int Channel = Shape.at(3);
        const int Width = Shape.at(2);
        const int Height = Shape.at(1);
        const int Batch = Shape.at(0);
        mTmpTensors[i] = std::make_shared<Tensor>(Tensor::createDevice<float>(std::vector<int>{Batch, Channel, Height, Width}));
        mOpenCLBackend->onAcquireBuffer(mTmpTensors[i].get(), Backend::DYNAMIC);       

        Unit unit;
        _TileOrPackTensor(input, mTmpTensors[i].get(), unit.kernel, unit.globalWorkSize, unit.localWorkSize, Width, Height, Channel, Batch, runTime, "tile_buf", mBuildOptions);
        mUnits.emplace_back(unit);
     }

     for(int i = 0; i < cmd->iterIndexes()->size(); ++i){
        if (mIter[i] >= 0) {
            auto input = mTensors[cmd->iterIndexes()->data()[i]];
            std::vector<int> Shape = tensorShapeFormat(input);
            const int Channel = Shape.at(3);
            const int Width = Shape.at(2);
            const int Height = Shape.at(1);
            const int Batch = Shape.at(0);
            mOffsetTensors.emplace_back(std::make_shared<Tensor>(Tensor::createDevice<float>(std::vector<int>{Batch, Channel, Height, Width})));
            mOpenCLBackend->onAcquireBuffer(mOffsetTensors.back().get(), Backend::DYNAMIC);

            Unit unit;
            _TileOrPackTensor(input, mOffsetTensors.back().get(), unit.kernel, unit.globalWorkSize, unit.localWorkSize, Width, Height, Channel, Batch, runTime, "tile_buf", mBuildOptions);
            mUnits.emplace_back(unit);
        }
     }

     // matmul
     {
        mTmpTensors[0] = std::make_shared<Tensor>(Tensor::createDevice<float>(std::vector<int>{1, n, e, h}));
        mOpenCLBackend->onAcquireBuffer(mTmpTensors[0].get(), Backend::DYNAMIC);
        int offset_index = 0;

        Unit unit;
        std::string KernelName = "batch_matmul_buf";
        if (mHasBias) {
            mBuildOptions.emplace("-DBIAS");
        }
        if (mTransposeA) {
            mBuildOptions.emplace("-DTRANSPOSE_A");
        }
        if (mTransposeB) {
            mBuildOptions.emplace("-DTRANSPOSE_B");
        }
        unit.kernel = runTime->buildKernel("loop_buf", KernelName, mBuildOptions);
        uint32_t mMaxWorkGroupSize = static_cast<uint32_t>(runTime->getMaxWorkGroupSize(unit.kernel));
        std::vector<uint32_t> mGlobalWorkSize = {(uint32_t)(h), (uint32_t)(e),(uint32_t)(n)};

        uint32_t index = 0;
        unit.kernel.setArg(index++, mGlobalWorkSize[0]);
        unit.kernel.setArg(index++, mGlobalWorkSize[1]);
        unit.kernel.setArg(index++, mGlobalWorkSize[2]);
        unit.kernel.setArg(index++, openCLBuffer(mTmpTensors[0].get()));
        unit.kernel.setArg(index++, openCLBuffer(mTmpTensors[1].get()));
        unit.kernel.setArg(index++, openCLBuffer(mTmpTensors[2].get()));
        if (mHasBias) {
            unit.kernel.setArg(index++, openCLBuffer(mTmpTensors[3].get()));
        }
        for (int i = 0; i < cmd->iterIndexes()->size(); ++i) {
            if (mIter[i] >= 0) {
                unit.kernel.setArg(index++, openCLBuffer(mOffsetTensors[offset_index++].get()));
            } else {
                unit.kernel.setArg(index++, openCLBuffer(mTensors[cmd->indexes()->data()[1]]));
            }
        }
        unit.kernel.setArg(index++, e);
        unit.kernel.setArg(index++, l);
        unit.kernel.setArg(index++, h);
        unit.kernel.setArg(index++, sizeof(mOffset), mOffset);
        unit.kernel.setArg(index++, sizeof(mIter), mIter);       
        unit.kernel.setArg(index++, sizeof(mStep), mStep);        

        std::vector<uint32_t> mLocalWorkSize = localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, runTime, KernelName, unit.kernel).first;

        unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1], mGlobalWorkSize[2]};
        unit.localWorkSize  = {mLocalWorkSize[0], mLocalWorkSize[1], mLocalWorkSize[2]};
        mUnits.emplace_back(unit);
     }

     //pack output
     {
        auto output = mTensors[cmd->indexes()->data()[0]];
        std::vector<int> Shape = tensorShapeFormat(output);
        const int Channel = Shape.at(3);
        const int Width = Shape.at(2);
        const int Height = Shape.at(1);
        const int Batch = Shape.at(0);
        Unit unit;
        _TileOrPackTensor(mTmpTensors[0].get(), output, unit.kernel, unit.globalWorkSize, unit.localWorkSize, Width, Height, Channel, Batch, runTime, "pack_buf", mBuildOptions);
        mUnits.emplace_back(unit);
     }

    for (int i = 0; i < cmd->indexes()->size(); ++i) {
         mOpenCLBackend->onReleaseBuffer(mTmpTensors[i].get(), Backend::DYNAMIC);
    }
    for (int i = 0; i < mOffsetTensors.size(); ++i) {
         mOpenCLBackend->onReleaseBuffer(mOffsetTensors[i].get(), Backend::DYNAMIC);
    }

    return NO_ERROR;
}

class LoopBufCreator : public OpenCLBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        auto loop = op->main_as_LoopParam();
        if (nullptr == loop || loop->commands() == nullptr) {
            return nullptr;
        }
        if (nullptr != loop->initCommand()) {
            return nullptr;
        }
        // Make Tensor Stack
        if (1 == loop->commands()->size()) {
            auto cmd   = loop->commands()->GetAs<RegionCommand>(0);
            auto subop = cmd->op();
            if (OpType_UnaryOp == subop->type() && nullptr == subop->main() && cmd->fuse() < 0) {
                return new LoopGatherBufExecution(loop, op, backend);
            }
            if (OpType_MatMul == subop->type() && loop->parallel()) {
                return new LoopBatchMatMulBufExecution(loop, op, backend);
            }
        }
        return nullptr;
    }
};

OpenCLCreatorRegister<LoopBufCreator> __LoopBuf_op(OpType_While, BUFFER);

} // namespace OpenCL
} // namespace MNN
#endif /* MNN_OPENCL_BUFFER_CLOSED */
