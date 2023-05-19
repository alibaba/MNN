//
//  LoopExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//


#include "backend/opencl/execution/image/LoopExecution.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {
namespace OpenCL {

static void _TileTensor(Tensor *input, cl::Buffer *output, cl::Kernel& kernel, cl::NDRange &globalWorkSize,
                        cl::NDRange &localWorkSize, const int Width, const int Height, const int Channel,
                        const int Batch, OpenCLRuntime *runTime, const std::set<std::string> &buildOptions) {
    kernel = runTime->buildKernel("loop", "tile", buildOptions);
    uint32_t mMaxWorkGroupSize  = static_cast<uint32_t>(runTime->getMaxWorkGroupSize(kernel));
    std::vector<uint32_t> mGlobalWorkSize = {(uint32_t)(Width * Height), (uint32_t)(UP_DIV(Channel, 4)), (uint32_t)(Batch)};

    uint32_t index = 0;
    kernel.setArg(index++, mGlobalWorkSize[0]);
    kernel.setArg(index++, mGlobalWorkSize[1]);
    kernel.setArg(index++, mGlobalWorkSize[2]);
    kernel.setArg(index++, openCLImage(input));
    kernel.setArg(index++, *output);
    kernel.setArg(index++, Width);
    kernel.setArg(index++, Height);
    kernel.setArg(index++, Channel);

    std::vector<uint32_t> mLocalWorkSize = localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, runTime, "tile", kernel).first;

    globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1], mGlobalWorkSize[2]};
    localWorkSize  = {mLocalWorkSize[0], mLocalWorkSize[1], mLocalWorkSize[2]};
}

static void _PackTensor(cl::Buffer *input, Tensor *output, cl::Kernel& kernel, cl::NDRange &globalWorkSize,
                        cl::NDRange &localWorkSize, const int Width, const int Height, const int Channel,
                        const int Batch, OpenCLRuntime *runTime, const std::set<std::string> &buildOptions) {
    kernel = runTime->buildKernel("loop", "pack", buildOptions);
    uint32_t mMaxWorkGroupSize  = static_cast<uint32_t>(runTime->getMaxWorkGroupSize(kernel));
    std::vector<uint32_t> mGlobalWorkSize = {(uint32_t)(Width * Height), (uint32_t)(UP_DIV(Channel, 4)), (uint32_t)(Batch)};

    uint32_t index = 0;
    kernel.setArg(index++, mGlobalWorkSize[0]);
    kernel.setArg(index++, mGlobalWorkSize[1]);
    kernel.setArg(index++, mGlobalWorkSize[2]);
    kernel.setArg(index++, *input);
    kernel.setArg(index++, openCLImage(output));
    kernel.setArg(index++, Width);
    kernel.setArg(index++, Height);
    kernel.setArg(index++, Channel);

    std::vector<uint32_t> mLocalWorkSize = localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, runTime, "pack", kernel).first;

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


 LoopGatherExecution::LoopGatherExecution(const LoopParam *loop, const MNN::Op *op, Backend *bn)
     : CommonExecution(bn, op) {
     mLoop = loop;
     mTensors.resize(mLoop->tensorNumber());
     auto cmd = loop->commands()->GetAs<RegionCommand>(0);
     mOpType = op->type();
 }
 ErrorCode LoopGatherExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
     auto cmd                      = mLoop->commands()->GetAs<RegionCommand>(0);
     OpenCLBackend *mOpenCLBackend = (OpenCLBackend *)backend();
     auto runTime                  = mOpenCLBackend->getOpenCLRuntime();
     auto bufferPool               = mOpenCLBackend->getBufferPool();
     auto bufferUnitSize           = runTime->isSupportedFP16() ? sizeof(half_float::half) : sizeof(float);
     _setTensorStack(mTensors, inputs, outputs, mLoop);
     mUnits.clear();
     mOffsetBuffers.clear();
     mTmpBuffers.resize(2);
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
        mTmpBuffers[1] = bufferPool->alloc(input->elementSize() * bufferUnitSize); 

        Unit unit;
        _TileTensor(mTensors[cmd->indexes()->data()[1]], mTmpBuffers[1], unit.kernel, unit.globalWorkSize, unit.localWorkSize, Width, Height,Channel, Batch, runTime, mBuildOptions);
        mUnits.emplace_back(unit);
     }

     for(int i = 0; i < cmd->iterIndexes()->size(); ++i){
        if (mIter[i] >= 0) {
            auto input = mTensors[cmd->iterIndexes()->data()[i]];
            std::vector<int> Shape = tensorShapeFormat(input);
            const int Channel = Shape.at(3);
            const int Width = Shape.at(2);
            const int Height = Shape.at(1);
            const int Batch        = Shape.at(0);
            mOffsetBuffers.emplace_back(bufferPool->alloc(input->elementSize() * bufferUnitSize)); 

            Unit unit;
            _TileTensor(input, mOffsetBuffers.back(), unit.kernel, unit.globalWorkSize, unit.localWorkSize, Width, Height, Channel, Batch, runTime, mBuildOptions);
            mUnits.emplace_back(unit);
        }
     }
     
     // gather
     {
        mTmpBuffers[0] = bufferPool->alloc(n * z * y * x * bufferUnitSize); 
        int offset_index = 0;
        Unit unit;
        std::string KernelName = "batch_gather";
        unit.kernel = runTime->buildKernel("loop", KernelName, mBuildOptions);
        uint32_t mMaxWorkGroupSize = static_cast<uint32_t>(runTime->getMaxWorkGroupSize(unit.kernel));
        std::vector<uint32_t> mGlobalWorkSize = {(uint32_t)(x * y), (uint32_t)(z), (uint32_t)(n)};

        uint32_t index = 0;
        unit.kernel.setArg(index++, mGlobalWorkSize[0]);
        unit.kernel.setArg(index++, mGlobalWorkSize[1]);
        unit.kernel.setArg(index++, mGlobalWorkSize[2]);
        unit.kernel.setArg(index++, *mTmpBuffers[0]);
        unit.kernel.setArg(index++, *mTmpBuffers[1]);
        for (int i = 0; i < cmd->iterIndexes()->size(); ++i) {
            if (mIter[i] >= 0) {
                unit.kernel.setArg(index++, *mOffsetBuffers[offset_index++]);
            } else {
                unit.kernel.setArg(index++, *mTmpBuffers[0]);
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
        _PackTensor(mTmpBuffers[0], mTensors[cmd->indexes()->data()[0]], unit.kernel, unit.globalWorkSize, unit.localWorkSize, Width, Height, Channel, Batch, runTime, mBuildOptions);
        mUnits.emplace_back(unit);
     }

     for (int i = 0; i < mTmpBuffers.size(); ++i) {
        bufferPool->recycle(mTmpBuffers[i]);
     }
     for (int i = 0; i < mOffsetBuffers.size(); ++i) {
        bufferPool->recycle(mOffsetBuffers[i]);
     }

     return NO_ERROR;
 }


LoopBatchMatMulExecution::LoopBatchMatMulExecution(const LoopParam *loop, const MNN::Op *op, Backend *bn)
     : CommonExecution(bn, op) {
     mLoop = loop;
     mTensors.resize(mLoop->tensorNumber());
     auto cmd = loop->commands()->GetAs<RegionCommand>(0);
     mHasBias = cmd->indexes()->size() > 3;
     mTransposeA = cmd->op()->main_as_MatMul()->transposeA();
     mTransposeB = cmd->op()->main_as_MatMul()->transposeB();
}
ErrorCode LoopBatchMatMulExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
     auto cmd     = mLoop->commands()->GetAs<RegionCommand>(0);
     OpenCLBackend *mOpenCLBackend = (OpenCLBackend *)backend();
     auto runTime = mOpenCLBackend->getOpenCLRuntime();
     auto bufferPool = mOpenCLBackend->getBufferPool();
     auto bufferUnitSize = runTime->isSupportedFP16() ? sizeof(half_float::half) : sizeof(float);
     _setTensorStack(mTensors, inputs, outputs, mLoop);

     mOffset[0] = cmd->view()->GetAs<View>(0)->offset();
     mOffset[1] = cmd->view()->GetAs<View>(1)->offset();
     mOffset[2] = cmd->view()->GetAs<View>(2)->offset();
     mUnits.clear();
     mOffsetBuffers.clear();
     mTmpBuffers.resize(3);
     if (mHasBias) {
        mTmpBuffers.resize(4);
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
        mTmpBuffers[i] = bufferPool->alloc(input->elementSize() * bufferUnitSize); 

        Unit unit;
        _TileTensor(input, mTmpBuffers[i], unit.kernel, unit.globalWorkSize, unit.localWorkSize, Width, Height, Channel, Batch, runTime, mBuildOptions);
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
            mOffsetBuffers.emplace_back(bufferPool->alloc(input->elementSize() * bufferUnitSize)); 

            Unit unit;
            _TileTensor(input, mOffsetBuffers.back(), unit.kernel, unit.globalWorkSize, unit.localWorkSize, Width, Height, Channel, Batch, runTime, mBuildOptions);
            mUnits.emplace_back(unit);
        }
     }

     // matmul
     {
        mTmpBuffers[0] = bufferPool->alloc(n * e * h * bufferUnitSize); 
        int offset_index = 0;

        Unit unit;
        std::string KernelName = "batch_matmul";
        if (mHasBias) {
            mBuildOptions.emplace("-DBIAS");
        }
        if (mTransposeA) {
            mBuildOptions.emplace("-DTRANSPOSE_A");
        }
        if (mTransposeB) {
            mBuildOptions.emplace("-DTRANSPOSE_B");
        }
        unit.kernel = runTime->buildKernel("loop", KernelName, mBuildOptions);
        uint32_t mMaxWorkGroupSize = static_cast<uint32_t>(runTime->getMaxWorkGroupSize(unit.kernel));
        std::vector<uint32_t> mGlobalWorkSize = {(uint32_t)(h), (uint32_t)(e),(uint32_t)(n)};

        uint32_t index = 0;
        unit.kernel.setArg(index++, mGlobalWorkSize[0]);
        unit.kernel.setArg(index++, mGlobalWorkSize[1]);
        unit.kernel.setArg(index++, mGlobalWorkSize[2]);
        unit.kernel.setArg(index++, *mTmpBuffers[0]);
        unit.kernel.setArg(index++, *mTmpBuffers[1]);
        unit.kernel.setArg(index++, *mTmpBuffers[2]);
        if (mHasBias) {
            unit.kernel.setArg(index++, *mTmpBuffers[3]);
        }
        for (int i = 0; i < cmd->iterIndexes()->size(); ++i) {
            if (mIter[i] >= 0) {
                unit.kernel.setArg(index++, *mOffsetBuffers[offset_index++]);
            } else {
                unit.kernel.setArg(index++, *mTmpBuffers[0]);
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
        _PackTensor(mTmpBuffers[0], output, unit.kernel, unit.globalWorkSize, unit.localWorkSize, Width, Height, Channel, Batch, runTime, mBuildOptions);
        mUnits.emplace_back(unit);
     }

    for (int i = 0; i < mTmpBuffers.size(); ++i) {
         bufferPool->recycle(mTmpBuffers[i]);
    }
    for (int i = 0; i < mOffsetBuffers.size(); ++i) {
         bufferPool->recycle(mOffsetBuffers[i]);
    }

    return NO_ERROR;
}

class LoopCreator : public OpenCLBackend::Creator {
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
                return new LoopGatherExecution(loop, op, backend);
            }
            if (OpType_MatMul == subop->type() && loop->parallel()) {
                return new LoopBatchMatMulExecution(loop, op, backend);
            }
        }
        return nullptr;
    }
};

OpenCLCreatorRegister<LoopCreator> __Loop_op(OpType_While, IMAGE);

} // namespace OpenCL
} // namespace MNN
