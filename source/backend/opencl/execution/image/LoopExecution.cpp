//
//  LoopExecution.cpp
//  MNN
//
//  Created by MNN on 2023/05/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/execution/image/LoopExecution.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {
namespace OpenCL {

static void _TileTensor(Tensor *input, cl::Buffer *output, cl::Kernel& kernel, cl::NDRange &globalWorkSize,
                        cl::NDRange &localWorkSize, const int Width, const int Height, const int Channel,
                        const int Batch, OpenCLRuntime *runTime, std::set<std::string> buildOptions) {
    
    if (TensorUtils::getDescribe(input)->dimensionFormat == MNN::MNN_DATA_FORMAT_NHWC){
        buildOptions.emplace("-DMNN_NHWC");
    }
    kernel = runTime->buildKernel("loop", "tile", buildOptions);
    uint32_t mMaxWorkGroupSize  = static_cast<uint32_t>(runTime->getMaxWorkGroupSize(kernel));
    std::vector<uint32_t> mGlobalWorkSize = {(uint32_t)(Width * Height), (uint32_t)(UP_DIV(Channel, 4)), (uint32_t)(Batch)};

    uint32_t index = 0;
    cl_int ret = CL_SUCCESS;
    ret |= kernel.setArg(index++, mGlobalWorkSize[0]);
    ret |= kernel.setArg(index++, mGlobalWorkSize[1]);
    ret |= kernel.setArg(index++, mGlobalWorkSize[2]);
    ret |= kernel.setArg(index++, openCLImage(input));
    ret |= kernel.setArg(index++, *output);
    ret |= kernel.setArg(index++, Width);
    ret |= kernel.setArg(index++, Height);
    ret |= kernel.setArg(index++, Channel);
    MNN_CHECK_CL_SUCCESS(ret, "setArg Loop _PackTensor");

    std::vector<uint32_t> mLocalWorkSize = localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, runTime, "tile", kernel).first;

    globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1], mGlobalWorkSize[2]};
    localWorkSize  = {mLocalWorkSize[0], mLocalWorkSize[1], mLocalWorkSize[2]};
    
    recordKernel3d(kernel, mGlobalWorkSize, mLocalWorkSize, runTime);
}

static void _PackTensor(cl::Buffer *input, Tensor *output, cl::Kernel& kernel, cl::NDRange &globalWorkSize,
                        cl::NDRange &localWorkSize, const int Width, const int Height, const int Channel,
                        const int Batch, OpenCLRuntime *runTime, std::set<std::string> buildOptions) {
    if (TensorUtils::getDescribe(output)->dimensionFormat == MNN::MNN_DATA_FORMAT_NHWC){
        buildOptions.emplace("-DMNN_NHWC");
    }
    kernel = runTime->buildKernel("loop", "pack", buildOptions);
    uint32_t mMaxWorkGroupSize  = static_cast<uint32_t>(runTime->getMaxWorkGroupSize(kernel));
    std::vector<uint32_t> mGlobalWorkSize = {(uint32_t)(Width * Height), (uint32_t)(UP_DIV(Channel, 4)), (uint32_t)(Batch)};

    uint32_t index = 0;
    cl_int ret = CL_SUCCESS;
    ret |= kernel.setArg(index++, mGlobalWorkSize[0]);
    ret |= kernel.setArg(index++, mGlobalWorkSize[1]);
    ret |= kernel.setArg(index++, mGlobalWorkSize[2]);
    ret |= kernel.setArg(index++, *input);
    ret |= kernel.setArg(index++, openCLImage(output));
    ret |= kernel.setArg(index++, Width);
    ret |= kernel.setArg(index++, Height);
    ret |= kernel.setArg(index++, Channel);
    MNN_CHECK_CL_SUCCESS(ret, "setArg Loop _PackTensor");

    std::vector<uint32_t> mLocalWorkSize = localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, runTime, "pack", kernel).first;

    globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1], mGlobalWorkSize[2]};
    localWorkSize  = {mLocalWorkSize[0], mLocalWorkSize[1], mLocalWorkSize[2]};
    recordKernel3d(kernel, mGlobalWorkSize, mLocalWorkSize, runTime);
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
 }
 ErrorCode LoopGatherExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
     auto cmd                      = mLoop->commands()->GetAs<RegionCommand>(0);
     OpenCLBackend *mOpenCLBackend = (OpenCLBackend *)backend();
     auto runTime                  = mOpenCLBackend->getOpenCLRuntime();
     startRecord(runTime, mRecording);
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
        cl_int ret = CL_SUCCESS;
        ret |= unit.kernel.setArg(index++, mGlobalWorkSize[0]);
        ret |= unit.kernel.setArg(index++, mGlobalWorkSize[1]);
        ret |= unit.kernel.setArg(index++, mGlobalWorkSize[2]);
        ret |= unit.kernel.setArg(index++, *mTmpBuffers[0]);
        ret |= unit.kernel.setArg(index++, *mTmpBuffers[1]);
        for (int i = 0; i < cmd->iterIndexes()->size(); ++i) {
            if (mIter[i] >= 0) {
                ret |= unit.kernel.setArg(index++, *mOffsetBuffers[offset_index++]);
            } else {
                ret |= unit.kernel.setArg(index++, *mTmpBuffers[0]);
            }
        }
        ret |= unit.kernel.setArg(index++, x);
        ret |= unit.kernel.setArg(index++, sizeof(mStride_src), mStride_src);
        ret |= unit.kernel.setArg(index++, sizeof(mStride_dst), mStride_dst);
        ret |= unit.kernel.setArg(index++, sizeof(mStep), mStep);
        ret |= unit.kernel.setArg(index++, sizeof(mIter), mIter);
        MNN_CHECK_CL_SUCCESS(ret, "setArg LoopGatherExecution");

        std::vector<uint32_t> mLocalWorkSize = localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, runTime, KernelName, unit.kernel).first;

        unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1], mGlobalWorkSize[2]};
        unit.localWorkSize  = {mLocalWorkSize[0], mLocalWorkSize[1], mLocalWorkSize[2]};
        recordKernel3d(unit.kernel, mGlobalWorkSize, mLocalWorkSize, runTime);
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
     endRecord(runTime, mRecording);

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
     startRecord(runTime, mRecording);
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
        mTmpBuffers[i] = bufferPool->alloc(Batch * Channel * ROUND_UP(Height, 4) * ROUND_UP(Width, 4) * bufferUnitSize);

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
        mBuildOptions.emplace("-DH_LEAVES=" + std::to_string(h % 4));
        unit.kernel = runTime->buildKernel("loop", KernelName, mBuildOptions);
        uint32_t mMaxWorkGroupSize = static_cast<uint32_t>(runTime->getMaxWorkGroupSize(unit.kernel));
        std::vector<uint32_t> mGlobalWorkSize = {(uint32_t)(UP_DIV(h, 4)), (uint32_t)(UP_DIV(e, 4)),(uint32_t)(n)};

        uint32_t index = 0;
        cl_int ret = CL_SUCCESS;
        ret |= unit.kernel.setArg(index++, mGlobalWorkSize[0]);
        ret |= unit.kernel.setArg(index++, mGlobalWorkSize[1]);
        ret |= unit.kernel.setArg(index++, mGlobalWorkSize[2]);
        ret |= unit.kernel.setArg(index++, *mTmpBuffers[0]);
        ret |= unit.kernel.setArg(index++, *mTmpBuffers[1]);
        ret |= unit.kernel.setArg(index++, *mTmpBuffers[2]);
        if (mHasBias) {
            ret |= unit.kernel.setArg(index++, *mTmpBuffers[3]);
        }
        for (int i = 0; i < cmd->iterIndexes()->size(); ++i) {
            if (mIter[i] >= 0) {
                ret |= unit.kernel.setArg(index++, *mOffsetBuffers[offset_index++]);
            } else {
                ret |= unit.kernel.setArg(index++, *mTmpBuffers[0]);
            }
        }
        ret |= unit.kernel.setArg(index++, e);
        ret |= unit.kernel.setArg(index++, l);
        ret |= unit.kernel.setArg(index++, h);
        ret |= unit.kernel.setArg(index++, sizeof(mOffset), mOffset);
        ret |= unit.kernel.setArg(index++, sizeof(mIter), mIter);
        ret |= unit.kernel.setArg(index++, sizeof(mStep), mStep);
        MNN_CHECK_CL_SUCCESS(ret, "setArg LoopBatchMatMulExecution");

        std::vector<uint32_t> mLocalWorkSize = localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, runTime, KernelName, unit.kernel).first;

        unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1], mGlobalWorkSize[2]};
        unit.localWorkSize  = {mLocalWorkSize[0], mLocalWorkSize[1], mLocalWorkSize[2]};
        mUnits.emplace_back(unit);
        recordKernel3d(unit.kernel, mGlobalWorkSize, mLocalWorkSize, runTime);
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
    endRecord(runTime, mRecording);

    return NO_ERROR;
}

LoopBinaryExecution::LoopBinaryExecution(const LoopParam *loop, const std::string &compute, const MNN::Op *op, Backend *bn)
    : CommonExecution(bn, op) {
    mLoop = loop;
    mTensors.resize(mLoop->tensorNumber());
    auto cmd = loop->commands()->GetAs<RegionCommand>(0);
    mBuildOptions.emplace("-DLOOP_BINARY_OPERATOR=" + compute);
}
ErrorCode LoopBinaryExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto cmd                      = mLoop->commands()->GetAs<RegionCommand>(0);
    OpenCLBackend *mOpenCLBackend = (OpenCLBackend *)backend();
    auto runTime                  = mOpenCLBackend->getOpenCLRuntime();
    startRecord(runTime, mRecording);
    _setTensorStack(mTensors, inputs, outputs, mLoop);
    mUnits.clear();
    Unit unit;
    auto input0 = mTensors[cmd->indexes()->data()[1]];
    std::vector<int> Input0Shape = tensorShapeFormat(input0);
    int Input0Size[4] = {Input0Shape.at(2), Input0Shape.at(1),Input0Shape.at(3),Input0Shape.at(0)};
         
    auto input1 = mTensors[cmd->indexes()->data()[2]];
    std::vector<int> Input1Shape = tensorShapeFormat(input1);
    int Input1Size[4] = {Input1Shape.at(2), Input1Shape.at(1),Input1Shape.at(3),Input1Shape.at(0)};
         
    auto output = mTensors[cmd->indexes()->data()[0]];
    std::vector<int> Shape = tensorShapeFormat(output);
    const int Channel = Shape.at(3);
    const int Width = Shape.at(2);
    const int Height = Shape.at(1);
    const int Batch = Shape.at(0);
    const int ChannelBlock = UP_DIV(Channel, 4);
    auto BuildOptions = mBuildOptions;
    if(Input0Size[2] != Input1Size[2]){
        BuildOptions.emplace("-DBROADCAST_CHANNEL");
    }
    std::string KernelName = "broadcast_binary";
    unit.kernel = runTime->buildKernel("loop", KernelName, BuildOptions);
    uint32_t mMaxWorkGroupSize = static_cast<uint32_t>(runTime->getMaxWorkGroupSize(unit.kernel));
        
       
    std::vector<uint32_t> mGlobalWorkSize = {(uint32_t)(Width), (uint32_t)(Height), (uint32_t)(Batch * ChannelBlock)};

    uint32_t index = 0;
    cl_int ret = CL_SUCCESS;
    ret |= unit.kernel.setArg(index++, mGlobalWorkSize[0]);
    ret |= unit.kernel.setArg(index++, mGlobalWorkSize[1]);
    ret |= unit.kernel.setArg(index++, mGlobalWorkSize[2]);
    ret |= unit.kernel.setArg(index++, openCLImage(output));
    ret |= unit.kernel.setArg(index++, openCLImage(input0));
    ret |= unit.kernel.setArg(index++, openCLImage(input1));
    ret |= unit.kernel.setArg(index++, sizeof(Input0Size), Input0Size);
    ret |= unit.kernel.setArg(index++, sizeof(Input1Size), Input1Size);
    ret |= unit.kernel.setArg(index++, Width);
    ret |= unit.kernel.setArg(index++, Height);
    ret |= unit.kernel.setArg(index++, ChannelBlock);
    MNN_CHECK_CL_SUCCESS(ret, "setArg LoopBinaryExecution");

    std::vector<uint32_t> mLocalWorkSize = localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, runTime, KernelName, unit.kernel).first;

    unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1], mGlobalWorkSize[2]};
    unit.localWorkSize  = {mLocalWorkSize[0], mLocalWorkSize[1], mLocalWorkSize[2]};
    recordKernel3d(unit.kernel, mGlobalWorkSize, mLocalWorkSize, runTime);
    mUnits.emplace_back(unit);

    endRecord(runTime, mRecording);

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
            if (OpType_BinaryOp == subop->type() && loop->parallel()) {
                switch (subop->main_as_BinaryOp()->opType()) {
                    case BinaryOpOperation_MUL:
                        return new LoopBinaryExecution(loop, "in0*in1", op, backend);
                    case BinaryOpOperation_ADD:
                        return new LoopBinaryExecution(loop, "in0+in1", op, backend);
                    case BinaryOpOperation_SUB:
                        return new LoopBinaryExecution(loop, "in0-in1", op, backend);
                    case BinaryOpOperation_REALDIV:
                        return new LoopBinaryExecution(loop, "sign(in1)*in0/(fabs(in1)>(FLOAT4)((FLOAT)0.0000001)?fabs(in1):(FLOAT4)((FLOAT)0.0000001))", op, backend);
                    case BinaryOpOperation_MINIMUM:
                        return new LoopBinaryExecution(loop, "in0>in1?in1:in0", op, backend);
                    case BinaryOpOperation_MAXIMUM:
                        return new LoopBinaryExecution(loop, "in0>in1?in0:in1", op, backend);
                    case BinaryOpOperation_GREATER:
                        return new LoopBinaryExecution(loop, "convert_float4(-isgreater(in0,in1))", op, backend);
                    case BinaryOpOperation_LESS:
                        return new LoopBinaryExecution(loop, "convert_float4(-isless(in0,in1))", op, backend);
                    case BinaryOpOperation_LESS_EQUAL:
                        return new LoopBinaryExecution(loop, "convert_float4(-islessequal(in0,in1))", op, backend);
                    case BinaryOpOperation_GREATER_EQUAL:
                        return new LoopBinaryExecution(loop, "convert_float4(-isgreaterequal(in0,in1))", op, backend);
                    case BinaryOpOperation_EQUAL:
                        return new LoopBinaryExecution(loop, "convert_float4(-isequal(in0,in1))", op, backend);
                    case BinaryOpOperation_FLOORDIV:
                        return new LoopBinaryExecution(loop, "floor(sign(in1)*in0/(fabs(in1)>(FLOAT4)((FLOAT)0.0000001)?fabs(in1):(FLOAT4)((FLOAT)0.0000001)))", op, backend);
                    case BinaryOpOperation_FLOORMOD:
                        return new LoopBinaryExecution(loop, "in0-floor(sign(in1)*in0/(fabs(in1)>(FLOAT4)((FLOAT)0.0000001)?fabs(in1):(FLOAT4)((FLOAT)0.0000001)))*in1", op, backend);
                    case BinaryOpOperation_POW:
                        return new LoopBinaryExecution(loop, "pow(in0,in1)", op, backend);
                    case BinaryOpOperation_SquaredDifference:
                        return new LoopBinaryExecution(loop, "(in0-in1)*(in0-in1)", op, backend);
                    case BinaryOpOperation_ATAN2:
                        return new LoopBinaryExecution(loop, "atan(sign(in1)*in0/(fabs(in1)>(FLOAT4)((FLOAT)0.0000001)?fabs(in1):(FLOAT4)((FLOAT)0.0000001)))", op, backend);
                    case BinaryOpOperation_NOTEQUAL:
                        return new LoopBinaryExecution(loop, "convert_float4(-isnotequal(in0,in1))", op, backend);
                    case BinaryOpOperation_MOD:
                        return new LoopBinaryExecution(loop, "in0-floor(sign(in1)*in0/(fabs(in1)>(FLOAT4)((FLOAT)0.0000001)?fabs(in1):(FLOAT4)((FLOAT)0.0000001)))*in1", op, backend);
                    default:
                        break;
                }
                return nullptr;
            }
        }
        return nullptr;
    }
};

OpenCLCreatorRegister<LoopCreator> __Loop_op(OpType_While, IMAGE);

} // namespace OpenCL
} // namespace MNN
