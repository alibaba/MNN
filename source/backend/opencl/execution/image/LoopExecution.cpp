//
//  LoopExecution.cpp
//  MNN
//
//  Created by MNN on 2023/05/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/execution/image/LoopExecution.hpp"
#include "core/TensorUtils.hpp"

namespace MNN {
namespace OpenCL {

static void _TileTensor(Tensor *input, cl::Buffer *output, std::shared_ptr<KernelWrap>& kernelW, cl::NDRange &globalWorkSize,
                        cl::NDRange &localWorkSize, const int Width, const int Height, const int Channel,
                        const int Batch, OpenCLBackend *bn, std::set<std::string> buildOptions) {
    
    if (TensorUtils::getDescribe(input)->dimensionFormat == MNN::MNN_DATA_FORMAT_NHWC){
        buildOptions.emplace("-DMNN_NHWC");
    }
    kernelW = bn->getOpenCLRuntime()->buildKernel("loop", "tile", buildOptions, bn->getPrecision(), input, input);
    uint32_t mMaxWorkGroupSize  = static_cast<uint32_t>(bn->getOpenCLRuntime()->getMaxWorkGroupSize(kernelW));
    std::vector<uint32_t> mGlobalWorkSize = {(uint32_t)(Width * Height), (uint32_t)(UP_DIV(Channel, 4)), (uint32_t)(Batch)};
    auto kernel = kernelW->get();
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

    std::vector<uint32_t> mLocalWorkSize = localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, bn->getOpenCLRuntime(), "tile", kernelW, bn->getCLTuneLevel()).first;

    globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1], mGlobalWorkSize[2]};
    localWorkSize  = {mLocalWorkSize[0], mLocalWorkSize[1], mLocalWorkSize[2]};
    
    bn->recordKernel3d(kernelW, mGlobalWorkSize, mLocalWorkSize);
}

static void _PackTensor(cl::Buffer *input, Tensor *output, std::shared_ptr<KernelWrap>& kernelW, cl::NDRange &globalWorkSize,
                        cl::NDRange &localWorkSize, const int Width, const int Height, const int Channel,
                        const int Batch, OpenCLBackend *bn, std::set<std::string> buildOptions) {
    if (TensorUtils::getDescribe(output)->dimensionFormat == MNN::MNN_DATA_FORMAT_NHWC){
        buildOptions.emplace("-DMNN_NHWC");
    }
    kernelW = bn->getOpenCLRuntime()->buildKernel("loop", "pack", buildOptions, bn->getPrecision(), output, output);
    uint32_t mMaxWorkGroupSize  = static_cast<uint32_t>(bn->getOpenCLRuntime()->getMaxWorkGroupSize(kernelW));
    std::vector<uint32_t> mGlobalWorkSize = {(uint32_t)(Width * Height), (uint32_t)(UP_DIV(Channel, 4)), (uint32_t)(Batch)};
    auto kernel = kernelW->get();
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

    std::vector<uint32_t> mLocalWorkSize = localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, bn->getOpenCLRuntime(), "pack", kernelW, bn->getCLTuneLevel()).first;

    globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1], mGlobalWorkSize[2]};
    localWorkSize  = {mLocalWorkSize[0], mLocalWorkSize[1], mLocalWorkSize[2]};
    bn->recordKernel3d(kernelW, mGlobalWorkSize, mLocalWorkSize);
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
 }
ErrorCode LoopGatherExecution::InitCommandOnEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs){
    auto cmd                      = mLoop->initCommand()->GetAs<RegionCommand>(0);
    OpenCLBackend *mOpenCLBackend = (OpenCLBackend *)backend();
    auto runTime                  = mOpenCLBackend->getOpenCLRuntime();
    auto bufferPool               = mOpenCLBackend->getBufferPool();
    auto bufferUnitSize           = mOpenCLBackend->getPrecision() != BackendConfig::Precision_High ? sizeof(half_float::half) : sizeof(float);
    
    if (cmd->op() == nullptr){
        Unit unit;
        auto output = mTensors[cmd->indexes()->data()[0]];
        auto outputShape    = tensorShapeFormat(output);
        int region[] = {outputShape[0], UP_DIV(outputShape[3], 4), outputShape[1], outputShape[2]};//nhwc
        unit.kernel         = runTime->buildKernel("raster", "image_set_zero", {}, mOpenCLBackend->getPrecision(), output, output);
        unit.localWorkSize  = {8, 8};
        unit.globalWorkSize = {(uint32_t)UP_DIV((region[1] * region[3]), 16)*16,
                               (uint32_t)UP_DIV((region[0] * region[2]), 16)*16};

        int global_dim0 = region[1] * region[3];
        int global_dim1 = region[0] * region[2];

        uint32_t idx   = 0;
        cl_int ret = CL_SUCCESS;
        ret |= unit.kernel->get().setArg(idx++, global_dim0);
        ret |= unit.kernel->get().setArg(idx++, global_dim1);
        ret |= unit.kernel->get().setArg(idx++, openCLImage(output));
        if(ret != CL_SUCCESS)
        {
            MNN_PRINT("setArg err %d\n", (int)ret);
        }
        mOpenCLBackend->recordKernel2d(unit.kernel,
            {(uint32_t)UP_DIV((region[1] * region[3]), 16)*16,
            (uint32_t)UP_DIV((region[0] * region[2]), 16)*16},
            {8, 8});
        mUnits.emplace_back(unit);
        return NO_ERROR;
    }
    
    mTmpInitBuffers.resize(2);
    int x = cmd->size()->data()[0];
    int y = cmd->size()->data()[1];
    int z = cmd->size()->data()[2];
    int inputSize = mTensors[cmd->indexes()->data()[1]]->elementSize();

    auto srcStride = cmd->view()->GetAs<View>(1)->stride()->data();
    auto dstStride = cmd->view()->GetAs<View>(0)->stride()->data();
    for(int i = 0; i < 3; ++i) {
        mStride_src[i] = srcStride[i];
        mStride_dst[i] = dstStride[i];
    }

    mStride_src[3] = 0;
    mStride_dst[3] = 0;
    ::memset(mStep, 0, 2 * sizeof(int));

    // tile input
    {
       auto input = mTensors[cmd->indexes()->data()[1]];
       std::vector<int> Shape = tensorShapeFormat(input);
       const int Channel = Shape.at(3);
       const int Width = Shape.at(2);
       const int Height = Shape.at(1);
       const int Batch = Shape.at(0);
        mTmpInitBuffers[1] = bufferPool->alloc(input->elementSize() * bufferUnitSize);

       Unit unit;
       _TileTensor(mTensors[cmd->indexes()->data()[1]], mTmpInitBuffers[1], unit.kernel, unit.globalWorkSize, unit.localWorkSize, Width, Height,Channel, Batch, mOpenCLBackend, mBuildOptions);
       mUnits.emplace_back(unit);
    }
    
    // tile output
    {
       auto output = mTensors[cmd->indexes()->data()[0]];
       std::vector<int> Shape = tensorShapeFormat(output);
       const int Channel = Shape.at(3);
       const int Width = Shape.at(2);
       const int Height = Shape.at(1);
       const int Batch = Shape.at(0);
       mTmpInitBuffers[0] = bufferPool->alloc(output->elementSize() * bufferUnitSize);

       Unit unit;
       _TileTensor(mTensors[cmd->indexes()->data()[0]], mTmpInitBuffers[0], unit.kernel, unit.globalWorkSize, unit.localWorkSize, Width, Height,Channel, Batch, mOpenCLBackend, mBuildOptions);
       mUnits.emplace_back(unit);
    }

    // gather
    {
       int offset_index = 0;
       Unit unit;
       std::string KernelName = "batch_gather";
       unit.kernel = runTime->buildKernel("loop", KernelName, mBuildOptions, mOpenCLBackend->getPrecision(), mTensors[cmd->indexes()->data()[1]], mTensors[cmd->indexes()->data()[0]]);
       uint32_t mMaxWorkGroupSize = static_cast<uint32_t>(runTime->getMaxWorkGroupSize(unit.kernel));
       std::vector<uint32_t> mGlobalWorkSize = {(uint32_t)(x * y), (uint32_t)(z), (uint32_t)(1)};

       uint32_t index = 0;
       cl_int ret = CL_SUCCESS;
       ret |= unit.kernel->get().setArg(index++, mGlobalWorkSize[0]);
       ret |= unit.kernel->get().setArg(index++, mGlobalWorkSize[1]);
       ret |= unit.kernel->get().setArg(index++, mGlobalWorkSize[2]);
       ret |= unit.kernel->get().setArg(index++, *mTmpInitBuffers[0]);
       ret |= unit.kernel->get().setArg(index++, *mTmpInitBuffers[1]);
       ret |= unit.kernel->get().setArg(index++, x);
       ret |= unit.kernel->get().setArg(index++, sizeof(mStride_src), mStride_src);
       ret |= unit.kernel->get().setArg(index++, sizeof(mStride_dst), mStride_dst);
       ret |= unit.kernel->get().setArg(index++, sizeof(mStep), mStep);
       ret |= unit.kernel->get().setArg(index++, inputSize);
       MNN_CHECK_CL_SUCCESS(ret, "setArg LoopGatherExecution");

       std::vector<uint32_t> mLocalWorkSize = localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, runTime, KernelName, unit.kernel, mOpenCLBackend->getCLTuneLevel()).first;

       unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1], mGlobalWorkSize[2]};
       unit.localWorkSize  = {mLocalWorkSize[0], mLocalWorkSize[1], mLocalWorkSize[2]};
       mOpenCLBackend->recordKernel3d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
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
       _PackTensor(mTmpInitBuffers[0], mTensors[cmd->indexes()->data()[0]], unit.kernel, unit.globalWorkSize, unit.localWorkSize, Width, Height, Channel, Batch, mOpenCLBackend, mBuildOptions);
       mUnits.emplace_back(unit);
    }

    return NO_ERROR;
}
 ErrorCode LoopGatherExecution::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
     auto cmd                      = mLoop->commands()->GetAs<RegionCommand>(0);
     OpenCLBackend *mOpenCLBackend = (OpenCLBackend *)backend();
     auto runTime                  = mOpenCLBackend->getOpenCLRuntime();
     auto bufferPool               = mOpenCLBackend->getBufferPool();
     auto bufferUnitSize           = mOpenCLBackend->getPrecision() != BackendConfig::Precision_High ? sizeof(half_float::half) : sizeof(float);
     _setTensorStack(mTensors, inputs, outputs, mLoop);
     mUnits.clear();
     
     if(mLoop->initCommand() != nullptr){
         InitCommandOnEncode(inputs, outputs);
     }
     
     mOffsetBuffers.clear();
     mTmpBuffers.resize(2);
     int x = cmd->size()->data()[0];
     int y = cmd->size()->data()[1];
     int z = cmd->size()->data()[2];
     int n = mLoop->loopNumber();
     int inputSize = mTensors[cmd->indexes()->data()[1]]->elementSize();

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
        _TileTensor(mTensors[cmd->indexes()->data()[1]], mTmpBuffers[1], unit.kernel, unit.globalWorkSize, unit.localWorkSize, Width, Height,Channel, Batch, mOpenCLBackend, mBuildOptions);
        mUnits.emplace_back(unit);
     }
     
     // tile output
     {
        auto output = mTensors[cmd->indexes()->data()[0]];
        std::vector<int> Shape = tensorShapeFormat(output);
        const int Channel = Shape.at(3);
        const int Width = Shape.at(2);
        const int Height = Shape.at(1);
        const int Batch = Shape.at(0);
        mTmpBuffers[0] = bufferPool->alloc(output->elementSize() * bufferUnitSize);

        Unit unit;
        _TileTensor(mTensors[cmd->indexes()->data()[0]], mTmpBuffers[0], unit.kernel, unit.globalWorkSize, unit.localWorkSize, Width, Height,Channel, Batch, mOpenCLBackend, mBuildOptions);
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
            _TileTensor(input, mOffsetBuffers.back(), unit.kernel, unit.globalWorkSize, unit.localWorkSize, Width, Height, Channel, Batch, mOpenCLBackend, mBuildOptions);
            mUnits.emplace_back(unit);
        }
     }
     
     // gather
     {
        int offset_index = 0;
        std::set<std::string> buildOptions = mBuildOptions;
        if (mIter[0] >= 0) {
            buildOptions.emplace("-DOFFSET_DST");
        }
        if (mIter[1] >= 0) {
            buildOptions.emplace("-DOFFSET_SRC");
        }
        Unit unit;
        std::string KernelName = "batch_gather";
        unit.kernel = runTime->buildKernel("loop", KernelName, buildOptions, mOpenCLBackend->getPrecision(), mTensors[cmd->indexes()->data()[1]], mTensors[cmd->indexes()->data()[0]]);
        uint32_t mMaxWorkGroupSize = static_cast<uint32_t>(runTime->getMaxWorkGroupSize(unit.kernel));
        std::vector<uint32_t> mGlobalWorkSize = {(uint32_t)(x * y), (uint32_t)(z), (uint32_t)(n)};

        uint32_t index = 0;
        cl_int ret = CL_SUCCESS;
        ret |= unit.kernel->get().setArg(index++, mGlobalWorkSize[0]);
        ret |= unit.kernel->get().setArg(index++, mGlobalWorkSize[1]);
        ret |= unit.kernel->get().setArg(index++, mGlobalWorkSize[2]);
        ret |= unit.kernel->get().setArg(index++, *mTmpBuffers[0]);
        ret |= unit.kernel->get().setArg(index++, *mTmpBuffers[1]);
        for (int i = 0; i < cmd->iterIndexes()->size(); ++i) {
            if (mIter[i] >= 0) {
                ret |= unit.kernel->get().setArg(index++, *mOffsetBuffers[offset_index++]);
            }
        }
        ret |= unit.kernel->get().setArg(index++, x);
        ret |= unit.kernel->get().setArg(index++, sizeof(mStride_src), mStride_src);
        ret |= unit.kernel->get().setArg(index++, sizeof(mStride_dst), mStride_dst);
        ret |= unit.kernel->get().setArg(index++, sizeof(mStep), mStep);
        ret |= unit.kernel->get().setArg(index++, inputSize);
        MNN_CHECK_CL_SUCCESS(ret, "setArg LoopGatherExecution");

        std::vector<uint32_t> mLocalWorkSize = localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, runTime, KernelName, unit.kernel, mOpenCLBackend->getCLTuneLevel()).first;

        unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1], mGlobalWorkSize[2]};
        unit.localWorkSize  = {mLocalWorkSize[0], mLocalWorkSize[1], mLocalWorkSize[2]};
        mOpenCLBackend->recordKernel3d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
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
        _PackTensor(mTmpBuffers[0], mTensors[cmd->indexes()->data()[0]], unit.kernel, unit.globalWorkSize, unit.localWorkSize, Width, Height, Channel, Batch, mOpenCLBackend, mBuildOptions);
        mUnits.emplace_back(unit);
     }

     for (int i = 0; i < mTmpBuffers.size(); ++i) {
        bufferPool->recycle(mTmpBuffers[i]);
     }
     for (int i = 0; i < mOffsetBuffers.size(); ++i) {
        bufferPool->recycle(mOffsetBuffers[i]);
     }
     if(mLoop->initCommand() != nullptr){
         for (int i = 0; i < mTmpInitBuffers.size(); ++i) {
             bufferPool->recycle(mTmpInitBuffers[i]);
         }
     }

     return NO_ERROR;
 }


LoopBatchMatMulExecution::LoopBatchMatMulExecution(const LoopParam *loop, const MNN::Op *op, Backend *bn)
     : CommonExecution(bn, op) {
     mLoop = loop;
     mTensors.resize(mLoop->tensorNumber());
}
ErrorCode LoopBatchMatMulExecution::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
     auto cmd     = mLoop->commands()->GetAs<RegionCommand>(0);
     mHasBias = cmd->indexes()->size() > 3;
     mTransposeA = cmd->op()->main_as_MatMul()->transposeA();
     mTransposeB = cmd->op()->main_as_MatMul()->transposeB();
     OpenCLBackend *mOpenCLBackend = (OpenCLBackend *)backend();
     auto runTime = mOpenCLBackend->getOpenCLRuntime();
     auto bufferPool = mOpenCLBackend->getBufferPool();
     auto bufferUnitSize = mOpenCLBackend->getPrecision() != BackendConfig::Precision_High ? sizeof(half_float::half) : sizeof(float);
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
        _TileTensor(input, mTmpBuffers[i], unit.kernel, unit.globalWorkSize, unit.localWorkSize, Width, Height, Channel, Batch, mOpenCLBackend, mBuildOptions);
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
            _TileTensor(input, mOffsetBuffers.back(), unit.kernel, unit.globalWorkSize, unit.localWorkSize, Width, Height, Channel, Batch, mOpenCLBackend, mBuildOptions);
            mUnits.emplace_back(unit);
        }
     }

     // matmul
     {
        mTmpBuffers[0] = bufferPool->alloc(n * e * h * bufferUnitSize);
        int offset_index = 0;

        Unit unit;
        std::string KernelName = "batch_matmul";
        std::set<std::string> buildOptions = mBuildOptions;
        if (mHasBias) {
            buildOptions.emplace("-DBIAS");
        }
        if (mTransposeA) {
            buildOptions.emplace("-DTRANSPOSE_A");
        }
        if (mTransposeB) {
            buildOptions.emplace("-DTRANSPOSE_B");
        }
        buildOptions.emplace("-DH_LEAVES=" + std::to_string(h % 4));
        unit.kernel = runTime->buildKernel("loop", KernelName, buildOptions, mOpenCLBackend->getPrecision(), mTensors[cmd->indexes()->data()[1]], mTensors[cmd->indexes()->data()[0]]);
        uint32_t mMaxWorkGroupSize = static_cast<uint32_t>(runTime->getMaxWorkGroupSize(unit.kernel));
        std::vector<uint32_t> mGlobalWorkSize = {(uint32_t)(UP_DIV(h, 4)), (uint32_t)(UP_DIV(e, 4)),(uint32_t)(n)};

        uint32_t index = 0;
        cl_int ret = CL_SUCCESS;
        ret |= unit.kernel->get().setArg(index++, mGlobalWorkSize[0]);
        ret |= unit.kernel->get().setArg(index++, mGlobalWorkSize[1]);
        ret |= unit.kernel->get().setArg(index++, mGlobalWorkSize[2]);
        ret |= unit.kernel->get().setArg(index++, *mTmpBuffers[0]);
        ret |= unit.kernel->get().setArg(index++, *mTmpBuffers[1]);
        ret |= unit.kernel->get().setArg(index++, *mTmpBuffers[2]);
        if (mHasBias) {
            ret |= unit.kernel->get().setArg(index++, *mTmpBuffers[3]);
        }
        for (int i = 0; i < cmd->iterIndexes()->size(); ++i) {
            if (mIter[i] >= 0) {
                ret |= unit.kernel->get().setArg(index++, *mOffsetBuffers[offset_index++]);
            } else {
                ret |= unit.kernel->get().setArg(index++, *mTmpBuffers[0]);
            }
        }
        ret |= unit.kernel->get().setArg(index++, e);
        ret |= unit.kernel->get().setArg(index++, l);
        ret |= unit.kernel->get().setArg(index++, h);
        ret |= unit.kernel->get().setArg(index++, sizeof(mOffset), mOffset);
        ret |= unit.kernel->get().setArg(index++, sizeof(mIter), mIter);
        ret |= unit.kernel->get().setArg(index++, sizeof(mStep), mStep);
        MNN_CHECK_CL_SUCCESS(ret, "setArg LoopBatchMatMulExecution");

        std::vector<uint32_t> mLocalWorkSize = localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, runTime, KernelName, unit.kernel, mOpenCLBackend->getCLTuneLevel()).first;

        unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1], mGlobalWorkSize[2]};
        unit.localWorkSize  = {mLocalWorkSize[0], mLocalWorkSize[1], mLocalWorkSize[2]};
        mUnits.emplace_back(unit);
        mOpenCLBackend->recordKernel3d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
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
        _PackTensor(mTmpBuffers[0], output, unit.kernel, unit.globalWorkSize, unit.localWorkSize, Width, Height, Channel, Batch, mOpenCLBackend, mBuildOptions);
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

LoopBinaryExecution::LoopBinaryExecution(const LoopParam *loop, const std::string &compute, const MNN::Op *op, Backend *bn)
    : CommonExecution(bn, op) {
    mLoop = loop;
    mTensors.resize(mLoop->tensorNumber());
    mBuildOptions.emplace("-DLOOP_BINARY_OPERATOR=" + compute);
}

ErrorCode LoopBinaryExecution::cumSumOnEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto cmd                      = mLoop->commands()->GetAs<RegionCommand>(0);
    OpenCLBackend *mOpenCLBackend = (OpenCLBackend *)backend();
    auto bufferPool = mOpenCLBackend->getBufferPool();
    auto runTime                  = mOpenCLBackend->getOpenCLRuntime();
    auto bufferUnitSize           = mOpenCLBackend->getPrecision() != BackendConfig::Precision_High ? sizeof(half_float::half) : sizeof(float);
    mUnits.clear();
    mTmpBuffers.resize(2);
    mOffset[0] = cmd->view()->GetAs<View>(0)->offset();
    mOffset[1] = cmd->view()->GetAs<View>(1)->offset();
    mOffset[2] = cmd->view()->GetAs<View>(2)->offset();
    ::memcpy(mStep, cmd->steps()->data(), cmd->steps()->size() * sizeof(int));
    int loopNumber = mLoop->loopNumber();
    int z = cmd->size()->data()[0];
    int y = cmd->size()->data()[1];
    int x = cmd->size()->data()[2];
    int n = mLoop->loopNumber();
    int inputSize = mTensors[cmd->indexes()->data()[1]]->elementSize();
    
    auto src0Stride = cmd->view()->GetAs<View>(1)->stride()->data();
    auto src1Stride = cmd->view()->GetAs<View>(2)->stride()->data();
    auto dstStride = cmd->view()->GetAs<View>(0)->stride()->data();
    for (int i = 0; i < 3; ++i) {
        mStride_src0[i] = src0Stride[i];
        mStride_src1[i] = src1Stride[i];
        mStride_dst[i] = dstStride[i];
    }

    // tile input
    // mTensors cmd->indexes()->data() = {2, 0, 1} -> {output, input0, input1}, output = input0
    for (int i = 1; i < cmd->indexes()->size(); ++i) {
       auto input = mTensors[cmd->indexes()->data()[i]];
       std::vector<int> Shape = tensorShapeFormat(input);
       const int Channel = Shape.at(3);
       const int Width = Shape.at(2);
       const int Height = Shape.at(1);
       const int Batch = Shape.at(0);
       mTmpBuffers[i - 1] = bufferPool->alloc(Batch * Channel * ROUND_UP(Height, 4) * ROUND_UP(Width, 4) * bufferUnitSize);

       Unit unit;
       _TileTensor(input, mTmpBuffers[i - 1], unit.kernel, unit.globalWorkSize, unit.localWorkSize, Width, Height, Channel, Batch, mOpenCLBackend, mBuildOptions);
       mUnits.emplace_back(unit);
    }
    
    {
        Unit unit;
        std::set<std::string> buildOptions = mBuildOptions;
        buildOptions.emplace("-DCOMPUTE_CUMSUM");
        unit.kernel = runTime->buildKernel("loop", "loop_cumsum", buildOptions, mOpenCLBackend->getPrecision(), mTensors[cmd->indexes()->data()[1]], mTensors[cmd->indexes()->data()[0]]);
        uint32_t mMaxWorkGroupSize = static_cast<uint32_t>(runTime->getMaxWorkGroupSize(unit.kernel));
        
        std::vector<uint32_t> mGlobalWorkSize = {(uint32_t)(x), (uint32_t)(y), (uint32_t)(z)};
        uint32_t index = 0;
        cl_int ret = CL_SUCCESS;
        ret |= unit.kernel->get().setArg(index++, mGlobalWorkSize[0]);
        ret |= unit.kernel->get().setArg(index++, mGlobalWorkSize[1]);
        ret |= unit.kernel->get().setArg(index++, mGlobalWorkSize[2]);
        ret |= unit.kernel->get().setArg(index++, *mTmpBuffers[0]); // cumsum input0 == output -> mTmpBuffers[0] == mTmpBuffers[2]
        ret |= unit.kernel->get().setArg(index++, *mTmpBuffers[0]);
        ret |= unit.kernel->get().setArg(index++, *mTmpBuffers[1]);
        ret |= unit.kernel->get().setArg(index++, mStride_src0[0]);
        ret |= unit.kernel->get().setArg(index++, mStride_src0[1]);
        ret |= unit.kernel->get().setArg(index++, mStride_src0[2]);
        ret |= unit.kernel->get().setArg(index++, mStride_src1[0]);
        ret |= unit.kernel->get().setArg(index++, mStride_src1[1]);
        ret |= unit.kernel->get().setArg(index++, mStride_src1[2]);
        ret |= unit.kernel->get().setArg(index++, mStride_dst[0]);
        ret |= unit.kernel->get().setArg(index++, mStride_dst[1]);
        ret |= unit.kernel->get().setArg(index++, mStride_dst[2]);
        ret |= unit.kernel->get().setArg(index++, loopNumber);
        ret |= unit.kernel->get().setArg(index++, sizeof(mOffset), mOffset);
        ret |= unit.kernel->get().setArg(index++, sizeof(mStep), mStep);
        MNN_CHECK_CL_SUCCESS(ret, "setArg LoopCumsumExecution");
        
        std::vector<uint32_t> mLocalWorkSize = localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, runTime, "loop_cumsum", unit.kernel, mOpenCLBackend->getCLTuneLevel()).first;
        
        unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1], mGlobalWorkSize[2]};
        unit.localWorkSize  = {mLocalWorkSize[0], mLocalWorkSize[1], mLocalWorkSize[2]};
        mUnits.emplace_back(unit);
        mOpenCLBackend->recordKernel3d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
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
       _PackTensor(mTmpBuffers[0], output, unit.kernel, unit.globalWorkSize, unit.localWorkSize, Width, Height, Channel, Batch, mOpenCLBackend, mBuildOptions);
       mUnits.emplace_back(unit);
    }

   for (int i = 0; i < mTmpBuffers.size(); ++i) {
        bufferPool->recycle(mTmpBuffers[i]);
   }
    
    return NO_ERROR;
}


ErrorCode LoopBinaryExecution::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto cmd                      = mLoop->commands()->GetAs<RegionCommand>(0);
    if(cmd->op()->main_as_BinaryOp()->opType() == BinaryOpOperation_MOD && (outputs[0]->getType().code == halide_type_int || outputs[0]->getType().code == halide_type_uint)){
        mBuildOptions.emplace("-DINT_COMPUTE_MOD");
    }
    OpenCLBackend *mOpenCLBackend = (OpenCLBackend *)backend();
    auto runTime                  = mOpenCLBackend->getOpenCLRuntime();
    _setTensorStack(mTensors, inputs, outputs, mLoop);
    
    // cumsum
    if(!mLoop->parallel())
        return cumSumOnEncode(inputs, outputs);
    
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
    
    bool broadcastInput0 = false;
    bool broadcastInput1 = false;
    int input0Shape[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    int input1Shape[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    int outputShape[8] = {1, 1, 1, 1, 1, 1, 1, 1};

    int offset0 = output->dimensions() - input0->dimensions();
    int offset1 = output->dimensions() - input1->dimensions();
    for (int i = 0; i < input0->dimensions(); ++i) {
        input0Shape[i + offset0] = input0->length(i);
    }
    for (int i = 0; i < input1->dimensions(); ++i) {
        input1Shape[i + offset1] = input1->length(i);
    }
    for(int i =0;i<output->dimensions();++i){
        outputShape[i] = output->length(i);
    }
    if (TensorUtils::getDescribe(input0)->dimensionFormat == MNN::MNN_DATA_FORMAT_NHWC)
    {
        int iN = input0Shape[0];
        int iH = input0Shape[1];
        int iW = input0Shape[2];
        int iC = input0Shape[3];
            
        if(input0->dimensions() > 4)
        {
            for(int i = 4; i < input0->dimensions(); i++)
            {
                iC *= input0Shape[i];
            }
        }
        input0Shape[0] = iN;
        input0Shape[1] = iC;
        input0Shape[2] = iH;
        input0Shape[3] = iW;
        input0Shape[4] = 1;
    }
    if (TensorUtils::getDescribe(input1)->dimensionFormat == MNN::MNN_DATA_FORMAT_NHWC)
    {
        int iN = input1Shape[0];
        int iH = input1Shape[1];
        int iW = input1Shape[2];
        int iC = input1Shape[3];
            
        if(input1->dimensions() > 4)
        {
            for(int i = 4; i < input1->dimensions(); i++)
            {
                iC *= input1Shape[i];
            }
        }
        input1Shape[0] = iN;
        input1Shape[1] = iC;
        input1Shape[2] = iH;
        input1Shape[3] = iW;
        input1Shape[4] = 1;
    }
    if (TensorUtils::getDescribe(output)->dimensionFormat == MNN::MNN_DATA_FORMAT_NHWC)
    {
        int iN = outputShape[0];
        int iH = outputShape[1];
        int iW = outputShape[2];
        int iC = outputShape[3];
            
        if(input1->dimensions() > 4)
        {
            for(int i = 4; i < input1->dimensions(); i++)
            {
                iC *= outputShape[i];
            }
        }
        input1Shape[0] = iN;
        outputShape[1] = iC;
        outputShape[2] = iH;
        outputShape[3] = iW;
        outputShape[4] = 1;
    }
        
    const int Channel = Shape.at(3);
    const int Width = Shape.at(2);
    const int Height = Shape.at(1);
    const int Batch = Shape.at(0);
    const int ChannelBlock = UP_DIV(Channel, 4);
    auto BuildOptions = mBuildOptions;
    std::string KernelName = "broadcast_binary";
    unit.kernel = runTime->buildKernel("loop", KernelName, BuildOptions, mOpenCLBackend->getPrecision(), input0, output);
    uint32_t mMaxWorkGroupSize = static_cast<uint32_t>(runTime->getMaxWorkGroupSize(unit.kernel));
       
    std::vector<uint32_t> mGlobalWorkSize = {(uint32_t)(Width), (uint32_t)(Height), (uint32_t)(Batch * ChannelBlock)};

    uint32_t index = 0;
    cl_int ret = CL_SUCCESS;
    ret |= unit.kernel->get().setArg(index++, mGlobalWorkSize[0]);
    ret |= unit.kernel->get().setArg(index++, mGlobalWorkSize[1]);
    ret |= unit.kernel->get().setArg(index++, mGlobalWorkSize[2]);
    ret |= unit.kernel->get().setArg(index++, openCLImage(output));
    ret |= unit.kernel->get().setArg(index++, openCLImage(input0));
    ret |= unit.kernel->get().setArg(index++, openCLImage(input1));
    ret |= unit.kernel->get().setArg(index++, sizeof(input0Shape), input0Shape);
    ret |= unit.kernel->get().setArg(index++, sizeof(Input0Size), Input0Size);
    ret |= unit.kernel->get().setArg(index++, sizeof(input1Shape), input1Shape);
    ret |= unit.kernel->get().setArg(index++, sizeof(Input1Size), Input1Size);
    ret |= unit.kernel->get().setArg(index++, sizeof(outputShape), outputShape);
    ret |= unit.kernel->get().setArg(index++, Width);
    ret |= unit.kernel->get().setArg(index++, Height);
    ret |= unit.kernel->get().setArg(index++, Channel);
    ret |= unit.kernel->get().setArg(index++, ChannelBlock);
    MNN_CHECK_CL_SUCCESS(ret, "setArg LoopBinaryExecution");

    std::vector<uint32_t> mLocalWorkSize = localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, runTime, KernelName, unit.kernel, mOpenCLBackend->getCLTuneLevel()).first;

    unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1], mGlobalWorkSize[2]};
    unit.localWorkSize  = {mLocalWorkSize[0], mLocalWorkSize[1], mLocalWorkSize[2]};
    mOpenCLBackend->recordKernel3d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
    mUnits.emplace_back(unit);


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
        // Make Tensor Stack
        if (1 == loop->commands()->size()) {
            auto cmd   = loop->commands()->GetAs<RegionCommand>(0);
            auto subop = cmd->op();
            if (OpType_UnaryOp == subop->type() && nullptr == subop->main() && cmd->fuse() < 0) {
                return new LoopGatherExecution(loop, op, backend);
            }
            if (OpType_MatMul == subop->type() && loop->parallel() && nullptr == loop->initCommand()) {
                return new LoopBatchMatMulExecution(loop, op, backend);
            }
            if (OpType_BinaryOp == subop->type() && nullptr == loop->initCommand()) {
                switch (subop->main_as_BinaryOp()->opType()) {
                    case BinaryOpOperation_MUL:
                        return new LoopBinaryExecution(loop, "in0*in1", op, backend);
                    case BinaryOpOperation_ADD:
                        return new LoopBinaryExecution(loop, "in0+in1", op, backend);
                    case BinaryOpOperation_SUB:
                        return new LoopBinaryExecution(loop, "in0-in1", op, backend);
                    case BinaryOpOperation_REALDIV:
                        return new LoopBinaryExecution(loop, "sign(in1)*in0/(fabs(in1)>(float4)((float)0.0000001)?fabs(in1):(float4)((float)0.0000001))", op, backend);
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
                        return new LoopBinaryExecution(loop, "floor(sign(in1)*in0/(fabs(in1)>(float4)((float)0.0000001)?fabs(in1):(float4)((float)0.0000001)))", op, backend);
                    case BinaryOpOperation_FLOORMOD:
                        return new LoopBinaryExecution(loop, "in0-floor(sign(in1)*in0/(fabs(in1)>(float4)((float)0.0000001)?fabs(in1):(float4)((float)0.0000001)))*in1", op, backend);
                    case BinaryOpOperation_POW:
                        return new LoopBinaryExecution(loop, "pow(in0,in1)", op, backend);
                    case BinaryOpOperation_SquaredDifference:
                        return new LoopBinaryExecution(loop, "(in0-in1)*(in0-in1)", op, backend);
                    case BinaryOpOperation_ATAN2:
                        return new LoopBinaryExecution(loop, "(in1==(float4)0?(sign(in0)*(float4)(PI/2)):(atan(in0/in1)+(in1>(float4)0?(float4)0:sign(in0)*(float4)PI)))", op, backend);
                    case BinaryOpOperation_NOTEQUAL:
                        return new LoopBinaryExecution(loop, "convert_float4(-isnotequal(in0,in1))", op, backend);
                    case BinaryOpOperation_MOD:
                        return new LoopBinaryExecution(loop, "in0-floor(sign(in1)*in0/(fabs(in1)>(float4)((float)0.0000001)?fabs(in1):(float4)((float)0.0000001)))*in1", op, backend);
                    default:
                        break;
                }
                return nullptr;
            }
        }
        return nullptr;
    }
};

REGISTER_OPENCL_OP_CREATOR(LoopCreator, OpType_While, IMAGE);

} // namespace OpenCL
} // namespace MNN
