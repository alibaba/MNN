//
//  LoopBufExecution.cpp
//  MNN
//
//  Created by MNN on 2023/04/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED

#include "backend/opencl/execution/buffer/LoopBufExecution.hpp"

namespace MNN {
namespace OpenCL {

static void _TileOrPackTensor(Tensor *input, Tensor *output, std::shared_ptr<KernelWrap>& kernelW, cl::NDRange &globalWorkSize,
                        cl::NDRange &localWorkSize, const int Width, const int Height, const int Channel,
                        const int Batch, OpenCLBackend *bn, const std::string &KernelName, std::set<std::string> buildOptions) {
    if (TensorUtils::getDescribe(output)->dimensionFormat == MNN::MNN_DATA_FORMAT_NHWC || TensorUtils::getDescribe(input)->dimensionFormat == MNN::MNN_DATA_FORMAT_NHWC){
        buildOptions.emplace("-DMNN_NHWC");
    }
    if(input->getType().code == halide_type_int){
        kernelW = bn->getOpenCLRuntime()->buildKernel("loop_buf", KernelName, buildOptions, input, input);
    }else if (output->getType().code == halide_type_int){
        kernelW = bn->getOpenCLRuntime()->buildKernel("loop_buf", KernelName, buildOptions, output, output);
    }else {
        kernelW = bn->getOpenCLRuntime()->buildKernel("loop_buf", KernelName, buildOptions, input, output);
    }
    auto kernel = kernelW->get();
    
    uint32_t mMaxWorkGroupSize  = static_cast<uint32_t>(bn->getOpenCLRuntime()->getMaxWorkGroupSize(kernelW));
    std::vector<uint32_t> mGlobalWorkSize = {(uint32_t)(Width * Height), (uint32_t)(UP_DIV(Channel, 4)), (uint32_t)(Batch)};

    uint32_t index = 0;
    cl_int ret = CL_SUCCESS;
    ret |= kernel.setArg(index++, mGlobalWorkSize[0]);
    ret |= kernel.setArg(index++, mGlobalWorkSize[1]);
    ret |= kernel.setArg(index++, mGlobalWorkSize[2]);
    ret |= kernel.setArg(index++, openCLBuffer(input));
    ret |= kernel.setArg(index++, openCLBuffer(output));
    ret |= kernel.setArg(index++, Width);
    ret |= kernel.setArg(index++, Height);
    ret |= kernel.setArg(index++, Channel);
    MNN_CHECK_CL_SUCCESS(ret, "setArg LoopBuf _TileOrPackTensor");

    std::vector<uint32_t> mLocalWorkSize = localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, bn->getOpenCLRuntime(), KernelName, kernelW).first;

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


 LoopGatherBufExecution::LoopGatherBufExecution(const LoopParam *loop, const MNN::Op *op, Backend *bn)
     : CommonExecution(bn, op) {
     mLoop = loop;
     mTensors.resize(mLoop->tensorNumber());
     auto cmd = loop->commands()->GetAs<RegionCommand>(0);
 }
 ErrorCode LoopGatherBufExecution::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
     auto cmd                      = mLoop->commands()->GetAs<RegionCommand>(0);
     OpenCLBackend *mOpenCLBackend = (OpenCLBackend *)backend();
     auto runTime                  = mOpenCLBackend->getOpenCLRuntime();
     _setTensorStack(mTensors, inputs, outputs, mLoop);
     mUnits.clear();
     mOffsetTensors.clear();
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
     
     // gather
     {
         auto input = mTensors[cmd->indexes()->data()[1]];
         auto output = mTensors[cmd->indexes()->data()[0]];
         std::vector<int> inputShape = tensorShapeFormat(input);
         std::vector<int> outputShape = tensorShapeFormat(output);
         int inputShapeVec[4] = {inputShape[2], inputShape[1], inputShape[3], inputShape[0]};
         int outputShapeVec[4] = {outputShape[2], outputShape[1], outputShape[3], inputShape[0]};
         int offset_index = 0;
         
         Unit unit;
         std::set<std::string> buildOptions;
         if (TensorUtils::getDescribe(output)->dimensionFormat == MNN::MNN_DATA_FORMAT_NHWC){
             buildOptions.emplace("-DGATHER_OUTPUT_NHWC");
         }
         if (TensorUtils::getDescribe(input)->dimensionFormat == MNN::MNN_DATA_FORMAT_NHWC){
             buildOptions.emplace("-DGATHER_INPUT_NHWC");
         }
         if (mIter[0] >= 0) {
             auto tensor = mTensors[cmd->iterIndexes()->data()[0]];
             std::vector<int> shape = tensorShapeFormat(tensor);
             if(shape[3] == 1){
                 buildOptions.emplace("-DOFFSET_DST_CHANNEL_STRIDE4");
             }
         }
         if (mIter[1] >= 0) {
             auto tensor = mTensors[cmd->iterIndexes()->data()[1]];
             std::vector<int> shape = tensorShapeFormat(tensor);
             if(shape[3] == 1){
                 buildOptions.emplace("-DOFFSET_SRC_CHANNEL_STRIDE4");
             }
         }
         unit.kernel = runTime->buildKernel("gather_buf", "batch_gather_buf", buildOptions, input, output);
         uint32_t mMaxWorkGroupSize = static_cast<uint32_t>(runTime->getMaxWorkGroupSize(unit.kernel));
         std::vector<uint32_t> mGlobalWorkSize = {(uint32_t)(x * y), (uint32_t)(z), (uint32_t)(n)};

         uint32_t index = 0;
         cl_int ret = CL_SUCCESS;
         ret |= unit.kernel->get().setArg(index++, mGlobalWorkSize[0]);
         ret |= unit.kernel->get().setArg(index++, mGlobalWorkSize[1]);
         ret |= unit.kernel->get().setArg(index++, mGlobalWorkSize[2]);
         ret |= unit.kernel->get().setArg(index++, openCLBuffer(output));
         ret |= unit.kernel->get().setArg(index++, openCLBuffer(input));
         for (int i = 0; i < cmd->iterIndexes()->size(); ++i) {
             if (mIter[i] >= 0) {
                 ret |= unit.kernel->get().setArg(index++, openCLBuffer(mTensors[cmd->iterIndexes()->data()[i]]));
             } else {
                 ret |= unit.kernel->get().setArg(index++, openCLBuffer(input));
             }
         }
         ret |= unit.kernel->get().setArg(index++, x);
         ret |= unit.kernel->get().setArg(index++, sizeof(mStride_src), mStride_src);
         ret |= unit.kernel->get().setArg(index++, sizeof(mStride_dst), mStride_dst);
         ret |= unit.kernel->get().setArg(index++, sizeof(mStep), mStep);
         ret |= unit.kernel->get().setArg(index++, sizeof(mIter), mIter);
         ret |= unit.kernel->get().setArg(index++, sizeof(outputShapeVec), outputShapeVec);
         ret |= unit.kernel->get().setArg(index++, sizeof(inputShapeVec), inputShapeVec);
         ret |= unit.kernel->get().setArg(index++, inputSize);
         MNN_CHECK_CL_SUCCESS(ret, "setArg LoopGatherBufExecution");

         std::vector<uint32_t> mLocalWorkSize = localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, runTime, "batch_gather_buf", unit.kernel).first;

         unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1], mGlobalWorkSize[2]};
         unit.localWorkSize  = {mLocalWorkSize[0], mLocalWorkSize[1], mLocalWorkSize[2]};
         mUnits.emplace_back(unit);
         mOpenCLBackend->recordKernel3d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
     }


     return NO_ERROR;
 }


LoopBatchMatMulBufExecution::LoopBatchMatMulBufExecution(const LoopParam *loop, const MNN::Op *op, Backend *bn)
     : CommonExecution(bn, op) {
     mLoop = loop;
     mTensors.resize(mLoop->tensorNumber());
     auto cmd = loop->commands()->GetAs<RegionCommand>(0);
}
ErrorCode LoopBatchMatMulBufExecution::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
     auto cmd     = mLoop->commands()->GetAs<RegionCommand>(0);
     mHasBias = cmd->indexes()->size() > 3;
     mTransposeA = cmd->op()->main_as_MatMul()->transposeA();
     mTransposeB = cmd->op()->main_as_MatMul()->transposeB();
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
        mTmpTensors[i] = std::make_shared<Tensor>(Tensor::createDevice<float>(std::vector<int>{Batch, Channel,  ROUND_UP(Height, 4), ROUND_UP(Width, 4)}, Tensor::CAFFE));
        mOpenCLBackend->onAcquireBuffer(mTmpTensors[i].get(), Backend::DYNAMIC);       

        Unit unit;
        _TileOrPackTensor(input, mTmpTensors[i].get(), unit.kernel, unit.globalWorkSize, unit.localWorkSize, Width, Height, Channel, Batch, mOpenCLBackend, "tile_buf", mBuildOptions);
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
            mOffsetTensors.emplace_back(std::make_shared<Tensor>(Tensor::createDevice<float>(std::vector<int>{Batch, Channel, Height, Width}, Tensor::CAFFE)));
            mOpenCLBackend->onAcquireBuffer(mOffsetTensors.back().get(), Backend::DYNAMIC);

            Unit unit;
            _TileOrPackTensor(input, mOffsetTensors.back().get(), unit.kernel, unit.globalWorkSize, unit.localWorkSize, Width, Height, Channel, Batch, mOpenCLBackend, "tile_buf", mBuildOptions);
            mUnits.emplace_back(unit);
        }
     }

     // matmul
     {
        mTmpTensors[0] = std::make_shared<Tensor>(Tensor::createDevice<float>(std::vector<int>{1, n, e, h}, Tensor::CAFFE));
        mOpenCLBackend->onAcquireBuffer(mTmpTensors[0].get(), Backend::DYNAMIC);
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
        unit.kernel = runTime->buildKernel("loop", KernelName, buildOptions, mTensors[cmd->indexes()->data()[1]], mTensors[cmd->indexes()->data()[0]]);
        uint32_t mMaxWorkGroupSize = static_cast<uint32_t>(runTime->getMaxWorkGroupSize(unit.kernel));
        std::vector<uint32_t> mGlobalWorkSize = {(uint32_t)(UP_DIV(h, 4)), (uint32_t)(UP_DIV(e, 4)),(uint32_t)(n)};

        uint32_t index = 0;
        cl_int ret = CL_SUCCESS;
        ret |= unit.kernel->get().setArg(index++, mGlobalWorkSize[0]);
        ret |= unit.kernel->get().setArg(index++, mGlobalWorkSize[1]);
        ret |= unit.kernel->get().setArg(index++, mGlobalWorkSize[2]);
        ret |= unit.kernel->get().setArg(index++, openCLBuffer(mTmpTensors[0].get()));
        ret |= unit.kernel->get().setArg(index++, openCLBuffer(mTmpTensors[1].get()));
        ret |= unit.kernel->get().setArg(index++, openCLBuffer(mTmpTensors[2].get()));
        if (mHasBias) {
            ret |= unit.kernel->get().setArg(index++, openCLBuffer(mTmpTensors[3].get()));
        }
        for (int i = 0; i < cmd->iterIndexes()->size(); ++i) {
            if (mIter[i] >= 0) {
                ret |= unit.kernel->get().setArg(index++, openCLBuffer(mOffsetTensors[offset_index++].get()));
            } else {
                ret |= unit.kernel->get().setArg(index++, openCLBuffer(mTensors[cmd->indexes()->data()[1]]));
            }
        }
        ret |= unit.kernel->get().setArg(index++, e);
        ret |= unit.kernel->get().setArg(index++, l);
        ret |= unit.kernel->get().setArg(index++, h);
        ret |= unit.kernel->get().setArg(index++, sizeof(mOffset), mOffset);
        ret |= unit.kernel->get().setArg(index++, sizeof(mIter), mIter);
        ret |= unit.kernel->get().setArg(index++, sizeof(mStep), mStep);
        MNN_CHECK_CL_SUCCESS(ret, "setArg LoopBatchMatMulBufExecution");

        std::vector<uint32_t> mLocalWorkSize = localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, runTime, KernelName, unit.kernel).first;

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
        _TileOrPackTensor(mTmpTensors[0].get(), output, unit.kernel, unit.globalWorkSize, unit.localWorkSize, Width, Height, Channel, Batch, mOpenCLBackend, "pack_buf", mBuildOptions);
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

LoopBinaryBufExecution::LoopBinaryBufExecution(const LoopParam *loop, const std::string &compute, const MNN::Op *op, Backend *bn)
    : CommonExecution(bn, op) {
    mLoop = loop;
    mTensors.resize(mLoop->tensorNumber());
    mBuildOptions.emplace("-DLOOP_BINARY_OPERATOR=" + compute);
}

ErrorCode LoopBinaryBufExecution::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto cmd                      = mLoop->commands()->GetAs<RegionCommand>(0);
    OpenCLBackend *mOpenCLBackend = (OpenCLBackend *)backend();
    auto runTime                  = mOpenCLBackend->getOpenCLRuntime();
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
    std::string KernelName = "broadcast_binary_buf";
    unit.kernel = runTime->buildKernel("loop_buf", KernelName, BuildOptions, input0, output);
    uint32_t mMaxWorkGroupSize = static_cast<uint32_t>(runTime->getMaxWorkGroupSize(unit.kernel));

    std::vector<uint32_t> mGlobalWorkSize = {(uint32_t)(Width), (uint32_t)(Height), (uint32_t)(Batch * ChannelBlock)};

    uint32_t index = 0;
    cl_int ret = CL_SUCCESS;
    ret |= unit.kernel->get().setArg(index++, mGlobalWorkSize[0]);
    ret |= unit.kernel->get().setArg(index++, mGlobalWorkSize[1]);
    ret |= unit.kernel->get().setArg(index++, mGlobalWorkSize[2]);
    ret |= unit.kernel->get().setArg(index++, openCLBuffer(output));
    ret |= unit.kernel->get().setArg(index++, openCLBuffer(input0));
    ret |= unit.kernel->get().setArg(index++, openCLBuffer(input1));
    ret |= unit.kernel->get().setArg(index++, sizeof(input0Shape), input0Shape);
    ret |= unit.kernel->get().setArg(index++, sizeof(Input0Size), Input0Size);
    ret |= unit.kernel->get().setArg(index++, sizeof(input1Shape), input1Shape);
    ret |= unit.kernel->get().setArg(index++, sizeof(Input1Size), Input1Size);
    ret |= unit.kernel->get().setArg(index++, sizeof(outputShape), outputShape);
    ret |= unit.kernel->get().setArg(index++, Width);
    ret |= unit.kernel->get().setArg(index++, Height);
    ret |= unit.kernel->get().setArg(index++, Channel);
    ret |= unit.kernel->get().setArg(index++, ChannelBlock);
    MNN_CHECK_CL_SUCCESS(ret, "setArg LoopBinaryBufExecution");

    std::vector<uint32_t> mLocalWorkSize = localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, runTime, KernelName, unit.kernel).first;

    unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1], mGlobalWorkSize[2]};
    unit.localWorkSize  = {mLocalWorkSize[0], mLocalWorkSize[1], mLocalWorkSize[2]};
    mUnits.emplace_back(unit);
    mOpenCLBackend->recordKernel3d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
    return NO_ERROR;
}

class LoopBufCreator : public OpenCLBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
         for (int i = 0; i < inputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(inputs[i], false);
         }
         for (int i = 0; i < outputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(outputs[i], false);
         }
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
            if (OpType_BinaryOp == subop->type() && loop->parallel()) {
                switch (subop->main_as_BinaryOp()->opType()) {
                    case BinaryOpOperation_MUL:
                        return new LoopBinaryBufExecution(loop, "in0*in1", op, backend);
                    case BinaryOpOperation_ADD:
                        return new LoopBinaryBufExecution(loop, "in0+in1", op, backend);
                    case BinaryOpOperation_SUB:
                        return new LoopBinaryBufExecution(loop, "in0-in1", op, backend);
                    case BinaryOpOperation_REALDIV:
                        return new LoopBinaryBufExecution(loop, "sign(in1)*in0/(fabs(in1)>(float4)((float)0.0000001)?fabs(in1):(float4)((float)0.0000001))", op, backend);
                    case BinaryOpOperation_MINIMUM:
                        return new LoopBinaryBufExecution(loop, "in0>in1?in1:in0", op, backend);
                    case BinaryOpOperation_MAXIMUM:
                        return new LoopBinaryBufExecution(loop, "in0>in1?in0:in1", op, backend);
                    case BinaryOpOperation_GREATER:
                        return new LoopBinaryBufExecution(loop, "convert_float4(-isgreater(in0,in1))", op, backend);
                    case BinaryOpOperation_LESS:
                        return new LoopBinaryBufExecution(loop, "convert_float4(-isless(in0,in1))", op, backend);
                    case BinaryOpOperation_LESS_EQUAL:
                        return new LoopBinaryBufExecution(loop, "convert_float4(-islessequal(in0,in1))", op, backend);
                    case BinaryOpOperation_GREATER_EQUAL:
                        return new LoopBinaryBufExecution(loop, "convert_float4(-isgreaterequal(in0,in1))", op, backend);
                    case BinaryOpOperation_EQUAL:
                        return new LoopBinaryBufExecution(loop, "convert_float4(-isequal(in0,in1))", op, backend);
                    case BinaryOpOperation_FLOORDIV:
                        return new LoopBinaryBufExecution(loop, "floor(sign(in1)*in0/(fabs(in1)>(float4)((float)0.0000001)?fabs(in1):(float4)((float)0.0000001)))", op, backend);
                    case BinaryOpOperation_FLOORMOD:
                        return new LoopBinaryBufExecution(loop, "in0-floor(sign(in1)*in0/(fabs(in1)>(float4)((float)0.0000001)?fabs(in1):(float4)((float)0.0000001)))*in1", op, backend);
                    case BinaryOpOperation_POW:
                        return new LoopBinaryBufExecution(loop, "pow(in0,in1)", op, backend);
                    case BinaryOpOperation_SquaredDifference:
                        return new LoopBinaryBufExecution(loop, "(in0-in1)*(in0-in1)", op, backend);
                    case BinaryOpOperation_ATAN2:
                        return new LoopBinaryBufExecution(loop, "(in1==(float4)0?(sign(in0)*(float4)(PI/2)):(atan(in0/in1)+(in1>(float4)0?(float4)0:sign(in0)*(float4)PI)))", op, backend);
                    case BinaryOpOperation_NOTEQUAL:
                        return new LoopBinaryBufExecution(loop, "convert_float4(-isnotequal(in0,in1))", op, backend);
                    case BinaryOpOperation_MOD:
                        return new LoopBinaryBufExecution(loop, "in0-floor(sign(in1)*in0/(fabs(in1)>(float4)((float)0.0000001)?fabs(in1):(float4)((float)0.0000001)))*in1", op, backend);
                    default:
                        break;
                }
                return nullptr;
            }
        }
        return nullptr;
    }
};

REGISTER_OPENCL_OP_CREATOR(LoopBufCreator, OpType_While, BUFFER);

} // namespace OpenCL
} // namespace MNN
#endif /* MNN_OPENCL_BUFFER_CLOSED */
