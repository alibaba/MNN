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
        Unit unit;
        auto input = mTensors[cmd->indexes()->data()[1]];
        auto output = mTensors[cmd->indexes()->data()[0]];
        std::set<std::string> buildOptions;
        if (mIter[0] >= 0) {
            buildOptions.emplace("-DOFFSET_DST");
        }
        if (mIter[1] >= 0) {
            buildOptions.emplace("-DOFFSET_SRC");
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
                auto tensor = mTensors[cmd->iterIndexes()->data()[i]];
                std::vector<int> shape = tensorShapeFormat(tensor);
                int offsetShapeVec[4] = {shape[2], shape[1], shape[3], shape[0]};// WHCN
                ret |= unit.kernel->get().setArg(index++, openCLBuffer(tensor));
                ret |= unit.kernel->get().setArg(index++, sizeof(offsetShapeVec), offsetShapeVec);
            }
        }
        ret |= unit.kernel->get().setArg(index++, x);
        ret |= unit.kernel->get().setArg(index++, sizeof(mStride_src), mStride_src);
        ret |= unit.kernel->get().setArg(index++, sizeof(mStride_dst), mStride_dst);
        ret |= unit.kernel->get().setArg(index++, sizeof(mStep), mStep);
        ret |= unit.kernel->get().setArg(index++, sizeof(mIter), mIter);
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
    if (mHasBias) {
        mOffset[3] = cmd->view()->GetAs<View>(3)->offset();
    }
    
    ::memcpy(mStep, cmd->steps()->data(), cmd->steps()->size() * sizeof(int));
    ::memcpy(mIter, cmd->iterIndexes()->data(), cmd->iterIndexes()->size() * sizeof(int));
    int e = cmd->size()->data()[0];
    int l = cmd->size()->data()[1];
    int h = cmd->size()->data()[2];
    int n = mLoop->loopNumber();
    
    {
       // matmul
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
       ret |= unit.kernel->get().setArg(index++, openCLBuffer(mTensors[cmd->indexes()->data()[0]]));
       ret |= unit.kernel->get().setArg(index++, openCLBuffer(mTensors[cmd->indexes()->data()[1]]));
       ret |= unit.kernel->get().setArg(index++, openCLBuffer(mTensors[cmd->indexes()->data()[2]]));
       if (mHasBias) {
           ret |= unit.kernel->get().setArg(index++, openCLBuffer(mTensors[cmd->indexes()->data()[3]]));
       }
       for (int i = 0; i < cmd->iterIndexes()->size(); ++i) {
           if (mIter[i] >= 0) {
               ret |= unit.kernel->get().setArg(index++, openCLBuffer(mTensors[cmd->iterIndexes()->data()[i]]));
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
    
    auto input0 = mTensors[cmd->indexes()->data()[1]];
    auto input1 = mTensors[cmd->indexes()->data()[2]];
    auto output = mTensors[cmd->indexes()->data()[0]];
    unit.kernel = runTime->buildKernel("loop_buf", "loop_binary_buf", mBuildOptions, input0, output);
    uint32_t mMaxWorkGroupSize = static_cast<uint32_t>(runTime->getMaxWorkGroupSize(unit.kernel));
    
    std::vector<uint32_t> mGlobalWorkSize = {(uint32_t)(x), (uint32_t)(y), (uint32_t)(z)};

    uint32_t index = 0;
    cl_int ret = CL_SUCCESS;
    ret |= unit.kernel->get().setArg(index++, mGlobalWorkSize[0]);
    ret |= unit.kernel->get().setArg(index++, mGlobalWorkSize[1]);
    ret |= unit.kernel->get().setArg(index++, mGlobalWorkSize[2]);
    ret |= unit.kernel->get().setArg(index++, openCLBuffer(output));
    ret |= unit.kernel->get().setArg(index++, openCLBuffer(input0));
    ret |= unit.kernel->get().setArg(index++, openCLBuffer(input1));
    ret |= unit.kernel->get().setArg(index++, mStride_src0[0]);
    ret |= unit.kernel->get().setArg(index++, mStride_src0[1]);
    ret |= unit.kernel->get().setArg(index++, mStride_src0[2]);
    ret |= unit.kernel->get().setArg(index++, mStride_src1[0]);
    ret |= unit.kernel->get().setArg(index++, mStride_src1[1]);
    ret |= unit.kernel->get().setArg(index++, mStride_src1[2]);
    ret |= unit.kernel->get().setArg(index++, mStride_dst[0]);
    ret |= unit.kernel->get().setArg(index++, mStride_dst[1]);
    ret |= unit.kernel->get().setArg(index++, mStride_dst[2]);
    MNN_CHECK_CL_SUCCESS(ret, "setArg LoopBinaryBufExecution");

    std::vector<uint32_t> mLocalWorkSize = localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, runTime, "loop_binary_buf", unit.kernel).first;

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
                        return new LoopBinaryBufExecution(loop, "sign(in1)*in0/(fabs(in1)>(float)((float)0.0000001)?fabs(in1):(float)((float)0.0000001))", op, backend);
                    case BinaryOpOperation_MINIMUM:
                        return new LoopBinaryBufExecution(loop, "in0>in1?in1:in0", op, backend);
                    case BinaryOpOperation_MAXIMUM:
                        return new LoopBinaryBufExecution(loop, "in0>in1?in0:in1", op, backend);
                    case BinaryOpOperation_GREATER:
                        return new LoopBinaryBufExecution(loop, "(float)(isgreater(in0,in1))", op, backend);
                    case BinaryOpOperation_LESS:
                        return new LoopBinaryBufExecution(loop, "(float)(isless(in0,in1))", op, backend);
                    case BinaryOpOperation_LESS_EQUAL:
                        return new LoopBinaryBufExecution(loop, "(float)(islessequal(in0,in1))", op, backend);
                    case BinaryOpOperation_GREATER_EQUAL:
                        return new LoopBinaryBufExecution(loop, "(float)(isgreaterequal(in0,in1))", op, backend);
                    case BinaryOpOperation_EQUAL:
                        return new LoopBinaryBufExecution(loop, "(float)(isequal(in0,in1))", op, backend);
                    case BinaryOpOperation_FLOORDIV:
                        return new LoopBinaryBufExecution(loop, "floor(sign(in1)*in0/(fabs(in1)>(float)((float)0.0000001)?fabs(in1):(float)((float)0.0000001)))", op, backend);
                    case BinaryOpOperation_FLOORMOD:
                        return new LoopBinaryBufExecution(loop, "in0-floor(sign(in1)*in0/(fabs(in1)>(float)((float)0.0000001)?fabs(in1):(float)((float)0.0000001)))*in1", op, backend);
                    case BinaryOpOperation_POW:
                        return new LoopBinaryBufExecution(loop, "pow(in0,in1)", op, backend);
                    case BinaryOpOperation_SquaredDifference:
                        return new LoopBinaryBufExecution(loop, "(in0-in1)*(in0-in1)", op, backend);
                    case BinaryOpOperation_ATAN2:
                        return new LoopBinaryBufExecution(loop, "(in1==(float)0?(sign(in0)*(float)(PI/2)):(atan(in0/in1)+(in1>(float)0?(float)0:sign(in0)*(float)PI)))", op, backend);
                    case BinaryOpOperation_NOTEQUAL:
                        return new LoopBinaryBufExecution(loop, "(float)(isnotequal(in0,in1))", op, backend);
                    case BinaryOpOperation_MOD:
                        return new LoopBinaryBufExecution(loop, "in0-floor(sign(in1)*in0/(fabs(in1)>(float)((float)0.0000001)?fabs(in1):(float)((float)0.0000001)))*in1", op, backend);
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
