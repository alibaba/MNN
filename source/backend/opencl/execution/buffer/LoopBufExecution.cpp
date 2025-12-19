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

static std::string getComputeOption(MNN::BinaryOpOperation type){
    std::string compute;
    switch (type) {
        case BinaryOpOperation_MUL:
            compute = "in0*in1";break;
        case BinaryOpOperation_ADD:
            compute = "in0+in1";break;
        case BinaryOpOperation_SUB:
            compute = "in0-in1";break;
        case BinaryOpOperation_REALDIV:
            compute = "sign(in1)*in0/(fabs(in1)>(float)((float)0.0000001)?fabs(in1):(float)((float)0.0000001))";break;
        case BinaryOpOperation_MINIMUM:
            compute = "in0>in1?in1:in0";break;
        case BinaryOpOperation_MAXIMUM:
            compute = "in0>in1?in0:in1";break;
        case BinaryOpOperation_GREATER:
            compute = "(float)(isgreater(in0,in1))";break;
        case BinaryOpOperation_LESS:
            compute = "(float)(isless(in0,in1))";break;
        case BinaryOpOperation_LESS_EQUAL:
            compute = "(float)(islessequal(in0,in1))";break;
        case BinaryOpOperation_GREATER_EQUAL:
            compute = "(float)(isgreaterequal(in0,in1))";break;
        case BinaryOpOperation_EQUAL:
            compute = "(float)(isequal(in0,in1))";break;
        case BinaryOpOperation_FLOORDIV:
            compute = "floor(sign(in1)*in0/(fabs(in1)>(float)((float)0.0000001)?fabs(in1):(float)((float)0.0000001)))";break;
        case BinaryOpOperation_FLOORMOD:
            compute = "in0-floor(sign(in1)*in0/(fabs(in1)>(float)((float)0.0000001)?fabs(in1):(float)((float)0.0000001)))*in1";break;
        case BinaryOpOperation_POW:
            compute = "pow(in0,in1)";break;
        case BinaryOpOperation_SquaredDifference:
            compute = "(in0-in1)*(in0-in1)";break;
        case BinaryOpOperation_ATAN2:
            compute = "(in1==(float)0?(sign(in0)*(float)(PI/2)):(atan(in0/in1)+(in1>(float)0?(float)0:sign(in0)*(float)PI)))";break;
        case BinaryOpOperation_NOTEQUAL:
            compute = "(float)(isnotequal(in0,in1))";break;
        case BinaryOpOperation_MOD:
            compute = "in0-floor(sign(in1)*in0/(fabs(in1)>(float)((float)0.0000001)?fabs(in1):(float)((float)0.0000001)))*in1";break;
        default:
            break;
    }
    return compute;
}

static std::string getUnaryComputeOption(MNN::UnaryOpOperation type){
    std::string compute;
    switch (type) {
        case UnaryOpOperation_ABS:
            compute = "fabs((float)(in))"; break;
        case UnaryOpOperation_SQUARE:
            compute = "in*in"; break;
        case UnaryOpOperation_RSQRT:
            compute = "rsqrt((float))(in)>(float)(0.000001)?(float))(in):(float)(0.000001))"; break;
        case UnaryOpOperation_NEG:
            compute = "-(in)"; break;
        case UnaryOpOperation_EXP:
            compute = "exp((float))(in))"; break;
        case UnaryOpOperation_COS:
            compute = "cos((float)(in))"; break;
        case UnaryOpOperation_SIN:
            compute = "sin((float)(in))"; break;
        case UnaryOpOperation_TAN:
            compute = "tan((float)(in))"; break;
        case UnaryOpOperation_ATAN:
            compute = "atan((float)(in))"; break;
        case UnaryOpOperation_SQRT:
            compute = "sqrt((float)(in))"; break;
        case UnaryOpOperation_CEIL:
            compute = "ceil((float)(in))"; break;
        case UnaryOpOperation_RECIPROCAL:
            compute = "native_recip((float)(in))"; break;
        case UnaryOpOperation_LOG1P:
            compute = "log1p((float)(in))"; break;
        case UnaryOpOperation_LOG:
            compute = "native_log((float)(in)>(float)(0.0000001)?(float)(in):(float)(0.0000001))"; break;
        case UnaryOpOperation_FLOOR:
            compute = "floor((float)(in))"; break;
        case UnaryOpOperation_BNLL:
            compute = "in>(float)((float)0)?(in+native_log(exp((float)(-(in)))+(float)(1.0))):(native_log(exp((float)(in))+(float)(1.0)))"; break;
        case UnaryOpOperation_ACOSH:
            compute = "acosh((float)(in))"; break;
        case UnaryOpOperation_SINH:
            compute = "sinh((float)(in))"; break;
        case UnaryOpOperation_ASINH:
            compute = "asinh((float)(in))"; break;
        case UnaryOpOperation_ATANH:
            compute = "atanh((float)(in))"; break;
        case UnaryOpOperation_SIGN:
            compute = "sign((float)(in))"; break;
        case UnaryOpOperation_ROUND:
            compute = "round((float)(in))"; break;
        case UnaryOpOperation_COSH:
            compute = "cosh((float)(in))"; break;
        case UnaryOpOperation_ERF:
            compute = "erf((float)(in))"; break;
        case UnaryOpOperation_ERFC:
            compute = "erfc((float)(in))"; break;
        case UnaryOpOperation_EXPM1:
            compute = "expm1((float)(in))"; break;
        case UnaryOpOperation_SIGMOID:
            compute = "native_recip((float)1+native_exp((float)(-in)))"; break;
        case UnaryOpOperation_SILU:
            compute = "((float)(in)*native_recip((float)1+native_exp((float)(-in))))"; break;
        case UnaryOpOperation_TANH:
            compute = "tanh((float)(in))"; break;
        case UnaryOpOperation_HARDSWISH:
            compute = "(float)(in)>(float)(-3.0f)?((float)(in)<(float)(3.0f)?(((float)(in)*((float)(in)+(float)3.0f))/(float)6.0f):(float)(in)):(float)(0.0f)"; break;
        case UnaryOpOperation_GELU:
            compute = "gelu((float)(in))"; break;
        case UnaryOpOperation_GELU_STANDARD:
            compute = "(erf((float)(in)*(float)0.7071067932881648)+(float)1.0)*(float)(in)*(float)0.5"; break;
        default:
            break;
    }
    return compute;
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

LoopBufExecution::LoopBufExecution(const LoopParam *loop, const MNN::Op *op, Backend *bn)
: CommonExecution(bn, op) {
    mLoop = loop;
    mTensors.resize(mLoop->tensorNumber());
}

ErrorCode LoopBufExecution::InitCommandOnEncode(){
    for (int i=0; i<mLoop->initCommand()->size(); ++i) {
        auto cmd                      = mLoop->initCommand()->GetAs<RegionCommand>(i);
        OpenCLBackend *mOpenCLBackend = (OpenCLBackend *)backend();
        auto runTime                  = mOpenCLBackend->getOpenCLRuntime();
        int mStride_src[4];
        int mStride_dst[4];
        int mStep[2];
        int mIter[2];
        if (cmd->op() == nullptr){
            Unit unit;
            auto output = mTensors[cmd->indexes()->data()[0]];
            auto outputShape = tensorShapeFormat(output);
            auto outputDes = TensorUtils::getDescribe(output);
            int region[] = {outputShape[0], outputShape[3], outputShape[1], outputShape[2]};//nchw
            if(MNN_DATA_FORMAT_NC4HW4 == outputDes->dimensionFormat){
                region[1] = ROUND_UP(outputShape[3], 4);
            }
            unit.kernel         = runTime->buildKernel("loop", "set_zero", {}, mOpenCLBackend->getPrecision(), output, output);
            unit.localWorkSize  = {8, 8};
            unit.globalWorkSize = {(uint32_t)UP_DIV((region[2] * region[3]), 8)*8,
                (uint32_t)UP_DIV((region[0] * region[1]), 8)*8};
            
            int global_dim0 = region[2] * region[3];
            int global_dim1 = region[0] * region[1];
            
            uint32_t idx   = 0;
            cl_int ret = CL_SUCCESS;
            ret |= unit.kernel->get().setArg(idx++, global_dim0);
            ret |= unit.kernel->get().setArg(idx++, global_dim1);
            ret |= unit.kernel->get().setArg(idx++, openCLBuffer(output));
            MNN_CHECK_CL_SUCCESS(ret, "setArg set_zero buffer");
            mOpenCLBackend->recordKernel2d(unit.kernel, {(uint32_t)UP_DIV((region[2] * region[3]), 8)*8,
                (uint32_t)UP_DIV((region[0] * region[1]), 8)*8},  {8, 8});
            mUnits.emplace_back(unit);
            return NO_ERROR;
        }
        int x = cmd->size()->data()[0];
        int y = cmd->size()->data()[1];
        int z = cmd->size()->data()[2];
        
        int inputSize = mTensors[cmd->indexes()->data()[1]]->elementSize();
        int outputSize = mTensors[cmd->indexes()->data()[0]]->elementSize();
        
        auto srcStride = cmd->view()->GetAs<View>(1)->stride()->data();
        auto dstStride = cmd->view()->GetAs<View>(0)->stride()->data();
        for (int i = 0; i < 3; ++i) {
            mStride_src[i] = srcStride[i];
            mStride_dst[i] = dstStride[i];
        }
        
        mStride_src[3] = 0;
        mStride_dst[3] = 0;
        ::memset(mStep, 0, 2 * sizeof(int));
        
        // gather
        {
            Unit unit;
            auto input = mTensors[cmd->indexes()->data()[1]];
            auto output = mTensors[cmd->indexes()->data()[0]];
            std::set<std::string> buildOptions;
            
            unit.kernel = runTime->buildKernel("loop", "batch_gather", buildOptions, mOpenCLBackend->getPrecision(), input, output);
            uint32_t mMaxWorkGroupSize = static_cast<uint32_t>(runTime->getMaxWorkGroupSize(unit.kernel));
            std::vector<uint32_t> mGlobalWorkSize = {(uint32_t)(x * y), (uint32_t)(z), (uint32_t)(1)};
            
            uint32_t index = 0;
            cl_int ret = CL_SUCCESS;
            ret |= unit.kernel->get().setArg(index++, mGlobalWorkSize[0]);
            ret |= unit.kernel->get().setArg(index++, mGlobalWorkSize[1]);
            ret |= unit.kernel->get().setArg(index++, mGlobalWorkSize[2]);
            ret |= unit.kernel->get().setArg(index++, openCLBuffer(output));
            ret |= unit.kernel->get().setArg(index++, openCLBuffer(input));
            ret |= unit.kernel->get().setArg(index++, x);
            ret |= unit.kernel->get().setArg(index++, 0);
            ret |= unit.kernel->get().setArg(index++, sizeof(mStride_src), mStride_src);
            ret |= unit.kernel->get().setArg(index++, sizeof(mStride_dst), mStride_dst);
            ret |= unit.kernel->get().setArg(index++, sizeof(mStep), mStep);
            ret |= unit.kernel->get().setArg(index++, inputSize);
            ret |= unit.kernel->get().setArg(index++, outputSize);
            MNN_CHECK_CL_SUCCESS(ret, "setArg LoopInitGatherBufExecution");
            
            std::vector<uint32_t> mLocalWorkSize = localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, runTime, "batch_gather", unit.kernel, mOpenCLBackend->getCLTuneLevel(), "loop").first;
            
            unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1], mGlobalWorkSize[2]};
            unit.localWorkSize  = {mLocalWorkSize[0], mLocalWorkSize[1], mLocalWorkSize[2]};
            mUnits.emplace_back(unit);
            mOpenCLBackend->recordKernel3d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
        }
    }
    return NO_ERROR;
}
ErrorCode LoopBufExecution::LoopGather(const Tensor *output, int cmdIndex, int iter) {
    auto cmd = mLoop->commands()->GetAs<RegionCommand>(cmdIndex);
    auto op = cmd->op();
    OpenCLBackend *mOpenCLBackend = (OpenCLBackend *)backend();
    auto runTime                  = mOpenCLBackend->getOpenCLRuntime();
    
    int x = cmd->size()->data()[0];
    int y = cmd->size()->data()[1];
    int z = cmd->size()->data()[2];
    int n = mLoop->parallel() ? mLoop->loopNumber() : 1;
    if(mLoop->commands()->size() == 1 && OpType_UnaryOp == op->type() && nullptr == op->main() && cmd->fuse() < 0){
        // only one gather
        n = mLoop->loopNumber();
    }
    
    int mStride_src[4];
    int mStride_dst[4];
    int mStep[2];
    int mIter[2];
    int inputSize = mTensors[cmd->indexes()->data()[1]]->elementSize();
    int outputSize = output->elementSize();
    
    auto srcStride = cmd->view()->GetAs<View>(1)->stride()->data();
    auto dstStride = cmd->view()->GetAs<View>(0)->stride()->data();
    for (int i = 0; i < 3; ++i) {
        mStride_src[i] = srcStride[i];
        mStride_dst[i] = dstStride[i];
    }
    if(cmd->fuse() >= 0){
        mStride_dst[0] = y * z;
        mStride_dst[1] = z;
        mStride_dst[2] = 1;
    }
    
    mStride_src[3] = cmd->view()->GetAs<View>(1)->offset();
    mStride_dst[3] = cmd->view()->GetAs<View>(0)->offset();
    ::memcpy(mStep, cmd->steps()->data(), cmd->steps()->size() * sizeof(int));
    ::memcpy(mIter, cmd->iterIndexes()->data(), cmd->iterIndexes()->size() * sizeof(int));
    
    // gather
    Unit unit;
    auto input = mTensors[cmd->indexes()->data()[1]];
    std::set<std::string> buildOptions;
    
    if(op->main() != nullptr){
        std::string compute = getUnaryComputeOption(cmd->op()->main_as_UnaryOp()->opType());
        buildOptions.emplace("-DUNARY_OPERATOR=" + compute);
    }
    if (mIter[0] >= 0) {
        buildOptions.emplace("-DOFFSET_DST");
    }
    if (mIter[1] >= 0) {
        buildOptions.emplace("-DOFFSET_SRC");
    }
    
    unit.kernel = runTime->buildKernel("loop", "batch_gather", buildOptions, mOpenCLBackend->getPrecision(), input, output);
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
        }
    }
    ret |= unit.kernel->get().setArg(index++, x);
    ret |= unit.kernel->get().setArg(index++, iter);
    ret |= unit.kernel->get().setArg(index++, sizeof(mStride_src), mStride_src);
    ret |= unit.kernel->get().setArg(index++, sizeof(mStride_dst), mStride_dst);
    ret |= unit.kernel->get().setArg(index++, sizeof(mStep), mStep);
    ret |= unit.kernel->get().setArg(index++, inputSize);
    ret |= unit.kernel->get().setArg(index++, outputSize);
    MNN_CHECK_CL_SUCCESS(ret, "setArg LoopGatherBufExecution");
    
    std::vector<uint32_t> mLocalWorkSize = localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, runTime, "batch_gather", unit.kernel, mOpenCLBackend->getCLTuneLevel(), "loop").first;
    
    unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1], mGlobalWorkSize[2]};
    unit.localWorkSize  = {mLocalWorkSize[0], mLocalWorkSize[1], mLocalWorkSize[2]};
    mUnits.emplace_back(unit);
    mOpenCLBackend->recordKernel3d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
    
    if(cmd->fuse() >= 0){
        FuseOutput(cmdIndex, mStride_dst, cmd->size()->data()[0], cmd->size()->data()[1], cmd->size()->data()[2], n, iter);
    }
    return NO_ERROR;
}


ErrorCode LoopBufExecution::LoopBatchMatMul(const Tensor *output, int cmdIndex, int iter) {
    auto cmd     = mLoop->commands()->GetAs<RegionCommand>(cmdIndex);
    bool mHasBias = cmd->indexes()->size() > 3;
    OpenCLBackend *mOpenCLBackend = (OpenCLBackend *)backend();
    auto runTime = mOpenCLBackend->getOpenCLRuntime();
    
    int mOffset[4];
    int mStep[4];
    int mIter[4];
    
    mOffset[0] = cmd->view()->GetAs<View>(0)->offset();
    mOffset[1] = cmd->view()->GetAs<View>(1)->offset();
    mOffset[2] = cmd->view()->GetAs<View>(2)->offset();
    if (mHasBias) {
        mOffset[3] = cmd->view()->GetAs<View>(3)->offset();
    }
    
    ::memcpy(mStep, cmd->steps()->data(), cmd->steps()->size() * sizeof(int));
    ::memcpy(mIter, cmd->iterIndexes()->data(), cmd->iterIndexes()->size() * sizeof(int));
    int e = cmd->size()->data()[0];
    int l = cmd->size()->data()[1];
    int h = cmd->size()->data()[2];
    int n = mLoop->parallel() ? mLoop->loopNumber() : 1;
    // matmul
    Unit unit;
    std::string KernelName = "batch_matmul";
    std::set<std::string> buildOptions;
    if (mHasBias) {
        buildOptions.emplace("-DBIAS");
    }
    if (cmd->op()->main_as_MatMul()->transposeA()) {
        buildOptions.emplace("-DTRANSPOSE_A");
    }
    if (cmd->op()->main_as_MatMul()->transposeB()) {
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
    ret |= unit.kernel->get().setArg(index++, openCLBuffer(output));
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
    ret |= unit.kernel->get().setArg(index++, iter);
    ret |= unit.kernel->get().setArg(index++, sizeof(mOffset), mOffset);
    ret |= unit.kernel->get().setArg(index++, sizeof(mIter), mIter);
    ret |= unit.kernel->get().setArg(index++, sizeof(mStep), mStep);
    MNN_CHECK_CL_SUCCESS(ret, "setArg LoopBatchMatMulBufExecution");
    
    std::vector<uint32_t> mLocalWorkSize = localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, runTime, KernelName, unit.kernel, mOpenCLBackend->getCLTuneLevel(), "loop").first;
    
    unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1], mGlobalWorkSize[2]};
    unit.localWorkSize  = {mLocalWorkSize[0], mLocalWorkSize[1], mLocalWorkSize[2]};
    mUnits.emplace_back(unit);
    mOpenCLBackend->recordKernel3d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
    
    if(cmd->fuse() >= 0){
        int mStride_dst[4];
        mStride_dst[0] = h * e;
        mStride_dst[1] = h;
        mStride_dst[2] = 1;
        mStride_dst[3] = 1;
        FuseOutput(cmdIndex, mStride_dst, 1, e, h, n, iter);
    }

    return NO_ERROR;
}

ErrorCode LoopBufExecution::LoopBinary(const Tensor *output, int cmdIndex, int iter) {
    auto cmd = mLoop->commands()->GetAs<RegionCommand>(cmdIndex);
    std::string compute = getComputeOption(cmd->op()->main_as_BinaryOp()->opType());
    std::set<std::string> buildOptions;
    buildOptions.emplace("-DOPERATOR=" + compute);
    if(cmd->op()->main_as_BinaryOp()->opType() == BinaryOpOperation_MOD && (output->getType().code == halide_type_int || output->getType().code == halide_type_uint)){
        buildOptions.emplace("-DINT_COMPUTE_MOD");
    }
    OpenCLBackend *mOpenCLBackend = (OpenCLBackend *)backend();
    auto runTime                  = mOpenCLBackend->getOpenCLRuntime();
    
    int mOffset[4];
    int mStep[4];
    int mIter[4];
    int mStride_src0[3];
    int mStride_src1[3];
    int mStride_dst[3];
    
    Unit unit;
    int z = cmd->size()->data()[0];
    int y = cmd->size()->data()[1];
    int x = cmd->size()->data()[2];
    int n = mLoop->parallel() ? mLoop->loopNumber() : 1;
    int inputSize = mTensors[cmd->indexes()->data()[1]]->elementSize();
    int outputSize = output->elementSize();
    
    auto src0Stride = cmd->view()->GetAs<View>(1)->stride()->data();
    auto src1Stride = cmd->view()->GetAs<View>(2)->stride()->data();
    auto dstStride = cmd->view()->GetAs<View>(0)->stride()->data();
    for (int i = 0; i < 3; ++i) {
        mStride_src0[i] = src0Stride[i];
        mStride_src1[i] = src1Stride[i];
        mStride_dst[i] = dstStride[i];
    }
    if(cmd->fuse() >= 0){
        mStride_dst[0] = y * x;
        mStride_dst[1] = x;
        mStride_dst[2] = 1;
    }
    
    auto input0 = mTensors[cmd->indexes()->data()[1]];
    auto input1 = mTensors[cmd->indexes()->data()[2]];
    
    ::memcpy(mStep, cmd->steps()->data(), cmd->steps()->size() * sizeof(int));
    ::memcpy(mIter, cmd->iterIndexes()->data(), cmd->iterIndexes()->size() * sizeof(int));
    mOffset[0] = cmd->view()->GetAs<View>(0)->offset();
    mOffset[1] = cmd->view()->GetAs<View>(1)->offset();
    mOffset[2] = cmd->view()->GetAs<View>(2)->offset();
    
    if (mIter[0] >= 0) {
        buildOptions.emplace("-DOFFSET_DST");
    }
    if (mIter[1] >= 0) {
        buildOptions.emplace("-DOFFSET_SRC0");
    }
    if (mIter[2] >= 0) {
        buildOptions.emplace("-DOFFSET_SRC1");
    }
    unit.kernel = runTime->buildKernel("loop", "loop_binary", buildOptions, mOpenCLBackend->getPrecision(), input0, output);
    uint32_t mMaxWorkGroupSize = static_cast<uint32_t>(runTime->getMaxWorkGroupSize(unit.kernel));
    
    std::vector<uint32_t> mGlobalWorkSize = {(uint32_t)(x), (uint32_t)(y), (uint32_t)(z*n)};
    uint32_t index = 0;
    cl_int ret = CL_SUCCESS;
    ret |= unit.kernel->get().setArg(index++, mGlobalWorkSize[0]);
    ret |= unit.kernel->get().setArg(index++, mGlobalWorkSize[1]);
    ret |= unit.kernel->get().setArg(index++, mGlobalWorkSize[2]);
    ret |= unit.kernel->get().setArg(index++, openCLBuffer(output));
    ret |= unit.kernel->get().setArg(index++, openCLBuffer(input0));
    ret |= unit.kernel->get().setArg(index++, openCLBuffer(input1));
    for (int i = 0; i < cmd->iterIndexes()->size(); ++i) {
        if (mIter[i] >= 0) {
            ret |= unit.kernel->get().setArg(index++, openCLBuffer(mTensors[cmd->iterIndexes()->data()[i]]));
        }
    }
    ret |= unit.kernel->get().setArg(index++, mStride_src0[0]);
    ret |= unit.kernel->get().setArg(index++, mStride_src0[1]);
    ret |= unit.kernel->get().setArg(index++, mStride_src0[2]);
    ret |= unit.kernel->get().setArg(index++, mStride_src1[0]);
    ret |= unit.kernel->get().setArg(index++, mStride_src1[1]);
    ret |= unit.kernel->get().setArg(index++, mStride_src1[2]);
    ret |= unit.kernel->get().setArg(index++, mStride_dst[0]);
    ret |= unit.kernel->get().setArg(index++, mStride_dst[1]);
    ret |= unit.kernel->get().setArg(index++, mStride_dst[2]);
    ret |= unit.kernel->get().setArg(index++, iter);
    ret |= unit.kernel->get().setArg(index++, z);
    ret |= unit.kernel->get().setArg(index++, sizeof(mOffset), mOffset);
    ret |= unit.kernel->get().setArg(index++, sizeof(mStep), mStep);
    ret |= unit.kernel->get().setArg(index++, outputSize);
    MNN_CHECK_CL_SUCCESS(ret, "setArg LoopBinaryBufExecution");

    std::vector<uint32_t> mLocalWorkSize = localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, runTime, "loop_binary", unit.kernel, mOpenCLBackend->getCLTuneLevel(), "loop").first;

    unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1], mGlobalWorkSize[2]};
    unit.localWorkSize  = {mLocalWorkSize[0], mLocalWorkSize[1], mLocalWorkSize[2]};
    mUnits.emplace_back(unit);
    mOpenCLBackend->recordKernel3d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
    
    if(cmd->fuse() >= 0){
        FuseOutput(cmdIndex, mStride_dst, cmd->size()->data()[0], cmd->size()->data()[1], cmd->size()->data()[2], n, iter);
    }
    return NO_ERROR;
}

ErrorCode LoopBufExecution::LoopCumsum(const Tensor *output) {
    auto cmd = mLoop->commands()->GetAs<RegionCommand>(0);
    std::string compute = getComputeOption(cmd->op()->main_as_BinaryOp()->opType());
    std::set<std::string> buildOptions;
    buildOptions.emplace("-DOPERATOR=" + compute);
    if(cmd->op()->main_as_BinaryOp()->opType() == BinaryOpOperation_MOD && (output->getType().code == halide_type_int || output->getType().code == halide_type_uint)){
        buildOptions.emplace("-DINT_COMPUTE_MOD");
    }
    OpenCLBackend *mOpenCLBackend = (OpenCLBackend *)backend();
    auto runTime                  = mOpenCLBackend->getOpenCLRuntime();
    
    int mOffset[4];
    int mStep[4];
    int mIter[4];
    int mStride_src0[3];
    int mStride_src1[3];
    int mStride_dst[3];
    
    Unit unit;
    int z = cmd->size()->data()[0];
    int y = cmd->size()->data()[1];
    int x = cmd->size()->data()[2];
    int n = mLoop->parallel() ? mLoop->loopNumber() : 1;
    int inputSize = mTensors[cmd->indexes()->data()[1]]->elementSize();
    int outputSize = output->elementSize();
    
    auto src0Stride = cmd->view()->GetAs<View>(1)->stride()->data();
    auto src1Stride = cmd->view()->GetAs<View>(2)->stride()->data();
    auto dstStride = cmd->view()->GetAs<View>(0)->stride()->data();
    for (int i = 0; i < 3; ++i) {
        mStride_src0[i] = src0Stride[i];
        mStride_src1[i] = src1Stride[i];
        mStride_dst[i] = dstStride[i];
    }
    if(cmd->fuse() >= 0){
        mStride_dst[0] = y * x;
        mStride_dst[1] = x;
        mStride_dst[2] = 1;
    }
    
    auto input0 = mTensors[cmd->indexes()->data()[1]];
    auto input1 = mTensors[cmd->indexes()->data()[2]];
    
    // cumsum
    // mTensors cmd->indexes()->data() = {2, 0, 1} -> {output, input0, input1}, output = input0
    int loopNumber = mLoop->loopNumber();
    
    ::memcpy(mStep, cmd->steps()->data(), cmd->steps()->size() * sizeof(int));
    mOffset[0] = cmd->view()->GetAs<View>(0)->offset();
    mOffset[1] = cmd->view()->GetAs<View>(1)->offset();
    mOffset[2] = cmd->view()->GetAs<View>(2)->offset();
    unit.kernel = runTime->buildKernel("loop", "loop_cumsum", buildOptions, mOpenCLBackend->getPrecision(), input0, output);
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
    ret |= unit.kernel->get().setArg(index++, loopNumber);
    ret |= unit.kernel->get().setArg(index++, sizeof(mOffset), mOffset);
    ret |= unit.kernel->get().setArg(index++, sizeof(mStep), mStep);
    ret |= unit.kernel->get().setArg(index++, outputSize);
    MNN_CHECK_CL_SUCCESS(ret, "setArg LoopCumsumBufExecution");
    
    std::vector<uint32_t> mLocalWorkSize = localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, runTime, "loop_cumsum", unit.kernel, mOpenCLBackend->getCLTuneLevel(), "loop").first;
    
    
    unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1], mGlobalWorkSize[2]};
    unit.localWorkSize  = {mLocalWorkSize[0], mLocalWorkSize[1], mLocalWorkSize[2]};
    mUnits.emplace_back(unit);
    mOpenCLBackend->recordKernel3d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
    
    return NO_ERROR;
}

ErrorCode LoopBufExecution::FuseOutput(int iter, int* inputStride, int sizeZ, int sizeY, int SizeX, int n, int n_offset) {
    auto cmd = mLoop->commands()->GetAs<RegionCommand>(iter);
    std::string compute = getComputeOption(MNN::BinaryOpOperation(cmd->fuse()));
    std::set<std::string> buildOptions;
    buildOptions.emplace("-DOPERATOR=" + compute);
    OpenCLBackend *mOpenCLBackend = (OpenCLBackend *)backend();
    auto runTime = mOpenCLBackend->getOpenCLRuntime();
    
    int mOffset[4];
    int mStep[4];
    int mIter[4];
    int mStride_src0[3];
    int mStride_src1[3];
    int mStride_dst[3];
    auto input = mFuseTensor.get();
    auto output = mTensors[cmd->indexes()->data()[0]];
    int outputSize = output->elementSize();
    
    Unit unit;
    int z = sizeZ;
    int y = sizeY;
    int x = SizeX;
    
    auto dstStride = cmd->view()->GetAs<View>(0)->stride()->data();
    for (int i = 0; i < 3; ++i) {
        mStride_src0[i] = dstStride[i];
        mStride_src1[i] = inputStride[i];
        mStride_dst[i] = dstStride[i];
    }
    
    for(int i = 0; i < 4; ++i){
        mStep[i] = cmd->steps()->data()[0];
    }
    ::memcpy(mIter, cmd->iterIndexes()->data(), cmd->iterIndexes()->size() * sizeof(int));
    mOffset[0] = cmd->view()->GetAs<View>(0)->offset();
    mOffset[1] = cmd->view()->GetAs<View>(0)->offset();
    mOffset[2] = cmd->view()->GetAs<View>(0)->offset();
    
    if (mIter[0] >= 0) {
        buildOptions.emplace("-DOFFSET_DST");
    }
    if (mIter[0] >= 0) {
        buildOptions.emplace("-DOFFSET_SRC0");
    }
    if (mIter[0] >= 0) {
        buildOptions.emplace("-DOFFSET_SRC1");
    }
    unit.kernel = runTime->buildKernel("loop", "loop_binary", buildOptions, mOpenCLBackend->getPrecision(), input, output);
    uint32_t mMaxWorkGroupSize = static_cast<uint32_t>(runTime->getMaxWorkGroupSize(unit.kernel));
    
    std::vector<uint32_t> mGlobalWorkSize = {(uint32_t)(x), (uint32_t)(y), (uint32_t)(z*n)};

    uint32_t index = 0;
    cl_int ret = CL_SUCCESS;
    ret |= unit.kernel->get().setArg(index++, mGlobalWorkSize[0]);
    ret |= unit.kernel->get().setArg(index++, mGlobalWorkSize[1]);
    ret |= unit.kernel->get().setArg(index++, mGlobalWorkSize[2]);
    ret |= unit.kernel->get().setArg(index++, openCLBuffer(output));
    ret |= unit.kernel->get().setArg(index++, openCLBuffer(output));
    ret |= unit.kernel->get().setArg(index++, openCLBuffer(input));
    if (mIter[0] >= 0) {
        ret |= unit.kernel->get().setArg(index++, openCLBuffer(mTensors[cmd->iterIndexes()->data()[0]]));
        ret |= unit.kernel->get().setArg(index++, openCLBuffer(mTensors[cmd->iterIndexes()->data()[0]]));
        ret |= unit.kernel->get().setArg(index++, openCLBuffer(mTensors[cmd->iterIndexes()->data()[0]]));
    }
    ret |= unit.kernel->get().setArg(index++, mStride_src0[0]);
    ret |= unit.kernel->get().setArg(index++, mStride_src0[1]);
    ret |= unit.kernel->get().setArg(index++, mStride_src0[2]);
    ret |= unit.kernel->get().setArg(index++, mStride_src1[0]);
    ret |= unit.kernel->get().setArg(index++, mStride_src1[1]);
    ret |= unit.kernel->get().setArg(index++, mStride_src1[2]);
    ret |= unit.kernel->get().setArg(index++, mStride_dst[0]);
    ret |= unit.kernel->get().setArg(index++, mStride_dst[1]);
    ret |= unit.kernel->get().setArg(index++, mStride_dst[2]);
    ret |= unit.kernel->get().setArg(index++, n_offset);
    ret |= unit.kernel->get().setArg(index++, z);
    ret |= unit.kernel->get().setArg(index++, sizeof(mOffset), mOffset);
    ret |= unit.kernel->get().setArg(index++, sizeof(mStep), mStep);
    ret |= unit.kernel->get().setArg(index++, outputSize);
    MNN_CHECK_CL_SUCCESS(ret, "setArg LoopBinaryBufExecution");

    std::vector<uint32_t> mLocalWorkSize = localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, runTime, "loop_binary", unit.kernel, mOpenCLBackend->getCLTuneLevel(), "loop").first;

    unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1], mGlobalWorkSize[2]};
    unit.localWorkSize  = {mLocalWorkSize[0], mLocalWorkSize[1], mLocalWorkSize[2]};
    mUnits.emplace_back(unit);
    mOpenCLBackend->recordKernel3d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
    return NO_ERROR;
}

ErrorCode LoopBufExecution::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs){
    OpenCLBackend *mOpenCLBackend = (OpenCLBackend *)backend();
    auto runTime                  = mOpenCLBackend->getOpenCLRuntime();
    _setTensorStack(mTensors, inputs, outputs, mLoop);
    // Make Temp output buffer
    int bufferUnitSize = mOpenCLBackend->getPrecision() != BackendConfig::Precision_High ? sizeof(half_float::half) : sizeof(float);
    int mMaxFuseBufferSize = 0;
    int loopNumber = mLoop->parallel() ? 1 : mLoop->loopNumber();
    for (int i=0; i<mLoop->commands()->size(); ++i) {
        auto cmd = mLoop->commands()->GetAs<RegionCommand>(i);
        auto op = cmd->op();
        if (cmd->fuse() >= 0) {
            // Make Temp output buffer
            auto size = cmd->size()->data();
            if (cmd->op()->type() == OpType_MatMul) {
                mMaxFuseBufferSize = std::max(mMaxFuseBufferSize, bufferUnitSize * size[0] * size[2]);
            } else {
                mMaxFuseBufferSize = std::max(mMaxFuseBufferSize, bufferUnitSize * size[0] * size[1] * size[2]);
            }
        }
    }
    if(mMaxFuseBufferSize != 0){
        mFuseTensor.reset(Tensor::createDevice<float>({loopNumber * mMaxFuseBufferSize}));
        mOpenCLBackend->onAcquireBuffer(mFuseTensor.get(), Backend::DYNAMIC);
        mOpenCLBackend->onReleaseBuffer(mFuseTensor.get(), Backend::DYNAMIC);
    }
    mUnits.clear();
    if(mLoop->initCommand() != nullptr){
        InitCommandOnEncode();
    }
    if (1 == mLoop->commands()->size()) {
        auto cmd = mLoop->commands()->GetAs<RegionCommand>(0);
        auto op = cmd->op();
        if (OpType_UnaryOp == op->type() && nullptr == op->main() && cmd->fuse() < 0) {
            return LoopGather(mTensors[cmd->indexes()->data()[0]], 0, 0);
        }
        if(OpType_BinaryOp == op->type() && mLoop->parallel() == false && cmd->fuse() < 0){
            return LoopCumsum(mTensors[cmd->indexes()->data()[0]]);
        }
    }
    for(int iter = 0; iter < loopNumber; ++iter){
        for (int index = 0; index<mLoop->commands()->size(); ++index) {
            auto cmd = mLoop->commands()->GetAs<RegionCommand>(index);
            auto op = cmd->op();
            Tensor *originOutput = mTensors[cmd->indexes()->data()[0]];
            Tensor *output = originOutput;
            if(cmd->fuse() >= 0){
                output = mFuseTensor.get();
            }
            if (OpType_UnaryOp == op->type()){
                LoopGather(output, index, iter);
            }else if (OpType_MatMul == op->type()){
                LoopBatchMatMul(output, index, iter);
            }else if(OpType_BinaryOp == op->type()){
                LoopBinary(output, index, iter);
            }
        }
    }
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
        return new LoopBufExecution(loop, op, backend);
    }
};

REGISTER_OPENCL_OP_CREATOR(LoopBufCreator, OpType_While, BUFFER);

} // namespace OpenCL
} // namespace MNN
#endif /* MNN_OPENCL_BUFFER_CLOSED */
