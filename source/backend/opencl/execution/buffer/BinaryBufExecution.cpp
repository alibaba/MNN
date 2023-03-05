//
//  BinaryBufExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED

#include "backend/opencl/execution/buffer/BinaryBufExecution.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {
namespace OpenCL {

BinaryBufExecution::BinaryBufExecution(const std::vector<Tensor *> &inputs, const std::string &compute, const MNN::Op *op, Backend *backend)
    : CommonExecution(backend), mCompute(compute) {
    mBuildOptions.emplace("-DOPERATOR=" + compute);
    mOp = op;
}

uint32_t BinaryBufExecution::realSize(const Tensor* tensor) {
    uint32_t num = 1;
    for(int i = 0; i < tensor->dimensions(); i++) {
        num *= tensor->length(i);
    }
    return num;
}


ErrorCode BinaryBufExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(inputs.size() >= 2);
    mUnits.resize(inputs.size() - 1);
    
    auto openCLBackend = static_cast<OpenCLBackend*>(backend());
    auto output = outputs[0];
    auto inputShape0 = tensorShapeFormat(inputs[0]);
    auto inputShape1 = tensorShapeFormat(inputs[1]);
    auto outputShape = tensorShapeFormat(output);
    auto runTime     = ((OpenCLBackend *)backend())->getOpenCLRuntime();
    int shape[4] = {outputShape[0], outputShape[1], outputShape[2], UP_DIV(outputShape[3], 4)};
    int fullCount[2] = {1, 1};
    
    int activationType = 0;
    if(mOp->type() == OpType_BinaryOp) {
        activationType = mOp->main_as_BinaryOp()->activationType();
    }
    auto &unit = mUnits[0];
    unit.kernel = runTime->buildKernel("binary_buf", "binary_buf", mBuildOptions);
    mMaxWorkGroupSize      = static_cast<uint32_t>(runTime->getMaxWorkGroupSize(unit.kernel));

    mGlobalWorkSize =  {(uint32_t)UP_DIV(outputShape[3], 4) * outputShape[0],
                                        (uint32_t)outputShape[1]*outputShape[2]};
    fullCount[0] = realSize(inputs[0]) == 1 ? 0 : 1;
    fullCount[1] = realSize(inputs[1]) == 1 ? 0 : 1;
    
    uint32_t index = 0;
    unit.kernel.setArg(index++, mGlobalWorkSize[0]);
    unit.kernel.setArg(index++, mGlobalWorkSize[1]);
    unit.kernel.setArg(index++, openCLBuffer(inputs[0]));
    unit.kernel.setArg(index++, openCLBuffer(inputs[1]));
    unit.kernel.setArg(index++, openCLBuffer(output));
    unit.kernel.setArg(index++, shape);
    unit.kernel.setArg(index++, fullCount);
    unit.kernel.setArg(index++, activationType);

    std::string name = "binary_buf";
    mLocalWorkSize = localWS2DDefault(mGlobalWorkSize, mMaxWorkGroupSize, openCLBackend->getOpenCLRuntime(), name, unit.kernel).first;
    
    unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1]};
    unit.localWorkSize  = {mLocalWorkSize[0], mLocalWorkSize[1]};
    for (int i = 2; i < inputs.size(); ++i) {
        fullCount[0] = 1;
        fullCount[1] = realSize(inputs[i]) == 1 ? 0 : 1;
        auto &unit = mUnits[i-1];
        unit.kernel = runTime->buildKernel("binary_buf", "binary_buf", mBuildOptions);

        uint32_t index = 0;
        unit.kernel.setArg(index++, mGlobalWorkSize[0]);
        unit.kernel.setArg(index++, mGlobalWorkSize[1]);
        unit.kernel.setArg(index++, openCLBuffer(output));
        unit.kernel.setArg(index++, openCLBuffer(inputs[i]));
        unit.kernel.setArg(index++, openCLBuffer(output));
        unit.kernel.setArg(index++, shape);
        unit.kernel.setArg(index++, fullCount);
        unit.kernel.setArg(index++, activationType);

        unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1]};
        unit.localWorkSize  = {mLocalWorkSize[0], mLocalWorkSize[1]};
    }
    return NO_ERROR;
}

class BinaryBufCreator : public OpenCLBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        if (op->type() == OpType_Eltwise) {
            switch (op->main_as_Eltwise()->type()) {
                case EltwiseType_SUM:
                    return new BinaryBufExecution(inputs, "in0+in1", op, backend);
                case EltwiseType_PROD:
                    return new BinaryBufExecution(inputs, "in0*in1", op, backend);
                case EltwiseType_SUB:
                    return new BinaryBufExecution(inputs, "in0-in1", op, backend);
                case EltwiseType_MAXIMUM:
                    return new BinaryBufExecution(inputs, "in0>in1?in0:in1", op, backend);
                default:
                    break;
            }
            return nullptr;
        }

        if (op->type() == OpType_BinaryOp) {
            MNN_ASSERT(inputs.size() > 1);

            switch (op->main_as_BinaryOp()->opType()) {
                case BinaryOpOperation_MUL:
                    return new BinaryBufExecution(inputs, "in0*in1", op, backend);
                case BinaryOpOperation_ADD:
                    return new BinaryBufExecution(inputs, "in0+in1", op, backend);
                case BinaryOpOperation_SUB:
                    return new BinaryBufExecution(inputs, "in0-in1", op, backend);
                case BinaryOpOperation_REALDIV:
                    return new BinaryBufExecution(inputs, "sign(in1)*in0/(fabs(in1)>(FLOAT4)((FLOAT)0.0000001)?fabs(in1):(FLOAT4)((FLOAT)0.0000001))", op, backend);
                case BinaryOpOperation_MINIMUM:
                    return new BinaryBufExecution(inputs, "in0>in1?in1:in0", op, backend);
                case BinaryOpOperation_MAXIMUM:
                    return new BinaryBufExecution(inputs, "in0>in1?in0:in1", op, backend);
                case BinaryOpOperation_GREATER:
                    return new BinaryBufExecution(inputs, "convert_float4(-isgreater(in0,in1))", op, backend);
                case BinaryOpOperation_LESS:
                    return new BinaryBufExecution(inputs, "convert_float4(-isless(in0,in1))", op, backend);
                case BinaryOpOperation_LESS_EQUAL:
                    return new BinaryBufExecution(inputs, "convert_float4(-islessequal(in0,in1))", op, backend);
                case BinaryOpOperation_GREATER_EQUAL:
                    return new BinaryBufExecution(inputs, "convert_float4(-isgreaterequal(in0,in1))", op, backend);
                case BinaryOpOperation_EQUAL:
                    return new BinaryBufExecution(inputs, "convert_float4(-isequal(in0,in1))", op, backend);
                case BinaryOpOperation_FLOORDIV:
                    return new BinaryBufExecution(inputs, "floor(sign(in1)*in0/(fabs(in1)>(FLOAT4)((FLOAT)0.0000001)?fabs(in1):(FLOAT4)((FLOAT)0.0000001)))", op, backend);
                case BinaryOpOperation_FLOORMOD:
                    return new BinaryBufExecution(inputs, "in0-floor(sign(in1)*in0/(fabs(in1)>(FLOAT4)((FLOAT)0.0000001)?fabs(in1):(FLOAT4)((FLOAT)0.0000001)))*in1", op, backend);
                case BinaryOpOperation_POW:
                    return new BinaryBufExecution(inputs, "pow(in0,in1)", op, backend);
                case BinaryOpOperation_SquaredDifference:
                    return new BinaryBufExecution(inputs, "(in0-in1)*(in0-in1)", op, backend);
                case BinaryOpOperation_ATAN2:
                    return new BinaryBufExecution(inputs, "atan(sign(in1)*in0/(fabs(in1)>(FLOAT4)((FLOAT)0.0000001)?fabs(in1):(FLOAT4)((FLOAT)0.0000001)))", op, backend);
                case BinaryOpOperation_NOTEQUAL:
                    return new BinaryBufExecution(inputs, "convert_float4(-isnotequal(in0,in1))", op, backend);
                case BinaryOpOperation_MOD:
                    return new BinaryBufExecution(inputs, "in0-sign(in1)*in0/(fabs(in1)>(FLOAT4)((FLOAT)0.0000001)?fabs(in1):(FLOAT4)((FLOAT)0.0000001))", op, backend);
                default:
                    break;
            }
            return nullptr;
        }
        return nullptr;
    }
};

OpenCLCreatorRegister<BinaryBufCreator> __eltwiseBuf_op(OpType_Eltwise, BUFFER);
OpenCLCreatorRegister<BinaryBufCreator> __binaryBuf_op(OpType_BinaryOp, BUFFER);

} // namespace OpenCL
} // namespace MNN
#endif /* MNN_OPENCL_BUFFER_CLOSED */
