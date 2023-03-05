//
//  EltwiseExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/execution/image/EltwiseExecution.hpp"

#include "core/Macro.h"
#include <string.h>
#include <string>
#include "core/TensorUtils.hpp"

using std::string;
namespace MNN {
namespace OpenCL {

static string swapComputeIn0In1(const string& computeOrigin) {
    string compute = computeOrigin;
    for (int i = 2; i < compute.length(); ++i) {
        if (compute.substr(i - 2, 2) == "in") {
            compute[i] = (compute[i] == '0' ? '1' : '0');
        }
    }
    return compute;
}

EltwiseExecution::EltwiseExecution(const std::vector<Tensor *> &inputs, const std::string &compute, const MNN::Op *op, Backend *backend)
    : CommonExecution(backend), mCompute(compute) {
    mBuildOptions.emplace("-DOPERATOR=" + compute);
    mOp = op;

}

uint32_t EltwiseExecution::realSize(const Tensor* tensor) {
    uint32_t num = 1;
    for(int i = 0; i < tensor->dimensions(); i++) {
        num *= tensor->length(i);
    }
    return num;
}

ErrorCode EltwiseExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
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
    unit.kernel = runTime->buildKernel("binary", "binary", mBuildOptions);
    mMaxWorkGroupSize  = static_cast<uint32_t>(runTime->getMaxWorkGroupSize(unit.kernel));

    mGlobalWorkSize =  {(uint32_t)UP_DIV(outputShape[3], 4)*outputShape[2],
                        (uint32_t)outputShape[0] * outputShape[1]};
    
    if(inputs.size() == 2) {
        fullCount[0] = realSize(inputs[0]) == 1 ? 0 : 1;
        fullCount[1] = realSize(inputs[1]) == 1 ? 0 : 1;
        
        uint32_t index = 0;
        unit.kernel.setArg(index++, mGlobalWorkSize[0]);
        unit.kernel.setArg(index++, mGlobalWorkSize[1]);
        unit.kernel.setArg(index++, openCLImage(inputs[0]));
        unit.kernel.setArg(index++, openCLImage(inputs[1]));
        unit.kernel.setArg(index++, openCLImage(output));
        unit.kernel.setArg(index++, shape);
        unit.kernel.setArg(index++, fullCount);
        unit.kernel.setArg(index++, activationType);

        std::string name = "binary";
        mLocalWorkSize = localWS2DDefault(mGlobalWorkSize, mMaxWorkGroupSize, openCLBackend->getOpenCLRuntime(), name, unit.kernel).first;
        
        unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1]};
        unit.localWorkSize  = {mLocalWorkSize[0], mLocalWorkSize[1]};
        
        return NO_ERROR;
    }
    
    if (inputs.size() > 2) {
        auto output = outputs[0];
        mTempOutput.reset(Tensor::createDevice(output->shape(), output->getType(), output->getDimensionType()));
        bool res = openCLBackend->onAcquireBuffer(mTempOutput.get(), Backend::DYNAMIC);
        if (!res) {
            return OUT_OF_MEMORY;
        }
        openCLBackend->onReleaseBuffer(mTempOutput.get(), Backend::DYNAMIC);
    }

    bool useTempAsOutput = (inputs.size() % 2 != 0);
    fullCount[1] = 1;

    for (int i = 0; i < inputs.size(); ++i) {
        if (i == 1)
            continue;

        auto &unit  = (i >= 2) ? mUnits[i - 1] : mUnits[i];
        unit.kernel = runTime->buildKernel("binary", "binary", mBuildOptions);

        auto input0 = inputs[0];
        fullCount[0] = realSize(input0) == 1 ? 0 : 1;
        if (i >= 2) {
            input0 = useTempAsOutput ? outputs[0] : mTempOutput.get();
            fullCount[0] = 1;
        }
        
        auto input1 = (i >= 2) ? inputs[i] : inputs[i + 1];
        fullCount[1] = realSize(input1) == 1 ? 0 : 1;

        auto output = useTempAsOutput ? mTempOutput.get() : outputs[0];
        useTempAsOutput = !useTempAsOutput;
        
        uint32_t index = 0;
        unit.kernel.setArg(index++, mGlobalWorkSize[0]);
        unit.kernel.setArg(index++, mGlobalWorkSize[1]);
        unit.kernel.setArg(index++, openCLImage(input0));
        unit.kernel.setArg(index++, openCLImage(input1));
        unit.kernel.setArg(index++, openCLImage(output));
        unit.kernel.setArg(index++, shape);
        unit.kernel.setArg(index++, fullCount);
        unit.kernel.setArg(index++, activationType);

        if(i == 0) {
            std::string name = "binary";
            mLocalWorkSize = localWS2DDefault(mGlobalWorkSize, mMaxWorkGroupSize, openCLBackend->getOpenCLRuntime(), name, unit.kernel).first;
        }
        
        unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1]};
        unit.localWorkSize  = {mLocalWorkSize[0], mLocalWorkSize[1]};
    }
    return NO_ERROR;
}

class EltwiseCreator : public OpenCLBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        if (op->type() == OpType_Eltwise) {
            switch (op->main_as_Eltwise()->type()) {
                case EltwiseType_SUM:
                    return new EltwiseExecution(inputs, "in0+in1", op, backend);
                case EltwiseType_SUB:
                    return new EltwiseExecution(inputs, "in0-in1", op, backend);
                case EltwiseType_PROD:
                    return new EltwiseExecution(inputs, "in0*in1", op, backend);
                case EltwiseType_MAXIMUM:
                    return new EltwiseExecution(inputs, "in0>in1?in0:in1", op, backend);
                default:
                    break;
            }
            return nullptr;
        }

        if (op->type() == OpType_BinaryOp) {
            MNN_ASSERT(inputs.size() > 1);

            switch (op->main_as_BinaryOp()->opType()) {
                case BinaryOpOperation_MUL:
                    return new EltwiseExecution(inputs, "in0*in1", op, backend);
                case BinaryOpOperation_ADD:
                    return new EltwiseExecution(inputs, "in0+in1", op, backend);
                case BinaryOpOperation_SUB:
                    return new EltwiseExecution(inputs, "in0-in1", op, backend);
                case BinaryOpOperation_REALDIV:
                    return new EltwiseExecution(inputs, "sign(in1)*in0/(fabs(in1)>(FLOAT4)((FLOAT)0.0000001)?fabs(in1):(FLOAT4)((FLOAT)0.0000001))", op, backend);
                case BinaryOpOperation_MINIMUM:
                    return new EltwiseExecution(inputs, "in0>in1?in1:in0", op, backend);
                case BinaryOpOperation_MAXIMUM:
                    return new EltwiseExecution(inputs, "in0>in1?in0:in1", op, backend);
                case BinaryOpOperation_GREATER:
                    return new EltwiseExecution(inputs, "convert_float4(-isgreater(in0,in1))", op, backend);
                case BinaryOpOperation_LESS:
                    return new EltwiseExecution(inputs, "convert_float4(-isless(in0,in1))", op, backend);
                case BinaryOpOperation_LESS_EQUAL:
                    return new EltwiseExecution(inputs, "convert_float4(-islessequal(in0,in1))", op, backend);
                case BinaryOpOperation_GREATER_EQUAL:
                    return new EltwiseExecution(inputs, "convert_float4(-isgreaterequal(in0,in1))", op, backend);
                case BinaryOpOperation_EQUAL:
                    return new EltwiseExecution(inputs, "convert_float4(-isequal(in0,in1))", op, backend);
                case BinaryOpOperation_FLOORDIV:
                    return new EltwiseExecution(inputs, "floor(sign(in1)*in0/(fabs(in1)>(FLOAT4)((FLOAT)0.0000001)?fabs(in1):(FLOAT4)((FLOAT)0.0000001)))", op, backend);
                case BinaryOpOperation_FLOORMOD:
                    return new EltwiseExecution(inputs, "in0-floor(sign(in1)*in0/(fabs(in1)>(FLOAT4)((FLOAT)0.0000001)?fabs(in1):(FLOAT4)((FLOAT)0.0000001)))*in1", op, backend);
                case BinaryOpOperation_POW:
                    return new EltwiseExecution(inputs, "pow(in0,in1)", op, backend);
                case BinaryOpOperation_SquaredDifference:
                    return new EltwiseExecution(inputs, "(in0-in1)*(in0-in1)", op, backend);
                case BinaryOpOperation_ATAN2:
                    return new EltwiseExecution(inputs, "atan(sign(in1)*in0/(fabs(in1)>(FLOAT4)((FLOAT)0.0000001)?fabs(in1):(FLOAT4)((FLOAT)0.0000001)))", op, backend);
                case BinaryOpOperation_NOTEQUAL:
                    return new EltwiseExecution(inputs, "convert_float4(-isnotequal(in0,in1))", op, backend);
                case BinaryOpOperation_MOD:
                    return new EltwiseExecution(inputs, "in0-sign(in1)*in0/(fabs(in1)>(FLOAT4)((FLOAT)0.0000001)?fabs(in1):(FLOAT4)((FLOAT)0.0000001))", op, backend);
                default:
                    break;
            }
            return nullptr;
        }
        return nullptr;
    }
};

OpenCLCreatorRegister<EltwiseCreator> __eltwise_op(OpType_Eltwise, IMAGE);
OpenCLCreatorRegister<EltwiseCreator> __binary_op(OpType_BinaryOp, IMAGE);

} // namespace OpenCL
} // namespace MNN
