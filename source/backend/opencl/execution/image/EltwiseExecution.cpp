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

EltwiseExecution::EltwiseExecution(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, const std::string &compute, const MNN::Op *op, Backend *backend)
    : CommonExecution(backend, op), mCompute(compute) {
    MNN_ASSERT(inputs.size() >= 2);
    mUnits.resize(inputs.size() - 1);
    mMaxWorkGroupSize.resize(inputs.size() - 1);
    auto runTime = static_cast<OpenCLBackend*>(backend)->getOpenCLRuntime();
    std::set<std::string> buildOptions;
    buildOptions.emplace("-DOPERATOR=" + compute);
    for(int i = 0; i < mUnits.size(); ++i){
        mUnits[i].kernel = runTime->buildKernel("binary", "binary", buildOptions, inputs[i], outputs[0]);
        mMaxWorkGroupSize[i]  = static_cast<uint32_t>(runTime->getMaxWorkGroupSize(mUnits[i].kernel));
    }
}

uint32_t EltwiseExecution::realSize(const Tensor* tensor) {
    uint32_t num = 1;
    for(int i = 0; i < tensor->dimensions(); i++) {
        num *= tensor->length(i);
    }
    return num;
}

ErrorCode EltwiseExecution::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(inputs.size() >= 2);
    
    auto openCLBackend = static_cast<OpenCLBackend*>(backend());
    auto runTime     = openCLBackend->getOpenCLRuntime();

    auto output = outputs[0];
    auto inputShape0 = tensorShapeFormat(inputs[0]);
    auto inputShape1 = tensorShapeFormat(inputs[1]);
    auto outputShape = tensorShapeFormat(output);
    int shape[4] = {outputShape[0], outputShape[1], outputShape[2], UP_DIV(outputShape[3], 4)};
    int fullCount[2] = {1, 1};
    int activationType = 0;
    if(mOp->type() == OpType_BinaryOp) {
        activationType = mOp->main_as_BinaryOp()->activationType();
    }
    
    auto &unit = mUnits[0];

    std::vector<uint32_t> globalWorkSize =  {(uint32_t)UP_DIV(outputShape[3], 4)*outputShape[2],
                        (uint32_t)outputShape[0] * outputShape[1]};
    
    if(inputs.size() == 2) {
        fullCount[0] = realSize(inputs[0]) == 1 ? 0 : 1;
        fullCount[1] = realSize(inputs[1]) == 1 ? 0 : 1;
        
        uint32_t index = 0;
        cl_int ret = CL_SUCCESS;
        ret |= unit.kernel->get().setArg(index++, globalWorkSize[0]);
        ret |= unit.kernel->get().setArg(index++, globalWorkSize[1]);
        ret |= unit.kernel->get().setArg(index++, openCLImage(inputs[0]));
        ret |= unit.kernel->get().setArg(index++, openCLImage(inputs[1]));
        ret |= unit.kernel->get().setArg(index++, openCLImage(output));
        ret |= unit.kernel->get().setArg(index++, shape);
        ret |= unit.kernel->get().setArg(index++, fullCount);
        ret |= unit.kernel->get().setArg(index++, activationType);
        MNN_CHECK_CL_SUCCESS(ret, "setArg eltwiseExecution");

        std::string name = "binary";
        std::vector<uint32_t> localWorkSize = localWS2DDefault(globalWorkSize, mMaxWorkGroupSize[0], openCLBackend->getOpenCLRuntime(), name, unit.kernel).first;
        
        unit.globalWorkSize = {globalWorkSize[0], globalWorkSize[1]};
        unit.localWorkSize  = {localWorkSize[0], localWorkSize[1]};
        
        openCLBackend->recordKernel2d(unit.kernel, globalWorkSize, localWorkSize);
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
    std::vector<uint32_t> lws;
    for (int i = 0; i < inputs.size(); ++i) {
        if (i == 1)
            continue;

        auto &unit  = (i >= 2) ? mUnits[i - 1] : mUnits[i];

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
        cl_int ret = CL_SUCCESS;
        ret |= unit.kernel->get().setArg(index++, globalWorkSize[0]);
        ret |= unit.kernel->get().setArg(index++, globalWorkSize[1]);
        ret |= unit.kernel->get().setArg(index++, openCLImage(input0));
        ret |= unit.kernel->get().setArg(index++, openCLImage(input1));
        ret |= unit.kernel->get().setArg(index++, openCLImage(output));
        ret |= unit.kernel->get().setArg(index++, shape);
        ret |= unit.kernel->get().setArg(index++, fullCount);
        ret |= unit.kernel->get().setArg(index++, activationType);
        MNN_CHECK_CL_SUCCESS(ret, "setArg eltwiseExecution multiinput");

        if(i == 0) {
            std::string name = "binary";
            lws = localWS2DDefault(globalWorkSize, mMaxWorkGroupSize[i], openCLBackend->getOpenCLRuntime(), name, unit.kernel).first;
        }
        
        unit.globalWorkSize = {globalWorkSize[0], globalWorkSize[1]};
        unit.localWorkSize  = {lws[0], lws[1]};
        
        openCLBackend->recordKernel2d(unit.kernel, globalWorkSize, lws);
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
                    return new EltwiseExecution(inputs, outputs, "in0+in1", op, backend);
                case EltwiseType_SUB:
                    return new EltwiseExecution(inputs, outputs, "in0-in1", op, backend);
                case EltwiseType_PROD:
                    return new EltwiseExecution(inputs, outputs, "in0*in1", op, backend);
                case EltwiseType_MAXIMUM:
                    return new EltwiseExecution(inputs, outputs, "in0>in1?in0:in1", op, backend);
                default:
                    break;
            }
            return nullptr;
        }

        if (op->type() == OpType_BinaryOp) {
            MNN_ASSERT(inputs.size() > 1);

            switch (op->main_as_BinaryOp()->opType()) {
                case BinaryOpOperation_MUL:
                    return new EltwiseExecution(inputs, outputs, "in0*in1", op, backend);
                case BinaryOpOperation_ADD:
                    return new EltwiseExecution(inputs, outputs, "in0+in1", op, backend);
                case BinaryOpOperation_SUB:
                    return new EltwiseExecution(inputs, outputs, "in0-in1", op, backend);
                case BinaryOpOperation_REALDIV:
                    return new EltwiseExecution(inputs, outputs, "sign(in1)*in0/(fabs(in1)>(float4)((float)0.0000001)?fabs(in1):(float4)((float)0.0000001))", op, backend);
                case BinaryOpOperation_MINIMUM:
                    return new EltwiseExecution(inputs, outputs, "in0>in1?in1:in0", op, backend);
                case BinaryOpOperation_MAXIMUM:
                    return new EltwiseExecution(inputs, outputs, "in0>in1?in0:in1", op, backend);
                case BinaryOpOperation_GREATER:
                    return new EltwiseExecution(inputs, outputs, "convert_float4(-isgreater(in0,in1))", op, backend);
                case BinaryOpOperation_LESS:
                    return new EltwiseExecution(inputs, outputs, "convert_float4(-isless(in0,in1))", op, backend);
                case BinaryOpOperation_LESS_EQUAL:
                    return new EltwiseExecution(inputs, outputs, "convert_float4(-islessequal(in0,in1))", op, backend);
                case BinaryOpOperation_GREATER_EQUAL:
                    return new EltwiseExecution(inputs, outputs, "convert_float4(-isgreaterequal(in0,in1))", op, backend);
                case BinaryOpOperation_EQUAL:
                    return new EltwiseExecution(inputs, outputs, "convert_float4(-isequal(in0,in1))", op, backend);
                case BinaryOpOperation_FLOORDIV:
                    return new EltwiseExecution(inputs, outputs, "floor(sign(in1)*in0/(fabs(in1)>(float4)((float)0.0000001)?fabs(in1):(float4)((float)0.0000001)))", op, backend);
                case BinaryOpOperation_FLOORMOD:
                    return new EltwiseExecution(inputs, outputs, "in0-floor(sign(in1)*in0/(fabs(in1)>(float4)((float)0.0000001)?fabs(in1):(float4)((float)0.0000001)))*in1", op, backend);
                case BinaryOpOperation_POW:
                    return new EltwiseExecution(inputs, outputs, "pow(in0,in1)", op, backend);
                case BinaryOpOperation_SquaredDifference:
                    return new EltwiseExecution(inputs, outputs, "(in0-in1)*(in0-in1)", op, backend);
                case BinaryOpOperation_ATAN2:
                    return new EltwiseExecution(inputs, outputs, "(in1==(float4)0?(sign(in0)*(float4)(PI/2)):(atan(in0/in1)+(in1>(float4)0?(float4)0:sign(in0)*(float4)PI)))", op, backend);
                case BinaryOpOperation_NOTEQUAL:
                    return new EltwiseExecution(inputs, outputs, "convert_float4(-isnotequal(in0,in1))", op, backend);
                case BinaryOpOperation_MOD:
                    return new EltwiseExecution(inputs, outputs, "in0-floor(sign(in1)*in0/(fabs(in1)>(float4)((float)0.0000001)?fabs(in1):(float4)((float)0.0000001)))*in1", op, backend);
                default:
                    break;
            }
            return nullptr;
        }
        return nullptr;
    }
};

REGISTER_OPENCL_OP_CREATOR(EltwiseCreator, OpType_Eltwise, IMAGE);
REGISTER_OPENCL_OP_CREATOR(EltwiseCreator, OpType_BinaryOp, IMAGE);

} // namespace OpenCL
} // namespace MNN
