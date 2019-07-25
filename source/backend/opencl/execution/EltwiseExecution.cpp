//
//  EltwiseExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "EltwiseExecution.hpp"

#include <Macro.h>
#include <string.h>
#include "TensorUtils.hpp"

namespace MNN {
namespace OpenCL {

EltwiseExecution::EltwiseExecution(const std::vector<Tensor *> &inputs, const std::string &compute, Backend *backend, float operatorData, bool broadCast)
    : CommonExecution(backend) {
    mBroadCast = broadCast;
    mOperatorData = operatorData;
    mBuildOptions.emplace("-DOPERATOR=" + compute);
}

ErrorCode EltwiseExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(inputs.size() >= 2);
    mUnits.resize(inputs.size() - 1);

    auto nhwc       = tensorShapeFormat(outputs[0]);
    int nhwcArray[] = {nhwc[0], nhwc[1], nhwc[2], UP_DIV(nhwc[3], 4)};

    auto imageWidth  = nhwcArray[2] * nhwcArray[3];
    auto imageHeight = nhwcArray[0] * nhwcArray[1];

    int wh[]               = {nhwc[2], nhwc[1]};
    int input1Stride[]     = {1, 1, 1, 1};
    cl::NDRange localSize  = {16, 16};
    cl::NDRange globalSize = {(uint32_t)UP_DIV(imageWidth, 16) * 16, (uint32_t)UP_DIV(imageHeight, 16) * 16};

    auto runTime     = ((OpenCLBackend *)backend())->getOpenCLRuntime();

    if(mBroadCast == true){
        mUnits[0].kernel = runTime->buildKernel("binary", "binary_broadcast", mBuildOptions);
        mUnits[0].kernel.setArg(0, openCLImage(inputs[0]));
        mUnits[0].kernel.setArg(1, mOperatorData);
        mUnits[0].kernel.setArg(2, openCLImage(outputs[0]));
        mUnits[0].kernel.setArg(3, nhwcArray);
        mUnits[0].kernel.setArg(4, wh);
        mUnits[0].kernel.setArg(5, input1Stride);
        mUnits[0].globalWorkSize = globalSize;
        mUnits[0].localWorkSize  = localSize;
        return NO_ERROR;
    }
    mUnits[0].kernel = runTime->buildKernel("binary", "binary", mBuildOptions);
    mUnits[0].kernel.setArg(0, openCLImage(inputs[0]));
    mUnits[0].kernel.setArg(1, openCLImage(inputs[1]));
    mUnits[0].kernel.setArg(2, openCLImage(outputs[0]));
    mUnits[0].kernel.setArg(3, nhwcArray);
    mUnits[0].kernel.setArg(4, wh);
    mUnits[0].kernel.setArg(5, input1Stride);
    mUnits[0].globalWorkSize = globalSize;
    mUnits[0].localWorkSize  = localSize;
    for (int i = 2; i < inputs.size(); ++i) {
        auto &unit  = mUnits[i - 1];
        unit.kernel = runTime->buildKernel("binary", "binary", mBuildOptions);
        unit.kernel.setArg(0, openCLImage(inputs[i]));
        unit.kernel.setArg(1, openCLImage(outputs[0]));
        unit.kernel.setArg(2, openCLImage(outputs[0]));
        unit.kernel.setArg(3, nhwcArray);
        unit.kernel.setArg(4, wh);
        unit.kernel.setArg(5, input1Stride);
        unit.globalWorkSize = globalSize;
        unit.localWorkSize  = localSize;
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
                    return new EltwiseExecution(inputs, "in0+in1", backend);
                case EltwiseType_PROD:
                    return new EltwiseExecution(inputs, "in0*in1", backend);
                case EltwiseType_MAXIMUM:
                    return new EltwiseExecution(inputs, "fmax(in0, in1)", backend);
                default:
                    break;
            }
            return nullptr;
        }
        if (op->type() == OpType_BinaryOp) {
            MNN_ASSERT(inputs.size() > 1);
            auto input0 = inputs[0];
            // Don't support broatcast
            for (int i = 1; i < inputs.size(); ++i) {
                auto input = inputs[i];
                if (input0->dimensions() != input->dimensions()) {
                    if(input->dimensions() == 0){
                        float operatorData = input->host<float>()[0];
                        switch (op->main_as_BinaryOp()->opType()) {
                                case BinaryOpOperation_REALDIV:
                                return new EltwiseExecution(inputs, "in0/in1", backend, operatorData, true);
                            default:
                                break;
                        }
                    }
                    return nullptr;
                }
                auto dim = input0->dimensions();
                for (int l = 0; l < dim; ++l) {
                    if (input0->length(l) != input->length(l) && input->dimensions() != 0) {
                        return nullptr;
                    }
                }
            }
            
            
            switch (op->main_as_BinaryOp()->opType()) {
                case BinaryOpOperation_ADD:
                    return new EltwiseExecution(inputs, "in0+in1", backend);
                case BinaryOpOperation_SUB:
                    return new EltwiseExecution(inputs, "in0-in1", backend);
                case BinaryOpOperation_MUL:
                    return new EltwiseExecution(inputs, "in0*in1", backend);
                case BinaryOpOperation_POW:
                    return new EltwiseExecution(inputs, "pow(in0, in1)", backend);
                case BinaryOpOperation_DIV:
                    return new EltwiseExecution(inputs, "in0/in1", backend);
                case BinaryOpOperation_MAXIMUM:
                    return new EltwiseExecution(inputs, "fmax(in0,in1)", backend);
                case BinaryOpOperation_MINIMUM:
                    return new EltwiseExecution(inputs, "fmin(in0,in1)", backend);
                case BinaryOpOperation_REALDIV:
                    return new EltwiseExecution(inputs, "in0/in1", backend);
                default:
                    break;
            }
            return nullptr;
        }
        return nullptr;
    }
};

OpenCLCreatorRegister<EltwiseCreator> __eltwise_op(OpType_Eltwise);
OpenCLCreatorRegister<EltwiseCreator> __binary_op(OpType_BinaryOp);

} // namespace OpenCL
} // namespace MNN
