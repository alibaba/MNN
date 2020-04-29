//
//  EltwiseExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/execution/EltwiseExecution.hpp"

#include "core/Macro.h"
#include <string.h>
#include "core/TensorUtils.hpp"

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
    
    auto output = outputs[0];
    auto openCLBackend = static_cast<OpenCLBackend*>(backend());
    std::shared_ptr<Tensor> myTensor;
    if (inputs[0] == output) {
        myTensor.reset(new Tensor(output, output->getDimensionType(), false));
        auto success = openCLBackend->onAcquireBuffer(myTensor.get(), Backend::DYNAMIC);
        if (!success) {
            return OUT_OF_MEMORY;
        }
        openCLBackend->onReleaseBuffer(myTensor.get(), Backend::DYNAMIC);
        output = myTensor.get();
    }

    auto nhwc0     = tensorShapeFormat(inputs[0]);
    auto nhwc       = tensorShapeFormat(outputs[0]);

    int nhwcArray[] = {nhwc[0], nhwc[1], nhwc[2], UP_DIV(nhwc[3], 4)};
    auto imageWidth  = nhwcArray[2] * nhwcArray[3];
    auto imageHeight = nhwcArray[0] * nhwcArray[1];

    int wh0[]             = {nhwc0[2], nhwc0[1]};
    int wh[]              = {nhwc[2], nhwc[1]};

    int input1Stride[]     = {1, 1, 1, 1};
    cl::NDRange localSize  = {16, 16};
    cl::NDRange globalSize = {(uint32_t)UP_DIV(imageWidth, 16) * 16, (uint32_t)UP_DIV(imageHeight, 16) * 16};

    auto runTime     = ((OpenCLBackend *)backend())->getOpenCLRuntime();
    for (int i = 0; i < inputs.size(); ++i) {
        if (i == 1)
            continue;

        auto &unit  = (i >= 2) ? mUnits[i - 1] : mUnits[i];
        int dimension = (i >= 2) ? inputs[i]->dimensions() : inputs[i + 1]->dimensions();
        int nums = 1;
        const auto& shape = (i >= 2) ? inputs[i]->shape() : inputs[i + 1]->shape();
        for (auto axis_len:shape) {
            nums*=axis_len;
        }
        const Tensor* input0 = (i >= 2) ? outputs[0] : inputs[0];
        if(dimension == 0 || nums == 1) {
            auto input = (i >= 2) ? inputs[i] : inputs[i + 1];
            unit.kernel = runTime->buildKernel("binary", "binary_value", mBuildOptions);
            unit.kernel.setArg(0, openCLImage(input0));
            unit.kernel.setArg(1, openCLImage(input));
            unit.kernel.setArg(2, openCLImage(output));
            unit.kernel.setArg(3, nhwcArray);
            unit.kernel.setArg(4, wh);
            unit.kernel.setArg(5, input1Stride);
        } else {
            const Tensor* input = (i >= 2) ? inputs[i] : inputs[i + 1];
            auto nhwc_0  = (i >= 2) ? nhwc : nhwc0;
            auto wh_v = (i >= 2) ? wh : wh0;
            int wh_0[] = {wh_v[0], wh_v[1]};
            auto nhwc_1 = tensorShapeFormat(input);
            int wh1[] = {nhwc_1[2], nhwc_1[1]};
            for (int dim = 0; dim < nhwc_0.size(); dim++) {
                if (nhwc_0[dim] != nhwc_1[dim]) {
                    mBroadCast = true;
                    break;
                }
            }

            if (mBroadCast) {
                if (nhwc_0[3] != nhwc_1[3]) {
                    if (nhwc_0[3] == 1) {
                        unit.kernel = (wh_0[0] != 1 && wh_0[1] != 1) ?
                            runTime->buildKernel("binary",
                                "binary_1toM_channel_broadcast_on_awh", mBuildOptions) :
                            runTime->buildKernel("binary",
                                "binary_1toM_channel_broadcast_on_1wh", mBuildOptions);
                        unit.kernel.setArg(0, openCLImage(input0));
                        unit.kernel.setArg(1, openCLImage(input));
                        unit.kernel.setArg(4, wh_0);
                        unit.kernel.setArg(5, wh1);
                    } else {
                        unit.kernel = (wh1[0] != 1 && wh1[1] != 1) ?
                            runTime->buildKernel("binary",
                                "binary_1toM_channel_broadcast_on_awh", mBuildOptions) :
                            runTime->buildKernel("binary",
                                "binary_1toM_channel_broadcast_on_1wh", mBuildOptions);
                        unit.kernel.setArg(0, openCLImage(input));
                        unit.kernel.setArg(1, openCLImage(input0));
                        unit.kernel.setArg(4, wh1);
                        unit.kernel.setArg(5, wh_0);
                    }
                    unit.kernel.setArg(2, openCLImage(output));
                    unit.kernel.setArg(3, nhwcArray);
                    unit.kernel.setArg(6, wh);
                } else {
                    unit.kernel = runTime->buildKernel("binary",
                            "binary_same_channel_broadcast", mBuildOptions);
                    if (wh_0[0] == 1 || wh_0[1] == 1) {
                        unit.kernel.setArg(0, openCLImage(input0));
                        unit.kernel.setArg(1, openCLImage(input));
                        unit.kernel.setArg(4, wh_0);
                        unit.kernel.setArg(5, wh1);

                    } else {
                        unit.kernel.setArg(0, openCLImage(input));
                        unit.kernel.setArg(1, openCLImage(input0));
                        unit.kernel.setArg(4, wh1);
                        unit.kernel.setArg(5, wh_0);
                    }
                    unit.kernel.setArg(2, openCLImage(output));
                    unit.kernel.setArg(3, nhwcArray);
                    unit.kernel.setArg(6, wh);
                }
            } else {
                unit.kernel = runTime->buildKernel("binary", "binary", mBuildOptions);
                unit.kernel.setArg(0, openCLImage(input0));
                unit.kernel.setArg(1, openCLImage(input));
                unit.kernel.setArg(2, openCLImage(output));
                unit.kernel.setArg(3, nhwcArray);
                unit.kernel.setArg(4, wh);
                unit.kernel.setArg(5, input1Stride);
            }
        }
        unit.globalWorkSize = globalSize;
        unit.localWorkSize  = localSize;
    }
    if (output != outputs[0]) {
        Unit unit;
        unit.kernel = runTime->buildKernel("binary", "imageCopy", mBuildOptions);
        unit.kernel.setArg(0, openCLImage(output));
        unit.kernel.setArg(1, openCLImage(outputs[0]));
        unit.localWorkSize = cl::NullRange;
        unit.globalWorkSize = {static_cast<uint32_t>(imageWidth), static_cast<uint32_t>(imageHeight)};
        mUnits.push_back(unit);
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
                    return new EltwiseExecution(inputs, "fmax(in0,in1)", backend);
                default:
                    break;
            }
            return nullptr;
        }

        if (op->type() == OpType_BinaryOp) {
            MNN_ASSERT(inputs.size() > 1);

            switch (op->main_as_BinaryOp()->opType()) {
                case BinaryOpOperation_ADD:
                    return new EltwiseExecution(inputs, "in0+in1", backend);
                case BinaryOpOperation_SUB:
                    return new EltwiseExecution(inputs, "in0-in1", backend);
                case BinaryOpOperation_MUL:
                    return new EltwiseExecution(inputs, "in0*in1", backend);
                case BinaryOpOperation_POW:
                    return new EltwiseExecution(inputs, "pow(in0,in1)", backend);
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
