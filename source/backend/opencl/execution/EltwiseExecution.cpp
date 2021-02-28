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

EltwiseExecution::EltwiseExecution(const std::vector<Tensor *> &inputs, const std::string &compute, const MNN::Op *op, Backend *backend,
                                   float operatorData, bool broadCast)
    : CommonExecution(backend), mCompute(compute), mBroadCast(broadCast), mOperatorData(operatorData) {
    mBuildOptions.emplace("-DOPERATOR=" + compute);
    mOp = op;

}

ErrorCode EltwiseExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(inputs.size() >= 2);
    mUnits.resize(inputs.size() - 1);
    
    auto openCLBackend = static_cast<OpenCLBackend*>(backend());

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
    if (inputs.size() > 2) {
        auto output = outputs[0];
        mTempOutput.reset(Tensor::createDevice(output->shape(), output->getType(), output->getDimensionType()));
        bool res = openCLBackend->onAcquireBuffer(mTempOutput.get(), Backend::DYNAMIC);
        if (!res) {
            return OUT_OF_MEMORY;
        }
        openCLBackend->onReleaseBuffer(mTempOutput.get(), Backend::DYNAMIC);
    }

    auto runTime     = ((OpenCLBackend *)backend())->getOpenCLRuntime();
    bool useTempAsOutput = (inputs.size() % 2 != 0);
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
        /*
         DONT REMOVE THIS!!!!!
         When we do binary operation on many (>= 3) input image2d_t, we need:
         fun(outputs[0], inputs[i]) -> temp, then fun(temp, inputs[i+1]) -> outputs[0] and so on,
         instead of fun(outputs[0], inputs[i]) -> outputs[0]
         
         It's very very important for correctness on many common GPUs (Intel Iris GPU on MacBook Pro 15, for example) on Opencl 1.2.
         Opencl 1.2 do not guarantee correctness for kernel using same image2d_t as input and output, because Opencl 1.2 specification
         only support __read_only and __write_only, no include __read_write which is support on Opencl 2.x
         Your device may support it and get right result if remove this, but it is defined by the specification.
         If you insist on modifying this, please please contact hebin first. Thank you very much.
         */
        const Tensor* input0 = inputs[0];
        if (i >= 2) {
            input0 = useTempAsOutput ? outputs[0] : mTempOutput.get();
        }
        auto output = useTempAsOutput ? mTempOutput.get() : outputs[0];
        useTempAsOutput = !useTempAsOutput;
        
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
                        mBuildOptions.erase("-DOPERATOR=" + mCompute);
                        mBuildOptions.emplace("-DOPERATOR=" + swapComputeIn0In1(mCompute));
                        
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
                        mBuildOptions.erase("-DOPERATOR=" + mCompute);
                        mBuildOptions.emplace("-DOPERATOR=" + swapComputeIn0In1(mCompute));
                        
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
                case EltwiseType_PROD:
                    return new EltwiseExecution(inputs, "in0*in1", op, backend);
                case EltwiseType_MAXIMUM:
                    return new EltwiseExecution(inputs, "fmax(in0,in1)", op, backend);
                default:
                    break;
            }
            return nullptr;
        }

        if (op->type() == OpType_BinaryOp) {
            MNN_ASSERT(inputs.size() > 1);

            switch (op->main_as_BinaryOp()->opType()) {
                case BinaryOpOperation_ADD:
                    return new EltwiseExecution(inputs, "in0+in1", op, backend);
                case BinaryOpOperation_SUB:
                    return new EltwiseExecution(inputs, "in0-in1", op, backend);
                case BinaryOpOperation_MUL:
                    return new EltwiseExecution(inputs, "in0*in1", op, backend);
                case BinaryOpOperation_POW:
                    return new EltwiseExecution(inputs, "pow(in0,in1)", op, backend);
                case BinaryOpOperation_DIV:
                    return new EltwiseExecution(inputs, "sign(in1)*in0/fmax(fabs(in1), 0.0000001)", op, backend);
                case BinaryOpOperation_MAXIMUM:
                    return new EltwiseExecution(inputs, "fmax(in0,in1)", op, backend);
                case BinaryOpOperation_MINIMUM:
                    return new EltwiseExecution(inputs, "fmin(in0,in1)", op, backend);
                case BinaryOpOperation_REALDIV:
                    return new EltwiseExecution(inputs, "sign(in1)*in0/fmax(fabs(in1), 0.0000001)", op, backend);
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
