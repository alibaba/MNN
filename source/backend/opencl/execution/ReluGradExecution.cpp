//
//  ReluGradExecution.cpp
//  MNN
//
//  Created by MNN on 2019/10/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/execution/ReluGradExecution.hpp"
#include "core/TensorUtils.hpp"

namespace MNN {
namespace OpenCL {

ReluGradExecution::ReluGradExecution(const MNN::Op *op, Backend *backend)
    : CommonExecution(backend) {
    if (op->type() == OpType_ReluGrad) {
        mKernelName = "relu_grad";
    } else if (op->type() == OpType_Relu6Grad) {
        mKernelName = "relu6_grad";
    } else {
        MNN_ERROR("unknown relu type\n");
        return;
    }
}

ReluGradExecution::~ReluGradExecution() {
    // do nothing
}

ErrorCode ReluGradExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mUnits.clear();
    mUnits.resize(1);

    auto nhwc = tensorShapeFormat(outputs[0]);
    uint32_t imageHeight = nhwc[0] * nhwc[1];
    uint32_t imageWidth = nhwc[2] * UP_DIV(nhwc[3], 4);

    auto runTime = ((OpenCLBackend *)backend())->getOpenCLRuntime();
    cl::Kernel kernel = runTime->buildKernel("binary_grad", mKernelName, {});
    kernel.setArg(0, openCLImage(inputs[0]));  // original input
    kernel.setArg(1, openCLImage(inputs[1]));  // grad for output
    kernel.setArg(2, openCLImage(outputs[0])); // grad for input
    mUnits[0].kernel = kernel;
    mUnits[0].localWorkSize = cl::NullRange;
    mUnits[0].globalWorkSize = {imageWidth, imageHeight};

    return NO_ERROR;
}

class ReluGradCreator : public OpenCLBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        return new ReluGradExecution(op, backend);
    }
};

OpenCLCreatorRegister<ReluGradCreator> __Relu_grad_op(OpType_ReluGrad);
OpenCLCreatorRegister<ReluGradCreator> __Relu6_grad_op(OpType_Relu6Grad);

}
}
