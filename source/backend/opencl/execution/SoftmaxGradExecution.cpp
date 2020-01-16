//
//  SoftmaxGradExecution.cpp
//  MNN
//
//  Created by MNN on 2019/10/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/execution/SoftmaxGradExecution.hpp"
#include "core/TensorUtils.hpp"

namespace MNN {
namespace OpenCL {

// current opencl image2d_t underly layout, support N-D data, (N,[spatial_shape depth (if have), height], C,W)
static std::vector<int> openclTensorShape(const Tensor* input) {
    int dim = input->dimensions();
    MNN_ASSERT(dim > 0);
    if (dim == 1) {
        return std::vector<int>({input->length(0), 1, 1, 1});
    }
    std::vector<int> res; // NCHW or NHWC
    for (int i = 0; i < dim; ++i) {
        res.push_back(input->length(i));
    }
    auto layout = TensorUtils::getDescribe(input)->dimensionFormat;
    // convert NCHW to NHWC
    if (layout == MNN_DATA_FORMAT_NCHW || layout == MNN_DATA_FORMAT_NC4HW4) {
        int channel = res[1];
        for (int i = 2; i < dim; ++i) {
            res[i - 1] = res[i];
        }
        res[dim - 1] = channel;
    }
    if (dim <= 3) {
        res.insert(res.begin() + 1, 1);
    }
    if (dim <= 2) {
        res.insert(res.begin() + 1, 1);
    }
    dim = res.size();
    std::swap(res[dim - 1], res[dim - 2]); // swap (W,C) to (C,W)
    return res;
}

SoftmaxGradExecution::SoftmaxGradExecution(Backend *backend, int axis)
    : CommonExecution(backend), mAxis(axis) {
    // do nothing
}

SoftmaxGradExecution::~SoftmaxGradExecution() {
    // do nothing
}

ErrorCode SoftmaxGradExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mUnits.clear();
    mUnits.resize(1);

    auto output = outputs[0];
    auto shape = openclTensorShape(output);
    int axis = mAxis;
    if (mAxis != 0) {
        const int dim = output->dimensions();
        auto layout = TensorUtils::getDescribe(output)->dimensionFormat;
        // convert axis from NCHW to NHWC
        if (layout == MNN_DATA_FORMAT_NCHW || layout == MNN_DATA_FORMAT_NC4HW4) {
            if (axis == 1) {
                axis = dim - 1;
            } else if (axis > 1) {
                axis = axis - 1;
            }
        }
        // convert axis from NHWC to NHCW (current opencl image2d_t underly layout, N..H,CW)
        if (dim > 2) {
            if (axis == dim - 1) {
                axis = axis - 1;
            } else if (axis == dim - 2) {
                axis = axis + 1;
            }
        } else {
            axis = 2;
        }
    }
    const int channelAxis = shape.size() - 2;
    int number = shape[axis], step = 1, remain = 1;
    int axisOnC4 = (axis == channelAxis ? 1 : 0); // softmax axis is channel dim (NH,CW)
    for (int i = 0; i < shape.size(); ++i) {
        int temp = shape[i];
        if (i == channelAxis) { // align up channel dim (NH,CW)
            temp = UP_DIV(temp, 4);
        }
        if (i > axis) {
            step *= temp;
        } else if (i < axis) {
            remain *= temp;
        }
    }
    auto runTime = ((OpenCLBackend *)backend())->getOpenCLRuntime();
    cl::Kernel kernel = runTime->buildKernel("softmax_grad", "softmax_grad", {});
    kernel.setArg(0, openCLImage(inputs[0]));  // original input
    kernel.setArg(1, openCLImage(inputs[1]));  // grad for output
    kernel.setArg(2, openCLImage(outputs[0])); // grad for input
    kernel.setArg(3, step);
    kernel.setArg(4, number);
    kernel.setArg(5, axisOnC4);
    mUnits[0].kernel = kernel;
    mUnits[0].localWorkSize = cl::NullRange;
    mUnits[0].globalWorkSize = {
        static_cast<uint32_t>(remain),
        static_cast<uint32_t>(step)
    };

    return NO_ERROR;
}

class SoftmaxGradCreator : public OpenCLBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        int axis = op->main_as_Axis()->axis();
        if (axis < 0) {
            axis = inputs[0]->dimensions() + axis;
        }
        return new SoftmaxGradExecution(backend, axis);
    }
};

OpenCLCreatorRegister<SoftmaxGradCreator> __Softmax_grad_op(OpType_SoftmaxGrad);
}
}
