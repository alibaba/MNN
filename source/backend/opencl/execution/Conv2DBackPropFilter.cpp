//
//  Conv2DBackPropFilter.cpp
//  MNN
//
//  Created by MNN on 2019/10/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string>
#include <vector>
#include "core/TensorUtils.hpp"
#include "backend/opencl/core/OpenCLBackend.hpp"
#include "backend/opencl/execution/Conv2DBackPropFilter.hpp"

namespace MNN {
namespace OpenCL {

Conv2DBackPropFilter::Conv2DBackPropFilter(const MNN::Op *op, Backend *backend) : CommonExecution(backend) {
    auto common = op->main_as_Convolution2D()->common();
    mStrides = {common->strideY(), common->strideX()};
    mDilations = {common->dilateY(), common->dilateX()};
    mKernels = {common->kernelY(), common->kernelX()};

    mPaddings = {common->padY(), common->padX()};
    if (common->padMode() == PadMode_VALID) {
        mPaddings[0] = mPaddings[1] = 0;
    }
}

Conv2DBackPropFilter::~Conv2DBackPropFilter() {
    // do nothing
}

ErrorCode Conv2DBackPropFilter::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mUnits.clear();
    mUnits.resize(2);

    auto originLayout = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
    auto openclBackend = static_cast<OpenCLBackend *>(backend());
    auto runtime = openclBackend->getOpenCLRuntime();

    const int weightSize = inputs[0]->elementSize();
    auto bufferPool = openclBackend->getBufferPool();
    auto bufferPtr = bufferPool->alloc(weightSize * sizeof(float), false);
    if (bufferPtr == nullptr) {
        return OUT_OF_MEMORY;
    }
    bufferPool->recycle(bufferPtr, false);

    {
        auto inputShape_ = tensorShapeFormat(inputs[1]);
        auto shape_ = tensorShapeFormat(inputs[2]);
        const int kernelHeight = mKernels[0], kernelWidth = mKernels[1];
        const int outputChannel = inputs[0]->length(0), inputChannel = inputs[0]->length(1);
        const int batch = inputs[1]->length(0), kernelSize = kernelWidth * kernelHeight;

        int inputShape[] = {inputShape_.at(2), inputShape_.at(1)};
        int shape[] = {shape_.at(2), shape_.at(1)};
        int kernelShape[] = {kernelWidth, kernelHeight};
        int strides[] = {mStrides[1], mStrides[0]};
        int pads[] = {mPaddings[1], mPaddings[0]};
        int dilates[] = {mDilations[1], mDilations[0]};

        cl::Kernel kernel = runtime->buildKernel("conv2d_backprop", "conv2d_backprop_filter", {});
        kernel.setArg(0, openCLImage(inputs[1]));
        kernel.setArg(1, openCLImage(inputs[2]));
        kernel.setArg(2, *bufferPtr);
        kernel.setArg(3, batch);
        kernel.setArg(4, outputChannel);
        kernel.setArg(5, inputChannel);
        kernel.setArg(6, sizeof(inputShape), inputShape);
        kernel.setArg(7, sizeof(shape), shape);
        kernel.setArg(8, sizeof(kernelShape), kernelShape);
        kernel.setArg(9, sizeof(strides), strides);
        kernel.setArg(10, sizeof(pads), pads);
        kernel.setArg(11, sizeof(dilates), dilates);

        const uint32_t maxWorkGroupSize = runtime->getMaxWorkGroupSize(kernel);
        std::vector<uint32_t> gws = {
            static_cast<uint32_t>(UP_DIV(outputChannel, 4)),
            static_cast<uint32_t>(UP_DIV(inputChannel, 4)),
            static_cast<uint32_t>(kernelSize)
        };
        std::vector<uint32_t> lws = {
            static_cast<uint32_t>(ALIMIN(maxWorkGroupSize / kernelSize, 32)), 1,
            static_cast<uint32_t>(kernelSize)
        };
        if (kernelSize == 1) {
            lws[1] = ALIMIN(maxWorkGroupSize / lws[0], 4);
        }
        for (size_t i = 0; i < lws.size(); ++i) {
            gws[i] = ROUND_UP(gws[i], lws[i]);
        }

        mUnits[0].kernel = kernel;
        mUnits[1].localWorkSize = {lws[0], lws[1], lws[2]};
        mUnits[0].globalWorkSize = {gws[0], gws[1], gws[2]};
    }
    // transform kernel from normal format (oc,ic,kh,kw) to image2d (NHCW)
    {
        std::string kernelName = "";
        if (originLayout == MNN_DATA_FORMAT_NCHW) {
            kernelName = "nchw_buffer_to_image";
        } else if (originLayout == MNN_DATA_FORMAT_NHWC) {
            kernelName = "nhwc_buffer_to_image";
        }
        auto shape = tensorShapeFormat(inputs[0]);
        std::vector<uint32_t> gws = {
            static_cast<uint32_t>(shape[2] * UP_DIV(shape[3], 4)),
            static_cast<uint32_t>(shape[0] * shape[1])
        };

        cl::Kernel kernel = runtime->buildKernel("buffer_to_image", kernelName, {});
        kernel.setArg(0, gws[0]);
        kernel.setArg(1, gws[1]);
        kernel.setArg(2, *bufferPtr);
        kernel.setArg(3, shape[1]);
        kernel.setArg(4, shape[2]);
        kernel.setArg(5, shape[3]);
        kernel.setArg(6, openCLImage(outputs[0]));

        const uint32_t maxWorkGroupSize = runtime->getMaxWorkGroupSize(kernel);
        std::vector<uint32_t> lws = {16, std::max((uint32_t)1, maxWorkGroupSize / 16)};
        for (size_t i = 0; i < lws.size(); ++i) {
            gws[i] = ROUND_UP(gws[i], lws[i]);
        }

        mUnits[1].kernel = kernel;
        mUnits[1].localWorkSize = {lws[0], lws[1]};
        mUnits[1].globalWorkSize = {gws[0], gws[1]};
    }
    //MNN_PRINT("flag\n");

    return NO_ERROR;
}

class Conv2DBackPropFilterCreator : public OpenCLBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        return new Conv2DBackPropFilter(op, backend);
    }
};

OpenCLCreatorRegister<Conv2DBackPropFilterCreator> __conv_backprop_filter_grad_op(OpType_Conv2DBackPropFilter);

}
}
