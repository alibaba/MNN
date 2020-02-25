//
//  MultiInputConvExecution.cpp
//  MNN
//
//  Created by MNN on 2019/10/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string>
#include <vector>
#include "core/TensorUtils.hpp"
#include "backend/opencl/core/OpenCLBackend.hpp"
#include "backend/opencl/execution/MultiInputConvExecution.hpp"

namespace MNN {
namespace OpenCL {

MultiInputConvExecution::MultiInputConvExecution(const MNN::Op *op, Backend *backend) : CommonExecution(backend) {
    auto common = op->main_as_Convolution2D()->common();
    mPadMode = common->padMode();
    mStrides = {common->strideY(), common->strideX()};
    mDilations = {common->dilateY(), common->dilateX()};
    if (mPadMode != PadMode_SAME) {
        mPaddings = {common->padY() * 2, common->padX() * 2};
    }
}

MultiInputConvExecution::~MultiInputConvExecution() {
    // do nothing
}

ErrorCode MultiInputConvExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mUnits.clear();
    mUnits.resize(3);

    auto originLayout = TensorUtils::getDescribe(inputs[1])->dimensionFormat;
    auto openclBackend = static_cast<OpenCLBackend *>(backend());
    auto runtime = openclBackend->getOpenCLRuntime();

    auto inputShape  = tensorShapeFormat(inputs[0]);
    auto outputShape = tensorShapeFormat(outputs[0]);
    const int batch = outputShape.at(0);
    const int outputChannel = outputShape.at(3), inputChannel = inputShape.at(3);
    const int inputHeight = inputShape.at(1), inputWidth = inputShape.at(2);
    const int height = outputShape.at(1), width = outputShape.at(2);
    const int kernelY = inputs[1]->length(2), kernelX = inputs[1]->length(3);
    int kernelShape[2] = {kernelY, kernelX};

    if (mPadMode == PadMode_SAME) {
        int padNeededHeight = (height - 1) * mStrides[0] + (kernelY - 1) * mDilations[0] + 1 - inputHeight;
        int padNeededWidth = (width - 1) * mStrides[1] + (kernelX - 1) * mDilations[1] + 1 - inputWidth;
        mPaddings[0] = padNeededHeight;
        mPaddings[1] = padNeededWidth;
    }

    const int weightSize = inputs[1]->elementSize();
    auto bufferPool = openclBackend->getBufferPool();
    auto bufferPtr = bufferPool->alloc(weightSize * sizeof(float), false);
    if (bufferPtr == nullptr) {
        return OUT_OF_MEMORY;
    }
    mFilter.reset(Tensor::createDevice<float>({1, UP_DIV(outputChannel, 4) * kernelY * kernelX, 1, inputChannel * 4}));
    bool succ = openclBackend->onAcquireBuffer(mFilter.get(), Backend::DYNAMIC);
    bufferPool->recycle(bufferPtr, false);
    if (!succ) {
        return OUT_OF_MEMORY;
    }
    openclBackend->onReleaseBuffer(mFilter.get(), Backend::DYNAMIC);

    // transform kernel from image2d (NHCW) to original form (maybe NCHW or NHWC)
    {
        std::string kernelName = "";
        if (originLayout == MNN_DATA_FORMAT_NCHW) {
            kernelName = "image_to_nchw_buffer";
        } else if (originLayout == MNN_DATA_FORMAT_NHWC) {
            kernelName = "image_to_nhwc_buffer";
        }
        auto shape = tensorShapeFormat(inputs[1]);
        std::vector<uint32_t> gws = {static_cast<uint32_t>(shape[2] * UP_DIV(shape[3], 4)), static_cast<uint32_t>(shape[0] * shape[1])};

        cl::Kernel kernel = runtime->buildKernel("buffer_to_image", kernelName, {});
        kernel.setArg(0, gws[0]);
        kernel.setArg(1, gws[1]);
        kernel.setArg(2, *bufferPtr);
        kernel.setArg(3, shape[1]);
        kernel.setArg(4, shape[2]);
        kernel.setArg(5, shape[3]);
        kernel.setArg(6, openCLImage(inputs[1]));

        const uint32_t maxWorkGroupSize = runtime->getMaxWorkGroupSize(kernel);
        std::vector<uint32_t> lws = {16, std::max((uint32_t)1, maxWorkGroupSize / 16)};
        for (size_t i = 0; i < lws.size(); ++i) {
            gws[i] = ROUND_UP(gws[i], lws[i]);
        }

        mUnits[0].kernel = kernel;
        mUnits[0].localWorkSize = {lws[0], lws[1]};
        mUnits[0].globalWorkSize = {gws[0], gws[1]};
    }

    // transform kernel from original form (maybe NCHW or NHWC) to filter format
    {
        std::vector<uint32_t> gws = {static_cast<uint32_t>(inputChannel), static_cast<uint32_t>(UP_DIV(outputChannel, 4) * kernelY * kernelX)};

        cl::Kernel kernel = runtime->buildKernel("buffer_to_image", "conv2d_filter_buffer_to_image", {});
        kernel.setArg(0, gws[0]);
        kernel.setArg(1, gws[1]);
        kernel.setArg(2, *bufferPtr);
        kernel.setArg(3, outputChannel);
        kernel.setArg(4, sizeof(kernelShape), kernelShape);
        kernel.setArg(5, inputChannel * kernelY * kernelX);
        kernel.setArg(6, kernelY * kernelX);
        kernel.setArg(7, openCLImage(mFilter.get()));

        const uint32_t maxWorkGroupSize = runtime->getMaxWorkGroupSize(kernel);
        std::vector<uint32_t> lws = {16, std::max((uint32_t)1, maxWorkGroupSize / 16)};
        for (size_t i = 0; i < lws.size(); ++i) {
            gws[i] = ROUND_UP(gws[i], lws[i]);
        }

        mUnits[1].kernel = kernel;
        mUnits[1].localWorkSize = {lws[0], lws[1]};
        mUnits[1].globalWorkSize = {gws[0], gws[1]};
    }

    {
        std::vector<uint32_t> gws = {static_cast<uint32_t>(UP_DIV(outputChannel, 4) * UP_DIV(width, 4)), static_cast<uint32_t>(height * batch)};
        int inputImageShape[2]  = {inputHeight, inputWidth};
        int outputImageShape[2] = {height, width};
        int strideShape[2]      = {mStrides[0], mStrides[1]};
        int paddingShape[2]     = {mPaddings[0] / 2, mPaddings[1] / 2};
        int dilationShape[2]    = {mDilations[0], mDilations[1]};

        std::set<std::string> buildOptions;
        if (inputs.size() > 2) {
            buildOptions.emplace("-DBIAS");
        }
        cl::Kernel kernel = runtime->buildKernel("conv_2d", "conv_2d", buildOptions);
        int idx = 0;
        kernel.setArg(idx++, gws[0]);
        kernel.setArg(idx++, gws[1]);
        kernel.setArg(idx++, openCLImage(inputs[0]));
        kernel.setArg(idx++, openCLImage(mFilter.get()));
        if (inputs.size() > 2) {
            kernel.setArg(idx++, openCLImage(inputs[2]));
        }
        kernel.setArg(idx++, openCLImage(outputs[0]));
        kernel.setArg(idx++, sizeof(inputImageShape), inputImageShape);
        kernel.setArg(idx++, UP_DIV(inputChannel, 4));
        kernel.setArg(idx++, sizeof(outputImageShape), outputImageShape);
        kernel.setArg(idx++, sizeof(kernelShape), kernelShape);
        kernel.setArg(idx++, sizeof(strideShape), strideShape);
        kernel.setArg(idx++, sizeof(paddingShape), paddingShape);
        kernel.setArg(idx++, sizeof(dilationShape), dilationShape);
        kernel.setArg(idx++, UP_DIV(width, 4));

        std::vector<uint32_t> lws = {runtime->deviceComputeUnits() * 2, 4, 1};
        for (int i = 0; i < 2; ++i) {
            gws[i] = ROUND_UP(gws[i], std::max((uint32_t)1, lws[i]));
        }

        mUnits[2].kernel = kernel;
        mUnits[2].localWorkSize = {lws[0], lws[1]};
        mUnits[2].globalWorkSize = {gws[0], gws[1]};
    }

    return NO_ERROR;
}

}
}
