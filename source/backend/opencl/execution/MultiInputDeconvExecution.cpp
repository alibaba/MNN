//
//  MultiInputDeconvExecution.cpp
//  MNN
//
//  Created by MNN on 2019/10/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "core/TensorUtils.hpp"
#include "backend/opencl/core/OpenCLBackend.hpp"
#include "backend/opencl/execution/MultiInputDeconvExecution.hpp"

namespace MNN {
namespace OpenCL {

MultiInputDeconvExecution::MultiInputDeconvExecution(const MNN::Op *op, Backend *backend) : CommonExecution(backend) {
    auto common = op->main_as_Convolution2D()->common();

    mStrides = {common->strideY(), common->strideX()};
    MNN_ASSERT(mStrides[0] > 0 && mStrides[1] > 0);

    mDilations = {common->dilateY(), common->dilateX()};
    mPaddings = {
        (common->kernelY() - 1 - common->padY()) * 2,
        (common->kernelX() - 1 - common->padX()) * 2
    };
    if (common->padMode() == PadMode_VALID) {
        mPaddings[0] = mPaddings[1] = 0;
    }
}

MultiInputDeconvExecution::~MultiInputDeconvExecution() {
    // do nothing
}

ErrorCode MultiInputDeconvExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mUnits.clear();
    mUnits.resize(4);

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

    const int weightSize = inputs[1]->elementSize();
    auto bufferPool = openclBackend->getBufferPool();
    auto rawBufferPtr = bufferPool->alloc(weightSize * sizeof(float), false);
    if (rawBufferPtr == nullptr) {
        return OUT_OF_MEMORY;
    }
    auto bufferPtr = bufferPool->alloc(weightSize * sizeof(float), false);
    if (bufferPtr == nullptr) {
        bufferPool->recycle(rawBufferPtr, false);
        return OUT_OF_MEMORY;
    }
    mFilter.reset(Tensor::createDevice<float>({1, UP_DIV(outputChannel, 4) * kernelY * kernelX, 1, inputChannel * 4}));
    bool succ = openclBackend->onAcquireBuffer(mFilter.get(), Backend::DYNAMIC);
    bufferPool->recycle(rawBufferPtr, false);
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
        kernel.setArg(2, *rawBufferPtr);
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

    // convert kernel from IOHW to OIHW, similar to DeconvExecution.cpp
    {
        cl::Kernel kernel = runtime->buildKernel("deconv_2d", "iohw2oihw", {});
        kernel.setArg(0, *rawBufferPtr);
        kernel.setArg(1, *bufferPtr);
        kernel.setArg(2, kernelY * kernelX);
        kernel.setArg(3, inputChannel);
        kernel.setArg(4, outputChannel);

        mUnits[1].kernel = kernel;
        mUnits[1].localWorkSize = cl::NullRange;
        mUnits[1].globalWorkSize = {
            static_cast<uint32_t>(inputChannel),
            static_cast<uint32_t>(outputChannel)
        };
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

        mUnits[2].kernel = kernel;
        mUnits[2].localWorkSize = {lws[0], lws[1]};
        mUnits[2].globalWorkSize = {gws[0], gws[1]};
    }

    {
        std::vector<uint32_t> gws = {
            static_cast<uint32_t>(UP_DIV(outputChannel, 4)),
            static_cast<uint32_t>(width),
            static_cast<uint32_t>(height * batch)
        };
        int inputImageShape[] = {inputHeight, inputWidth};
        int outputImageShape[] = {height, width};
        int strideShape[] = {mStrides[0], mStrides[1]};
        int paddingShape[] = {UP_DIV(mPaddings[0], 2), UP_DIV(mPaddings[1], 2)};
        int alignShape[] = {mStrides[0] - 1 - paddingShape[0], mStrides[1] - 1 - paddingShape[1]};
        std::set<std::string> buildOptions;
        if (inputs.size() > 2) {
            buildOptions.emplace("-DBIAS");
        }
        auto kernel = runtime->buildKernel("deconv_2d", "deconv_2d", buildOptions);
        int index = 0;
        kernel.setArg(index++, gws[0]);
        kernel.setArg(index++, gws[1]);
        kernel.setArg(index++, gws[2]);
        kernel.setArg(index++, openCLImage(inputs[0]));
        kernel.setArg(index++, openCLImage(mFilter.get()));
        if (inputs.size() > 2) {
            kernel.setArg(index++, openCLImage(inputs[2]));
        }
        kernel.setArg(index++, openCLImage(outputs[0]));
        kernel.setArg(index++, sizeof(inputImageShape), inputImageShape);
        kernel.setArg(index++, sizeof(outputImageShape), outputImageShape);
        kernel.setArg(index++, sizeof(strideShape), strideShape);
        kernel.setArg(index++, sizeof(alignShape), alignShape);
        kernel.setArg(index++, sizeof(paddingShape), paddingShape);
        kernel.setArg(index++, sizeof(kernelShape), kernelShape);
        kernel.setArg(index++, static_cast<int32_t>(kernelX * kernelY));
        kernel.setArg(index++, static_cast<int32_t>(UP_DIV(inputChannel, 4)));
        kernel.setArg(index++, static_cast<int32_t>(UP_DIV(outputChannel, 4)));

        const uint32_t maxWorkGroupSize = runtime->getMaxWorkGroupSize(kernel);
        auto lws = localWS3DDefault(gws, maxWorkGroupSize, runtime);
        for (size_t i = 0; i < 3; ++i) {
            gws[i] = ROUND_UP(gws[i], std::max((uint32_t)1, lws[i]));
        }

        mUnits[3].kernel = kernel;
        mUnits[3].localWorkSize = {lws[0], lws[1], lws[2]};
        mUnits[3].globalWorkSize = {gws[0], gws[1], gws[2]};
    }

    return NO_ERROR;
}

}
}
