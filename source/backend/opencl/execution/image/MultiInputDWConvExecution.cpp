//
//  MultiInputDWConvExecution.cpp
//  MNN
//
//  Created by MNN on 2019/10/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string>
#include <vector>
#include "core/TensorUtils.hpp"
#include "backend/opencl/core/OpenCLBackend.hpp"
#include "backend/opencl/execution/image/MultiInputDWConvExecution.hpp"

namespace MNN {
namespace OpenCL {

MultiInputDWConvExecution::MultiInputDWConvExecution(const MNN::Op *op, Backend *backend) : CommonExecution(backend, op) {
    auto common = op->main_as_Convolution2D()->common();
    mPadMode = common->padMode();
    mStrides = {common->strideY(), common->strideX()};
    mDilations = {common->dilateY(), common->dilateX()};
    if (mPadMode != PadMode_SAME) {
        mPaddings = {common->padY() * 2, common->padX() * 2};
    }
    isRelu = common->relu();
    isRelu6 = common->relu6();
}

MultiInputDWConvExecution::~MultiInputDWConvExecution() {
    // do nothing
}

ErrorCode MultiInputDWConvExecution::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
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
    
    mFilter.reset(Tensor::createDevice<float>({1, UP_DIV(outputChannel, 4), 1, 4 * kernelY * kernelX}));
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

        auto kernelW = runtime->buildKernel("buffer_to_image", kernelName, {}, inputs[1], inputs[1]);
        auto kernel = kernelW->get();
        cl_int ret = CL_SUCCESS;
        ret |= kernel.setArg(0, gws[0]);
        ret |= kernel.setArg(1, gws[1]);
        ret |= kernel.setArg(2, *bufferPtr);
        ret |= kernel.setArg(3, shape[1]);
        ret |= kernel.setArg(4, shape[2]);
        ret |= kernel.setArg(5, shape[3]);
        ret |= kernel.setArg(6, openCLImage(inputs[1]));
        MNN_CHECK_CL_SUCCESS(ret, "setArg MultiInputDWConvExecution transform input");

        const uint32_t maxWorkGroupSize = runtime->getMaxWorkGroupSize(kernelW);
        std::vector<uint32_t> lws = {16, std::max((uint32_t)1, maxWorkGroupSize / 16)};
        for (size_t i = 0; i < lws.size(); ++i) {
            gws[i] = ROUND_UP(gws[i], lws[i]);
        }

        mUnits[0].kernel = kernelW;
        mUnits[0].localWorkSize = {lws[0], lws[1]};
        mUnits[0].globalWorkSize = {gws[0], gws[1]};
        openclBackend->recordKernel2d(mUnits[0].kernel, gws, lws);
    }
    
    
    
    // transform kernel from original form (maybe NCHW or NHWC) to filter format
    {
        std::vector<int> filterShape{1, outputChannel, kernelY, kernelX};
        std::shared_ptr<Tensor> filterBuffer(Tensor::createDevice<float>(filterShape));
        filterBuffer->buffer().device = (uint64_t)(bufferPtr);
        
        auto buffer = filterBuffer.get();
        auto image =mFilter.get();
        auto formattedBufferShape = tensorShapeFormat(filterBuffer.get());
       std::vector<size_t> imageShape;
       getImageShape(formattedBufferShape, MNN::OpenCL::DW_CONV2D_FILTER, &imageShape);

       uint32_t gws[2] = {static_cast<uint32_t>(imageShape[0]), static_cast<uint32_t>(imageShape[1])};

       std::string kernelName = "dw_filter_buffer_to_image";


        std::set<std::string> buildOptions;
        auto kernelW = runtime->buildKernel("buffer_to_image", kernelName, buildOptions, buffer, image);
        auto kernel = kernelW->get();

       uint32_t idx = 0;
       cl_int ret = CL_SUCCESS;
       ret |= kernel.setArg(idx++, gws[0]);
       ret |= kernel.setArg(idx++, gws[1]);
       ret |= kernel.setArg(idx++, openCLBuffer(buffer));

       const int heightWidthSumSize = buffer->buffer().dim[2].extent * buffer->buffer().dim[3].extent;
       int kernelShape[4] = {buffer->buffer().dim[0].extent, buffer->buffer().dim[1].extent, buffer->buffer().dim[2].extent, buffer->buffer().dim[3].extent};
       ret |= kernel.setArg(idx++, sizeof(kernelShape),kernelShape);
       ret |= kernel.setArg(idx++, static_cast<uint32_t>(heightWidthSumSize));
       ret |= kernel.setArg(idx++, openCLImage(image));
       MNN_CHECK_CL_SUCCESS(ret, "setArg MultiInputDWConvExecution transform kernel");

    
        const uint32_t maxWorkGroupSize = runtime->getMaxWorkGroupSize(kernelW);
        std::vector<uint32_t> lws = {16, std::max((uint32_t)1, maxWorkGroupSize / 16)};
        for (size_t i = 0; i < lws.size(); ++i) {
            gws[i] = ROUND_UP(gws[i], lws[i]);
        }

        mUnits[1].kernel = kernelW;
        mUnits[1].localWorkSize = {lws[0], lws[1]};
        mUnits[1].globalWorkSize = {gws[0], gws[1]};
        openclBackend->recordKernel2d(mUnits[1].kernel, {gws[0], gws[1]}, {lws[0], lws[1]});
    }

    {
        std::vector<int> inputShape  = tensorShapeFormat(inputs[0]);
        std::vector<int> outputShape = tensorShapeFormat(outputs[0]);

        std::vector<uint32_t> gws = {static_cast<uint32_t>(UP_DIV(outputShape.at(3), 4) * UP_DIV(outputShape.at(2), 4)),
                           static_cast<uint32_t>(outputShape.at(0) * outputShape.at(1))};


        const int outputHeight = outputShape.at(1);
        const int outputWidth  = outputShape.at(2);

        const int inputHeight   = inputShape.at(1);
        const int inputWidth    = inputShape.at(2);
        const int inputChannels = inputShape.at(3);

        const int inputChannelBlocks = UP_DIV(inputChannels, 4);
        const int filterHeight       = kernelY;
        const int filterWidth        = kernelX;
        uint32_t idx                 = 0;

        int inputImageShape[2]  = {inputHeight, inputWidth};
        int outputImageShape[2] = {outputHeight, outputWidth};
        int strideShape[2]      = {mStrides[0], mStrides[1]};
        int paddingShape[2]     = {mPaddings[0] / 2, mPaddings[1] / 2};
        int kernelShape[2]      = {filterHeight, filterWidth};
        int dilationShape[2]    = {mDilations[0], mDilations[1]};

        std::set<std::string> buildOptions;
        std::string kernelName = "depthwise_conv2d";
        if (mStrides[0] == 1 && mStrides[1] == 1 &&
            mDilations[0] == 1 && mDilations[1] == 1) {
            kernelName = "depthwise_conv2d_s1";
        }

        if (isRelu == true) {
            buildOptions.emplace("-DRELU");
        } else if (isRelu6 == true) {
            buildOptions.emplace("-DRELU6");
        }
        if(inputs.size() == 2) {
            buildOptions.emplace("-DNO_BIAS");
        }

        auto kernelW = runtime->buildKernel("depthwise_conv2d", kernelName, buildOptions);
        auto kernel = kernelW->get();
        cl_int ret = CL_SUCCESS;
        ret |= kernel.setArg(idx++, gws[0]);
        ret |= kernel.setArg(idx++, gws[1]);
        ret |= kernel.setArg(idx++, openCLImage(inputs[0]));
        ret |= kernel.setArg(idx++, openCLImage(mFilter.get()));
        if (inputs.size() > 2) {
            ret |= kernel.setArg(idx++, openCLImage(inputs[2]));
        }
        ret |= kernel.setArg(idx++, openCLImage(outputs[0]));
        ret |= kernel.setArg(idx++, sizeof(inputImageShape), inputImageShape);
        ret |= kernel.setArg(idx++, static_cast<int>(inputChannelBlocks));
        ret |= kernel.setArg(idx++, sizeof(outputImageShape), outputImageShape);
        ret |= kernel.setArg(idx++, sizeof(kernelShape), kernelShape);
        ret |= kernel.setArg(idx++, sizeof(paddingShape), paddingShape);
        if (mStrides[0] != 1 || mStrides[1] != 1 || mDilations[0] != 1 || mDilations[1] != 1) {
            ret |= kernel.setArg(idx++, sizeof(dilationShape), dilationShape);
            ret |= kernel.setArg(idx++, sizeof(strideShape), strideShape);
        }
        MNN_CHECK_CL_SUCCESS(ret, "setArg MultiInputDWConvExecution");

        mUnits[2].kernel = kernelW;
        mUnits[2].localWorkSize = {1, 1};
        mUnits[2].globalWorkSize = {gws[0], gws[1]};
        
        openclBackend->recordKernel2d(mUnits[2].kernel, gws, {1, 1});
    }

    return NO_ERROR;
}

}
}
