//
//  MultiInputDWDeconvExecution.cpp
//  MNN
//
//  Created by MNN on 2019/10/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "core/TensorUtils.hpp"
#include "backend/opencl/core/OpenCLBackend.hpp"
#include "backend/opencl/execution/image/MultiInputDWDeconvExecution.hpp"

namespace MNN {
namespace OpenCL {

MultiInputDWDeconvExecution::MultiInputDWDeconvExecution(const MNN::Op *op, Backend *backend) : CommonExecution(backend, op) {
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
    
    isRelu = common->relu();
    isRelu6 = common->relu6();
}

MultiInputDWDeconvExecution::~MultiInputDWDeconvExecution() {
    // do nothing
}

ErrorCode MultiInputDWDeconvExecution::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
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
    
    mFilter.reset(Tensor::createDevice<float>({1, UP_DIV(outputChannel, 4), 1, 4 * kernelY * kernelX}));
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

        auto kernelW = runtime->buildKernel("buffer_to_image", kernelName, {}, inputs[1], inputs[1]);
        auto kernel = kernelW->get();
        cl_int ret = CL_SUCCESS;
        ret |= kernel.setArg(0, gws[0]);
        ret |= kernel.setArg(1, gws[1]);
        ret |= kernel.setArg(2, *rawBufferPtr);
        ret |= kernel.setArg(3, shape[1]);
        ret |= kernel.setArg(4, shape[2]);
        ret |= kernel.setArg(5, shape[3]);
        ret |= kernel.setArg(6, openCLImage(inputs[1]));
        MNN_CHECK_CL_SUCCESS(ret, "setArg MultiInputDWDeconvExecution transform input");

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
    
    // convert kernel from IOHW to OIHW, similar to DeconvExecution.cpp
    {
        auto shape = tensorShapeFormat(inputs[1]);
        
        auto kernelW = runtime->buildKernel("deconv_2d", "iohw2oihw", {});
        auto kernel = kernelW->get();
        cl_int ret = CL_SUCCESS;
        ret |= kernel.setArg(0, *rawBufferPtr);
        ret |= kernel.setArg(1, *bufferPtr);
        ret |= kernel.setArg(2, kernelY * kernelX);
        ret |= kernel.setArg(3, shape[3]);
        ret |= kernel.setArg(4, shape[0]);
        MNN_CHECK_CL_SUCCESS(ret, "setArg MultiInputDWDeconvExecution transform kernel");

        mUnits[1].kernel = kernelW;
        mUnits[1].localWorkSize = cl::NullRange;
        mUnits[1].globalWorkSize = {
            static_cast<uint32_t>(shape[3]),
            static_cast<uint32_t>(shape[0])
        };
        openclBackend->recordKernel2d(mUnits[1].kernel, {
            static_cast<uint32_t>(shape[3]),
            static_cast<uint32_t>(shape[0])
        }, {0, 0});
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
        MNN_CHECK_CL_SUCCESS(ret, "setArg MultiInputDWDeconvExecution transfor kernel format");

        const int heightWidthSumSize = buffer->buffer().dim[2].extent * buffer->buffer().dim[3].extent;
        int kernelShape[4] = {buffer->buffer().dim[0].extent, buffer->buffer().dim[1].extent, buffer->buffer().dim[2].extent, buffer->buffer().dim[3].extent};
        ret |= kernel.setArg(idx++, sizeof(kernelShape),kernelShape);
        ret |= kernel.setArg(idx++, static_cast<uint32_t>(heightWidthSumSize));
        ret |= kernel.setArg(idx++, openCLImage(image));
    
    
        const uint32_t maxWorkGroupSize = runtime->getMaxWorkGroupSize(kernelW);
        std::vector<uint32_t> lws = {16, std::max((uint32_t)1, maxWorkGroupSize / 16)};
        for (size_t i = 0; i < lws.size(); ++i) {
            gws[i] = ROUND_UP(gws[i], lws[i]);
        }

        mUnits[2].kernel = kernelW;
        mUnits[2].localWorkSize = {lws[0], lws[1]};
        mUnits[2].globalWorkSize = {gws[0], gws[1]};
        openclBackend->recordKernel2d(mUnits[2].kernel, {gws[0], gws[1]}, {lws[0], lws[1]});
    }

    {
        std::vector<int> inputShape  = tensorShapeFormat(inputs[0]);
        std::vector<int> outputShape = tensorShapeFormat(outputs[0]);

        const int outputBatch    = outputShape.at(0);
        const int outputHeight   = outputShape.at(1);
        const int outputWidth    = outputShape.at(2);
        const int outputChannels = outputShape.at(3);

        const int inputHeight   = inputShape.at(1);
        const int inputWidth    = inputShape.at(2);
        const int inputChannels = inputShape.at(3);

        const int strideHeight = mStrides[0];
        const int strideWidth  = mStrides[1];

        const int channelBlocks = UP_DIV(outputChannels, 4);

        const int paddingHeight = UP_DIV(mPaddings[0], 2);
        const int paddingWidth  = UP_DIV(mPaddings[1], 2);

        const int alignHeight = strideHeight - 1 - paddingHeight;
        const int alignWidth  = strideWidth - 1 - paddingWidth;

        const int filterHeight = kernelY;
        const int filterWidth  = kernelX;
        const int kernelSize   = filterHeight * filterWidth;

        std::vector<uint32_t> gws  = {static_cast<uint32_t>(channelBlocks), static_cast<uint32_t>(outputWidth),
                static_cast<uint32_t>(outputHeight * outputBatch)};

        int inputImageShape[2]  = {inputHeight, inputWidth};
        int outputImageShape[2] = {outputHeight, outputWidth};
        int strideShape[2]      = {strideHeight, strideWidth};
        int paddingShape[2]     = {paddingHeight, paddingWidth};
        int alignShape[2]       = {alignHeight, alignWidth};
        int kernelShape[2]      = {filterHeight, filterWidth};
        
        std::set<std::string> buildOptions;
        
        std::string kernelName = "depthwise_deconv2d";
        if (isRelu == true) {
            buildOptions.emplace("-DRELU");
        } else if (isRelu6 == true) {
            buildOptions.emplace("-DRELU6");
        }
        if(inputs.size() == 2) {
            buildOptions.emplace("-DNO_BIAS");
        }

        auto kernelW = runtime->buildKernel("depthwise_deconv2d", kernelName, buildOptions);
        auto kernel = kernelW->get();
        int index = 0;
        
        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;
        ret |= kernel.setArg(idx++, gws[0]);
        ret |= kernel.setArg(idx++, gws[1]);
        ret |= kernel.setArg(idx++, gws[2]);

        ret |= kernel.setArg(idx++, openCLImage(inputs[0]));
        ret |= kernel.setArg(idx++, openCLImage(mFilter.get()));
        if(inputs.size() > 2) {
            ret |= kernel.setArg(idx++, openCLImage(inputs[2]));
        }
        ret |= kernel.setArg(idx++, openCLImage(outputs[0]));
        ret |= kernel.setArg(idx++, sizeof(inputImageShape), inputImageShape);
        ret |= kernel.setArg(idx++, sizeof(outputImageShape), outputImageShape);
        ret |= kernel.setArg(idx++, sizeof(strideShape), strideShape);
        ret |= kernel.setArg(idx++, sizeof(alignShape), alignShape);
        ret |= kernel.setArg(idx++, sizeof(paddingShape), paddingShape);
        ret |= kernel.setArg(idx++, sizeof(kernelShape), kernelShape);
        ret |= kernel.setArg(idx++, static_cast<int32_t>(kernelSize));
        ret |= kernel.setArg(idx++, static_cast<int32_t>(channelBlocks));
        MNN_CHECK_CL_SUCCESS(ret, "setArg MultiInputDWDeconvExecution");

        const uint32_t maxWorkGroupSize = runtime->getMaxWorkGroupSize(kernelW);
        std::string name = "depthwiseDeconv";
        auto lws = localWS3DDefault(gws, maxWorkGroupSize, runtime, name, kernelW).first;
        for (size_t i = 0; i < 3; ++i) {
            gws[i] = ROUND_UP(gws[i], std::max((uint32_t)1, lws[i]));
        }

        mUnits[3].kernel = kernelW;
        mUnits[3].localWorkSize = {lws[0], lws[1], lws[2]};
        mUnits[3].globalWorkSize = {gws[0], gws[1], gws[2]};
        openclBackend->recordKernel3d(mUnits[3].kernel, gws, lws);
    }
   
    return NO_ERROR;
}

}
}
