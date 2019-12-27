//
//  ReshapeExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/execution/ReshapeExecution.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {
namespace OpenCL {

ReshapeExecution::ReshapeExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend)
    : Execution(backend) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start ReshapeExecution init !\n");
#endif
    mDimType = op->main_as_Reshape()->dimType();
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
#ifdef LOG_VERBOSE
    MNN_PRINT("end ReshapeExecution init !\n");
#endif
}

ErrorCode ReshapeExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input = inputs[0];
    auto output = outputs[0];
#ifdef LOG_VERBOSE
    MNN_PRINT("mDimType = %d , %d\n", mDimType, TensorUtils::getDescribe(input)->dimensionFormat);
    MNN_PRINT("%d, %d, %d, %d -> %d, %d, %d, %d\n", input->batch(), input->width(), input->height(), input->channel(), output->batch(), output->width(),
              output->height(), output->channel());
#endif
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    std::string mImageToBufferKernelname;
    std::string mBufferToImageKernelname;
    {
        auto inputFormat  = TensorUtils::getDescribe(input)->dimensionFormat;
        std::map<MNN_DATA_FORMAT, std::string> formatMap = {
            {MNN_DATA_FORMAT_NCHW, "image_to_nchw_buffer"},
            {MNN_DATA_FORMAT_NHWC, "image_to_nhwc_buffer"},
        };
        if(inputFormat == MNN_DATA_FORMAT_NC4HW4){
            mImageToBufferKernelname = formatMap[mDimType];
        }else{
            mImageToBufferKernelname = formatMap[inputFormat];
        }
    }

    {
        auto outputFormat = TensorUtils::getDescribe(output)->dimensionFormat;
        std::map<MNN_DATA_FORMAT, std::string> formatMap = {
            {MNN_DATA_FORMAT_NCHW, "nchw_buffer_to_image"},
            {MNN_DATA_FORMAT_NHWC, "nhwc_buffer_to_image"},
        };
        if(outputFormat == MNN_DATA_FORMAT_NC4HW4){
            mBufferToImageKernelname = formatMap[mDimType];
        }else{
            mBufferToImageKernelname = formatMap[outputFormat];
        }
    }

    if (mImageToBufferKernel.get() == nullptr) {
        std::set<std::string> buildOptions;
        mImageToBufferKernel = runtime->buildKernel("buffer_to_image", mImageToBufferKernelname, buildOptions);
    }

    if (mBufferToImageKernel.get() == nullptr) {
        std::set<std::string> buildOptions;
        mBufferToImageKernel = runtime->buildKernel("buffer_to_image", mBufferToImageKernelname, buildOptions);
    }

    auto bufferPool = mOpenCLBackend->getBufferPool();
    mInterBuffer    = bufferPool->alloc(input->size());
    bufferPool->recycle(mInterBuffer);

    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);

    uint32_t inputGlobalWorkSize[2]  = {static_cast<uint32_t>(UP_DIV(inputShape[3], 4) * inputShape[2]),
                                       static_cast<uint32_t>(inputShape[0] * inputShape[1])};
    uint32_t outputGlobalWorkSize[2] = {static_cast<uint32_t>(UP_DIV(outputShape[3], 4) * outputShape[2]),
                                        static_cast<uint32_t>(outputShape[0] * outputShape[1])};

    // image->buffer
    {
        uint32_t idx = 0;
        mImageToBufferKernel.setArg(idx++, inputGlobalWorkSize[0]);
        mImageToBufferKernel.setArg(idx++, inputGlobalWorkSize[1]);
        mImageToBufferKernel.setArg(idx++, *mInterBuffer);
        mImageToBufferKernel.setArg(idx++, static_cast<uint32_t>(inputShape[1]));
        mImageToBufferKernel.setArg(idx++, static_cast<uint32_t>(inputShape[2]));
        mImageToBufferKernel.setArg(idx++, static_cast<uint32_t>(inputShape[3]));
        mImageToBufferKernel.setArg(idx++, openCLImage(inputs[0]));
        const uint32_t maxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mImageToBufferKernel));
        mLocalWorkSize                  = {16, maxWorkGroupSize / 16};
        for (size_t i = 0; i < mLocalWorkSize.size(); ++i) {
            mImageToBufferRoundUpGWS[i] = ROUND_UP(inputGlobalWorkSize[i], std::max((uint32_t)1, mLocalWorkSize[i]));
        }
    }

    // buffer->image
    {
        uint32_t idx = 0;
        mBufferToImageKernel.setArg(idx++, outputGlobalWorkSize[0]);
        mBufferToImageKernel.setArg(idx++, outputGlobalWorkSize[1]);
        mBufferToImageKernel.setArg(idx++, *mInterBuffer);
        mBufferToImageKernel.setArg(idx++, static_cast<uint32_t>(outputShape[1]));
        mBufferToImageKernel.setArg(idx++, static_cast<uint32_t>(outputShape[2]));
        mBufferToImageKernel.setArg(idx++, static_cast<uint32_t>(outputShape[3]));
        mBufferToImageKernel.setArg(idx++, openCLImage(outputs[0]));

        for (size_t i = 0; i < mLocalWorkSize.size(); ++i) {
            mBufferToImageRoundUpGWS[i] = ROUND_UP(outputGlobalWorkSize[i], std::max((uint32_t)1, mLocalWorkSize[i]));
        }
    }
    return NO_ERROR;
}

ErrorCode ReshapeExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start ReshapeExecution onExecute !\n");
#endif

    auto runtime = mOpenCLBackend->getOpenCLRuntime();

    cl_int error;
    error = runtime->commandQueue().enqueueNDRangeKernel(
        mImageToBufferKernel, cl::NullRange, cl::NDRange(mImageToBufferRoundUpGWS[0], mImageToBufferRoundUpGWS[1]),
        cl::NDRange(mLocalWorkSize[0], mLocalWorkSize[1]), nullptr, nullptr);
    MNN_CHECK_CL_SUCCESS(error);

    error = runtime->commandQueue().enqueueNDRangeKernel(
        mBufferToImageKernel, cl::NullRange, cl::NDRange(mBufferToImageRoundUpGWS[0], mBufferToImageRoundUpGWS[1]),
        cl::NDRange(mLocalWorkSize[0], mLocalWorkSize[1]), nullptr, nullptr);
    MNN_CHECK_CL_SUCCESS(error);

#ifdef LOG_VERBOSE
    MNN_PRINT("end ReshapeExecution onExecute !\n");
#endif
    return NO_ERROR;
}


class ReshapeCreator : public OpenCLBackend::Creator {
public:
    virtual ~ReshapeCreator() = default;
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        if(inputs[0]->dimensions() == 3 || outputs[0]->dimensions() == 3){
            MNN_PRINT("reshape not support dimensions == 3 \n");
            return nullptr;
        }
        return new ReshapeExecution(inputs, op, backend);
    }
};

OpenCLCreatorRegister<ReshapeCreator> __reshape_op(OpType_Reshape);

} // namespace OpenCL
} // namespace MNN
