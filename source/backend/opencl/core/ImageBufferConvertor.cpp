//
//  ImageBufferConvertor.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ImageBufferConvertor.hpp"

namespace MNN {
namespace OpenCL {
bool convertNCHWBufferToImage(const Tensor *input, Tensor *output, cl::Kernel &bufferToImageKernel,
                              OpenCLRuntime *runtime, bool needWait) {
    std::vector<int> outputShape = tensorShapeFormat(input);

    uint32_t outputGlobalWorkSize[2] = {static_cast<uint32_t>(UP_DIV(outputShape[3], 4) * outputShape[2]),
                                        static_cast<uint32_t>(outputShape[0] * outputShape[1])};
    if (bufferToImageKernel.get() == nullptr) {
        std::set<std::string> buildOptions;
        bufferToImageKernel = runtime->buildKernel("buffer_to_image", "nchw_buffer_to_image", buildOptions);
    }
    uint32_t idx = 0;
    bufferToImageKernel.setArg(idx++, outputGlobalWorkSize[0]);
    bufferToImageKernel.setArg(idx++, outputGlobalWorkSize[1]);
    bufferToImageKernel.setArg(idx++, openCLBuffer(input));
    bufferToImageKernel.setArg(idx++, static_cast<uint32_t>(outputShape[1]));
    bufferToImageKernel.setArg(idx++, static_cast<uint32_t>(outputShape[2]));
    bufferToImageKernel.setArg(idx++, static_cast<uint32_t>(outputShape[3]));
    bufferToImageKernel.setArg(idx++, openCLImage(output));

    const uint32_t maxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(bufferToImageKernel));
    const std::vector<uint32_t> lws = {16, std::max((uint32_t)1, maxWorkGroupSize / 16)};
    cl::Event event;
    cl_int error;
    std::vector<uint32_t> roundUpGroupWorkSize(lws.size());
    for (size_t i = 0; i < lws.size(); ++i) {
        roundUpGroupWorkSize[i] = ROUND_UP(outputGlobalWorkSize[i], lws[i]);
    }
    error = runtime->commandQueue().enqueueNDRangeKernel(bufferToImageKernel, cl::NullRange,
                                                         cl::NDRange(roundUpGroupWorkSize[0], roundUpGroupWorkSize[1]),
                                                         cl::NDRange(lws[0], lws[1]), nullptr, &event);
    MNN_CHECK_CL_SUCCESS(error);
    if (true == needWait) {
        event.wait();
    }
    return true;
}

bool convertNHWCBufferToImage(const Tensor *input, Tensor *output, cl::Kernel &bufferToImageKernel,
                              OpenCLRuntime *runtime, bool needWait) {
    std::vector<int> outputShape = tensorShapeFormat(input);

    uint32_t outputGlobalWorkSize[2] = {static_cast<uint32_t>(UP_DIV(outputShape[3], 4) * outputShape[2]),
                                        static_cast<uint32_t>(outputShape[0] * outputShape[1])};
    if (bufferToImageKernel.get() == nullptr) {
        std::set<std::string> buildOptions;
        bufferToImageKernel = runtime->buildKernel("buffer_to_image", "nhwc_buffer_to_image", buildOptions);
    }
    uint32_t idx = 0;
    bufferToImageKernel.setArg(idx++, outputGlobalWorkSize[0]);
    bufferToImageKernel.setArg(idx++, outputGlobalWorkSize[1]);
    bufferToImageKernel.setArg(idx++, openCLBuffer(input));
    bufferToImageKernel.setArg(idx++, static_cast<uint32_t>(outputShape[1]));
    bufferToImageKernel.setArg(idx++, static_cast<uint32_t>(outputShape[2]));
    bufferToImageKernel.setArg(idx++, static_cast<uint32_t>(outputShape[3]));
    bufferToImageKernel.setArg(idx++, openCLImage(output));

    const uint32_t maxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(bufferToImageKernel));
    const std::vector<uint32_t> lws = {16, std::max((uint32_t)1, maxWorkGroupSize / 16)};
    cl::Event event;
    cl_int error;
    std::vector<uint32_t> roundUpGroupWorkSize(lws.size());
    for (size_t i = 0; i < lws.size(); ++i) {
        roundUpGroupWorkSize[i] = ROUND_UP(outputGlobalWorkSize[i], lws[i]);
    }
    error = runtime->commandQueue().enqueueNDRangeKernel(bufferToImageKernel, cl::NullRange,
                                                         cl::NDRange(roundUpGroupWorkSize[0], roundUpGroupWorkSize[1]),
                                                         cl::NDRange(lws[0], lws[1]), nullptr, &event);
    MNN_CHECK_CL_SUCCESS(error);
    if (true == needWait) {
        event.wait();
    }
    return true;
}

bool convertImageToNCHWBuffer(const Tensor *input, Tensor *output, cl::Kernel &imageToBufferKernel,
                              OpenCLRuntime *runtime, bool needWait) {
    std::vector<int> inputShape = tensorShapeFormat(input);
    uint32_t in_gws[2]          = {static_cast<uint32_t>(UP_DIV(inputShape[3], 4) * inputShape[2]),
                          static_cast<uint32_t>(inputShape[0] * inputShape[1])};

    if (imageToBufferKernel.get() == nullptr) {
        std::set<std::string> buildOptions;
        imageToBufferKernel = runtime->buildKernel("buffer_to_image", "image_to_nchw_buffer", buildOptions);
    }

    uint32_t idx = 0;
    imageToBufferKernel.setArg(idx++, in_gws[0]);
    imageToBufferKernel.setArg(idx++, in_gws[1]);
    imageToBufferKernel.setArg(idx++, openCLBuffer(output));
    imageToBufferKernel.setArg(idx++, static_cast<uint32_t>(inputShape[1]));
    imageToBufferKernel.setArg(idx++, static_cast<uint32_t>(inputShape[2]));
    imageToBufferKernel.setArg(idx++, static_cast<uint32_t>(inputShape[3]));
    imageToBufferKernel.setArg(idx++, openCLImage(input));
    const uint32_t maxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(imageToBufferKernel));
    const std::vector<uint32_t> lws = {16, std::max((uint32_t)1, maxWorkGroupSize / 16)};
    cl::Event event;
    cl_int error;
    std::vector<uint32_t> roundUpGroupWorkSize(lws.size());
    for (size_t i = 0; i < lws.size(); ++i) {
        roundUpGroupWorkSize[i] = ROUND_UP(in_gws[i], lws[i]);
    }
    error = runtime->commandQueue().enqueueNDRangeKernel(imageToBufferKernel, cl::NullRange,
                                                         cl::NDRange(roundUpGroupWorkSize[0], roundUpGroupWorkSize[1]),
                                                         cl::NDRange(lws[0], lws[1]), nullptr, &event);
    MNN_CHECK_CL_SUCCESS(error);

    if (true == needWait) {
        event.wait();
    }
    return true;
}

bool convertNC4HW4BufferToImage(const Tensor *input, Tensor *output, cl::Kernel &bufferToImageKernel,
                                OpenCLRuntime *runtime, bool needWait) {
    std::vector<int> outputShape = tensorShapeFormat(input);

    uint32_t outputGlobalWorkSize[2] = {static_cast<uint32_t>(UP_DIV(outputShape[3], 4) * outputShape[2]),
                                        static_cast<uint32_t>(outputShape[0] * outputShape[1])};
    if (bufferToImageKernel.get() == nullptr) {
        std::set<std::string> buildOptions;
        bufferToImageKernel = runtime->buildKernel("buffer_to_image", "nc4hw4_buffer_to_image", buildOptions);
    }
    int channelUp4 = ((outputShape[3] + 3) / 4) * 4;
    uint32_t idx   = 0;
    int outputImageShape[2] = {outputShape[1], outputShape[2]};
    bufferToImageKernel.setArg(idx++, outputGlobalWorkSize[0]);
    bufferToImageKernel.setArg(idx++, outputGlobalWorkSize[1]);
    bufferToImageKernel.setArg(idx++, openCLBuffer(input));
    bufferToImageKernel.setArg(idx++, sizeof(outputImageShape), outputImageShape);
    bufferToImageKernel.setArg(idx++, channelUp4);
    bufferToImageKernel.setArg(idx++, openCLImage(output));

    const uint32_t maxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(bufferToImageKernel));
    const std::vector<uint32_t> lws = {16, std::max((uint32_t)1, maxWorkGroupSize / 16)};
    cl::Event event;
    cl_int error;
    std::vector<uint32_t> roundUpGroupWorkSize(lws.size());
    for (size_t i = 0; i < lws.size(); ++i) {
        roundUpGroupWorkSize[i] = ROUND_UP(outputGlobalWorkSize[i], lws[i]);
    }
    error = runtime->commandQueue().enqueueNDRangeKernel(bufferToImageKernel, cl::NullRange,
                                                         cl::NDRange(roundUpGroupWorkSize[0], roundUpGroupWorkSize[1]),
                                                         cl::NDRange(lws[0], lws[1]), nullptr, &event);
    MNN_CHECK_CL_SUCCESS(error);
    if (true == needWait) {
        event.wait();
    }
    return true;
}

/**
 * @brief convert image to nc/4hwc%4 buffer.
 * @param input      input tensor.
 * @param output     output tensor.
 * @param bufferToImageKernel    opencl kernel reference.
 * @param runtime    opencl runtime instance pointer.
 * @param needWait   whether need wait opencl complete before return or not, default false.
 * @return true if success, false otherwise.
 */
bool convertImageToNC4HW4Buffer(const Tensor *input, Tensor *output, cl::Kernel &imageToBufferKernel,
                                OpenCLRuntime *runtime, bool needWait) {
    std::vector<int> inputShape = tensorShapeFormat(input);
    uint32_t in_gws[2]          = {static_cast<uint32_t>(UP_DIV(inputShape[3], 4) * inputShape[2]),
                          static_cast<uint32_t>(inputShape[0] * inputShape[1])};

    if (imageToBufferKernel.get() == nullptr) {
        std::set<std::string> buildOptions;
        imageToBufferKernel = runtime->buildKernel("buffer_to_image", "image_to_nc4hw4_buffer", buildOptions);
    }

    int channelUp4 = ((inputShape[3] + 3) / 4) * 4;
    uint32_t idx   = 0;
    int outputImageShape[2] = {inputShape[1], inputShape[2]};
    imageToBufferKernel.setArg(idx++, in_gws[0]);
    imageToBufferKernel.setArg(idx++, in_gws[1]);
    imageToBufferKernel.setArg(idx++, openCLBuffer(output));
    imageToBufferKernel.setArg(idx++, sizeof(outputImageShape), outputImageShape);
    imageToBufferKernel.setArg(idx++, static_cast<uint32_t>(channelUp4));
    imageToBufferKernel.setArg(idx++, openCLImage(input));
    const uint32_t maxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(imageToBufferKernel));
    const std::vector<uint32_t> lws = {16, std::max((uint32_t)1, maxWorkGroupSize / 16)};
    cl::Event event;
    cl_int error;
    std::vector<uint32_t> roundUpGroupWorkSize(lws.size());
    for (size_t i = 0; i < lws.size(); ++i) {
        roundUpGroupWorkSize[i] = ROUND_UP(in_gws[i], lws[i]);
    }
    error = runtime->commandQueue().enqueueNDRangeKernel(imageToBufferKernel, cl::NullRange,
                                                         cl::NDRange(roundUpGroupWorkSize[0], roundUpGroupWorkSize[1]),
                                                         cl::NDRange(lws[0], lws[1]), nullptr, &event);
    MNN_CHECK_CL_SUCCESS(error);

    if (true == needWait) {
        event.wait();
    }
    return true;
}

bool convertImageToNHWCBuffer(const Tensor *input, Tensor *output, cl::Kernel &imageToBufferKernel,
                              OpenCLRuntime *runtime, bool needWait) {
    std::vector<int> inputShape = tensorShapeFormat(input);
    uint32_t in_gws[2]          = {static_cast<uint32_t>(UP_DIV(inputShape[3], 4) * inputShape[2]),
                          static_cast<uint32_t>(inputShape[0] * inputShape[1])};

    if (imageToBufferKernel.get() == nullptr) {
        std::set<std::string> buildOptions;
        imageToBufferKernel = runtime->buildKernel("buffer_to_image", "image_to_nhwc_buffer", buildOptions);
    }

    uint32_t idx = 0;
    imageToBufferKernel.setArg(idx++, in_gws[0]);
    imageToBufferKernel.setArg(idx++, in_gws[1]);
    imageToBufferKernel.setArg(idx++, openCLBuffer(output));
    imageToBufferKernel.setArg(idx++, static_cast<uint32_t>(inputShape[1]));
    imageToBufferKernel.setArg(idx++, static_cast<uint32_t>(inputShape[2]));
    imageToBufferKernel.setArg(idx++, static_cast<uint32_t>(inputShape[3]));
    imageToBufferKernel.setArg(idx++, openCLImage(input));
    const uint32_t maxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(imageToBufferKernel));
    const std::vector<uint32_t> lws = {16, std::max((uint32_t)1, maxWorkGroupSize / 16)};
    cl::Event event;
    cl_int error;
    std::vector<uint32_t> roundUpGroupWorkSize(lws.size());
    for (size_t i = 0; i < lws.size(); ++i) {
        roundUpGroupWorkSize[i] = ROUND_UP(in_gws[i], lws[i]);
    }
    error = runtime->commandQueue().enqueueNDRangeKernel(imageToBufferKernel, cl::NullRange,
                                                         cl::NDRange(roundUpGroupWorkSize[0], roundUpGroupWorkSize[1]),
                                                         cl::NDRange(lws[0], lws[1]), nullptr, &event);
    MNN_CHECK_CL_SUCCESS(error);

    if (true == needWait) {
        event.wait();
    }
    return true;
}
bool ImageBufferConvertor::convertImageToBuffer(const Tensor *image, const OpenCLBufferFormat type, Tensor *buffer,
                                                bool needWait) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start convertImageToBuffer !\n");
#endif
    auto formattedBufferShape = tensorShapeFormat(image);

    auto runtime = mOpenCLRuntime;

    std::string kernelName;
    if (type == NHWC_BUFFER) {
        kernelName = "image_to_nhwc_buffer";
    } else if (type == NCHW_BUFFER) {
        kernelName = "image_to_nchw_buffer";
    } else if (type == CONV2D_FILTER) {
        kernelName = "conv2d_filter_image_to_buffer";
    } else if (type == ARGUMENT) {
        kernelName = "arg_image_to_buffer";
    } else {
        MNN_PRINT("not support such type !!! \n");
    }

    if (mImageToBufferKernel.get() == nullptr || mImageToBufferKernelName != kernelName) {
        mImageToBufferKernelName = kernelName;
        std::set<std::string> buildOptions;

        mImageToBufferKernel = runtime->buildKernel("buffer_to_image", kernelName, buildOptions);
    }

    std::vector<size_t> gws;
    getImageShape(formattedBufferShape, type, &gws);

    uint32_t idx = 0;
    mImageToBufferKernel.setArg(idx++, gws[0]);
    mImageToBufferKernel.setArg(idx++, gws[1]);

    mImageToBufferKernel.setArg(idx++, openCLBuffer(buffer));
    if (type == CONV2D_FILTER) {
        const int channelHeightWidthSumSize =
            buffer->buffer().dim[1].extent * buffer->buffer().dim[2].extent * buffer->buffer().dim[3].extent;
        const int heightWidthSumSize = buffer->buffer().dim[2].extent * buffer->buffer().dim[3].extent;
        int kernelShape[2] = {buffer->buffer().dim[2].extent, buffer->buffer().dim[3].extent};
        mImageToBufferKernel.setArg(idx++, static_cast<uint32_t>(buffer->buffer().dim[0].extent));
        mImageToBufferKernel.setArg(idx++, sizeof(kernelShape), kernelShape);
        mImageToBufferKernel.setArg(idx++, static_cast<uint32_t>(channelHeightWidthSumSize));
        mImageToBufferKernel.setArg(idx++, static_cast<uint32_t>(heightWidthSumSize));
    } else if (type == ARGUMENT) {
        mImageToBufferKernel.setArg(idx++, static_cast<uint32_t>(buffer->buffer().dim[0].extent));
    } else {
        mImageToBufferKernel.setArg(idx++, static_cast<uint32_t>(formattedBufferShape[1]));
        mImageToBufferKernel.setArg(idx++, static_cast<uint32_t>(formattedBufferShape[2]));
        mImageToBufferKernel.setArg(idx++, static_cast<uint32_t>(formattedBufferShape[3]));
    }
    mImageToBufferKernel.setArg(idx++, openCLImage(image));

    const uint32_t maxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mImageToBufferKernel));
    const std::vector<uint32_t> lws = {16, std::max((uint32_t)1, maxWorkGroupSize / 16)};

    cl::Event event;
    cl_int error;

    std::vector<uint32_t> roundUpGroupWorkSize(lws.size());
    for (size_t i = 0; i < lws.size(); ++i) {
        roundUpGroupWorkSize[i] = ROUND_UP(gws[i], lws[i]);
    }

    error = runtime->commandQueue().enqueueNDRangeKernel(mImageToBufferKernel, cl::NullRange,
                                                         cl::NDRange(roundUpGroupWorkSize[0], roundUpGroupWorkSize[1]),
                                                         cl::NDRange(lws[0], lws[1]), nullptr, &event);

    MNN_CHECK_CL_SUCCESS(error);

    if (needWait) {
        event.wait();
    }
#ifdef LOG_VERBOSE
    MNN_PRINT("end convertImageToBuffer !\n");
#endif
    return true;
}

bool ImageBufferConvertor::convertBufferToImage(const Tensor *buffer, const OpenCLBufferFormat type, Tensor *image,
                                                bool needWait) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start convertBufferToImage !\n");
#endif
    auto formattedBufferShape = tensorShapeFormat(buffer);
    std::vector<size_t> imageShape;
    getImageShape(formattedBufferShape, type, &imageShape);

    uint32_t gws[2] = {static_cast<uint32_t>(imageShape[0]), static_cast<uint32_t>(imageShape[1])};

    auto runtime = mOpenCLRuntime;
    std::string kernelName;
    switch (type) {
        case CONV2D_FILTER:
            kernelName = "conv2d_filter_buffer_to_image";
            break;
        case DW_CONV2D_FILTER:
            kernelName = "dw_filter_buffer_to_image";
            break;
        case NHWC_BUFFER:
            kernelName = "nhwc_buffer_to_image";
            break;
        case NCHW_BUFFER:
            kernelName = "nchw_buffer_to_image";
            break;
        case ARGUMENT:
            kernelName = "arg_buffer_to_image";
            break;
        default:
            break;
    }
    if (mBufferToImageKernel.get() == nullptr || mBufferToImageKernelName != kernelName) {
        mBufferToImageKernelName = kernelName;
        std::set<std::string> buildOptions;

        mBufferToImageKernel = runtime->buildKernel("buffer_to_image", kernelName, buildOptions);
    }

    uint32_t idx = 0;
    mBufferToImageKernel.setArg(idx++, gws[0]);
    mBufferToImageKernel.setArg(idx++, gws[1]);

    mBufferToImageKernel.setArg(idx++, openCLBuffer(buffer));

    if (type == CONV2D_FILTER) {
        const int channelHeightWidthSumSize =
            buffer->buffer().dim[1].extent * buffer->buffer().dim[2].extent * buffer->buffer().dim[3].extent;
        const int heightWidthSumSize = buffer->buffer().dim[2].extent * buffer->buffer().dim[3].extent;
        int kernelShape[2] = {buffer->buffer().dim[2].extent, buffer->buffer().dim[3].extent}; 
        mBufferToImageKernel.setArg(idx++, static_cast<uint32_t>(buffer->buffer().dim[0].extent));
        mBufferToImageKernel.setArg(idx++, sizeof(kernelShape),kernelShape);
        mBufferToImageKernel.setArg(idx++, static_cast<uint32_t>(channelHeightWidthSumSize));
        mBufferToImageKernel.setArg(idx++, static_cast<uint32_t>(heightWidthSumSize));
    } else if (type == DW_CONV2D_FILTER) {
        const int heightWidthSumSize = buffer->buffer().dim[2].extent * buffer->buffer().dim[3].extent;
        int kernelShape[4] = {buffer->buffer().dim[0].extent, buffer->buffer().dim[1].extent, buffer->buffer().dim[2].extent, buffer->buffer().dim[3].extent};  
        mBufferToImageKernel.setArg(idx++, sizeof(kernelShape),kernelShape);
        mBufferToImageKernel.setArg(idx++, static_cast<uint32_t>(heightWidthSumSize));
    } else if (type == ARGUMENT) {
        mBufferToImageKernel.setArg(idx++, static_cast<uint32_t>(buffer->buffer().dim[0].extent));
    } else {
        mBufferToImageKernel.setArg(idx++, static_cast<uint32_t>(formattedBufferShape[1]));
        mBufferToImageKernel.setArg(idx++, static_cast<uint32_t>(formattedBufferShape[2]));
        mBufferToImageKernel.setArg(idx++, static_cast<uint32_t>(formattedBufferShape[3]));
    }

    mBufferToImageKernel.setArg(idx++, openCLImage(image));

    const uint32_t maxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mBufferToImageKernel));
    const std::vector<uint32_t> lws = {16, std::max((uint32_t)1, maxWorkGroupSize / 16)};

    cl::Event event;
    cl_int error;

    std::vector<uint32_t> roundUpGroupWorkSize(lws.size());
    for (size_t i = 0; i < lws.size(); ++i) {
        roundUpGroupWorkSize[i] = ROUND_UP(gws[i], lws[i]);
    }

    error = runtime->commandQueue().enqueueNDRangeKernel(mBufferToImageKernel, cl::NullRange,
                                                         cl::NDRange(roundUpGroupWorkSize[0], roundUpGroupWorkSize[1]),
                                                         cl::NDRange(lws[0], lws[1]), nullptr, &event);
    MNN_CHECK_CL_SUCCESS(error);

    if (needWait) {
        event.wait();
    }
#ifdef LOG_VERBOSE
    MNN_PRINT("end convertBufferToImage !\n");
#endif
    return true;
}

} // namespace OpenCL
} // namespace MNN
