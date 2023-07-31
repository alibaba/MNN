//
//  BufferConvertor.cpp
//  MNN
//
//  Created by MNN on 2020/09/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED

#include "backend/opencl/core/BufferConvertor.hpp"

namespace MNN {
namespace OpenCL {

bool converNCHWOrNHWCBufferToNC4HW4OrNC16HW16Buffer(const Tensor *input, Tensor *output, cl::Kernel &convertBufferKernel, const std::string Name, OpenCLRuntime *runtime, bool needTrans, bool needWait, bool svmFlag) {
    std::vector<int> outputShape = tensorShapeFormat(input);
    std::string kernelName = Name;
    std::string sourceName = "buffer_convert_buf";
    uint32_t cPack = 4;
    auto inputpad = TensorUtils::getDescribe(input)->mPads;
    auto outputpad = TensorUtils::getDescribe(output)->mPads;
#ifdef MNN_SUPPORT_INTEL_SUBGROUP
    cPack = TensorUtils::getTensorChannelPack(output);
    if(cPack == 16)
    {
        sourceName =  "buffer_convert_subgroup_buf";
    }
#endif
    uint32_t outputGlobalWorkSize[2] = {static_cast<uint32_t>(UP_DIV(outputShape[3], cPack) * outputShape[2]),
                                        static_cast<uint32_t>(outputShape[0] * outputShape[1])};
    if (convertBufferKernel.get() == nullptr) {
        std::set<std::string> buildOptions;
        if(needTrans) {
            kernelName += "_floatin";
        }
        convertBufferKernel = runtime->buildKernel(sourceName, kernelName, buildOptions);
    }
    uint32_t idx = 0;
    cl_int ret = CL_SUCCESS;
    ret |= convertBufferKernel.setArg(idx++, outputGlobalWorkSize[0]);
    ret |= convertBufferKernel.setArg(idx++, outputGlobalWorkSize[1]);
#ifdef MNN_OPENCL_SVM_ENABLE
    if(svmFlag == true) {
        ret |= clSetKernelArgSVMPointer(convertBufferKernel.get(), idx++, (const void *)input->deviceId());
    }
    else
#endif
    {
        ret |= convertBufferKernel.setArg(idx++, openCLBuffer(input));
    }
    ret |= convertBufferKernel.setArg(idx++, static_cast<uint32_t>(outputShape[1]));
    ret |= convertBufferKernel.setArg(idx++, static_cast<uint32_t>(outputShape[2]));
    ret |= convertBufferKernel.setArg(idx++, static_cast<uint32_t>(outputShape[3]));
    ret |= convertBufferKernel.setArg(idx++, openCLBuffer(output));
    if(cPack == 16)
    {
        ret |= convertBufferKernel.setArg(idx++, static_cast<uint32_t>(inputpad.left));
        ret |= convertBufferKernel.setArg(idx++, static_cast<uint32_t>(inputpad.right));
        ret |= convertBufferKernel.setArg(idx++, static_cast<uint32_t>(outputpad.left));
        ret |= convertBufferKernel.setArg(idx++, static_cast<uint32_t>(outputpad.right));
    }
    MNN_CHECK_CL_SUCCESS(ret, "setArg converNCHWOrNHWCBufferToNC4HW4OrNC16HW16Buffer");

    const uint32_t maxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(convertBufferKernel));
    const std::vector<uint32_t> lws = {16, std::max((uint32_t)1, maxWorkGroupSize / 16)};
    cl::Event event;
    cl_int res;
    std::vector<uint32_t> roundUpGroupWorkSize(lws.size());
    for (size_t i = 0; i < lws.size(); ++i) {
        roundUpGroupWorkSize[i] = ROUND_UP(outputGlobalWorkSize[i], lws[i]);
    }
    res = runtime->commandQueue().enqueueNDRangeKernel(convertBufferKernel, cl::NullRange,
                                                         cl::NDRange(roundUpGroupWorkSize[0], roundUpGroupWorkSize[1]),
                                                         cl::NDRange(lws[0], lws[1]), nullptr, &event);
    MNN_CHECK_CL_SUCCESS(res, kernelName.c_str());
    
    if (true == needWait) {
        event.wait();
    }
    return true;
}

bool convertNC4HW4BufferToNC4HW4Buffer(const Tensor *input, Tensor *output, cl::Kernel &convertBufferKernel,
                                OpenCLRuntime *runtime, TransType formatTrans, bool needWait, bool svmFlag, bool srcswap, bool dstswap) {
    std::vector<int> outputShape = tensorShapeFormat(input);
    uint32_t outputGlobalWorkSize[2] = {static_cast<uint32_t>(UP_DIV(outputShape[3], 4) * outputShape[2]),
                                        static_cast<uint32_t>(outputShape[0] * outputShape[1])};
    if (convertBufferKernel.get() == nullptr) {
        std::set<std::string> buildOptions;
        std::string kernelName = "nc4hw4_buffer_to_nc4hw4_buffer";
        switch (formatTrans) {
            case InpTrans:
                kernelName += "_floatin";
                break;
            case OutTrans:
                kernelName += "_floatout";
                break;
            default:
                break;
        }
        convertBufferKernel = runtime->buildKernel("buffer_convert_buf", kernelName, buildOptions);
    }
    uint32_t idx   = 0;
    int outputImageShape[2] = {input->height(), input->width()};
    int channelC4 = UP_DIV(input->channel(), 4);
    int batch  = input->batch();
    int srcStride[2] = {
        channelC4,
        1
    };
    int dstStride[2] = {
        channelC4,
        1
    };
    if (srcswap) {
        srcStride[0] = 1;
        srcStride[1] = batch;
    }
    if (dstswap) {
        dstStride[0] = 1;
        dstStride[1] = batch;
    }
    cl_int ret = CL_SUCCESS;
    ret |= convertBufferKernel.setArg(idx++, outputGlobalWorkSize[0]);
    ret |= convertBufferKernel.setArg(idx++, outputGlobalWorkSize[1]);
#ifdef MNN_OPENCL_SVM_ENABLE
    if(svmFlag == true)
    {
        ret |= clSetKernelArgSVMPointer(convertBufferKernel.get(), idx++, (const void *)input->buffer().device);
    }
    else
#endif
    {
        ret |= convertBufferKernel.setArg(idx++, openCLBuffer(input));
    }
    ret |= convertBufferKernel.setArg(idx++, sizeof(outputImageShape), outputImageShape);
    ret |= convertBufferKernel.setArg(idx++, sizeof(srcStride), srcStride);
    ret |= convertBufferKernel.setArg(idx++, sizeof(dstStride), dstStride);
    ret |= convertBufferKernel.setArg(idx++, openCLBuffer(output));
    MNN_CHECK_CL_SUCCESS(ret, "setArg convertNC4HW4BufferToNC4HW4Buffer");

    const uint32_t maxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(convertBufferKernel));
    const std::vector<uint32_t> lws = {16, std::max((uint32_t)1, maxWorkGroupSize / 16)};
    cl::Event event;
    cl_int res;
    std::vector<uint32_t> roundUpGroupWorkSize(lws.size());
    for (size_t i = 0; i < lws.size(); ++i) {
        roundUpGroupWorkSize[i] = ROUND_UP(outputGlobalWorkSize[i], lws[i]);
    }
    res = runtime->commandQueue().enqueueNDRangeKernel(convertBufferKernel, cl::NullRange,
                                                         cl::NDRange(roundUpGroupWorkSize[0], roundUpGroupWorkSize[1]),
                                                         cl::NDRange(lws[0], lws[1]), nullptr, &event);
    MNN_CHECK_CL_SUCCESS(res, "nc4hw4_buffer_to_nc4hw4_buffer");
    if (true == needWait) {
        event.wait();
    }
    return true;
}

#ifdef MNN_SUPPORT_INTEL_SUBGROUP
bool convertNC4HW4BufferBetweenNC16HW16Buffer(const Tensor *input, Tensor *output, cl::Kernel &convertBufferKernel, const std::string Name,
                                       OpenCLRuntime *runtime, TransType formatTrans, bool needWait, bool svmFlag,
                                       bool srcswap, bool dstswap) {
    std::vector<int> outputShape     = tensorShapeFormat(input);
    uint32_t outputGlobalWorkSize[2] = {static_cast<uint32_t>(UP_DIV(outputShape[3], 16) * outputShape[2]),
                                        static_cast<uint32_t>(outputShape[0] * outputShape[1])};
    std::string kernelName = Name;
    auto inputpad = TensorUtils::getDescribe(input)->mPads;
    auto outputpad = TensorUtils::getDescribe(output)->mPads;
    if (convertBufferKernel.get() == nullptr) {
        std::set<std::string> buildOptions;
        switch (formatTrans) {
            case InpTrans:
                kernelName += "_floatin";
                break;
            case OutTrans:
                kernelName += "_floatout";
                break;
            default:
                break;
        }
        convertBufferKernel = runtime->buildKernel("buffer_convert_subgroup_buf", kernelName, buildOptions);
    }
    uint32_t idx            = 0;
    int outputImageShape[2] = {input->height(), input->width()};
    int inchannelPack           = UP_DIV(input->channel(), TensorUtils::getTensorChannelPack(input));
    int outchannelPack          = UP_DIV(output->channel(), TensorUtils::getTensorChannelPack(output));
    int batch               = input->batch();
    int srcStride[2]        = {inchannelPack, 1};
    int dstStride[2]        = {outchannelPack, 1};
    if (srcswap) {
        srcStride[0] = 1;
        srcStride[1] = batch;
    }
    if (dstswap) {
        dstStride[0] = 1;
        dstStride[1] = batch;
    }
    cl_int ret = CL_SUCCESS;
    ret |= convertBufferKernel.setArg(idx++, outputGlobalWorkSize[0]);
    ret |= convertBufferKernel.setArg(idx++, outputGlobalWorkSize[1]);
#ifdef MNN_OPENCL_SVM_ENABLE
    if (svmFlag == true) {
        ret |= clSetKernelArgSVMPointer(convertBufferKernel.get(), idx++, (const void *)input->buffer().device);
    } else
#endif
    {
        ret |= convertBufferKernel.setArg(idx++, openCLBuffer(input));
    }
    ret |= convertBufferKernel.setArg(idx++, sizeof(outputImageShape), outputImageShape);
    ret |= convertBufferKernel.setArg(idx++, sizeof(srcStride), srcStride);
    ret |= convertBufferKernel.setArg(idx++, sizeof(dstStride), dstStride);
    ret |= convertBufferKernel.setArg(idx++, openCLBuffer(output));
    ret |= convertBufferKernel.setArg(idx++, static_cast<uint32_t>(inputpad.left));
    ret |= convertBufferKernel.setArg(idx++, static_cast<uint32_t>(inputpad.right));
    ret |= convertBufferKernel.setArg(idx++, static_cast<uint32_t>(outputpad.left));
    ret |= convertBufferKernel.setArg(idx++, static_cast<uint32_t>(outputpad.right));
    ret |= convertBufferKernel.setArg(idx++, static_cast<uint32_t>(outchannelPack));
    MNN_CHECK_CL_SUCCESS(ret, "setArg convertNC4HW4BufferBetweenNC16HW16Buffer");

    const uint32_t maxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(convertBufferKernel));
    const std::vector<uint32_t> lws = {16, std::max((uint32_t)1, maxWorkGroupSize / 16)};
    cl::Event event;
    cl_int res;
    std::vector<uint32_t> roundUpGroupWorkSize(lws.size());
    for (size_t i = 0; i < lws.size(); ++i) {
        roundUpGroupWorkSize[i] = ROUND_UP(outputGlobalWorkSize[i], lws[i]);
    }
    res = runtime->commandQueue().enqueueNDRangeKernel(convertBufferKernel, cl::NullRange,
                                                       cl::NDRange(roundUpGroupWorkSize[0], roundUpGroupWorkSize[1]),
                                                       cl::NDRange(lws[0], lws[1]), nullptr, &event);
    MNN_CHECK_CL_SUCCESS(res, Name.c_str());
    if (true == needWait) {
        event.wait();
    }
    return true;
}
#endif /* MNN_SUPPORT_INTEL_SUBGROUP */

bool convertNC4HW4OrNC16HW16BufferToNCHWOrNHWCBuffer(const Tensor *input, Tensor *output, cl::Kernel &convertBufferKernel, const std::string Name, OpenCLRuntime *runtime, bool needOutTrans, bool needWait, bool svmFlag) {
    std::vector<int> inputShape = tensorShapeFormat(input);
    std::string kernelName      = Name;
    std::string sourceName = "buffer_convert_buf";
    uint32_t cPack = 4;
    auto inputpad = TensorUtils::getDescribe(input)->mPads;
    auto outputpad = TensorUtils::getDescribe(output)->mPads;
#ifdef MNN_SUPPORT_INTEL_SUBGROUP
    cPack = TensorUtils::getTensorChannelPack(input);
    if(cPack == 16)
    {
        sourceName =  "buffer_convert_subgroup_buf";
    }
#endif
    uint32_t in_gws[2]          = {static_cast<uint32_t>(UP_DIV(inputShape[3], cPack) * inputShape[2]),
                          static_cast<uint32_t>(inputShape[0] * inputShape[1])};

    if (convertBufferKernel.get() == nullptr) {
        std::set<std::string> buildOptions;
        if(needOutTrans) {
            kernelName += "_floatout";
        }
        convertBufferKernel = runtime->buildKernel(sourceName, kernelName, buildOptions);
    }

    uint32_t idx = 0;
    cl_int ret = CL_SUCCESS;
    ret |= convertBufferKernel.setArg(idx++, in_gws[0]);
    ret |= convertBufferKernel.setArg(idx++, in_gws[1]);
#ifdef MNN_OPENCL_SVM_ENABLE
    if(svmFlag == true)
    {
        ret |= clSetKernelArgSVMPointer(convertBufferKernel.get(), idx++, (const void *)output->deviceId());
    }
    else
#endif
    {
        ret |= convertBufferKernel.setArg(idx++, openCLBuffer(output));
    }
    ret |= convertBufferKernel.setArg(idx++, static_cast<uint32_t>(inputShape[1]));
    ret |= convertBufferKernel.setArg(idx++, static_cast<uint32_t>(inputShape[2]));
    ret |= convertBufferKernel.setArg(idx++, static_cast<uint32_t>(inputShape[3]));
    ret |= convertBufferKernel.setArg(idx++, openCLBuffer(input));
    if(cPack == 16)
    {
        ret |= convertBufferKernel.setArg(idx++, static_cast<uint32_t>(inputpad.left));
        ret |= convertBufferKernel.setArg(idx++, static_cast<uint32_t>(inputpad.right));
        ret |= convertBufferKernel.setArg(idx++, static_cast<uint32_t>(outputpad.left));
        ret |= convertBufferKernel.setArg(idx++, static_cast<uint32_t>(outputpad.right));
    }
    MNN_CHECK_CL_SUCCESS(ret, "setArg convertNC4HW4OrNC16HW16BufferToNCHWOrNHWCBuffer");

    const uint32_t maxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(convertBufferKernel));
    const std::vector<uint32_t> lws = {16, std::max((uint32_t)1, maxWorkGroupSize / 16)};
    cl::Event event;
    cl_int res;
    std::vector<uint32_t> roundUpGroupWorkSize(lws.size());
    for (size_t i = 0; i < lws.size(); ++i) {
        roundUpGroupWorkSize[i] = ROUND_UP(in_gws[i], lws[i]);
    }
    res = runtime->commandQueue().enqueueNDRangeKernel(convertBufferKernel, cl::NullRange,
                                                         cl::NDRange(roundUpGroupWorkSize[0], roundUpGroupWorkSize[1]),
                                                         cl::NDRange(lws[0], lws[1]), nullptr, &event);
    MNN_CHECK_CL_SUCCESS(res, kernelName.c_str());

    if (true == needWait) {
        event.wait();
    }
    return true;
}

bool BufferConvertor::convertToNC4HW4Buffer(const Tensor *buffer, const OpenCLBufferFormat type, Tensor *image, bool needTrans, bool needWait) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start convertBufferToNC4HW4Buffer !\n");
#endif
    auto formattedBufferShape = tensorShapeFormat(buffer);//NHWC
    std::vector<size_t> imageShape;
    getImageShape(formattedBufferShape, type, &imageShape);

    uint32_t gws[2] = {static_cast<uint32_t>(imageShape[0]), static_cast<uint32_t>(imageShape[1])};

    auto runtime = mOpenCLRuntime;
    std::string kernelName;
    switch (type) {
        case CONV2D_FILTER:
            kernelName = "conv2d_filter_buffer_to_nc4hw4_buffer";//NC4HW4 (1, 4*ic/4, kw*kh*oc/4, 1)*4
            break;
        case DW_CONV2D_FILTER:
            kernelName = "dw_filter_buffer_to_nc4hw4_buffer";//NC4HW4 (1, kw*kh, oc/4, 1)*4
        case NHWC_BUFFER:
        case NCHW_BUFFER:
        case ARGUMENT:
            break;
        default:
            break;
    }
    if (mBufferToImageKernel.get() == nullptr || mBufferToImageKernelName != kernelName) {
        mBufferToImageKernelName = kernelName;
        std::set<std::string> buildOptions;
        if(needTrans) {
            //buildOptions.emplace("-DBUFFER_FORMAT_INP_TRANS");
            kernelName += "_floatin";
        }
        mBufferToImageKernel = runtime->buildKernel("buffer_convert_buf", kernelName, buildOptions);
    }

    uint32_t idx = 0;
    cl_int ret = CL_SUCCESS;
    ret |= mBufferToImageKernel.setArg(idx++, gws[0]);
    ret |= mBufferToImageKernel.setArg(idx++, gws[1]);

    ret |= mBufferToImageKernel.setArg(idx++, openCLBuffer(buffer));

    if (type == CONV2D_FILTER) {
        const int channelHeightWidthSumSize =
            buffer->buffer().dim[1].extent * buffer->buffer().dim[2].extent * buffer->buffer().dim[3].extent;
        const int heightWidthSumSize = buffer->buffer().dim[2].extent * buffer->buffer().dim[3].extent;
        int kernelShape[2] = {buffer->buffer().dim[2].extent, buffer->buffer().dim[3].extent};
        ret |= mBufferToImageKernel.setArg(idx++, static_cast<uint32_t>(buffer->buffer().dim[0].extent));
        ret |= mBufferToImageKernel.setArg(idx++, sizeof(kernelShape),kernelShape);
        ret |= mBufferToImageKernel.setArg(idx++, static_cast<uint32_t>(channelHeightWidthSumSize));
        ret |= mBufferToImageKernel.setArg(idx++, static_cast<uint32_t>(heightWidthSumSize));
    } else if (type == DW_CONV2D_FILTER) {
        const int heightWidthSumSize = buffer->buffer().dim[2].extent * buffer->buffer().dim[3].extent;
        int kernelShape[4] = {buffer->buffer().dim[0].extent, buffer->buffer().dim[1].extent, buffer->buffer().dim[2].extent, buffer->buffer().dim[3].extent};
        ret |= mBufferToImageKernel.setArg(idx++, sizeof(kernelShape),kernelShape);
        ret |= mBufferToImageKernel.setArg(idx++, static_cast<uint32_t>(heightWidthSumSize));
    } else {
        MNN_PRINT("convertToNC4HW4Buffer type not support!\n");
        return false;
    }

    ret |= mBufferToImageKernel.setArg(idx++, openCLBuffer(image));
    MNN_CHECK_CL_SUCCESS(ret, "setArg convertToNC4HW4Buffer");

    const uint32_t maxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mBufferToImageKernel));
    const std::vector<uint32_t> lws = {16, std::max((uint32_t)1, maxWorkGroupSize / 16)};

    cl::Event event;
    cl_int res;

    std::vector<uint32_t> roundUpGroupWorkSize(lws.size());
    for (size_t i = 0; i < lws.size(); ++i) {
        roundUpGroupWorkSize[i] = ROUND_UP(gws[i], lws[i]);
    }

    res = runtime->commandQueue().enqueueNDRangeKernel(mBufferToImageKernel, cl::NullRange,
                                                         cl::NDRange(roundUpGroupWorkSize[0], roundUpGroupWorkSize[1]),
                                                         cl::NDRange(lws[0], lws[1]), nullptr, &event);
    MNN_CHECK_CL_SUCCESS(res, "convertToNC4HW4Buffer");

    if (needWait) {
        event.wait();
    }
#ifdef LOG_VERBOSE
    MNN_PRINT("end convertBufferToNC4HW4Buffer !\n");
#endif
    return true;
}
} // namespace OpenCL
} // namespace MNN
#endif /* MNN_OPENCL_BUFFER_CLOSED */
