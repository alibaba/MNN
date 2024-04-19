//
//  ImageBufferConvertor.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/core/ImageBufferConvertor.hpp"

namespace MNN {
namespace OpenCL {
static void AddBuildOptionOfDataTypeForImage(const Tensor *input, const Tensor *output, std::set<std::string> &buildOptions, bool isfp16, bool toDevice, bool toHost){
    if(input->getType().code == halide_type_int) {
        buildOptions.emplace("-DINPUT_TYPE_I=int");
        buildOptions.emplace("-DINPUT_TYPE_I4=int4");
        if(input->getType().bits == 8){
            buildOptions.emplace("-DINPUT_TYPE=char");
            buildOptions.emplace("-DINPUT_TYPE4=char4");
            buildOptions.emplace("-DRI_DATA=read_imagei");
        } else if(input->getType().bits == 32){
            buildOptions.emplace("-DINPUT_TYPE=int");
            buildOptions.emplace("-DINPUT_TYPE4=int4");
            buildOptions.emplace("-DRI_DATA=read_imagei");
        } else {
            MNN_PRINT("opencl input datatype not support, bit:%d\n", input->getType().bits);
            MNN_ASSERT(false);
        }
    } else if(input->getType().code == halide_type_uint){
        buildOptions.emplace("-DINPUT_TYPE_I=uint");
        buildOptions.emplace("-DINPUT_TYPE_I4=uint4");
        if(input->getType().bits == 8){
            buildOptions.emplace("-DINPUT_TYPE=uchar");
            buildOptions.emplace("-DINPUT_TYPE4=uchar4");
            buildOptions.emplace("-DRI_DATA=read_imageui");
        } else if(input->getType().bits == 32){
            buildOptions.emplace("-DINPUT_TYPE=uint");
            buildOptions.emplace("-DINPUT_TYPE4=uint4");
            buildOptions.emplace("-DRI_DATA=read_imageui");
        } else {
            MNN_PRINT("opencl input datatype not support, bit:%d\n", input->getType().bits);
            MNN_ASSERT(false);
        }
    } else {
        if(isfp16 && toHost){
            buildOptions.emplace("-DINPUT_TYPE_I=half");
            buildOptions.emplace("-DINPUT_TYPE_I4=half4");
            buildOptions.emplace("-DINPUT_TYPE=half");
            buildOptions.emplace("-DINPUT_TYPE4=half4");
            buildOptions.emplace("-DRI_DATA=read_imageh");
        }else{
            buildOptions.emplace("-DINPUT_TYPE_I=float");
            buildOptions.emplace("-DINPUT_TYPE_I4=float4");
            buildOptions.emplace("-DINPUT_TYPE=float");
            buildOptions.emplace("-DINPUT_TYPE4=float4");
            buildOptions.emplace("-DRI_DATA=read_imagef");
        }
    }
    
    if(output->getType().code == halide_type_int) {
        buildOptions.emplace("-DOUTPUT_TYPE_I=int");
        buildOptions.emplace("-DOUTPUT_TYPE_I4=int4");
        buildOptions.emplace("-DCONVERT_OUTPUT_I4=convert_int4");
        if(output->getType().bits == 8){
            buildOptions.emplace("-DOUTPUT_TYPE=char");
            buildOptions.emplace("-DOUTPUT_TYPE4=char4");
            buildOptions.emplace("-DCONVERT_OUTPUT4=convert_char4");
            buildOptions.emplace("-DWI_DATA=write_imagei");
        } else if(output->getType().bits == 32){
            buildOptions.emplace("-DOUTPUT_TYPE=int");
            buildOptions.emplace("-DOUTPUT_TYPE4=int4");
            buildOptions.emplace("-DCONVERT_OUTPUT4=convert_int4");
            buildOptions.emplace("-DWI_DATA=write_imagei");
        } else {
            MNN_PRINT("opencl input datatype not support, bit:%d\n", output->getType().bits);
            MNN_ASSERT(false);
        }
    } else if(output->getType().code == halide_type_uint){
        buildOptions.emplace("-DOUTPUT_TYPE_I=uint");
        buildOptions.emplace("-DOUTPUT_TYPE_I4=uint4");
        buildOptions.emplace("-DCONVERT_OUTPUT_I4=convert_uint4");
        if(output->getType().bits == 8){
            buildOptions.emplace("-DOUTPUT_TYPE=uchar");
            buildOptions.emplace("-DOUTPUT_TYPE4=uchar4");
            buildOptions.emplace("-DCONVERT_OUTPUT4=convert_uchar4");
            buildOptions.emplace("-DWI_DATA=write_imageui");
        } else if(output->getType().bits == 32){
            buildOptions.emplace("-DOUTPUT_TYPE=uint");
            buildOptions.emplace("-DOUTPUT_TYPE4=uint4");
            buildOptions.emplace("-DCONVERT_OUTPUT4=convert_uint4");
            buildOptions.emplace("-DWI_DATA=write_imageui");
        } else {
            MNN_PRINT("opencl input datatype not support, bit:%d\n", output->getType().bits);
            MNN_ASSERT(false);
        }
    } else {
        if(isfp16 && toDevice){
            buildOptions.emplace("-DOUTPUT_TYPE_I=half");
            buildOptions.emplace("-DOUTPUT_TYPE_I4=half4");
            buildOptions.emplace("-DCONVERT_OUTPUT_I4=convert_half4");
            buildOptions.emplace("-DOUTPUT_TYPE=half");
            buildOptions.emplace("-DOUTPUT_TYPE4=half4");
            buildOptions.emplace("-DCONVERT_OUTPUT4=convert_half4");
            buildOptions.emplace("-DWI_DATA=write_imageh");
        }else{
            buildOptions.emplace("-DOUTPUT_TYPE_I=float");
            buildOptions.emplace("-DOUTPUT_TYPE_I4=float4");
            buildOptions.emplace("-DCONVERT_OUTPUT_I4=convert_float4");
            buildOptions.emplace("-DOUTPUT_TYPE=float");
            buildOptions.emplace("-DOUTPUT_TYPE4=float4");
            buildOptions.emplace("-DCONVERT_OUTPUT4=convert_float4");
            buildOptions.emplace("-DWI_DATA=write_imagef");
        }
    }
}

bool convertNCHWBufferToImage(const Tensor *input, Tensor *output,
                              OpenCLRuntime *runtime, bool needWait, bool svmFlag) {
    std::vector<int> outputShape = tensorShapeFormat(input);

    uint32_t outputGlobalWorkSize[2] = {static_cast<uint32_t>(UP_DIV(outputShape[3], 4) * outputShape[2]),
                                        static_cast<uint32_t>(outputShape[0] * outputShape[1])};
    std::set<std::string> buildOptions;
    AddBuildOptionOfDataTypeForImage(input, output, buildOptions, runtime->isSupportedFP16(), true, false);
    auto bufferToImageKernelW = runtime->buildKernelWithCache("buffer_to_image", "nchw_buffer_to_image", buildOptions);
    auto bufferToImageKernel = bufferToImageKernelW->get();
    uint32_t idx = 0;
    cl_int ret = CL_SUCCESS;
    ret |= bufferToImageKernel.setArg(idx++, outputGlobalWorkSize[0]);
    ret |= bufferToImageKernel.setArg(idx++, outputGlobalWorkSize[1]);
#ifdef MNN_OPENCL_SVM_ENABLE
    if(svmFlag == true)
    {
        ret |= clSetKernelArgSVMPointer(bufferToImageKernel.get(), idx++, (const void *)input->deviceId());
    }
    else
#endif
    {
        ret |= bufferToImageKernel.setArg(idx++, openCLBuffer(input));
    }
    ret |= bufferToImageKernel.setArg(idx++, static_cast<uint32_t>(outputShape[1]));
    ret |= bufferToImageKernel.setArg(idx++, static_cast<uint32_t>(outputShape[2]));
    ret |= bufferToImageKernel.setArg(idx++, static_cast<uint32_t>(outputShape[3]));
    ret |= bufferToImageKernel.setArg(idx++, openCLImage(output));
    MNN_CHECK_CL_SUCCESS(ret, "setArg convertNCHWBufferToImage");

    const uint32_t maxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(bufferToImageKernelW));
    const std::vector<uint32_t> lws = {16, std::max((uint32_t)1, maxWorkGroupSize / 16)};
    cl::Event event;
    cl_int res;
    std::vector<uint32_t> roundUpGroupWorkSize(lws.size());
    for (size_t i = 0; i < lws.size(); ++i) {
        roundUpGroupWorkSize[i] = ROUND_UP(outputGlobalWorkSize[i], lws[i]);
    }
    res = runtime->commandQueue().enqueueNDRangeKernel(bufferToImageKernel, cl::NullRange,
                                                         cl::NDRange(roundUpGroupWorkSize[0], roundUpGroupWorkSize[1]),
                                                         cl::NDRange(lws[0], lws[1]), nullptr, &event);
    MNN_CHECK_CL_SUCCESS(res, "nchw_buffer_to_image");
    
    if (true == needWait) {
        event.wait();
    }
    
    #ifdef ENABLE_OPENCL_TIME_PROFILER
    runtime->pushEvent({"inputFormatTransform", event});
    #endif
    return true;
}

bool convertNHWCBufferToImage(const Tensor *input, Tensor *output,
                              OpenCLRuntime *runtime, bool needWait, bool svmFlag) {
    std::vector<int> outputShape = tensorShapeFormat(input);
    uint32_t outputGlobalWorkSize[2] = {static_cast<uint32_t>(UP_DIV(outputShape[3], 4) * outputShape[2]),
                                        static_cast<uint32_t>(outputShape[0] * outputShape[1])};
    
    std::set<std::string> buildOptions;
    AddBuildOptionOfDataTypeForImage(input, output, buildOptions, runtime->isSupportedFP16(), true, false);
    auto bufferToImageKernelW = runtime->buildKernelWithCache("buffer_to_image", "nhwc_buffer_to_image", buildOptions);
    auto bufferToImageKernel = bufferToImageKernelW->get();
    uint32_t idx = 0;
    cl_int ret = CL_SUCCESS;
    ret |= bufferToImageKernel.setArg(idx++, outputGlobalWorkSize[0]);
    ret |= bufferToImageKernel.setArg(idx++, outputGlobalWorkSize[1]);
#ifdef MNN_OPENCL_SVM_ENABLE
    if(svmFlag == true) {
        ret |= clSetKernelArgSVMPointer(bufferToImageKernel.get(), idx++, (const void *)input->deviceId());
    }
    else
#endif
    {
        ret |= bufferToImageKernel.setArg(idx++, openCLBuffer(input));
    }
    ret |= bufferToImageKernel.setArg(idx++, static_cast<uint32_t>(outputShape[1]));
    ret |= bufferToImageKernel.setArg(idx++, static_cast<uint32_t>(outputShape[2]));
    ret |= bufferToImageKernel.setArg(idx++, static_cast<uint32_t>(outputShape[3]));
    ret |= bufferToImageKernel.setArg(idx++, openCLImage(output));
    MNN_CHECK_CL_SUCCESS(ret, "setArg convertNHWCBufferToImage");

    const uint32_t maxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(bufferToImageKernelW));
    const std::vector<uint32_t> lws = {16, std::max((uint32_t)1, maxWorkGroupSize / 16)};
    cl::Event event;
    cl_int res;
    std::vector<uint32_t> roundUpGroupWorkSize(lws.size());
    for (size_t i = 0; i < lws.size(); ++i) {
        roundUpGroupWorkSize[i] = ROUND_UP(outputGlobalWorkSize[i], lws[i]);
    }
    res = runtime->commandQueue().enqueueNDRangeKernel(bufferToImageKernel, cl::NullRange,
                                                         cl::NDRange(roundUpGroupWorkSize[0], roundUpGroupWorkSize[1]),
                                                         cl::NDRange(lws[0], lws[1]), nullptr, &event);
    MNN_CHECK_CL_SUCCESS(res, "nhwc_buffer_to_image");
    if (true == needWait) {
        event.wait();
    }
    
    #ifdef ENABLE_OPENCL_TIME_PROFILER
        runtime->pushEvent({"inputFormatTransform", event});
    #endif
    return true;
}

bool convertImageToNCHWBuffer(const Tensor *input, Tensor *output,
                              OpenCLRuntime *runtime, bool needWait, bool svmFlag) {
    std::vector<int> inputShape = tensorShapeFormat(input);
    uint32_t in_gws[2]          = {static_cast<uint32_t>(UP_DIV(inputShape[3], 4) * inputShape[2]),
                          static_cast<uint32_t>(inputShape[0] * inputShape[1])};

    
    std::set<std::string> buildOptions;
    AddBuildOptionOfDataTypeForImage(input, output, buildOptions, runtime->isSupportedFP16(), false, true);
    auto imageToBufferKernelW = runtime->buildKernelWithCache("buffer_to_image", "image_to_nchw_buffer", buildOptions);
    auto imageToBufferKernel = imageToBufferKernelW->get();

    uint32_t idx = 0;
    cl_int ret = CL_SUCCESS;
    ret |= imageToBufferKernel.setArg(idx++, in_gws[0]);
    ret |= imageToBufferKernel.setArg(idx++, in_gws[1]);
#ifdef MNN_OPENCL_SVM_ENABLE
    if(svmFlag == true)
    {
        ret |= clSetKernelArgSVMPointer(imageToBufferKernel.get(), idx++, (const void *)output->deviceId());
    }
    else
#endif
    {
        ret |= imageToBufferKernel.setArg(idx++, openCLBuffer(output));
    }
    ret |= imageToBufferKernel.setArg(idx++, static_cast<uint32_t>(inputShape[1]));
    ret |= imageToBufferKernel.setArg(idx++, static_cast<uint32_t>(inputShape[2]));
    ret |= imageToBufferKernel.setArg(idx++, static_cast<uint32_t>(inputShape[3]));
    ret |= imageToBufferKernel.setArg(idx++, openCLImage(input));
    MNN_CHECK_CL_SUCCESS(ret, "setArg convertImageToNCHWBuffer");

    const uint32_t maxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(imageToBufferKernelW));
    const std::vector<uint32_t> lws = {16, std::max((uint32_t)1, maxWorkGroupSize / 16)};
    cl::Event event;
    cl_int res;
    std::vector<uint32_t> roundUpGroupWorkSize(lws.size());
    for (size_t i = 0; i < lws.size(); ++i) {
        roundUpGroupWorkSize[i] = ROUND_UP(in_gws[i], lws[i]);
    }
    res = runtime->commandQueue().enqueueNDRangeKernel(imageToBufferKernel, cl::NullRange,
                                                         cl::NDRange(roundUpGroupWorkSize[0], roundUpGroupWorkSize[1]),
                                                         cl::NDRange(lws[0], lws[1]), nullptr, &event);
    MNN_CHECK_CL_SUCCESS(res, "image_to_nchw_buffer");

    if (true == needWait) {
        event.wait();
    }
    
    #ifdef ENABLE_OPENCL_TIME_PROFILER
        runtime->pushEvent({"outputFormatTransform", event});
    #endif
    return true;
}

bool convertNC4HW4BufferToImage(const Tensor *input, Tensor *output,
                                OpenCLRuntime *runtime, bool needWait, bool svmFlag) {

    uint32_t outputGlobalWorkSize[2] = {static_cast<uint32_t>(UP_DIV(input->channel(), 4) * input->width()),
                                        static_cast<uint32_t>(input->batch() * input->height())};
    std::set<std::string> buildOptions;
    AddBuildOptionOfDataTypeForImage(input, output, buildOptions, runtime->isSupportedFP16(), true, false);
    auto bufferToImageKernelW = runtime->buildKernelWithCache("buffer_to_image", "nc4hw4_buffer_to_image", buildOptions);
    auto bufferToImageKernel = bufferToImageKernelW->get();
    
    uint32_t idx   = 0;
    cl_int ret = CL_SUCCESS;
    int outputImageShape[2] = {input->height(), input->width()};
    ret |= bufferToImageKernel.setArg(idx++, outputGlobalWorkSize[0]);
    ret |= bufferToImageKernel.setArg(idx++, outputGlobalWorkSize[1]);
#ifdef MNN_OPENCL_SVM_ENABLE
    if(svmFlag == true)
    {
        ret |= clSetKernelArgSVMPointer(bufferToImageKernel.get(), idx++, (const void *)input->deviceId());
    }
    else
#endif
    {
        ret |= bufferToImageKernel.setArg(idx++, openCLBuffer(input));
    }
    ret |= bufferToImageKernel.setArg(idx++, sizeof(outputImageShape), outputImageShape);
    ret |= bufferToImageKernel.setArg(idx++, input->batch());
    ret |= bufferToImageKernel.setArg(idx++, openCLImage(output));
    MNN_CHECK_CL_SUCCESS(ret, "setArg convertNC4HW4BufferToImage");

    const uint32_t maxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(bufferToImageKernelW));
    const std::vector<uint32_t> lws = {16, std::max((uint32_t)1, maxWorkGroupSize / 16)};
    cl::Event event;
    cl_int res;
    std::vector<uint32_t> roundUpGroupWorkSize(lws.size());
    for (size_t i = 0; i < lws.size(); ++i) {
        roundUpGroupWorkSize[i] = ROUND_UP(outputGlobalWorkSize[i], lws[i]);
    }
    res = runtime->commandQueue().enqueueNDRangeKernel(bufferToImageKernel, cl::NullRange,
                                                         cl::NDRange(roundUpGroupWorkSize[0], roundUpGroupWorkSize[1]),
                                                         cl::NDRange(lws[0], lws[1]), nullptr, &event);
    MNN_CHECK_CL_SUCCESS(res, "nc4hw4_buffer_to_image");
    if (true == needWait) {
        event.wait();
    }
    
    #ifdef ENABLE_OPENCL_TIME_PROFILER
        runtime->pushEvent({"inputFormatTransform", event});
    #endif
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
bool convertImageToNC4HW4Buffer(const Tensor *input, Tensor *output,
                                OpenCLRuntime *runtime, bool needWait, bool svmFlag) {
    auto inputShape = tensorShapeFormat(input);
    uint32_t in_gws[2]          = {static_cast<uint32_t>(UP_DIV(inputShape.at(3), 4) * inputShape.at(2)),
                          static_cast<uint32_t>(inputShape.at(0) * inputShape.at(1))};

    std::set<std::string> buildOptions;
    AddBuildOptionOfDataTypeForImage(input, output, buildOptions, runtime->isSupportedFP16(), false, true);
    auto imageToBufferKernelW = runtime->buildKernelWithCache("buffer_to_image", "image_to_nc4hw4_buffer", buildOptions);
    auto imageToBufferKernel = imageToBufferKernelW->get();

    uint32_t idx   = 0;
    int outputImageShape[2] = {inputShape.at(1), inputShape.at(2)};
    cl_int ret = CL_SUCCESS;
    ret |= imageToBufferKernel.setArg(idx++, in_gws[0]);
    ret |= imageToBufferKernel.setArg(idx++, in_gws[1]);
#ifdef MNN_OPENCL_SVM_ENABLE
    if(svmFlag == true)
    {
        ret |= clSetKernelArgSVMPointer(imageToBufferKernel.get(), idx++, (const void *)output->deviceId());
    }
    else
#endif
    {
        ret |= imageToBufferKernel.setArg(idx++, openCLBuffer(output));
    }
    ret |= imageToBufferKernel.setArg(idx++, sizeof(outputImageShape), outputImageShape);
    ret |= imageToBufferKernel.setArg(idx++, input->batch());
    ret |= imageToBufferKernel.setArg(idx++, openCLImage(input));
    MNN_CHECK_CL_SUCCESS(ret, "setArg convertImageToNC4HW4Buffer");

    const uint32_t maxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(imageToBufferKernelW));
    const std::vector<uint32_t> lws = {16, std::max((uint32_t)1, maxWorkGroupSize / 16)};
    cl::Event event;
    cl_int res;
    std::vector<uint32_t> roundUpGroupWorkSize(lws.size());
    for (size_t i = 0; i < lws.size(); ++i) {
        roundUpGroupWorkSize[i] = ROUND_UP(in_gws[i], lws[i]);
    }
    res = runtime->commandQueue().enqueueNDRangeKernel(imageToBufferKernel, cl::NullRange,
                                                         cl::NDRange(roundUpGroupWorkSize[0], roundUpGroupWorkSize[1]),
                                                         cl::NDRange(lws[0], lws[1]), nullptr, &event);
    MNN_CHECK_CL_SUCCESS(res, "image_to_nc4hw4_buffer");

    if (true == needWait) {
        event.wait();
    }
    
    #ifdef ENABLE_OPENCL_TIME_PROFILER
        runtime->pushEvent({"outputFormatTransform", event});
    #endif
    return true;
}

bool convertImageToNHWCBuffer(const Tensor *input, Tensor *output,
                              OpenCLRuntime *runtime, bool needWait, bool svmFlag) {
    std::vector<int> inputShape = tensorShapeFormat(input);
    uint32_t in_gws[2]          = {static_cast<uint32_t>(UP_DIV(inputShape[3], 4) * inputShape[2]),
                          static_cast<uint32_t>(inputShape[0] * inputShape[1])};

    
    std::set<std::string> buildOptions;
    AddBuildOptionOfDataTypeForImage(input, output, buildOptions, runtime->isSupportedFP16(), false, true);
    auto imageToBufferKernelW = runtime->buildKernelWithCache("buffer_to_image", "image_to_nhwc_buffer", buildOptions);
    auto imageToBufferKernel = imageToBufferKernelW->get();

    uint32_t idx = 0;
    cl_int ret = CL_SUCCESS;
    ret |= imageToBufferKernel.setArg(idx++, in_gws[0]);
    ret |= imageToBufferKernel.setArg(idx++, in_gws[1]);
#ifdef MNN_OPENCL_SVM_ENABLE
    if(svmFlag == true)
    {
        ret |= clSetKernelArgSVMPointer(imageToBufferKernel.get(), idx++, (const void *)output->deviceId());
    }
    else
#endif
    {
        ret |= imageToBufferKernel.setArg(idx++, openCLBuffer(output));
    }
    ret |= imageToBufferKernel.setArg(idx++, static_cast<uint32_t>(inputShape[1]));
    ret |= imageToBufferKernel.setArg(idx++, static_cast<uint32_t>(inputShape[2]));
    ret |= imageToBufferKernel.setArg(idx++, static_cast<uint32_t>(inputShape[3]));
    ret |= imageToBufferKernel.setArg(idx++, openCLImage(input));
    MNN_CHECK_CL_SUCCESS(ret, "setArg convertImageToNHWCBuffer");

    const uint32_t maxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(imageToBufferKernelW));
    const std::vector<uint32_t> lws = {16, std::max((uint32_t)1, maxWorkGroupSize / 16)};
    cl::Event event;
    cl_int res;
    std::vector<uint32_t> roundUpGroupWorkSize(lws.size());
    for (size_t i = 0; i < lws.size(); ++i) {
        roundUpGroupWorkSize[i] = ROUND_UP(in_gws[i], lws[i]);
    }
    res = runtime->commandQueue().enqueueNDRangeKernel(imageToBufferKernel, cl::NullRange,
                                                         cl::NDRange(roundUpGroupWorkSize[0], roundUpGroupWorkSize[1]),
                                                         cl::NDRange(lws[0], lws[1]), nullptr, &event);
    MNN_CHECK_CL_SUCCESS(res, "image_to_nhwc_buffer");

    if (true == needWait) {
        event.wait();
    }
    
    #ifdef ENABLE_OPENCL_TIME_PROFILER
        runtime->pushEvent({"outputFormatTransform", event});
    #endif

    return true;
}
bool ImageBufferConvertor::convertImageToBuffer(const Tensor *image, const OpenCLBufferFormat type, Tensor *buffer,
                                                bool needWait, bool svmFlag) {
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

        mImageToBufferKernel = runtime->buildKernelWithCache("buffer_to_image", kernelName, buildOptions, image, buffer);
    }
    auto kernel = mImageToBufferKernel->get();
    std::vector<size_t> gws;
    getImageShape(formattedBufferShape, type, &gws);

    uint32_t idx = 0;
    cl_int ret = CL_SUCCESS;
    ret |= kernel.setArg(idx++, gws[0]);
    ret |= kernel.setArg(idx++, gws[1]);

    ret |= kernel.setArg(idx++, openCLBuffer(buffer));
    if (type == CONV2D_FILTER) {
        const int channelHeightWidthSumSize =
            buffer->buffer().dim[1].extent * buffer->buffer().dim[2].extent * buffer->buffer().dim[3].extent;
        const int heightWidthSumSize = buffer->buffer().dim[2].extent * buffer->buffer().dim[3].extent;
        int kernelShape[2] = {buffer->buffer().dim[2].extent, buffer->buffer().dim[3].extent};
        ret |= kernel.setArg(idx++, static_cast<uint32_t>(buffer->buffer().dim[0].extent));
        ret |= kernel.setArg(idx++, sizeof(kernelShape), kernelShape);
        ret |= kernel.setArg(idx++, static_cast<uint32_t>(channelHeightWidthSumSize));
        ret |= kernel.setArg(idx++, static_cast<uint32_t>(heightWidthSumSize));
    } else if (type == ARGUMENT) {
        ret |= kernel.setArg(idx++, static_cast<uint32_t>(buffer->buffer().dim[0].extent));
    } else {
        ret |= kernel.setArg(idx++, static_cast<uint32_t>(formattedBufferShape[1]));
        ret |= kernel.setArg(idx++, static_cast<uint32_t>(formattedBufferShape[2]));
        ret |= kernel.setArg(idx++, static_cast<uint32_t>(formattedBufferShape[3]));
    }
    ret |= kernel.setArg(idx++, openCLImage(image));
    MNN_CHECK_CL_SUCCESS(ret, "setArg convertImageToBuffer");

    const uint32_t maxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mImageToBufferKernel));
    const std::vector<uint32_t> lws = {16, std::max((uint32_t)1, maxWorkGroupSize / 16)};

    cl::Event event;
    cl_int res;

    std::vector<uint32_t> roundUpGroupWorkSize(lws.size());
    for (size_t i = 0; i < lws.size(); ++i) {
        roundUpGroupWorkSize[i] = ROUND_UP(gws[i], lws[i]);
    }

    res = runtime->commandQueue().enqueueNDRangeKernel(kernel, cl::NullRange,
                                                         cl::NDRange(roundUpGroupWorkSize[0], roundUpGroupWorkSize[1]),
                                                         cl::NDRange(lws[0], lws[1]), nullptr, &event);

    MNN_CHECK_CL_SUCCESS(res, "convertImageToBuffer");
#ifdef ENABLE_OPENCL_TIME_PROFILER
    runtime->pushEvent({"convertBufferToImage", event});
#endif
    if (needWait) {
        event.wait();
    }
#ifdef LOG_VERBOSE
    MNN_PRINT("end convertImageToBuffer !\n");
#endif
    return true;
}

bool ImageBufferConvertor::convertBufferToImage(const Tensor *buffer, const OpenCLBufferFormat type, Tensor *image, bool needWait, const std::string &buildOption) {
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
        case CONV2D1x1_OPT_FILTER:
            kernelName = "conv2d1x1_opt_filter_buffer_to_image";
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
        buildOptions.emplace(buildOption);
        mBufferToImageKernel = runtime->buildKernelWithCache("buffer_to_image", kernelName, buildOptions, buffer, image);
    }
    auto kernel = mBufferToImageKernel->get();

    uint32_t idx = 0;
    cl_int ret = CL_SUCCESS;
    ret |= kernel.setArg(idx++, gws[0]);
    ret |= kernel.setArg(idx++, gws[1]);

    ret |= kernel.setArg(idx++, openCLBuffer(buffer));

    if (type == CONV2D_FILTER) {
        const int channelHeightWidthSumSize =
            buffer->buffer().dim[1].extent * buffer->buffer().dim[2].extent * buffer->buffer().dim[3].extent;
        const int heightWidthSumSize = buffer->buffer().dim[2].extent * buffer->buffer().dim[3].extent;
        int kernelShape[2] = {buffer->buffer().dim[2].extent, buffer->buffer().dim[3].extent};
        ret |= kernel.setArg(idx++, static_cast<uint32_t>(buffer->buffer().dim[0].extent));
        ret |= kernel.setArg(idx++, sizeof(kernelShape),kernelShape);
        ret |= kernel.setArg(idx++, static_cast<uint32_t>(channelHeightWidthSumSize));
        ret |= kernel.setArg(idx++, static_cast<uint32_t>(heightWidthSumSize));
    } else if (type == DW_CONV2D_FILTER) {
        const int heightWidthSumSize = buffer->buffer().dim[2].extent * buffer->buffer().dim[3].extent;
        int kernelShape[4] = {buffer->buffer().dim[0].extent, buffer->buffer().dim[1].extent, buffer->buffer().dim[2].extent, buffer->buffer().dim[3].extent};
        ret |= kernel.setArg(idx++, sizeof(kernelShape),kernelShape);
        ret |= kernel.setArg(idx++, static_cast<uint32_t>(heightWidthSumSize));
    } else if (type == ARGUMENT) {
        ret |= kernel.setArg(idx++, static_cast<uint32_t>(buffer->buffer().dim[0].extent));
    } else if(type == CONV2D1x1_OPT_FILTER){
        const int channelHeightWidthSumSize =
            buffer->buffer().dim[1].extent * buffer->buffer().dim[2].extent * buffer->buffer().dim[3].extent;
        const int heightWidthSumSize = buffer->buffer().dim[2].extent * buffer->buffer().dim[3].extent;
        int kernelShape[2] = {buffer->buffer().dim[2].extent, buffer->buffer().dim[3].extent};
        ret |= kernel.setArg(idx++, static_cast<uint32_t>(buffer->buffer().dim[1].extent));
        ret |= kernel.setArg(idx++, sizeof(kernelShape),kernelShape);
        ret |= kernel.setArg(idx++, static_cast<uint32_t>(channelHeightWidthSumSize));
        ret |= kernel.setArg(idx++, static_cast<uint32_t>(heightWidthSumSize));
    }else {
        ret |= kernel.setArg(idx++, static_cast<uint32_t>(formattedBufferShape[1]));
        ret |= kernel.setArg(idx++, static_cast<uint32_t>(formattedBufferShape[2]));
        ret |= kernel.setArg(idx++, static_cast<uint32_t>(formattedBufferShape[3]));
    }

    ret |= kernel.setArg(idx++, openCLImage(image));
    MNN_CHECK_CL_SUCCESS(ret, "setArg convertBufferToImage");

    const uint32_t maxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mBufferToImageKernel));
    const std::vector<uint32_t> lws = {16, std::max((uint32_t)1, maxWorkGroupSize / 16)};

    cl::Event event;
    cl_int res;

    std::vector<uint32_t> roundUpGroupWorkSize(lws.size());
    for (size_t i = 0; i < lws.size(); ++i) {
        roundUpGroupWorkSize[i] = ROUND_UP(gws[i], lws[i]);
    }

    res = runtime->commandQueue().enqueueNDRangeKernel(kernel, cl::NullRange,
                                                         cl::NDRange(roundUpGroupWorkSize[0], roundUpGroupWorkSize[1]),
                                                         cl::NDRange(lws[0], lws[1]), nullptr, &event);
    MNN_CHECK_CL_SUCCESS(res, "convertBufferToImage");
#ifdef ENABLE_OPENCL_TIME_PROFILER
    runtime->pushEvent({"convertBufferToImage", event});
#endif
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
