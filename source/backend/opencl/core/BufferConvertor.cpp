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

static void AddBuildOptionOfDataType(const Tensor *input, const Tensor *output, std::set<std::string> &buildOptions, bool isfp16, bool toDevice, bool toHost){
    if(input->getType().code == halide_type_int) {
        if(input->getType().bits == 8){
            buildOptions.emplace("-DINPUT_TYPE=char");
            buildOptions.emplace("-DINPUT_TYPE4=char4");
            buildOptions.emplace("-DINPUT_TYPE16=char16");
        } else if(input->getType().bits == 32){
            buildOptions.emplace("-DINPUT_TYPE=int");
            buildOptions.emplace("-DINPUT_TYPE4=int4");
            buildOptions.emplace("-DINPUT_TYPE16=int16");
        } else {
            MNN_PRINT("opencl input datatype not support, bit:%d\n", input->getType().bits);
            MNN_ASSERT(false);
        }
    } else if(input->getType().code == halide_type_uint){
        if(input->getType().bits == 8){
            buildOptions.emplace("-DINPUT_TYPE=uchar");
            buildOptions.emplace("-DINPUT_TYPE4=uchar4");
            buildOptions.emplace("-DINPUT_TYPE16=uchar16");
        } else if(input->getType().bits == 32){
            buildOptions.emplace("-DINPUT_TYPE=uint");
            buildOptions.emplace("-DINPUT_TYPE4=uint4");
            buildOptions.emplace("-DINPUT_TYPE16=uint16");
        } else {
            MNN_PRINT("opencl input datatype not support, bit:%d\n", input->getType().bits);
            MNN_ASSERT(false);
        }
    } else {
        if(isfp16 && toHost){
            buildOptions.emplace("-DINPUT_TYPE=half");
            buildOptions.emplace("-DINPUT_TYPE4=half4");
            buildOptions.emplace("-DINPUT_TYPE16=half16");
        }else{
            buildOptions.emplace("-DINPUT_TYPE=float");
            buildOptions.emplace("-DINPUT_TYPE4=float4");
            buildOptions.emplace("-DINPUT_TYPE16=float16");
        }
    }
    
    if(output->getType().code == halide_type_int) {
        if(output->getType().bits == 8){
            buildOptions.emplace("-DOUTPUT_TYPE=char");
            buildOptions.emplace("-DOUTPUT_TYPE4=char4");
            buildOptions.emplace("-DOUTPUT_TYPE16=char16");
            buildOptions.emplace("-DCONVERT_OUTPUT4=convert_char4");
            buildOptions.emplace("-DCONVERT_OUTPUT16=convert_char16");
        } else if(output->getType().bits == 32){
            buildOptions.emplace("-DOUTPUT_TYPE=int");
            buildOptions.emplace("-DOUTPUT_TYPE4=int4");
            buildOptions.emplace("-DOUTPUT_TYPE16=int16");
            buildOptions.emplace("-DCONVERT_OUTPUT4=convert_int4");
            buildOptions.emplace("-DCONVERT_OUTPUT16=convert_int16");
        } else {
            MNN_PRINT("opencl input datatype not support, bit:%d\n", output->getType().bits);
            MNN_ASSERT(false);
        }
    } else if(output->getType().code == halide_type_uint){
        if(output->getType().bits == 8){
            buildOptions.emplace("-DOUTPUT_TYPE=uchar");
            buildOptions.emplace("-DOUTPUT_TYPE4=uchar4");
            buildOptions.emplace("-DOUTPUT_TYPE16=uchar16");
            buildOptions.emplace("-DCONVERT_OUTPUT4=convert_uchar4");
            buildOptions.emplace("-DCONVERT_OUTPUT16=convert_uchar16");
        } else if(output->getType().bits == 32){
            buildOptions.emplace("-DOUTPUT_TYPE=uint");
            buildOptions.emplace("-DOUTPUT_TYPE4=uint4");
            buildOptions.emplace("-DOUTPUT_TYPE16=uint16");
            buildOptions.emplace("-DCONVERT_OUTPUT4=convert_uint4");
            buildOptions.emplace("-DCONVERT_OUTPUT16=convert_uint16");
        } else {
            MNN_PRINT("opencl input datatype not support, bit:%d\n", output->getType().bits);
            MNN_ASSERT(false);
        }
    } else {
        if(isfp16 && toDevice){
            buildOptions.emplace("-DOUTPUT_TYPE=half");
            buildOptions.emplace("-DOUTPUT_TYPE4=half4");
            buildOptions.emplace("-DOUTPUT_TYPE16=half16");
            buildOptions.emplace("-DCONVERT_OUTPUT4=convert_half4");
            buildOptions.emplace("-DCONVERT_OUTPUT16=convert_half16");
        }else{
            buildOptions.emplace("-DOUTPUT_TYPE=float");
            buildOptions.emplace("-DOUTPUT_TYPE4=float4");
            buildOptions.emplace("-DOUTPUT_TYPE16=float16");
            buildOptions.emplace("-DCONVERT_OUTPUT4=convert_float4");
            buildOptions.emplace("-DCONVERT_OUTPUT16=convert_float16");
        }
    }
}

bool converNCHWOrNHWCBufferToNC4HW4OrNC16HW16Buffer(const Tensor *input, Tensor *output, const std::string Name, OpenCLRuntime *runtime, bool needTrans, bool needWait, bool svmFlag) {
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
    std::set<std::string> buildOptions;
    AddBuildOptionOfDataType(input, output, buildOptions, runtime->isSupportedFP16(), true, false);
    auto convertBufferKernelW = runtime->buildKernelWithCache(sourceName, kernelName, buildOptions);
    auto convertBufferKernel = convertBufferKernelW->get();
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

    const uint32_t maxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(convertBufferKernelW));
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

#ifdef MNN_SUPPORT_INTEL_SUBGROUP
bool convertNC4HW4BufferBetweenNC16HW16Buffer(const Tensor *input, Tensor *output, const std::string Name,
                                       OpenCLRuntime *runtime, TransType formatTrans, bool needWait, bool svmFlag,
                                       bool srcswap, bool dstswap) {
    std::vector<int> outputShape     = tensorShapeFormat(input);
    uint32_t outputGlobalWorkSize[2] = {static_cast<uint32_t>(UP_DIV(outputShape[3], 16) * outputShape[2]),
                                        static_cast<uint32_t>(outputShape[0] * outputShape[1])};
    std::string kernelName = Name;
    auto inputpad = TensorUtils::getDescribe(input)->mPads;
    auto outputpad = TensorUtils::getDescribe(output)->mPads;
    std::set<std::string> buildOptions;
    switch (formatTrans) {
        case InpTrans:
            AddBuildOptionOfDataType(input, output, buildOptions, runtime->isSupportedFP16(), true, false);
            break;
        case OutTrans:
            AddBuildOptionOfDataType(input, output, buildOptions, runtime->isSupportedFP16(), false, true);
            break;
        default:
            AddBuildOptionOfDataType(input, output, buildOptions, runtime->isSupportedFP16(), true, true);
            break;
    }
    auto convertBufferKernelW = runtime->buildKernelWithCache("buffer_convert_subgroup_buf", kernelName, buildOptions);
    auto convertBufferKernel = convertBufferKernelW->get();
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

    const uint32_t maxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(convertBufferKernelW));
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

bool convertNC4HW4OrNC16HW16BufferToNCHWOrNHWCBuffer(const Tensor *input, Tensor *output, const std::string Name, OpenCLRuntime *runtime, bool needOutTrans, bool needWait, bool svmFlag) {
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
    std::set<std::string> buildOptions;
    AddBuildOptionOfDataType(input, output, buildOptions, runtime->isSupportedFP16(), false, true);
    auto convertBufferKernelW = runtime->buildKernelWithCache(sourceName, kernelName, buildOptions);
    auto convertBufferKernel = convertBufferKernelW->get();

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

    const uint32_t maxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(convertBufferKernelW));
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

bool BufferConvertor::convertToNC4HW4Buffer(const Tensor *buffer, const OpenCLBufferFormat type, Tensor *image, bool needTrans, bool needWait, bool lowMemory, int quantBit) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start convertBufferToNC4HW4Buffer !\n");
#endif
    auto formattedBufferShape = tensorShapeFormat(buffer);//NHWC
    std::vector<size_t> imageShape;
    getImageShape(formattedBufferShape, type, &imageShape);

    uint32_t gws[2] = {static_cast<uint32_t>(imageShape[0]), static_cast<uint32_t>(imageShape[1])};

    auto runtime = mOpenCLRuntime;
    std::string kernelName;
    std::string kernelFile = "buffer_convert_buf";
    switch (type) {
        case CONV2D_FILTER:
#ifdef MNN_LOW_MEMORY
            if (lowMemory) {
                if (quantBit != 8 && quantBit != 4) {
                    MNN_ERROR("For Opencl Backend, only support low memory mode of int8 or int4 dequantization currently.\n");
                    MNN_ASSERT(false);
                }
                kernelFile = "buffer_convert_quant";
                // shared part for all cases
                if (quantBit == 8) {
                    kernelName = "conv2d_filter_buffer_to_nc4hw4_buffer_int8"; //NC4HW4 (1, 4*ic/4, kw*kh*oc/4, 1)*4
                } else if (quantBit == 4){
                    kernelName = "conv2d_filter_buffer_to_nc4hw4_buffer_int4"; //NC4HW4 (1, 4*ic/4, kw*kh*oc/4, 1)*4
                } else {/* More types to be supported. */}
            } else
#endif
            {
                kernelName = "conv2d_filter_buffer_to_nc4hw4_buffer";//NC4HW4 (1, 4*ic/4, kw*kh*oc/4, 1)*4
            }
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
#ifdef MNN_LOW_MEMORY
        if (lowMemory) {
            if (quantBit == 8) {
                // int8 case
                buildOptions.emplace("-DUSE_LOW_BIT_WEIGHT_INT8");
            } else if (quantBit == 4){
                // int4 case
                buildOptions.emplace("-DUSE_LOW_BIT_WEIGHT_INT4");
            } else {/* More types to be supported. */}
        }
#endif
        mBufferToImageKernel = runtime->buildKernelWithCache(kernelFile, kernelName, buildOptions, buffer, image);
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
    } else {
        MNN_PRINT("convertToNC4HW4Buffer type not support!\n");
        return false;
    }

    ret |= kernel.setArg(idx++, openCLBuffer(image));
    MNN_CHECK_CL_SUCCESS(ret, "setArg convertToNC4HW4Buffer");

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
    MNN_CHECK_CL_SUCCESS(res, "convertToNC4HW4Buffer");

    if (needWait) {
        event.wait();
    }
#ifdef LOG_VERBOSE
    MNN_PRINT("end convertBufferToNC4HW4Buffer !\n");
#endif
    return true;
}

bool convertBufferToBuffer(Tensor *input, Tensor *output, OpenCLRuntime *runtime, bool toDevice, bool toHost, bool needWait, bool svmFlag) {
    std::vector<int> outputShape = tensorShapeFormat(input);
    int shape[4] = {outputShape[0], outputShape[3], outputShape[1], outputShape[2]};//N C H W
    auto srcDimensionFormat = TensorUtils::getDescribe(input)->dimensionFormat;
    auto dstDimensionFormat = TensorUtils::getDescribe(output)->dimensionFormat;
    if (MNN_DATA_FORMAT_NC4HW4 == dstDimensionFormat && srcDimensionFormat != dstDimensionFormat && (outputShape[3] % 4) != 0){
        int region[] = {outputShape[0], ROUND_UP(outputShape[3], 4), outputShape[1], outputShape[2]};//nchw
        
        auto kernelW = runtime->buildKernelWithCache("raster_buf", "buffer_set_zero", {}, output, output);
        auto kernel = kernelW->get();
        uint32_t lws[2] = {8, 8};
        uint32_t gws[2] = {(uint32_t)UP_DIV((region[2] * region[3]), 8)*8, (uint32_t)UP_DIV((region[0] * region[1]), 8)*8};
    
        int global_dim0 = region[2] * region[3];
        int global_dim1 = region[0] * region[1];
    
        uint32_t idx   = 0;
        cl_int res = CL_SUCCESS;
        res |= kernel.setArg(idx++, global_dim0);
        res |= kernel.setArg(idx++, global_dim1);
        res |= kernel.setArg(idx++, openCLBuffer(output));
        MNN_CHECK_CL_SUCCESS(res, "setArg buffer_set_zero");
    
        res = runtime->commandQueue().enqueueNDRangeKernel(kernel, cl::NullRange,
                                                         cl::NDRange(gws[0], gws[1]),
                                                         cl::NDRange(lws[0], lws[1]), nullptr, nullptr);
        MNN_CHECK_CL_SUCCESS(res, "buffer_set_zero");
    }
    if (srcDimensionFormat == dstDimensionFormat && MNN_DATA_FORMAT_NC4HW4 != dstDimensionFormat){
        int size = outputShape[0] * outputShape[1] * outputShape[2] * outputShape[3];
        uint32_t gws[2] = {static_cast<uint32_t>(UP_DIV(size, 4)), static_cast<uint32_t>(1)};
        std::set<std::string> buildOptions;
        if(size % 4 != 0){
            buildOptions.emplace("-DPACK_LEAVE");
        }
        AddBuildOptionOfDataType(input, output, buildOptions, runtime->isSupportedFP16(), toDevice, toHost);
        auto convertBufferKernelW = runtime->buildKernelWithCache("buffer_convert_buf", "buffer_copy_to_buffer", buildOptions);
        auto convertBufferKernel = convertBufferKernelW->get();
        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;
        ret |= convertBufferKernel.setArg(idx++, gws[0]);
        ret |= convertBufferKernel.setArg(idx++, gws[1]);
#ifdef MNN_OPENCL_SVM_ENABLE
        if(svmFlag == true && toDevice) {
            ret |= clSetKernelArgSVMPointer(convertBufferKernel.get(), idx++, (const void *)input->deviceId());
        }
        else
#endif
        {
            ret |= convertBufferKernel.setArg(idx++, openCLBuffer(input));
        }
#ifdef MNN_OPENCL_SVM_ENABLE
        if(svmFlag == true && toHost) {
            ret |= clSetKernelArgSVMPointer(convertBufferKernel.get(), idx++, (const void *)output->deviceId());
        }
        else
#endif
        {
            ret |= convertBufferKernel.setArg(idx++, openCLBuffer(output));
        }
        ret |= convertBufferKernel.setArg(idx++, size);
        MNN_CHECK_CL_SUCCESS(ret, "setArg buffer_convert_to_buffer");

        const uint32_t maxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(convertBufferKernelW));
        const std::vector<uint32_t> lws = {16, std::max((uint32_t)1, maxWorkGroupSize / 16)};
        cl::Event event;
        cl_int res;
        std::vector<uint32_t> roundUpGroupWorkSize(lws.size());
        for (size_t i = 0; i < lws.size(); ++i) {
            roundUpGroupWorkSize[i] = ROUND_UP(gws[i], lws[i]);
        }
        
        res = runtime->commandQueue().enqueueNDRangeKernel(convertBufferKernel, cl::NullRange,
                                                             cl::NDRange(roundUpGroupWorkSize[0], roundUpGroupWorkSize[1]),
                                                             cl::NDRange(lws[0], lws[1]), nullptr, &event);
        MNN_CHECK_CL_SUCCESS(res, "buffer_convert_to_buffer");
        
        if (true == needWait) {
            event.wait();
        }
    } else{
        uint32_t gws[3] = {static_cast<uint32_t>(shape[2] * shape[3]),
                                      static_cast<uint32_t>(shape[1]),
                                      static_cast<uint32_t>(shape[0])};
        std::set<std::string> buildOptions;
        buildOptions.emplace("-DINPUT_FORMAT=" + std::to_string(srcDimensionFormat));
        buildOptions.emplace("-DOUTPUT_FORMAT=" + std::to_string(dstDimensionFormat));
        AddBuildOptionOfDataType(input, output, buildOptions, runtime->isSupportedFP16(), toDevice, toHost);
        auto convertBufferKernelW = runtime->buildKernelWithCache("buffer_convert_buf", "buffer_convert_to_buffer", buildOptions);
        auto convertBufferKernel = convertBufferKernelW->get();
        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;
        ret |= convertBufferKernel.setArg(idx++, gws[0]);
        ret |= convertBufferKernel.setArg(idx++, gws[1]);
        ret |= convertBufferKernel.setArg(idx++, gws[2]);
#ifdef MNN_OPENCL_SVM_ENABLE
        if(svmFlag == true && toDevice) {
            ret |= clSetKernelArgSVMPointer(convertBufferKernel.get(), idx++, (const void *)input->deviceId());
        }
        else
#endif
        {
            ret |= convertBufferKernel.setArg(idx++, openCLBuffer(input));
        }
        
        ret |= convertBufferKernel.setArg(idx++, sizeof(shape), shape);
#ifdef MNN_OPENCL_SVM_ENABLE
        if(svmFlag == true && toHost) {
            ret |= clSetKernelArgSVMPointer(convertBufferKernel.get(), idx++, (const void *)output->deviceId());
        }
        else
#endif
        {
            ret |= convertBufferKernel.setArg(idx++, openCLBuffer(output));
        }
        MNN_CHECK_CL_SUCCESS(ret, "setArg buffer_convert_to_buffer");
        
        const uint32_t maxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(convertBufferKernelW));
        const std::vector<uint32_t> lws = {16, std::max((uint32_t)1, maxWorkGroupSize / 16), 1};
        cl::Event event;
        cl_int res;
        std::vector<uint32_t> roundUpGroupWorkSize(lws.size());
        for (size_t i = 0; i < lws.size(); ++i) {
            roundUpGroupWorkSize[i] = ROUND_UP(gws[i], lws[i]);
        }
        
        res = runtime->commandQueue().enqueueNDRangeKernel(convertBufferKernel, cl::NullRange,
                                                           cl::NDRange(roundUpGroupWorkSize[0], roundUpGroupWorkSize[1], roundUpGroupWorkSize[2]),
                                                           cl::NDRange(lws[0], lws[1], lws[2]), nullptr, &event);
        MNN_CHECK_CL_SUCCESS(res, "buffer_convert_to_buffer");
        
        if (true == needWait) {
            event.wait();
        }
    }
    return true;
}

bool convertBetweenAHDandCLmem(const Tensor *input, const Tensor *output, OpenCLRuntime *runtime, int memType, bool toDevice, bool toHost) {
    std::set<std::string> buildOptions;
    auto srcDimensionFormat = TensorUtils::getDescribe(input)->dimensionFormat;
    auto dstDimensionFormat = TensorUtils::getDescribe(output)->dimensionFormat;
    if(runtime->getGpuMemType() == IMAGE){
        buildOptions.emplace("-DUSE_IMAGE");
    }
    
    buildOptions.emplace("-DINPUT_FORMAT=" + std::to_string(srcDimensionFormat));
    buildOptions.emplace("-DOUTPUT_FORMAT=" + std::to_string(dstDimensionFormat));
    std::vector<int> outputShape;
    std::shared_ptr<KernelWrap> kernelW;
    if(toDevice){
        buildOptions.emplace("-DSHARED_TO_CL");
        kernelW = runtime->buildKernelWithCache("glmem_convert", "gl_to_cl", buildOptions, nullptr, output);
        outputShape = tensorShapeFormat(output);
    } else if(toHost){
        buildOptions.emplace("-DCL_TO_SHARED");
        kernelW = runtime->buildKernelWithCache("glmem_convert", "cl_to_gl", buildOptions, input, nullptr);
        outputShape = tensorShapeFormat(input);
    }else{
        MNN_PRINT("convertGLMemBetweenCLmem only support toDevice or toHost!\n");
        return false;
    }
    
    int shape[4] = {outputShape[0], outputShape[3], outputShape[1], outputShape[2]};//N C H W
    uint32_t gws[3] = {static_cast<uint32_t>(UP_DIV(shape[3], 4)),
                                  static_cast<uint32_t>(UP_DIV(shape[1], 4)),
                                  static_cast<uint32_t>(shape[0] * shape[2])};
    auto Kernel = kernelW->get();
    uint32_t idx = 0;
    cl_int ret = CL_SUCCESS;
    ret |= Kernel.setArg(idx++, gws[0]);
    ret |= Kernel.setArg(idx++, gws[1]);
    ret |= Kernel.setArg(idx++, gws[2]);
    if(toDevice){
        ret |= Kernel.setArg(idx++, *((CLSharedMemReleaseBuffer*)TensorUtils::getSharedMem(input))->getMem());
    }else{
        if(runtime->getGpuMemType() == IMAGE) {
            ret |= Kernel.setArg(idx++, openCLImage(input));
        }
        else {
            ret |= Kernel.setArg(idx++, openCLBuffer(input));
        }
    }
    if (toHost){
        ret |= Kernel.setArg(idx++, *((CLSharedMemReleaseBuffer*)TensorUtils::getSharedMem(output))->getMem());
    }else{
        if(runtime->getGpuMemType() == IMAGE) {
            ret |= Kernel.setArg(idx++, openCLImage(output));
        } else {
            ret |= Kernel.setArg(idx++, openCLBuffer(output));
        }
    }
    ret |= Kernel.setArg(idx++, sizeof(shape), shape);
    MNN_CHECK_CL_SUCCESS(ret, "setArg glmem_convert");
    
    const uint32_t maxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(kernelW));
    const std::vector<uint32_t> lws = {16, std::max((uint32_t)1, maxWorkGroupSize / 16), 1};
    cl::Event event;
    cl_int res;
    std::vector<uint32_t> roundUpGroupWorkSize(lws.size());
    for (size_t i = 0; i < lws.size(); ++i) {
        roundUpGroupWorkSize[i] = ROUND_UP(gws[i], lws[i]);
    }
    
    res = runtime->commandQueue().enqueueNDRangeKernel(Kernel, cl::NullRange,
                                                       cl::NDRange(roundUpGroupWorkSize[0], roundUpGroupWorkSize[1], roundUpGroupWorkSize[2]),
                                                       cl::NDRange(lws[0], lws[1], lws[2]), nullptr, &event);
    event.wait();
    MNN_CHECK_CL_SUCCESS(res, "glmem_convert");
    return true;
}

} // namespace OpenCL
} // namespace MNN
#endif /* MNN_OPENCL_BUFFER_CLOSED */
