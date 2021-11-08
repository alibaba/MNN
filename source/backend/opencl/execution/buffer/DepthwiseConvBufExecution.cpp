//
//  DepthwiseConvBufExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED

#include "backend/opencl/execution/buffer/DepthwiseConvBufExecution.hpp"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"
#include "core/ConvolutionCommon.hpp"

namespace MNN {
namespace OpenCL {

DepthwiseConvBufExecution::DepthwiseConvBufExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend)
    : ConvBufCommonExecution(op->main_as_Convolution2D(), backend) {
    mOpenCLBackend      = static_cast<OpenCLBackend *>(backend);
    mCon2dParams        = op->main_as_Convolution2D();
    mConv2dCommonParams = mCon2dParams->common();
    mStrides            = {mConv2dCommonParams->strideY(), mConv2dCommonParams->strideX()};
    mDilations          = {mConv2dCommonParams->dilateY(), mConv2dCommonParams->dilateX()};

    int kernelWidth   = mConv2dCommonParams->kernelX();
    int kernelHeight  = mConv2dCommonParams->kernelY();
    int outputChannel = mConv2dCommonParams->outputCount();

    std::vector<int> filterShape{1, outputChannel, kernelHeight, kernelWidth};
    std::vector<int> filterImageShape{(int)kernelHeight * kernelWidth, (int)UP_DIV(outputChannel, 4)};

        
    const float* filterDataPtr = nullptr;
    int filterDataSize   = 0;
    std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;
    ConvolutionCommon::getConvParameters(&quanCommon, mCon2dParams, &filterDataPtr, &filterDataSize);

    mFilter.reset(Tensor::createDevice<float>({1, ROUND_UP(filterImageShape[1], 2)/*for kernel C8 read*/, 1, 4 * filterImageShape[0]}));
    std::shared_ptr<Tensor> filterBuffer(Tensor::createDevice<float>(filterShape));
        
    int buffer_size = filterBuffer->elementSize();
    if(mOpenCLBackend->getOpenCLRuntime()->isWeightCpuTransHalf()) {
        buffer_size *= sizeof(half_float::half);
    } else {
        buffer_size *= sizeof(float);
    }
    cl::Buffer filterBufferCL(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buffer_size);
    filterBuffer->buffer().device = (uint64_t)(&filterBufferCL);
    cl_int error;
    auto ptrCL = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(filterBufferCL, true, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &error);
    if(ptrCL != nullptr && error == CL_SUCCESS){
        if(mOpenCLBackend->getOpenCLRuntime()->isWeightCpuTransHalf()){
            for (int i = 0; i < filterBuffer->elementSize(); i++) {
                ((half_float::half *)ptrCL)[i] = (half_float::half)(filterDataPtr[i]);
            }
        } else {
            ::memcpy(ptrCL, filterDataPtr, filterBuffer->size());
        }
    }else{
        MNN_ERROR("Map error ptrCL == nullptr \n");
    }
    mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(filterBufferCL, ptrCL);

    mOpenCLBackend->onAcquireBuffer(mFilter.get(), Backend::STATIC);
    MNN::OpenCL::BufferConvertor bufferConvertor{mOpenCLBackend->getOpenCLRuntime()};
        
    bool needTrans = false;
    if(mOpenCLBackend->getOpenCLRuntime()->isWeightCpuTransHalf() == false){
        needTrans = true;
    }
    bufferConvertor.convertToNC4HW4Buffer(filterBuffer.get(), MNN::OpenCL::DW_CONV2D_FILTER, mFilter.get(), needTrans);
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    std::string kernelName = "depthwise_conv2d_c4h1w2";
    if (mConv2dCommonParams->strideX() == 1 && mConv2dCommonParams->strideY() == 1 &&
        mConv2dCommonParams->dilateX() == 1 && mConv2dCommonParams->dilateY() == 1) {
        mStride_1 = true;
    }
    if(mStride_1) {
        kernelName = "depthwise_conv2d_s1_c4h1w4";
    }
    
    if (mConv2dCommonParams->relu() == true) {
        mBuildOptions.emplace("-DRELU");
    } else if (mConv2dCommonParams->relu6() == true) {
        mBuildOptions.emplace("-DRELU6");
    }

    mKernel           = runtime->buildKernel("depthwise_conv2d_buf", kernelName, mBuildOptions);
    mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel));
}

DepthwiseConvBufExecution::~DepthwiseConvBufExecution() {
    mOpenCLBackend->onReleaseBuffer(mFilter.get(), Backend::STATIC);
}

ErrorCode DepthwiseConvBufExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input                   = inputs[0];
    auto output                  = outputs[0];
    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);

    auto padding = ConvolutionCommon::convolutionPad(input, output, mConv2dCommonParams);
    mPaddings[0] = padding.second;//padY
    mPaddings[1] = padding.first;//padX
    
    const int outputHeight = outputShape.at(1);
    const int outputWidth  = outputShape.at(2);
    const int outputChannel  = outputShape.at(3);

    const int inputHeight   = inputShape.at(1);
    const int inputWidth    = inputShape.at(2);
    const int inputChannels = inputShape.at(3);

    const int inputChannelBlocks = UP_DIV(inputChannels, 4);
    const int filterHeight       = mCon2dParams->common()->kernelY();
    const int filterWidth        = mCon2dParams->common()->kernelX();
    
    int inputImageShape[2]  = {inputHeight, inputWidth};
    int outputImageShape[2] = {outputHeight, outputWidth};
    int strideShape[2]      = {mStrides[0], mStrides[1]};
    int paddingShape[2]     = {mPaddings[0], mPaddings[1]};
    int kernelShape[2]      = {filterHeight, filterWidth};
    int dilationShape[2]    = {mDilations[0], mDilations[1]};
    
    if(mStride_1) {
        // {"depthwise_conv2d_s1_c4h1w4", "depthwise_conv2d_s1_c8h1w4", "depthwise_conv2d_s1_c8h1w2"};
        const int total_kernel = 3;
        std::string kernelName[total_kernel] = {"depthwise_conv2d_s1_c4h1w4", "depthwise_conv2d_s1_c8h1w4", "depthwise_conv2d_s1_c8h1w2"};
        int itemC[total_kernel] = {4, 8, 8};
        int itemW[total_kernel] = {4, 4, 2};
        int itemH[total_kernel] = {1, 1, 1};

        int actual_kernel = total_kernel;
        
        
        if(kernelShape[0]==3 && kernelShape[1]==3 && paddingShape[0]==1 && paddingShape[1]==1) {
            //{"depthwise_conv2d_k3s1p1_c4h1w2", "depthwise_conv2d_k3s1p1_c4h2w2"}
            actual_kernel = 2;
            kernelName[0] = "depthwise_conv2d_k3s1p1_c4h1w2";
            itemC[0]      = 4;
            itemW[0]      = 2;
            itemH[0]      = 1;

            kernelName[1] = "depthwise_conv2d_k3s1p1_c4h2w2";
            itemC[1]      = 4;
            itemW[1]      = 2;
            itemH[1]      = 2;
        }
        
        if(mOpenCLBackend->getOpenCLRuntime()->getCLTuneLevel() == Normal || mOpenCLBackend->getOpenCLRuntime()->getCLTuneLevel() == Fast || mOpenCLBackend->getOpenCLRuntime()->getCLTuneLevel() == None) {
            actual_kernel = 1;
        }

        cl::Kernel kernel[total_kernel];
        std::vector<uint32_t> globalWorkSize[total_kernel];
        std::vector<uint32_t> localWorkSize[total_kernel];
        std::pair<int, int> min_cost(INT_MAX, 0);//(min_time, min_index)
        for(int knl_idx = 0; knl_idx < actual_kernel; knl_idx++) {
            kernel[knl_idx]        = mOpenCLBackend->getOpenCLRuntime()->buildKernel("depthwise_conv2d_buf", kernelName[knl_idx], mBuildOptions);
            uint32_t maxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(kernel[knl_idx]));
                        
            globalWorkSize[knl_idx] = {static_cast<uint32_t>(UP_DIV(outputShape.at(3), itemC[knl_idx]) * UP_DIV(outputShape.at(2), itemW[knl_idx])), static_cast<uint32_t>(outputShape.at(0) * UP_DIV(outputShape.at(1), itemH[knl_idx]))};
            
            uint32_t idx            = 0;
            kernel[knl_idx].setArg(idx++, globalWorkSize[knl_idx][0]);
            kernel[knl_idx].setArg(idx++, globalWorkSize[knl_idx][1]);
            kernel[knl_idx].setArg(idx++, openCLBuffer(input));
            kernel[knl_idx].setArg(idx++, openCLBuffer(mFilter.get()));
            kernel[knl_idx].setArg(idx++, openCLBuffer(mBias.get()));
            kernel[knl_idx].setArg(idx++, openCLBuffer(output));
            kernel[knl_idx].setArg(idx++, sizeof(inputImageShape), inputImageShape);
            kernel[knl_idx].setArg(idx++, static_cast<int>(inputChannels));
            kernel[knl_idx].setArg(idx++, sizeof(outputImageShape), outputImageShape);
            kernel[knl_idx].setArg(idx++, sizeof(kernelShape), kernelShape);
            kernel[knl_idx].setArg(idx++, sizeof(paddingShape), paddingShape);
            kernel[knl_idx].setArg(idx++, sizeof(dilationShape), dilationShape);
            kernel[knl_idx].setArg(idx++, sizeof(strideShape), strideShape);
            kernel[knl_idx].setArg(idx++, UP_DIV(outputWidth, itemW[knl_idx]));
            kernel[knl_idx].setArg(idx++, UP_DIV(outputChannel, 4));
            
            std::pair<std::vector<uint32_t>, int> retTune;
            retTune = gws2dLwsTune(kernel[knl_idx], globalWorkSize[knl_idx], kernelName[knl_idx], maxWorkGroupSize);
            //printf("depthwiseCovs1 %d, %d\n", knl_idx, retTune.second);
            if(min_cost.first > retTune.second) {
                min_cost.first = retTune.second;
                min_cost.second = knl_idx;
                mLocalWorkSize = {retTune.first[0], retTune.first[1]};
            }
        }
        int min_index  = min_cost.second;
        mGlobalWorkSize = {globalWorkSize[min_index][0], globalWorkSize[min_index][1]};
        
        mKernel        = mOpenCLBackend->getOpenCLRuntime()->buildKernel("depthwise_conv2d_buf", kernelName[min_index], mBuildOptions);
        
        uint32_t idx = 0;
        mKernel.setArg(idx++, mGlobalWorkSize[0]);
        mKernel.setArg(idx++, mGlobalWorkSize[1]);
        mKernel.setArg(idx++, openCLBuffer(input));
        mKernel.setArg(idx++, openCLBuffer(mFilter.get()));
        mKernel.setArg(idx++, openCLBuffer(mBias.get()));
        mKernel.setArg(idx++, openCLBuffer(output));
        mKernel.setArg(idx++, sizeof(inputImageShape), inputImageShape);
        mKernel.setArg(idx++, static_cast<int>(inputChannels));
        mKernel.setArg(idx++, sizeof(outputImageShape), outputImageShape);
        mKernel.setArg(idx++, sizeof(kernelShape), kernelShape);
        mKernel.setArg(idx++, sizeof(paddingShape), paddingShape);
        mKernel.setArg(idx++, sizeof(dilationShape), dilationShape);
        mKernel.setArg(idx++, sizeof(strideShape), strideShape);
        mKernel.setArg(idx++, UP_DIV(outputWidth, itemW[min_index]));
        mKernel.setArg(idx++, UP_DIV(outputChannel, 4));
        
        //printf("DepthwiseConvBufs1 %d, %d %d, %d %d, %d %d\n", min_index, mGlobalWorkSize[0], mGlobalWorkSize[1], mLocalWorkSize[0], mLocalWorkSize[1], outputChannel, outputWidth);

    } else {
        // {"depthwise_conv2d_c4h1w4", "depthwise_conv2d_c4h1w2", "depthwise_conv2d_c4h1w1"};
        const int total_kernel = 3;
        const std::string kernelName[total_kernel] = {"depthwise_conv2d_c4h1w1", "depthwise_conv2d_c4h1w4", "depthwise_conv2d_c4h1w2"};
        int itemC[total_kernel] = {4, 4, 4};
        int itemW[total_kernel] = {1, 4, 2};
        
        int actual_kernel = total_kernel;
        if(mOpenCLBackend->getOpenCLRuntime()->getCLTuneLevel() == Normal || mOpenCLBackend->getOpenCLRuntime()->getCLTuneLevel() == Fast || mOpenCLBackend->getOpenCLRuntime()->getCLTuneLevel() == None) {
            actual_kernel = 1;
        }

        cl::Kernel kernel[total_kernel];
        std::vector<uint32_t> globalWorkSize[total_kernel];
        std::vector<uint32_t> localWorkSize[total_kernel];
        std::pair<int, int> min_cost(INT_MAX, 0);//(min_time, min_index)
        for(int knl_idx = 0; knl_idx < actual_kernel; knl_idx++) {
            kernel[knl_idx]        = mOpenCLBackend->getOpenCLRuntime()->buildKernel("depthwise_conv2d_buf", kernelName[knl_idx], mBuildOptions);
            uint32_t maxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(kernel[knl_idx]));
                        
            globalWorkSize[knl_idx] = {static_cast<uint32_t>(UP_DIV(outputShape.at(3), itemC[knl_idx]) * UP_DIV(outputShape.at(2), itemW[knl_idx])), static_cast<uint32_t>(outputShape.at(0) * outputShape.at(1))};
            
            uint32_t idx            = 0;
            kernel[knl_idx].setArg(idx++, globalWorkSize[knl_idx][0]);
            kernel[knl_idx].setArg(idx++, globalWorkSize[knl_idx][1]);
            kernel[knl_idx].setArg(idx++, openCLBuffer(input));
            kernel[knl_idx].setArg(idx++, openCLBuffer(mFilter.get()));
            kernel[knl_idx].setArg(idx++, openCLBuffer(mBias.get()));
            kernel[knl_idx].setArg(idx++, openCLBuffer(output));
            kernel[knl_idx].setArg(idx++, sizeof(inputImageShape), inputImageShape);
            kernel[knl_idx].setArg(idx++, static_cast<int>(inputChannels));
            kernel[knl_idx].setArg(idx++, sizeof(outputImageShape), outputImageShape);
            kernel[knl_idx].setArg(idx++, sizeof(kernelShape), kernelShape);
            kernel[knl_idx].setArg(idx++, sizeof(paddingShape), paddingShape);
            kernel[knl_idx].setArg(idx++, sizeof(dilationShape), dilationShape);
            kernel[knl_idx].setArg(idx++, sizeof(strideShape), strideShape);
            kernel[knl_idx].setArg(idx++, UP_DIV(outputWidth, itemW[knl_idx]));
            kernel[knl_idx].setArg(idx++, UP_DIV(outputChannel, 4));
            
            std::pair<std::vector<uint32_t>, int> retTune;
            retTune = gws2dLwsTune(kernel[knl_idx], globalWorkSize[knl_idx], kernelName[knl_idx], maxWorkGroupSize);
            //printf("depthwiseCov!! %d, %d\n", knl_idx, retTune.second);
            if(min_cost.first > retTune.second) {
                min_cost.first = retTune.second;
                min_cost.second = knl_idx;
                mLocalWorkSize = {retTune.first[0], retTune.first[1]};
            }
        }
        int min_index  = min_cost.second;
        mGlobalWorkSize = {globalWorkSize[min_index][0], globalWorkSize[min_index][1]};
        
        mKernel        = mOpenCLBackend->getOpenCLRuntime()->buildKernel("depthwise_conv2d_buf", kernelName[min_index], mBuildOptions);
        
        
        uint32_t idx = 0;
        mKernel.setArg(idx++, mGlobalWorkSize[0]);
        mKernel.setArg(idx++, mGlobalWorkSize[1]);
        mKernel.setArg(idx++, openCLBuffer(input));
        mKernel.setArg(idx++, openCLBuffer(mFilter.get()));
        mKernel.setArg(idx++, openCLBuffer(mBias.get()));
        mKernel.setArg(idx++, openCLBuffer(output));
        mKernel.setArg(idx++, sizeof(inputImageShape), inputImageShape);
        mKernel.setArg(idx++, static_cast<int>(inputChannels));
        mKernel.setArg(idx++, sizeof(outputImageShape), outputImageShape);
        mKernel.setArg(idx++, sizeof(kernelShape), kernelShape);
        mKernel.setArg(idx++, sizeof(paddingShape), paddingShape);
        mKernel.setArg(idx++, sizeof(dilationShape), dilationShape);
        mKernel.setArg(idx++, sizeof(strideShape), strideShape);
        mKernel.setArg(idx++, UP_DIV(outputWidth, itemW[min_index]));
        mKernel.setArg(idx++, UP_DIV(outputChannel, 4));
        
        //printf("DepthwiseConvBuf!! %d, %d %d, %d %d, %d %d\n", min_index, mGlobalWorkSize[0], mGlobalWorkSize[1], mLocalWorkSize[0], mLocalWorkSize[1], outputChannel, outputWidth);
    }
    return NO_ERROR;
}

ErrorCode DepthwiseConvBufExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start DepthwiseConvBufExecution onExecute !\n");
#endif

#ifdef ENABLE_OPENCL_TIME_PROFILER
    cl::Event event;
    runKernel2D(mKernel, mGlobalWorkSize, mLocalWorkSize,
                mOpenCLBackend->getOpenCLRuntime(),
                &event);
    
    int costTime = (int)mOpenCLBackend->getOpenCLRuntime()->getCostTime(&event);
    MNN_PRINT("kernel cost:%d    us DepthwiseConvBuf\n",costTime);
#else
    runKernel2D(mKernel, mGlobalWorkSize, mLocalWorkSize,
                mOpenCLBackend->getOpenCLRuntime());
#endif

#ifdef LOG_VERBOSE
    MNN_PRINT("end DepthwiseConvBufExecution onExecute !\n");
#endif
    return NO_ERROR;
}

class DepthwiseConvolutionBufCreator : public OpenCLBackend::Creator {
public:
    virtual ~DepthwiseConvolutionBufCreator() = default;
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        
        MNN_ASSERT(inputs.size() <= 3);
        if (inputs.size() == 3) {
            MNN_PRINT("multi input depthwise conv for opencl buffer not supoort!\n");
            return nullptr;
        }
        
        MNN_ASSERT(inputs.size() == 1);
        return new DepthwiseConvBufExecution(inputs, op, backend);
    }
};

OpenCLCreatorRegister<DepthwiseConvolutionBufCreator> __DepthwiseConvBuf_op(OpType_ConvolutionDepthwise, BUFFER);

} // namespace OpenCL
} // namespace MNN
#endif /* MNN_OPENCL_BUFFER_CLOSED */
