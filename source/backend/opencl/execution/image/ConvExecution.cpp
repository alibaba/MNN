//
//  ConvExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ConvExecution.hpp"
#include "ConvWinograd.hpp"
#include "core/ConvolutionCommon.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "backend/opencl/core/OpenCLBackend.hpp"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"

#define UNIT 4
namespace MNN {
namespace OpenCL {

ConvCommonExecution::ConvCommonExecution(const Convolution2D *conv2dParams, Backend *backend) : Execution(backend) {
    auto openclBackend       = (OpenCLBackend *)backend;
    int biasSize             = conv2dParams->bias()->size();
    const float *biasDataPtr = conv2dParams->bias()->data();
    
    int buffer_size = ALIGN_UP4(biasSize);
    if(openclBackend->getOpenCLRuntime()->isWeightCpuTransHalf()) {
        buffer_size *= sizeof(half_float::half);
    } else {
        buffer_size *= sizeof(float);
    }
    cl::Buffer biasBuffer(openclBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buffer_size);
    cl_int error;
    auto biasPtrCL = openclBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(
        biasBuffer, true, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &error);
    if(biasPtrCL != nullptr && error == CL_SUCCESS){
        if(openclBackend->getOpenCLRuntime()->isWeightCpuTransHalf()){
            for(int i=0; i<biasSize; i++) {
                ((half_float::half*)biasPtrCL)[i] = (half_float::half)(biasDataPtr[i]);
            }
            for(int i=biasSize; i<ALIGN_UP4(biasSize); i++) {
                ((half_float::half*)biasPtrCL)[i] = (half_float::half)(0.0f);
            }
        }else{
            ::memset(biasPtrCL, 0, ALIGN_UP4(biasSize) * sizeof(float));
            ::memcpy(biasPtrCL, biasDataPtr, biasSize * sizeof(float));
        }
    }else{
        MNN_ERROR("Map error biasPtrCL == nullptr \n");
    }
    openclBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(biasBuffer, biasPtrCL);
    mBias.reset(Tensor::createDevice<float>({1, 1, 1, biasSize}));
    backend->onAcquireBuffer(mBias.get(), Backend::STATIC);
    copyBufferToImage(openclBackend->getOpenCLRuntime(), biasBuffer, openCLImage(mBias.get()), UP_DIV(biasSize, 4), 1);
}
ConvCommonExecution::~ConvCommonExecution() {
    MNN_ASSERT(nullptr != mBias);
    backend()->onReleaseBuffer(mBias.get(), Backend::STATIC);
}

ConvExecution::ConvExecution(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, const MNN::Op *op, Backend *backend)
    : ConvCommonExecution(op->main_as_Convolution2D(), backend) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start ConvExecution init !\n");
#endif
    mOpenCLBackend                 = static_cast<OpenCLBackend *>(backend);
    const auto *conv2dParams       = op->main_as_Convolution2D();
    const auto *conv2dCommonParams = conv2dParams->common();
    mConv2dCommonParams            = conv2dCommonParams;
    mStrides                       = {conv2dCommonParams->strideY(), conv2dCommonParams->strideX()};
    mDilations                     = {conv2dCommonParams->dilateY(), conv2dCommonParams->dilateX()};

    auto pad = ConvolutionCommon::convolutionPad(inputs[0], outputs[0], mConv2dCommonParams);
    mPaddings[0] = pad.second;
    mPaddings[1] = pad.first;

    int kernelWidth   = conv2dCommonParams->kernelX();
    int kernelHeight  = conv2dCommonParams->kernelY();
    int outputChannel = conv2dCommonParams->outputCount();

    int weightSize             = 0;
    const float *filterDataPtr = nullptr;

    std::shared_ptr<MNN::ConvolutionCommon::Int8Common> quanCommon;
    if (nullptr != conv2dParams->quanParameter()) {
        quanCommon = ConvolutionCommon::load(conv2dParams->quanParameter(), true);
        if (nullptr == quanCommon) {
            MNN_ERROR("Memory not Enough, can't extract IDST Convolution: %s \n", op->name()->c_str());
        }
        if (quanCommon->weightFloat.get() == nullptr) {
            MNN_PRINT("quanCommon->weightFloat.get() == nullptr \n");
        }
        // Back to float
        filterDataPtr = quanCommon->weightFloat.get();
        weightSize    = quanCommon->weightFloat.size();
    } else if (nullptr == conv2dParams->weight() || nullptr == conv2dParams->bias()) {
        MNN_ERROR("%s has no weight or bias. The model may be benchmark model, please revert the weight/bias firstly\n", op->name()->c_str());
    }

    if (nullptr == filterDataPtr) {
        weightSize    = conv2dParams->weight()->size();
        filterDataPtr = conv2dParams->weight()->data();
    }
    int inputChannel = weightSize / (kernelWidth * kernelHeight * outputChannel);

    auto gpuType = mOpenCLBackend->getOpenCLRuntime()->getGpuType();

    //select opt conv method
    std::string kernelName = "conv_2d";
    if (kernelHeight == kernelWidth && kernelHeight == 1 && mPaddings[0] == 0 &&
        mPaddings[1] == 0) {
        mConv1x1Opt = (mStrides[0] == 1 && mStrides[1] == 1 && gpuType == GpuType::MALI);
#if 0
        if((gpuType == GpuType::ADRENO)){
            uint64_t useLocalSize = UNIT*UNIT*4*sizeof(float)*4;
            if(useLocalSize >= mOpenCLBackend->getOpenCLRuntime()->getMaxLocalMem()){
                mUseLocalMem = false;
            }else{
                kernelName = "conv_2d_1x1_local";
                mUseLocalMem=true;
            }
        }
#endif
        if(!mUseLocalMem){
            if(mConv1x1Opt){
                kernelName = "conv_2d_1x1_mali";
            }else{
                kernelName = "conv_2d_1x1";
            }
        }
    }

    if(mConv1x1Opt && !mUseLocalMem){
        cl_int error;
        std::shared_ptr<Tensor> filterBuffer(Tensor::createDevice<float>({UP_DIV(outputChannel, 4)*4, UP_DIV(inputChannel, 4)*4, kernelWidth, kernelHeight}));
        
        int buffer_size = filterBuffer->elementSize();
        if(mOpenCLBackend->getOpenCLRuntime()->isSupportedFP16()) {
            buffer_size *= sizeof(half_float::half);
        } else {
            buffer_size *= sizeof(float);
        }
        
        mKernelBuffer.reset(new cl::Buffer(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buffer_size));
        auto kernelBufferPtr = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(*(mKernelBuffer.get()), true, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &error);
        if(kernelBufferPtr != nullptr && error == CL_SUCCESS){
            ::memset(kernelBufferPtr, 0, buffer_size);
            for(int o = 0; o < outputChannel; o++){
                for(int i = 0 ; i < inputChannel; i++){
                    int bufferIdx = (o/4) * ROUND_UP(inputChannel, 4)*4 + (i/4)*16 + (o%4)*4 + (i%4);
                    int filterIdx = o*inputChannel + i;
                    if(mOpenCLBackend->getOpenCLRuntime()->isSupportedFP16()){
                        ((half_float::half*)kernelBufferPtr)[bufferIdx] = (half_float::half)(filterDataPtr[filterIdx]);
                    }else{
                        ((float*)kernelBufferPtr)[bufferIdx] = (float)(filterDataPtr[filterIdx]);
                    }
                }
            }
        }else{
            MNN_ERROR("Map error ptrCL == nullptr \n");
        }
        mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(*(mKernelBuffer.get()), kernelBufferPtr);

        //bias
        int biasSize             = conv2dParams->bias()->size();
        const float *biasDataPtr = conv2dParams->bias()->data();
        
        buffer_size = ALIGN_UP4(biasSize);
        if(mOpenCLBackend->getOpenCLRuntime()->isSupportedFP16()) {
            buffer_size *= sizeof(half_float::half);
        } else {
            buffer_size *= sizeof(float);
        }
        
        mBiasBuffer.reset(new cl::Buffer(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buffer_size));
        auto biasPtrCL = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(
            *(mBiasBuffer.get()), true, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &error);
        if(biasPtrCL != nullptr && error == CL_SUCCESS){
            if(mOpenCLBackend->getOpenCLRuntime()->isSupportedFP16()){
                for (int i = 0; i < biasSize; i++)
                {
                    ((half_float::half*)biasPtrCL)[i] = (half_float::half)(biasDataPtr[i]);
                }
                for(int i=biasSize; i<ALIGN_UP4(biasSize); i++) {
                    ((half_float::half*)biasPtrCL)[i] = (half_float::half)(0.0f);
                }
            }else{
                ::memset(biasPtrCL, 0, ALIGN_UP4(biasSize) * sizeof(float));
                ::memcpy(biasPtrCL, biasDataPtr, biasSize * sizeof(float));
            }
        }else{
            MNN_ERROR("Map error biasPtrCL == nullptr \n");
        }
        mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(*(mBiasBuffer.get()), biasPtrCL);

    }else{
        std::vector<int> filterImageShape{(int)inputChannel, (int)(UP_DIV(outputChannel, 4) * kernelWidth * kernelHeight)};
        std::shared_ptr<Tensor> filterBuffer(
            Tensor::createDevice<float>({outputChannel, inputChannel, kernelWidth, kernelHeight}));
        
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
        if(ptrCL != nullptr && error == CL_SUCCESS) {
            ::memset(ptrCL, 0, buffer_size);
            if(mOpenCLBackend->getOpenCLRuntime()->isWeightCpuTransHalf()){
                for(int i = 0 ; i < filterBuffer->elementSize(); i++){
                    ((half_float::half*)ptrCL)[i] = (half_float::half)(filterDataPtr[i]);
                }
            }else{
                ::memcpy(ptrCL, filterDataPtr, filterBuffer->size());
            }
        }else{
            MNN_ERROR("Map error ptrCL == nullptr \n");
        }
        mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(filterBufferCL, ptrCL);

        mFilter.reset(Tensor::createDevice<float>({1, filterImageShape[1], 1, 4 * filterImageShape[0]}));
        mOpenCLBackend->onAcquireBuffer(mFilter.get(), Backend::STATIC);
        MNN::OpenCL::ImageBufferConvertor imageBufferConvertor{mOpenCLBackend->getOpenCLRuntime()};

        std::string buildOption = "";
        if(mOpenCLBackend->getOpenCLRuntime()->isWeightCpuTransHalf() == false){
            buildOption = "-DBUFFER_INP_FP32";
        }
        imageBufferConvertor.convertBufferToImage(filterBuffer.get(), MNN::OpenCL::CONV2D_FILTER, mFilter.get(), false, buildOption);
    }

    // Create Kernel
    std::set<std::string> buildOptions;
    buildOptions.emplace("-DBIAS");
    if (mConv2dCommonParams->relu()) {
        buildOptions.emplace("-DRELU");
    } else if (mConv2dCommonParams->relu6()) {
        buildOptions.emplace("-DRELU6");
    }


    mKernel           = mOpenCLBackend->getOpenCLRuntime()->buildKernel("conv_2d", kernelName, buildOptions);
    mMaxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(mKernel));

#ifdef LOG_VERBOSE
    MNN_PRINT("end ConvExecution init !\n");
#endif
}

ConvExecution::~ConvExecution() {
    if(mUseLocalMem || !mConv1x1Opt){
        mOpenCLBackend->onReleaseBuffer(mFilter.get(), Backend::STATIC);
    }
}

ErrorCode ConvExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start ConvExecution onResize !\n");
#endif
    auto input  = inputs[0];
    auto output = outputs[0];

    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);
    const int height             = outputShape.at(1);
    const int width              = outputShape.at(2);

    const int inputHeight   = inputShape.at(1);
    const int inputWidth    = inputShape.at(2);
    const int inputChannels = inputShape.at(3);

    const int inputChannelBlocks = UP_DIV(inputChannels, 4);
    int kernelHeight = mConv2dCommonParams->kernelY();
    int kernelWidth  = mConv2dCommonParams->kernelX();
    
    auto pad = ConvolutionCommon::convolutionPad(input, output, mConv2dCommonParams);
    mPaddings[0] = pad.second;
    mPaddings[1] = pad.first;

    if (kernelHeight == kernelWidth && kernelHeight == 1 && mPaddings[0] == 0 && mPaddings[1] == 0) {
        if(mConv1x1Opt){

            auto kernel             = &mKernel;
            uint32_t idx            = 0;

            if(mUseLocalMem){
                mGlobalWorkSize = {static_cast<uint32_t>(UP_DIV(outputShape.at(3), 4)), static_cast<uint32_t>(UP_DIV(outputShape.at(2), 4)),
                static_cast<uint32_t>(outputShape.at(0) * outputShape.at(1))};
                std::vector<uint32_t> lws{UNIT, UNIT, 1};
                mLocalWorkSize = lws;
                kernel->setArg(idx++, mGlobalWorkSize[0]);
                kernel->setArg(idx++, mGlobalWorkSize[1]);
                kernel->setArg(idx++, mGlobalWorkSize[2]);
                kernel->setArg(idx++, openCLImage(input));
                kernel->setArg(idx++, openCLImage(mFilter.get()));
                kernel->setArg(idx++, openCLImage(mBias.get()));
                kernel->setArg(idx++, openCLImage(output));
                kernel->setArg(idx++, static_cast<int>(inputChannelBlocks));
                kernel->setArg(idx++, height);
                kernel->setArg(idx++, width);
            }else{
                mGlobalWorkSize = {static_cast<uint32_t>(UP_DIV(outputShape.at(3), 4) * UP_DIV(outputShape.at(2), 4)),
                           static_cast<uint32_t>(outputShape.at(0) * outputShape.at(1))};
                kernel->setArg(idx++, mGlobalWorkSize[0]);
                kernel->setArg(idx++, mGlobalWorkSize[1]);
                kernel->setArg(idx++, UP_DIV(width, 4));
                kernel->setArg(idx++, openCLImage(input));
                kernel->setArg(idx++, *mKernelBuffer.get());
                kernel->setArg(idx++, *mBiasBuffer.get());
                kernel->setArg(idx++, openCLImage(output));
                kernel->setArg(idx++, static_cast<int>(inputChannelBlocks));
                kernel->setArg(idx++, height);
                kernel->setArg(idx++, width);
                
                std::string kernelName = "conv_2d_1x1_mali";
                mLocalWorkSize = localWS2DDefault(mGlobalWorkSize, mMaxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), kernelName, mKernel).first;
            }


        }else{
            mGlobalWorkSize = {
            static_cast<uint32_t>(UP_DIV(outputShape.at(3), 4) * static_cast<uint32_t>(UP_DIV(outputShape.at(2), 4))),
            static_cast<uint32_t>(outputShape.at(0) * outputShape.at(1))};
            
            auto kernel             = &mKernel;
            uint32_t idx            = 0;
            int inputImageShape[2]  = {inputHeight, inputWidth};
            int outputImageShape[2] = {height, width};
            int stideShape[2]       = {mStrides[0], mStrides[1]};
            kernel->setArg(idx++, mGlobalWorkSize[0]);
            kernel->setArg(idx++, mGlobalWorkSize[1]);
            kernel->setArg(idx++, openCLImage(input));
            kernel->setArg(idx++, openCLImage(mFilter.get()));
            kernel->setArg(idx++, openCLImage(mBias.get()));
            kernel->setArg(idx++, openCLImage(output));
            kernel->setArg(idx++, sizeof(inputImageShape), inputImageShape);
            kernel->setArg(idx++, static_cast<int>(inputChannelBlocks));
            kernel->setArg(idx++, sizeof(outputImageShape), outputImageShape);
            kernel->setArg(idx++, sizeof(stideShape), stideShape);
            kernel->setArg(idx++, UP_DIV(width, 4));
            std::string kernelName = "conv_2d_1x1";
            mLocalWorkSize = localWS2DDefault(mGlobalWorkSize, mMaxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), kernelName, mKernel).first;
        }
    } else {
        mGlobalWorkSize         = {static_cast<uint32_t>(UP_DIV(outputShape.at(3), 4) * UP_DIV(outputShape.at(2), 4)),
                           static_cast<uint32_t>(outputShape.at(0) * outputShape.at(1))};

        int inputImageShape[2]  = {inputHeight, inputWidth};
        int outputImageShape[2] = {height, width};
        int kernelShape[2]      = {kernelHeight, kernelWidth};
        int strideShape[2]      = {mStrides[0], mStrides[1]};
        int paddingShape[2]     = {mPaddings[0], mPaddings[1]};
        int dilationShape[2]    = {mDilations[0], mDilations[1]};
        uint32_t idx            = 0;
        auto kernel             = &mKernel;
        kernel->setArg(idx++, mGlobalWorkSize[0]);
        kernel->setArg(idx++, mGlobalWorkSize[1]);
        kernel->setArg(idx++, openCLImage(input));
        kernel->setArg(idx++, openCLImage(mFilter.get()));
        kernel->setArg(idx++, openCLImage(mBias.get()));
        kernel->setArg(idx++, openCLImage(output));
        kernel->setArg(idx++, sizeof(inputImageShape), inputImageShape);
        kernel->setArg(idx++, inputChannelBlocks);
        kernel->setArg(idx++, sizeof(outputImageShape), outputImageShape);
        kernel->setArg(idx++, sizeof(kernelShape), kernelShape);
        kernel->setArg(idx++, sizeof(strideShape), strideShape);
        kernel->setArg(idx++, sizeof(paddingShape), paddingShape);
        kernel->setArg(idx++, sizeof(dilationShape), dilationShape);
        kernel->setArg(idx++, UP_DIV(width, 4));
        
        std::string kernelName = "conv_2d";
        mLocalWorkSize = localWS2DDefault(mGlobalWorkSize, mMaxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), kernelName, mKernel).first;
    }

#ifdef LOG_VERBOSE
    MNN_PRINT("end ConvExecution onResize !\n");
#endif
    return NO_ERROR;
}

ErrorCode ConvExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start ConvExecution onExecute !\n");
#endif
    if(mUseLocalMem){
    #ifdef ENABLE_OPENCL_TIME_PROFILER
        cl::Event event;
        run3DKernelDefault(mKernel, mGlobalWorkSize, mLocalWorkSize,
                           mOpenCLBackend->getOpenCLRuntime(), &event);
        
        float costTime = mOpenCLBackend->getOpenCLRuntime()->getCostTime(&event);
        MNN_PRINT("kernel cost:%f    us Conv UseLocalMem\n",costTime);
    #else
        run3DKernelDefault(mKernel, mGlobalWorkSize, mLocalWorkSize,
                           mOpenCLBackend->getOpenCLRuntime());
    #endif
    }
    
#ifdef ENABLE_OPENCL_TIME_PROFILER
    cl::Event event;
    runKernel2D(mKernel, mGlobalWorkSize, mLocalWorkSize,
                mOpenCLBackend->getOpenCLRuntime(), &event);
    
    int costTime = (int)mOpenCLBackend->getOpenCLRuntime()->getCostTime(&event);
    MNN_PRINT("kernel cost:%d    us Conv2D\n",costTime);
#else
    runKernel2D(mKernel, mGlobalWorkSize, mLocalWorkSize,
                mOpenCLBackend->getOpenCLRuntime());
#endif
    
#ifdef LOG_VERBOSE
    MNN_PRINT("end ConvExecution onExecute !\n");
#endif
    return NO_ERROR;
}

class ConvolutionCreator : public OpenCLBackend::Creator {
public:
    virtual ~ConvolutionCreator() = default;
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        if (inputs.size() > 1) {
            return nullptr;
        }
        if (nullptr != op->main_as_Convolution2D()->quanParameter()) {
            auto quan = op->main_as_Convolution2D()->quanParameter();
            if (1 == quan->type() || 2 == quan->type()) {
                if (quan->has_scaleInt()) {
                    // Don't support IDST-int8 because of error
                    return nullptr;
                }
            }
        }

        auto conv2D = op->main_as_Convolution2D();
        if (ConvWinograd::valid(conv2D->common(), inputs[0])) {
            return new ConvWinograd(conv2D, backend);
        }

        return new ConvExecution(inputs, outputs, op, backend);
    }
};

OpenCLCreatorRegister<ConvolutionCreator> __conv_op(OpType_Convolution, IMAGE);

} // namespace OpenCL
} // namespace MNN
