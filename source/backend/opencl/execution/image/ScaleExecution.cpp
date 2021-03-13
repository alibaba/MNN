//
//  ScaleExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/execution/image/ScaleExecution.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"

namespace MNN {
namespace OpenCL {

ScaleExecution::ScaleExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend)
    : Execution(backend) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start ScaleExecution init !\n");
#endif
    auto openclBackend        = (OpenCLBackend *)backend;
    mOpenCLBackend            = static_cast<OpenCLBackend *>(backend);
    const auto *scaleParams   = op->main_as_Scale();
    int scaleSize             = scaleParams->scaleData()->size();
    const float *scaleDataPtr = scaleParams->scaleData()->data();
        
    int buffer_size = ALIGN_UP4(scaleSize);
    if(mOpenCLBackend->getOpenCLRuntime()->isWeightCpuTransHalf()) {
        buffer_size *= sizeof(half_float::half);
    } else {
        buffer_size *= sizeof(float);
    }
    cl::Buffer scaleBuffer(openclBackend->getOpenCLRuntime()->context(), CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, buffer_size);
    cl_int error;
    auto scalePtrCL = openclBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(
        scaleBuffer, true, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &error);
    if(nullptr != scalePtrCL && error == CL_SUCCESS){
        if(mOpenCLBackend->getOpenCLRuntime()->isWeightCpuTransHalf()){
            for (int i = 0; i < scaleSize; i++) {
                ((half_float::half *)scalePtrCL)[i] = (half_float::half)(scaleDataPtr[i]);
            }
            for(int i=scaleSize; i<ALIGN_UP4(scaleSize); i++) {
                ((half_float::half*)scalePtrCL)[i] = (half_float::half)(0.0f);
            }
        } else {
            ::memset(scalePtrCL, 0, buffer_size);
            ::memcpy(scalePtrCL, scaleDataPtr, scaleSize * sizeof(float));
        }
    }else{
        MNN_ERROR("Map error scalePtrCL == nullptr \n");
    }
    openclBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(scaleBuffer, scalePtrCL);

    mScale.reset(Tensor::createDevice<float>({1, 1, 1, scaleSize}));
    backend->onAcquireBuffer(mScale.get(), Backend::STATIC);
    copyBufferToImage(openclBackend->getOpenCLRuntime(), scaleBuffer, openCLImage(mScale.get()), UP_DIV(scaleSize, 4),
                      1);

    std::set<std::string> buildOptions;
    if (nullptr != scaleParams->biasData() && nullptr != scaleParams->biasData()->data()) {
        int biasSize = scaleParams->biasData()->size();
        MNN_ASSERT(biasSize == scaleSize);
        const float *biasDataPtr = scaleParams->biasData()->data();
        
        int buffer_size = ALIGN_UP4(biasSize);
        if(openclBackend->getOpenCLRuntime()->isWeightCpuTransHalf()) {
            buffer_size *= sizeof(half_float::half);
        } else {
            buffer_size *= sizeof(float);
        }
        cl::Buffer biasBuffer(openclBackend->getOpenCLRuntime()->context(), CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, buffer_size);
        cl_int error;
        auto biasPtrCL = openclBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(
            biasBuffer, true, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &error);
        if(nullptr != biasPtrCL && error == CL_SUCCESS){
            if(mOpenCLBackend->getOpenCLRuntime()->isWeightCpuTransHalf()){
                for (int i = 0; i < biasSize; i++) {
                    ((half_float::half *)biasPtrCL)[i] = (half_float::half)(biasDataPtr[i]);
                }
                for(int i=biasSize; i<ALIGN_UP4(biasSize); i++) {
                    ((half_float::half*)biasPtrCL)[i] = (half_float::half)(0.0f);
                }
            } else {
                ::memset(biasPtrCL, 0, buffer_size);
                ::memcpy(biasPtrCL, biasDataPtr, biasSize * sizeof(float));
            }
        }else{
            MNN_ERROR("Map error biasPtrCL == nullptr \n");
        }
        openclBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(biasBuffer, biasPtrCL);
        std::shared_ptr<Tensor> bias;
        bias.reset(Tensor::createDevice<float>({1, 1, 1, biasSize}));
        backend->onAcquireBuffer(bias.get(), Backend::STATIC);
        copyBufferToImage(openclBackend->getOpenCLRuntime(), biasBuffer, openCLImage(bias.get()), UP_DIV(biasSize, 4),
                          1);
        mBias = bias;
        buildOptions.emplace("-DHAS_BIAS");
        mHasBias = true;
    }
    std::string kernelName = "scale";
    auto runtime           = mOpenCLBackend->getOpenCLRuntime();
    mKernel                = runtime->buildKernel("scale", kernelName, buildOptions);
    mMaxWorkGroupSize      = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel));

#ifdef LOG_VERBOSE
    MNN_PRINT("end ScaleExecution init !\n");
#endif
}

ScaleExecution::~ScaleExecution() {
    if (nullptr != mBias) {
        mOpenCLBackend->onReleaseBuffer(mBias.get(), Backend::STATIC);
    }
    mOpenCLBackend->onReleaseBuffer(mScale.get(), Backend::STATIC);
}

ErrorCode ScaleExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start ScaleExecution onResize !\n");
#endif

#ifdef LOG_VERBOSE
    MNN_PRINT("end ScaleExecution onResize !\n");
#endif
    std::vector<int> inputShape = tensorShapeFormat(inputs[0]);

    const int batch    = inputShape.at(0);
    const int height   = inputShape.at(1);
    const int width    = inputShape.at(2);
    const int channels = inputShape.at(3);

    const int channelBlocks = UP_DIV(channels, 4);

    const std::vector<uint32_t> &gws = {static_cast<uint32_t>(channelBlocks),
                                        static_cast<uint32_t>(width),
                                        static_cast<uint32_t>(height * batch)};
    
    uint32_t idx                     = 0;
    mKernel.setArg(idx++, gws[0]);
    mKernel.setArg(idx++, gws[1]);
    mKernel.setArg(idx++, gws[2]);

    mKernel.setArg(idx++, openCLImage(inputs[0]));
    mKernel.setArg(idx++, openCLImage(mScale.get()));
    if (mHasBias) {
        mKernel.setArg(idx++, openCLImage(mBias.get()));
    }
    mKernel.setArg(idx++, openCLImage(outputs[0]));
    
    std::string name = "scale";
    mLWS = localWS3DDefault(gws, mMaxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), name, mKernel).first;
    for (size_t i = 0; i < mLWS.size(); ++i) {
        mGWS[i] = ROUND_UP(gws[i], std::max((uint32_t)1, mLWS[i]));
    }
    return NO_ERROR;
}

ErrorCode ScaleExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start ScaleExecution onExecute !\n");
#endif
 
#ifdef ENABLE_OPENCL_TIME_PROFILER
    cl::Event event;
    run3DKernelDefault(mKernel, mGWS, mLWS, mOpenCLBackend->getOpenCLRuntime(), &event);
    
    int costTime = (int)mOpenCLBackend->getOpenCLRuntime()->getCostTime(&event);
    MNN_PRINT("kernel cost:%d    us Softmax\n",costTime);
#else
    run3DKernelDefault(mKernel, mGWS, mLWS, mOpenCLBackend->getOpenCLRuntime());
#endif

#ifdef LOG_VERBOSE
    MNN_PRINT("end ScaleExecution onExecute !\n");
#endif
    return NO_ERROR;
}

OpenCLCreatorRegister<TypedCreator<ScaleExecution>> __scale_op(OpType_Scale, IMAGE);

} // namespace OpenCL
} // namespace MNN
