//
//  ScaleBufExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED

#include "backend/opencl/execution/buffer/ScaleBufExecution.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {
namespace OpenCL {

ScaleBufExecution::ScaleBufExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend)
    : Execution(backend) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start ScaleBufExecution init !\n");
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

    mScale.reset(Tensor::createDevice<float>({1, 1, 1, ALIGN_UP4(scaleSize)}));
    backend->onAcquireBuffer(mScale.get(), Backend::STATIC);
        
    cl::Buffer &scaleBuffer = openCLBuffer(mScale.get());
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
        
        mBias.reset(Tensor::createDevice<float>({1, 1, 1, ALIGN_UP4(biasSize)}));
        backend->onAcquireBuffer(mBias.get(), Backend::STATIC);
        cl::Buffer &biasBuffer = openCLBuffer(mBias.get());
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

        buildOptions.emplace("-DBIAS");
        mHasBias = true;
    }

    auto runtime           = mOpenCLBackend->getOpenCLRuntime();
    mKernel                = runtime->buildKernel("scale_buf", "scale_buf", buildOptions);
    mMaxWorkGroupSize      = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel));

#ifdef LOG_VERBOSE
    MNN_PRINT("end ScaleBufExecution init !\n");
#endif
}

ScaleBufExecution::~ScaleBufExecution() {
    if (nullptr != mBias) {
        mOpenCLBackend->onReleaseBuffer(mBias.get(), Backend::STATIC);
    }
    mOpenCLBackend->onReleaseBuffer(mScale.get(), Backend::STATIC);
}

ErrorCode ScaleBufExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start ScaleBufExecution onResize !\n");
#endif

    std::vector<int> inputShape = tensorShapeFormat(inputs[0]);

    const int batch    = inputShape.at(0);
    const int height   = inputShape.at(1);
    const int width    = inputShape.at(2);
    const int channels = inputShape.at(3);

    const int channelBlocks = UP_DIV(channels, 4);

    mGlobalWorkSize = {static_cast<uint32_t>(width * channelBlocks),
            static_cast<uint32_t>(height * batch)};
    
    int shape[4] = {batch, height, width, channelBlocks};
    uint32_t idx = 0;
    mKernel.setArg(idx++, mGlobalWorkSize[0]);
    mKernel.setArg(idx++, mGlobalWorkSize[1]);
    mKernel.setArg(idx++, openCLBuffer(inputs[0]));
    mKernel.setArg(idx++, openCLBuffer(mScale.get()));
    if (mHasBias) {
        mKernel.setArg(idx++, openCLBuffer(mBias.get()));
    }
    mKernel.setArg(idx++, openCLBuffer(outputs[0]));
    mKernel.setArg(idx++, shape);

    std::string name = "scale_buf";
    mLocalWorkSize = localWS2DDefault(mGlobalWorkSize, mMaxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), name, mKernel).first;
    
    return NO_ERROR;
}

ErrorCode ScaleBufExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start ScaleBufExecution onExecute !\n");
#endif
 
#ifdef ENABLE_OPENCL_TIME_PROFILER
    cl::Event event;
    runKernel2D(mKernel, mGlobalWorkSize, mLocalWorkSize,
                mOpenCLBackend->getOpenCLRuntime(), &event);
    
    int costTime = (int)mOpenCLBackend->getOpenCLRuntime()->getCostTime(&event);
    MNN_PRINT("kernel cost:%d    us Scale\n",costTime);
#else
    runKernel2D(mKernel, mGlobalWorkSize, mLocalWorkSize,
                mOpenCLBackend->getOpenCLRuntime());
#endif
        
#ifdef LOG_VERBOSE
    MNN_PRINT("end ScaleBufExecution onExecute !\n");
#endif
    return NO_ERROR;
}

OpenCLCreatorRegister<TypedCreator<ScaleBufExecution>> __scaleBuf_op(OpType_Scale, BUFFER);

} // namespace OpenCL
} // namespace MNN
#endif /* MNN_OPENCL_BUFFER_CLOSED */
