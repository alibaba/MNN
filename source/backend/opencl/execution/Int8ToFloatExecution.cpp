//
//  Int8ToFloatExecution.cpp
//  MNN
//
//  Created by MNN on 2019/5/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/execution/Int8ToFloatExecution.hpp"
#include "backend/opencl/execution/InterpExecution.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "core/Concurrency.h"
#include "backend/cpu/compute/Int8FunctionsOpt.h"
#include "core/Macro.h"
#include "backend/opencl/core/OpenCLBackend.hpp"
namespace MNN {
namespace OpenCL {

Int8ToFloatExecution::Int8ToFloatExecution(Backend* backend, const MNN::Op* param) : Execution(backend) {
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
    auto scale         = param->main_as_QuantizedFloatParam();
    const int scaleLen = scale->tensorScale()->size();

    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    
    int buffer_size = ALIGN_UP4(scaleLen);
    if(mOpenCLBackend->getOpenCLRuntime()->isWeightCpuTransHalf()) {
        buffer_size *= sizeof(half_float::half);
    } else {
        buffer_size *= sizeof(float);
    }
    mScaleBuffer.reset(new cl::Buffer(runtime->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buffer_size));

    auto DeviceBuffer = (cl::Buffer*)mScaleBuffer.get();
    cl_int error                = CL_SUCCESS;
    auto bufferPtr = runtime->commandQueue().enqueueMapBuffer(*DeviceBuffer, CL_TRUE, CL_MAP_WRITE, 0,
                                                            buffer_size, nullptr, nullptr, &error);
    if (error != CL_SUCCESS) {
        MNN_ERROR("Error to map buffer in copy buffer, error=%d\n", error);
        return;
    }
    if(bufferPtr != nullptr){
        if(mOpenCLBackend->getOpenCLRuntime()->isWeightCpuTransHalf()){
            for (int i = 0; i < scaleLen; i++) {
                ((half_float::half *)bufferPtr)[i] = (half_float::half)(scale->tensorScale()->data()[i]);
            }
            for(int i=scaleLen; i<ALIGN_UP4(scaleLen); i++) {
                ((half_float::half*)bufferPtr)[i] = (half_float::half)(0.0f);
            }
        } else {
            memset(bufferPtr, 0, ALIGN_UP4(scaleLen) * sizeof(float));
            memcpy(bufferPtr, scale->tensorScale()->data(), scaleLen * sizeof(float));
        }
    }

    runtime->commandQueue().enqueueUnmapMemObject(*DeviceBuffer, bufferPtr);

    std::set<std::string> buildOptions;
    std::string kernelName = "int8_to_float";
    if(mOpenCLBackend->getOpenCLRuntime()->isWeightCpuTransHalf() == false){
        buildOptions.emplace("-DBUFFER_INP_FP32");
    }
    mKernel                = runtime->buildKernel("Int8ToFloat", kernelName, buildOptions);
    mMaxWorkGroupSize      = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel));
}
Int8ToFloatExecution::~Int8ToFloatExecution() {

}

ErrorCode Int8ToFloatExecution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    const auto input = inputs[0];
    Tensor* output = outputs[0];
    std::vector<int> inputShape  = tensorShapeFormat(input);
    const int channels      = inputShape[3];
    const int batch         = inputShape[0];
    const int width         = inputShape[2];
    const int height        = inputShape[1];
    
    const int icDiv4        = UP_DIV(channels, 4);
    mGWS = {static_cast<uint32_t>(icDiv4),
            static_cast<uint32_t>(width),
            static_cast<uint32_t>(height * batch)
            };
    
    int idx = 0;
    mKernel.setArg(idx++, mGWS[0]);
    mKernel.setArg(idx++, mGWS[1]);
    mKernel.setArg(idx++, mGWS[2]);
    mKernel.setArg(idx++, openCLImage(input));
    mKernel.setArg(idx++, openCLBuffer(output));
    mKernel.setArg(idx++, *(mScaleBuffer.get()));
    mKernel.setArg(idx++, height);
    mKernel.setArg(idx++, width);
    std::string name = "int8Tofloat";
    mLWS = localWS3DDefault(mGWS, mMaxWorkGroupSize,
                           mOpenCLBackend->getOpenCLRuntime(), name, mKernel);
    return NO_ERROR;
}

ErrorCode Int8ToFloatExecution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    
#ifdef LOG_VERBOSE
    MNN_PRINT("Start Int8ToFloatExecution onExecute... \n");
#endif
    
#ifdef ENABLE_OPENCL_TIME_PROFILER
    cl::Event event;
    run3DKernelDefault(mKernel, mGWS, mLWS,
                       mOpenCLBackend->getOpenCLRuntime(),
                       &event);
    
    int costTime = (int)mOpenCLBackend->getOpenCLRuntime()->getCostTime(&event);
    MNN_PRINT("kernel cost:%d    us FloatToInt8\n",costTime);
#else
    run3DKernelDefault(mKernel, mGWS, mLWS,
                       mOpenCLBackend->getOpenCLRuntime());
#endif
    
#ifdef LOG_VERBOSE
    MNN_PRINT("End Int8ToFloatExecution onExecute... \n");
#endif
    
    return NO_ERROR;
}

class Int8ToFloatExecutionCreator : public OpenCLBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        return new Int8ToFloatExecution(backend, op);
    }
};

OpenCLCreatorRegister<Int8ToFloatExecutionCreator> __int8_to_float_op_(OpType_Int8ToFloat);
}
} // namespace MNN
