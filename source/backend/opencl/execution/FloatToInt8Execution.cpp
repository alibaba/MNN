//
//  FloatToInt8Execution.cpp
//  MNN
//
//  Created by MNN on 2019/5/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/execution/FloatToInt8Execution.hpp"
#include "backend/opencl/execution/InterpExecution.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "core/Concurrency.h"
#include "backend/cpu/compute/Int8FunctionsOpt.h"
#include "core/Macro.h"
#include "backend/opencl/core/OpenCLBackend.hpp"

namespace MNN {
namespace OpenCL {

FloatToInt8Execution::FloatToInt8Execution(Backend* backend, const MNN::Op* param) : Execution(backend) {
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
    auto scale         = param->main_as_QuantizedFloatParam();
    const int scaleLen = scale->tensorScale()->size();

    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    mScaleBuffer.reset(new cl::Buffer(runtime->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, ALIGN_UP4(scaleLen)*sizeof(float)));

    auto DeviceBuffer = (cl::Buffer*)mScaleBuffer.get();
    cl_int error                = CL_SUCCESS;
    auto bufferPtr = runtime->commandQueue().enqueueMapBuffer(*DeviceBuffer, CL_TRUE, CL_MAP_WRITE, 0,
                                                                         scaleLen, nullptr, nullptr, &error);
    if (error != CL_SUCCESS) {
        MNN_ERROR("Error to map buffer in copy buffer, error=%d\n", error);
        return;
    }
    if(bufferPtr != nullptr){
        memset(bufferPtr, 0, ALIGN_UP4(scaleLen) * sizeof(float));
        memcpy(bufferPtr, scale->tensorScale()->data(), scaleLen * sizeof(float));
    }

    runtime->commandQueue().enqueueUnmapMemObject(*DeviceBuffer, bufferPtr);

    std::set<std::string> buildOptions;
    std::string kernelName = "float_to_int8";
    mKernel                = runtime->buildKernel("FloatToInt8", kernelName, buildOptions);
    mMaxWorkGroupSize      = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel));
}
FloatToInt8Execution::~FloatToInt8Execution() {

}

ErrorCode FloatToInt8Execution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    const auto input = inputs[0];
    Tensor* output = outputs[0];
    const int channels      = input->channel();
    const int batch         = input->batch();
    const int width         = input->width();
    const int height        = input->height();
    const int icDiv4        = UP_DIV(channels, 4);
    mGWS = {static_cast<uint32_t>(icDiv4),
            static_cast<uint32_t>(width),
            static_cast<uint32_t>(height * batch)
            };
    
    mLWS = localWS3DDefault(mGWS, mMaxWorkGroupSize,
                            mOpenCLBackend->getOpenCLRuntime());
    
    int idx = 0;
    mKernel.setArg(idx++, mGWS[0]);
    mKernel.setArg(idx++, mGWS[1]);
    mKernel.setArg(idx++, mGWS[2]);
    mKernel.setArg(idx++, openCLImage(input));
    mKernel.setArg(idx++, openCLBuffer(output));
    mKernel.setArg(idx++, *(mScaleBuffer.get()));
    mKernel.setArg(idx++, height);
    mKernel.setArg(idx++, width);
    return NO_ERROR;
}

ErrorCode FloatToInt8Execution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start FloatToInt8Execution onExecute... \n");
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
    MNN_PRINT("End FloatToInt8Execution onExecute... \n");
#endif
    return NO_ERROR;
}

class FloatToInt8ExecutionCreator : public OpenCLBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        return new FloatToInt8Execution(backend, op);
    }
};

OpenCLCreatorRegister<FloatToInt8ExecutionCreator> __float_to_int8_op_(OpType_FloatToInt8);
}
} // namespace MNN
