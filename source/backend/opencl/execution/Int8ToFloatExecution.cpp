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
    std::string kernelName = "int8_to_float";
    mKernel                = runtime->buildKernel("Int8ToFloat", kernelName, buildOptions);
    mMaxWorkGroupSize      = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel));
}
Int8ToFloatExecution::~Int8ToFloatExecution() {

}

ErrorCode Int8ToFloatExecution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    const auto input = inputs[0];
    Tensor* output = outputs[0];
    const int channels      = input->channel();
    const int batch         = input->batch();
    const int width         = input->width();
    const int height        = input->height();
    const int icDiv4        = UP_DIV(channels, 4);
    const std::vector<uint32_t> gws = {static_cast<uint32_t>(icDiv4), static_cast<uint32_t>(width),
                static_cast<uint32_t>(height * batch)};
    int idx = 0;
    mKernel.setArg(idx++, gws[0]);
    mKernel.setArg(idx++, gws[1]);
    mKernel.setArg(idx++, gws[2]);
    mKernel.setArg(idx++, openCLImage(input));
    mKernel.setArg(idx++, openCLBuffer(output));
    mKernel.setArg(idx++, *(mScaleBuffer.get()));
    mKernel.setArg(idx++, height);
    mKernel.setArg(idx++, width);
    return NO_ERROR;
}

ErrorCode Int8ToFloatExecution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    const auto input = inputs[0];
    auto output      = outputs[0];

    const auto inputDataPtr = input->host<float>();
    auto outputDataPtr      = output->host<int8_t>();
    const auto scaleDataPtr = mScales->host<float>();
    const int channels      = input->channel();
    const int icDiv4        = UP_DIV(channels, 4);
    const int batch         = input->batch();
    const int batchStride   = input->stride(0);
    const int width         = input->width();
    const int height        = input->height();

    auto runtime                    = mOpenCLBackend->getOpenCLRuntime();
    const std::vector<uint32_t> gws = {static_cast<uint32_t>(icDiv4), static_cast<uint32_t>(width),
                static_cast<uint32_t>(height * batch)};

    const std::vector<uint32_t> lws = localWS3DDefault(gws, mMaxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime());
    run3DKernelDefault(mKernel, gws, lws, mOpenCLBackend->getOpenCLRuntime());

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
