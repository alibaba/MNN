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

ErrorCode FloatToInt8Execution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
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
