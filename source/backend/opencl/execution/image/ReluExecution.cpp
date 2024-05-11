//
//  ReluExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/execution/image/ReluExecution.hpp"
#include "core/TensorUtils.hpp"
#include "backend/opencl/execution/image/UnaryExecution.hpp"
#include <string.h>
namespace MNN {
namespace OpenCL {

ReluExecution::ReluExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend)
    : CommonExecution(backend, op) {
    auto mOpenCLBackend       = static_cast<OpenCLBackend *>(backend);
    auto mPreluParamPtr       = op->main_as_PRelu();
    int preluSize             = mPreluParamPtr->slopeCount();
    const float *preluDataPtr = mPreluParamPtr->slope()->data();
    
    size_t buffer_size = ALIGN_UP4(preluSize) * sizeof(float);
    cl::Buffer preluBuffer(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, buffer_size);
    cl_int error;
    auto preluDataPtrCL = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(
        preluBuffer, true, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &error);
    if(preluDataPtrCL != nullptr && error == CL_SUCCESS){
        ::memset(preluDataPtrCL, 0, buffer_size);
        ::memcpy(preluDataPtrCL, preluDataPtr, preluSize * sizeof(float));
    }else{
        MNN_ERROR("Map error preluDataPtrCL == nullptr \n");
    }
    mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(preluBuffer, preluDataPtrCL);
    mPreluParam.reset(Tensor::createDevice<float>({1, 1, 1, preluSize}));
    mOpenCLBackend->onAcquireBuffer(mPreluParam.get(), Backend::STATIC);
    copyBufferToImage(mOpenCLBackend->getOpenCLRuntime(), preluBuffer, openCLImage(mPreluParam.get()),
                      UP_DIV(preluSize, 4), 1);
}
ReluExecution::~ReluExecution() {
    backend()->onReleaseBuffer(mPreluParam.get(), Backend::STATIC);
}

ErrorCode ReluExecution::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mUnits.resize(1);
    auto nhwc              = tensorShapeFormat(outputs[0]);
    int nhwcArray[4]        = {nhwc[0], nhwc[1], nhwc[2], UP_DIV(nhwc[3], 4)};

    auto imageWidth        = nhwc[2] * UP_DIV(nhwc[3], 4);
    auto imageHeight       = nhwc[0] * nhwc[1];
    int reluImageWH[2]      = {1, 1};
    int reluStride[4]       = {0, 0, 0, 1};
    cl::NDRange localSize  = {4, 4};
    cl::NDRange globalSize = {(uint32_t)UP_DIV(imageWidth, 4) * 4, (uint32_t)UP_DIV(imageHeight, 4) * 4};
    
    auto mOpenCLBackend  = static_cast<OpenCLBackend *>(backend());
    mUnits[0].kernel = mOpenCLBackend->getOpenCLRuntime()->buildKernel("binary", "binary_prelu", {"-DOPERATOR=select(in0*in1,in0,in0>=(float4)0)"}, inputs[0], outputs[0]);
    cl_int ret = CL_SUCCESS;
    ret |= mUnits[0].kernel->get().setArg(0, openCLImage(inputs[0]));
    ret |= mUnits[0].kernel->get().setArg(1, openCLImage(mPreluParam.get()));
    ret |= mUnits[0].kernel->get().setArg(2, openCLImage(outputs[0]));
    ret |= mUnits[0].kernel->get().setArg(3, nhwcArray);
    ret |= mUnits[0].kernel->get().setArg(4, reluImageWH);
    ret |= mUnits[0].kernel->get().setArg(5, reluStride);
    MNN_CHECK_CL_SUCCESS(ret, "setArg ReluExecution");

    mUnits[0].globalWorkSize = globalSize;
    mUnits[0].localWorkSize  = localSize;
    mOpenCLBackend->recordKernel2d(mUnits[0].kernel, {(uint32_t)UP_DIV(imageWidth, 4) * 4, (uint32_t)UP_DIV(imageHeight, 4) * 4}, {4, 4});
    return NO_ERROR;
}
class ReluCreator : public OpenCLBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        // There seems to be a bug on OpenCL compiler of AMD Radeon HD 7000 series.
        // When use build option -Dname=definition, definition will be truncated by
        // a comma, which violate opencl specification (quote, 'In particular, the definition will
        // be truncated by embedded newline characters'.)
        // So we use ternary operation (A ? B: C) instead of function call with comma
        // (e.g, fmax(in,(float4)(0))), when there is a Radeon GPU.
        bool isRadeonGpu = (static_cast<OpenCLBackend*>(backend)->getOpenCLRuntime()->getGpuType() == RADEON);

        if (op->type() == OpType_ReLU6) {
            char storage[256];
            float minValue = 0.0f;
            float maxValue = 6.0f;
            if (nullptr != op->main_as_Relu6()) {
                minValue = op->main_as_Relu6()->minValue();
                maxValue = op->main_as_Relu6()->maxValue();
            }
            if (isRadeonGpu) {
                std::string temp = "(in<=(float4)((float)%f)?(float4)((float)%f):(in>=(float4)((float)%f)?(float4)((float)%f):in))";
                sprintf(storage, temp.c_str(), minValue, minValue, maxValue, maxValue);
                return new UnaryExecution(storage, op, backend);
            }
            std::string temp = "clamp(in,(float4)((float)%f),(float4)((float)%f))";
            sprintf(storage, temp.c_str(), minValue, maxValue);
            return new UnaryExecution(storage, op, backend);
        }
        if (op->type() == OpType_ReLU) {
            if (op->main_as_Relu()->slope() == 0.0f) {
                if (isRadeonGpu) {
                    return new UnaryExecution("(in>(float4)((float)0)?in:(float4)((float)0))", op, backend);
                }
                return new UnaryExecution("fmax(in,(float4)((float)0))", op, backend);
            }
            auto slope         = op->main_as_Relu()->slope();
            char slopeCStr[30] = {};
            sprintf(slopeCStr, "%.8f", slope);
            std::string slopeStr = slopeCStr;
            if (isRadeonGpu) {
                return new UnaryExecution("in<(float4)((float)0)?(float)(" + slopeStr + "f)*in:in", op, backend);
            }
            return new UnaryExecution("select((float)(" + slopeStr + "f)*in,in,in>=(float4)((float)0))", op, backend);
        }
        if (op->type() == OpType_PReLU) {
            if (op->main_as_PRelu()->slopeCount() == 1) {
                auto slope         = op->main_as_PRelu()->slope()->data()[0];
                char slopeCStr[30] = {};
                sprintf(slopeCStr, "%.8f", slope);
                std::string slopeStr = slopeCStr;
                if (isRadeonGpu) {
                    return new UnaryExecution("in<(float4)((float)0)?(float)(" + slopeStr + "f)*in:in", op, backend);
                }
                return new UnaryExecution("select((float)(" + slopeStr + "f)*in,in,in>=(float4)((float)0))", op, backend);
            }
            // FUNC_PRINT(1);
            return new ReluExecution(inputs, op, backend);
        }
        return nullptr;
    }
};
REGISTER_OPENCL_OP_CREATOR(ReluCreator, OpType_ReLU, IMAGE);
REGISTER_OPENCL_OP_CREATOR(ReluCreator, OpType_PReLU, IMAGE);
REGISTER_OPENCL_OP_CREATOR(ReluCreator, OpType_ReLU6, IMAGE);

} // namespace OpenCL
} // namespace MNN
