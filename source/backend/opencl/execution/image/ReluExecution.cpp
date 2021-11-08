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
    : CommonExecution(backend) {
    auto mOpenCLBackend       = static_cast<OpenCLBackend *>(backend);
    auto mPreluParamPtr       = op->main_as_PRelu();
    int preluSize             = mPreluParamPtr->slopeCount();
    const float *preluDataPtr = mPreluParamPtr->slope()->data();
    
    int buffer_size = ALIGN_UP4(preluSize);
    if(mOpenCLBackend->getOpenCLRuntime()->isWeightCpuTransHalf()) {
        buffer_size *= sizeof(half_float::half);
    } else {
        buffer_size *= sizeof(float);
    }
    cl::Buffer preluBuffer(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, buffer_size);
    cl_int error;
    auto preluDataPtrCL = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(
        preluBuffer, true, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &error);
    if(preluDataPtrCL != nullptr && error == CL_SUCCESS){
        if(mOpenCLBackend->getOpenCLRuntime()->isWeightCpuTransHalf()){
            for(int i=0; i<preluSize; i++) {
                ((half_float::half*)preluDataPtrCL)[i] = (half_float::half)(preluDataPtr[i]);
            }
            for(int i=preluSize; i<ALIGN_UP4(preluSize); i++) {
                ((half_float::half*)preluDataPtrCL)[i] = (half_float::half)(0.0f);
            }
        }else{
            ::memset(preluDataPtrCL, 0, buffer_size);
            ::memcpy(preluDataPtrCL, preluDataPtr, preluSize * sizeof(float));
        }
    }else{
        MNN_ERROR("Map error preluDataPtrCL == nullptr \n");
    }
    mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(preluBuffer, preluDataPtrCL);
    mPreluParam.reset(Tensor::createDevice<float>({1, 1, 1, preluSize}));
    mOpenCLBackend->onAcquireBuffer(mPreluParam.get(), Backend::STATIC);
    copyBufferToImage(mOpenCLBackend->getOpenCLRuntime(), preluBuffer, openCLImage(mPreluParam.get()),
                      UP_DIV(preluSize, 4), 1);
    mOp = op;
}
ReluExecution::~ReluExecution() {
    backend()->onReleaseBuffer(mPreluParam.get(), Backend::STATIC);
}

ErrorCode ReluExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mUnits.resize(1);
    auto nhwc              = tensorShapeFormat(outputs[0]);
    int nhwcArray[4]        = {nhwc[0], nhwc[1], nhwc[2], UP_DIV(nhwc[3], 4)};

    auto imageWidth        = nhwc[2] * UP_DIV(nhwc[3], 4);
    auto imageHeight       = nhwc[0] * nhwc[1];
    int reluImageWH[2]      = {1, 1};
    int reluStride[4]       = {0, 0, 0, 1};
    cl::NDRange localSize  = {4, 4};
    cl::NDRange globalSize = {(uint32_t)UP_DIV(imageWidth, 4) * 4, (uint32_t)UP_DIV(imageHeight, 4) * 4};

    auto runTime     = ((OpenCLBackend *)backend())->getOpenCLRuntime();
    mUnits[0].kernel = runTime->buildKernel("binary", "binary_prelu", {"-DOPERATOR=select(in0*in1,in0,in0>=(FLOAT4)0)"});
    mUnits[0].kernel.setArg(0, openCLImage(inputs[0]));
    mUnits[0].kernel.setArg(1, openCLImage(mPreluParam.get()));
    mUnits[0].kernel.setArg(2, openCLImage(outputs[0]));
    mUnits[0].kernel.setArg(3, nhwcArray);
    mUnits[0].kernel.setArg(4, reluImageWH);
    mUnits[0].kernel.setArg(5, reluStride);
    mUnits[0].globalWorkSize = globalSize;
    mUnits[0].localWorkSize  = localSize;

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
            char storage[128];
            float minValue = 0.0f;
            float maxValue = 6.0f;
            if (nullptr != op->main_as_Relu6()) {
                minValue = op->main_as_Relu6()->minValue();
                maxValue = op->main_as_Relu6()->maxValue();
            }
            if (isRadeonGpu) {
                std::string temp = "(in<=(FLOAT4)((FLOAT)%f)?(FLOAT4)((FLOAT)%f):(in>=(FLOAT4)((FLOAT)%f)?(FLOAT4)((FLOAT)%f):in))";
                sprintf(storage, temp.c_str(), minValue, minValue, maxValue, maxValue);
                return new UnaryExecution(storage, backend);
            }
            std::string temp = "clamp(in,(FLOAT4)((FLOAT)%f),(FLOAT4)((FLOAT)%f))";
            sprintf(storage, temp.c_str(), minValue, maxValue);
            return new UnaryExecution(storage, backend);
        }
        if (op->type() == OpType_ReLU) {
            if (op->main_as_Relu()->slope() == 0.0f) {
                if (isRadeonGpu) {
                    return new UnaryExecution("(in>(FLOAT4)((FLOAT)0)?in:(FLOAT4)((FLOAT)0))", backend);
                }
                return new UnaryExecution("fmax(in,(FLOAT4)((FLOAT)0))", backend);
            }
            auto slope         = op->main_as_Relu()->slope();
            char slopeCStr[30] = {};
            sprintf(slopeCStr, "%.8f", slope);
            std::string slopeStr = slopeCStr;
            if (isRadeonGpu) {
                return new UnaryExecution("in<(FLOAT4)((FLOAT)0)?(FLOAT)(" + slopeStr + "f)*in:in", backend);
            }
            return new UnaryExecution("select((FLOAT)(" + slopeStr + "f)*in,in,in>=(FLOAT4)((FLOAT)0))", backend);
        }
        if (op->type() == OpType_PReLU) {
            if (op->main_as_PRelu()->slopeCount() == 1) {
                auto slope         = op->main_as_PRelu()->slope()->data()[0];
                char slopeCStr[30] = {};
                sprintf(slopeCStr, "%.8f", slope);
                std::string slopeStr = slopeCStr;
                if (isRadeonGpu) {
                    return new UnaryExecution("in<(FLOAT4)((FLOAT)0)?(FLOAT)(" + slopeStr + "f)*in:in", backend);
                }
                return new UnaryExecution("select((FLOAT)(" + slopeStr + "f)*in,in,in>=(FLOAT4)((FLOAT)0))", backend);
            }
            // FUNC_PRINT(1);
            return new ReluExecution(inputs, op, backend);
        }
        return nullptr;
    }
};

OpenCLCreatorRegister<ReluCreator> __Relu_op(OpType_ReLU, IMAGE);
OpenCLCreatorRegister<ReluCreator> __PRelu_op(OpType_PReLU, IMAGE);
OpenCLCreatorRegister<ReluCreator> __Relu6_op(OpType_ReLU6, IMAGE);

} // namespace OpenCL
} // namespace MNN
