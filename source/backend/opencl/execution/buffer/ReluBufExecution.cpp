//
//  ReluBufExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED

#include "backend/opencl/execution/buffer/ReluBufExecution.hpp"
#include "core/TensorUtils.hpp"
#include "backend/opencl/execution/buffer/UnaryBufExecution.hpp"
namespace MNN {
namespace OpenCL {

ReluBufExecution::ReluBufExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend)
    : CommonExecution(backend) {
    mOpenCLBackend       = static_cast<OpenCLBackend *>(backend);
    auto mPreluParamPtr       = op->main_as_PRelu();
    int preluSize             = mPreluParamPtr->slopeCount();
    const float *preluDataPtr = mPreluParamPtr->slope()->data();
    
    int buffer_size = ALIGN_UP4(preluSize);
    if(mOpenCLBackend->getOpenCLRuntime()->isWeightCpuTransHalf()) {
        buffer_size *= sizeof(half_float::half);
    } else {
        buffer_size *= sizeof(float);
    }
        
    mPreluParam.reset(Tensor::createDevice<float>({1, 1, 1, ALIGN_UP4(preluSize)}));
    mOpenCLBackend->onAcquireBuffer(mPreluParam.get(), Backend::STATIC);
    cl::Buffer &preluBuffer = openCLBuffer(mPreluParam.get());
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
        
    mOp = op;
}

ReluBufExecution::~ReluBufExecution() {
    mOpenCLBackend->onReleaseBuffer(mPreluParam.get(), Backend::STATIC);
}

ErrorCode ReluBufExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mUnits.resize(1);
    auto nhwc              = tensorShapeFormat(outputs[0]);
    int nhwcArray[4]        = {nhwc[0], nhwc[1], nhwc[2], UP_DIV(nhwc[3], 4)};
    auto imageWidth        = nhwc[0] * UP_DIV(nhwc[3], 4);
    auto imageHeight       = nhwc[1] * nhwc[2];
    
    std::vector<uint32_t> localSize  = {1, 1};
    std::vector<uint32_t> globalSize = {(uint32_t)imageWidth, (uint32_t)imageHeight};

    auto runTime     = mOpenCLBackend->getOpenCLRuntime();
    
    mUnits[0].kernel = runTime->buildKernel("binary_buf", "prelu_buf", {"-DOPERATOR=select(in0*in1,in0,in0>=(FLOAT4)0)"});
    mMaxWorkGroupSize      = static_cast<uint32_t>(runTime->getMaxWorkGroupSize(mUnits[0].kernel));
    int fullCount[2] = {1, 1};
    
    uint32_t index = 0;
    mUnits[0].kernel.setArg(index++, globalSize[0]);
    mUnits[0].kernel.setArg(index++, globalSize[1]);
    mUnits[0].kernel.setArg(index++, openCLBuffer(inputs[0]));
    mUnits[0].kernel.setArg(index++, openCLBuffer(mPreluParam.get()));
    mUnits[0].kernel.setArg(index++, openCLBuffer(outputs[0]));
    mUnits[0].kernel.setArg(index++, nhwcArray);

    std::string name = "prelu_buf";
    localSize = localWS2DDefault(globalSize, mMaxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), name, mUnits[0].kernel).first;
    
    mUnits[0].globalWorkSize = {globalSize[0], globalSize[1]};
    mUnits[0].localWorkSize  = {localSize[0], localSize[1]};
    return NO_ERROR;
}
class ReluBufCreator : public OpenCLBackend::Creator {
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
                return new UnaryBufExecution(storage, backend);
            }
            std::string temp = "clamp(in,(FLOAT4)((FLOAT)%f),(FLOAT4)((FLOAT)%f))";
            sprintf(storage, temp.c_str(), minValue, maxValue);
            return new UnaryBufExecution(storage, backend);
        }
        if (op->type() == OpType_ReLU) {
            if (op->main_as_Relu()->slope() == 0.0f) {
                if (isRadeonGpu) {
                    return new UnaryBufExecution("(in>(FLOAT4)((FLOAT)0)?in:(FLOAT4)((FLOAT)0))", backend);
                }
                return new UnaryBufExecution("fmax(in,(FLOAT4)((FLOAT)0))", backend);
            }
            auto slope         = op->main_as_Relu()->slope();
            char slopeCStr[30] = {};
            sprintf(slopeCStr, "%.8f", slope);
            std::string slopeStr = slopeCStr;
            if (isRadeonGpu) {
                return new UnaryBufExecution("in<(FLOAT4)((FLOAT)0)?(FLOAT)(" + slopeStr + "f)*in:in", backend);
            }
            return new UnaryBufExecution("select((FLOAT)(" + slopeStr + "f)*in,in,in>=(FLOAT4)((FLOAT)0))", backend);
        }
        if (op->type() == OpType_PReLU) {
            if (op->main_as_PRelu()->slopeCount() == 1) {
                auto slope         = op->main_as_PRelu()->slope()->data()[0];
                char slopeCStr[30] = {};
                sprintf(slopeCStr, "%.8f", slope);
                std::string slopeStr = slopeCStr;
                if (isRadeonGpu) {
                    return new UnaryBufExecution("in<(FLOAT4)((FLOAT)0)?(FLOAT)(" + slopeStr + "f)*in:in", backend);
                }
                return new UnaryBufExecution("select((FLOAT)(" + slopeStr + "f)*in,in,in>=(FLOAT4)((FLOAT)0))", backend);
            }
            return new ReluBufExecution(inputs, op, backend);
        }
        return nullptr;
    }
};

OpenCLCreatorRegister<ReluBufCreator> __ReluBuf_op(OpType_ReLU, BUFFER);
OpenCLCreatorRegister<ReluBufCreator> __PReluBuf_op(OpType_PReLU, BUFFER);
OpenCLCreatorRegister<ReluBufCreator> __Relu6Buf_op(OpType_ReLU6, BUFFER);

} // namespace OpenCL
} // namespace MNN
#endif /* MNN_OPENCL_BUFFER_CLOSED */
