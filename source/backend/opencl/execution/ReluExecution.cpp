//
//  ReluExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ReluExecution.hpp"
#include "TensorUtils.hpp"
#include "UnaryExecution.hpp"

namespace MNN {
namespace OpenCL {

ReluExecution::ReluExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend)
    : CommonExecution(backend) {
    auto mOpenCLBackend       = static_cast<OpenCLBackend *>(backend);
    auto mPreluParamPtr       = op->main_as_PRelu();
    int preluSize             = mPreluParamPtr->slopeCount();
    const float *preluDataPtr = mPreluParamPtr->slope()->data();
    auto preluSizeAlign       = UP_DIV(preluSize, 4) * 4;
    cl::Buffer preluBuffer(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                           preluSizeAlign * sizeof(float));
    cl_int error;
    auto preluDataPtrCL = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(
        preluBuffer, true, CL_MAP_WRITE, 0, preluSizeAlign * sizeof(float), nullptr, nullptr, &error);
    if(preluDataPtrCL != nullptr && error == CL_SUCCESS){
        ::memset(preluDataPtrCL, 0, sizeof(float) * preluSizeAlign);
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

ErrorCode ReluExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mUnits.resize(1);
    auto nhwc              = tensorShapeFormat(outputs[0]);
    int nhwcArray[]        = {nhwc[0], nhwc[1], nhwc[2], UP_DIV(nhwc[3], 4)};
    auto imageWidth        = nhwc[2] * UP_DIV(nhwc[3], 4);
    auto imageHeight       = nhwc[0] * nhwc[1];
    int reluImageWH[]      = {1, 1};
    int reluStride[]       = {0, 0, 0, 1};
    cl::NDRange localSize  = {16, 16};
    cl::NDRange globalSize = {(uint32_t)UP_DIV(imageWidth, 16) * 16, (uint32_t)UP_DIV(imageHeight, 16) * 16};

    auto runTime     = ((OpenCLBackend *)backend())->getOpenCLRuntime();
    mUnits[0].kernel = runTime->buildKernel("binary", "binary", {"-DOPERATOR=select(in0*in1,in0,in0>=(FLOAT4)0)"});
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
        if (op->type() == OpType_ReLU6) {
            return new UnaryExecution("clamp(in,(float4)0,(float4)6)", backend);
        }
        if (op->type() == OpType_ReLU) {
            if (op->main_as_Relu()->slope() == 0.0f) {
                return new UnaryExecution("fmax(in,(float4)0)", backend);
            }
            auto slope         = op->main_as_Relu()->slope();
            char slopeCStr[30] = {};
            sprintf(slopeCStr, "%.8f", slope);
            std::string slopeStr = slopeCStr;

            return new UnaryExecution("select(" + slopeStr + "f*in,in,in>=(float4)0)", backend);
        }
        if (op->type() == OpType_PReLU) {
            if (op->main_as_PRelu()->slopeCount() == 1) {
                auto slope         = op->main_as_PRelu()->slope()->data()[0];
                char slopeCStr[30] = {};
                sprintf(slopeCStr, "%.8f", slope);
                std::string slopeStr = slopeCStr;
                return new UnaryExecution("select(" + slopeStr + "f*in,in,in>=(float4)0)", backend);
            }
            // FUNC_PRINT(1);
            return new ReluExecution(inputs, op, backend);
        }
        return nullptr;
    }
};

OpenCLCreatorRegister<ReluCreator> __Relu_op(OpType_ReLU);
OpenCLCreatorRegister<ReluCreator> __PRelu_op(OpType_PReLU);
OpenCLCreatorRegister<ReluCreator> __Relu6_op(OpType_ReLU6);

} // namespace OpenCL
} // namespace MNN
