//
//  UnaryBufExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED
#include "backend/opencl/execution/buffer/UnaryBufExecution.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "backend/opencl/core/OpenCLBackend.hpp"

namespace MNN {
namespace OpenCL {

UnaryBufExecution::UnaryBufExecution(const std::string& compute, Backend* backend) : Execution(backend) {
    auto openCLBackend = static_cast<OpenCLBackend*>(backend);
    std::set<std::string> buildOptions;
    buildOptions.emplace(" -DOPERATOR=" + compute);
    // FUNC_PRINT_ALL(buildOptions.begin()->c_str(), s);
    auto runtime      = openCLBackend->getOpenCLRuntime();
    mKernel           = runtime->buildKernel("unary_buf", "unary_buf", buildOptions);
    mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel));
}
ErrorCode UnaryBufExecution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    Tensor* input      = inputs[0];
    Tensor* output     = outputs[0];
    auto openCLBackend = static_cast<OpenCLBackend*>(backend());

    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);

    int batch        = outputShape.at(0);
    int outputHeight = outputShape.at(1);
    int outputWidth  = outputShape.at(2);
    int channels     = outputShape.at(3);

    int channelBlocks = (channels + 3) / 4;

    mGlobalWorkSize = {
        static_cast<uint32_t>(channelBlocks),
        static_cast<uint32_t>(outputWidth),
        static_cast<uint32_t>(batch * outputHeight),
    };

    uint32_t idx = 0;
    mKernel.setArg(idx++, mGlobalWorkSize[0]);
    mKernel.setArg(idx++, mGlobalWorkSize[1]);
    mKernel.setArg(idx++, mGlobalWorkSize[2]);
    mKernel.setArg(idx++, openCLBuffer(input));
    mKernel.setArg(idx++, openCLBuffer(output));
    mKernel.setArg(idx++, outputHeight);

    std::string kernelName = "unary_buf";
    mLocalSize =
    localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, openCLBackend->getOpenCLRuntime(), kernelName, mKernel).first;
    return NO_ERROR;
}

ErrorCode UnaryBufExecution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start UnaryBufExecution onExecute...");
#endif
    auto mOpenCLBackend = static_cast<OpenCLBackend*>(backend());
    
#ifdef ENABLE_OPENCL_TIME_PROFILER
    cl::Event event;
    run3DKernelDefault(mKernel, mGlobalWorkSize, mLocalSize,
                       mOpenCLBackend->getOpenCLRuntime(), &event);
    
    int costTime = (int)mOpenCLBackend->getOpenCLRuntime()->getCostTime(&event);
    MNN_PRINT("kernel cost:%d    us Unary\n",costTime);
#else
    run3DKernelDefault(mKernel, mGlobalWorkSize, mLocalSize,
                       mOpenCLBackend->getOpenCLRuntime());
#endif

#ifdef LOG_VERBOSE
    MNN_PRINT("end UnaryBufExecution onExecute...");
#endif
    return NO_ERROR;
}

class UnaryBufCreator : public OpenCLBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        if (op->type() == OpType_UnaryOp) {
            switch (op->main_as_UnaryOp()->opType()) {
                case UnaryOpOperation_ABS:
                    return new UnaryBufExecution("fabs(convert_float4(in))", backend);
                case UnaryOpOperation_SQUARE:
                    return new UnaryBufExecution("in*in", backend);
                case UnaryOpOperation_RSQRT:
                    return new UnaryBufExecution("rsqrt(convert_float4(in))", backend);
                case UnaryOpOperation_NEG:
                    return new UnaryBufExecution("-(in)", backend);
                case UnaryOpOperation_EXP:
                    return new UnaryBufExecution("exp(convert_float4(in))", backend);
                case UnaryOpOperation_COS:
                    return new UnaryBufExecution("cos(convert_float4(in))", backend);
                case UnaryOpOperation_SIN:
                    return new UnaryBufExecution("sin(convert_float4(in))", backend);
                case UnaryOpOperation_TAN:
                    return new UnaryBufExecution("tan(convert_float4(in))", backend);
                case UnaryOpOperation_ATAN:
                    return new UnaryBufExecution("atan(convert_float4(in))", backend);
                case UnaryOpOperation_SQRT:
                    return new UnaryBufExecution("sqrt(convert_float4(in))", backend);
                case UnaryOpOperation_CEIL:
                    return new UnaryBufExecution("ceil(convert_float4(in))", backend);
                case UnaryOpOperation_RECIPROCAL:
                    return new UnaryBufExecution("native_recip(convert_float4(in))", backend);
                case UnaryOpOperation_LOG1P:
                    return new UnaryBufExecution("log1p(convert_float4(in))", backend);
                case UnaryOpOperation_LOG:
                    return new UnaryBufExecution("native_log(convert_float4(in)>(float4)(0.0000001)?convert_float4(in):(float4)(0.0000001))", backend);
                case UnaryOpOperation_FLOOR:
                    return new UnaryBufExecution("floor(convert_float4(in))", backend);
                case UnaryOpOperation_BNLL:
                    return new UnaryBufExecution("in>(FLOAT4)((FLOAT)0)?(in+native_log(exp(convert_float4(-(in)))+(float4)(1.0))):(native_log(exp(convert_float4(in))+(float4)(1.0)))", backend);
                case UnaryOpOperation_ACOSH:
                    return new UnaryBufExecution("acosh(convert_float4(in))", backend);
                case UnaryOpOperation_SINH:
                    return new UnaryBufExecution("sinh(convert_float4(in))", backend);
                case UnaryOpOperation_ASINH:
                    return new UnaryBufExecution("asinh(convert_float4(in))", backend);
                case UnaryOpOperation_ATANH:
                    return new UnaryBufExecution("atanh(convert_float4(in))", backend);
                case UnaryOpOperation_SIGN:
                    return new UnaryBufExecution("sign(convert_float4(in))", backend);
                case UnaryOpOperation_ROUND:
                    return new UnaryBufExecution("round(convert_float4(in))", backend);
                case UnaryOpOperation_COSH:
                    return new UnaryBufExecution("cosh(convert_float4(in))", backend);
               case UnaryOpOperation_ERF:
                    return new UnaryBufExecution("erf(convert_float4(in))", backend);
                case UnaryOpOperation_ERFC:
                    return new UnaryBufExecution("erfc(convert_float4(in))", backend);
                case UnaryOpOperation_EXPM1:
                    return new UnaryBufExecution("expm1(convert_float4(in))", backend);
                case UnaryOpOperation_SIGMOID:
                    return new UnaryBufExecution("native_recip((float4)1+native_exp(convert_float4(-in)))", backend);
                case UnaryOpOperation_TANH:
                    return new UnaryBufExecution("tanh(convert_float4(in))", backend);
                case UnaryOpOperation_HARDSWISH:
                    return new UnaryBufExecution("convert_float4(in)>(float4)(-3.0f)?(convert_float4(in)<(float4)(3.0f)?((convert_float4(in)*(convert_float4(in)+(float4)3.0f))/(float4)6.0f):convert_float4(in)):(float4)(0.0f)", backend);                
		default:
                    break;
            }
            return nullptr;
        }
        if (op->type() == OpType_Sigmoid) {
            return new UnaryBufExecution("native_recip((float4)(1.0)+native_exp(convert_float4(-(in))))", backend);
        }
        if (op->type() == OpType_TanH) {
            return new UnaryBufExecution("tanh(convert_float4(in))", backend);
        }
        return nullptr;
    }
};

OpenCLCreatorRegister<UnaryBufCreator> __UnaryBuf__(OpType_UnaryOp, BUFFER);
OpenCLCreatorRegister<UnaryBufCreator> __SigmoidBuf__(OpType_Sigmoid, BUFFER);
OpenCLCreatorRegister<UnaryBufCreator> __TanhBuf__(OpType_TanH, BUFFER);
} // namespace OpenCL
} // namespace MNN
#endif /* MNN_OPENCL_BUFFER_CLOSED */
