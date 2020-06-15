//
//  UnaryExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/execution/UnaryExecution.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "backend/opencl/core/OpenCLBackend.hpp"

namespace MNN {
namespace OpenCL {

UnaryExecution::UnaryExecution(const std::string& compute, Backend* backend) : Execution(backend) {
    auto openCLBackend = static_cast<OpenCLBackend*>(backend);
    std::set<std::string> buildOptions;
    buildOptions.emplace(" -DOPERATOR=" + compute);
    // FUNC_PRINT_ALL(buildOptions.begin()->c_str(), s);
    auto runtime      = openCLBackend->getOpenCLRuntime();
    mKernel           = runtime->buildKernel("unary", "unary", buildOptions);
    mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel));
}
ErrorCode UnaryExecution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
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
    mKernel.setArg(idx++, openCLImage(input));
    mKernel.setArg(idx++, openCLImage(output));

    const std::vector<uint32_t> lws =
        localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, openCLBackend->getOpenCLRuntime());
    mLocalSize = lws;
    return NO_ERROR;
}

ErrorCode UnaryExecution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start UnaryExecution onExecute...");
#endif
    auto openCLBackend = static_cast<OpenCLBackend*>(backend());
    run3DKernelDefault(mKernel, mGlobalWorkSize, mLocalSize, openCLBackend->getOpenCLRuntime());

#ifdef LOG_VERBOSE
    MNN_PRINT("end UnaryExecution onExecute...");
#endif
    return NO_ERROR;
}

class UnaryCreator : public OpenCLBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        if (op->type() == OpType_UnaryOp) {
            switch (op->main_as_UnaryOp()->opType()) {
                case UnaryOpOperation_SQUARE:
                    return new UnaryExecution("in*in", backend);
                case UnaryOpOperation_ERF:
                    return new UnaryExecution("erf(in)", backend);
                case UnaryOpOperation_ERFC:
                    return new UnaryExecution("erfc(in)", backend);
                case UnaryOpOperation_SQRT:
                    return new UnaryExecution("sqrt(in)", backend);
                case UnaryOpOperation_RSQRT:
                    return new UnaryExecution("rsqrt(in)", backend);
                case UnaryOpOperation_ABS:
                    return new UnaryExecution("fabs(in)", backend);
                case UnaryOpOperation_SIN:
                    return new UnaryExecution("sin(in)", backend);
                case UnaryOpOperation_COS:
                    return new UnaryExecution("cos(in)", backend);
                case UnaryOpOperation_SIGN:
                    return new UnaryExecution("sign(in)", backend);
                case UnaryOpOperation_EXP:
                    return new UnaryExecution("exp(in)", backend);
                case UnaryOpOperation_NEG:
                    return new UnaryExecution("-(in)", backend);
                case UnaryOpOperation_TAN:
                    return new UnaryExecution("tan(in)", backend);
                case UnaryOpOperation_CEIL:
                    return new UnaryExecution("ceil(in)", backend);
                case UnaryOpOperation_LOG1P:
                    return new UnaryExecution("log1p(in)", backend);
                case UnaryOpOperation_FLOOR:
                    return new UnaryExecution("floor(in)", backend);
                case UnaryOpOperation_ROUND:
                    return new UnaryExecution("round(in)", backend);
                case UnaryOpOperation_RECIPROCAL:
                    return new UnaryExecution("native_recip(in)", backend);
                case UnaryOpOperation_LOG:
                    return new UnaryExecution("native_log(in+(FLOAT4)(0.0000001))", backend);
                default:
                    break;
            }
            return nullptr;
        }
        if (op->type() == OpType_Sigmoid) {
            return new UnaryExecution("native_recip((float4)1+native_exp(-in))", backend);
        }
        if (op->type() == OpType_TanH) {
            return new UnaryExecution("tanh(in)", backend);
        }
        return nullptr;
    }
};

OpenCLCreatorRegister<UnaryCreator> __UnaryExecution(OpType_UnaryOp);
OpenCLCreatorRegister<UnaryCreator> __SigmoidExecution(OpType_Sigmoid);
OpenCLCreatorRegister<UnaryCreator> __TanhExecution(OpType_TanH);
} // namespace OpenCL
} // namespace MNN
