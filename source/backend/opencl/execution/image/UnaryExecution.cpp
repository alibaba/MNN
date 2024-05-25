//
//  UnaryExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/execution/image/UnaryExecution.hpp"

namespace MNN {
namespace OpenCL {

UnaryExecution::UnaryExecution(const std::string& compute, const MNN::Op *op, Backend* backend) : CommonExecution(backend, op) {
    mBuildOptions.emplace(" -DOPERATOR=" + compute);
}
ErrorCode UnaryExecution::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    mUnits.resize(1);
    auto &unit = mUnits[0];
    Tensor* input      = inputs[0];
    Tensor* output     = outputs[0];
    auto openCLBackend = static_cast<OpenCLBackend*>(backend());
    auto runtime = openCLBackend->getOpenCLRuntime();
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
    
    std::set<std::string> buildOptions = mBuildOptions;
    auto dataType = inputs[0]->getType();
    if (dataType.code == halide_type_int){
        buildOptions.emplace("-DOPENCL_INPUT_INT");
    }
    unit.kernel = runtime->buildKernel("unary", "unary", buildOptions, inputs[0], outputs[0]);
    mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(unit.kernel));

    uint32_t idx = 0;
    cl_int ret = CL_SUCCESS;
    ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[0]);
    ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[1]);
    ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[2]);
    ret |= unit.kernel->get().setArg(idx++, openCLImage(input));
    ret |= unit.kernel->get().setArg(idx++, openCLImage(output));
    MNN_CHECK_CL_SUCCESS(ret, "setArg UnaryExecution");

    std::string name = "unary";
    mLocalSize = localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, runtime, name, unit.kernel).first;
    openCLBackend->recordKernel3d(unit.kernel, mGlobalWorkSize, mLocalSize);
    unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1], mGlobalWorkSize[2]};
    unit.localWorkSize = {mLocalSize[0], mLocalSize[1], mLocalSize[2]};
    return NO_ERROR;
}


class UnaryCreator : public OpenCLBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        if (op->type() == OpType_UnaryOp) {
            switch (op->main_as_UnaryOp()->opType()) {
                case UnaryOpOperation_ABS:
                    return new UnaryExecution("fabs(convert_float4(in))", op, backend);
                case UnaryOpOperation_SQUARE:
                    return new UnaryExecution("in*in", op, backend);
                case UnaryOpOperation_RSQRT:
                    return new UnaryExecution("rsqrt(convert_float4(in)>(float4)(0.000001)?convert_float4(in):(float4)(0.000001))", op, backend);
                case UnaryOpOperation_NEG:
                    return new UnaryExecution("-(in)", op, backend);
                case UnaryOpOperation_EXP:
                    return new UnaryExecution("exp(convert_float4(in))", op, backend);
                case UnaryOpOperation_COS:
                    return new UnaryExecution("cos(convert_float4(in))", op, backend);
                case UnaryOpOperation_SIN:
                    return new UnaryExecution("sin(convert_float4(in))", op, backend);
                case UnaryOpOperation_TAN:
                    return new UnaryExecution("tan(convert_float4(in))", op, backend);
                case UnaryOpOperation_ATAN:
                    return new UnaryExecution("atan(convert_float4(in))", op, backend);
                case UnaryOpOperation_SQRT:
                    return new UnaryExecution("sqrt(convert_float4(in))", op, backend);
                case UnaryOpOperation_CEIL:
                    return new UnaryExecution("ceil(convert_float4(in))", op, backend);
                case UnaryOpOperation_RECIPROCAL:
                    return new UnaryExecution("native_recip(convert_float4(in))", op, backend);
                case UnaryOpOperation_LOG1P:
                    return new UnaryExecution("log1p(convert_float4(in))", op, backend);
                case UnaryOpOperation_LOG:
                    return new UnaryExecution("native_log(convert_float4(in)>(float4)(0.0000001)?convert_float4(in):(float4)(0.0000001))", op, backend);
                case UnaryOpOperation_FLOOR:
                    return new UnaryExecution("floor(convert_float4(in))", op, backend);
                case UnaryOpOperation_BNLL:
                    return new UnaryExecution("in>(float4)((float)0)?(in+native_log(exp(convert_float4(-(in)))+(float4)(1.0))):(native_log(exp(convert_float4(in))+(float4)(1.0)))", op, backend);
                case UnaryOpOperation_ACOSH:
                    return new UnaryExecution("acosh(convert_float4(in))", op, backend);
                case UnaryOpOperation_SINH:
                    return new UnaryExecution("sinh(convert_float4(in))", op, backend);
                case UnaryOpOperation_ASINH:
                    return new UnaryExecution("asinh(convert_float4(in))", op, backend);
                case UnaryOpOperation_ATANH:
                    return new UnaryExecution("atanh(convert_float4(in))", op, backend);
                case UnaryOpOperation_SIGN:
                    return new UnaryExecution("sign(convert_float4(in))", op, backend);
                case UnaryOpOperation_ROUND:
                    return new UnaryExecution("round(convert_float4(in))", op, backend);
                case UnaryOpOperation_COSH:
                    return new UnaryExecution("cosh(convert_float4(in))", op, backend);
               case UnaryOpOperation_ERF:
                    return new UnaryExecution("erf(convert_float4(in))", op, backend);
                case UnaryOpOperation_ERFC:
                    return new UnaryExecution("erfc(convert_float4(in))", op, backend);
                case UnaryOpOperation_EXPM1:
                    return new UnaryExecution("expm1(convert_float4(in))", op, backend);
                case UnaryOpOperation_SIGMOID:
                    return new UnaryExecution("native_recip((float4)1+native_exp(convert_float4(-in)))", op, backend);
                case UnaryOpOperation_TANH:
                    return new UnaryExecution("tanh(convert_float4(in))", op, backend);
                case UnaryOpOperation_HARDSWISH:
                    return new UnaryExecution("convert_float4(in)>(float4)(-3.0f)?(convert_float4(in)<(float4)(3.0f)?((convert_float4(in)*(convert_float4(in)+(float4)3.0f))/(float4)6.0f):convert_float4(in)):(float4)(0.0f)", op, backend);
                case UnaryOpOperation_GELU:
                    return new UnaryExecution("gelu(convert_float4(in))", op, backend);
		default:
                    break;
            }
            return nullptr;
        }
        if (op->type() == OpType_Sigmoid) {
            return new UnaryExecution("native_recip((float4)(1.0)+native_exp(convert_float4(-(in))))", op, backend);
        }
        if (op->type() == OpType_TanH) {
            return new UnaryExecution("tanh(convert_float4(in))", op, backend);
        }
        return nullptr;
    }
};

REGISTER_OPENCL_OP_CREATOR(UnaryCreator, OpType_UnaryOp, IMAGE);
REGISTER_OPENCL_OP_CREATOR(UnaryCreator, OpType_Sigmoid, IMAGE);
REGISTER_OPENCL_OP_CREATOR(UnaryCreator, OpType_TanH, IMAGE);

} // namespace OpenCL
} // namespace MNN
