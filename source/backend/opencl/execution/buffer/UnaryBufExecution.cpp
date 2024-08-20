//
//  UnaryBufExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED
#include "backend/opencl/execution/buffer/UnaryBufExecution.hpp"

namespace MNN {
namespace OpenCL {

UnaryBufExecution::UnaryBufExecution(const std::string& compute, const MNN::Op* op, Backend* backend) : CommonExecution(backend, op) {
    mBuildOptions.emplace(" -DOPERATOR=" + compute);
}
ErrorCode UnaryBufExecution::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    mUnits.resize(1);
    auto &unit = mUnits[0];
    Tensor* input      = inputs[0];
    Tensor* output     = outputs[0];
    auto openCLBackend = static_cast<OpenCLBackend*>(backend());
    auto runtime       = openCLBackend->getOpenCLRuntime();
    
    auto dataType = inputs[0]->getType();
    std::set<std::string> buildOptions = mBuildOptions;
    if (dataType.code == halide_type_int){
        buildOptions.emplace("-DOPENCL_INPUT_INT");
    }
#ifdef MNN_SUPPORT_INTEL_SUBGROUP
    if (runtime->isSupportedIntelSubgroup()) {
        return SubgrouponResize(inputs, outputs);
    }
#endif /* MNN_SUPPORT_INTEL_SUBGROUP */
    unit.kernel = runtime->buildKernel("unary_buf", "unary_buf", buildOptions, input, output);
    mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(unit.kernel));

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
    cl_int ret = CL_SUCCESS;
    ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[0]);
    ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[1]);
    ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[2]);
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(input));
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(output));
    ret |= unit.kernel->get().setArg(idx++, outputHeight);
    MNN_CHECK_CL_SUCCESS(ret, "setArg UnaryBufExecution");

    std::string kernelName = "unary_buf";
    mLocalSize = localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, openCLBackend->getOpenCLRuntime(), kernelName, unit.kernel).first;
    openCLBackend->recordKernel3d(unit.kernel, mGlobalWorkSize, mLocalSize);
    unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1], mGlobalWorkSize[2]};
    unit.localWorkSize = {mLocalSize[0], mLocalSize[1], mLocalSize[2]};
    return NO_ERROR;
}

#ifdef MNN_SUPPORT_INTEL_SUBGROUP
ErrorCode UnaryBufExecution::SubgrouponResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto &unit = mUnits[0];
    Tensor* input      = inputs[0];
    Tensor* output     = outputs[0];
    auto openCLBackend = static_cast<OpenCLBackend*>(backend());
    auto runtime       = openCLBackend->getOpenCLRuntime();

    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);

    int batch        = outputShape.at(0);
    int outputHeight = outputShape.at(1);
    int outputWidth  = outputShape.at(2);
    int channels     = outputShape.at(3);
    auto inputpad    = TensorUtils::getDescribe(input)->mPads;
    auto outputpad   = TensorUtils::getDescribe(output)->mPads;
    int input_c_pack = TensorUtils::getTensorChannelPack(input);
    int output_c_pack = TensorUtils::getTensorChannelPack(output);
    
    std::set<std::string> buildOptions = mBuildOptions;
    if(output->getType().code == halide_type_int) {
        if(output->getType().bits == 8){
            buildOptions.emplace("-DINTEL_DATA=uchar");
            buildOptions.emplace("-DAS_INPUT_DATA4=as_char4");
            buildOptions.emplace("-DAS_OUTPUT_DATA4=as_uchar4");
            buildOptions.emplace("-DINTEL_SUB_GROUP_READ4=intel_sub_group_block_read_uc4");
            buildOptions.emplace("-DINTEL_SUB_GROUP_WRITE4=intel_sub_group_block_write_uc4");
        } else if(output->getType().bits == 32){
            buildOptions.emplace("-DINTEL_DATA=uint");
            buildOptions.emplace("-DAS_INPUT_DATA4=as_int4");
            buildOptions.emplace("-DAS_OUTPUT_DATA4=as_uint4");
            buildOptions.emplace("-DINTEL_SUB_GROUP_READ4=intel_sub_group_block_read4");
            buildOptions.emplace("-DINTEL_SUB_GROUP_WRITE4=intel_sub_group_block_write4");
        }
    } else if(output->getType().code == halide_type_uint){
        if(output->getType().bits == 8){
            buildOptions.emplace("-DINTEL_DATA=uchar");
            buildOptions.emplace("-DAS_INPUT_DATA4=as_uchar4");
            buildOptions.emplace("-DAS_OUTPUT_DATA4=as_uchar4");
            buildOptions.emplace("-DINTEL_SUB_GROUP_READ4=intel_sub_group_block_read_uc4");
            buildOptions.emplace("-DINTEL_SUB_GROUP_WRITE4=intel_sub_group_block_write_uc4");
        } else if(output->getType().bits == 32){
            buildOptions.emplace("-DINTEL_DATA=uint");
            buildOptions.emplace("-DAS_INPUT_DATA4=as_uint4");
            buildOptions.emplace("-DAS_OUTPUT_DATA4=as_uint4");
            buildOptions.emplace("-DINTEL_SUB_GROUP_READ4=intel_sub_group_block_read4");
            buildOptions.emplace("-DINTEL_SUB_GROUP_WRITE4=intel_sub_group_block_write4");
        }
    } else {
        if(runtime->isSupportedFP16()){
            buildOptions.emplace("-DINTEL_DATA=ushort");
            buildOptions.emplace("-DAS_INPUT_DATA4=as_half4");
            buildOptions.emplace("-DAS_OUTPUT_DATA4=as_ushort4");
            buildOptions.emplace("-DINTEL_SUB_GROUP_READ4=intel_sub_group_block_read_us4");
            buildOptions.emplace("-DINTEL_SUB_GROUP_WRITE4=intel_sub_group_block_write_us4");
        }else{
            buildOptions.emplace("-DINTEL_DATA=uint");
            buildOptions.emplace("-DAS_INPUT_DATA4=as_float4");
            buildOptions.emplace("-DAS_OUTPUT_DATA4=as_uint4");
            buildOptions.emplace("-DINTEL_SUB_GROUP_READ4=intel_sub_group_block_read4");
            buildOptions.emplace("-DINTEL_SUB_GROUP_WRITE4=intel_sub_group_block_write4");
        }
    }
    std::string KernelName = "unary_buf_c" + std::to_string(input_c_pack) + "_c" + std::to_string(output_c_pack);
    unit.kernel       = runtime->buildKernel("unary_subgroup_buf", KernelName, buildOptions, input, output);
    mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(unit.kernel));

    int channelBlocks = (channels + 3) / 4;

    mGlobalWorkSize = {
        static_cast<uint32_t>(channelBlocks),
        static_cast<uint32_t>(outputWidth),
        static_cast<uint32_t>(batch * outputHeight),
    };

    if (runtime->isSupportedIntelSubgroup() && input_c_pack == 16) {
        channelBlocks = UP_DIV(channels, 16);
        mGlobalWorkSize[0] = ROUND_UP(channels, 16);
        mGlobalWorkSize[1] = UP_DIV(outputWidth, 4);
    }

    uint32_t idx = 0;
    cl_int ret = CL_SUCCESS;
    ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[0]);
    ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[1]);
    ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[2]);
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(input));
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(output));
    ret |= unit.kernel->get().setArg(idx++, outputWidth);
    ret |= unit.kernel->get().setArg(idx++, outputHeight);
    ret |= unit.kernel->get().setArg(idx++, channels);
    ret |= unit.kernel->get().setArg(idx++, static_cast<uint32_t>(inputpad.left));
    ret |= unit.kernel->get().setArg(idx++, static_cast<uint32_t>(inputpad.right));
    ret |= unit.kernel->get().setArg(idx++, static_cast<uint32_t>(outputpad.left));
    ret |= unit.kernel->get().setArg(idx++, static_cast<uint32_t>(outputpad.right));
    MNN_CHECK_CL_SUCCESS(ret, "setArg UnaryBufExecution SubGroup");

    std::string kernelName = "unary_buf";
    if (runtime->isSupportedIntelSubgroup() && input_c_pack == 16) {
        mLocalSize = {16, 1, 1};
    } else {
        mLocalSize = localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, openCLBackend->getOpenCLRuntime(), kernelName, unit.kernel).first;
    }
    openCLBackend->recordKernel3d(unit.kernel, mGlobalWorkSize, mLocalSize);
    unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1], mGlobalWorkSize[2]};
    unit.localWorkSize = {mLocalSize[0], mLocalSize[1], mLocalSize[2]};
    return NO_ERROR;
}
#endif /* MNN_SUPPORT_INTEL_SUBGROUP */

class UnaryBufCreator : public OpenCLBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        for (int i = 0; i < inputs.size(); ++i) {
            int channel = inputs[i]->channel();
            if (channel >= 16 && static_cast<OpenCLBackend *>(backend)->getOpenCLRuntime()->isSupportedIntelSubgroup()) {
                TensorUtils::setTensorChannelPack(inputs[i], 16);
            }
        }
        if (op->type() == OpType_UnaryOp) {
            switch (op->main_as_UnaryOp()->opType()) {
                case UnaryOpOperation_ABS:
                    return new UnaryBufExecution("fabs(convert_float4(in))", op, backend);
                case UnaryOpOperation_SQUARE:
                    return new UnaryBufExecution("in*in", op, backend);
                case UnaryOpOperation_RSQRT:
                    return new UnaryBufExecution("rsqrt(convert_float4(in)>(float4)(0.000001)?convert_float4(in):(float4)(0.000001))", op, backend);
                case UnaryOpOperation_NEG:
                    return new UnaryBufExecution("-(in)", op, backend);
                case UnaryOpOperation_EXP:
                    return new UnaryBufExecution("exp(convert_float4(in))", op, backend);
                case UnaryOpOperation_COS:
                    return new UnaryBufExecution("cos(convert_float4(in))", op, backend);
                case UnaryOpOperation_SIN:
                    return new UnaryBufExecution("sin(convert_float4(in))", op, backend);
                case UnaryOpOperation_TAN:
                    return new UnaryBufExecution("tan(convert_float4(in))", op, backend);
                case UnaryOpOperation_ATAN:
                    return new UnaryBufExecution("atan(convert_float4(in))", op, backend);
                case UnaryOpOperation_SQRT:
                    return new UnaryBufExecution("sqrt(convert_float4(in))", op, backend);
                case UnaryOpOperation_CEIL:
                    return new UnaryBufExecution("ceil(convert_float4(in))", op, backend);
                case UnaryOpOperation_RECIPROCAL:
                    return new UnaryBufExecution("native_recip(convert_float4(in))", op, backend);
                case UnaryOpOperation_LOG1P:
                    return new UnaryBufExecution("log1p(convert_float4(in))", op, backend);
                case UnaryOpOperation_LOG:
                    return new UnaryBufExecution("native_log(convert_float4(in)>(float4)(0.0000001)?convert_float4(in):(float4)(0.0000001))", op, backend);
                case UnaryOpOperation_FLOOR:
                    return new UnaryBufExecution("floor(convert_float4(in))", op, backend);
                case UnaryOpOperation_BNLL:
                    return new UnaryBufExecution("in>(float4)((float)0)?(in+native_log(exp(convert_float4(-(in)))+(float4)(1.0))):(native_log(exp(convert_float4(in))+(float4)(1.0)))", op, backend);
                case UnaryOpOperation_ACOSH:
                    return new UnaryBufExecution("acosh(convert_float4(in))", op, backend);
                case UnaryOpOperation_SINH:
                    return new UnaryBufExecution("sinh(convert_float4(in))", op, backend);
                case UnaryOpOperation_ASINH:
                    return new UnaryBufExecution("asinh(convert_float4(in))", op, backend);
                case UnaryOpOperation_ATANH:
                    return new UnaryBufExecution("atanh(convert_float4(in))", op, backend);
                case UnaryOpOperation_SIGN:
                    return new UnaryBufExecution("sign(convert_float4(in))", op, backend);
                case UnaryOpOperation_ROUND:
                    return new UnaryBufExecution("round(convert_float4(in))", op, backend);
                case UnaryOpOperation_COSH:
                    return new UnaryBufExecution("cosh(convert_float4(in))", op, backend);
               case UnaryOpOperation_ERF:
                    return new UnaryBufExecution("erf(convert_float4(in))", op, backend);
                case UnaryOpOperation_ERFC:
                    return new UnaryBufExecution("erfc(convert_float4(in))", op, backend);
                case UnaryOpOperation_EXPM1:
                    return new UnaryBufExecution("expm1(convert_float4(in))", op, backend);
                case UnaryOpOperation_SIGMOID:
                    return new UnaryBufExecution("native_recip((float4)1+native_exp(convert_float4(-in)))", op, backend);
                case UnaryOpOperation_TANH:
                    return new UnaryBufExecution("tanh(convert_float4(in))", op, backend);
                case UnaryOpOperation_HARDSWISH:
                    return new UnaryBufExecution("convert_float4(in)>(float4)(-3.0f)?(convert_float4(in)<(float4)(3.0f)?((convert_float4(in)*(convert_float4(in)+(float4)3.0f))/(float4)6.0f):convert_float4(in)):(float4)(0.0f)", op, backend);
                case UnaryOpOperation_GELU:
                    return new UnaryBufExecution("gelu(convert_float4(in))", op, backend);
                case UnaryOpOperation_GELU_STANDARD:
                    return new UnaryBufExecution("(erf(convert_float4(in)*(float4)0.7071067932881648)+(float4)1.0)*convert_float4(in)*(float4)0.5", op, backend);

		default:
                    break;
            }
            return nullptr;
        }
        if (op->type() == OpType_Sigmoid) {
            return new UnaryBufExecution("native_recip((float4)(1.0)+native_exp(convert_float4(-(in))))", op, backend);
        }
        if (op->type() == OpType_TanH) {
            return new UnaryBufExecution("tanh(convert_float4(in))", op, backend);
        }
        return nullptr;
    }
};

REGISTER_OPENCL_OP_CREATOR(UnaryBufCreator, OpType_UnaryOp, BUFFER);
REGISTER_OPENCL_OP_CREATOR(UnaryBufCreator, OpType_Sigmoid, BUFFER);
REGISTER_OPENCL_OP_CREATOR(UnaryBufCreator, OpType_TanH, BUFFER);

} // namespace OpenCL
} // namespace MNN
#endif /* MNN_OPENCL_BUFFER_CLOSED */
