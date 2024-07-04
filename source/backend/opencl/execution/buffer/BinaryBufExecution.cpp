//
//  BinaryBufExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED

#include "backend/opencl/execution/buffer/BinaryBufExecution.hpp"

namespace MNN {
namespace OpenCL {

BinaryBufExecution::BinaryBufExecution(const std::vector<Tensor *> &inputs, const std::string &compute, const MNN::Op *op, Backend *backend)
    : CommonExecution(backend, op), mCompute(compute) {
    mBuildOptions.emplace("-DOPERATOR=" + compute);
}

uint32_t BinaryBufExecution::realSize(const Tensor* tensor) {
    uint32_t num = 1;
    for(int i = 0; i < tensor->dimensions(); i++) {
        num *= tensor->length(i);
    }
    return num;
}

#ifdef MNN_SUPPORT_INTEL_SUBGROUP
ErrorCode BinaryBufExecution::SubgroupOnResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto openCLBackend = static_cast<OpenCLBackend *>(backend());
    auto output        = outputs[0];
    auto inputShape0   = tensorShapeFormat(inputs[0]);
    auto inputShape1   = tensorShapeFormat(inputs[1]);
    auto outputShape   = tensorShapeFormat(output);
    auto runTime       = ((OpenCLBackend *)backend())->getOpenCLRuntime();
    int shape[4]       = {outputShape[0], outputShape[1], outputShape[2], outputShape[3]};

    int fullCount[2] = {1, 1};

    int input0_c_pack = TensorUtils::getTensorChannelPack(inputs[0]);
    int input1_c_pack = TensorUtils::getTensorChannelPack(inputs[1]);
    int output_c_pack = TensorUtils::getTensorChannelPack(output);
    
    int activationType = 0;
    if(mOp->type() == OpType_BinaryOp) {
        activationType = mOp->main_as_BinaryOp()->activationType();
    }
    auto &unit = mUnits[0];
    std::set<std::string> buildOptions = mBuildOptions;
    if(output->getType().code == halide_type_int) {
        if(output->getType().bits == 8){
            buildOptions.emplace("-DINTEL_DATA=uchar");
            buildOptions.emplace("-DAS_INPUT_DATA=as_char");
            buildOptions.emplace("-DAS_INPUT_DATA4=as_char4");
            buildOptions.emplace("-DAS_OUTPUT_DATA4=as_uchar4");
            buildOptions.emplace("-DINTEL_SUB_GROUP_READ=intel_sub_group_block_read_uc");
            buildOptions.emplace("-DINTEL_SUB_GROUP_READ4=intel_sub_group_block_read_uc4");
            buildOptions.emplace("-DINTEL_SUB_GROUP_WRITE4=intel_sub_group_block_write_uc4");
        } else if(output->getType().bits == 32){
            buildOptions.emplace("-DINTEL_DATA=uint");
            buildOptions.emplace("-DAS_INPUT_DATA=as_int");
            buildOptions.emplace("-DAS_INPUT_DATA4=as_int4");
            buildOptions.emplace("-DAS_OUTPUT_DATA4=as_uint4");
            buildOptions.emplace("-DINTEL_SUB_GROUP_READ=intel_sub_group_block_read");
            buildOptions.emplace("-DINTEL_SUB_GROUP_READ4=intel_sub_group_block_read4");
            buildOptions.emplace("-DINTEL_SUB_GROUP_WRITE4=intel_sub_group_block_write4");
        }
    } else if(output->getType().code == halide_type_uint){
        if(output->getType().bits == 8){
            buildOptions.emplace("-DINTEL_DATA=uchar");
            buildOptions.emplace("-DAS_INPUT_DATA=as_uchar");
            buildOptions.emplace("-DAS_INPUT_DATA4=as_uchar4");
            buildOptions.emplace("-DAS_OUTPUT_DATA4=as_uchar4");
            buildOptions.emplace("-DINTEL_SUB_GROUP_READ=intel_sub_group_block_read_uc");
            buildOptions.emplace("-DINTEL_SUB_GROUP_READ4=intel_sub_group_block_read_uc4");
            buildOptions.emplace("-DINTEL_SUB_GROUP_WRITE4=intel_sub_group_block_write_uc4");
        } else if(output->getType().bits == 32){
            buildOptions.emplace("-DINTEL_DATA=uint");
            buildOptions.emplace("-DAS_INPUT_DATA=as_uint");
            buildOptions.emplace("-DAS_INPUT_DATA4=as_uint4");
            buildOptions.emplace("-DAS_OUTPUT_DATA4=as_uint4");
            buildOptions.emplace("-DINTEL_SUB_GROUP_READ=intel_sub_group_block_read");
            buildOptions.emplace("-DINTEL_SUB_GROUP_READ4=intel_sub_group_block_read4");
            buildOptions.emplace("-DINTEL_SUB_GROUP_WRITE4=intel_sub_group_block_write4");
        }
    } else {
        if(runTime->isSupportedFP16()){
            buildOptions.emplace("-DINTEL_DATA=ushort");
            buildOptions.emplace("-DAS_INPUT_DATA=as_half");
            buildOptions.emplace("-DAS_INPUT_DATA4=as_half4");
            buildOptions.emplace("-DAS_OUTPUT_DATA4=as_ushort4");
            buildOptions.emplace("-DINTEL_SUB_GROUP_READ=intel_sub_group_block_read_us");
            buildOptions.emplace("-DINTEL_SUB_GROUP_READ4=intel_sub_group_block_read_us4");
            buildOptions.emplace("-DINTEL_SUB_GROUP_WRITE4=intel_sub_group_block_write_us4");
        }else{
            buildOptions.emplace("-DINTEL_DATA=uint");
            buildOptions.emplace("-DAS_INPUT_DATA=as_float");
            buildOptions.emplace("-DAS_INPUT_DATA4=as_float4");
            buildOptions.emplace("-DAS_OUTPUT_DATA4=as_uint4");
            buildOptions.emplace("-DINTEL_SUB_GROUP_READ=intel_sub_group_block_read");
            buildOptions.emplace("-DINTEL_SUB_GROUP_READ4=intel_sub_group_block_read4");
            buildOptions.emplace("-DINTEL_SUB_GROUP_WRITE4=intel_sub_group_block_write4");
        }
    }
    std::string kernelName = "binary_buf_c" + std::to_string(input0_c_pack) + "_c" + std::to_string(input1_c_pack) +
                                 "_c" + std::to_string(output_c_pack);
    unit.kernel = runTime->buildKernel("binary_subgroup_buf", kernelName, buildOptions, inputs[0], output);
    mMaxWorkGroupSize = static_cast<uint32_t>(runTime->getMaxWorkGroupSize(unit.kernel));

    fullCount[0] = realSize(inputs[0]) == 1 ? 0 : 1;
    fullCount[1] = realSize(inputs[1]) == 1 ? 0 : 1;

    auto input0pad = TensorUtils::getDescribe(inputs[0])->mPads;
    auto input1pad = TensorUtils::getDescribe(inputs[1])->mPads;
    auto outputpad = TensorUtils::getDescribe(output)->mPads;
    
    uint32_t index = 0;
    cl_int ret = CL_SUCCESS;
    if (input0_c_pack == 16 && input1_c_pack == 16) {
        mGlobalWorkSize = {(uint32_t)UP_DIV(outputShape[2], 4) * outputShape[1],
                               (uint32_t)ROUND_UP(outputShape[3], 16), (uint32_t)outputShape[0]};
        unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1], mGlobalWorkSize[2]};
        unit.localWorkSize  = {1, 16, 1};
        ret |= unit.kernel->get().setArg(index++, mGlobalWorkSize[0]);
        ret |= unit.kernel->get().setArg(index++, mGlobalWorkSize[1]);
        ret |= unit.kernel->get().setArg(index++, mGlobalWorkSize[2]);
        ret |= unit.kernel->get().setArg(index++, openCLBuffer(inputs[0]));
        ret |= unit.kernel->get().setArg(index++, openCLBuffer(inputs[1]));
        ret |= unit.kernel->get().setArg(index++, openCLBuffer(output));
        ret |= unit.kernel->get().setArg(index++, shape);
        ret |= unit.kernel->get().setArg(index++, fullCount);
        ret |= unit.kernel->get().setArg(index++, activationType);
        ret |= unit.kernel->get().setArg(index++, static_cast<uint32_t>(input0pad.left));
        ret |= unit.kernel->get().setArg(index++, static_cast<uint32_t>(input0pad.right));
        ret |= unit.kernel->get().setArg(index++, static_cast<uint32_t>(input1pad.left));
        ret |= ret |= unit.kernel->get().setArg(index++, static_cast<uint32_t>(input1pad.right));
        ret |= unit.kernel->get().setArg(index++, static_cast<uint32_t>(outputpad.left));
        ret |= unit.kernel->get().setArg(index++, static_cast<uint32_t>(outputpad.right));
        MNN_CHECK_CL_SUCCESS(ret, "setArg BinaryBufExecution C16");
        openCLBackend->recordKernel3d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
    } else {
        mGlobalWorkSize = {(uint32_t)outputShape[2] * outputShape[1], (uint32_t)UP_DIV(outputShape[3], 4),
                                    (uint32_t)outputShape[0]};

        ret |= unit.kernel->get().setArg(index++, mGlobalWorkSize[0]);
        ret |= unit.kernel->get().setArg(index++, mGlobalWorkSize[1]);
        ret |= unit.kernel->get().setArg(index++, mGlobalWorkSize[2]);
        ret |= unit.kernel->get().setArg(index++, openCLBuffer(inputs[0]));
        ret |= unit.kernel->get().setArg(index++, openCLBuffer(inputs[1]));
        ret |= unit.kernel->get().setArg(index++, openCLBuffer(output));
        ret |= unit.kernel->get().setArg(index++, shape);
        ret |= unit.kernel->get().setArg(index++, fullCount);
        ret |= unit.kernel->get().setArg(index++, activationType);
        ret |= unit.kernel->get().setArg(index++, static_cast<uint32_t>(input0pad.left));
        ret |= unit.kernel->get().setArg(index++, static_cast<uint32_t>(input0pad.right));
        ret |= unit.kernel->get().setArg(index++, static_cast<uint32_t>(input1pad.left));
        ret |= unit.kernel->get().setArg(index++, static_cast<uint32_t>(input1pad.right));
        ret |= unit.kernel->get().setArg(index++, static_cast<uint32_t>(outputpad.left));
        ret |= unit.kernel->get().setArg(index++, static_cast<uint32_t>(outputpad.right));
        MNN_CHECK_CL_SUCCESS(ret, "setArg BinaryBufExecution");

        mLocalWorkSize = localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, openCLBackend->getOpenCLRuntime(), kernelName, unit.kernel).first;

        unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1], mGlobalWorkSize[2]};
        unit.localWorkSize  = {mLocalWorkSize[0], mLocalWorkSize[1], mLocalWorkSize[2]};
        openCLBackend->recordKernel3d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
    }
    
    for (int i = 2; i < inputs.size(); ++i) {
        fullCount[0] = 1;
        fullCount[1] = realSize(inputs[i]) == 1 ? 0 : 1;
        auto &unit = mUnits[i-1];

        int input0_c_pack_tmp = TensorUtils::getTensorChannelPack(output);
        int input1_c_pack_tmp = TensorUtils::getTensorChannelPack(inputs[i]);
        int output_c_pack_tmp = TensorUtils::getTensorChannelPack(output);
        std::string kernelNameTmp = "binary_buf_c" + std::to_string(input0_c_pack_tmp) + "_c" + std::to_string(input1_c_pack_tmp) +
                                 "_c" + std::to_string(output_c_pack_tmp);
        unit.kernel = runTime->buildKernel("binary_subgroup_buf", kernelNameTmp, buildOptions, inputs[i], output);
        mMaxWorkGroupSize = static_cast<uint32_t>(runTime->getMaxWorkGroupSize(unit.kernel));

        auto input0padtmp = TensorUtils::getDescribe(output)->mPads;
        auto input1padtmp = TensorUtils::getDescribe(inputs[i])->mPads;
        auto outputpadtmp = TensorUtils::getDescribe(output)->mPads;
    

        uint32_t index = 0;
        if (input0_c_pack_tmp == 16 && input1_c_pack_tmp == 16) {
            mGlobalWorkSize     = {(uint32_t)UP_DIV(outputShape[2], 4) * outputShape[1],
                                   (uint32_t)ROUND_UP(outputShape[3], 16), (uint32_t)outputShape[0]};
            unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1], mGlobalWorkSize[2]};
            unit.localWorkSize  = {1, 16, 1};
            ret |= unit.kernel->get().setArg(index++, mGlobalWorkSize[0]);
            ret |= unit.kernel->get().setArg(index++, mGlobalWorkSize[1]);
            ret |= unit.kernel->get().setArg(index++, mGlobalWorkSize[2]);
            ret |= unit.kernel->get().setArg(index++, openCLBuffer(output));
            ret |= unit.kernel->get().setArg(index++, openCLBuffer(inputs[i]));
            ret |= unit.kernel->get().setArg(index++, openCLBuffer(output));
            ret |= unit.kernel->get().setArg(index++, shape);
            ret |= unit.kernel->get().setArg(index++, fullCount);
            ret |= unit.kernel->get().setArg(index++, activationType);
            ret |= unit.kernel->get().setArg(index++, static_cast<uint32_t>(input0padtmp.left));
            ret |= unit.kernel->get().setArg(index++, static_cast<uint32_t>(input0padtmp.right));
            ret |= unit.kernel->get().setArg(index++, static_cast<uint32_t>(input1padtmp.left));
            ret |= unit.kernel->get().setArg(index++, static_cast<uint32_t>(input1padtmp.right));
            ret |= unit.kernel->get().setArg(index++, static_cast<uint32_t>(outputpadtmp.left));
            ret |= unit.kernel->get().setArg(index++, static_cast<uint32_t>(outputpadtmp.right));
            MNN_CHECK_CL_SUCCESS(ret, "setArg BinaryBufExecution C16 MultiInput");
            openCLBackend->recordKernel3d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
        } else {
            mGlobalWorkSize = {(uint32_t)outputShape[2] * outputShape[1], (uint32_t)UP_DIV(outputShape[3], 4),
                                    (uint32_t)outputShape[0]};

            ret |= unit.kernel->get().setArg(index++, mGlobalWorkSize[0]);
            ret |= unit.kernel->get().setArg(index++, mGlobalWorkSize[1]);
            ret |= unit.kernel->get().setArg(index++, mGlobalWorkSize[2]);
            ret |= unit.kernel->get().setArg(index++, openCLBuffer(output));
            ret |= unit.kernel->get().setArg(index++, openCLBuffer(inputs[i]));
            ret |= unit.kernel->get().setArg(index++, openCLBuffer(output));
            ret |= unit.kernel->get().setArg(index++, shape);
            ret |= unit.kernel->get().setArg(index++, fullCount);
            ret |= unit.kernel->get().setArg(index++, activationType);
            ret |= unit.kernel->get().setArg(index++, static_cast<uint32_t>(input0padtmp.left));
            ret |= unit.kernel->get().setArg(index++, static_cast<uint32_t>(input0padtmp.right));
            ret |= unit.kernel->get().setArg(index++, static_cast<uint32_t>(input1padtmp.left));
            ret |= unit.kernel->get().setArg(index++, static_cast<uint32_t>(input1padtmp.right));
            ret |= unit.kernel->get().setArg(index++, static_cast<uint32_t>(outputpadtmp.left));
            ret |= unit.kernel->get().setArg(index++, static_cast<uint32_t>(outputpadtmp.right));
            MNN_CHECK_CL_SUCCESS(ret, "setArg BinaryBufExecution MultiInput");

            mLocalWorkSize = localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, openCLBackend->getOpenCLRuntime(), kernelNameTmp, unit.kernel).first;

            unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1], mGlobalWorkSize[2]};
            unit.localWorkSize  = {mLocalWorkSize[0], mLocalWorkSize[1], mLocalWorkSize[2]};
            openCLBackend->recordKernel3d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
        }
    }
    return NO_ERROR;
}
#endif /* MNN_SUPPORT_INTEL_SUBGROUP */

ErrorCode BinaryBufExecution::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(inputs.size() >= 2);
    mUnits.resize(inputs.size() - 1);
    
    auto openCLBackend = static_cast<OpenCLBackend*>(backend());
    auto output = outputs[0];
    auto inputShape0 = tensorShapeFormat(inputs[0]);
    auto inputShape1 = tensorShapeFormat(inputs[1]);
    auto outputShape = tensorShapeFormat(output);
    auto runTime     = ((OpenCLBackend *)backend())->getOpenCLRuntime();
#ifdef MNN_SUPPORT_INTEL_SUBGROUP
    if (runTime->isSupportedIntelSubgroup()) {
        return SubgroupOnResize(inputs, outputs);
    }
#endif /* MNN_SUPPORT_INTEL_SUBGROUP */
    int shape[4] = {outputShape[0], outputShape[1], outputShape[2], UP_DIV(outputShape[3], 4)};
    int fullCount[2] = {1, 1};
    fullCount[0] = realSize(inputs[0]) == 1 ? 0 : 1;
    fullCount[1] = realSize(inputs[1]) == 1 ? 0 : 1;
    
    int activationType = 0;
    if(mOp->type() == OpType_BinaryOp) {
        activationType = mOp->main_as_BinaryOp()->activationType();
    }
    auto &unit = mUnits[0];
    
    std::set<std::string> buildOptions = mBuildOptions;
    int wh_pack = 1;
    if((outputShape[1]*outputShape[2]) % 4 == 0) {
        wh_pack = 4;
        buildOptions.emplace("-DWH_PACK4");
    }
    if(fullCount[0] == 0) {
        buildOptions.emplace("-DA_SINGLE");
    }
    if(fullCount[1] == 0) {
        buildOptions.emplace("-DB_SINGLE");
    }
    unit.kernel = runTime->buildKernel("binary_buf", "binary_buf", buildOptions, inputs[0], output);
    mMaxWorkGroupSize      = static_cast<uint32_t>(runTime->getMaxWorkGroupSize(unit.kernel));

    mGlobalWorkSize =  {(uint32_t)UP_DIV(outputShape[3], 4) * outputShape[0],
                                        (uint32_t)UP_DIV(outputShape[1]*outputShape[2], wh_pack)};

    uint32_t index = 0;
    cl_int ret = CL_SUCCESS;
    ret |= unit.kernel->get().setArg(index++, mGlobalWorkSize[0]);
    ret |= unit.kernel->get().setArg(index++, mGlobalWorkSize[1]);
    ret |= unit.kernel->get().setArg(index++, openCLBuffer(inputs[0]));
    ret |= unit.kernel->get().setArg(index++, openCLBuffer(inputs[1]));
    ret |= unit.kernel->get().setArg(index++, openCLBuffer(output));
    ret |= unit.kernel->get().setArg(index++, shape);
    ret |= unit.kernel->get().setArg(index++, fullCount);
    ret |= unit.kernel->get().setArg(index++, activationType);
    MNN_CHECK_CL_SUCCESS(ret, "setArg BinaryBufExecution");

    std::string name = "binary_buf";
    mLocalWorkSize = {(uint32_t)16, (uint32_t)16};
    
    unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1]};
    unit.localWorkSize  = {mLocalWorkSize[0], mLocalWorkSize[1]};
    openCLBackend->recordKernel2d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
    for (int i = 2; i < inputs.size(); ++i) {
        fullCount[0] = 1;
        fullCount[1] = realSize(inputs[i]) == 1 ? 0 : 1;
        auto &unit = mUnits[i-1];
        
        std::set<std::string> buildOptions = mBuildOptions;
        if((outputShape[1]*outputShape[2]) % 4 == 0) {
            buildOptions.emplace("-DWH_PACK4");
        }
        if(fullCount[1] == 0) {
            buildOptions.emplace("-DB_SINGLE");
        }
        unit.kernel = runTime->buildKernel("binary_buf", "binary_buf", buildOptions, inputs[i], output);

        uint32_t index = 0;
        ret |= unit.kernel->get().setArg(index++, mGlobalWorkSize[0]);
        ret |= unit.kernel->get().setArg(index++, mGlobalWorkSize[1]);
        ret |= unit.kernel->get().setArg(index++, openCLBuffer(output));
        ret |= unit.kernel->get().setArg(index++, openCLBuffer(inputs[i]));
        ret |= unit.kernel->get().setArg(index++, openCLBuffer(output));
        ret |= unit.kernel->get().setArg(index++, shape);
        ret |= unit.kernel->get().setArg(index++, fullCount);
        ret |= unit.kernel->get().setArg(index++, activationType);
        MNN_CHECK_CL_SUCCESS(ret, "setArg BinaryBufExecution MultiInput");

        unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1]};
        unit.localWorkSize  = {mLocalWorkSize[0], mLocalWorkSize[1]};
        openCLBackend->recordKernel2d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
    }
    
    return NO_ERROR;
}

class BinaryBufCreator : public OpenCLBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        for (int i = 0; i < inputs.size(); ++i) {
            int channel = inputs[i]->channel();
            if (channel >= 16 && static_cast<OpenCLBackend *>(backend)->getOpenCLRuntime()->isSupportedIntelSubgroup()) {
                TensorUtils::setTensorChannelPack(inputs[i], 16);
            }
        }
        if (op->type() == OpType_Eltwise) {
            switch (op->main_as_Eltwise()->type()) {
                case EltwiseType_SUM:
                    return new BinaryBufExecution(inputs, "in0+in1", op, backend);
                case EltwiseType_PROD:
                    return new BinaryBufExecution(inputs, "in0*in1", op, backend);
                case EltwiseType_SUB:
                    return new BinaryBufExecution(inputs, "in0-in1", op, backend);
                case EltwiseType_MAXIMUM:
                    return new BinaryBufExecution(inputs, "in0>in1?in0:in1", op, backend);
                default:
                    break;
            }
            return nullptr;
        }

        if (op->type() == OpType_BinaryOp) {
            MNN_ASSERT(inputs.size() > 1);

            switch (op->main_as_BinaryOp()->opType()) {
                case BinaryOpOperation_MUL:
                    return new BinaryBufExecution(inputs, "in0*in1", op, backend);
                case BinaryOpOperation_ADD:
                    return new BinaryBufExecution(inputs, "in0+in1", op, backend);
                case BinaryOpOperation_SUB:
                    return new BinaryBufExecution(inputs, "in0-in1", op, backend);
                case BinaryOpOperation_REALDIV:
                    return new BinaryBufExecution(inputs, "sign(in1)*in0/(fabs(in1)>(float4)((float)0.0000001)?fabs(in1):(float4)((float)0.0000001))", op, backend);
                case BinaryOpOperation_MINIMUM:
                    return new BinaryBufExecution(inputs, "in0>in1?in1:in0", op, backend);
                case BinaryOpOperation_MAXIMUM:
                    return new BinaryBufExecution(inputs, "in0>in1?in0:in1", op, backend);
                case BinaryOpOperation_GREATER:
                    return new BinaryBufExecution(inputs, "convert_float4(-isgreater(in0,in1))", op, backend);
                case BinaryOpOperation_LESS:
                    return new BinaryBufExecution(inputs, "convert_float4(-isless(in0,in1))", op, backend);
                case BinaryOpOperation_LESS_EQUAL:
                    return new BinaryBufExecution(inputs, "convert_float4(-islessequal(in0,in1))", op, backend);
                case BinaryOpOperation_GREATER_EQUAL:
                    return new BinaryBufExecution(inputs, "convert_float4(-isgreaterequal(in0,in1))", op, backend);
                case BinaryOpOperation_EQUAL:
                    return new BinaryBufExecution(inputs, "convert_float4(-isequal(in0,in1))", op, backend);
                case BinaryOpOperation_FLOORDIV:
                    return new BinaryBufExecution(inputs, "floor(sign(in1)*in0/(fabs(in1)>(float4)((float)0.0000001)?fabs(in1):(float4)((float)0.0000001)))", op, backend);
                case BinaryOpOperation_FLOORMOD:
                    return new BinaryBufExecution(inputs, "in0-floor(sign(in1)*in0/(fabs(in1)>(float4)((float)0.0000001)?fabs(in1):(float4)((float)0.0000001)))*in1", op, backend);
                case BinaryOpOperation_POW:
                    return new BinaryBufExecution(inputs, "pow(in0,in1)", op, backend);
                case BinaryOpOperation_SquaredDifference:
                    return new BinaryBufExecution(inputs, "(in0-in1)*(in0-in1)", op, backend);
                case BinaryOpOperation_ATAN2:
                    return new BinaryBufExecution(inputs, "(in1==(float)0?(sign(in0)*(float4)(PI/2)):(atan(in0/in1)+(in1>(float4)0?(float4)0:sign(in0)*(float)PI)))", op, backend);
                case BinaryOpOperation_NOTEQUAL:
                    return new BinaryBufExecution(inputs, "convert_float4(-isnotequal(in0,in1))", op, backend);
                case BinaryOpOperation_MOD:
                    return new BinaryBufExecution(inputs, "in0-floor(sign(in1)*in0/(fabs(in1)>(float4)((float)0.0000001)?fabs(in1):(float4)((float)0.0000001)))*in1", op, backend);
                default:
                    break;
            }
            return nullptr;
        }
        return nullptr;
    }
};

REGISTER_OPENCL_OP_CREATOR(BinaryBufCreator, OpType_Eltwise, BUFFER);
REGISTER_OPENCL_OP_CREATOR(BinaryBufCreator, OpType_BinaryOp, BUFFER);

} // namespace OpenCL
} // namespace MNN
#endif /* MNN_OPENCL_BUFFER_CLOSED */
