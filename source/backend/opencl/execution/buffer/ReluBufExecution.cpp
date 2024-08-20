//
//  ReluBufExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED

#include "backend/opencl/execution/buffer/ReluBufExecution.hpp"
#include "backend/opencl/execution/buffer/UnaryBufExecution.hpp"
namespace MNN {
namespace OpenCL {

ReluBufExecution::ReluBufExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend)
    : CommonExecution(backend, op) {
    mOpenCLBackend       = static_cast<OpenCLBackend *>(backend);
    auto mPreluParamPtr       = op->main_as_PRelu();
    int preluSize             = mPreluParamPtr->slopeCount();
    const float *preluDataPtr = mPreluParamPtr->slope()->data();
    
    int buffer_size = ALIGN_UP4(preluSize);
    if (mOpenCLBackend->getOpenCLRuntime()->isSupportedFP16()) {
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
        if (mOpenCLBackend->getOpenCLRuntime()->isSupportedFP16()) {
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
}

ReluBufExecution::~ReluBufExecution() {
    // Do nothing
}

ErrorCode ReluBufExecution::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mUnits.resize(1);
    auto nhwc              = tensorShapeFormat(outputs[0]);
    int nhwcArray[4]        = {nhwc[0], nhwc[1], nhwc[2], UP_DIV(nhwc[3], 4)};
    auto imageWidth        = nhwc[0] * UP_DIV(nhwc[3], 4);
    auto imageHeight       = nhwc[1] * nhwc[2];
    
    std::vector<uint32_t> localSize  = {1, 1};
    std::vector<uint32_t> globalSize = {(uint32_t)imageWidth, (uint32_t)imageHeight};

    auto runTime     = mOpenCLBackend->getOpenCLRuntime();
#ifdef MNN_SUPPORT_INTEL_SUBGROUP
    if (runTime->isSupportedIntelSubgroup()){
        return SubgrouponResize(inputs, outputs);
    }
#endif /* MNN_SUPPORT_INTEL_SUBGROUP */
    mUnits[0].kernel = runTime->buildKernel("binary_buf", "prelu_buf", {"-DOPERATOR=select(in0*in1,in0,in0>=(float4)0)"}, inputs[0], outputs[0]);
    mMaxWorkGroupSize      = static_cast<uint32_t>(runTime->getMaxWorkGroupSize(mUnits[0].kernel));
    int fullCount[2] = {1, 1};
    
    uint32_t index = 0;
    cl_int ret = CL_SUCCESS;
    ret |= mUnits[0].kernel->get().setArg(index++, globalSize[0]);
    ret |= mUnits[0].kernel->get().setArg(index++, globalSize[1]);
    ret |= mUnits[0].kernel->get().setArg(index++, openCLBuffer(inputs[0]));
    ret |= mUnits[0].kernel->get().setArg(index++, openCLBuffer(mPreluParam.get()));
    ret |= mUnits[0].kernel->get().setArg(index++, openCLBuffer(outputs[0]));
    ret |= mUnits[0].kernel->get().setArg(index++, nhwcArray);
    MNN_CHECK_CL_SUCCESS(ret, "setArg ReluBufExecution");

    std::string name = "prelu_buf";
    localSize = localWS2DDefault(globalSize, mMaxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), name, mUnits[0].kernel).first;
    
    mUnits[0].globalWorkSize = {globalSize[0], globalSize[1]};
    mUnits[0].localWorkSize  = {localSize[0], localSize[1]};
    mOpenCLBackend->recordKernel2d(mUnits[0].kernel, globalSize, localSize);
    return NO_ERROR;
}

#ifdef MNN_SUPPORT_INTEL_SUBGROUP
ErrorCode ReluBufExecution::SubgrouponResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mUnits.resize(1);
    auto nhwc              = tensorShapeFormat(outputs[0]);
    int nhwcArray[4]        = {nhwc[0], nhwc[1], nhwc[2], nhwc[3]};

    auto runTime     = mOpenCLBackend->getOpenCLRuntime();
    int input_c_pack = TensorUtils::getTensorChannelPack(inputs[0]);
    int output_c_pack = TensorUtils::getTensorChannelPack(outputs[0]);
    auto inputpad = TensorUtils::getDescribe(inputs[0])->mPads;
    auto outputpad = TensorUtils::getDescribe(outputs[0])->mPads;
    std::string kernelName = "prelu_buf_c" + std::to_string(input_c_pack) + "_c" + std::to_string(output_c_pack);

    auto output            = outputs[0];
    std::set<std::string> buildOptions;
    if (output->getType().code == halide_type_int) {
        if (output->getType().bits == 8) {
            buildOptions.emplace("-DINTEL_DATA=uchar");
            buildOptions.emplace("-DAS_INPUT_DATA=as_char");
            buildOptions.emplace("-DAS_INPUT_DATA4=as_char4");
            buildOptions.emplace("-DAS_OUTPUT_DATA4=as_uchar4");
            buildOptions.emplace("-DINTEL_SUB_GROUP_READ=intel_sub_group_block_read_uc");
            buildOptions.emplace("-DINTEL_SUB_GROUP_READ4=intel_sub_group_block_read_uc4");
            buildOptions.emplace("-DINTEL_SUB_GROUP_WRITE4=intel_sub_group_block_write_uc4");
        } else if (output->getType().bits == 32) {
            buildOptions.emplace("-DINTEL_DATA=uint");
            buildOptions.emplace("-DAS_INPUT_DATA=as_int");
            buildOptions.emplace("-DAS_INPUT_DATA4=as_int4");
            buildOptions.emplace("-DAS_OUTPUT_DATA4=as_uint4");
            buildOptions.emplace("-DINTEL_SUB_GROUP_READ=intel_sub_group_block_read");
            buildOptions.emplace("-DINTEL_SUB_GROUP_READ4=intel_sub_group_block_read4");
            buildOptions.emplace("-DINTEL_SUB_GROUP_WRITE4=intel_sub_group_block_write4");
        }
    } else if (output->getType().code == halide_type_uint) {
        if (output->getType().bits == 8) {
            buildOptions.emplace("-DINTEL_DATA=uchar");
            buildOptions.emplace("-DAS_INPUT_DATA=as_uchar");
            buildOptions.emplace("-DAS_INPUT_DATA4=as_uchar4");
            buildOptions.emplace("-DAS_OUTPUT_DATA4=as_uchar4");
            buildOptions.emplace("-DINTEL_SUB_GROUP_READ=intel_sub_group_block_read_uc");
            buildOptions.emplace("-DINTEL_SUB_GROUP_READ4=intel_sub_group_block_read_uc4");
            buildOptions.emplace("-DINTEL_SUB_GROUP_WRITE4=intel_sub_group_block_write_uc4");
        } else if (output->getType().bits == 32) {
            buildOptions.emplace("-DINTEL_DATA=uint");
            buildOptions.emplace("-DAS_INPUT_DATA=as_uint");
            buildOptions.emplace("-DAS_INPUT_DATA4=as_uint4");
            buildOptions.emplace("-DAS_OUTPUT_DATA4=as_uint4");
            buildOptions.emplace("-DINTEL_SUB_GROUP_READ=intel_sub_group_block_read");
            buildOptions.emplace("-DINTEL_SUB_GROUP_READ4=intel_sub_group_block_read4");
            buildOptions.emplace("-DINTEL_SUB_GROUP_WRITE4=intel_sub_group_block_write4");
        }
    } else {
        if (runTime->isSupportedFP16()) {
            buildOptions.emplace("-DINTEL_DATA=ushort");
            buildOptions.emplace("-DAS_INPUT_DATA=as_half");
            buildOptions.emplace("-DAS_INPUT_DATA4=as_half4");
            buildOptions.emplace("-DAS_OUTPUT_DATA4=as_ushort4");
            buildOptions.emplace("-DINTEL_SUB_GROUP_READ=intel_sub_group_block_read_us");
            buildOptions.emplace("-DINTEL_SUB_GROUP_READ4=intel_sub_group_block_read_us4");
            buildOptions.emplace("-DINTEL_SUB_GROUP_WRITE4=intel_sub_group_block_write_us4");
        } else {
            buildOptions.emplace("-DINTEL_DATA=uint");
            buildOptions.emplace("-DAS_INPUT_DATA=as_float");
            buildOptions.emplace("-DAS_INPUT_DATA4=as_float4");
            buildOptions.emplace("-DAS_OUTPUT_DATA4=as_uint4");
            buildOptions.emplace("-DINTEL_SUB_GROUP_READ=intel_sub_group_block_read");
            buildOptions.emplace("-DINTEL_SUB_GROUP_READ4=intel_sub_group_block_read4");
            buildOptions.emplace("-DINTEL_SUB_GROUP_WRITE4=intel_sub_group_block_write4");
        }
    }
    buildOptions.emplace("-DOPERATOR=select(in0*in1,in0,in0>=(float4)0)");
    mUnits[0].kernel  = runTime->buildKernel("binary_subgroup_buf", kernelName, buildOptions, inputs[0], output);
    mMaxWorkGroupSize      = static_cast<uint32_t>(runTime->getMaxWorkGroupSize(mUnits[0].kernel));
    int fullCount[2] = {1, 1};
    
    uint32_t index = 0;
    cl_int ret = CL_SUCCESS;
    std::vector<uint32_t> gws = {(uint32_t)nhwc[2] * nhwc[1], (uint32_t)UP_DIV(nhwc[3], 4),
                                        (uint32_t)nhwc[0]};
    std::vector<uint32_t> lws  = {1, 16, 1};
    if (input_c_pack == 4) {
        mUnits[0].globalWorkSize         = {gws[0], gws[1], gws[2]};
        ret |= mUnits[0].kernel->get().setArg(index++, mUnits[0].globalWorkSize[0]);
        ret |= mUnits[0].kernel->get().setArg(index++, mUnits[0].globalWorkSize[1]);
        ret |= mUnits[0].kernel->get().setArg(index++, mUnits[0].globalWorkSize[2]);
        ret |= mUnits[0].kernel->get().setArg(index++, openCLBuffer(inputs[0]));
        ret |= mUnits[0].kernel->get().setArg(index++, openCLBuffer(mPreluParam.get()));
        ret |= mUnits[0].kernel->get().setArg(index++, openCLBuffer(output));
        ret |= mUnits[0].kernel->get().setArg(index++, nhwcArray);
        ret |= mUnits[0].kernel->get().setArg(index++, static_cast<uint32_t>(inputpad.left));
        ret |= mUnits[0].kernel->get().setArg(index++, static_cast<uint32_t>(inputpad.right));
        ret |= mUnits[0].kernel->get().setArg(index++, static_cast<uint32_t>(outputpad.left));
        ret |= mUnits[0].kernel->get().setArg(index++, static_cast<uint32_t>(outputpad.right));
        MNN_CHECK_CL_SUCCESS(ret, "setArg ReluBufExecution SubGroup C4");

        lws = localWS3DDefault(gws, mMaxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), kernelName, mUnits[0].kernel).first;
        mUnits[0].localWorkSize = {lws[0], lws[1], lws[2]};
    } else {
        gws = {(uint32_t)UP_DIV(nhwc[2], 4) * nhwc[1], (uint32_t)ROUND_UP(nhwc[3], 16),
            (uint32_t)nhwc[0]};
        mUnits[0].globalWorkSize         = {gws[0], gws[1], gws[2]};
        mUnits[0].localWorkSize = {lws[0], lws[1], lws[2]};

        ret |= mUnits[0].kernel->get().setArg(index++, mUnits[0].globalWorkSize[0]);
        ret |= mUnits[0].kernel->get().setArg(index++, mUnits[0].globalWorkSize[1]);
        ret |= mUnits[0].kernel->get().setArg(index++, mUnits[0].globalWorkSize[2]);
        ret |= mUnits[0].kernel->get().setArg(index++, openCLBuffer(inputs[0]));
        ret |= mUnits[0].kernel->get().setArg(index++, openCLBuffer(mPreluParam.get()));
        ret |= mUnits[0].kernel->get().setArg(index++, openCLBuffer(outputs[0]));
        ret |= mUnits[0].kernel->get().setArg(index++, nhwcArray);
        ret |= mUnits[0].kernel->get().setArg(index++, static_cast<uint32_t>(inputpad.left));
        ret |= mUnits[0].kernel->get().setArg(index++, static_cast<uint32_t>(inputpad.right));
        ret |= mUnits[0].kernel->get().setArg(index++, static_cast<uint32_t>(outputpad.left));
        ret |= mUnits[0].kernel->get().setArg(index++, static_cast<uint32_t>(outputpad.right));
        MNN_CHECK_CL_SUCCESS(ret, "setArg ReluBufExecution SubGroup");
    }
    mOpenCLBackend->recordKernel3d(mUnits[0].kernel, gws, lws);
    return NO_ERROR;
}
#endif /* MNN_SUPPORT_INTEL_SUBGROUP */

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
        for (int i = 0; i < inputs.size(); ++i) {
            int channel = inputs[i]->channel();
            if (channel >= 16 && static_cast<OpenCLBackend *>(backend)->getOpenCLRuntime()->isSupportedIntelSubgroup()) {
                TensorUtils::setTensorChannelPack(inputs[i], 16);
            }
        }

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
                return new UnaryBufExecution(storage, op, backend);
            }
            std::string temp = "clamp(in,(float4)((float)%f),(float4)((float)%f))";
            sprintf(storage, temp.c_str(), minValue, maxValue);
            return new UnaryBufExecution(storage, op, backend);
        }
        if (op->type() == OpType_ReLU) {
            if (op->main_as_Relu()->slope() == 0.0f) {
                if (isRadeonGpu) {
                    return new UnaryBufExecution("(in>(float4)((float)0)?in:(float4)((float)0))", op, backend);
                }
                return new UnaryBufExecution("fmax(in,(float4)((float)0))", op, backend);
            }
            auto slope         = op->main_as_Relu()->slope();
            char slopeCStr[30] = {};
            sprintf(slopeCStr, "%.8f", slope);
            std::string slopeStr = slopeCStr;
            if (isRadeonGpu) {
                return new UnaryBufExecution("in<(float4)((float)0)?(float)(" + slopeStr + "f)*in:in", op, backend);
            }
            return new UnaryBufExecution("select((float)(" + slopeStr + "f)*in,in,in>=(float4)((float)0))", op, backend);
        }
        if (op->type() == OpType_PReLU) {
            if (op->main_as_PRelu()->slopeCount() == 1) {
                auto slope         = op->main_as_PRelu()->slope()->data()[0];
                char slopeCStr[30] = {};
                sprintf(slopeCStr, "%.8f", slope);
                std::string slopeStr = slopeCStr;
                if (isRadeonGpu) {
                    return new UnaryBufExecution("in<(float4)((float)0)?(float)(" + slopeStr + "f)*in:in", op, backend);
                }
                return new UnaryBufExecution("select((float)(" + slopeStr + "f)*in,in,in>=(float4)((float)0))", op, backend);
            }
            return new ReluBufExecution(inputs, op, backend);
        }
        return nullptr;
    }
};

REGISTER_OPENCL_OP_CREATOR(ReluBufCreator, OpType_ReLU, BUFFER);
REGISTER_OPENCL_OP_CREATOR(ReluBufCreator, OpType_PReLU, BUFFER);
REGISTER_OPENCL_OP_CREATOR(ReluBufCreator, OpType_ReLU6, BUFFER);

} // namespace OpenCL
} // namespace MNN
#endif /* MNN_OPENCL_BUFFER_CLOSED */
