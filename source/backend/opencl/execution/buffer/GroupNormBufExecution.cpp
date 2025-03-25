
//
//  GroupNormBufExecution.cpp
//  MNN
//
//  Created by MNN on 2024/06/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include "backend/opencl/execution/buffer/GroupNormBufExecution.hpp"

namespace MNN {
namespace OpenCL {

GroupNormBufExecution::GroupNormBufExecution(const MNN::Op* op, Backend* backend) : CommonExecution(backend, op) {
    auto group_norm_param = op->main_as_GroupNorm();
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    mEpsilon = group_norm_param->epsilon();
    mBSwish = group_norm_param->bSwish();
    mGroup = group_norm_param->group();
    if (group_norm_param->gamma() && group_norm_param->beta()) {
        auto bufferUnitSize = runtime->isSupportedFP16() ? sizeof(half_float::half) : sizeof(float);

        mHasGammaBeta = true;
        int size = group_norm_param->gamma()->size();
        mGammaTensor.reset(Tensor::createDevice<float>({ALIGN_UP4(size)}));
        auto status = backend->onAcquireBuffer(mGammaTensor.get(), Backend::STATIC);
        if (!status) {
            MNN_ERROR("Out of memory when gamma is acquired in GroupNorm.\n");
        }

        cl::Buffer &gammaBuffer = openCLBuffer(mGammaTensor.get());
        
        cl_int res;
        auto GammaPtrCL = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(
            gammaBuffer, true, CL_MAP_WRITE, 0, ALIGN_UP4(size) * bufferUnitSize, nullptr, nullptr, &res);
        if(GammaPtrCL != nullptr && res == CL_SUCCESS){
            if(mOpenCLBackend->getOpenCLRuntime()->isSupportedFP16()){
                for (int i = 0; i < size; i++) {
                    ((half_float::half*)GammaPtrCL)[i] = (half_float::half)(group_norm_param->gamma()->data()[i]);
                }
                for(int i=size; i<ALIGN_UP4(size); i++) {
                    ((half_float::half*)GammaPtrCL)[i] = (half_float::half)(0.0f);
                }
            }else{
                ::memset(GammaPtrCL, 0, ALIGN_UP4(size) * sizeof(float));
                ::memcpy(GammaPtrCL, group_norm_param->gamma()->data(), size * sizeof(float));
            }
        } else {
            MNN_ERROR("GroupNorm Gamma map error:%d\n", res);
        }

        
        if (group_norm_param->beta()->size() != size) {
            MNN_ERROR("Size of gamma and beta are not match in GroupNorm.\n");
        }
        mBetaTensor.reset(Tensor::createDevice<float>({ALIGN_UP4(size)}));
        status = backend->onAcquireBuffer(mBetaTensor.get(), Backend::STATIC);
        if (!status) {
            MNN_ERROR("Out of memory when beta is acquired in GroupNorm.\n");
        }

        cl::Buffer &betaBuffer = openCLBuffer(mBetaTensor.get());

        auto BetaPtrCL = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(
             betaBuffer, true, CL_MAP_WRITE, 0, ALIGN_UP4(size) * bufferUnitSize, nullptr, nullptr, &res);
        if(BetaPtrCL != nullptr && res == CL_SUCCESS){
            if(mOpenCLBackend->getOpenCLRuntime()->isSupportedFP16()){
                for (int i = 0; i < size; i++) {
                    ((half_float::half*)BetaPtrCL)[i] = (half_float::half)(group_norm_param->beta()->data()[i]);
                }
                for(int i=size; i<ALIGN_UP4(size); i++) {
                    ((half_float::half*)BetaPtrCL)[i] = (half_float::half)(0.0f);
                }
            }else{
                ::memset(BetaPtrCL, 0, ALIGN_UP4(size) * sizeof(float));
                ::memcpy(BetaPtrCL, group_norm_param->beta()->data(), size * sizeof(float));
            }
        } else {
            MNN_ERROR("GroupNorm Beta map error:%d\n", res);
        }
	
	mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(gammaBuffer, GammaPtrCL);
        mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(betaBuffer, BetaPtrCL);
    }
}

int GroupNormBufExecution::getLocalSize(int size, int maxGroupSize){
    int local_size = 1;
    while(local_size * 2 <= maxGroupSize && local_size * 2 <= size){
        local_size *= 2;
    }
    return local_size;
}
ErrorCode GroupNormBufExecution::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto runtime = static_cast<OpenCLBackend*>(backend())->getOpenCLRuntime();

    MNN_ASSERT(outputs.size() == 1);
    auto input = inputs[0];
    auto output = outputs[0];

    mBatch = input->length(0);
    if(inputs.size() > 1) {
        MNN_ASSERT(inputs[1]->dimensions() == 2);
        MNN_ASSERT(inputs[1]->length(0) == inputs[0]->length(0));
        MNN_ASSERT(inputs[1]->length(1) == inputs[0]->length(1));
    }

    size_t outter_size = mBatch * mGroup;
    size_t inner_size = 1;
    for (int i = 1; i < input->dimensions(); i++) {
        inner_size *= inputs[0]->length(i);
    }
    inner_size /= mGroup;
    
    mUnits.clear();
    mUnits.resize(1);
    std::vector<int> inputShape = tensorShapeFormat(inputs[0]);
    int inputWH[]      = {inputShape[2], inputShape[1]};
    int region[]       = {inputShape[0], UP_DIV(inputShape[3], 4), inputShape[1], inputShape[2]};
    
    std::set<std::string> buildOptions;
    // do groupnorm
    {
        int area = inputWH[1] * inputWH[0];
        if(mHasGammaBeta){
            buildOptions.emplace("-DGAMMA_BETA");
        }
        if(mBSwish) {
            buildOptions.emplace("-DSWISH");
        }
        if(area % 4 == 0) {
            buildOptions.emplace("-DWH_4");
        }
        if(inputs.size() > 1) {
            buildOptions.emplace("-DDOUBLE_INPUTS");
        }
        auto MaxLocalSize = std::min(runtime->getMaxWorkItemSizes()[0], (uint32_t)256);

        auto &unit = mUnits[0];
        std::string kernelName = "groupnorm_plain_buf";
        int local_size = getLocalSize(UP_DIV(inner_size, 4), MaxLocalSize);
        buildOptions.emplace("-DLOCAL_SIZE=" + std::to_string(local_size));
        unit.kernel = runtime->buildKernel("groupnorm_buf", kernelName, buildOptions);
        
        mGWS = {static_cast<uint32_t>(local_size),
                static_cast<uint32_t>(1),
                static_cast<uint32_t>(outter_size)};
        
        mLWS = {static_cast<uint32_t>(local_size), 1, 1};

        unit.globalWorkSize  = {mGWS[0], mGWS[1], mGWS[2]};
        unit.localWorkSize   = {mLWS[0], mLWS[1], mLWS[2]};

        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;
        ret |= unit.kernel->get().setArg(idx++, mGWS[0]);
        ret |= unit.kernel->get().setArg(idx++, mGWS[1]);
        ret |= unit.kernel->get().setArg(idx++, mGWS[2]);
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(input));
        if(inputs.size() > 1) {
            ret |= unit.kernel->get().setArg(idx++, openCLBuffer(inputs[1]));
        }
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(output));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int32_t>(area));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int32_t>(mGroup));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int32_t>(inner_size));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int32_t>(outter_size));
        if(mHasGammaBeta){
            ret |= unit.kernel->get().setArg(idx++, openCLBuffer(mGammaTensor.get()));
            ret |= unit.kernel->get().setArg(idx++, openCLBuffer(mBetaTensor.get()));
        }
        ret |= unit.kernel->get().setArg(idx++, mEpsilon);
        MNN_CHECK_CL_SUCCESS(ret, "setArg GroupNormBufExecution with group, do group layernorm");
        mOpenCLBackend->recordKernel3d(unit.kernel, mGWS, mLWS);
    }
    mOpenCLBackend->endRecord(mRecording);

    return NO_ERROR;
    
}


class GroupNormBufCreator : public OpenCLBackend::Creator {
public:
    virtual ~GroupNormBufCreator() = default;
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        for (int i = 0; i < inputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(inputs[i], false);
        }
        for (int i = 0; i < outputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(outputs[i], false);
        }
        
        return new GroupNormBufExecution(op, backend);
    }
};

REGISTER_OPENCL_OP_CREATOR_TRANSFORMER(GroupNormBufCreator, OpType_GroupNorm, BUFFER);
} // namespace OpenCL
} // namespace MNN
#endif/* MNN_SUPPORT_TRANSFORMER_FUSE */
