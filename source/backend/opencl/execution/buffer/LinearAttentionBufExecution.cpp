//
//  LinearAttentionBufExecution.cpp
//  MNN
//
//  Created by MNN on 2026/02/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include "LinearAttentionBufExecution.hpp"
#include "core/TensorUtils.hpp"
#include "core/OpCommonUtils.hpp"

namespace MNN {
namespace OpenCL {

LinearAttentionBufExecution::LinearAttentionBufExecution(const MNN::Op *op, Backend *backend)
    : CommonExecution(backend, op) {
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
    auto param = op->main_as_LinearAttentionParam();
    mAttentionType = param->attn_type()->str();
    mNumKHeads = param->num_k_heads();
    mNumVHeads = param->num_v_heads();
    mHeadKDim = param->head_k_dim();
    mHeadVDim = param->head_v_dim();
    mUseQKL2Norm = param->use_qk_l2norm();
    mStateCache.reset(new OpenCLStateCache);
}

ErrorCode LinearAttentionBufExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto qkv = inputs[0];
    int batch = qkv->length(0);
    int convDim = qkv->length(1);
    int seqLen = qkv->length(2);

    int H = mNumVHeads;
    int dk = mHeadKDim;
    int dv = mHeadVDim;
    int K_conv = inputs[3]->length(2);
    int convStateSize = K_conv - 1;
    int key_dim = mNumKHeads * dk;
    int val_dim = mNumVHeads * dv;
    int gqa_factor = (mNumVHeads > mNumKHeads) ? (mNumVHeads / mNumKHeads) : 1;
    float qScale = 1.0f / sqrt((float)dk);

    auto runtime = mOpenCLBackend->getOpenCLRuntime();

    // ─── Persistent state buffers (STATIC): allocate once, shared via onClone ───
    int bytesPerElement = mOpenCLBackend->fpBytes();
    if (mStateCache->mRecurrentState.get() == nullptr) {
        // First time: allocate and zero-initialize
        int rnnSize = batch * H * dk * dv;
        mStateCache->mRecurrentState.reset(Tensor::createDevice<float>({rnnSize}));
        bool success = backend()->onAcquireBuffer(mStateCache->mRecurrentState.get(), Backend::STATIC);
        if (!success) return OUT_OF_MEMORY;

        {
            cl_int res;
            int bufferBytes = rnnSize * bytesPerElement;
            void* mapPtr = runtime->commandQueue().enqueueMapBuffer(
                openCLBuffer(mStateCache->mRecurrentState.get()), true, CL_MAP_WRITE, 0, bufferBytes, nullptr, nullptr, &res);
            if (mapPtr != nullptr && res == CL_SUCCESS) {
                ::memset(mapPtr, 0, bufferBytes);
                runtime->commandQueue().enqueueUnmapMemObject(openCLBuffer(mStateCache->mRecurrentState.get()), mapPtr);
            }
        }

        if (convStateSize > 0) {
            int convStateTotal = batch * convDim * convStateSize;
            mStateCache->mConvState.reset(Tensor::createDevice<float>({convStateTotal}));
            success = backend()->onAcquireBuffer(mStateCache->mConvState.get(), Backend::STATIC);
            if (!success) return OUT_OF_MEMORY;

            cl_int res;
            int bufferBytes = convStateTotal * bytesPerElement;
            void* mapPtr = runtime->commandQueue().enqueueMapBuffer(
                openCLBuffer(mStateCache->mConvState.get()), true, CL_MAP_WRITE, 0, bufferBytes, nullptr, nullptr, &res);
            if (mapPtr != nullptr && res == CL_SUCCESS) {
                ::memset(mapPtr, 0, bufferBytes);
                runtime->commandQueue().enqueueUnmapMemObject(openCLBuffer(mStateCache->mConvState.get()), mapPtr);
            }
        }
    } else if (seqLen > 1) {
        // Prefill (seqLen > 1): reset state for new sequence
        {
            cl_int res;
            int bufferBytes = mStateCache->mRecurrentState->elementSize() * bytesPerElement;
            void* mapPtr = runtime->commandQueue().enqueueMapBuffer(
                openCLBuffer(mStateCache->mRecurrentState.get()), true, CL_MAP_WRITE, 0, bufferBytes, nullptr, nullptr, &res);
            if (mapPtr != nullptr && res == CL_SUCCESS) {
                ::memset(mapPtr, 0, bufferBytes);
                runtime->commandQueue().enqueueUnmapMemObject(openCLBuffer(mStateCache->mRecurrentState.get()), mapPtr);
            }
        }
        if (mStateCache->mConvState.get() != nullptr) {
            cl_int res;
            int bufferBytes = mStateCache->mConvState->elementSize() * bytesPerElement;
            void* mapPtr = runtime->commandQueue().enqueueMapBuffer(
                openCLBuffer(mStateCache->mConvState.get()), true, CL_MAP_WRITE, 0, bufferBytes, nullptr, nullptr, &res);
            if (mapPtr != nullptr && res == CL_SUCCESS) {
                ::memset(mapPtr, 0, bufferBytes);
                runtime->commandQueue().enqueueUnmapMemObject(openCLBuffer(mStateCache->mConvState.get()), mapPtr);
            }
        }
    }
    // Decode (seqLen == 1): keep existing state untouched

    // Allocate temporary conv output buffer
    mConvOut.reset(Tensor::createDevice<float>({batch * convDim * seqLen}));
    mOpenCLBackend->onAcquireBuffer(mConvOut.get(), Backend::DYNAMIC);
    mOpenCLBackend->onReleaseBuffer(mConvOut.get(), Backend::DYNAMIC);

    // Build kernels
    std::set<std::string> buildOptions;

    // Kernel 1: Conv1D + SiLU
    mKernelConvSilu = runtime->buildKernel("linear_attention_buf", "linear_attn_conv_silu", buildOptions, mOpenCLBackend->getPrecision());

    int totalConvSilu = batch * convDim * seqLen;
    mGWSConvSilu = {(uint32_t)totalConvSilu, 1, 1};
    auto maxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernelConvSilu));
    uint32_t lwsConv = std::min(maxWorkGroupSize, (uint32_t)256);
    lwsConv = std::min(lwsConv, (uint32_t)totalConvSilu);
    mLWSConvSilu = {lwsConv, 1, 1};

    // Kernel 2: Conv state update
    if (convStateSize > 0) {
        mKernelConvStateUpdate = runtime->buildKernel("linear_attention_buf", "linear_attn_conv_state_update", buildOptions, mOpenCLBackend->getPrecision());

        int totalConvUpdate = batch * convDim * convStateSize;
        mGWSConvStateUpdate = {(uint32_t)totalConvUpdate, 1, 1};
        maxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernelConvStateUpdate));
        uint32_t lwsUpdate = std::min(maxWorkGroupSize, (uint32_t)256);
        lwsUpdate = std::min(lwsUpdate, (uint32_t)totalConvUpdate);
        mLWSConvStateUpdate = {lwsUpdate, 1, 1};
    }

    // Kernel 3: Gated Delta Rule
    mKernelGatedDeltaRule = runtime->buildKernel("linear_attention_buf", "linear_attn_gated_delta_rule", buildOptions, mOpenCLBackend->getPrecision());

    int totalHeads = batch * H;
    mGWSGatedDeltaRule = {(uint32_t)totalHeads, 1, 1};
    maxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernelGatedDeltaRule));
    uint32_t lwsDelta = std::min(maxWorkGroupSize, (uint32_t)256);
    lwsDelta = std::min(lwsDelta, (uint32_t)totalHeads);
    mLWSGatedDeltaRule = {lwsDelta, 1, 1};

    // Set kernel arguments
    // Kernel 1: conv_silu
    {
        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;
        ret |= mKernelConvSilu->get().setArg(idx++, totalConvSilu);
        ret |= mKernelConvSilu->get().setArg(idx++, openCLBuffer(inputs[0]));          // qkv
        ret |= mKernelConvSilu->get().setArg(idx++, openCLBuffer(mStateCache->mConvState.get()));   // conv_state
        ret |= mKernelConvSilu->get().setArg(idx++, openCLBuffer(inputs[3]));          // conv_weight
        ret |= mKernelConvSilu->get().setArg(idx++, openCLBuffer(mConvOut.get()));     // conv_out
        ret |= mKernelConvSilu->get().setArg(idx++, batch);
        ret |= mKernelConvSilu->get().setArg(idx++, convDim);
        ret |= mKernelConvSilu->get().setArg(idx++, seqLen);
        ret |= mKernelConvSilu->get().setArg(idx++, K_conv);
        ret |= mKernelConvSilu->get().setArg(idx++, convStateSize);
        MNN_CHECK_CL_SUCCESS(ret, "setArg linear_attn_conv_silu");
    }

    // Kernel 2: conv_state_update
    if (convStateSize > 0) {
        int totalConvUpdate = batch * convDim * convStateSize;
        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;
        ret |= mKernelConvStateUpdate->get().setArg(idx++, totalConvUpdate);
        ret |= mKernelConvStateUpdate->get().setArg(idx++, openCLBuffer(inputs[0]));           // qkv
        ret |= mKernelConvStateUpdate->get().setArg(idx++, openCLBuffer(mStateCache->mConvState.get()));    // conv_state
        ret |= mKernelConvStateUpdate->get().setArg(idx++, batch);
        ret |= mKernelConvStateUpdate->get().setArg(idx++, convDim);
        ret |= mKernelConvStateUpdate->get().setArg(idx++, seqLen);
        ret |= mKernelConvStateUpdate->get().setArg(idx++, convStateSize);
        MNN_CHECK_CL_SUCCESS(ret, "setArg linear_attn_conv_state_update");
    }

    // Kernel 3: gated_delta_rule
    {
        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;
        ret |= mKernelGatedDeltaRule->get().setArg(idx++, totalHeads);
        ret |= mKernelGatedDeltaRule->get().setArg(idx++, openCLBuffer(mConvOut.get()));           // conv_out
        ret |= mKernelGatedDeltaRule->get().setArg(idx++, openCLBuffer(inputs[1]));                // gate
        ret |= mKernelGatedDeltaRule->get().setArg(idx++, openCLBuffer(inputs[2]));                // beta
        ret |= mKernelGatedDeltaRule->get().setArg(idx++, openCLBuffer(mStateCache->mRecurrentState.get()));    // recurrent_state
        ret |= mKernelGatedDeltaRule->get().setArg(idx++, openCLBuffer(outputs[0]));               // attn_out
        ret |= mKernelGatedDeltaRule->get().setArg(idx++, batch);
        ret |= mKernelGatedDeltaRule->get().setArg(idx++, convDim);
        ret |= mKernelGatedDeltaRule->get().setArg(idx++, seqLen);
        ret |= mKernelGatedDeltaRule->get().setArg(idx++, mNumKHeads);
        ret |= mKernelGatedDeltaRule->get().setArg(idx++, mNumVHeads);
        ret |= mKernelGatedDeltaRule->get().setArg(idx++, dk);
        ret |= mKernelGatedDeltaRule->get().setArg(idx++, dv);
        ret |= mKernelGatedDeltaRule->get().setArg(idx++, key_dim);
        ret |= mKernelGatedDeltaRule->get().setArg(idx++, val_dim);
        ret |= mKernelGatedDeltaRule->get().setArg(idx++, gqa_factor);
        ret |= mKernelGatedDeltaRule->get().setArg(idx++, (int)mUseQKL2Norm);
        ret |= mKernelGatedDeltaRule->get().setArg(idx++, qScale);
        MNN_CHECK_CL_SUCCESS(ret, "setArg linear_attn_gated_delta_rule");
    }

    // Round up global work sizes to multiples of local work sizes
    mGWSConvSilu[0] = ROUND_UP(mGWSConvSilu[0], mLWSConvSilu[0]);
    if (convStateSize > 0) {
        mGWSConvStateUpdate[0] = ROUND_UP(mGWSConvStateUpdate[0], mLWSConvStateUpdate[0]);
    }
    mGWSGatedDeltaRule[0] = ROUND_UP(mGWSGatedDeltaRule[0], mLWSGatedDeltaRule[0]);

    // Record kernels for queue recording optimization
    mOpenCLBackend->startRecord(mRecording);
    mOpenCLBackend->recordKernel3d(mKernelConvSilu, mGWSConvSilu, mLWSConvSilu);
    if (convStateSize > 0) {
        mOpenCLBackend->recordKernel3d(mKernelConvStateUpdate, mGWSConvStateUpdate, mLWSConvStateUpdate);
    }
    mOpenCLBackend->recordKernel3d(mKernelGatedDeltaRule, mGWSGatedDeltaRule, mLWSGatedDeltaRule);
    mOpenCLBackend->endRecord(mRecording);

    return NO_ERROR;
}

ErrorCode LinearAttentionBufExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    int convStateSize = inputs[3]->length(2) - 1;

#ifdef ENABLE_OPENCL_TIME_PROFILER
    {
        cl::Event event;
        run3DKernelDefault(mKernelConvSilu, mGWSConvSilu, mLWSConvSilu, runtime, &event);
        runtime->pushEvent({"linear_attn_conv_silu", event});
    }
    if (convStateSize > 0) {
        cl::Event event;
        run3DKernelDefault(mKernelConvStateUpdate, mGWSConvStateUpdate, mLWSConvStateUpdate, runtime, &event);
        runtime->pushEvent({"linear_attn_conv_state_update", event});
    }
    {
        cl::Event event;
        run3DKernelDefault(mKernelGatedDeltaRule, mGWSGatedDeltaRule, mLWSGatedDeltaRule, runtime, &event);
        runtime->pushEvent({"linear_attn_gated_delta_rule", event});
    }
#else
    if(mOpenCLBackend->isUseRecordQueue()){
        mOpenCLBackend->addRecord(mRecording, mOpRecordUpdateInfo);
        return NO_ERROR;
    }
    run3DKernelDefault(mKernelConvSilu, mGWSConvSilu, mLWSConvSilu, runtime);
    if (convStateSize > 0) {
        run3DKernelDefault(mKernelConvStateUpdate, mGWSConvStateUpdate, mLWSConvStateUpdate, runtime);
    }
    run3DKernelDefault(mKernelGatedDeltaRule, mGWSGatedDeltaRule, mLWSGatedDeltaRule, runtime);
#endif

    return NO_ERROR;
}

bool LinearAttentionBufExecution::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (nullptr == dst) {
        return true;
    }
    auto exe = new LinearAttentionBufExecution(op, bn);
    // Share persistent state buffers between prefill and decode Executions
    exe->mStateCache = mStateCache;
    *dst = exe;
    return true;
}

class LinearAttentionBufCreator : public OpenCLBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        for (int i = 0; i < inputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(inputs[i], false);
        }
        for (int i = 0; i < outputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(outputs[i], false);
        }
        return new LinearAttentionBufExecution(op, backend);
    }
};
REGISTER_OPENCL_OP_CREATOR_TRANSFORMER(LinearAttentionBufCreator, OpType_LinearAttention, BUFFER);

} // namespace OpenCL
} // namespace MNN
#endif/* MNN_SUPPORT_TRANSFORMER_FUSE */
