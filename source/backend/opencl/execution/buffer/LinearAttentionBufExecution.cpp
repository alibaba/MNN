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

    // ─── Chunked prefill: fully independent branch ───
    mUseChunkedPrefill = (seqLen > 1);
    if (mUseChunkedPrefill) {
        return onResizeChunkedPrefill(inputs, outputs);
    }

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
            success &= backend()->onAcquireBuffer(mStateCache->mConvState.get(), Backend::STATIC);
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
    int local_size = 16;
    buildOptions.emplace("-DLOCAL_SIZE=" + std::to_string(local_size));
    buildOptions.emplace("-DK_SIZE=" + std::to_string(dv));

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
    auto gateDeltaRuleBuildOptions = buildOptions;
    if(seqLen == 1){
        gateDeltaRuleBuildOptions.emplace("-DDECODE_PHASE");
    }
    mKernelGatedDeltaRule = runtime->buildKernel("linear_attention_buf", "linear_attn_gated_delta_rule", gateDeltaRuleBuildOptions, mOpenCLBackend->getPrecision());

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

    // Kernel 2.5: l2
    if(mUseQKL2Norm){
        auto l2BuildOptions = buildOptions;
        if(seqLen > 1){
            l2BuildOptions.emplace("-DUSE_VEC");
        }
        mKernell2Norm = runtime->buildKernel("linear_attention_buf", "l2_norm", l2BuildOptions, mOpenCLBackend->getPrecision());

        mGWSl2Norm = {128, (uint32_t)(H * UP_DIV(seqLen, 4)), (uint32_t)(batch * 2)};
        mLWSl2Norm = {128, 1, 1};
        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;
        ret |= mKernell2Norm->get().setArg(idx++, openCLBuffer(mConvOut.get()));    // conv_out
        ret |= mKernell2Norm->get().setArg(idx++, openCLBuffer(mConvOut.get()));    // conv_out
        ret |= mKernell2Norm->get().setArg(idx++, convDim);
        ret |= mKernell2Norm->get().setArg(idx++, dk);
        ret |= mKernell2Norm->get().setArg(idx++, gqa_factor);
        ret |= mKernell2Norm->get().setArg(idx++, key_dim);
        ret |= mKernell2Norm->get().setArg(idx++, seqLen);
        MNN_CHECK_CL_SUCCESS(ret, "setArg l2 norm");
    }

    // Kernel 3: gated_delta_rule
    {
        mGWSGatedDeltaRule = {(uint32_t)local_size, (uint32_t)UP_DIV(dv, 4) * H * batch};
        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;
        ret |= mKernelGatedDeltaRule->get().setArg(idx++, openCLBuffer(mConvOut.get()));           // conv_out
        ret |= mKernelGatedDeltaRule->get().setArg(idx++, openCLBuffer(inputs[1]));                // gate
        ret |= mKernelGatedDeltaRule->get().setArg(idx++, openCLBuffer(inputs[2]));                // beta
        ret |= mKernelGatedDeltaRule->get().setArg(idx++, openCLBuffer(mStateCache->mRecurrentState.get()));    // recurrent_state id = 6
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
        ret |= mKernelGatedDeltaRule->get().setArg(idx++, qScale);
        MNN_CHECK_CL_SUCCESS(ret, "setArg linear_attn_gated_delta_rule");
        maxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernelGatedDeltaRule));
        mLWSGatedDeltaRule = {(uint32_t)local_size, 1};
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
    if(mUseQKL2Norm){
        mOpenCLBackend->recordKernel3d(mKernell2Norm, mGWSl2Norm, mLWSl2Norm);
    }
    mOpenCLBackend->recordKernel2d(mKernelGatedDeltaRule, mGWSGatedDeltaRule, mLWSGatedDeltaRule);
    mOpenCLBackend->endRecord(mRecording);

    return NO_ERROR;
}

ErrorCode LinearAttentionBufExecution::onResizeChunkedPrefill(
    const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    
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
    int gqa_factor = (mNumVHeads > mNumKHeads) ? (mNumVHeads / mNumKHeads) : 1;
    float qScale = 1.0f / sqrt((float)dk);

    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    int bytesPerElement = mOpenCLBackend->fpBytes();

    // ─── Persistent state buffers (STATIC): allocate once, shared via onClone ───
    if (mStateCache->mRecurrentState.get() == nullptr) {
        int rnnSize = batch * H * dk * dv;
        mStateCache->mRecurrentState.reset(Tensor::createDevice<float>({rnnSize}));
        mStateCache->mRecurrentStateTune.reset(Tensor::createDevice<float>({rnnSize}));
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
            success &= backend()->onAcquireBuffer(mStateCache->mConvState.get(), Backend::STATIC);
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
    } else {
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

    // ─── Allocate temporary buffers ───
    // IMPORTANT: All DYNAMIC buffers that are used together during execution must have
    // overlapping lifetimes (acquire all before releasing any) to prevent the memory
    // planner from aliasing them. mConvOutPrefill is read by C2-C5 and C7 concurrently
    // with chunk buffers, so they must all be alive simultaneously.
    mConvOutPrefill.reset(Tensor::createDevice<float>({batch * convDim * seqLen}));
    mOpenCLBackend->onAcquireBuffer(mStateCache->mRecurrentStateTune.get(), Backend::DYNAMIC);
    mOpenCLBackend->onReleaseBuffer(mStateCache->mRecurrentStateTune.get(), Backend::DYNAMIC);

    int chunkSize = mChunkSize;
    int numChunks = UP_DIV(seqLen, chunkSize);
    mNumChunks = numChunks;

    // Allocate intermediate buffers
    // Critical buffers use float32 for precision (allocate 2x elements in half mode to get 4N bytes = N floats)
    // Non-critical buffers use FLOAT (matches onAcquireBuffer precision)
    int fpBytes = mOpenCLBackend->fpBytes();
    auto f32Elems = [fpBytes](int n) { return (n * 4 + fpBytes - 1) / fpBytes; };
    mGCumsumBuf.reset(Tensor::createDevice<float>({f32Elems(batch * H * numChunks * chunkSize)}));
    mAttnMatrixBuf.reset(Tensor::createDevice<float>({f32Elems(batch * H * numChunks * chunkSize * chunkSize)}));
    mVCorrectedBuf.reset(Tensor::createDevice<float>({f32Elems(batch * H * numChunks * chunkSize * dv)}));
    mKCumdecayBuf.reset(Tensor::createDevice<float>({f32Elems(batch * H * numChunks * chunkSize * dk)}));
    mVNewBuf.reset(Tensor::createDevice<float>({f32Elems(batch * H * chunkSize * dv)}));
    // Acquire all buffers used concurrently during execution BEFORE releasing any
    mOpenCLBackend->onAcquireBuffer(mConvOutPrefill.get(), Backend::DYNAMIC);
    mOpenCLBackend->onAcquireBuffer(mGCumsumBuf.get(), Backend::DYNAMIC);
    mOpenCLBackend->onAcquireBuffer(mAttnMatrixBuf.get(), Backend::DYNAMIC);
    mOpenCLBackend->onAcquireBuffer(mVCorrectedBuf.get(), Backend::DYNAMIC);
    mOpenCLBackend->onAcquireBuffer(mKCumdecayBuf.get(), Backend::DYNAMIC);
    mOpenCLBackend->onAcquireBuffer(mVNewBuf.get(), Backend::DYNAMIC);
    // Release all together — planner now sees overlapping lifetimes, no aliasing
    mOpenCLBackend->onReleaseBuffer(mConvOutPrefill.get(), Backend::DYNAMIC);
    mOpenCLBackend->onReleaseBuffer(mGCumsumBuf.get(), Backend::DYNAMIC);
    mOpenCLBackend->onReleaseBuffer(mAttnMatrixBuf.get(), Backend::DYNAMIC);
    mOpenCLBackend->onReleaseBuffer(mVCorrectedBuf.get(), Backend::DYNAMIC);
    mOpenCLBackend->onReleaseBuffer(mKCumdecayBuf.get(), Backend::DYNAMIC);
    mOpenCLBackend->onReleaseBuffer(mVNewBuf.get(), Backend::DYNAMIC);

    // ─── Build common kernels for prefill ───
    std::set<std::string> buildOptions;
    
    int local_size = 16;
    buildOptions.emplace("-DLOCAL_SIZE=" + std::to_string(local_size));
    buildOptions.emplace("-DK_SIZE=" + std::to_string(dv));
    // Conv1D + SiLU
    mKernelConvSiluPrefill = runtime->buildKernel("linear_attention_buf", "linear_attn_conv_silu", buildOptions, mOpenCLBackend->getPrecision());
    int totalConvSilu = batch * convDim * seqLen;
    mGWSConvSiluPrefill = {(uint32_t)totalConvSilu, 1, 1};
    {
        auto maxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernelConvSiluPrefill));
        uint32_t lwsConv = std::min(maxWorkGroupSize, (uint32_t)256);
        lwsConv = std::min(lwsConv, (uint32_t)totalConvSilu);
        mLWSConvSiluPrefill = {lwsConv, 1, 1};
    }
    {
        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;
        ret |= mKernelConvSiluPrefill->get().setArg(idx++, totalConvSilu);
        ret |= mKernelConvSiluPrefill->get().setArg(idx++, openCLBuffer(inputs[0]));
        ret |= mKernelConvSiluPrefill->get().setArg(idx++, openCLBuffer(mStateCache->mConvState.get()));
        ret |= mKernelConvSiluPrefill->get().setArg(idx++, openCLBuffer(inputs[3]));
        ret |= mKernelConvSiluPrefill->get().setArg(idx++, openCLBuffer(mConvOutPrefill.get()));
        ret |= mKernelConvSiluPrefill->get().setArg(idx++, batch);
        ret |= mKernelConvSiluPrefill->get().setArg(idx++, convDim);
        ret |= mKernelConvSiluPrefill->get().setArg(idx++, seqLen);
        ret |= mKernelConvSiluPrefill->get().setArg(idx++, K_conv);
        ret |= mKernelConvSiluPrefill->get().setArg(idx++, convStateSize);
        MNN_CHECK_CL_SUCCESS(ret, "setArg linear_attn_conv_silu (prefill)");
    }
    mGWSConvSiluPrefill[0] = ROUND_UP(mGWSConvSiluPrefill[0], mLWSConvSiluPrefill[0]);

    // Conv state update
    if (convStateSize > 0) {
        mKernelConvStateUpdatePrefill = runtime->buildKernel("linear_attention_buf", "linear_attn_conv_state_update", buildOptions, mOpenCLBackend->getPrecision());
        int totalConvUpdate = batch * convDim * convStateSize;
        mGWSConvStateUpdatePrefill = {(uint32_t)totalConvUpdate, 1, 1};
        {
            auto maxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernelConvStateUpdatePrefill));
            uint32_t lwsUpdate = std::min(maxWorkGroupSize, (uint32_t)256);
            lwsUpdate = std::min(lwsUpdate, (uint32_t)totalConvUpdate);
            mLWSConvStateUpdatePrefill = {lwsUpdate, 1, 1};
        }
        {
            uint32_t idx = 0;
            cl_int ret = CL_SUCCESS;
            ret |= mKernelConvStateUpdatePrefill->get().setArg(idx++, totalConvUpdate);
            ret |= mKernelConvStateUpdatePrefill->get().setArg(idx++, openCLBuffer(inputs[0]));
            ret |= mKernelConvStateUpdatePrefill->get().setArg(idx++, openCLBuffer(mStateCache->mConvState.get()));
            ret |= mKernelConvStateUpdatePrefill->get().setArg(idx++, batch);
            ret |= mKernelConvStateUpdatePrefill->get().setArg(idx++, convDim);
            ret |= mKernelConvStateUpdatePrefill->get().setArg(idx++, seqLen);
            ret |= mKernelConvStateUpdatePrefill->get().setArg(idx++, convStateSize);
            MNN_CHECK_CL_SUCCESS(ret, "setArg linear_attn_conv_state_update (prefill)");
        }
        mGWSConvStateUpdatePrefill[0] = ROUND_UP(mGWSConvStateUpdatePrefill[0], mLWSConvStateUpdatePrefill[0]);
    }

    // L2 norm
    if (mUseQKL2Norm) {
        auto l2BuildOptions = buildOptions;
        l2BuildOptions.emplace("-DUSE_VEC");
        mKernell2NormPrefill = runtime->buildKernel("linear_attention_buf", "l2_norm", l2BuildOptions, mOpenCLBackend->getPrecision());
        mGWSl2NormPrefill = {128, (uint32_t)(H * UP_DIV(seqLen, 4)), (uint32_t)(batch * 2)};
        mLWSl2NormPrefill = {128, 1, 1};
        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;
        ret |= mKernell2NormPrefill->get().setArg(idx++, openCLBuffer(mConvOutPrefill.get()));
        ret |= mKernell2NormPrefill->get().setArg(idx++, openCLBuffer(mConvOutPrefill.get()));
        ret |= mKernell2NormPrefill->get().setArg(idx++, convDim);
        ret |= mKernell2NormPrefill->get().setArg(idx++, dk);
        ret |= mKernell2NormPrefill->get().setArg(idx++, gqa_factor);
        ret |= mKernell2NormPrefill->get().setArg(idx++, key_dim);
        ret |= mKernell2NormPrefill->get().setArg(idx++, seqLen);
        MNN_CHECK_CL_SUCCESS(ret, "setArg l2 norm (prefill)");
    }

    // ─── Build chunked prefill kernels ───
    std::set<std::string> chunkOpts = buildOptions;
    chunkOpts.emplace("-DCHUNK_PREFILL");
    chunkOpts.emplace("-DCHUNK_SIZE=" + std::to_string(chunkSize));

    // C1: chunk_g_cumsum
    mKernelChunkGCumsum = runtime->buildKernel("linear_attention_buf", "chunk_g_cumsum", chunkOpts, mOpenCLBackend->getPrecision());
    mGWSChunkGCumsum = {(uint32_t)H, (uint32_t)numChunks, (uint32_t)batch};
    mLWSChunkGCumsum = {1, 1, 1};
    {
        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;
        ret |= mKernelChunkGCumsum->get().setArg(idx++, openCLBuffer(inputs[1]));         // gate
        ret |= mKernelChunkGCumsum->get().setArg(idx++, openCLBuffer(mGCumsumBuf.get())); // g_cumsum
        ret |= mKernelChunkGCumsum->get().setArg(idx++, H);
        ret |= mKernelChunkGCumsum->get().setArg(idx++, seqLen);
        ret |= mKernelChunkGCumsum->get().setArg(idx++, numChunks);
        MNN_CHECK_CL_SUCCESS(ret, "setArg chunk_g_cumsum");
    }
    
    {
        {
            // C2: chunk_build_neumann_attn
            mKernelChunkNeumannAttn0 = runtime->buildKernel("linear_attention_buf", "chunk_build_neumann_attn_step0", chunkOpts, mOpenCLBackend->getPrecision());
            mGWSChunkNeumannAttn0 = {(uint32_t)chunkSize * chunkSize, (uint32_t)(H * numChunks), (uint32_t)batch};
            uint32_t idx = 0;
            cl_int ret = CL_SUCCESS;
            ret |= mKernelChunkNeumannAttn0->get().setArg(idx++, openCLBuffer(mConvOutPrefill.get()));
            ret |= mKernelChunkNeumannAttn0->get().setArg(idx++, openCLBuffer(inputs[2]));             // beta
            ret |= mKernelChunkNeumannAttn0->get().setArg(idx++, openCLBuffer(mGCumsumBuf.get()));
            ret |= mKernelChunkNeumannAttn0->get().setArg(idx++, openCLBuffer(mAttnMatrixBuf.get()));
            ret |= mKernelChunkNeumannAttn0->get().setArg(idx++, batch);
            ret |= mKernelChunkNeumannAttn0->get().setArg(idx++, convDim);
            ret |= mKernelChunkNeumannAttn0->get().setArg(idx++, seqLen);
            ret |= mKernelChunkNeumannAttn0->get().setArg(idx++, H);
            ret |= mKernelChunkNeumannAttn0->get().setArg(idx++, dk);
            ret |= mKernelChunkNeumannAttn0->get().setArg(idx++, key_dim);
            ret |= mKernelChunkNeumannAttn0->get().setArg(idx++, gqa_factor);
            ret |= mKernelChunkNeumannAttn0->get().setArg(idx++, numChunks);
            MNN_CHECK_CL_SUCCESS(ret, "setArg chunk_build_neumann_attn_step0");
            auto maxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernelChunkNeumannAttn0));
            mLWSChunkNeumannAttn0 = localWS3DDefault(mGWSChunkNeumannAttn0, maxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), "chunk_build_neumann_attn_step0", mKernelChunkNeumannAttn0, mOpenCLBackend->getCLTuneLevel(), "linear_attention_buf").first;
        }
        {
            // C2: chunk_build_neumann_attn
            mKernelChunkNeumannAttn1 = runtime->buildKernel("linear_attention_buf", "chunk_build_neumann_attn_step1", chunkOpts, mOpenCLBackend->getPrecision());
            mGWSChunkNeumannAttn1 = {(uint32_t)chunkSize, (uint32_t)(H * numChunks), (uint32_t)batch};
            mLWSChunkNeumannAttn1 = {(uint32_t)chunkSize, 1, 1};
            uint32_t idx = 0;
            cl_int ret = CL_SUCCESS;
            ret |= mKernelChunkNeumannAttn1->get().setArg(idx++, openCLBuffer(mAttnMatrixBuf.get()));
            ret |= mKernelChunkNeumannAttn1->get().setArg(idx++, batch);
            ret |= mKernelChunkNeumannAttn1->get().setArg(idx++, convDim);
            ret |= mKernelChunkNeumannAttn1->get().setArg(idx++, seqLen);
            ret |= mKernelChunkNeumannAttn1->get().setArg(idx++, H);
            ret |= mKernelChunkNeumannAttn1->get().setArg(idx++, dk);
            ret |= mKernelChunkNeumannAttn1->get().setArg(idx++, key_dim);
            ret |= mKernelChunkNeumannAttn1->get().setArg(idx++, gqa_factor);
            ret |= mKernelChunkNeumannAttn1->get().setArg(idx++, numChunks);
            MNN_CHECK_CL_SUCCESS(ret, "setArg chunk_build_neumann_attn_step1");
        }
    }

    // C3: chunk_correct_v
    mKernelChunkCorrectV = runtime->buildKernel("linear_attention_buf", "chunk_correct_v", chunkOpts, mOpenCLBackend->getPrecision());
    mGWSChunkCorrectV = {(uint32_t)UP_DIV(dv, 4), (uint32_t)(chunkSize * numChunks), (uint32_t)(batch * H)};
    {
        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;
        ret |= mKernelChunkCorrectV->get().setArg(idx++, openCLBuffer(mAttnMatrixBuf.get()));
        ret |= mKernelChunkCorrectV->get().setArg(idx++, openCLBuffer(mConvOutPrefill.get()));
        ret |= mKernelChunkCorrectV->get().setArg(idx++, openCLBuffer(inputs[2]));                // beta
        ret |= mKernelChunkCorrectV->get().setArg(idx++, openCLBuffer(mGCumsumBuf.get()));
        ret |= mKernelChunkCorrectV->get().setArg(idx++, openCLBuffer(mVCorrectedBuf.get()));
        ret |= mKernelChunkCorrectV->get().setArg(idx++, openCLBuffer(mKCumdecayBuf.get()));
        ret |= mKernelChunkCorrectV->get().setArg(idx++, mGWSChunkCorrectV[0]);
        ret |= mKernelChunkCorrectV->get().setArg(idx++, mGWSChunkCorrectV[1]);
        ret |= mKernelChunkCorrectV->get().setArg(idx++, mGWSChunkCorrectV[2]);
        ret |= mKernelChunkCorrectV->get().setArg(idx++, convDim);
        ret |= mKernelChunkCorrectV->get().setArg(idx++, seqLen);
        ret |= mKernelChunkCorrectV->get().setArg(idx++, H);
        ret |= mKernelChunkCorrectV->get().setArg(idx++, dk);
        ret |= mKernelChunkCorrectV->get().setArg(idx++, dv);
        ret |= mKernelChunkCorrectV->get().setArg(idx++, key_dim);
        ret |= mKernelChunkCorrectV->get().setArg(idx++, numChunks);
        MNN_CHECK_CL_SUCCESS(ret, "setArg chunk_correct_v");
        auto maxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernelChunkCorrectV));
        mLWSChunkCorrectV = localWS3DDefault(mGWSChunkCorrectV, maxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), "chunk_correct_v", mKernelChunkCorrectV, mOpenCLBackend->getCLTuneLevel(), "linear_attention_buf").first;
    }

    // C5: chunk_qk_attn (reuses attn_matrix buffer)
    mKernelChunkQKAttn = runtime->buildKernel("linear_attention_buf", "chunk_qk_attn", chunkOpts, mOpenCLBackend->getPrecision());
    mGWSChunkQKAttn = {(uint32_t)chunkSize, (uint32_t)(chunkSize * numChunks), (uint32_t)(batch * H)};
    {
        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;
        ret |= mKernelChunkQKAttn->get().setArg(idx++, openCLBuffer(mConvOutPrefill.get()));
        ret |= mKernelChunkQKAttn->get().setArg(idx++, openCLBuffer(mGCumsumBuf.get()));
        ret |= mKernelChunkQKAttn->get().setArg(idx++, openCLBuffer(mAttnMatrixBuf.get()));  // overwrite
        ret |= mKernelChunkQKAttn->get().setArg(idx++, mGWSChunkQKAttn[0]);
        ret |= mKernelChunkQKAttn->get().setArg(idx++, mGWSChunkQKAttn[1]);
        ret |= mKernelChunkQKAttn->get().setArg(idx++, mGWSChunkQKAttn[2]);
        ret |= mKernelChunkQKAttn->get().setArg(idx++, convDim);
        ret |= mKernelChunkQKAttn->get().setArg(idx++, seqLen);
        ret |= mKernelChunkQKAttn->get().setArg(idx++, H);
        ret |= mKernelChunkQKAttn->get().setArg(idx++, dk);
        ret |= mKernelChunkQKAttn->get().setArg(idx++, key_dim);
        ret |= mKernelChunkQKAttn->get().setArg(idx++, gqa_factor);
        ret |= mKernelChunkQKAttn->get().setArg(idx++, numChunks);
        ret |= mKernelChunkQKAttn->get().setArg(idx++, qScale);
        MNN_CHECK_CL_SUCCESS(ret, "setArg chunk_qk_attn");
        auto maxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernelChunkQKAttn));
        mLWSChunkQKAttn = localWS3DDefault(mGWSChunkQKAttn, maxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), "chunk_qk_attn", mKernelChunkQKAttn, mOpenCLBackend->getCLTuneLevel(), "linear_attention_buf").first;
    }

    // C6: chunk_compute_vnew (per-chunk, chunk_idx=11 set dynamically)
    mKernelChunkVnew = runtime->buildKernel("linear_attention_buf", "chunk_compute_vnew", chunkOpts, mOpenCLBackend->getPrecision());
    mGWSChunkVnew = {(uint32_t)UP_DIV(dv, 4), (uint32_t)chunkSize, (uint32_t)(batch * H)};
    {
        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;
        ret |= mKernelChunkVnew->get().setArg(idx++, openCLBuffer(mVCorrectedBuf.get()));
        ret |= mKernelChunkVnew->get().setArg(idx++, openCLBuffer(mKCumdecayBuf.get()));
        ret |= mKernelChunkVnew->get().setArg(idx++, openCLBuffer(mStateCache->mRecurrentStateTune.get()));  // arg 2: state (tune first)
        ret |= mKernelChunkVnew->get().setArg(idx++, openCLBuffer(mVNewBuf.get()));
        ret |= mKernelChunkVnew->get().setArg(idx++, mGWSChunkVnew[0]);
        ret |= mKernelChunkVnew->get().setArg(idx++, mGWSChunkVnew[1]);
        ret |= mKernelChunkVnew->get().setArg(idx++, mGWSChunkVnew[2]);
        ret |= mKernelChunkVnew->get().setArg(idx++, dk);
        ret |= mKernelChunkVnew->get().setArg(idx++, dv);
        ret |= mKernelChunkVnew->get().setArg(idx++, H);
        ret |= mKernelChunkVnew->get().setArg(idx++, numChunks);
        ret |= mKernelChunkVnew->get().setArg(idx++, 0);  // chunk_idx placeholder
        MNN_CHECK_CL_SUCCESS(ret, "setArg chunk_compute_vnew");
        auto maxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernelChunkVnew));
        mLWSChunkVnew = localWS3DDefault(mGWSChunkVnew, maxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), "chunk_compute_vnew", mKernelChunkVnew, mOpenCLBackend->getCLTuneLevel(), "linear_attention_buf").first;
        // Swap to real state buffer after tuning
        ret |= mKernelChunkVnew->get().setArg(2, openCLBuffer(mStateCache->mRecurrentState.get()));
    }
    
    // C6.5: chunk_output (per-chunk, chunk_idx=17 set dynamically)
    mKernelChunkOutput = runtime->buildKernel("linear_attention_buf", "chunk_output", chunkOpts, mOpenCLBackend->getPrecision());
    mGWSChunkOutput = {(uint32_t)UP_DIV(dv, 4) * chunkSize,  (uint32_t)H, (uint32_t)batch};
    {
        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;
        ret |= mKernelChunkOutput->get().setArg(idx++, openCLBuffer(mConvOutPrefill.get()));
        ret |= mKernelChunkOutput->get().setArg(idx++, openCLBuffer(mAttnMatrixBuf.get()));  // qk_attn after C5
        ret |= mKernelChunkOutput->get().setArg(idx++, openCLBuffer(mVNewBuf.get()));
        ret |= mKernelChunkOutput->get().setArg(idx++, openCLBuffer(mGCumsumBuf.get()));
        ret |= mKernelChunkOutput->get().setArg(idx++, openCLBuffer(mStateCache->mRecurrentState.get()));  // arg 4: state (tune first)
        ret |= mKernelChunkOutput->get().setArg(idx++, openCLBuffer(outputs[0]));
        ret |= mKernelChunkOutput->get().setArg(idx++, mGWSChunkOutput[0]);
        ret |= mKernelChunkOutput->get().setArg(idx++, mGWSChunkOutput[1]);
        ret |= mKernelChunkOutput->get().setArg(idx++, mGWSChunkOutput[2]);
        ret |= mKernelChunkOutput->get().setArg(idx++, convDim);
        ret |= mKernelChunkOutput->get().setArg(idx++, seqLen);
        ret |= mKernelChunkOutput->get().setArg(idx++, H);
        ret |= mKernelChunkOutput->get().setArg(idx++, dk);
        ret |= mKernelChunkOutput->get().setArg(idx++, dv);
        ret |= mKernelChunkOutput->get().setArg(idx++, key_dim);
        ret |= mKernelChunkOutput->get().setArg(idx++, gqa_factor);
        ret |= mKernelChunkOutput->get().setArg(idx++, numChunks);
        ret |= mKernelChunkOutput->get().setArg(idx++, 0);  // chunk_idx placeholder
        ret |= mKernelChunkOutput->get().setArg(idx++, qScale);
        MNN_CHECK_CL_SUCCESS(ret, "setArg chunk_output");
        auto maxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernelChunkOutput));
        mLWSChunkOutput = localWS3DDefault(mGWSChunkOutput, maxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), "chunk_output", mKernelChunkOutput, mOpenCLBackend->getCLTuneLevel(), "linear_attention_buf").first;
    }

    // C7: chunk_output_state_update (per-chunk, chunk_idx=17 set dynamically)
    mKernelChunkOutputUpdate = runtime->buildKernel("linear_attention_buf", "chunk_output_state_update", chunkOpts, mOpenCLBackend->getPrecision());
    mGWSChunkOutputUpdate = {(uint32_t)UP_DIV(dv, 4) * dk, (uint32_t)H, (uint32_t)batch};
    {
        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;
        ret |= mKernelChunkOutputUpdate->get().setArg(idx++, openCLBuffer(mConvOutPrefill.get()));
        ret |= mKernelChunkOutputUpdate->get().setArg(idx++, openCLBuffer(mAttnMatrixBuf.get()));  // qk_attn after C5
        ret |= mKernelChunkOutputUpdate->get().setArg(idx++, openCLBuffer(mVNewBuf.get()));
        ret |= mKernelChunkOutputUpdate->get().setArg(idx++, openCLBuffer(mGCumsumBuf.get()));
        ret |= mKernelChunkOutputUpdate->get().setArg(idx++, openCLBuffer(mStateCache->mRecurrentStateTune.get()));  // arg 4: state (tune first)
        ret |= mKernelChunkOutputUpdate->get().setArg(idx++, openCLBuffer(outputs[0]));
        ret |= mKernelChunkOutputUpdate->get().setArg(idx++, mGWSChunkOutputUpdate[0]);
        ret |= mKernelChunkOutputUpdate->get().setArg(idx++, mGWSChunkOutputUpdate[1]);
        ret |= mKernelChunkOutputUpdate->get().setArg(idx++, mGWSChunkOutputUpdate[2]);
        ret |= mKernelChunkOutputUpdate->get().setArg(idx++, convDim);
        ret |= mKernelChunkOutputUpdate->get().setArg(idx++, seqLen);
        ret |= mKernelChunkOutputUpdate->get().setArg(idx++, H);
        ret |= mKernelChunkOutputUpdate->get().setArg(idx++, dk);
        ret |= mKernelChunkOutputUpdate->get().setArg(idx++, dv);
        ret |= mKernelChunkOutputUpdate->get().setArg(idx++, key_dim);
        ret |= mKernelChunkOutputUpdate->get().setArg(idx++, gqa_factor);
        ret |= mKernelChunkOutputUpdate->get().setArg(idx++, numChunks);
        ret |= mKernelChunkOutputUpdate->get().setArg(idx++, 0);  // chunk_idx placeholder
        ret |= mKernelChunkOutputUpdate->get().setArg(idx++, qScale);
        MNN_CHECK_CL_SUCCESS(ret, "setArg chunk_output_state_update");
        auto maxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernelChunkOutputUpdate));
        mLWSChunkOutputUpdate = localWS3DDefault(mGWSChunkOutputUpdate, maxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), "chunk_output_state_update", mKernelChunkOutputUpdate, mOpenCLBackend->getCLTuneLevel(), "linear_attention_buf").first;
        // Swap to real state buffer after tuning
        ret |= mKernelChunkOutputUpdate->get().setArg(4, openCLBuffer(mStateCache->mRecurrentState.get()));
    }

    // Round up chunked GWS
    for (auto& gws_lws : std::vector<std::pair<std::vector<uint32_t>*, std::vector<uint32_t>*>>{
        {&mGWSChunkCorrectV, &mLWSChunkCorrectV}, {&mGWSChunkNeumannAttn0, &mLWSChunkNeumannAttn0}, {&mGWSChunkNeumannAttn1, &mLWSChunkNeumannAttn1},
        {&mGWSChunkQKAttn, &mLWSChunkQKAttn}, {&mGWSChunkVnew, &mLWSChunkVnew},{&mGWSChunkOutput, &mLWSChunkOutput},
        {&mGWSChunkOutputUpdate, &mLWSChunkOutputUpdate}}) {
        for (int d = 0; d < 3; ++d) {
            (*gws_lws.first)[d] = ROUND_UP((*gws_lws.first)[d], std::max((uint32_t)1, (*gws_lws.second)[d]));
        }
    }

    return NO_ERROR;
}

ErrorCode LinearAttentionBufExecution::onExecuteChunkedPrefill(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    int convStateSize = inputs[3]->length(2) - 1;

#ifdef ENABLE_OPENCL_TIME_PROFILER
    { cl::Event event; run3DKernelDefault(mKernelConvSiluPrefill, mGWSConvSiluPrefill, mLWSConvSiluPrefill, runtime, &event); runtime->pushEvent({"linear_attn_conv_silu", event}); }
    if (convStateSize > 0) {
        cl::Event event; run3DKernelDefault(mKernelConvStateUpdatePrefill, mGWSConvStateUpdatePrefill, mLWSConvStateUpdatePrefill, runtime, &event); runtime->pushEvent({"linear_attn_conv_state_update", event});
    }
    if (mUseQKL2Norm) {
        cl::Event event; run3DKernelDefault(mKernell2NormPrefill, mGWSl2NormPrefill, mLWSl2NormPrefill, runtime, &event); runtime->pushEvent({"l2_norm", event});
    }
    { cl::Event e; run3DKernelDefault(mKernelChunkGCumsum, mGWSChunkGCumsum, mLWSChunkGCumsum, runtime, &e); runtime->pushEvent({"chunk_g_cumsum", e}); }
    { cl::Event e; run3DKernelDefault(mKernelChunkNeumannAttn0, mGWSChunkNeumannAttn0, mLWSChunkNeumannAttn0, runtime, &e); runtime->pushEvent({"chunk_build_neumann_attn0", e}); }
    { cl::Event e; run3DKernelDefault(mKernelChunkNeumannAttn1, mGWSChunkNeumannAttn1, mLWSChunkNeumannAttn1, runtime, &e); runtime->pushEvent({"chunk_build_neumann_attn1", e}); }
    { cl::Event e; run3DKernelDefault(mKernelChunkCorrectV, mGWSChunkCorrectV, mLWSChunkCorrectV, runtime, &e); runtime->pushEvent({"chunk_correct_v", e}); }
    { cl::Event e; run3DKernelDefault(mKernelChunkQKAttn, mGWSChunkQKAttn, mLWSChunkQKAttn, runtime, &e); runtime->pushEvent({"chunk_qk_attn", e}); }
    for (int c = 0; c < mNumChunks; ++c) {
        mKernelChunkVnew->get().setArg(11, c);
        mKernelChunkOutput->get().setArg(17, c);
        mKernelChunkOutputUpdate->get().setArg(17, c);
        { cl::Event e; run3DKernelDefault(mKernelChunkVnew, mGWSChunkVnew, mLWSChunkVnew, runtime, &e); runtime->pushEvent({"chunk_vnew_" + std::to_string(c), e}); }
        { cl::Event e; runKernel2D(mKernelChunkOutput, mGWSChunkOutput, mLWSChunkOutput, runtime, &e); runtime->pushEvent({"chunk_output_" + std::to_string(c), e}); }
        { cl::Event e; run3DKernelDefault(mKernelChunkOutputUpdate, mGWSChunkOutputUpdate, mLWSChunkOutputUpdate, runtime, &e); runtime->pushEvent({"chunk_update" + std::to_string(c), e}); }
    }
#else
    // Common kernels
    run3DKernelDefault(mKernelConvSiluPrefill, mGWSConvSiluPrefill, mLWSConvSiluPrefill, runtime);
    if (convStateSize > 0) {
        run3DKernelDefault(mKernelConvStateUpdatePrefill, mGWSConvStateUpdatePrefill, mLWSConvStateUpdatePrefill, runtime);
    }
    if (mUseQKL2Norm) {
        run3DKernelDefault(mKernell2NormPrefill, mGWSl2NormPrefill, mLWSl2NormPrefill, runtime);
    }
    // Chunked prefill: C1 → C2 → C3, C4 → C5 → loop(C6, C7)
    run3DKernelDefault(mKernelChunkGCumsum, mGWSChunkGCumsum, mLWSChunkGCumsum, runtime);
    run3DKernelDefault(mKernelChunkNeumannAttn0, mGWSChunkNeumannAttn0, mLWSChunkNeumannAttn0, runtime);
    run3DKernelDefault(mKernelChunkNeumannAttn1, mGWSChunkNeumannAttn1, mLWSChunkNeumannAttn1, runtime);
    run3DKernelDefault(mKernelChunkCorrectV, mGWSChunkCorrectV, mLWSChunkCorrectV, runtime);
    run3DKernelDefault(mKernelChunkQKAttn, mGWSChunkQKAttn, mLWSChunkQKAttn, runtime);
    for (int c = 0; c < mNumChunks; ++c) {
        mKernelChunkVnew->get().setArg(11, c);           // chunk_idx at position 11
        mKernelChunkOutput->get().setArg(17, c);   // chunk_idx at position 17
        mKernelChunkOutputUpdate->get().setArg(17, c);   // chunk_idx at position 17
        run3DKernelDefault(mKernelChunkVnew, mGWSChunkVnew, mLWSChunkVnew, runtime);
        runKernel2D(mKernelChunkOutput, mGWSChunkOutput, mLWSChunkOutput, runtime);
        run3DKernelDefault(mKernelChunkOutputUpdate, mGWSChunkOutputUpdate, mLWSChunkOutputUpdate, runtime);
    }
#endif

    return NO_ERROR;
}

ErrorCode LinearAttentionBufExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    if (mUseChunkedPrefill) {
        return onExecuteChunkedPrefill(inputs, outputs);
    }
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
    if(mUseQKL2Norm){
        cl::Event event;
        run3DKernelDefault(mKernell2Norm, mGWSl2Norm, mLWSl2Norm, runtime, &event);
        runtime->pushEvent({"l2_norm", event});
    }
    {
        cl::Event event;
        runKernel2D(mKernelGatedDeltaRule, mGWSGatedDeltaRule, mLWSGatedDeltaRule, runtime, &event);
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
    if(mUseQKL2Norm){
        run3DKernelDefault(mKernell2Norm, mGWSl2Norm, mLWSl2Norm, runtime);
    }
    runKernel2D(mKernelGatedDeltaRule, mGWSGatedDeltaRule, mLWSGatedDeltaRule, runtime);
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
        OPENCL_CREATOR_CHECK(new LinearAttentionBufExecution(op, backend));
    }
};
REGISTER_OPENCL_OP_CREATOR_TRANSFORMER(LinearAttentionBufCreator, OpType_LinearAttention, BUFFER);

} // namespace OpenCL
} // namespace MNN
#endif/* MNN_SUPPORT_TRANSFORMER_FUSE */
