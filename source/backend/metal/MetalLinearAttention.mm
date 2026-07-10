//
//  MetalLinearAttention.mm
//  MNN
//
//  Created by MNN on 2026/02/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "MetalLinearAttention.hpp"
#import "MNNMetalContext.h"
#import "MetalLinearAttentionShader.hpp"
#import "core/TensorUtils.hpp"

#if MNN_METAL_ENABLED
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

namespace MNN {

// Must match LinearAttnParam in MetalLinearAttentionShader.hpp
struct LinearAttnParam {
    int batch;
    int conv_dim;
    int seq_len;
    int kernel_size;
    int conv_state_size;
    int num_k_heads;
    int num_v_heads;
    int head_k_dim;
    int head_v_dim;
    int key_dim;
    int val_dim;
    int gqa_factor;
    int use_l2norm;
    int qkv_c4;
    int gate_c4;
    int beta_c4;
    int output_c4;
    float q_scale;
};

static void linearAttentionDims(const Tensor* qkv, int& batch, int& convDim, int& seqLen) {
    if (TensorUtils::getDescribe(qkv)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
        batch = 1;
        seqLen = qkv->length(0);
        convDim = qkv->length(1);
        return;
    }
    batch = qkv->length(0);
    convDim = qkv->length(1);
    seqLen = qkv->length(2);
}

MetalLinearAttention::MetalLinearAttention(Backend *backend, const MNN::Op* op)
    : MetalExecution(backend) {
    auto param = op->main_as_LinearAttentionParam();
    mAttentionType = param->attn_type()->str();
    mNumKHeads = param->num_k_heads();
    mNumVHeads = param->num_v_heads();
    mHeadKDim = param->head_k_dim();
    mHeadVDim = param->head_v_dim();
    mUseQKL2Norm = param->use_qk_l2norm();
    mStateCache.reset(new MetalStateCache);

    auto mtbn = static_cast<MetalBackend *>(backend);
    mMeta = (KVMeta*)(mtbn->getMetaPtr());
    auto context = (__bridge MNNMetalContext *)mtbn->context();
    mParamBuffer = [context newDeviceBuffer:sizeof(LinearAttnParam) access:CPUWriteOnly];

    // Compile shader pipelines
    MTLCompileOptions *option = [[MTLCompileOptions alloc] init];
    auto rt = (MetalRuntime *)mtbn->runtime();
    bool useFp16 = mtbn->useFp16InsteadFp32();
    if (useFp16) {
        option.preprocessorMacros = @{@"MNN_METAL_FLOAT16_STORAGE" : @"1"};
    }

    if (mAttentionType == "short_conv") {
        std::vector<std::string> commonKeys;
        if (useFp16) {
            commonKeys.emplace_back("MNN_METAL_FLOAT16_STORAGE");
        }
        auto buildShortPipeline = [&](const char* kernel) -> id<MTLComputePipelineState> {
            auto keys = commonKeys;
            keys.insert(keys.begin(), kernel);
            id<MTLComputePipelineState> pipeline = rt->findPipeline(keys);
            if (nil == pipeline) {
                pipeline = mtbn->makeComputePipelineWithSourceOption(gLinearAttnShortConv, kernel, option);
                rt->insertPipeline(keys, pipeline);
            }
            return pipeline;
        };
        mShortConvPipeline = buildShortPipeline("linear_attn_short_conv_nosilu");
        mShortConvStateUpdatePipeline = buildShortPipeline("linear_attn_short_conv_state_update");
        mShortConvOutputPipeline = buildShortPipeline("linear_attn_short_conv_output");
        return;
    }

    // Conv + SiLU pipeline (includes both conv_silu and conv_state_update kernels)
    {
        std::vector<std::string> keys = {"linear_attn_conv_silu"};
        if (useFp16) keys.emplace_back("MNN_METAL_FLOAT16_STORAGE");
        mConvSiluPipeline = rt->findPipeline(keys);
        if (nil == mConvSiluPipeline) {
            mConvSiluPipeline = mtbn->makeComputePipelineWithSourceOption(gLinearAttnConvSilu, "linear_attn_conv_silu", option);
            rt->insertPipeline(keys, mConvSiluPipeline);
        }
    }
    {
        std::vector<std::string> keys = {"linear_attn_conv_state_update"};
        if (useFp16) keys.emplace_back("MNN_METAL_FLOAT16_STORAGE");
        mConvStateUpdatePipeline = rt->findPipeline(keys);
        if (nil == mConvStateUpdatePipeline) {
            mConvStateUpdatePipeline = mtbn->makeComputePipelineWithSourceOption(gLinearAttnConvSilu, "linear_attn_conv_state_update", option);
            rt->insertPipeline(keys, mConvStateUpdatePipeline);
        }
    }
    // QKV prep pipeline
    {
        std::vector<std::string> keys = {"linear_attn_qkv_prep"};
        if (useFp16) keys.emplace_back("MNN_METAL_FLOAT16_STORAGE");
        mQKVPrepPipeline = rt->findPipeline(keys);
        if (nil == mQKVPrepPipeline) {
            mQKVPrepPipeline = mtbn->makeComputePipelineWithSourceOption(gLinearAttnGatedDeltaRule, "linear_attn_qkv_prep", option);
            rt->insertPipeline(keys, mQKVPrepPipeline);
        }
    }
    // Gated delta rule pipelines
    mUseSimdGroupOpt = rt->supportSimdGroupReduce();
    if (mUseSimdGroupOpt) {
        // Compute SIMD_ITERS from actual head_k_dim and inject as compile-time macro
        int simdIters = (mHeadKDim + 31) / 32;
        NSString *simdItersStr = [NSString stringWithFormat:@"%d", simdIters];
        MTLCompileOptions *sgOption = [[MTLCompileOptions alloc] init];
        NSMutableDictionary *sgMacros = [NSMutableDictionary dictionary];
        if (useFp16) {
            sgMacros[@"MNN_METAL_FLOAT16_STORAGE"] = @"1";
        }
        sgMacros[@"SIMD_ITERS"] = simdItersStr;
        sgOption.preprocessorMacros = sgMacros;

        std::string simdItersKey = "SIMD_ITERS_" + std::to_string(simdIters);
        // Non-fused simdgroup version for prefill (reads pre-arranged Q/K/V)
        {
            std::vector<std::string> keys = {"linear_attn_gated_delta_rule_sg", simdItersKey};
            if (useFp16) keys.emplace_back("MNN_METAL_FLOAT16_STORAGE");
            mGatedDeltaRuleSGPipeline = rt->findPipeline(keys);
            if (nil == mGatedDeltaRuleSGPipeline) {
                mGatedDeltaRuleSGPipeline = mtbn->makeComputePipelineWithSourceOption(gLinearAttnGatedDeltaRuleSG, "linear_attn_gated_delta_rule_sg", sgOption);
                rt->insertPipeline(keys, mGatedDeltaRuleSGPipeline);
            }
        }
        // Fused simdgroup version for decode (reads conv_out directly, skips qkv_prep)
        {
            std::vector<std::string> keys = {"linear_attn_fused_sg", simdItersKey};
            if (useFp16) keys.emplace_back("MNN_METAL_FLOAT16_STORAGE");
            mGatedDeltaRuleFusedSGPipeline = rt->findPipeline(keys);
            if (nil == mGatedDeltaRuleFusedSGPipeline) {
                mGatedDeltaRuleFusedSGPipeline = mtbn->makeComputePipelineWithSourceOption(gLinearAttnFusedSG, "linear_attn_fused_sg", sgOption);
                rt->insertPipeline(keys, mGatedDeltaRuleFusedSGPipeline);
            }
        }
    }
    {
        // Scalar fallback
        std::vector<std::string> keys = {"linear_attn_gated_delta_rule"};
        if (useFp16) keys.emplace_back("MNN_METAL_FLOAT16_STORAGE");
        mGatedDeltaRulePipeline = rt->findPipeline(keys);
        if (nil == mGatedDeltaRulePipeline) {
            mGatedDeltaRulePipeline = mtbn->makeComputePipelineWithSourceOption(gLinearAttnGatedDeltaRule, "linear_attn_gated_delta_rule", option);
            rt->insertPipeline(keys, mGatedDeltaRulePipeline);
        }
    }
}

ErrorCode MetalLinearAttention::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto qkv = inputs[0];
    int batch = 0, convDim = 0, seqLen = 0;
    linearAttentionDims(qkv, batch, convDim, seqLen);
    int K_conv = inputs[3]->length(2);
    int convStateSize = K_conv - 1;
    int H = mNumVHeads;
    int dk = mHeadKDim;
    int dv = mHeadVDim;
    int convChannels = mAttentionType == "short_conv" ? mHeadVDim : convDim;
    bool needRecurrentState = mAttentionType != "short_conv";

    // ─── Persistent state buffers (STATIC): allocate once, shared via onClone ───
    auto mtbn = static_cast<MetalBackend *>(backend());
    int bytesPerElement = mtbn->useFp16InsteadFp32() ? 2 : 4;
    const bool needConvStateInit = mStateCache->mConvState.get() == nullptr;
    const bool needRecurrentStateInit = needRecurrentState && mStateCache->mRecurrentState.get() == nullptr;
    if (needConvStateInit || needRecurrentStateInit) {
        // First time: allocate and zero-initialize
        if (needConvStateInit) {
            int convStateTotal = ALIMAX(batch * convChannels * convStateSize, 1);
            mStateCache->mConvState.reset(Tensor::createDevice<float>({convStateTotal}));
            bool success = backend()->onAcquireBuffer(mStateCache->mConvState.get(), Backend::STATIC);
            if (!success) return OUT_OF_MEMORY;
            auto convDevice = (id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)mStateCache->mConvState->deviceId())->getBuffer();
            auto convPtr = (uint8_t*)convDevice.contents + TensorUtils::getDescribeOrigin(mStateCache->mConvState.get())->offset;
            ::memset(convPtr, 0, convStateTotal * bytesPerElement);
        }

        if (needRecurrentStateInit) {
            mStateCache->mRecurrentState.reset(Tensor::createDevice<float>({batch, H, dk, dv}));
            bool success = backend()->onAcquireBuffer(mStateCache->mRecurrentState.get(), Backend::STATIC);
            if (!success) return OUT_OF_MEMORY;
            auto rnnDevice = (id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)mStateCache->mRecurrentState->deviceId())->getBuffer();
            auto rnnPtr = (uint8_t*)rnnDevice.contents + TensorUtils::getDescribeOrigin(mStateCache->mRecurrentState.get())->offset;
            ::memset(rnnPtr, 0, batch * H * dk * dv * bytesPerElement);
        }
    } else if (seqLen > 1) {
        // Prefill: reset state for new sequence, UNLESS:
        // 1. Loading from prefix cache (PendingRead), or
        // 2. Reusing KV from previous inference (reuse_kv=true, i.e. previous != remove)
        bool loadingFromDisk = (mMeta != nullptr && mMeta->file_flag == KVMeta::PendingRead && mMeta->file_name.size() > 0);
        bool reusingKV = (mMeta != nullptr && mMeta->previous != mMeta->remove);
        if (!loadingFromDisk && !reusingKV) {
            if (mStateCache->mConvState.get() != nullptr) {
                auto convDevice = (id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)mStateCache->mConvState->deviceId())->getBuffer();
                auto convPtr = (uint8_t*)convDevice.contents + TensorUtils::getDescribeOrigin(mStateCache->mConvState.get())->offset;
                ::memset(convPtr, 0, mStateCache->mConvState->elementSize() * bytesPerElement);
            }
            if (mStateCache->mRecurrentState.get() != nullptr) {
                auto rnnDevice = (id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)mStateCache->mRecurrentState->deviceId())->getBuffer();
                auto rnnPtr = (uint8_t*)rnnDevice.contents + TensorUtils::getDescribeOrigin(mStateCache->mRecurrentState.get())->offset;
                ::memset(rnnPtr, 0, mStateCache->mRecurrentState->elementSize() * bytesPerElement);
            }
        }
    }
    // Decode (seqLen == 1): keep existing state untouched

    mConvOut.reset(Tensor::createDevice<float>({batch, convChannels, seqLen}));
    bool success = backend()->onAcquireBuffer(mConvOut.get(), Backend::DYNAMIC);

    // Fused decode path (simd + L=1) reads conv_out directly, no Q/K/V needed
    bool needQKV = mAttentionType != "short_conv" && !(mUseSimdGroupOpt && seqLen == 1);
    if (needQKV) {
        mQ.reset(Tensor::createDevice<float>({batch, seqLen, H, dk}));
        mK.reset(Tensor::createDevice<float>({batch, seqLen, H, dk}));
        mV.reset(Tensor::createDevice<float>({batch, seqLen, H, dv}));
        success = success && backend()->onAcquireBuffer(mQ.get(), Backend::DYNAMIC);
        success = success && backend()->onAcquireBuffer(mK.get(), Backend::DYNAMIC);
        success = success && backend()->onAcquireBuffer(mV.get(), Backend::DYNAMIC);
    }
    if (!success) return OUT_OF_MEMORY;

    if (needQKV) {
        backend()->onReleaseBuffer(mV.get(), Backend::DYNAMIC);
        backend()->onReleaseBuffer(mK.get(), Backend::DYNAMIC);
        backend()->onReleaseBuffer(mQ.get(), Backend::DYNAMIC);
    }
    backend()->onReleaseBuffer(mConvOut.get(), Backend::DYNAMIC);

    return NO_ERROR;
}

void MetalLinearAttention::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) {
    // onResize() may be skipped when shapes are unchanged. Ensure state is reset here too.
    int resetBatch = 0, resetConvDim = 0, resetSeqLen = 0;
    linearAttentionDims(inputs[0], resetBatch, resetConvDim, resetSeqLen);
    if (resetSeqLen > 1 && mMeta != nullptr && mMeta->previous == mMeta->remove) {
        bool loadingFromDisk = (mMeta->file_flag == KVMeta::PendingRead && mMeta->file_name.size() > 0);
        if (!loadingFromDisk) {
            auto mtbn = static_cast<MetalBackend *>(backend());
            int bytesPerElement = mtbn->useFp16InsteadFp32() ? 2 : 4;
            if (mStateCache->mConvState.get() != nullptr) {
                auto convDevice = (id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)mStateCache->mConvState->deviceId())->getBuffer();
                auto convPtr = (uint8_t*)convDevice.contents + TensorUtils::getDescribeOrigin(mStateCache->mConvState.get())->offset;
                ::memset(convPtr, 0, mStateCache->mConvState->elementSize() * bytesPerElement);
            }
            if (mStateCache->mRecurrentState.get() != nullptr) {
                auto rnnDevice = (id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)mStateCache->mRecurrentState->deviceId())->getBuffer();
                auto rnnPtr = (uint8_t*)rnnDevice.contents + TensorUtils::getDescribeOrigin(mStateCache->mRecurrentState.get())->offset;
                ::memset(rnnPtr, 0, mStateCache->mRecurrentState->elementSize() * bytesPerElement);
            }
        }
    }

    auto qkv = inputs[0];
    int batch = 0, convDim = 0, seqLen = 0;
    linearAttentionDims(qkv, batch, convDim, seqLen);
    int K_conv = inputs[3]->length(2);
    int convStateSize = K_conv - 1;
    int H = mNumVHeads;
    int dk = mHeadKDim;
    int dv = mHeadVDim;
    int key_dim = mNumKHeads * dk;
    int val_dim = mNumVHeads * dv;
    int gqa_factor = (mNumVHeads > mNumKHeads) ? (mNumVHeads / mNumKHeads) : 1;

    // Update param buffer
    auto paramPtr = (LinearAttnParam *)mParamBuffer.contents;
    paramPtr->batch = batch;
    paramPtr->conv_dim = convDim;
    paramPtr->seq_len = seqLen;
    paramPtr->kernel_size = K_conv;
    paramPtr->conv_state_size = convStateSize;
    paramPtr->num_k_heads = mNumKHeads;
    paramPtr->num_v_heads = mNumVHeads;
    paramPtr->head_k_dim = dk;
    paramPtr->head_v_dim = dv;
    paramPtr->key_dim = key_dim;
    paramPtr->val_dim = val_dim;
    paramPtr->gqa_factor = gqa_factor;
    paramPtr->use_l2norm = mUseQKL2Norm ? 1 : 0;
    paramPtr->qkv_c4 = TensorUtils::getDescribe(inputs[0])->dimensionFormat == MNN_DATA_FORMAT_NC4HW4 ? 1 : 0;
    paramPtr->gate_c4 = TensorUtils::getDescribe(inputs[1])->dimensionFormat == MNN_DATA_FORMAT_NC4HW4 ? 1 : 0;
    paramPtr->beta_c4 = TensorUtils::getDescribe(inputs[2])->dimensionFormat == MNN_DATA_FORMAT_NC4HW4 ? 1 : 0;
    paramPtr->output_c4 = TensorUtils::getDescribe(outputs[0])->dimensionFormat == MNN_DATA_FORMAT_NC4HW4 ? 1 : 0;
    paramPtr->q_scale = 1.0f / sqrtf((float)dk);

    if (mAttentionType == "short_conv") {
        int total = batch * mHeadVDim * seqLen;
        NSUInteger threadGroupSize = MIN((NSUInteger)256, mShortConvPipeline.maxTotalThreadsPerThreadgroup);
        threadGroupSize = MIN(threadGroupSize, (NSUInteger)total);

        [encoder setComputePipelineState:mShortConvPipeline];
        MetalBackend::setTensor(inputs[0], encoder, 0);
        MetalBackend::setTensor(mStateCache->mConvState.get(), encoder, 1);
        MetalBackend::setTensor(inputs[3], encoder, 2);
        MetalBackend::setTensor(mConvOut.get(), encoder, 3);
        [encoder setBuffer:mParamBuffer offset:0 atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake((total + threadGroupSize - 1) / threadGroupSize, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threadGroupSize, 1, 1)];

        if (convStateSize > 0) {
            int stateTotal = batch * mHeadVDim * convStateSize;
            NSUInteger stateThreadGroupSize =
                MIN((NSUInteger)256, mShortConvStateUpdatePipeline.maxTotalThreadsPerThreadgroup);
            stateThreadGroupSize = MIN(stateThreadGroupSize, (NSUInteger)stateTotal);
            [encoder setComputePipelineState:mShortConvStateUpdatePipeline];
            MetalBackend::setTensor(inputs[0], encoder, 0);
            MetalBackend::setTensor(mStateCache->mConvState.get(), encoder, 1);
            [encoder setBuffer:mParamBuffer offset:0 atIndex:2];
            [encoder dispatchThreadgroups:MTLSizeMake((stateTotal + stateThreadGroupSize - 1) / stateThreadGroupSize,
                                                       1, 1)
                    threadsPerThreadgroup:MTLSizeMake(stateThreadGroupSize, 1, 1)];
        }

        [encoder setComputePipelineState:mShortConvOutputPipeline];
        MetalBackend::setTensor(inputs[0], encoder, 0);
        MetalBackend::setTensor(mConvOut.get(), encoder, 1);
        MetalBackend::setTensor(outputs[0], encoder, 2);
        [encoder setBuffer:mParamBuffer offset:0 atIndex:3];
        NSUInteger outputThreadGroupSize =
            MIN((NSUInteger)256, mShortConvOutputPipeline.maxTotalThreadsPerThreadgroup);
        outputThreadGroupSize = MIN(outputThreadGroupSize, (NSUInteger)total);
        [encoder dispatchThreadgroups:MTLSizeMake((total + outputThreadGroupSize - 1) / outputThreadGroupSize, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(outputThreadGroupSize, 1, 1)];
        return;
    }

    // Kernel 1: Conv1D + SiLU
    {
        [encoder setComputePipelineState:mConvSiluPipeline];
        // [batch, convDim, seqLen]
        MetalBackend::setTensor(inputs[0], encoder, 0);                    // qkv
        // [batch, convDim, K_conv-1]
        MetalBackend::setTensor(mStateCache->mConvState.get(), encoder, 1); // conv_state
        // [convDim, 1, K_conv]
        MetalBackend::setTensor(inputs[3], encoder, 2);                    // conv_weight
        // [batch, convDim, seqLen]
        MetalBackend::setTensor(mConvOut.get(), encoder, 3);               // conv_out
        [encoder setBuffer:mParamBuffer offset:0 atIndex:4];     // param

        int totalConvSilu = batch * convDim * seqLen;
        NSUInteger threadGroupSize = MIN((NSUInteger)256, mConvSiluPipeline.maxTotalThreadsPerThreadgroup);
        threadGroupSize = MIN(threadGroupSize, (NSUInteger)totalConvSilu);
        [encoder dispatchThreadgroups:MTLSizeMake((totalConvSilu + threadGroupSize - 1) / threadGroupSize, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threadGroupSize, 1, 1)];
    }

    // Kernel 2: Conv state update
    if (convStateSize > 0) {
        [encoder setComputePipelineState:mConvStateUpdatePipeline];
        // [batch, convDim, seqLen]
        MetalBackend::setTensor(inputs[0], encoder, 0);                    // qkv
        // [batch, convDim, K_conv-1]
        MetalBackend::setTensor(mStateCache->mConvState.get(), encoder, 1); // conv_state
        [encoder setBuffer:mParamBuffer offset:0 atIndex:2];               // param

        int totalUpdate = batch * convDim * convStateSize;
        NSUInteger threadGroupSize = MIN((NSUInteger)256, mConvStateUpdatePipeline.maxTotalThreadsPerThreadgroup);
        threadGroupSize = MIN(threadGroupSize, (NSUInteger)totalUpdate);
        [encoder dispatchThreadgroups:MTLSizeMake((totalUpdate + threadGroupSize - 1) / threadGroupSize, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threadGroupSize, 1, 1)];
    }

    if (mUseSimdGroupOpt && seqLen == 1) {
        // Decode: Fused QKV-prep + Delta Rule (skip qkv_prep, read conv_out directly)
        // conv_out stride = L = 1, so reads are coalesced
        [encoder setComputePipelineState:mGatedDeltaRuleFusedSGPipeline];
        MetalBackend::setTensor(mConvOut.get(), encoder, 0);               // conv_out
        MetalBackend::setTensor(inputs[1], encoder, 1);                    // gate
        MetalBackend::setTensor(inputs[2], encoder, 2);                    // beta
        MetalBackend::setTensor(mStateCache->mRecurrentState.get(), encoder, 3);  // recurrent_state
        MetalBackend::setTensor(outputs[0], encoder, 4);                   // attn_out
        [encoder setBuffer:mParamBuffer offset:0 atIndex:5];               // param

        int totalSimdgroups = batch * H * dv;
        int simdgroupsPerTG = 4;
        NSUInteger threadGroupSize = simdgroupsPerTG * 32; // 128 threads
        int numThreadgroups = (totalSimdgroups + simdgroupsPerTG - 1) / simdgroupsPerTG;
        [encoder dispatchThreadgroups:MTLSizeMake(numThreadgroups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threadGroupSize, 1, 1)];
    } else {
        // Prefill or scalar fallback: QKV prep + separate Delta Rule
        // Kernel 3: QKV prep
        {
            [encoder setComputePipelineState:mQKVPrepPipeline];
            MetalBackend::setTensor(mConvOut.get(), encoder, 0);
            MetalBackend::setTensor(mQ.get(), encoder, 1);
            MetalBackend::setTensor(mK.get(), encoder, 2);
            MetalBackend::setTensor(mV.get(), encoder, 3);
            [encoder setBuffer:mParamBuffer offset:0 atIndex:4];

            int totalPrep = batch * seqLen * H;
            NSUInteger threadGroupSize = MIN((NSUInteger)256, mQKVPrepPipeline.maxTotalThreadsPerThreadgroup);
            threadGroupSize = MIN(threadGroupSize, (NSUInteger)totalPrep);
            [encoder dispatchThreadgroups:MTLSizeMake((totalPrep + threadGroupSize - 1) / threadGroupSize, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(threadGroupSize, 1, 1)];
        }
        // Kernel 4: Gated Delta Rule
        if (mUseSimdGroupOpt) {
            // Simdgroup-optimized (prefill path)
            [encoder setComputePipelineState:mGatedDeltaRuleSGPipeline];
            MetalBackend::setTensor(mQ.get(), encoder, 0);
            MetalBackend::setTensor(mK.get(), encoder, 1);
            MetalBackend::setTensor(mV.get(), encoder, 2);
            MetalBackend::setTensor(inputs[1], encoder, 3);
            MetalBackend::setTensor(inputs[2], encoder, 4);
            MetalBackend::setTensor(mStateCache->mRecurrentState.get(), encoder, 5);
            MetalBackend::setTensor(outputs[0], encoder, 6);
            [encoder setBuffer:mParamBuffer offset:0 atIndex:7];

            int totalSimdgroups = batch * H * dv;
            int simdgroupsPerTG = 4;
            NSUInteger threadGroupSize = simdgroupsPerTG * 32;
            int numThreadgroups = (totalSimdgroups + simdgroupsPerTG - 1) / simdgroupsPerTG;
            [encoder dispatchThreadgroups:MTLSizeMake(numThreadgroups, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(threadGroupSize, 1, 1)];
        } else {
            // Scalar fallback
            [encoder setComputePipelineState:mGatedDeltaRulePipeline];
            MetalBackend::setTensor(mQ.get(), encoder, 0);
            MetalBackend::setTensor(mK.get(), encoder, 1);
            MetalBackend::setTensor(mV.get(), encoder, 2);
            MetalBackend::setTensor(inputs[1], encoder, 3);
            MetalBackend::setTensor(inputs[2], encoder, 4);
            MetalBackend::setTensor(mStateCache->mRecurrentState.get(), encoder, 5);
            MetalBackend::setTensor(outputs[0], encoder, 6);
            [encoder setBuffer:mParamBuffer offset:0 atIndex:7];

            int total = batch * H * dv;
            NSUInteger threadGroupSize = MIN((NSUInteger)256, mGatedDeltaRulePipeline.maxTotalThreadsPerThreadgroup);
            threadGroupSize = MIN(threadGroupSize, (NSUInteger)total);
            [encoder dispatchThreadgroups:MTLSizeMake((total + threadGroupSize - 1) / threadGroupSize, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(threadGroupSize, 1, 1)];
        }
    }
}

bool MetalLinearAttention::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (nullptr == dst) {
        return true;
    }
    auto tmp = new MetalLinearAttention(bn, op);
    // Share persistent state buffers between prefill and decode Executions
    tmp->mStateCache = mStateCache;
    *dst = tmp;
    return true;
}

class MetalLinearAttentionCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op,
                                Backend *backend, const std::vector<Tensor *> &outputs) const {
        return new MetalLinearAttention(backend, op);
    }
};
REGISTER_METAL_OP_TRANSFORMER_CREATOR(MetalLinearAttentionCreator, OpType_LinearAttention);

} // namespace MNN

#endif /* MNN_SUPPORT_TRANSFORMER_FUSE */
#endif /* MNN_METAL_ENABLED */
