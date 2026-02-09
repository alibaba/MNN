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
    float q_scale;
};

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
    auto context = (__bridge MNNMetalContext *)mtbn->context();
    mParamBuffer = [context newDeviceBuffer:sizeof(LinearAttnParam) access:CPUWriteOnly];

    // Compile shader pipelines
    MTLCompileOptions *option = [[MTLCompileOptions alloc] init];
    auto rt = (MetalRuntime *)mtbn->runtime();
    bool useFp16 = mtbn->useFp16InsteadFp32();
    if (useFp16) {
        option.preprocessorMacros = @{@"MNN_METAL_FLOAT16_STORAGE" : @"1"};
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
    // Gated delta rule pipeline
    {
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
    int batch = qkv->length(0);
    int convDim = qkv->length(1);
    int seqLen = qkv->length(2);
    int K_conv = inputs[3]->length(2);
    int convStateSize = K_conv - 1;
    int H = mNumVHeads;
    int dk = mHeadKDim;
    int dv = mHeadVDim;

    // ─── Persistent state buffers (STATIC): allocate once, shared via onClone ───
    auto mtbn = static_cast<MetalBackend *>(backend());
    int bytesPerElement = mtbn->useFp16InsteadFp32() ? 2 : 4;
    if (mStateCache->mRecurrentState.get() == nullptr) {
        // First time: allocate and zero-initialize
        if (convStateSize > 0) {
            mStateCache->mConvState.reset(Tensor::createDevice<float>({batch, convDim, convStateSize}));
            bool success = backend()->onAcquireBuffer(mStateCache->mConvState.get(), Backend::STATIC);
            if (!success) return OUT_OF_MEMORY;
            auto convDevice = (id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)mStateCache->mConvState->deviceId())->getBuffer();
            auto convPtr = (uint8_t*)convDevice.contents + TensorUtils::getDescribe(mStateCache->mConvState.get())->extra.offset;
            ::memset(convPtr, 0, batch * convDim * convStateSize * bytesPerElement);
        }

        mStateCache->mRecurrentState.reset(Tensor::createDevice<float>({batch, H, dk, dv}));
        bool success = backend()->onAcquireBuffer(mStateCache->mRecurrentState.get(), Backend::STATIC);
        if (!success) return OUT_OF_MEMORY;
        auto rnnDevice = (id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)mStateCache->mRecurrentState->deviceId())->getBuffer();
        auto rnnPtr = (uint8_t*)rnnDevice.contents + TensorUtils::getDescribe(mStateCache->mRecurrentState.get())->extra.offset;
        ::memset(rnnPtr, 0, batch * H * dk * dv * bytesPerElement);
    } else if (seqLen > 1) {
        // Prefill (seqLen > 1): reset state for new sequence
        if (mStateCache->mConvState.get() != nullptr) {
            auto convDevice = (id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)mStateCache->mConvState->deviceId())->getBuffer();
            auto convPtr = (uint8_t*)convDevice.contents + TensorUtils::getDescribe(mStateCache->mConvState.get())->extra.offset;
            ::memset(convPtr, 0, mStateCache->mConvState->elementSize() * bytesPerElement);
        }
        auto rnnDevice = (id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)mStateCache->mRecurrentState->deviceId())->getBuffer();
        auto rnnPtr = (uint8_t*)rnnDevice.contents + TensorUtils::getDescribe(mStateCache->mRecurrentState.get())->extra.offset;
        ::memset(rnnPtr, 0, mStateCache->mRecurrentState->elementSize() * bytesPerElement);
    }
    // Decode (seqLen == 1): keep existing state untouched

    // Allocate temporary conv output
    mConvOut.reset(Tensor::createDevice<float>({batch, convDim, seqLen}));
    bool success = backend()->onAcquireBuffer(mConvOut.get(), Backend::DYNAMIC);
    if (!success) return OUT_OF_MEMORY;
    backend()->onReleaseBuffer(mConvOut.get(), Backend::DYNAMIC);

    return NO_ERROR;
}

void MetalLinearAttention::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) {
    auto qkv = inputs[0];
    int batch = qkv->length(0);
    int convDim = qkv->length(1);
    int seqLen = qkv->length(2);
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
    paramPtr->q_scale = 1.0f / sqrtf((float)dk);

    // Kernel 1: Conv1D + SiLU
    {
        [encoder setComputePipelineState:mConvSiluPipeline];
        MetalBackend::setTensor(inputs[0], encoder, 0);                    // qkv
        MetalBackend::setTensor(mStateCache->mConvState.get(), encoder, 1); // conv_state
        MetalBackend::setTensor(inputs[3], encoder, 2);                    // conv_weight
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
        MetalBackend::setTensor(inputs[0], encoder, 0);                    // qkv
        MetalBackend::setTensor(mStateCache->mConvState.get(), encoder, 1); // conv_state
        [encoder setBuffer:mParamBuffer offset:0 atIndex:2];               // param

        int totalUpdate = batch * convDim * convStateSize;
        NSUInteger threadGroupSize = MIN((NSUInteger)256, mConvStateUpdatePipeline.maxTotalThreadsPerThreadgroup);
        threadGroupSize = MIN(threadGroupSize, (NSUInteger)totalUpdate);
        [encoder dispatchThreadgroups:MTLSizeMake((totalUpdate + threadGroupSize - 1) / threadGroupSize, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threadGroupSize, 1, 1)];
    }

    // Kernel 3: Gated Delta Rule
    {
        [encoder setComputePipelineState:mGatedDeltaRulePipeline];
        MetalBackend::setTensor(mConvOut.get(), encoder, 0);         // conv_out
        MetalBackend::setTensor(inputs[1], encoder, 1);              // gate
        MetalBackend::setTensor(inputs[2], encoder, 2);              // beta
        MetalBackend::setTensor(mStateCache->mRecurrentState.get(), encoder, 3);  // recurrent_state
        MetalBackend::setTensor(outputs[0], encoder, 4);             // attn_out
        [encoder setBuffer:mParamBuffer offset:0 atIndex:5];         // param

        int totalHeads = batch * H;
        NSUInteger threadGroupSize = MIN((NSUInteger)256, mGatedDeltaRulePipeline.maxTotalThreadsPerThreadgroup);
        threadGroupSize = MIN(threadGroupSize, (NSUInteger)totalHeads);
        [encoder dispatchThreadgroups:MTLSizeMake((totalHeads + threadGroupSize - 1) / threadGroupSize, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threadGroupSize, 1, 1)];
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
