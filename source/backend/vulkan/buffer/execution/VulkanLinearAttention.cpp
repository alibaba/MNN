//
//  VulkanLinearAttention.cpp
//  MNN
//
//  Created by MNN on 2026/02/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include <cmath>
#include <cstring>
#include <vector>
#include "VulkanLinearAttention.hpp"
#include "backend/vulkan/vulkan/vulkan_wrapper.h"
#include "core/Macro.h"
#include "core/OpCommonUtils.hpp"
#include "core/TensorUtils.hpp"

namespace MNN {

namespace {

static bool _supportLinearAttentionSubgroup(const VulkanDevice& device) {
    const auto& subgroup = device.getSubgroupInfo();
    if (0 == subgroup.size) {
        return false;
    }
    if (0 == (subgroup.stages & VK_SHADER_STAGE_COMPUTE_BIT)) {
        return false;
    }
    const VkSubgroupFeatureFlags required = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT;
    return (subgroup.ops & required) == required;
}

struct LinearAttnConvSiluParams {
    ivec4 size0; // batch, conv_dim, seq_len, kernel_size
    ivec4 size1; // conv_state_size, total, qkv_c4, 0
};

struct LinearAttnConvStateUpdateParams {
    ivec4 size0; // batch, conv_dim, seq_len, conv_state_size
    ivec4 size1; // channel_count, qkv_c4, 0, 0
};

struct LinearAttnQKVPrepParams {
    ivec4 size0; // batch, conv_dim, seq_len, num_k_heads
    ivec4 size1; // num_v_heads, head_k_dim, head_v_dim, key_dim
    ivec4 size2; // val_dim, gqa_factor, use_l2norm, total
    vec4 size3;  // q_scale, 0, 0, 0
};

struct LinearAttnRecurrentParams {
    ivec4 size0; // batch, seq_len, num_v_heads, head_k_dim
    ivec4 size1; // head_v_dim, total_rows, layout_flags, 0
};

struct LinearAttnShortConvParams {
    ivec4 size0; // batch, conv_dim, seq_len, kernel_size
    ivec4 size1; // conv_state_size, hidden, qkv_c4, output_c4
};

static bool _isC4(const Tensor* tensor) {
    return TensorUtils::getDescribe(tensor)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4;
}

static void _linearAttentionDims(const Tensor* qkv, int& batch, int& convDim, int& seqLen) {
    if (_isC4(qkv)) {
        batch = 1;
        seqLen = qkv->length(0);
        convDim = qkv->length(1);
        return;
    }
    batch = qkv->length(0);
    convDim = qkv->length(1);
    seqLen = qkv->length(2);
}

} // namespace

VulkanLinearAttention::VulkanLinearAttention(const MNN::Op* op, Backend* backend)
    : VulkanBasicExecution(backend) {
    auto param = op->main_as_LinearAttentionParam();
    mAttentionType = param->attn_type()->str();
    mNumKHeads = param->num_k_heads();
    mNumVHeads = param->num_v_heads();
    mHeadKDim = param->head_k_dim();
    mHeadVDim = param->head_v_dim();
    mUseQKL2Norm = param->use_qk_l2norm();
    mStateCache.reset(new VulkanLinearAttentionState);

    auto vkBn = static_cast<VulkanBackend*>(backend);
    mUseFP16 = vkBn->useFP16();
    mSubgroupSize = vkBn->getDevice().getSubgroupSize();
    mUseSubgroup = _supportLinearAttentionSubgroup(vkBn->getDevice());
    mLaneCount = mSubgroupSize > 0 ? mSubgroupSize : 32;
    mMeta = reinterpret_cast<KVMeta*>(vkBn->getMetaPtr());

    auto shaderKey = [this](const char* base) {
        std::string key = base;
        if (mUseFP16) {
            key += "_FP16";
        }
        key += "_comp";
        return key;
    };

    if (mAttentionType == "short_conv") {
        std::vector<VkDescriptorType> convTypes{
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        };
        mShortConvPipeline = vkBn->getPipeline(shaderKey("glsl_linear_attn_short_conv"), convTypes);
        MNN_ASSERT(nullptr != mShortConvPipeline);
        mShortConvDesSet.reset(mShortConvPipeline->createSet());

        std::vector<VkDescriptorType> stateTypes{
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        };
        mShortConvStateUpdatePipeline =
            vkBn->getPipeline(shaderKey("glsl_linear_attn_short_conv_state_update"), stateTypes);
        MNN_ASSERT(nullptr != mShortConvStateUpdatePipeline);
        mShortConvStateUpdateDesSet.reset(mShortConvStateUpdatePipeline->createSet());

        std::vector<VkDescriptorType> outputTypes{
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        };
        mShortConvOutputPipeline = vkBn->getPipeline(shaderKey("glsl_linear_attn_short_conv_output"), outputTypes);
        MNN_ASSERT(nullptr != mShortConvOutputPipeline);
        mShortConvOutputDesSet.reset(mShortConvOutputPipeline->createSet());
        mShortConvParam = vkBn->allocUniform();
        return;
    }

    {
        std::vector<VkDescriptorType> types{
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        };
        mConvSiluPipeline = vkBn->getPipeline(shaderKey("glsl_linear_attn_conv_silu"), types);
        MNN_ASSERT(nullptr != mConvSiluPipeline);
        mConvSiluDesSet.reset(mConvSiluPipeline->createSet());
        mConvSiluParam = vkBn->allocUniform();
    }
    {
        std::vector<VkDescriptorType> types{
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        };
        mConvStateUpdatePipeline = vkBn->getPipeline(shaderKey("glsl_linear_attn_conv_state_update"), types);
        MNN_ASSERT(nullptr != mConvStateUpdatePipeline);
        mConvStateUpdateDesSet.reset(mConvStateUpdatePipeline->createSet());
        mConvStateUpdateParam = vkBn->allocUniform();
    }
    {
        std::vector<VkDescriptorType> types{
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        };
        mQKVPrepPipeline = vkBn->getPipeline(shaderKey("glsl_linear_attn_qkv_prep"), types);
        MNN_ASSERT(nullptr != mQKVPrepPipeline);
        mQKVPrepDesSet.reset(mQKVPrepPipeline->createSet());
        mQKVPrepParam = vkBn->allocUniform();
    }
    {
        std::vector<VkDescriptorType> types{
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        };
        const char* prefillBase = mUseSubgroup ? "glsl_linear_attn_gated_delta_rule_prefill"
                                               : "glsl_linear_attn_gated_delta_rule_prefill_nosubgroup";
        const char* decodeBase  = mUseSubgroup ? "glsl_linear_attn_gated_delta_rule_decode"
                                               : "glsl_linear_attn_gated_delta_rule_decode_nosubgroup";
        mPrefillPipeline = vkBn->getPipeline(shaderKey(prefillBase), types,
                                             {mLaneCount, mSubgroupsPerWorkgroup, 1});
        mDecodePipeline = vkBn->getPipeline(shaderKey(decodeBase), types,
                                            {mLaneCount, mSubgroupsPerWorkgroup, 1});
#ifdef MNN_VULKAN_LINEAR_ATTN_VERBOSE
        MNN_PRINT("[VulkanLinearAttention] path=%s, laneCount=%u, rowsPerGroup=%u\n",
                  mUseSubgroup ? "subgroup" : "shared_memory", mLaneCount, mSubgroupsPerWorkgroup);
#endif
        MNN_ASSERT(nullptr != mPrefillPipeline);
        MNN_ASSERT(nullptr != mDecodePipeline);
        mPrefillDesSet.reset(mPrefillPipeline->createSet());
        mDecodeDesSet.reset(mDecodePipeline->createSet());
        mPrefillParam = vkBn->allocUniform();
        mDecodeParam = vkBn->allocUniform();
    }
}

VulkanLinearAttention::~VulkanLinearAttention() {
    auto vkBn = static_cast<VulkanBackend*>(backend());
    auto recycle = [vkBn](std::shared_ptr<VulkanBuffer>& buffer) {
        if (buffer)
            vkBn->recycleUniform(buffer);
    };
    recycle(mConvSiluParam);
    recycle(mConvStateUpdateParam);
    recycle(mQKVPrepParam);
    recycle(mPrefillParam);
    recycle(mDecodeParam);
    recycle(mShortConvParam);
}

ErrorCode VulkanLinearAttention::ensurePersistentState(VulkanBackend* vkBn, int batch, int convDim, int convStateSize) {
    const int recurrentSize = batch * mNumVHeads * mHeadVDim * mHeadKDim;
    const int convSize = batch * convDim * convStateSize;
    const bool needRecurrentState = mAttentionType != "short_conv";
    const bool needRealloc = (needRecurrentState && nullptr == mStateCache->mRecurrentState.get()) ||
                             mStateCache->mBatch != batch || mStateCache->mConvDim != convDim ||
                             mStateCache->mConvStateSize != convStateSize || mStateCache->mNumVHeads != mNumVHeads ||
                             mStateCache->mHeadKDim != mHeadKDim || mStateCache->mHeadVDim != mHeadVDim;
    if (!needRealloc) {
        return NO_ERROR;
    }

    mStateCache->mConvState.reset();
    mStateCache->mRecurrentState.reset();

    if (convStateSize > 0) {
        mStateCache->mConvState.reset(Tensor::createDevice<float>({convSize}));
        if (!backend()->onAcquireBuffer(mStateCache->mConvState.get(), Backend::STATIC)) {
            return OUT_OF_MEMORY;
        }
    }

    if (needRecurrentState) {
        mStateCache->mRecurrentState.reset(Tensor::createDevice<float>({recurrentSize}));
        if (!backend()->onAcquireBuffer(mStateCache->mRecurrentState.get(), Backend::STATIC)) {
            return OUT_OF_MEMORY;
        }
    }

    mStateCache->mBatch = batch;
    mStateCache->mConvDim = convDim;
    mStateCache->mConvStateSize = convStateSize;
    mStateCache->mNumVHeads = mNumVHeads;
    mStateCache->mHeadKDim = mHeadKDim;
    mStateCache->mHeadVDim = mHeadVDim;
    return resetPersistentState(vkBn);
}

ErrorCode VulkanLinearAttention::resetPersistentState(VulkanBackend* vkBn) {
    if (mStateCache->mConvState.get() != nullptr) {
        std::vector<uint8_t> zeros(vkBn->getTensorSize(mStateCache->mConvState.get()), 0);
        auto buf = vkBn->getBuffer(mStateCache->mConvState.get());
        vkBn->copyToGPUBuffer(zeros.data(), std::get<0>(buf), zeros.size(), std::get<2>(buf));
    }
    if (mStateCache->mRecurrentState.get() != nullptr) {
        std::vector<uint8_t> zeros(vkBn->getTensorSize(mStateCache->mRecurrentState.get()), 0);
        auto buf = vkBn->getBuffer(mStateCache->mRecurrentState.get());
        vkBn->copyToGPUBuffer(zeros.data(), std::get<0>(buf), zeros.size(), std::get<2>(buf));
    }
    return NO_ERROR;
}

ErrorCode VulkanLinearAttention::onBeforeExecute(const std::vector<Tensor*>& inputs,
                                                 const std::vector<Tensor*>& outputs) {
    int batch = 0, convDim = 0, seqLen = 0;
    _linearAttentionDims(inputs[0], batch, convDim, seqLen);
    if (seqLen > 1 && mMeta != nullptr && mMeta->previous == mMeta->remove) {
        const bool loadingFromDisk = mMeta->file_flag == KVMeta::PendingRead && !mMeta->file_name.empty();
        if (!loadingFromDisk) {
            auto code = resetPersistentState(static_cast<VulkanBackend*>(backend()));
            if (code != NO_ERROR) {
                return code;
            }
        }
    }
    if (mMeta != nullptr && !mMeta->file_name.empty() && mMeta->layer_nums > 0 &&
        (mMeta->file_flag == KVMeta::PendingWrite || mMeta->file_flag == KVMeta::PendingRead) &&
        mMeta->previous == mMeta->remove) {
        // Keep the shared prefix index aligned with full-attention layers in hybrid models.
        mMeta->layer_index = (mMeta->layer_index + 1) % mMeta->layer_nums;
    }
    return NO_ERROR;
}

ErrorCode VulkanLinearAttention::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                          const VulkanCommandPool::Buffer* cmdBuffer) {
    auto vkBn = static_cast<VulkanBackend*>(backend());
    auto cmd = cmdBuffer->get();

    MNN_ASSERT(inputs.size() >= 4);
    MNN_ASSERT(outputs.size() >= 1);

    auto qkv = inputs[0];
    int batch = 0, convDim = 0, seqLen = 0;
    _linearAttentionDims(qkv, batch, convDim, seqLen);
    const int kernelSize = inputs[3]->length(2);
    const int convStateSize = kernelSize - 1;
    const int keyDim = mNumKHeads * mHeadKDim;
    const int valDim = mNumVHeads * mHeadVDim;
    const int gqaFactor = (mNumVHeads > mNumKHeads) ? (mNumVHeads / mNumKHeads) : 1;
    const float qScale = 1.0f / ::sqrtf((float)mHeadKDim);
    const bool shortConv = mAttentionType == "short_conv";
    const bool gatedDelta = mAttentionType == "gated_delta_rule";
    const bool qkvC4 = _isC4(qkv);
    const bool gateC4 = _isC4(inputs[1]);
    const bool betaC4 = _isC4(inputs[2]);
    const bool outputC4 = _isC4(outputs[0]);
    const int convChannels = shortConv ? convDim / 3 : convDim;

    if ((!shortConv && !gatedDelta) || batch <= 0 || convDim <= 0 || seqLen <= 0 || kernelSize <= 0 ||
        mNumKHeads <= 0 || mNumVHeads <= 0 || mHeadKDim <= 0 || mHeadVDim <= 0 ||
        (shortConv && (mNumKHeads != 1 || mNumVHeads != 1 || convDim % 3 != 0 || convDim / 3 != mHeadVDim))) {
        MNN_ERROR("Vulkan LinearAttention: invalid type, shape, or head configuration.\n");
        return INVALID_VALUE;
    }
    if (qkvC4 && (qkv->dimensions() != 4 || qkv->length(2) != 1 || qkv->length(3) != 1)) {
        MNN_ERROR("Vulkan LinearAttention: invalid C4 qkv layout.\n");
        return INVALID_VALUE;
    }

    auto code = ensurePersistentState(vkBn, batch, convChannels, convStateSize);
    if (NO_ERROR != code) {
        return code;
    }
    const bool reusingKV = (nullptr != mMeta && mMeta->previous != mMeta->remove);
    const bool loadingFromDisk = (mMeta != nullptr && mMeta->file_flag == KVMeta::PendingRead && mMeta->file_name.size() > 0);
    if (seqLen > 1 && !reusingKV && !loadingFromDisk) {
        code = resetPersistentState(vkBn);
        if (NO_ERROR != code) {
            return code;
        }
    }

    const int convOutSize = batch * convChannels * seqLen;
    const int qSize = batch * seqLen * mNumVHeads * mHeadKDim;
    const int vSize = batch * seqLen * mNumVHeads * mHeadVDim;
    mConvOut.reset(Tensor::createDevice<float>({convOutSize}));
    bool success = backend()->onAcquireBuffer(mConvOut.get(), Backend::DYNAMIC);
    if (gatedDelta) {
        mQ.reset(Tensor::createDevice<float>({qSize}));
        mK.reset(Tensor::createDevice<float>({qSize}));
        mV.reset(Tensor::createDevice<float>({vSize}));
        success = success && backend()->onAcquireBuffer(mQ.get(), Backend::DYNAMIC);
        success = success && backend()->onAcquireBuffer(mK.get(), Backend::DYNAMIC);
        success = success && backend()->onAcquireBuffer(mV.get(), Backend::DYNAMIC);
    }
    if (!success) {
        return OUT_OF_MEMORY;
    }
    if (gatedDelta) {
        backend()->onReleaseBuffer(mV.get(), Backend::DYNAMIC);
        backend()->onReleaseBuffer(mK.get(), Backend::DYNAMIC);
        backend()->onReleaseBuffer(mQ.get(), Backend::DYNAMIC);
    }
    backend()->onReleaseBuffer(mConvOut.get(), Backend::DYNAMIC);

#ifdef ENABLE_VULKAN_TIME_PROFILE
    auto dispatchWithProfile = [&](const char* name, const VulkanPipeline* pipeline,
                                   const std::shared_ptr<VulkanLayout::DescriptorSet>& set, uint32_t x, uint32_t y,
                                   uint32_t z) {
        auto* profiler = vkBn->timeProfiler();
        if (nullptr != profiler) {
            VulkanTimeProfileScope scope(profiler, cmd, name, VulkanTimeProfiler::Kind::Shader);
            pipeline->bind(cmd, set->get());
            vkCmdDispatch(cmd, x, y, z);
            return;
        }
        pipeline->bind(cmd, set->get());
        vkCmdDispatch(cmd, x, y, z);
    };
#else
    auto dispatchWithProfile = [&](const char*, const VulkanPipeline* pipeline,
                                   const std::shared_ptr<VulkanLayout::DescriptorSet>& set, uint32_t x, uint32_t y,
                                   uint32_t z) {
        pipeline->bind(cmd, set->get());
        vkCmdDispatch(cmd, x, y, z);
    };
#endif

    if (shortConv) {
        LinearAttnShortConvParams params;
        params.size0[0] = batch;
        params.size0[1] = convDim;
        params.size0[2] = seqLen;
        params.size0[3] = kernelSize;
        params.size1[0] = convStateSize;
        params.size1[1] = mHeadVDim;
        params.size1[2] = qkvC4 ? 1 : 0;
        params.size1[3] = outputC4 ? 1 : 0;
        ::memcpy(mShortConvParam->map(), &params, sizeof(params));
        mShortConvParam->unmap();

        mShortConvDesSet->writeBuffer(vkBn->getBuffer(inputs[0]), 0);
        if (mStateCache->mConvState.get() != nullptr) {
            mShortConvDesSet->writeBuffer(vkBn->getBuffer(mStateCache->mConvState.get()), 1);
        } else {
            mShortConvDesSet->writeBuffer(vkBn->getBuffer(mConvOut.get()), 1);
        }
        mShortConvDesSet->writeBuffer(vkBn->getBuffer(inputs[3]), 2);
        mShortConvDesSet->writeBuffer(vkBn->getBuffer(mConvOut.get()), 3);
        mShortConvDesSet->writeBuffer(mShortConvParam->buffer(), 4, mShortConvParam->size());
        dispatchWithProfile("linear_attn_short_conv", mShortConvPipeline, mShortConvDesSet,
                            UP_DIV((uint32_t)(batch * mHeadVDim * seqLen), 256), 1, 1);
        cmdBuffer->barrierSource(vkBn->getBuffer(mConvOut.get()));

        if (convStateSize > 0) {
            mShortConvStateUpdateDesSet->writeBuffer(vkBn->getBuffer(inputs[0]), 0);
            mShortConvStateUpdateDesSet->writeBuffer(vkBn->getBuffer(mStateCache->mConvState.get()), 1);
            mShortConvStateUpdateDesSet->writeBuffer(mShortConvParam->buffer(), 2, mShortConvParam->size());
            dispatchWithProfile("linear_attn_short_conv_state_update", mShortConvStateUpdatePipeline,
                                mShortConvStateUpdateDesSet, UP_DIV((uint32_t)(batch * mHeadVDim), 64), 1, 1);
            cmdBuffer->barrierSource(vkBn->getBuffer(mStateCache->mConvState.get()));
        }

        mShortConvOutputDesSet->writeBuffer(vkBn->getBuffer(inputs[0]), 0);
        mShortConvOutputDesSet->writeBuffer(vkBn->getBuffer(mConvOut.get()), 1);
        mShortConvOutputDesSet->writeBuffer(vkBn->getBuffer(outputs[0]), 2);
        mShortConvOutputDesSet->writeBuffer(mShortConvParam->buffer(), 3, mShortConvParam->size());
        dispatchWithProfile("linear_attn_short_conv_output", mShortConvOutputPipeline, mShortConvOutputDesSet,
                            UP_DIV((uint32_t)(batch * mHeadVDim * seqLen), 256), 1, 1);
        return NO_ERROR;
    }

    {
        LinearAttnConvSiluParams params;
        params.size0[0] = batch;
        params.size0[1] = convDim;
        params.size0[2] = seqLen;
        params.size0[3] = kernelSize;
        params.size1[0] = convStateSize;
        params.size1[1] = batch * convDim * seqLen;
        params.size1[2] = qkvC4 ? 1 : 0;
        params.size1[3] = 0;
        ::memcpy(mConvSiluParam->map(), &params, sizeof(params));
        mConvSiluParam->unmap();

        mConvSiluDesSet->writeBuffer(vkBn->getBuffer(inputs[0]), 0);
        if (mStateCache->mConvState.get() != nullptr) {
            mConvSiluDesSet->writeBuffer(vkBn->getBuffer(mStateCache->mConvState.get()), 1);
        } else {
            mConvSiluDesSet->writeBuffer(vkBn->getBuffer(mConvOut.get()), 1);
        }
        mConvSiluDesSet->writeBuffer(vkBn->getBuffer(inputs[3]), 2);
        mConvSiluDesSet->writeBuffer(vkBn->getBuffer(mConvOut.get()), 3);
        mConvSiluDesSet->writeBuffer(mConvSiluParam->buffer(), 4, mConvSiluParam->size());

        dispatchWithProfile("linear_attn_conv_silu", mConvSiluPipeline, mConvSiluDesSet,
                            UP_DIV((uint32_t)(batch * convDim * seqLen), 256), 1, 1);
        cmdBuffer->barrierSource(vkBn->getBuffer(mConvOut.get()));
    }

    if (convStateSize > 0) {
        LinearAttnConvStateUpdateParams params;
        params.size0[0] = batch;
        params.size0[1] = convDim;
        params.size0[2] = seqLen;
        params.size0[3] = convStateSize;
        params.size1[0] = batch * convDim;
        params.size1[1] = qkvC4 ? 1 : 0;
        params.size1[2] = 0;
        params.size1[3] = 0;
        ::memcpy(mConvStateUpdateParam->map(), &params, sizeof(params));
        mConvStateUpdateParam->unmap();

        mConvStateUpdateDesSet->writeBuffer(vkBn->getBuffer(inputs[0]), 0);
        mConvStateUpdateDesSet->writeBuffer(vkBn->getBuffer(mStateCache->mConvState.get()), 1);
        mConvStateUpdateDesSet->writeBuffer(mConvStateUpdateParam->buffer(), 2, mConvStateUpdateParam->size());

        dispatchWithProfile("linear_attn_conv_state_update", mConvStateUpdatePipeline, mConvStateUpdateDesSet,
                            UP_DIV((uint32_t)(batch * convDim), 256), 1, 1);
        cmdBuffer->barrierSource(vkBn->getBuffer(mStateCache->mConvState.get()));
    }

    {
        LinearAttnQKVPrepParams params;
        params.size0[0] = batch;
        params.size0[1] = convDim;
        params.size0[2] = seqLen;
        params.size0[3] = mNumKHeads;
        params.size1[0] = mNumVHeads;
        params.size1[1] = mHeadKDim;
        params.size1[2] = mHeadVDim;
        params.size1[3] = keyDim;
        params.size2[0] = valDim;
        params.size2[1] = gqaFactor;
        params.size2[2] = mUseQKL2Norm ? 1 : 0;
        params.size2[3] = batch * seqLen * mNumVHeads;
        params.size3[0] = qScale;
        params.size3[1] = 0.0f;
        params.size3[2] = 0.0f;
        params.size3[3] = 0.0f;
        ::memcpy(mQKVPrepParam->map(), &params, sizeof(params));
        mQKVPrepParam->unmap();

        mQKVPrepDesSet->writeBuffer(vkBn->getBuffer(mConvOut.get()), 0);
        mQKVPrepDesSet->writeBuffer(vkBn->getBuffer(mQ.get()), 1);
        mQKVPrepDesSet->writeBuffer(vkBn->getBuffer(mK.get()), 2);
        mQKVPrepDesSet->writeBuffer(vkBn->getBuffer(mV.get()), 3);
        mQKVPrepDesSet->writeBuffer(mQKVPrepParam->buffer(), 4, mQKVPrepParam->size());

        dispatchWithProfile("linear_attn_qkv_prep", mQKVPrepPipeline, mQKVPrepDesSet,
                            UP_DIV((uint32_t)(batch * seqLen * mNumVHeads), 256), 1, 1);
        cmdBuffer->barrierSource(vkBn->getBuffer(mQ.get()));
        cmdBuffer->barrierSource(vkBn->getBuffer(mK.get()));
        cmdBuffer->barrierSource(vkBn->getBuffer(mV.get()));
    }

    {
        LinearAttnRecurrentParams params;
        params.size0[0] = batch;
        params.size0[1] = seqLen;
        params.size0[2] = mNumVHeads;
        params.size0[3] = mHeadKDim;
        params.size1[0] = mHeadVDim;
        params.size1[1] = batch * mNumVHeads * mHeadVDim;
        params.size1[2] = (gateC4 ? 1 : 0) | (betaC4 ? 2 : 0) | (outputC4 ? 4 : 0);
        params.size1[3] = 0;

        auto recurrentParam = seqLen == 1 ? mDecodeParam : mPrefillParam;
        ::memcpy(recurrentParam->map(), &params, sizeof(params));
        recurrentParam->unmap();

        auto recurrentSet = seqLen == 1 ? mDecodeDesSet : mPrefillDesSet;
        recurrentSet->writeBuffer(vkBn->getBuffer(mQ.get()), 0);
        recurrentSet->writeBuffer(vkBn->getBuffer(mK.get()), 1);
        recurrentSet->writeBuffer(vkBn->getBuffer(mV.get()), 2);
        recurrentSet->writeBuffer(vkBn->getBuffer(inputs[1]), 3);
        recurrentSet->writeBuffer(vkBn->getBuffer(inputs[2]), 4);
        recurrentSet->writeBuffer(vkBn->getBuffer(mStateCache->mRecurrentState.get()), 5);
        recurrentSet->writeBuffer(vkBn->getBuffer(outputs[0]), 6);
        recurrentSet->writeBuffer(recurrentParam->buffer(), 7, recurrentParam->size());

        auto recurrentPipeline = seqLen == 1 ? mDecodePipeline : mPrefillPipeline;
        const uint32_t groupsX = UP_DIV((uint32_t)(batch * mNumVHeads * mHeadVDim), mSubgroupsPerWorkgroup);
        dispatchWithProfile(seqLen == 1 ? "linear_attn_gated_delta_rule_decode" : "linear_attn_gated_delta_rule_prefill",
                            recurrentPipeline, recurrentSet, groupsX, 1, 1);
        cmdBuffer->barrierSource(vkBn->getBuffer(mStateCache->mRecurrentState.get()));
    }

    return NO_ERROR;
}

bool VulkanLinearAttention::onClone(Backend* bn, const Op* op, VulkanBasicExecution** dst) {
    if (nullptr == dst) {
        return true;
    }
    auto res = new VulkanLinearAttention(op, bn);
    res->mStateCache = mStateCache;
    *dst = res;
    return true;
}

class VulkanLinearAttentionCreator : public VulkanBackend::Creator {
public:
    virtual VulkanBasicExecution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                           const MNN::Op* op, Backend* backend) const override {
        auto param = op->main_as_LinearAttentionParam();
        if (nullptr == param || nullptr == param->attn_type()) {
            return nullptr;
        }
        const auto type = param->attn_type()->str();
        if (type != "gated_delta_rule" && type != "short_conv") {
            return nullptr;
        }
        return new VulkanLinearAttention(op, backend);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_LinearAttention, new VulkanLinearAttentionCreator);
    return true;
}();

} // namespace MNN

#endif /* MNN_SUPPORT_TRANSFORMER_FUSE */
