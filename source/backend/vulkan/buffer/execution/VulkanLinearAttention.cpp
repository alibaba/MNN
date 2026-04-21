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
    ivec4 size1; // conv_state_size, total, 0, 0
};

struct LinearAttnConvStateUpdateParams {
    ivec4 size0; // batch, conv_dim, seq_len, conv_state_size
    ivec4 size1; // total, 0, 0, 0
};

struct LinearAttnQKVPrepParams {
    ivec4 size0; // batch, conv_dim, seq_len, num_k_heads
    ivec4 size1; // num_v_heads, head_k_dim, head_v_dim, key_dim
    ivec4 size2; // val_dim, gqa_factor, use_l2norm, total
    vec4 size3;  // q_scale, 0, 0, 0
};

struct LinearAttnRecurrentParams {
    ivec4 size0; // batch, seq_len, num_v_heads, head_k_dim
    ivec4 size1; // head_v_dim, total_rows, 0, 0
};

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
    vkBn->recycleUniform(mConvSiluParam);
    vkBn->recycleUniform(mConvStateUpdateParam);
    vkBn->recycleUniform(mQKVPrepParam);
    vkBn->recycleUniform(mPrefillParam);
    vkBn->recycleUniform(mDecodeParam);
}

ErrorCode VulkanLinearAttention::ensurePersistentState(VulkanBackend* vkBn, int batch, int convDim, int convStateSize) {
    const int recurrentSize = batch * mNumVHeads * mHeadVDim * mHeadKDim;
    const int convSize = batch * convDim * convStateSize;
    const bool needRealloc = nullptr == mStateCache->mRecurrentState.get() || mStateCache->mBatch != batch ||
                             mStateCache->mConvDim != convDim || mStateCache->mConvStateSize != convStateSize ||
                             mStateCache->mNumVHeads != mNumVHeads || mStateCache->mHeadKDim != mHeadKDim ||
                             mStateCache->mHeadVDim != mHeadVDim;
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

    mStateCache->mRecurrentState.reset(Tensor::createDevice<float>({recurrentSize}));
    if (!backend()->onAcquireBuffer(mStateCache->mRecurrentState.get(), Backend::STATIC)) {
        return OUT_OF_MEMORY;
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

ErrorCode VulkanLinearAttention::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                          const VulkanCommandPool::Buffer* cmdBuffer) {
    auto vkBn = static_cast<VulkanBackend*>(backend());
    auto cmd = cmdBuffer->get();

    MNN_ASSERT(inputs.size() >= 4);
    MNN_ASSERT(outputs.size() >= 1);

    auto qkv = inputs[0];
    const int batch = qkv->length(0);
    const int convDim = qkv->length(1);
    const int seqLen = qkv->length(2);
    const int kernelSize = inputs[3]->length(2);
    const int convStateSize = kernelSize - 1;
    const int keyDim = mNumKHeads * mHeadKDim;
    const int valDim = mNumVHeads * mHeadVDim;
    const int gqaFactor = (mNumVHeads > mNumKHeads) ? (mNumVHeads / mNumKHeads) : 1;
    const float qScale = 1.0f / ::sqrtf((float)mHeadKDim);

    auto code = ensurePersistentState(vkBn, batch, convDim, convStateSize);
    if (NO_ERROR != code) {
        return code;
    }
    const bool reusingKV = (nullptr != mMeta && mMeta->previous != mMeta->remove);
    if (seqLen > 1 && !reusingKV) {
        code = resetPersistentState(vkBn);
        if (NO_ERROR != code) {
            return code;
        }
    }

    const int convOutSize = batch * convDim * seqLen;
    const int qSize = batch * seqLen * mNumVHeads * mHeadKDim;
    const int vSize = batch * seqLen * mNumVHeads * mHeadVDim;
    mConvOut.reset(Tensor::createDevice<float>({convOutSize}));
    mQ.reset(Tensor::createDevice<float>({qSize}));
    mK.reset(Tensor::createDevice<float>({qSize}));
    mV.reset(Tensor::createDevice<float>({vSize}));
    bool success = backend()->onAcquireBuffer(mConvOut.get(), Backend::DYNAMIC);
    success = success && backend()->onAcquireBuffer(mQ.get(), Backend::DYNAMIC);
    success = success && backend()->onAcquireBuffer(mK.get(), Backend::DYNAMIC);
    success = success && backend()->onAcquireBuffer(mV.get(), Backend::DYNAMIC);
    if (!success) {
        return OUT_OF_MEMORY;
    }
    backend()->onReleaseBuffer(mV.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mK.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mQ.get(), Backend::DYNAMIC);
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

    {
        LinearAttnConvSiluParams params;
        params.size0[0] = batch;
        params.size0[1] = convDim;
        params.size0[2] = seqLen;
        params.size0[3] = kernelSize;
        params.size1[0] = convStateSize;
        params.size1[1] = batch * convDim * seqLen;
        params.size1[2] = 0;
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
        params.size1[0] = batch * convDim * convStateSize;
        params.size1[1] = 0;
        params.size1[2] = 0;
        params.size1[3] = 0;
        ::memcpy(mConvStateUpdateParam->map(), &params, sizeof(params));
        mConvStateUpdateParam->unmap();

        mConvStateUpdateDesSet->writeBuffer(vkBn->getBuffer(inputs[0]), 0);
        mConvStateUpdateDesSet->writeBuffer(vkBn->getBuffer(mStateCache->mConvState.get()), 1);
        mConvStateUpdateDesSet->writeBuffer(mConvStateUpdateParam->buffer(), 2, mConvStateUpdateParam->size());

        dispatchWithProfile("linear_attn_conv_state_update", mConvStateUpdatePipeline, mConvStateUpdateDesSet,
                            UP_DIV((uint32_t)(batch * convDim * convStateSize), 256), 1, 1);
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
        params.size1[2] = 0;
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
        if (nullptr == param || nullptr == param->attn_type() || param->attn_type()->str() != "gated_delta_rule") {
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