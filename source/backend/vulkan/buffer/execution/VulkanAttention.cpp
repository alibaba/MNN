#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include "VulkanAttention.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "backend/vulkan/vulkan/vulkan_wrapper.h"
#include <climits>

namespace MNN {

static inline float _invSqrt(float x) {
    return 1.0f / ::sqrtf(x);
}

static uint32_t _selectSoftmaxLocalSize(int totalLen, uint32_t maxSizeX, uint32_t maxInvocations) {
    if (totalLen <= 1) {
        return 1;
    }
    uint32_t cap = 128;
    cap = ALIMIN(cap, maxSizeX);
    cap = ALIMIN(cap, maxInvocations);
    cap = ALIMIN(cap, (uint32_t)totalLen);
    uint32_t localSize = 1;
    while ((localSize << 1) <= cap) {
        localSize <<= 1;
    }
    return localSize;
}

static constexpr int kAttentionPrefillKBlock = 512;
static constexpr int kAttentionDecodeTwoStageMinPastLen = 640;
static constexpr int kAttentionDecodeTwoStageMaxLen = 2048;
static constexpr int kAttentionDecodeTwoStageSoftmaxStride = 2048;
static constexpr int kAttentionDecodeTwoStageQkLocalCap = 256;
static constexpr int kAttentionDecodeTwoStageQkvD4Pack = 2;
static constexpr int kAttentionDecodeIndirectFusedSlot = 0;
static constexpr int kAttentionDecodeIndirectTwoStageQkSlot = 1;
static constexpr int kAttentionDecodeIndirectTwoStageQkvSlot = 2;
static constexpr int kAttentionDecodeTwoStageIndirectCmdCount = 3;

static VkDeviceSize _dispatchIndirectOffset(int slot) {
    return (VkDeviceSize)slot * (VkDeviceSize)sizeof(VkDispatchIndirectCommand);
}

static bool _supportDecodeSubgroup(const VulkanDevice& device) {
    const auto& subgroup = device.getSubgroupInfo();
    if (0 == subgroup.size) {
        return false;
    }
    if (0 == (subgroup.stages & VK_SHADER_STAGE_COMPUTE_BIT)) {
        return false;
    }
    const VkSubgroupFeatureFlags required = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT;
    if ((subgroup.ops & required) != required) {
        return false;
    }
    return true;
}

static int _decodeHeadDimAlignedIndex(int headDim) {
    if (headDim < 64 || headDim > 256 || 0 != (headDim & 63)) {
        return -1;
    }
    return headDim / 64 - 1;
}

static std::string _decodeSubgroupFusedShaderName(const char* mode, bool fp16, int index) {
    static const char* kHeadDimMacros[4] = {"", "D128", "D192", "D256"};
    std::string name = "glsl_attention_decode_";
    name += mode;
    name += "_subgroup_fused_";
    if (index > 0) {
        name += kHeadDimMacros[index];
        name += "_";
    }
    if (fp16) {
        name += "FP16_";
    }
    name += "comp";
    return name;
}

static std::string _decodeSingleFusedShaderName(bool fp16, int index) {
    return _decodeSubgroupFusedShaderName("single", fp16, index);
}

static std::string _decodeSmallFusedShaderName(bool fp16, int index) {
    return _decodeSubgroupFusedShaderName("small", fp16, index);
}

static std::string _decodeTwoStageShaderName(const char* kernel, bool fp16, int index = -1) {
    static const char* kHeadDimMacros[4] = {"", "D128", "D192", "D256"};
    std::string name = "glsl_attention_decode_";
    name += kernel;
    name += "_";
    if (index > 0) {
        name += kHeadDimMacros[index];
        name += "_";
    }
    if (fp16) {
        name += "FP16_";
    }
    name += "comp";
    return name;
}

static std::string _prefillKBlockShaderName(const char* kernel, bool fp16) {
    std::string name = "glsl_attention_prefill_kblock_";
    name += kernel;
    name += "_";
    if (fp16) {
        name += "FP16_";
    }
    name += "comp";
    return name;
}

void VulkanAttention::KVCache::reset() {
    maxLen = 0;
    kvHeadNum = 0;
    headDim = 0;
    fp16 = false;
    key = nullptr;
    value = nullptr;
}

bool VulkanAttention::KVCache::ensureCapacity(VulkanBackend* vkBn, int requiredLen, int kvH, int d, bool useFP16) {
    MNN_ASSERT(requiredLen >= 0);
    MNN_ASSERT(kvH > 0);
    MNN_ASSERT(d > 0);
    if (kvHeadNum != kvH || headDim != d || fp16 != useFP16 || nullptr == key || nullptr == value) {
        reset();
        kvHeadNum = kvH;
        headDim = d;
        fp16 = useFP16;
        maxLen = requiredLen + expandChunk;
        maxLen = ALIMAX(maxLen, expandChunk);
        const size_t bytes = fp16 ? sizeof(uint16_t) : sizeof(float);
        const size_t bufSize = (size_t)maxLen * (size_t)kvHeadNum * (size_t)headDim * bytes;
        key.reset(new VulkanBuffer(vkBn->getMemoryPool(), false, bufSize, nullptr,
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                       VK_BUFFER_USAGE_TRANSFER_DST_BIT));
        value.reset(new VulkanBuffer(vkBn->getMemoryPool(), false, bufSize, nullptr,
                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                         VK_BUFFER_USAGE_TRANSFER_DST_BIT));
        if (nullptr == key || nullptr == value || key->buffer() == VK_NULL_HANDLE ||
            value->buffer() == VK_NULL_HANDLE) {
            reset();
            return false;
        }
        return true;
    }
    if (requiredLen <= maxLen) {
        return true;
    }
    const int oldMaxLen = maxLen;
    maxLen = requiredLen + expandChunk;
    const size_t bytes = fp16 ? sizeof(uint16_t) : sizeof(float);
    const size_t newSize = (size_t)maxLen * (size_t)kvHeadNum * (size_t)headDim * bytes;
    std::shared_ptr<VulkanBuffer> newKey(new VulkanBuffer(
        vkBn->getMemoryPool(), false, newSize, nullptr,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT));
    std::shared_ptr<VulkanBuffer> newValue(new VulkanBuffer(
        vkBn->getMemoryPool(), false, newSize, nullptr,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT));
    if (nullptr == newKey || nullptr == newValue || newKey->buffer() == VK_NULL_HANDLE ||
        newValue->buffer() == VK_NULL_HANDLE) {
        return false;
    }
    // Preserve old content.
    //
    // cacheKey is packed as [kvHeadNum, headDim/4, maxLen, 4], so changing maxLen changes the row stride and we must
    // repack. cacheValue is kvh-major as [kvHeadNum, maxLen, headDim], so changing maxLen changes the kvh stride and we
    // must repack too.
    const size_t oldSize = key->size();
    if (oldSize > 0) {
        // Value: repack kvh blocks with new stride.
        {
            const VkDeviceSize rowBytes = (VkDeviceSize)oldMaxLen * (VkDeviceSize)headDim * (VkDeviceSize)bytes;
            const VkDeviceSize srcStride = rowBytes;
            const VkDeviceSize dstStride = (VkDeviceSize)maxLen * (VkDeviceSize)headDim * (VkDeviceSize)bytes;
            std::vector<VkBufferCopy> regions;
            regions.reserve((size_t)kvHeadNum);
            for (int kvh = 0; kvh < kvHeadNum; ++kvh) {
                VkBufferCopy c;
                c.srcOffset = (VkDeviceSize)kvh * srcStride;
                c.dstOffset = (VkDeviceSize)kvh * dstStride;
                c.size = rowBytes;
                regions.emplace_back(c);
            }
            vkBn->copyGPUToGPUBufferRegions(value->buffer(), newValue->buffer(), regions.data(),
                                            (uint32_t)regions.size());
        }

        // Key: repack rows with new stride.
        const int d4Size = headDim / 4;
        MNN_ASSERT(d4Size > 0);
        const uint32_t rowCount = (uint32_t)kvHeadNum * (uint32_t)d4Size;
        const VkDeviceSize vec4Bytes = (VkDeviceSize)(4 * bytes);
        const VkDeviceSize srcRowStride = (VkDeviceSize)oldMaxLen * vec4Bytes;
        const VkDeviceSize dstRowStride = (VkDeviceSize)maxLen * vec4Bytes;
        std::vector<VkBufferCopy> regions;
        regions.reserve(rowCount);
        for (uint32_t r = 0; r < rowCount; ++r) {
            VkBufferCopy c;
            c.srcOffset = (VkDeviceSize)r * srcRowStride;
            c.dstOffset = (VkDeviceSize)r * dstRowStride;
            c.size = srcRowStride;
            regions.emplace_back(c);
        }
        vkBn->copyGPUToGPUBufferRegions(key->buffer(), newKey->buffer(), regions.data(), (uint32_t)regions.size());
    }
    key = newKey;
    value = newValue;
    return true;
}

VulkanAttention::VulkanAttention(const Op* op, Backend* bn) : VulkanBasicExecution(bn) {
    auto vkBn = static_cast<VulkanBackend*>(bn);
    mUseFP16 = vkBn->useFP16();
    mMeta = reinterpret_cast<KVMeta*>(vkBn->getMetaPtr());
    mNeedKvCache = nullptr != mMeta;
    if (nullptr != op && nullptr != op->main_as_AttentionParam()) {
        auto param = op->main_as_AttentionParam();
        mOutputC4 = param->output_c4();
        mAttnScale = param->attnScale();
    }
    mKVCache.reset(new KVCache);
    mParam = vkBn->allocUniform(nullptr, sizeof(GpuParam));

    if (!mNeedKvCache && !ensureLegacyPipeline(vkBn)) {
        MNN_ERROR("VulkanAttention create legacy pipeline failed\n");
    }
    if (!ensureUpdatePipeline(vkBn) || !ensurePrefillPipelines(vkBn) || !ensureAttentionPipeline(vkBn) ||
        !ensureDecodeSmallFusedPipelines(vkBn)) {
        MNN_ERROR("VulkanAttention create pipeline failed\n");
    }
}

bool VulkanAttention::ensureLegacyPipeline(VulkanBackend* vkBn) {
    if (mAttentionLegacyPipeline && mAttentionLegacySet) {
        return true;
    }
    std::vector<VkDescriptorType> typesAttn(7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    typesAttn.emplace_back(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    std::string attnName = "glsl_attention_fused_";
    if (mUseFP16) {
        attnName += "FP16_";
    }
    attnName += "comp";
    mAttentionLegacyPipeline = vkBn->getPipeline(attnName, typesAttn);
    if (nullptr == mAttentionLegacyPipeline) {
        return false;
    }
    mAttentionLegacySet.reset(mAttentionLegacyPipeline->createSet());
    return nullptr != mAttentionLegacySet;
}

bool VulkanAttention::ensureUpdatePipeline(VulkanBackend* vkBn) {
    if (mUpdatePipeline && mUpdateSet) {
        return true;
    }
    std::vector<VkDescriptorType> typesUpdate{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                              VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
    std::string updateName = "glsl_attention_kvcache_update_";
    if (mUseFP16) {
        updateName += "FP16_";
    }
    updateName += "comp";
    mUpdatePipeline = vkBn->getPipeline(updateName, typesUpdate);
    if (nullptr == mUpdatePipeline) {
        return false;
    }
    mUpdateSet.reset(mUpdatePipeline->createSet());
    return nullptr != mUpdateSet;
}

bool VulkanAttention::ensurePrefillPipelines(VulkanBackend* vkBn) {
    if (mRearrangeQPipeline && mInitStatePipeline && mQKBlockPipeline && mQKBlockFullPipeline &&
        mSoftmaxOnlinePipeline && mQKVAccPipeline && mQKVAccFullPipeline && mQKVAccFinalFullPipeline &&
        mFinalizePipeline && mRearrangeQSet && mInitStateSet && mQKBlockSet && mQKBlockFullSet && mSoftmaxOnlineSet &&
        mQKVAccSet && mQKVAccFullSet && mQKVAccFinalFullSet && mFinalizeSet) {
        return true;
    }
    if (!ensureUpdatePipeline(vkBn)) {
        return false;
    }
    {
        std::vector<VkDescriptorType> types{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
        std::string name = "glsl_attention_prefill_rearrange_q_";
        if (mUseFP16) {
            name += "FP16_";
        }
        name += "comp";
        mRearrangeQPipeline = vkBn->getPipeline(name, types);
        if (nullptr == mRearrangeQPipeline)
            return false;
        mRearrangeQSet.reset(mRearrangeQPipeline->createSet());
    }
    {
        std::vector<VkDescriptorType> types{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
        std::string name = "glsl_attention_prefill_kblock_init_state_";
        if (mUseFP16) {
            name += "FP16_";
        }
        name += "comp";
        mInitStatePipeline = vkBn->getPipeline(name, types);
        if (nullptr == mInitStatePipeline)
            return false;
        mInitStateSet.reset(mInitStatePipeline->createSet());
    }
    {
        std::vector<VkDescriptorType> types{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
        std::string name = "glsl_attention_prefill_kblock_qk_";
        if (mUseFP16) {
            name += "FP16_";
        }
        name += "comp";
        mQKBlockPipeline = vkBn->getPipeline(name, types);
        if (nullptr == mQKBlockPipeline)
            return false;
        mQKBlockSet.reset(mQKBlockPipeline->createSet());

        std::string fullName = "glsl_attention_prefill_kblock_qk_full_";
        if (mUseFP16) {
            fullName += "FP16_";
        }
        fullName += "comp";
        mQKBlockFullPipeline = vkBn->getPipeline(fullName, types);
        if (nullptr == mQKBlockFullPipeline)
            return false;
        mQKBlockFullSet.reset(mQKBlockFullPipeline->createSet());
    }
    {
        std::vector<VkDescriptorType> types{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
        std::string name = "glsl_attention_prefill_kblock_softmax_online_";
        if (mUseFP16) {
            name += "FP16_";
        }
        name += "comp";
        const auto& limits = vkBn->getDevice().proty().limits;
        const int kBlock4 = UP_DIV(kAttentionPrefillKBlock, 4) * 4;
        const int maxK4 = UP_DIV(kBlock4, 4);
        uint32_t localSize = _selectSoftmaxLocalSize(maxK4, (uint32_t)limits.maxComputeWorkGroupSize[0],
                                                     (uint32_t)limits.maxComputeWorkGroupInvocations);
        mSoftmaxOnlinePipeline = vkBn->getPipeline(name, types, {localSize});
        if (nullptr == mSoftmaxOnlinePipeline)
            return false;
        mSoftmaxOnlineSet.reset(mSoftmaxOnlinePipeline->createSet());
        mSoftmaxOnlineLocalSize = localSize;
    }
    {
        std::vector<VkDescriptorType> types{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
        std::string name = "glsl_attention_prefill_kblock_qkv_acc_";
        if (mUseFP16) {
            name += "FP16_";
        }
        name += "comp";
        mQKVAccPipeline = vkBn->getPipeline(name, types);
        if (nullptr == mQKVAccPipeline)
            return false;
        mQKVAccSet.reset(mQKVAccPipeline->createSet());

        std::string fullName = "glsl_attention_prefill_kblock_qkv_acc_full_";
        if (mUseFP16) {
            fullName += "FP16_";
        }
        fullName += "comp";
        mQKVAccFullPipeline = vkBn->getPipeline(fullName, types);
        if (nullptr == mQKVAccFullPipeline)
            return false;
        mQKVAccFullSet.reset(mQKVAccFullPipeline->createSet());

        std::vector<VkDescriptorType> finalTypes{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                 VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                 VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                 VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
        std::string finalFullName = "glsl_attention_prefill_kblock_qkv_acc_final_full_";
        if (mUseFP16) {
            finalFullName += "FP16_";
        }
        finalFullName += "comp";
        mQKVAccFinalFullPipeline = vkBn->getPipeline(finalFullName, finalTypes);
        if (nullptr == mQKVAccFinalFullPipeline)
            return false;
        mQKVAccFinalFullSet.reset(mQKVAccFinalFullPipeline->createSet());
    }
    {
        std::vector<VkDescriptorType> types{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
        std::string name = "glsl_attention_prefill_kblock_finalize_";
        if (mUseFP16) {
            name += "FP16_";
        }
        name += "comp";
        mFinalizePipeline = vkBn->getPipeline(name, types);
        if (nullptr == mFinalizePipeline)
            return false;
        mFinalizeSet.reset(mFinalizePipeline->createSet());
    }
    return mRearrangeQSet && mInitStateSet && mQKBlockSet && mQKBlockFullSet && mSoftmaxOnlineSet && mQKVAccSet &&
           mQKVAccFullSet && mQKVAccFinalFullSet && mFinalizeSet;
}

bool VulkanAttention::ensureAttentionPipeline(VulkanBackend* vkBn) {
    if (mAttentionPipeline && mAttentionSet) {
        return true;
    }
    std::vector<VkDescriptorType> typesAttn(7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    typesAttn.emplace_back(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    std::string attnName = "glsl_attention_fused_packed_";
    if (mUseFP16) {
        attnName += "FP16_";
    }
    attnName += "comp";
    mAttentionPipeline = vkBn->getPipeline(attnName, typesAttn);
    if (nullptr == mAttentionPipeline) {
        return false;
    }
    mAttentionSet.reset(mAttentionPipeline->createSet());
    return nullptr != mAttentionSet;
}

bool VulkanAttention::ensureDecodeSmallFusedPipelines(VulkanBackend* vkBn) {
    bool ready = true;
    for (int i = 0; i < 4; ++i) {
        ready = ready && mDecodeSingleFusedPipelines[i] && mDecodeSingleFusedSets[i] && mDecodeSmallFusedPipelines[i] &&
                mDecodeSmallFusedSets[i];
    }
    if (!mNeedKvCache || ready) {
        return true;
    }
    if (!_supportDecodeSubgroup(vkBn->getDevice())) {
        return true;
    }
    mDecodeSubgroupLocalSize = vkBn->getDevice().getSubgroupSize();
    if (mDecodeSubgroupLocalSize == 0) {
        return true;
    }

    std::vector<VkDescriptorType> typesAttn(7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    typesAttn.emplace_back(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    for (int i = 0; i < 4; ++i) {
        if (nullptr == mDecodeSingleFusedPipelines[i]) {
            const std::string name = _decodeSingleFusedShaderName(mUseFP16, i);
            mDecodeSingleFusedPipelines[i] = vkBn->getPipeline(name, typesAttn, {mDecodeSubgroupLocalSize});
            if (nullptr != mDecodeSingleFusedPipelines[i]) {
                mDecodeSingleFusedSets[i].reset(mDecodeSingleFusedPipelines[i]->createSet());
            }
        }
        if (mDecodeSmallFusedPipelines[i] && mDecodeSmallFusedSets[i]) {
            continue;
        }
        const std::string name = _decodeSmallFusedShaderName(mUseFP16, i);
        mDecodeSmallFusedPipelines[i] = vkBn->getPipeline(name, typesAttn, {mDecodeSubgroupLocalSize});
        if (nullptr != mDecodeSmallFusedPipelines[i]) {
            mDecodeSmallFusedSets[i].reset(mDecodeSmallFusedPipelines[i]->createSet());
        }
    }
    return true;
}

bool VulkanAttention::ensureDecodeTwoStagePipelines(VulkanBackend* vkBn) {
    bool ready = mDecodeTwoStagePipelineMaskMode == mDecodeMaskMode && mDecodeQkvPipeline && mDecodeQkvSet;
    for (int i = 0; i < 4; ++i) {
        ready = ready && mDecodeQkSoftmaxPipelines[i] && mDecodeQkSoftmaxSets[i];
    }
    if (!mNeedKvCache || ready) {
        return true;
    }
    if (!_supportDecodeSubgroup(vkBn->getDevice())) {
        return true;
    }
    const uint32_t subgroupSize = vkBn->getDevice().getSubgroupSize();
    if (subgroupSize == 0) {
        return true;
    }
    const auto& limits = vkBn->getDevice().proty().limits;
    uint32_t qkLocalSize = subgroupSize;
    const uint32_t qkLocalCap =
        ALIMIN((uint32_t)kAttentionDecodeTwoStageQkLocalCap,
               ALIMIN((uint32_t)limits.maxComputeWorkGroupInvocations, (uint32_t)limits.maxComputeWorkGroupSize[0]));
    while ((qkLocalSize << 1) <= qkLocalCap) {
        qkLocalSize <<= 1;
    }
    if (qkLocalSize < subgroupSize) {
        return true;
    }
    if (mDecodeTwoStagePipelineMaskMode != mDecodeMaskMode) {
        for (int i = 0; i < 4; ++i) {
            mDecodeQkSoftmaxPipelines[i] = nullptr;
            mDecodeQkSoftmaxSets[i].reset();
        }
    }

    std::vector<VkDescriptorType> qkTypes{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
    const std::vector<uint32_t> qkSpec{
        static_cast<uint32_t>(mDecodeMaskMode),
        2u,
        (uint32_t)kAttentionDecodeTwoStageSoftmaxStride,
    };
    for (int i = 0; i < 4; ++i) {
        if (nullptr == mDecodeQkSoftmaxPipelines[i]) {
            const std::string name = _decodeTwoStageShaderName("qk_softmax", mUseFP16, i);
            mDecodeQkSoftmaxPipelines[i] = vkBn->getPipeline(name, qkTypes, {qkLocalSize}, qkSpec);
            if (nullptr != mDecodeQkSoftmaxPipelines[i]) {
                mDecodeQkSoftmaxSets[i].reset(mDecodeQkSoftmaxPipelines[i]->createSet());
            }
        }
    }

    if (nullptr == mDecodeQkvPipeline) {
        std::vector<VkDescriptorType> qkvTypes{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                               VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
        const std::vector<uint32_t> qkvSpec{
            mOutputC4 ? 2u : 1u,
            2u,
            (uint32_t)kAttentionDecodeTwoStageSoftmaxStride,
        };
        const std::string name = _decodeTwoStageShaderName("qkv", mUseFP16);
        mDecodeQkvPipeline = vkBn->getPipeline(name, qkvTypes, {subgroupSize}, qkvSpec);
        if (nullptr != mDecodeQkvPipeline) {
            mDecodeQkvSet.reset(mDecodeQkvPipeline->createSet());
        }
    }
    mDecodeTwoStagePipelineMaskMode = mDecodeMaskMode;
    return true;
}

VulkanAttention::~VulkanAttention() {
    auto vkBn = static_cast<VulkanBackend*>(backend());
    if (mTempQuery) {
        vkBn->onReleaseBuffer(mTempQuery.get(), Backend::DYNAMIC);
        mTempQuery.reset();
    }
    if (mTempDecodeSoftmax) {
        vkBn->onReleaseBuffer(mTempDecodeSoftmax.get(), Backend::DYNAMIC);
        mTempDecodeSoftmax.reset();
    }
    if (mTempQKBlock) {
        vkBn->onReleaseBuffer(mTempQKBlock.get(), Backend::DYNAMIC);
        mTempQKBlock.reset();
    }
    if (mTempWBlock) {
        vkBn->onReleaseBuffer(mTempWBlock.get(), Backend::DYNAMIC);
        mTempWBlock.reset();
    }
    if (mTempM) {
        vkBn->onReleaseBuffer(mTempM.get(), Backend::DYNAMIC);
        mTempM.reset();
    }
    if (mTempL) {
        vkBn->onReleaseBuffer(mTempL.get(), Backend::DYNAMIC);
        mTempL.reset();
    }
    if (mTempAlpha) {
        vkBn->onReleaseBuffer(mTempAlpha.get(), Backend::DYNAMIC);
        mTempAlpha.reset();
    }
    if (mTempOAcc) {
        vkBn->onReleaseBuffer(mTempOAcc.get(), Backend::DYNAMIC);
        mTempOAcc.reset();
    }
    if (mTempCacheKey) {
        vkBn->onReleaseBuffer(mTempCacheKey.get(), Backend::DYNAMIC);
        mTempCacheKey.reset();
    }
    if (mTempCacheValue) {
        vkBn->onReleaseBuffer(mTempCacheValue.get(), Backend::DYNAMIC);
        mTempCacheValue.reset();
    }
    vkBn->recycleUniform(mParam);
}

bool VulkanAttention::onClone(Backend* bn, const Op* op, VulkanBasicExecution** dst) {
    if (nullptr == dst) {
        return true;
    }
    auto res = new VulkanAttention(op, bn);
    if (bn->getMetaPtr() == mMeta && nullptr != mMeta) {
        res->mKVCache = mKVCache;
        res->mMeta = mMeta;
    }
    *dst = res;
    return true;
}

ErrorCode VulkanAttention::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                    const VulkanCommandPool::Buffer* cmdBuffer) {
    MNN_ASSERT(!inputs.empty());
    MNN_ASSERT(!outputs.empty());
    auto query = inputs[0];
    auto key = inputs[1];
    auto value = inputs[2];
    MNN_ASSERT(nullptr != query && nullptr != key && nullptr != value);
    MNN_ASSERT(query->dimensions() == 4);
    MNN_ASSERT(key->dimensions() == 4);
    MNN_ASSERT(value->dimensions() == 4);
    MNN_ASSERT(query->length(0) == 1);
    MNN_ASSERT(key->length(0) == 1);
    MNN_ASSERT(value->length(0) == 1);
    mQueryLen = query->length(1);
    mKeyLen = key->length(1);
    mHeadNum = query->length(2);
    mHeadDim = query->length(3);
    mKvHeadNum = key->length(2);
    MNN_ASSERT(mHeadNum > 0 && mKvHeadNum > 0);
    MNN_ASSERT(mHeadNum % mKvHeadNum == 0);
    MNN_ASSERT(mHeadDim > 0);
    MNN_ASSERT((mHeadDim & 3) == 0);
    MNN_ASSERT(mHeadDim <= 256);
    MNN_ASSERT(value->length(1) == mKeyLen);
    MNN_ASSERT(value->length(2) == mKvHeadNum);
    MNN_ASSERT(value->length(3) == mHeadDim);

    auto vkBn = static_cast<VulkanBackend*>(backend());
    auto cmd = cmdBuffer->get();

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

    auto dispatchIndirectWithProfile = [&](const char* name, const VulkanPipeline* pipeline,
                                           const std::shared_ptr<VulkanLayout::DescriptorSet>& set, int slot) {
        MNN_ASSERT(nullptr != mDecodeIndirectBuffer);
#ifdef ENABLE_VULKAN_TIME_PROFILE
        auto* profiler = vkBn->timeProfiler();
        if (nullptr != profiler) {
            VulkanTimeProfileScope scope(profiler, cmd, name, VulkanTimeProfiler::Kind::Shader);
            pipeline->bind(cmd, set->get());
            vkCmdDispatchIndirect(cmd, mDecodeIndirectBuffer->buffer(), _dispatchIndirectOffset(slot));
            return;
        }
#endif
        pipeline->bind(cmd, set->get());
        vkCmdDispatchIndirect(cmd, mDecodeIndirectBuffer->buffer(), _dispatchIndirectOffset(slot));
    };

    int pastLenForRoute = 0;
    if (mNeedKvCache) {
        MNN_ASSERT(nullptr != mMeta);
        MNN_ASSERT(mMeta->n_reserve == 0);
        MNN_ASSERT(mMeta->computeReverseSize() == 0);
        const int previous = (int)mMeta->previous;
        const int remove = (int)mMeta->remove;
        MNN_ASSERT(previous >= 0);
        MNN_ASSERT(remove >= 0);
        MNN_ASSERT(remove <= previous);
        pastLenForRoute = previous - remove;
    }
    mDecodeMaskMode = 0;
    if (inputs.size() > 3 && nullptr != inputs[3]) {
        mDecodeMaskMode = inputs[3]->shape().empty() ? 2 : 1;
    }
    const int alignedDecodeIndex = _decodeHeadDimAlignedIndex(mHeadDim);
    const int decodeGroup = mHeadNum / mKvHeadNum;
    mUseDecodeTwoStageIndirect = false;
    mUseDecodeTwoStageDirect = false;
    const int totalLenForRoute = pastLenForRoute + mKeyLen;
    const bool mayNeedTwoStageDecode = pastLenForRoute == 0 || (pastLenForRoute >= kAttentionDecodeTwoStageMinPastLen &&
                                                                totalLenForRoute <= kAttentionDecodeTwoStageMaxLen);
    const bool decodeTwoStageCandidate = mNeedKvCache && mQueryLen == 1 && mKeyLen == 1 && mayNeedTwoStageDecode &&
                                         decodeGroup == 2 && mDecodeMaskMode != 1 && alignedDecodeIndex >= 0 &&
                                         nullptr != mDecodeSingleFusedPipelines[alignedDecodeIndex] &&
                                         nullptr != mDecodeSingleFusedSets[alignedDecodeIndex];
    if (decodeTwoStageCandidate && !ensureDecodeTwoStagePipelines(vkBn)) {
        return OUT_OF_MEMORY;
    }
    const bool decodeTwoStageReady = decodeTwoStageCandidate &&
                                     nullptr != mDecodeQkSoftmaxPipelines[alignedDecodeIndex] &&
                                     nullptr != mDecodeQkSoftmaxSets[alignedDecodeIndex] &&
                                     nullptr != mDecodeQkvPipeline && nullptr != mDecodeQkvSet;
    mUseDecodeTwoStageIndirect = decodeTwoStageReady && pastLenForRoute == 0;
    mUseDecodeTwoStageDirect = decodeTwoStageReady && !mUseDecodeTwoStageIndirect;
    const bool useSmallDecode =
        mNeedKvCache && mQueryLen <= 4 && !mUseDecodeTwoStageIndirect && !mUseDecodeTwoStageDirect;
    const bool usePrefill = mQueryLen > 1 && !useSmallDecode;
    mUsePrefill = usePrefill;
    if (!mUseDecodeTwoStageIndirect && !mUseDecodeTwoStageDirect) {
        mDecodeIndirectBuffer.reset();
        mDecodeIndirectCmdCount = 0;
        mDecodeIndirectCmdInitialized = false;
        mDecodeIndirectLastActive = false;
    }
    if (!mUseDecodeTwoStageIndirect && !mUseDecodeTwoStageDirect && mTempDecodeSoftmax) {
        vkBn->onReleaseBuffer(mTempDecodeSoftmax.get(), Backend::DYNAMIC);
        mTempDecodeSoftmax.reset();
    }

    if (mNeedKvCache) {
        if (!ensureUpdatePipeline(vkBn)) {
            return OUT_OF_MEMORY;
        }
        MNN_ASSERT(nullptr != mUpdatePipeline);
        MNN_ASSERT(nullptr != mUpdateSet);

        // Dispatch: KV update (x=dim/4, y=keyLen, z=kvHeadNum).
        dispatchWithProfile(mUseFP16 ? "glsl_attention_kvcache_update_FP16_comp" : "glsl_attention_kvcache_update_comp",
                            mUpdatePipeline, mUpdateSet, UP_DIV(mHeadDim / 4, 8), mKeyLen, mKvHeadNum);
        // NOTE: KV cache buffers may be reallocated in onBeforeExecute (descriptor set updated there), so we must not
        // record a VkBufferMemoryBarrier with a stale VkBuffer handle here. Use a global memory barrier instead.
        {
            VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier,
                                 0, nullptr, 0, nullptr);
        }
    }

    auto ensureDecodeSoftmaxTemp = [&]() -> bool {
        const int softmaxElements = mHeadNum * kAttentionDecodeTwoStageSoftmaxStride;
        if (mTempDecodeSoftmax && mTempDecodeSoftmax->elementSize() == softmaxElements) {
            return true;
        }
        if (mTempDecodeSoftmax) {
            vkBn->onReleaseBuffer(mTempDecodeSoftmax.get(), Backend::DYNAMIC);
            mTempDecodeSoftmax.reset();
        }
        mTempDecodeSoftmax.reset(Tensor::createDevice<float>({softmaxElements}));
        return vkBn->onAcquireBuffer(mTempDecodeSoftmax.get(), Backend::DYNAMIC);
    };

    if (mUseDecodeTwoStageDirect) {
        if (!ensureDecodeTwoStagePipelines(vkBn) || !ensureDecodeSoftmaxTemp()) {
            return OUT_OF_MEMORY;
        }
        MNN_ASSERT(alignedDecodeIndex >= 0);
        MNN_ASSERT(nullptr != mDecodeQkSoftmaxPipelines[alignedDecodeIndex]);
        MNN_ASSERT(nullptr != mDecodeQkSoftmaxSets[alignedDecodeIndex]);
        MNN_ASSERT(nullptr != mDecodeQkvPipeline);
        MNN_ASSERT(nullptr != mDecodeQkvSet);

        const auto softmaxBuf = vkBn->getTensorBuffer(mTempDecodeSoftmax.get());
        const size_t softmaxSize = vkBn->getTensorSize(mTempDecodeSoftmax.get());
        const std::string qkName = _decodeTwoStageShaderName("qk_softmax", mUseFP16, alignedDecodeIndex);
        dispatchWithProfile(qkName.c_str(), mDecodeQkSoftmaxPipelines[alignedDecodeIndex],
                            mDecodeQkSoftmaxSets[alignedDecodeIndex], (uint32_t)mKvHeadNum, 1, 1);
        cmdBuffer->barrierSource(softmaxBuf.first->buffer(), softmaxBuf.second, softmaxSize);

        const std::string qkvName = _decodeTwoStageShaderName("qkv", mUseFP16);
        dispatchWithProfile(qkvName.c_str(), mDecodeQkvPipeline, mDecodeQkvSet,
                            (uint32_t)UP_DIV(mHeadDim / 4, kAttentionDecodeTwoStageQkvD4Pack), (uint32_t)mHeadNum, 1);
        return NO_ERROR;
    }

    if (mUseDecodeTwoStageIndirect) {
        if (!ensureDecodeTwoStagePipelines(vkBn) || !ensureDecodeSoftmaxTemp()) {
            return OUT_OF_MEMORY;
        }
        MNN_ASSERT(alignedDecodeIndex >= 0);
        MNN_ASSERT(nullptr != mDecodeQkSoftmaxPipelines[alignedDecodeIndex]);
        MNN_ASSERT(nullptr != mDecodeQkSoftmaxSets[alignedDecodeIndex]);
        MNN_ASSERT(nullptr != mDecodeQkvPipeline);
        MNN_ASSERT(nullptr != mDecodeQkvSet);

        mDecodeIndirectCmdCount = kAttentionDecodeTwoStageIndirectCmdCount;
        mDecodeIndirectBuffer.reset(new VulkanBuffer(
            vkBn->getMemoryPool(), false, (size_t)mDecodeIndirectCmdCount * sizeof(VkDispatchIndirectCommand), nullptr,
            VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, VK_SHARING_MODE_EXCLUSIVE, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT));
        if (nullptr == mDecodeIndirectBuffer || mDecodeIndirectBuffer->buffer() == VK_NULL_HANDLE) {
            return OUT_OF_MEMORY;
        }
        mDecodeIndirectCmdInitialized = false;
        mDecodeIndirectLastActive = false;

        const auto softmaxBuf = vkBn->getTensorBuffer(mTempDecodeSoftmax.get());
        const size_t softmaxSize = vkBn->getTensorSize(mTempDecodeSoftmax.get());
        const std::string fusedName = _decodeSingleFusedShaderName(mUseFP16, alignedDecodeIndex);
        dispatchIndirectWithProfile(fusedName.c_str(), mDecodeSingleFusedPipelines[alignedDecodeIndex],
                                    mDecodeSingleFusedSets[alignedDecodeIndex], kAttentionDecodeIndirectFusedSlot);

        const std::string qkName = _decodeTwoStageShaderName("qk_softmax", mUseFP16, alignedDecodeIndex);
        dispatchIndirectWithProfile(qkName.c_str(), mDecodeQkSoftmaxPipelines[alignedDecodeIndex],
                                    mDecodeQkSoftmaxSets[alignedDecodeIndex], kAttentionDecodeIndirectTwoStageQkSlot);
        cmdBuffer->barrierSource(softmaxBuf.first->buffer(), softmaxBuf.second, softmaxSize);

        const std::string qkvName = _decodeTwoStageShaderName("qkv", mUseFP16);
        dispatchIndirectWithProfile(qkvName.c_str(), mDecodeQkvPipeline, mDecodeQkvSet,
                                    kAttentionDecodeIndirectTwoStageQkvSlot);
        return NO_ERROR;
    }

    if (usePrefill) {
        if (!ensurePrefillPipelines(vkBn)) {
            return OUT_OF_MEMORY;
        }
        constexpr int K_BLOCK = kAttentionPrefillKBlock;
        int pastLenForPrefill = 0;
        if (mNeedKvCache) {
            pastLenForPrefill = pastLenForRoute;
        }
        mPrefillTotalLen = pastLenForPrefill + mKeyLen;
        mQueryLen4 = UP_DIV(mQueryLen, 4) * 4;
        MNN_ASSERT(mPrefillTotalLen > 0);

        const int64_t queryElementsI64 = (int64_t)mHeadNum * (int64_t)mHeadDim * (int64_t)mQueryLen4;
        MNN_ASSERT(queryElementsI64 > 0 && queryElementsI64 <= (int64_t)INT_MAX);
        const int queryElements = (int)queryElementsI64;

        const int kBlock4 = UP_DIV(K_BLOCK, 4) * 4;
        const int64_t rowCountI64 = (int64_t)mQueryLen * (int64_t)mHeadNum;
        MNN_ASSERT(rowCountI64 > 0 && rowCountI64 <= (int64_t)INT_MAX);
        const int rowCount = (int)rowCountI64;

        const int64_t qkElementsI64 = (int64_t)rowCount * (int64_t)kBlock4;
        MNN_ASSERT(qkElementsI64 > 0 && qkElementsI64 <= (int64_t)INT_MAX);
        const int qkElements = (int)qkElementsI64;

        const int64_t oaccElementsI64 = (int64_t)rowCount * (int64_t)mHeadDim;
        MNN_ASSERT(oaccElementsI64 > 0 && oaccElementsI64 <= (int64_t)INT_MAX);
        const int oaccElements = (int)oaccElementsI64;

        // Acquire workspace tensors fresh each onEncode (Backend::DYNAMIC). They are released at the end of this
        // call so other ops within the same resize can reuse the pool chunks. Descriptor sets capture
        // (VkBuffer, offset) below; the underlying GPU memory stays alive past release because the parent
        // VkBuffer is owned by the pool, and command-buffer order + barriers serialize cross-op access.
        // M / L / Alpha / OAcc must be FP32 even when the backend runs FP16 -> int tensor forces 4-byte storage.
        auto acquireTemp = [&](std::shared_ptr<Tensor>& t, Tensor* dev) -> bool {
            t.reset(dev);
            return vkBn->onAcquireBuffer(t.get(), Backend::DYNAMIC);
        };
        std::pair<const VulkanBuffer*, size_t> tempCacheKeyBuf{nullptr, 0};
        std::pair<const VulkanBuffer*, size_t> tempCacheValueBuf{nullptr, 0};
        if (!mNeedKvCache) {
            if (!acquireTemp(mTempCacheKey, Tensor::createDevice<float>({mKvHeadNum, mKeyLen, mHeadDim})))
                return OUT_OF_MEMORY;
            if (!acquireTemp(mTempCacheValue, Tensor::createDevice<float>({mKvHeadNum, mKeyLen, mHeadDim})))
                return OUT_OF_MEMORY;
            tempCacheKeyBuf = vkBn->getTensorBuffer(mTempCacheKey.get());
            tempCacheValueBuf = vkBn->getTensorBuffer(mTempCacheValue.get());
        }
        if (!acquireTemp(mTempQuery, Tensor::createDevice<float>({queryElements})))
            return OUT_OF_MEMORY;
        if (!acquireTemp(mTempQKBlock, Tensor::createDevice<float>({qkElements})))
            return OUT_OF_MEMORY;
        if (!acquireTemp(mTempWBlock, Tensor::createDevice<float>({qkElements})))
            return OUT_OF_MEMORY;
        if (!acquireTemp(mTempM, Tensor::createDevice<int>({rowCount})))
            return OUT_OF_MEMORY;
        if (!acquireTemp(mTempL, Tensor::createDevice<int>({rowCount})))
            return OUT_OF_MEMORY;
        if (!acquireTemp(mTempAlpha, Tensor::createDevice<int>({rowCount})))
            return OUT_OF_MEMORY;
        if (!acquireTemp(mTempOAcc, Tensor::createDevice<int>({oaccElements})))
            return OUT_OF_MEMORY;

        MNN_ASSERT(nullptr != mRearrangeQPipeline);
        MNN_ASSERT(nullptr != mRearrangeQSet);
        MNN_ASSERT(nullptr != mInitStatePipeline);
        MNN_ASSERT(nullptr != mInitStateSet);
        MNN_ASSERT(nullptr != mQKBlockPipeline);
        MNN_ASSERT(nullptr != mQKBlockSet);
        MNN_ASSERT(nullptr != mQKBlockFullPipeline);
        MNN_ASSERT(nullptr != mQKBlockFullSet);
        MNN_ASSERT(nullptr != mSoftmaxOnlinePipeline);
        MNN_ASSERT(nullptr != mSoftmaxOnlineSet);
        MNN_ASSERT(nullptr != mQKVAccPipeline);
        MNN_ASSERT(nullptr != mQKVAccSet);
        MNN_ASSERT(nullptr != mQKVAccFullPipeline);
        MNN_ASSERT(nullptr != mQKVAccFullSet);
        MNN_ASSERT(nullptr != mQKVAccFinalFullPipeline);
        MNN_ASSERT(nullptr != mQKVAccFinalFullSet);
        MNN_ASSERT(nullptr != mFinalizePipeline);
        MNN_ASSERT(nullptr != mFinalizeSet);

        // Bind workspace + uniform descriptor slots here (their (VkBuffer, offset) is stable across executes).
        // Cache (cacheKey/cacheValue) and external I/O (query/mask/output) are bound in onBeforeExecute because
        // KV cache may be reallocated by ensureCapacity().
        auto tqBuf = vkBn->getTensorBuffer(mTempQuery.get());
        auto qkBuf = vkBn->getTensorBuffer(mTempQKBlock.get());
        auto wBuf = vkBn->getTensorBuffer(mTempWBlock.get());
        auto mBuf = vkBn->getTensorBuffer(mTempM.get());
        auto lBuf = vkBn->getTensorBuffer(mTempL.get());
        auto aBuf = vkBn->getTensorBuffer(mTempAlpha.get());
        auto oBuf = vkBn->getTensorBuffer(mTempOAcc.get());

        if (!mNeedKvCache) {
            auto keyBuf = vkBn->getTensorBuffer(key);
            auto valueBuf = vkBn->getTensorBuffer(value);
            mUpdateSet->writeBuffer(keyBuf.first->buffer(), 0, vkBn->getTensorSize(key), keyBuf.second);
            mUpdateSet->writeBuffer(valueBuf.first->buffer(), 1, vkBn->getTensorSize(value), valueBuf.second);
            mUpdateSet->writeBuffer(tempCacheKeyBuf.first->buffer(), 2, vkBn->getTensorSize(mTempCacheKey.get()),
                                    tempCacheKeyBuf.second);
            mUpdateSet->writeBuffer(tempCacheValueBuf.first->buffer(), 3, vkBn->getTensorSize(mTempCacheValue.get()),
                                    tempCacheValueBuf.second);
            mUpdateSet->writeBuffer(mParam->buffer(), 4, mParam->size());
        }

        mRearrangeQSet->writeBuffer(tqBuf.first->buffer(), 0, vkBn->getTensorSize(mTempQuery.get()), tqBuf.second);
        mRearrangeQSet->writeBuffer(mParam->buffer(), 2, mParam->size());

        mInitStateSet->writeBuffer(mBuf.first->buffer(), 0, vkBn->getTensorSize(mTempM.get()), mBuf.second);
        mInitStateSet->writeBuffer(lBuf.first->buffer(), 1, vkBn->getTensorSize(mTempL.get()), lBuf.second);
        mInitStateSet->writeBuffer(aBuf.first->buffer(), 2, vkBn->getTensorSize(mTempAlpha.get()), aBuf.second);
        mInitStateSet->writeBuffer(oBuf.first->buffer(), 3, vkBn->getTensorSize(mTempOAcc.get()), oBuf.second);
        mInitStateSet->writeBuffer(mParam->buffer(), 4, mParam->size());

        mQKBlockSet->writeBuffer(qkBuf.first->buffer(), 0, vkBn->getTensorSize(mTempQKBlock.get()), qkBuf.second);
        mQKBlockSet->writeBuffer(tqBuf.first->buffer(), 1, vkBn->getTensorSize(mTempQuery.get()), tqBuf.second);
        if (!mNeedKvCache) {
            mQKBlockSet->writeBuffer(tempCacheKeyBuf.first->buffer(), 2, vkBn->getTensorSize(mTempCacheKey.get()),
                                     tempCacheKeyBuf.second);
        }
        mQKBlockSet->writeBuffer(mParam->buffer(), 4, mParam->size());

        mQKBlockFullSet->writeBuffer(qkBuf.first->buffer(), 0, vkBn->getTensorSize(mTempQKBlock.get()), qkBuf.second);
        mQKBlockFullSet->writeBuffer(tqBuf.first->buffer(), 1, vkBn->getTensorSize(mTempQuery.get()), tqBuf.second);
        if (!mNeedKvCache) {
            mQKBlockFullSet->writeBuffer(tempCacheKeyBuf.first->buffer(), 2, vkBn->getTensorSize(mTempCacheKey.get()),
                                         tempCacheKeyBuf.second);
        }
        mQKBlockFullSet->writeBuffer(mParam->buffer(), 4, mParam->size());

        mSoftmaxOnlineSet->writeBuffer(wBuf.first->buffer(), 0, vkBn->getTensorSize(mTempWBlock.get()), wBuf.second);
        mSoftmaxOnlineSet->writeBuffer(qkBuf.first->buffer(), 1, vkBn->getTensorSize(mTempQKBlock.get()), qkBuf.second);
        mSoftmaxOnlineSet->writeBuffer(mBuf.first->buffer(), 2, vkBn->getTensorSize(mTempM.get()), mBuf.second);
        mSoftmaxOnlineSet->writeBuffer(lBuf.first->buffer(), 3, vkBn->getTensorSize(mTempL.get()), lBuf.second);
        mSoftmaxOnlineSet->writeBuffer(aBuf.first->buffer(), 4, vkBn->getTensorSize(mTempAlpha.get()), aBuf.second);
        mSoftmaxOnlineSet->writeBuffer(mParam->buffer(), 5, mParam->size());

        mQKVAccSet->writeBuffer(oBuf.first->buffer(), 0, vkBn->getTensorSize(mTempOAcc.get()), oBuf.second);
        mQKVAccSet->writeBuffer(wBuf.first->buffer(), 1, vkBn->getTensorSize(mTempWBlock.get()), wBuf.second);
        if (!mNeedKvCache) {
            mQKVAccSet->writeBuffer(tempCacheValueBuf.first->buffer(), 2, vkBn->getTensorSize(mTempCacheValue.get()),
                                    tempCacheValueBuf.second);
        }
        mQKVAccSet->writeBuffer(aBuf.first->buffer(), 3, vkBn->getTensorSize(mTempAlpha.get()), aBuf.second);
        mQKVAccSet->writeBuffer(mParam->buffer(), 4, mParam->size());

        mQKVAccFullSet->writeBuffer(oBuf.first->buffer(), 0, vkBn->getTensorSize(mTempOAcc.get()), oBuf.second);
        mQKVAccFullSet->writeBuffer(wBuf.first->buffer(), 1, vkBn->getTensorSize(mTempWBlock.get()), wBuf.second);
        if (!mNeedKvCache) {
            mQKVAccFullSet->writeBuffer(tempCacheValueBuf.first->buffer(), 2,
                                        vkBn->getTensorSize(mTempCacheValue.get()), tempCacheValueBuf.second);
        }
        mQKVAccFullSet->writeBuffer(aBuf.first->buffer(), 3, vkBn->getTensorSize(mTempAlpha.get()), aBuf.second);
        mQKVAccFullSet->writeBuffer(mParam->buffer(), 4, mParam->size());

        mQKVAccFinalFullSet->writeBuffer(oBuf.first->buffer(), 1, vkBn->getTensorSize(mTempOAcc.get()), oBuf.second);
        mQKVAccFinalFullSet->writeBuffer(wBuf.first->buffer(), 2, vkBn->getTensorSize(mTempWBlock.get()), wBuf.second);
        if (!mNeedKvCache) {
            mQKVAccFinalFullSet->writeBuffer(tempCacheValueBuf.first->buffer(), 3,
                                             vkBn->getTensorSize(mTempCacheValue.get()), tempCacheValueBuf.second);
        }
        mQKVAccFinalFullSet->writeBuffer(aBuf.first->buffer(), 4, vkBn->getTensorSize(mTempAlpha.get()), aBuf.second);
        mQKVAccFinalFullSet->writeBuffer(lBuf.first->buffer(), 5, vkBn->getTensorSize(mTempL.get()), lBuf.second);
        mQKVAccFinalFullSet->writeBuffer(mParam->buffer(), 6, mParam->size());

        mFinalizeSet->writeBuffer(oBuf.first->buffer(), 1, vkBn->getTensorSize(mTempOAcc.get()), oBuf.second);
        mFinalizeSet->writeBuffer(lBuf.first->buffer(), 2, vkBn->getTensorSize(mTempL.get()), lBuf.second);
        mFinalizeSet->writeBuffer(mParam->buffer(), 3, mParam->size());

        if (!mNeedKvCache) {
            dispatchWithProfile(mUseFP16 ? "glsl_attention_kvcache_update_FP16_comp"
                                         : "glsl_attention_kvcache_update_comp",
                                mUpdatePipeline, mUpdateSet, UP_DIV(mHeadDim / 4, 8), mKeyLen, mKvHeadNum);
            cmdBuffer->barrierSource(tempCacheKeyBuf.first->buffer(), tempCacheKeyBuf.second,
                                     vkBn->getTensorSize(mTempCacheKey.get()));
            cmdBuffer->barrierSource(tempCacheValueBuf.first->buffer(), tempCacheValueBuf.second,
                                     vkBn->getTensorSize(mTempCacheValue.get()));
        }

        // 1) Rearrange Q to packed-D Qtmp: (x=qLen4, y=headDim/4, z=headNum)
        dispatchWithProfile(
            mUseFP16 ? "glsl_attention_prefill_rearrange_q_FP16_comp" : "glsl_attention_prefill_rearrange_q_comp",
            mRearrangeQPipeline, mRearrangeQSet, UP_DIV(mQueryLen4, 8), UP_DIV(mHeadDim / 4, 8), mHeadNum);
        cmdBuffer->barrierSource(tqBuf.first->buffer(), tqBuf.second, vkBn->getTensorSize(mTempQuery.get()));

        // K-block prefill: online softmax in K dimension to avoid O(qLen*totalLen) intermediates.
        dispatchWithProfile(mUseFP16 ? "glsl_attention_prefill_kblock_init_state_FP16_comp"
                                     : "glsl_attention_prefill_kblock_init_state_comp",
                            mInitStatePipeline, mInitStateSet,
                            UP_DIV((uint32_t)mQueryLen * (uint32_t)mHeadNum * (uint32_t)mHeadDim, 256), 1, 1);
        cmdBuffer->barrierSource(mBuf.first->buffer(), mBuf.second, vkBn->getTensorSize(mTempM.get()));
        cmdBuffer->barrierSource(lBuf.first->buffer(), lBuf.second, vkBn->getTensorSize(mTempL.get()));
        cmdBuffer->barrierSource(aBuf.first->buffer(), aBuf.second, vkBn->getTensorSize(mTempAlpha.get()));
        cmdBuffer->barrierSource(oBuf.first->buffer(), oBuf.second, vkBn->getTensorSize(mTempOAcc.get()));

        struct QKPushConst {
            uint32_t kStart;
            uint32_t blockLen;
        };
        struct SoftmaxPushConst {
            uint32_t kStart;
            uint32_t blockLen;
        };

        auto dispatchWithPushConst = [&](const char* name, const VulkanPipeline* pipeline,
                                         const std::shared_ptr<VulkanLayout::DescriptorSet>& set, uint32_t x,
                                         uint32_t y, uint32_t z, const void* pcData, uint32_t pcSize) {
#ifdef ENABLE_VULKAN_TIME_PROFILE
            auto* profiler = vkBn->timeProfiler();
            if (nullptr != profiler) {
                VulkanTimeProfileScope scope(profiler, cmd, name, VulkanTimeProfiler::Kind::Shader);
                pipeline->bind(cmd, set->get());
                vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, pcSize, pcData);
                vkCmdDispatch(cmd, x, y, z);
                return;
            }
#endif
            pipeline->bind(cmd, set->get());
            vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, pcSize, pcData);
            vkCmdDispatch(cmd, x, y, z);
        };

        const int totalLen = mPrefillTotalLen;
        const int kBlock = K_BLOCK;
        bool finalWritten = false;
        for (int kStart = 0; kStart < totalLen; kStart += kBlock) {
            const int blockLen = ALIMIN(kBlock, totalLen - kStart);
            const int blockLen4 = UP_DIV(blockLen, 4) * 4;
            const int blockLen4_4 = UP_DIV(blockLen4, 4);

            // 2) QK block: (x=blockLen4/4, y=qLen4/4, z=headNum)
            QKPushConst pcQK{(uint32_t)kStart, (uint32_t)blockLen};
            const bool fullBlock = (blockLen == kBlock) && (kStart + kBlock <= totalLen);
            const VulkanPipeline* qkPipe = fullBlock ? mQKBlockFullPipeline : mQKBlockPipeline;
            const std::shared_ptr<VulkanLayout::DescriptorSet>& qkSet = fullBlock ? mQKBlockFullSet : mQKBlockSet;
            const std::string qkName = _prefillKBlockShaderName(fullBlock ? "qk_full" : "qk", mUseFP16);
            dispatchWithPushConst(qkName.c_str(), qkPipe, qkSet, UP_DIV((uint32_t)blockLen4_4, 8),
                                  UP_DIV((uint32_t)UP_DIV(mQueryLen4, 4), 8), (uint32_t)mHeadNum, &pcQK, sizeof(pcQK));
            cmdBuffer->barrierSource(qkBuf.first->buffer(), qkBuf.second, vkBn->getTensorSize(mTempQKBlock.get()));

            // 3) Softmax online: updates m/l and writes unnormalized w (x=headNum, y=qLen)
            SoftmaxPushConst pcSM{(uint32_t)kStart, (uint32_t)blockLen};
            const std::string softmaxName = _prefillKBlockShaderName("softmax_online", mUseFP16);
            dispatchWithPushConst(softmaxName.c_str(), mSoftmaxOnlinePipeline, mSoftmaxOnlineSet, (uint32_t)mHeadNum,
                                  (uint32_t)mQueryLen, 1, &pcSM, sizeof(pcSM));
            cmdBuffer->barrierSource(wBuf.first->buffer(), wBuf.second, vkBn->getTensorSize(mTempWBlock.get()));
            cmdBuffer->barrierSource(mBuf.first->buffer(), mBuf.second, vkBn->getTensorSize(mTempM.get()));
            cmdBuffer->barrierSource(lBuf.first->buffer(), lBuf.second, vkBn->getTensorSize(mTempL.get()));
            cmdBuffer->barrierSource(aBuf.first->buffer(), aBuf.second, vkBn->getTensorSize(mTempAlpha.get()));

            // 4) QKV accumulate: (x=headDim/4, y=qLen/2, z=headNum)
            const bool finalBlock = (kStart + blockLen) >= totalLen;
            const bool finalFullBlock = finalBlock && fullBlock;
            const VulkanPipeline* qkvPipe =
                finalFullBlock ? mQKVAccFinalFullPipeline : (fullBlock ? mQKVAccFullPipeline : mQKVAccPipeline);
            const std::shared_ptr<VulkanLayout::DescriptorSet>& qkvSet =
                finalFullBlock ? mQKVAccFinalFullSet : (fullBlock ? mQKVAccFullSet : mQKVAccSet);
            const char* qkvKernel = finalFullBlock ? "qkv_acc_final_full" : (fullBlock ? "qkv_acc_full" : "qkv_acc");
            const std::string qkvName = _prefillKBlockShaderName(qkvKernel, mUseFP16);
            dispatchWithPushConst(qkvName.c_str(), qkvPipe, qkvSet, UP_DIV((uint32_t)(mHeadDim / 4), 8),
                                  UP_DIV((uint32_t)UP_DIV(mQueryLen, 2), 8), (uint32_t)mHeadNum, &pcQK, sizeof(pcQK));
            if (finalFullBlock) {
                finalWritten = true;
            } else {
                cmdBuffer->barrierSource(oBuf.first->buffer(), oBuf.second, vkBn->getTensorSize(mTempOAcc.get()));
            }
        }

        // 5) Finalize: output = oAcc / l
        if (!finalWritten) {
            dispatchWithProfile(mUseFP16 ? "glsl_attention_prefill_kblock_finalize_FP16_comp"
                                         : "glsl_attention_prefill_kblock_finalize_comp",
                                mFinalizePipeline, mFinalizeSet, UP_DIV((uint32_t)(mHeadDim / 4), 8),
                                UP_DIV((uint32_t)UP_DIV(mQueryLen, 2), 8), (uint32_t)mHeadNum);
        }

        auto releaseTemp = [&](std::shared_ptr<Tensor>& t) {
            if (t) {
                vkBn->onReleaseBuffer(t.get(), Backend::DYNAMIC);
                t.reset();
            }
        };
        releaseTemp(mTempQuery);
        releaseTemp(mTempQKBlock);
        releaseTemp(mTempWBlock);
        releaseTemp(mTempM);
        releaseTemp(mTempL);
        releaseTemp(mTempAlpha);
        releaseTemp(mTempOAcc);
        releaseTemp(mTempCacheKey);
        releaseTemp(mTempCacheValue);
        return NO_ERROR;
    }

    // Decode (or kv_cache disabled): keep fused shader.
    mQueryLen4 = 0;
    if (mTempQuery) {
        vkBn->onReleaseBuffer(mTempQuery.get(), Backend::DYNAMIC);
        mTempQuery.reset();
    }
    if (mTempQKBlock) {
        vkBn->onReleaseBuffer(mTempQKBlock.get(), Backend::DYNAMIC);
        mTempQKBlock.reset();
    }
    if (mTempWBlock) {
        vkBn->onReleaseBuffer(mTempWBlock.get(), Backend::DYNAMIC);
        mTempWBlock.reset();
    }
    if (mTempM) {
        vkBn->onReleaseBuffer(mTempM.get(), Backend::DYNAMIC);
        mTempM.reset();
    }
    if (mTempL) {
        vkBn->onReleaseBuffer(mTempL.get(), Backend::DYNAMIC);
        mTempL.reset();
    }
    if (mTempAlpha) {
        vkBn->onReleaseBuffer(mTempAlpha.get(), Backend::DYNAMIC);
        mTempAlpha.reset();
    }
    if (mTempOAcc) {
        vkBn->onReleaseBuffer(mTempOAcc.get(), Backend::DYNAMIC);
        mTempOAcc.reset();
    }
    if (mTempCacheKey) {
        vkBn->onReleaseBuffer(mTempCacheKey.get(), Backend::DYNAMIC);
        mTempCacheKey.reset();
    }
    if (mTempCacheValue) {
        vkBn->onReleaseBuffer(mTempCacheValue.get(), Backend::DYNAMIC);
        mTempCacheValue.reset();
    }
    mPrefillTotalLen = 0;

    if (mNeedKvCache) {
        if (!ensureDecodeSmallFusedPipelines(vkBn)) {
            return OUT_OF_MEMORY;
        }
        const int alignedIndex = _decodeHeadDimAlignedIndex(mHeadDim);
        const bool useSingleDecodeFused = (mQueryLen == 1) && (alignedIndex >= 0) &&
                                          (nullptr != mDecodeSingleFusedPipelines[alignedIndex]) &&
                                          (nullptr != mDecodeSingleFusedSets[alignedIndex]);
        const bool useSmallDecodeFused = (mQueryLen > 1 && mQueryLen <= 4) && (alignedIndex >= 0) &&
                                         (nullptr != mDecodeSmallFusedPipelines[alignedIndex]) &&
                                         (nullptr != mDecodeSmallFusedSets[alignedIndex]);
        if (useSingleDecodeFused) {
            const std::string profileName = _decodeSingleFusedShaderName(mUseFP16, alignedIndex);
            dispatchWithProfile(profileName.c_str(), mDecodeSingleFusedPipelines[alignedIndex],
                                mDecodeSingleFusedSets[alignedIndex], (uint32_t)mHeadNum, 1, 1);
        } else if (useSmallDecodeFused) {
            const std::string profileName = _decodeSmallFusedShaderName(mUseFP16, alignedIndex);
            dispatchWithProfile(profileName.c_str(), mDecodeSmallFusedPipelines[alignedIndex],
                                mDecodeSmallFusedSets[alignedIndex], (uint32_t)mHeadNum, (uint32_t)mQueryLen, 1);
        } else {
            if (!ensureAttentionPipeline(vkBn)) {
                return OUT_OF_MEMORY;
            }
            MNN_ASSERT(nullptr != mAttentionPipeline);
            MNN_ASSERT(nullptr != mAttentionSet);
            dispatchWithProfile(mUseFP16 ? "glsl_attention_fused_packed_FP16_comp" : "glsl_attention_fused_packed_comp",
                                mAttentionPipeline, mAttentionSet, UP_DIV(mHeadNum, 8), UP_DIV(mQueryLen, 8), 1);
        }
    } else {
        if (!ensureLegacyPipeline(vkBn)) {
            return OUT_OF_MEMORY;
        }
        MNN_ASSERT(nullptr != mAttentionLegacyPipeline);
        MNN_ASSERT(nullptr != mAttentionLegacySet);
        dispatchWithProfile(mUseFP16 ? "glsl_attention_fused_FP16_comp" : "glsl_attention_fused_comp",
                            mAttentionLegacyPipeline, mAttentionLegacySet, UP_DIV(mHeadNum, 8), UP_DIV(mQueryLen, 8),
                            1);
    }

    return NO_ERROR;
}

ErrorCode VulkanAttention::onBeforeExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    MNN_ASSERT(!inputs.empty());
    MNN_ASSERT(!outputs.empty());
    auto query = inputs[0];
    auto key = inputs[1];
    auto value = inputs[2];
    auto output = outputs[0];
    MNN_ASSERT(nullptr != query && nullptr != key && nullptr != value && nullptr != output);
    MNN_ASSERT(query->length(1) == mQueryLen);
    MNN_ASSERT(key->length(1) == mKeyLen);
    MNN_ASSERT(query->length(2) == mHeadNum);
    MNN_ASSERT(key->length(2) == mKvHeadNum);
    MNN_ASSERT(query->length(3) == mHeadDim);
    MNN_ASSERT(key->length(3) == mHeadDim);
    MNN_ASSERT(value->length(1) == mKeyLen);
    MNN_ASSERT(value->length(2) == mKvHeadNum);
    MNN_ASSERT(value->length(3) == mHeadDim);
    MNN_ASSERT(query->length(0) == 1);

    auto vkBn = static_cast<VulkanBackend*>(backend());

    int pastLenForCompute = 0;
    if (mNeedKvCache) {
        MNN_ASSERT(nullptr != mMeta);
        MNN_ASSERT(mMeta->n_reserve == 0);
        MNN_ASSERT(mMeta->computeReverseSize() == 0);
        const int previous = (int)mMeta->previous;
        const int remove = (int)mMeta->remove;
        const int add = (int)mMeta->add;
        MNN_ASSERT(previous >= 0);
        MNN_ASSERT(remove >= 0);
        MNN_ASSERT(add >= 0);
        MNN_ASSERT(add <= mKeyLen);
        MNN_ASSERT(remove <= previous);
        pastLenForCompute = previous - remove;
        // Ensure capacity for compute window (pastLen + keyLen), because shaders read only from KV cache.
        if (!mKVCache->ensureCapacity(vkBn, pastLenForCompute + mKeyLen, mKvHeadNum, mHeadDim, mUseFP16)) {
            return OUT_OF_MEMORY;
        }
    }

    const int group = mHeadNum / mKvHeadNum;
    const int totalLenForCompute = pastLenForCompute + mKeyLen;

    int maskMode = 0;
    int maskQlen = 0;
    int maskKvlen = 0;
    const Tensor* mask = nullptr;
    if (inputs.size() > 3 && nullptr != inputs[3]) {
        mask = inputs[3];
        MNN_ASSERT(mask->getType() == halide_type_of<float>());
        if (mask->shape().empty()) {
            // Match the transformer-fuse attention convention used by the tests:
            // a shape-empty scalar mask is the lower-triangular causal sentinel.
            maskMode = 2;
        } else {
            const int md = mask->dimensions();
            MNN_ASSERT(md >= 2);
            maskQlen = mask->length(md - 2);
            maskKvlen = mask->length(md - 1);
            MNN_ASSERT(maskQlen == mQueryLen);
            MNN_ASSERT(maskKvlen > 0);
            maskMode = 1;
        }
    }

    auto gpuParam = reinterpret_cast<GpuParam*>(mParam->map());
    gpuParam->s0[0] = mQueryLen;
    gpuParam->s0[1] = mKeyLen;
    gpuParam->s0[2] = mHeadNum;
    gpuParam->s0[3] = mKvHeadNum;
    gpuParam->s1[0] = mHeadDim;
    gpuParam->s1[1] = group;
    gpuParam->s1[2] = pastLenForCompute;
    gpuParam->s1[3] = totalLenForCompute;
    gpuParam->s2[0] = maskQlen;
    gpuParam->s2[1] = maskKvlen;
    gpuParam->s2[2] = maskMode;
    gpuParam->s2[3] = mNeedKvCache ? mKVCache->maxLen : (mUsePrefill ? mKeyLen : 0);
    gpuParam->f0[0] = (mAttnScale == 0.0f) ? _invSqrt((float)mHeadDim) : mAttnScale;
    gpuParam->f0[1] = mOutputC4 ? 1.0f : 0.0f;
    gpuParam->f0[2] = (mUseDecodeTwoStageIndirect || mUseDecodeTwoStageDirect) ? 1.0f : 0.0f;
    gpuParam->f0[3] = 0.0f;
    mParam->unmap();

    // Bind buffers (update + attention). Note: when hasMask == 0, bind query buffer as placeholder.
    auto queryBuf = vkBn->getTensorBuffer(query);
    auto keyBuf = vkBn->getTensorBuffer(key);
    auto valueBuf = vkBn->getTensorBuffer(value);
    auto outBuf = vkBn->getTensorBuffer(output);
    const VkDeviceSize queryOffset = queryBuf.second;

    const VulkanBuffer* cacheKeyBuf = nullptr;
    const VulkanBuffer* cacheValueBuf = nullptr;
    VkDeviceSize cacheKeyOffset = 0;
    VkDeviceSize cacheValueOffset = 0;
    size_t cacheKeySize = 0;
    size_t cacheValueSize = 0;

    if (mNeedKvCache) {
        cacheKeyBuf = mKVCache->key.get();
        cacheValueBuf = mKVCache->value.get();
        MNN_ASSERT(nullptr != cacheKeyBuf && nullptr != cacheValueBuf);
        cacheKeySize = cacheKeyBuf->size();
        cacheValueSize = cacheValueBuf->size();
    } else {
        // KV cache disabled: alias cache buffers to current K/V (shaders read only from cache bindings).
        cacheKeyBuf = keyBuf.first;
        cacheValueBuf = valueBuf.first;
        cacheKeyOffset = keyBuf.second;
        cacheValueOffset = valueBuf.second;
        cacheKeySize = vkBn->getTensorSize(key);
        cacheValueSize = vkBn->getTensorSize(value);
    }

    // Update set (only when KV cache is enabled; kv_cache=false uses legacy fused shader directly on input K/V).
    if (mNeedKvCache) {
        mUpdateSet->writeBuffer(keyBuf.first->buffer(), 0, vkBn->getTensorSize(key), keyBuf.second);
        mUpdateSet->writeBuffer(valueBuf.first->buffer(), 1, vkBn->getTensorSize(value), valueBuf.second);
        mUpdateSet->writeBuffer(cacheKeyBuf->buffer(), 2, cacheKeySize, cacheKeyOffset);
        mUpdateSet->writeBuffer(cacheValueBuf->buffer(), 3, cacheValueSize, cacheValueOffset);
        mUpdateSet->writeBuffer(mParam->buffer(), 4, mParam->size());
    }

    auto writeAttentionSet = [&](const std::shared_ptr<VulkanLayout::DescriptorSet>& set) {
        MNN_ASSERT(nullptr != set);
        set->writeBuffer(outBuf.first->buffer(), 0, vkBn->getTensorSize(output), outBuf.second);
        set->writeBuffer(queryBuf.first->buffer(), 1, vkBn->getTensorSize(query), queryBuf.second);
        set->writeBuffer(keyBuf.first->buffer(), 2, vkBn->getTensorSize(key), keyBuf.second);
        set->writeBuffer(valueBuf.first->buffer(), 3, vkBn->getTensorSize(value), valueBuf.second);
        set->writeBuffer(cacheKeyBuf->buffer(), 4, cacheKeySize, cacheKeyOffset);
        set->writeBuffer(cacheValueBuf->buffer(), 5, cacheValueSize, cacheValueOffset);
        if (maskMode == 1) {
            auto maskBuf = vkBn->getTensorBuffer(mask);
            set->writeBuffer(maskBuf.first->buffer(), 6, vkBn->getTensorSize(mask), maskBuf.second);
        } else {
            set->writeBuffer(queryBuf.first->buffer(), 6, vkBn->getTensorSize(query), queryBuf.second);
        }
        set->writeBuffer(mParam->buffer(), 7, mParam->size());
    };

    auto writeDecodeTwoStageSet = [&](int alignedIndex) {
        MNN_ASSERT(nullptr != mTempDecodeSoftmax);
        MNN_ASSERT(alignedIndex >= 0);
        auto softmaxBuf = vkBn->getTensorBuffer(mTempDecodeSoftmax.get());
        const size_t softmaxSize = vkBn->getTensorSize(mTempDecodeSoftmax.get());

        mDecodeQkSoftmaxSets[alignedIndex]->writeBuffer(softmaxBuf.first->buffer(), 0, softmaxSize, softmaxBuf.second);
        mDecodeQkSoftmaxSets[alignedIndex]->writeBuffer(queryBuf.first->buffer(), 1, vkBn->getTensorSize(query),
                                                        queryBuf.second);
        mDecodeQkSoftmaxSets[alignedIndex]->writeBuffer(cacheKeyBuf->buffer(), 2, cacheKeySize, cacheKeyOffset);
        mDecodeQkSoftmaxSets[alignedIndex]->writeBuffer(mParam->buffer(), 3, mParam->size());

        mDecodeQkvSet->writeBuffer(outBuf.first->buffer(), 0, vkBn->getTensorSize(output), outBuf.second);
        mDecodeQkvSet->writeBuffer(softmaxBuf.first->buffer(), 1, softmaxSize, softmaxBuf.second);
        mDecodeQkvSet->writeBuffer(cacheValueBuf->buffer(), 2, cacheValueSize, cacheValueOffset);
        mDecodeQkvSet->writeBuffer(mParam->buffer(), 3, mParam->size());
    };

    auto writeDecodeTwoStageIndirect = [&](bool twoStageActive) {
        if (!mUseDecodeTwoStageIndirect) {
            return;
        }
        if (mDecodeIndirectCmdInitialized && mDecodeIndirectLastActive == twoStageActive) {
            return;
        }
        MNN_ASSERT(nullptr != mDecodeIndirectBuffer);
        auto* cmds = reinterpret_cast<VkDispatchIndirectCommand*>(mDecodeIndirectBuffer->map());
        for (int i = 0; i < mDecodeIndirectCmdCount; ++i) {
            cmds[i].x = 0;
            cmds[i].y = 0;
            cmds[i].z = 0;
        }
        if (twoStageActive) {
            cmds[kAttentionDecodeIndirectTwoStageQkSlot].x = (uint32_t)mKvHeadNum;
            cmds[kAttentionDecodeIndirectTwoStageQkSlot].y = 1;
            cmds[kAttentionDecodeIndirectTwoStageQkSlot].z = 1;
            cmds[kAttentionDecodeIndirectTwoStageQkvSlot].x =
                (uint32_t)UP_DIV(mHeadDim / 4, kAttentionDecodeTwoStageQkvD4Pack);
            cmds[kAttentionDecodeIndirectTwoStageQkvSlot].y = (uint32_t)mHeadNum;
            cmds[kAttentionDecodeIndirectTwoStageQkvSlot].z = 1;
        } else {
            cmds[kAttentionDecodeIndirectFusedSlot].x = (uint32_t)mHeadNum;
            cmds[kAttentionDecodeIndirectFusedSlot].y = 1;
            cmds[kAttentionDecodeIndirectFusedSlot].z = 1;
        }
        mDecodeIndirectBuffer->unmap();
        mDecodeIndirectCmdInitialized = true;
        mDecodeIndirectLastActive = twoStageActive;
    };

    if (mUseDecodeTwoStageIndirect) {
        const int alignedIndex = _decodeHeadDimAlignedIndex(mHeadDim);
        MNN_ASSERT(alignedIndex >= 0);
        writeAttentionSet(mDecodeSingleFusedSets[alignedIndex]);
        writeDecodeTwoStageSet(alignedIndex);

        const bool useTwoStage = pastLenForCompute >= kAttentionDecodeTwoStageMinPastLen &&
                                 totalLenForCompute <= kAttentionDecodeTwoStageMaxLen;
        writeDecodeTwoStageIndirect(useTwoStage);
        return NO_ERROR;
    }

    if (mUseDecodeTwoStageDirect) {
        const int alignedIndex = _decodeHeadDimAlignedIndex(mHeadDim);
        MNN_ASSERT(alignedIndex >= 0);
        writeDecodeTwoStageSet(alignedIndex);
        return NO_ERROR;
    }

    if (mUsePrefill) {
        MNN_ASSERT(totalLenForCompute == mPrefillTotalLen);
        MNN_ASSERT(mQueryLen4 == UP_DIV(mQueryLen, 4) * 4);

        // Only the bindings that may change between executes are rewritten below.

        MNN_ASSERT(nullptr != mRearrangeQSet);
        mRearrangeQSet->writeBuffer(queryBuf.first->buffer(), 1, vkBn->getTensorSize(query), queryBuf.second);

        if (mNeedKvCache) {
            mQKBlockSet->writeBuffer(cacheKeyBuf->buffer(), 2, cacheKeySize, cacheKeyOffset);
            mQKBlockFullSet->writeBuffer(cacheKeyBuf->buffer(), 2, cacheKeySize, cacheKeyOffset);
        }
        if (maskMode == 1) {
            auto maskBuf = vkBn->getTensorBuffer(mask);
            mQKBlockSet->writeBuffer(maskBuf.first->buffer(), 3, vkBn->getTensorSize(mask), maskBuf.second);
            mQKBlockFullSet->writeBuffer(maskBuf.first->buffer(), 3, vkBn->getTensorSize(mask), maskBuf.second);
        } else {
            mQKBlockSet->writeBuffer(queryBuf.first->buffer(), 3, vkBn->getTensorSize(query), queryBuf.second);
            mQKBlockFullSet->writeBuffer(queryBuf.first->buffer(), 3, vkBn->getTensorSize(query), queryBuf.second);
        }

        if (mNeedKvCache) {
            mQKVAccSet->writeBuffer(cacheValueBuf->buffer(), 2, cacheValueSize, cacheValueOffset);
            mQKVAccFullSet->writeBuffer(cacheValueBuf->buffer(), 2, cacheValueSize, cacheValueOffset);
            mQKVAccFinalFullSet->writeBuffer(cacheValueBuf->buffer(), 3, cacheValueSize, cacheValueOffset);
        }

        mFinalizeSet->writeBuffer(outBuf.first->buffer(), 0, vkBn->getTensorSize(output), outBuf.second);
        mQKVAccFinalFullSet->writeBuffer(outBuf.first->buffer(), 0, vkBn->getTensorSize(output), outBuf.second);
        return NO_ERROR;
    }

    // Attention set (fused). Small decode subgroup uses the same descriptor layout as packed fused fallback.
    if (mNeedKvCache) {
        if (mQueryLen <= 4) {
            const int alignedIndex = _decodeHeadDimAlignedIndex(mHeadDim);
            if (mQueryLen == 1 && alignedIndex >= 0 && nullptr != mDecodeSingleFusedSets[alignedIndex]) {
                writeAttentionSet(mDecodeSingleFusedSets[alignedIndex]);
            } else if (alignedIndex >= 0 && nullptr != mDecodeSmallFusedSets[alignedIndex]) {
                writeAttentionSet(mDecodeSmallFusedSets[alignedIndex]);
            } else if (nullptr != mAttentionSet) {
                writeAttentionSet(mAttentionSet);
            }
        } else if (nullptr != mAttentionSet) {
            writeAttentionSet(mAttentionSet);
        }
    } else {
        MNN_ASSERT(nullptr != mAttentionLegacySet);
        writeAttentionSet(mAttentionLegacySet);
    }

    return NO_ERROR;
}

class VulkanAttentionCreator : public VulkanBackend::Creator {
public:
    VulkanBasicExecution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                   const MNN::Op* op, Backend* backend) const override {
        return new VulkanAttention(op, backend);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_Attention, new VulkanAttentionCreator);
    return true;
}();

} // namespace MNN

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
