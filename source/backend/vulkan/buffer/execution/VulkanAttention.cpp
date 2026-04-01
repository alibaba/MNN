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

static bool _supportDecodeQ1Subgroup(const VulkanDevice& device) {
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

void VulkanAttention::KVCache::reset() {
    maxLen = 0;
    kvHeadNum = 0;
    headDim = 0;
    fp16 = false;
    key = nullptr;
    value = nullptr;
}

void VulkanAttention::KVCache::ensureCapacity(VulkanBackend* vkBn, int requiredLen, int kvH, int d, bool useFP16) {
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
        return;
    }
    if (requiredLen <= maxLen) {
        return;
    }
    const int oldMaxLen = maxLen;
    maxLen = requiredLen + expandChunk;
    const size_t bytes = fp16 ? sizeof(uint16_t) : sizeof(float);
    const size_t newSize = (size_t)maxLen * (size_t)kvHeadNum * (size_t)headDim * bytes;
    std::shared_ptr<VulkanBuffer> newKey(new VulkanBuffer(vkBn->getMemoryPool(), false, newSize, nullptr,
                                                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                              VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                                              VK_BUFFER_USAGE_TRANSFER_DST_BIT));
    std::shared_ptr<VulkanBuffer> newValue(new VulkanBuffer(vkBn->getMemoryPool(), false, newSize, nullptr,
                                                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                                VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                                                VK_BUFFER_USAGE_TRANSFER_DST_BIT));
    // Preserve old content.
    //
    // cacheKey is packed as [kvHeadNum, headDim/4, maxLen, 4], so changing maxLen changes the row stride and we must repack.
    // cacheValue is kvh-major as [kvHeadNum, maxLen, headDim], so changing maxLen changes the kvh stride and we must repack too.
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
            vkBn->copyGPUToGPUBufferRegions(value->buffer(), newValue->buffer(), regions.data(), (uint32_t)regions.size());
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
}

VulkanAttention::VulkanAttention(const Op* op, Backend* bn) : VulkanBasicExecution(bn), mOp(op) {
    auto vkBn = static_cast<VulkanBackend*>(bn);
    mUseFP16 = vkBn->useFP16();
    mMeta = reinterpret_cast<KVMeta*>(vkBn->getMetaPtr());
    if (nullptr != op && nullptr != op->main_as_AttentionParam()) {
        mNeedKvCache = op->main_as_AttentionParam()->kv_cache();
    }
    mKVCache.reset(new KVCache);
    mParam = vkBn->allocUniform(nullptr, sizeof(GpuParam));
    if (!mNeedKvCache) {
        std::vector<VkDescriptorType> typesAttn{
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // output
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // query
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // keyIn
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // valueIn
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // cacheKey
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // cacheValue
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // mask
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER  // param
        };
        std::string attnName = "glsl_attention_fused_";
        if (mUseFP16) {
            attnName += "FP16_";
        }
        attnName += "comp";
        mAttentionLegacyPipeline = vkBn->getPipeline(attnName, typesAttn);
        MNN_ASSERT(nullptr != mAttentionLegacyPipeline);
        mAttentionLegacySet.reset(mAttentionLegacyPipeline->createSet());
        return;
    }

    // kv_cache=true path: pre-create update/prefill/decode pipelines to avoid resize cold-start.
    {
        std::vector<VkDescriptorType> typesUpdate{
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // keyIn
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // valueIn
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // cacheKey
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // cacheValue
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER  // param
        };
        std::string updateName = "glsl_attention_kvcache_update_";
        if (mUseFP16) {
            updateName += "FP16_";
        }
        updateName += "comp";
        mUpdatePipeline = vkBn->getPipeline(updateName, typesUpdate);
        MNN_ASSERT(nullptr != mUpdatePipeline);
        mUpdateSet.reset(mUpdatePipeline->createSet());
    }

    {
        std::vector<VkDescriptorType> typesRearrange{
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // queryOut
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // queryIn
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER  // param
        };
        std::string rqName = "glsl_attention_prefill_rearrange_q_";
        if (mUseFP16) {
            rqName += "FP16_";
        }
        rqName += "comp";
        mRearrangeQPipeline = vkBn->getPipeline(rqName, typesRearrange);
        MNN_ASSERT(nullptr != mRearrangeQPipeline);
        mRearrangeQSet.reset(mRearrangeQPipeline->createSet());
    }

    {
        std::vector<VkDescriptorType> typesInit{
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // m
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // l
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // alpha
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // oAcc
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER  // param
        };
        std::string initName = "glsl_attention_prefill_kblock_init_state_";
        if (mUseFP16) {
            initName += "FP16_";
        }
        initName += "comp";
        mInitStatePipeline = vkBn->getPipeline(initName, typesInit);
        MNN_ASSERT(nullptr != mInitStatePipeline);
        mInitStateSet.reset(mInitStatePipeline->createSet());
    }

    {
        std::vector<VkDescriptorType> typesQK{
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // qk
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // query
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // cacheKey
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // mask
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER  // param
        };

        std::string qkName = "glsl_attention_prefill_kblock_qk_";
        if (mUseFP16) {
            qkName += "FP16_";
        }
        qkName += "comp";
        mQKBlockPipeline = vkBn->getPipeline(qkName, typesQK);
        MNN_ASSERT(nullptr != mQKBlockPipeline);
        mQKBlockSet.reset(mQKBlockPipeline->createSet());

        std::string qkFullName = "glsl_attention_prefill_kblock_qk_full_";
        if (mUseFP16) {
            qkFullName += "FP16_";
        }
        qkFullName += "comp";
        mQKBlockFullPipeline = vkBn->getPipeline(qkFullName, typesQK);
        MNN_ASSERT(nullptr != mQKBlockFullPipeline);
        mQKBlockFullSet.reset(mQKBlockFullPipeline->createSet());
    }

    {
        std::vector<VkDescriptorType> typesSoftmax{
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // w
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // qk
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // m
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // l
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // alpha
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER  // param
        };
        std::string softmaxName = "glsl_attention_prefill_kblock_softmax_online_";
        if (mUseFP16) {
            softmaxName += "FP16_";
        }
        softmaxName += "comp";
        const auto& limits = vkBn->getDevice().proty().limits;
        const int kBlock4 = UP_DIV(kAttentionPrefillKBlock, 4) * 4;
        const int maxK4 = UP_DIV(kBlock4, 4);
        uint32_t localSize = _selectSoftmaxLocalSize(maxK4, (uint32_t)limits.maxComputeWorkGroupSize[0],
                                                      (uint32_t)limits.maxComputeWorkGroupInvocations);
        mSoftmaxOnlinePipeline = vkBn->getPipeline(softmaxName, typesSoftmax, {localSize});
        MNN_ASSERT(nullptr != mSoftmaxOnlinePipeline);
        mSoftmaxOnlineSet.reset(mSoftmaxOnlinePipeline->createSet());
        mSoftmaxOnlineLocalSize = localSize;
    }

    {
        std::vector<VkDescriptorType> typesQKV{
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // oAcc
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // w
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // cacheValue
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // alpha
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER  // param
        };
        std::string qkvName = "glsl_attention_prefill_kblock_qkv_acc_";
        if (mUseFP16) {
            qkvName += "FP16_";
        }
        qkvName += "comp";
        mQKVAccPipeline = vkBn->getPipeline(qkvName, typesQKV);
        MNN_ASSERT(nullptr != mQKVAccPipeline);
        mQKVAccSet.reset(mQKVAccPipeline->createSet());

        std::string qkvFullName = "glsl_attention_prefill_kblock_qkv_acc_full_";
        if (mUseFP16) {
            qkvFullName += "FP16_";
        }
        qkvFullName += "comp";
        mQKVAccFullPipeline = vkBn->getPipeline(qkvFullName, typesQKV);
        MNN_ASSERT(nullptr != mQKVAccFullPipeline);
        mQKVAccFullSet.reset(mQKVAccFullPipeline->createSet());
    }

    {
        std::vector<VkDescriptorType> typesFinal{
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // output
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // oAcc
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // l
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER  // param
        };
        std::string finalName = "glsl_attention_prefill_kblock_finalize_";
        if (mUseFP16) {
            finalName += "FP16_";
        }
        finalName += "comp";
        mFinalizePipeline = vkBn->getPipeline(finalName, typesFinal);
        MNN_ASSERT(nullptr != mFinalizePipeline);
        mFinalizeSet.reset(mFinalizePipeline->createSet());
    }

    {
        std::vector<VkDescriptorType> typesAttn{
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // output
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // query
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // keyIn
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // valueIn
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // cacheKey
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // cacheValue
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // mask
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER  // param
        };
        std::string attnName = "glsl_attention_fused_packed_";
        if (mUseFP16) {
            attnName += "FP16_";
        }
        attnName += "comp";
        mAttentionPipeline = vkBn->getPipeline(attnName, typesAttn);
        MNN_ASSERT(nullptr != mAttentionPipeline);
        mAttentionSet.reset(mAttentionPipeline->createSet());

        if (_supportDecodeQ1Subgroup(vkBn->getDevice())) {
            mDecodeQ1SubgroupLocalSize = vkBn->getDevice().getSubgroupSize();
            if (mDecodeQ1SubgroupLocalSize > 0) {
                std::string decodeQ1Name = "glsl_attention_decode_q1_subgroup_";
                if (mUseFP16) {
                    decodeQ1Name += "FP16_";
                }
                decodeQ1Name += "comp";
                mDecodeQ1SubgroupPipeline = vkBn->getPipeline(decodeQ1Name, typesAttn, {mDecodeQ1SubgroupLocalSize});
                if (nullptr != mDecodeQ1SubgroupPipeline) {
                    mDecodeQ1SubgroupSet.reset(mDecodeQ1SubgroupPipeline->createSet());
                }

                std::string decodeQ1HD128Name = "glsl_attention_decode_q1_subgroup_hd128_";
                if (mUseFP16) {
                    decodeQ1HD128Name += "FP16_";
                }
                decodeQ1HD128Name += "comp";
                mDecodeQ1SubgroupHD128Pipeline = vkBn->getPipeline(decodeQ1HD128Name, typesAttn, {mDecodeQ1SubgroupLocalSize});
                if (nullptr != mDecodeQ1SubgroupHD128Pipeline) {
                    mDecodeQ1SubgroupHD128Set.reset(mDecodeQ1SubgroupHD128Pipeline->createSet());
                }
            }
        }
    }
}

VulkanAttention::~VulkanAttention() {
    auto vkBn = static_cast<VulkanBackend*>(backend());
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
    vkBn->recycleUniform(mParam);
}

bool VulkanAttention::onClone(Backend* bn, const Op* op, VulkanBasicExecution** dst) {
    if (nullptr == dst) {
        return true;
    }
    auto res = new VulkanAttention(op, bn);
    res->mKVCache = mKVCache;
    res->mMeta = mMeta;
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

    const bool usePrefill = mNeedKvCache && mQueryLen > 1;
    mUsePrefill = usePrefill;

    if (mNeedKvCache) {
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

    if (usePrefill) {
        constexpr int K_BLOCK = kAttentionPrefillKBlock;
        int pastLenForPrefill = 0;
        if (mNeedKvCache) {
            MNN_ASSERT(nullptr != mMeta);
            MNN_ASSERT(mMeta->n_reserve == 0);
            MNN_ASSERT(mMeta->computeReverseSize() == 0);
            const int previous = (int)mMeta->previous;
            const int remove = (int)mMeta->remove;
            MNN_ASSERT(previous >= 0);
            MNN_ASSERT(remove >= 0);
            MNN_ASSERT(remove <= previous);
            pastLenForPrefill = previous - remove;
        }
        mPrefillTotalLen = pastLenForPrefill + mKeyLen;
        mQueryLen4 = UP_DIV(mQueryLen, 4) * 4;
        MNN_ASSERT(mPrefillTotalLen > 0);

        const int64_t queryElementsI64 = (int64_t)mHeadNum * (int64_t)mHeadDim * (int64_t)mQueryLen4;
        MNN_ASSERT(queryElementsI64 > 0 && queryElementsI64 <= (int64_t)INT_MAX);
        const int queryElements = (int)queryElementsI64;

        if (!mTempQuery || (size_t)mTempQuery->elementSize() != (size_t)queryElements) {
            if (mTempQuery) {
                vkBn->onReleaseBuffer(mTempQuery.get(), Backend::DYNAMIC);
                mTempQuery.reset();
            }
            mTempQuery.reset(Tensor::createDevice<float>({queryElements}));
            bool res = vkBn->onAcquireBuffer(mTempQuery.get(), Backend::DYNAMIC);
            if (!res) {
                return OUT_OF_MEMORY;
            }
        }

        MNN_ASSERT(nullptr != mRearrangeQPipeline);
        MNN_ASSERT(nullptr != mRearrangeQSet);

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

        if (!mTempQKBlock || (size_t)mTempQKBlock->elementSize() != (size_t)qkElements) {
            if (mTempQKBlock) {
                vkBn->onReleaseBuffer(mTempQKBlock.get(), Backend::DYNAMIC);
                mTempQKBlock.reset();
            }
            mTempQKBlock.reset(Tensor::createDevice<float>({qkElements}));
            if (!vkBn->onAcquireBuffer(mTempQKBlock.get(), Backend::DYNAMIC)) {
                return OUT_OF_MEMORY;
            }
        }
        if (!mTempWBlock || (size_t)mTempWBlock->elementSize() != (size_t)qkElements) {
            if (mTempWBlock) {
                vkBn->onReleaseBuffer(mTempWBlock.get(), Backend::DYNAMIC);
                mTempWBlock.reset();
            }
            mTempWBlock.reset(Tensor::createDevice<float>({qkElements}));
            if (!vkBn->onAcquireBuffer(mTempWBlock.get(), Backend::DYNAMIC)) {
                return OUT_OF_MEMORY;
            }
        }

        // State buffers must be FP32 even when VulkanBackend runs in FP16 mode. Use int tensors to force 4-byte storage.
        if (!mTempM || (size_t)mTempM->elementSize() != (size_t)rowCount) {
            if (mTempM) {
                vkBn->onReleaseBuffer(mTempM.get(), Backend::DYNAMIC);
                mTempM.reset();
            }
            mTempM.reset(Tensor::createDevice<int>({rowCount}));
            if (!vkBn->onAcquireBuffer(mTempM.get(), Backend::DYNAMIC)) {
                return OUT_OF_MEMORY;
            }
        }
        if (!mTempL || (size_t)mTempL->elementSize() != (size_t)rowCount) {
            if (mTempL) {
                vkBn->onReleaseBuffer(mTempL.get(), Backend::DYNAMIC);
                mTempL.reset();
            }
            mTempL.reset(Tensor::createDevice<int>({rowCount}));
            if (!vkBn->onAcquireBuffer(mTempL.get(), Backend::DYNAMIC)) {
                return OUT_OF_MEMORY;
            }
        }
        if (!mTempAlpha || (size_t)mTempAlpha->elementSize() != (size_t)rowCount) {
            if (mTempAlpha) {
                vkBn->onReleaseBuffer(mTempAlpha.get(), Backend::DYNAMIC);
                mTempAlpha.reset();
            }
            mTempAlpha.reset(Tensor::createDevice<int>({rowCount}));
            if (!vkBn->onAcquireBuffer(mTempAlpha.get(), Backend::DYNAMIC)) {
                return OUT_OF_MEMORY;
            }
        }
        if (!mTempOAcc || (size_t)mTempOAcc->elementSize() != (size_t)oaccElements) {
            if (mTempOAcc) {
                vkBn->onReleaseBuffer(mTempOAcc.get(), Backend::DYNAMIC);
                mTempOAcc.reset();
            }
            mTempOAcc.reset(Tensor::createDevice<int>({oaccElements}));
            if (!vkBn->onAcquireBuffer(mTempOAcc.get(), Backend::DYNAMIC)) {
                return OUT_OF_MEMORY;
            }
        }

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

        MNN_ASSERT(nullptr != mFinalizePipeline);
        MNN_ASSERT(nullptr != mFinalizeSet);

        // 1) Rearrange Q to packed-D Qtmp: (x=qLen4, y=headDim/4, z=headNum)
        dispatchWithProfile(mUseFP16 ? "glsl_attention_prefill_rearrange_q_FP16_comp" : "glsl_attention_prefill_rearrange_q_comp",
                            mRearrangeQPipeline, mRearrangeQSet, UP_DIV(mQueryLen4, 8), UP_DIV(mHeadDim / 4, 8), mHeadNum);
        {
            auto qBuf = vkBn->getTensorBuffer(mTempQuery.get());
            cmdBuffer->barrierSource(qBuf.first->buffer(), qBuf.second, vkBn->getTensorSize(mTempQuery.get()));
        }

        // K-block prefill: online softmax in K dimension to avoid O(qLen*totalLen) intermediates.
        auto stateMBuf = vkBn->getTensorBuffer(mTempM.get());
        auto stateLBuf = vkBn->getTensorBuffer(mTempL.get());
        auto stateABuf = vkBn->getTensorBuffer(mTempAlpha.get());
        auto oaccBuf = vkBn->getTensorBuffer(mTempOAcc.get());

        dispatchWithProfile(mUseFP16 ? "glsl_attention_prefill_kblock_init_state_FP16_comp" : "glsl_attention_prefill_kblock_init_state_comp",
                            mInitStatePipeline, mInitStateSet, UP_DIV((uint32_t)mQueryLen * (uint32_t)mHeadNum * (uint32_t)mHeadDim, 256),
                            1, 1);
        cmdBuffer->barrierSource(stateMBuf.first->buffer(), stateMBuf.second, vkBn->getTensorSize(mTempM.get()));
        cmdBuffer->barrierSource(stateLBuf.first->buffer(), stateLBuf.second, vkBn->getTensorSize(mTempL.get()));
        cmdBuffer->barrierSource(stateABuf.first->buffer(), stateABuf.second, vkBn->getTensorSize(mTempAlpha.get()));
        cmdBuffer->barrierSource(oaccBuf.first->buffer(), oaccBuf.second, vkBn->getTensorSize(mTempOAcc.get()));

        struct QKPushConst {
            uint32_t kStart;
            uint32_t blockLen;
        };
        struct SoftmaxPushConst {
            uint32_t blockLen;
        };

        auto dispatchWithPushConst = [&](const char* name, const VulkanPipeline* pipeline,
                                         const std::shared_ptr<VulkanLayout::DescriptorSet>& set, uint32_t x, uint32_t y,
                                         uint32_t z, const void* pcData, uint32_t pcSize) {
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
        for (int kStart = 0; kStart < totalLen; kStart += kBlock) {
            const int blockLen = ALIMIN(kBlock, totalLen - kStart);
            const int blockLen4 = UP_DIV(blockLen, 4) * 4;
            const int blockLen4_4 = UP_DIV(blockLen4, 4);

            // 2) QK block: (x=blockLen4/4, y=qLen4/4, z=headNum)
            QKPushConst pcQK{(uint32_t)kStart, (uint32_t)blockLen};
            const bool fullBlock = (blockLen == kBlock) && (kStart + kBlock <= totalLen);
            const VulkanPipeline* qkPipe = fullBlock ? mQKBlockFullPipeline : mQKBlockPipeline;
            const std::shared_ptr<VulkanLayout::DescriptorSet>& qkSet = fullBlock ? mQKBlockFullSet : mQKBlockSet;
            const char* qkName = nullptr;
            if (fullBlock) {
                qkName = mUseFP16 ? "glsl_attention_prefill_kblock_qk_full_FP16_comp" : "glsl_attention_prefill_kblock_qk_full_comp";
            } else {
                qkName = mUseFP16 ? "glsl_attention_prefill_kblock_qk_FP16_comp" : "glsl_attention_prefill_kblock_qk_comp";
            }
            dispatchWithPushConst(qkName, qkPipe, qkSet, UP_DIV((uint32_t)blockLen4_4, 8),
                                  UP_DIV((uint32_t)UP_DIV(mQueryLen4, 4), 8), (uint32_t)mHeadNum, &pcQK, sizeof(pcQK));
            {
                auto qkBuf = vkBn->getTensorBuffer(mTempQKBlock.get());
                cmdBuffer->barrierSource(qkBuf.first->buffer(), qkBuf.second, vkBn->getTensorSize(mTempQKBlock.get()));
            }

            // 3) Softmax online: updates m/l and writes unnormalized w (x=headNum, y=qLen)
            SoftmaxPushConst pcSM{(uint32_t)blockLen};
            dispatchWithPushConst(mUseFP16 ? "glsl_attention_prefill_kblock_softmax_online_FP16_comp"
                                           : "glsl_attention_prefill_kblock_softmax_online_comp",
                                  mSoftmaxOnlinePipeline, mSoftmaxOnlineSet, (uint32_t)mHeadNum, (uint32_t)mQueryLen, 1, &pcSM,
                                  sizeof(pcSM));
            {
                auto wBuf = vkBn->getTensorBuffer(mTempWBlock.get());
                cmdBuffer->barrierSource(wBuf.first->buffer(), wBuf.second, vkBn->getTensorSize(mTempWBlock.get()));
                cmdBuffer->barrierSource(stateMBuf.first->buffer(), stateMBuf.second, vkBn->getTensorSize(mTempM.get()));
                cmdBuffer->barrierSource(stateLBuf.first->buffer(), stateLBuf.second, vkBn->getTensorSize(mTempL.get()));
                cmdBuffer->barrierSource(stateABuf.first->buffer(), stateABuf.second, vkBn->getTensorSize(mTempAlpha.get()));
            }

            // 4) QKV accumulate: (x=headDim/4, y=qLen/2, z=headNum)
            const VulkanPipeline* qkvPipe = fullBlock ? mQKVAccFullPipeline : mQKVAccPipeline;
            const std::shared_ptr<VulkanLayout::DescriptorSet>& qkvSet = fullBlock ? mQKVAccFullSet : mQKVAccSet;
            const char* qkvName = nullptr;
            if (fullBlock) {
                qkvName = mUseFP16 ? "glsl_attention_prefill_kblock_qkv_acc_full_FP16_comp"
                                   : "glsl_attention_prefill_kblock_qkv_acc_full_comp";
            } else {
                qkvName =
                    mUseFP16 ? "glsl_attention_prefill_kblock_qkv_acc_FP16_comp" : "glsl_attention_prefill_kblock_qkv_acc_comp";
            }
            dispatchWithPushConst(qkvName, qkvPipe, qkvSet, UP_DIV((uint32_t)(mHeadDim / 4), 8),
                                  UP_DIV((uint32_t)UP_DIV(mQueryLen, 2), 8), (uint32_t)mHeadNum, &pcQK, sizeof(pcQK));
            cmdBuffer->barrierSource(oaccBuf.first->buffer(), oaccBuf.second, vkBn->getTensorSize(mTempOAcc.get()));
        }

        // 5) Finalize: output = oAcc / l
        dispatchWithProfile(mUseFP16 ? "glsl_attention_prefill_kblock_finalize_FP16_comp" : "glsl_attention_prefill_kblock_finalize_comp",
                            mFinalizePipeline, mFinalizeSet, UP_DIV((uint32_t)(mHeadDim / 4), 8),
                            UP_DIV((uint32_t)UP_DIV(mQueryLen, 2), 8), (uint32_t)mHeadNum);
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
    mPrefillTotalLen = 0;

    if (mNeedKvCache) {
        const bool useDecodeQ1Subgroup =
            (mQueryLen == 1) && (nullptr != mDecodeQ1SubgroupPipeline) && (nullptr != mDecodeQ1SubgroupSet);
        if (useDecodeQ1Subgroup) {
            const bool useHD128 = (mHeadDim == 128) && (nullptr != mDecodeQ1SubgroupHD128Pipeline) &&
                                  (nullptr != mDecodeQ1SubgroupHD128Set);
            if (useHD128) {
                dispatchWithProfile(mUseFP16 ? "glsl_attention_decode_q1_subgroup_hd128_FP16_comp"
                                             : "glsl_attention_decode_q1_subgroup_hd128_comp",
                                    mDecodeQ1SubgroupHD128Pipeline, mDecodeQ1SubgroupHD128Set, (uint32_t)mHeadNum, 1, 1);
            } else {
                dispatchWithProfile(mUseFP16 ? "glsl_attention_decode_q1_subgroup_FP16_comp"
                                             : "glsl_attention_decode_q1_subgroup_comp",
                                    mDecodeQ1SubgroupPipeline, mDecodeQ1SubgroupSet, (uint32_t)mHeadNum, 1, 1);
            }
        } else {
            MNN_ASSERT(nullptr != mAttentionPipeline);
            MNN_ASSERT(nullptr != mAttentionSet);
            dispatchWithProfile(mUseFP16 ? "glsl_attention_fused_packed_FP16_comp" : "glsl_attention_fused_packed_comp",
                                mAttentionPipeline, mAttentionSet, UP_DIV(mHeadNum, 8), UP_DIV(mQueryLen, 8), 1);
        }
    } else {
        MNN_ASSERT(nullptr != mAttentionLegacyPipeline);
        MNN_ASSERT(nullptr != mAttentionLegacySet);
        dispatchWithProfile(mUseFP16 ? "glsl_attention_fused_FP16_comp" : "glsl_attention_fused_comp",
                            mAttentionLegacyPipeline, mAttentionLegacySet, UP_DIV(mHeadNum, 8), UP_DIV(mQueryLen, 8), 1);
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
        mKVCache->ensureCapacity(vkBn, pastLenForCompute + mKeyLen, mKvHeadNum, mHeadDim, mUseFP16);
    }

    const int group = mHeadNum / mKvHeadNum;
    const int totalLenForCompute = pastLenForCompute + mKeyLen;

    int hasMask = 0;
    int maskQlen = 0;
    int maskKvlen = 0;
    const Tensor* mask = nullptr;
    if (inputs.size() > 3 && nullptr != inputs[3]) {
        mask = inputs[3];
        hasMask = 1;
        MNN_ASSERT(mask->getType() == halide_type_of<float>());
        const int md = mask->dimensions();
        MNN_ASSERT(md >= 2);
        maskQlen = mask->length(md - 2);
        maskKvlen = mask->length(md - 1);
        MNN_ASSERT(maskQlen == mQueryLen);
        MNN_ASSERT(maskKvlen > 0);
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
    gpuParam->s2[2] = hasMask;
    gpuParam->s2[3] = mNeedKvCache ? mKVCache->maxLen : 0;
    gpuParam->f0[0] = _invSqrt((float)mHeadDim);
    gpuParam->f0[1] = 0.0f;
    gpuParam->f0[2] = 0.0f;
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

    if (mUsePrefill) {
        MNN_ASSERT(totalLenForCompute == mPrefillTotalLen);
        MNN_ASSERT(mQueryLen4 == UP_DIV(mQueryLen, 4) * 4);

        MNN_ASSERT(nullptr != mTempQuery);
        auto tqBuf = vkBn->getTensorBuffer(mTempQuery.get());

        // Rearrange Q set: queryTmp <- query
        MNN_ASSERT(nullptr != mRearrangeQSet);
        mRearrangeQSet->writeBuffer(tqBuf.first->buffer(), 0, vkBn->getTensorSize(mTempQuery.get()), tqBuf.second);
        mRearrangeQSet->writeBuffer(queryBuf.first->buffer(), 1, vkBn->getTensorSize(query), queryBuf.second);
        mRearrangeQSet->writeBuffer(mParam->buffer(), 2, mParam->size());

        MNN_ASSERT(nullptr != mTempQKBlock && nullptr != mTempWBlock);
        MNN_ASSERT(nullptr != mTempM && nullptr != mTempL && nullptr != mTempAlpha && nullptr != mTempOAcc);
        auto qkBuf = vkBn->getTensorBuffer(mTempQKBlock.get());
        auto wBuf = vkBn->getTensorBuffer(mTempWBlock.get());
        auto mBuf = vkBn->getTensorBuffer(mTempM.get());
        auto lBuf = vkBn->getTensorBuffer(mTempL.get());
        auto aBuf = vkBn->getTensorBuffer(mTempAlpha.get());
        auto oBuf = vkBn->getTensorBuffer(mTempOAcc.get());

        // Init state set
        mInitStateSet->writeBuffer(mBuf.first->buffer(), 0, vkBn->getTensorSize(mTempM.get()), mBuf.second);
        mInitStateSet->writeBuffer(lBuf.first->buffer(), 1, vkBn->getTensorSize(mTempL.get()), lBuf.second);
        mInitStateSet->writeBuffer(aBuf.first->buffer(), 2, vkBn->getTensorSize(mTempAlpha.get()), aBuf.second);
        mInitStateSet->writeBuffer(oBuf.first->buffer(), 3, vkBn->getTensorSize(mTempOAcc.get()), oBuf.second);
        mInitStateSet->writeBuffer(mParam->buffer(), 4, mParam->size());

        // QK block set
        mQKBlockSet->writeBuffer(qkBuf.first->buffer(), 0, vkBn->getTensorSize(mTempQKBlock.get()), qkBuf.second);
        mQKBlockSet->writeBuffer(tqBuf.first->buffer(), 1, vkBn->getTensorSize(mTempQuery.get()), tqBuf.second);
        mQKBlockSet->writeBuffer(cacheKeyBuf->buffer(), 2, cacheKeySize, cacheKeyOffset);
        if (hasMask) {
            auto maskBuf = vkBn->getTensorBuffer(mask);
            mQKBlockSet->writeBuffer(maskBuf.first->buffer(), 3, vkBn->getTensorSize(mask), maskBuf.second);
        } else {
            mQKBlockSet->writeBuffer(queryBuf.first->buffer(), 3, vkBn->getTensorSize(query), queryBuf.second);
        }
        mQKBlockSet->writeBuffer(mParam->buffer(), 4, mParam->size());

        // QK full-block set (same bindings as tail-safe set)
        mQKBlockFullSet->writeBuffer(qkBuf.first->buffer(), 0, vkBn->getTensorSize(mTempQKBlock.get()), qkBuf.second);
        mQKBlockFullSet->writeBuffer(tqBuf.first->buffer(), 1, vkBn->getTensorSize(mTempQuery.get()), tqBuf.second);
        mQKBlockFullSet->writeBuffer(cacheKeyBuf->buffer(), 2, cacheKeySize, cacheKeyOffset);
        if (hasMask) {
            auto maskBuf = vkBn->getTensorBuffer(mask);
            mQKBlockFullSet->writeBuffer(maskBuf.first->buffer(), 3, vkBn->getTensorSize(mask), maskBuf.second);
        } else {
            mQKBlockFullSet->writeBuffer(queryBuf.first->buffer(), 3, vkBn->getTensorSize(query), queryBuf.second);
        }
        mQKBlockFullSet->writeBuffer(mParam->buffer(), 4, mParam->size());

        // Softmax online set (writes w, updates m/l/alpha)
        mSoftmaxOnlineSet->writeBuffer(wBuf.first->buffer(), 0, vkBn->getTensorSize(mTempWBlock.get()), wBuf.second);
        mSoftmaxOnlineSet->writeBuffer(qkBuf.first->buffer(), 1, vkBn->getTensorSize(mTempQKBlock.get()), qkBuf.second);
        mSoftmaxOnlineSet->writeBuffer(mBuf.first->buffer(), 2, vkBn->getTensorSize(mTempM.get()), mBuf.second);
        mSoftmaxOnlineSet->writeBuffer(lBuf.first->buffer(), 3, vkBn->getTensorSize(mTempL.get()), lBuf.second);
        mSoftmaxOnlineSet->writeBuffer(aBuf.first->buffer(), 4, vkBn->getTensorSize(mTempAlpha.get()), aBuf.second);
        mSoftmaxOnlineSet->writeBuffer(mParam->buffer(), 5, mParam->size());

        // QKV accumulate set
        mQKVAccSet->writeBuffer(oBuf.first->buffer(), 0, vkBn->getTensorSize(mTempOAcc.get()), oBuf.second);
        mQKVAccSet->writeBuffer(wBuf.first->buffer(), 1, vkBn->getTensorSize(mTempWBlock.get()), wBuf.second);
        mQKVAccSet->writeBuffer(cacheValueBuf->buffer(), 2, cacheValueSize, cacheValueOffset);
        mQKVAccSet->writeBuffer(aBuf.first->buffer(), 3, vkBn->getTensorSize(mTempAlpha.get()), aBuf.second);
        mQKVAccSet->writeBuffer(mParam->buffer(), 4, mParam->size());

        // QKV accumulate full-block set (same bindings as tail-safe set)
        mQKVAccFullSet->writeBuffer(oBuf.first->buffer(), 0, vkBn->getTensorSize(mTempOAcc.get()), oBuf.second);
        mQKVAccFullSet->writeBuffer(wBuf.first->buffer(), 1, vkBn->getTensorSize(mTempWBlock.get()), wBuf.second);
        mQKVAccFullSet->writeBuffer(cacheValueBuf->buffer(), 2, cacheValueSize, cacheValueOffset);
        mQKVAccFullSet->writeBuffer(aBuf.first->buffer(), 3, vkBn->getTensorSize(mTempAlpha.get()), aBuf.second);
        mQKVAccFullSet->writeBuffer(mParam->buffer(), 4, mParam->size());

        // Finalize set
        mFinalizeSet->writeBuffer(outBuf.first->buffer(), 0, vkBn->getTensorSize(output), outBuf.second);
        mFinalizeSet->writeBuffer(oBuf.first->buffer(), 1, vkBn->getTensorSize(mTempOAcc.get()), oBuf.second);
        mFinalizeSet->writeBuffer(lBuf.first->buffer(), 2, vkBn->getTensorSize(mTempL.get()), lBuf.second);
        mFinalizeSet->writeBuffer(mParam->buffer(), 3, mParam->size());
        return NO_ERROR;
    }

    // Attention set (fused). Keep packed fused set for fallback even when decode-q1 subgroup is available.
    auto writeAttentionSet = [&](const std::shared_ptr<VulkanLayout::DescriptorSet>& set) {
        MNN_ASSERT(nullptr != set);
        set->writeBuffer(outBuf.first->buffer(), 0, vkBn->getTensorSize(output), outBuf.second);
        set->writeBuffer(queryBuf.first->buffer(), 1, vkBn->getTensorSize(query), queryBuf.second);
        set->writeBuffer(keyBuf.first->buffer(), 2, vkBn->getTensorSize(key), keyBuf.second);
        set->writeBuffer(valueBuf.first->buffer(), 3, vkBn->getTensorSize(value), valueBuf.second);
        set->writeBuffer(cacheKeyBuf->buffer(), 4, cacheKeySize, cacheKeyOffset);
        set->writeBuffer(cacheValueBuf->buffer(), 5, cacheValueSize, cacheValueOffset);
        if (hasMask) {
            auto maskBuf = vkBn->getTensorBuffer(mask);
            set->writeBuffer(maskBuf.first->buffer(), 6, vkBn->getTensorSize(mask), maskBuf.second);
        } else {
            set->writeBuffer(queryBuf.first->buffer(), 6, vkBn->getTensorSize(query), queryBuf.second);
        }
        set->writeBuffer(mParam->buffer(), 7, mParam->size());
    };
    if (mNeedKvCache) {
        MNN_ASSERT(nullptr != mAttentionSet);
        writeAttentionSet(mAttentionSet);
        if (mQueryLen == 1 && nullptr != mDecodeQ1SubgroupSet) {
            writeAttentionSet(mDecodeQ1SubgroupSet);
        }
        if (mQueryLen == 1 && nullptr != mDecodeQ1SubgroupHD128Set) {
            writeAttentionSet(mDecodeQ1SubgroupHD128Set);
        }
    } else {
        MNN_ASSERT(nullptr != mAttentionLegacySet);
        writeAttentionSet(mAttentionLegacySet);
    }

    return NO_ERROR;
}

class VulkanAttentionCreator : public VulkanBackend::Creator {
public:
    VulkanBasicExecution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                                   Backend* backend) const override {
        return new VulkanAttention(op, backend);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_Attention, new VulkanAttentionCreator);
    return true;
}();

} // namespace MNN

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
