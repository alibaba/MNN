#include "VulkanConv1x1General.hpp"
#include "VulkanBackend.hpp"
#include "VulkanSharedGather.hpp"
#include "core/Macro.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#ifdef ENABLE_VULKAN_TIME_PROFILE
#include "backend/vulkan/component/VulkanTimeProfiler.hpp"
#endif

namespace MNN {

namespace {

struct QuantWeightPrepareParams {
    uint32_t ci;
    uint32_t co;
    uint32_t padN;
    uint32_t weightStride;
    uint32_t srcBytes;
};

struct QuantMetaPrepareParams {
    uint32_t co;
    uint32_t padN;
    uint32_t blockCount;
    uint32_t blockStride;
    uint32_t soSize;
    uint32_t alphaSize;
};

static size_t _alignUp4(size_t size) {
    return (size + 3u) & ~size_t(3u);
}

static bool _prepareQuantBuffersGPU(VulkanBackend* vkBn, const ConvolutionCommon::Int8Common* quantCommon, bool useFP16,
                                    int ci, int co, uint32_t padN, uint32_t blockStride,
                                    uint32_t decodeWeightStrideWords, int quantBits,
                                    std::shared_ptr<VulkanBuffer>& quantWeightBuffer,
                                    std::shared_ptr<VulkanBuffer>& quantMetaBuffer) {
    if (nullptr == vkBn || nullptr == quantCommon || nullptr == quantCommon->weight.get()) {
        return false;
    }
    const int soSize = quantCommon->asymmetric ? 2 : 1;
    const int alphaSize = quantCommon->alpha.size();
    const int alphaDenominator = std::max(1, co * soSize);
    const int blockCount = std::max(1, alphaSize / alphaDenominator);
    const int8_t* qWeight = quantCommon->weight.get();
    const size_t rawWeightBytes = static_cast<size_t>(quantCommon->weight.size());
    const size_t alignedWeightBytes = std::max<size_t>(4u, _alignUp4(rawWeightBytes));
    const uint32_t wordsPerGroup = (quantBits == 3) ? 2u : 1u;
    const size_t decodeWeightBytes = static_cast<size_t>(padN) * static_cast<size_t>(decodeWeightStrideWords) *
                                     static_cast<size_t>(wordsPerGroup) * sizeof(uint32_t);
    const size_t metaElem = static_cast<size_t>(padN) * static_cast<size_t>(blockStride) * 2u;
    const size_t metaBytes = metaElem * (useFP16 ? sizeof(int16_t) : sizeof(float));

    const void* rawWeightSrc = qWeight;
    std::vector<uint8_t> weightAlignedHost;
    if (alignedWeightBytes != rawWeightBytes) {
        weightAlignedHost.resize(alignedWeightBytes, 0);
        if (rawWeightBytes > 0u) {
            ::memcpy(weightAlignedHost.data(), qWeight, rawWeightBytes);
        }
        rawWeightSrc = weightAlignedHost.data();
    }

    std::shared_ptr<VulkanBuffer> stagingWeightBuffer = vkBn->createHostBuffer(alignedWeightBytes);
    ::memcpy(stagingWeightBuffer->map(), rawWeightSrc, alignedWeightBytes);
    stagingWeightBuffer->unmap();
    std::shared_ptr<VulkanBuffer> rawWeightBuffer(new VulkanBuffer(
        vkBn->getMemoryPool(), false, alignedWeightBytes, nullptr,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE, 0));

    quantWeightBuffer.reset(new VulkanBuffer(vkBn->getMemoryPool(), false, decodeWeightBytes, nullptr,
                                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_SHARING_MODE_EXCLUSIVE, 0));

    const float* alphaPtr = quantCommon->alpha.get();
    const size_t rawAlphaBytes = static_cast<size_t>(std::max(alphaSize, 1)) * sizeof(float);
    const float zero = 0.0f;
    const void* rawAlphaSrc = (alphaSize > 0 && nullptr != alphaPtr) ? alphaPtr : &zero;
    std::shared_ptr<VulkanBuffer> stagingAlphaBuffer = vkBn->createHostBuffer(rawAlphaBytes);
    ::memcpy(stagingAlphaBuffer->map(), rawAlphaSrc, rawAlphaBytes);
    stagingAlphaBuffer->unmap();
    std::shared_ptr<VulkanBuffer> rawAlphaBuffer(new VulkanBuffer(
        vkBn->getMemoryPool(), false, rawAlphaBytes, nullptr,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE, 0));

    quantMetaBuffer.reset(new VulkanBuffer(vkBn->getMemoryPool(), false, metaBytes, nullptr,
                                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_SHARING_MODE_EXCLUSIVE, 0));

    const char* weightShader = nullptr;
    switch (quantBits) {
        case 2:
            weightShader = "glsl_conv1x1_int2_weight_prepare_comp";
            break;
        case 3:
            weightShader = "glsl_conv1x1_int3_weight_prepare_comp";
            break;
        case 4:
            weightShader = "glsl_conv1x1_int4_weight_prepare_comp";
            break;
        default:
            weightShader = "glsl_conv1x1_int8_weight_prepare_comp";
            break;
    }
    const char* metaShader =
        useFP16 ? "glsl_conv1x1_quant_meta_prepare_FP16_comp" : "glsl_conv1x1_quant_meta_prepare_comp";

    std::vector<VkDescriptorType> prepareTypes = {
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    };
    auto weightPipeline = vkBn->getPipeline(weightShader, prepareTypes);
    auto metaPipeline = vkBn->getPipeline(metaShader, prepareTypes);
    if (nullptr == weightPipeline || nullptr == metaPipeline) {
        return false;
    }

    std::shared_ptr<VulkanLayout::DescriptorSet> weightSet(weightPipeline->createSet());
    std::shared_ptr<VulkanLayout::DescriptorSet> metaSet(metaPipeline->createSet());
    if (nullptr == weightSet.get() || nullptr == metaSet.get()) {
        return false;
    }

    std::shared_ptr<VulkanCommandPool::Buffer> prepareCmd(vkBn->getPool().allocBuffer());
    prepareCmd->begin(0);

    {
        VkBufferCopy copy;
        copy.srcOffset = 0;
        copy.dstOffset = 0;
        copy.size = alignedWeightBytes;
        vkCmdCopyBuffer(prepareCmd->get(), stagingWeightBuffer->buffer(), rawWeightBuffer->buffer(), 1, &copy);
        copy.size = rawAlphaBytes;
        vkCmdCopyBuffer(prepareCmd->get(), stagingAlphaBuffer->buffer(), rawAlphaBuffer->buffer(), 1, &copy);
        prepareCmd->barrierSource(rawWeightBuffer->buffer(), 0, rawWeightBuffer->size());
        prepareCmd->barrierSource(rawAlphaBuffer->buffer(), 0, rawAlphaBuffer->size());
    }

    {
        QuantWeightPrepareParams pc;
        pc.ci = static_cast<uint32_t>(ci);
        pc.co = static_cast<uint32_t>(co);
        pc.padN = padN;
        pc.weightStride = decodeWeightStrideWords;
        pc.srcBytes = static_cast<uint32_t>(rawWeightBytes);

        weightSet->writeBuffer(rawWeightBuffer->buffer(), 0, rawWeightBuffer->size());
        weightSet->writeBuffer(quantWeightBuffer->buffer(), 1, quantWeightBuffer->size());
        weightPipeline->bind(prepareCmd->get(), weightSet->get());
        vkCmdPushConstants(prepareCmd->get(), weightPipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc),
                           &pc);
        vkCmdDispatch(prepareCmd->get(), UP_DIV(decodeWeightStrideWords, 16u), UP_DIV(padN, 16u), 1);
        prepareCmd->barrierSource(quantWeightBuffer->buffer(), 0, quantWeightBuffer->size());
    }

    {
        QuantMetaPrepareParams pc;
        pc.co = static_cast<uint32_t>(co);
        pc.padN = padN;
        pc.blockCount = static_cast<uint32_t>(blockCount);
        pc.blockStride = blockStride;
        pc.soSize = static_cast<uint32_t>(soSize);
        pc.alphaSize = static_cast<uint32_t>(alphaSize);

        metaSet->writeBuffer(rawAlphaBuffer->buffer(), 0, rawAlphaBuffer->size());
        metaSet->writeBuffer(quantMetaBuffer->buffer(), 1, quantMetaBuffer->size());
        metaPipeline->bind(prepareCmd->get(), metaSet->get());
        vkCmdPushConstants(prepareCmd->get(), metaPipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(prepareCmd->get(), UP_DIV(blockStride, 16u), UP_DIV(padN, 16u), 1);
        prepareCmd->barrierSource(quantMetaBuffer->buffer(), 0, quantMetaBuffer->size());
    }

    prepareCmd->end();
    vkBn->submitCommand(prepareCmd, {stagingWeightBuffer, stagingAlphaBuffer, rawWeightBuffer, rawAlphaBuffer},
                        {weightSet, metaSet});
    return true;
}

} // namespace

VulkanConv1x1General::VulkanConv1x1General(VulkanBackend* backend, const Convolution2DCommon* convOption,
                                           const float* biasPtr, int ci, int co,
                                           std::shared_ptr<ConvolutionCommon::Int8Common> quantInfo)
    : VulkanBasicExecution(backend), mCommon(convOption), mCi(ci), mCo(co), mQuantCommon(std::move(quantInfo)) {
    if (!_init(biasPtr, true)) {
        MNN_ERROR("VulkanConv1x1General init failed\n");
        MNN_ASSERT(false);
    }
}

VulkanConv1x1General::VulkanConv1x1General(VulkanBackend* backend, const Convolution2DCommon* convOption, int ci,
                                           int co, std::shared_ptr<ConvolutionCommon::Int8Common> quantInfo,
                                           bool initStaticResource)
    : VulkanBasicExecution(backend), mCommon(convOption), mCi(ci), mCo(co), mQuantCommon(std::move(quantInfo)) {
    if (!_init(nullptr, initStaticResource)) {
        MNN_ERROR("VulkanConv1x1General clone init failed\n");
        MNN_ASSERT(false);
    }
}

VulkanConv1x1General::~VulkanConv1x1General() {}

bool VulkanConv1x1General::_init(const float* biasPtr, bool initStaticResource) {
    auto vkBn = static_cast<VulkanBackend*>(backend());
    if (nullptr == vkBn || nullptr == mQuantCommon.get() || nullptr == mQuantCommon->weight.get()) {
        return false;
    }

    const bool useFP16 = vkBn->useFP16();
    if (mQuantCommon->canUseInt2) {
        mQuantBits = 2;
    } else if (mQuantCommon->canUseInt3) {
        mQuantBits = 3;
    } else if (mQuantCommon->canUseInt4) {
        mQuantBits = 4;
    } else {
        mQuantBits = 8;
    }
    mPadK = ROUND_UP(static_cast<uint32_t>(mCi), 4u);
    mPadN = ROUND_UP(static_cast<uint32_t>(mCo), 32u);

    if (mPadK == 0u || mPadN == 0u) {
        MNN_ERROR("VulkanConv1x1General invalid shape, ci=%d, co=%d\n", mCi, mCo);
        return false;
    }

    const int soSize = mQuantCommon->asymmetric ? 2 : 1;
    const int alphaSize = mQuantCommon->alpha.size();
    const int alphaDenominator = std::max(1, mCo * soSize);
    const int blockCount = std::max(1, alphaSize / alphaDenominator);
    mBlockSize = std::max<uint32_t>(1u, static_cast<uint32_t>(UP_DIV(mCi, blockCount)));

    mBlockStride = UP_DIV(mPadK, mBlockSize);
    mDecodeWeightStrideWords = (mQuantBits == 2 || mQuantBits == 3) ? UP_DIV(mPadK, 16u)
                               : (mQuantBits == 4)                  ? UP_DIV(mPadK, 8u)
                                                                    : (mPadK / 4u);

    if (initStaticResource) {
        std::vector<float> biasHost(mPadN, 0.0f);
        if (nullptr != biasPtr) {
            ::memcpy(biasHost.data(), biasPtr, static_cast<size_t>(mCo) * sizeof(float));
        }
        const size_t elementSize = useFP16 ? sizeof(int16_t) : sizeof(float);
        mBiasBuffer.reset(new VulkanBuffer(vkBn->getMemoryPool(), true, mPadN * elementSize, nullptr,
                                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
        auto biasMap = mBiasBuffer->map();
        if (nullptr == biasMap) {
            return false;
        }
        ::memset(biasMap, 0, mPadN * elementSize);
        if (useFP16) {
            std::vector<int16_t> biasHalf(mPadN);
            FLOAT_TO_HALF(biasHost.data(), biasHalf.data(), static_cast<int>(mPadN));
            ::memcpy(biasMap, biasHalf.data(), mPadN * sizeof(int16_t));
        } else {
            ::memcpy(biasMap, biasHost.data(), mPadN * sizeof(float));
        }
        mBiasBuffer->unmap();

        if (!_prepareQuantBuffersGPU(vkBn, mQuantCommon.get(), useFP16, mCi, mCo, mPadN, mBlockStride,
                                     mDecodeWeightStrideWords, mQuantBits, mQuantWeightBuffer, mQuantMetaBuffer)) {
            return false;
        }
    }

    int activation = 0;
    if (mCommon->relu()) {
        activation = 1;
    }
    if (mCommon->relu6()) {
        activation = 2;
    }

    {
        const auto& subgroup = vkBn->getDevice().getSubgroupInfo();
        const VkSubgroupFeatureFlags requiredOps = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT;
        mUseSubgroup = subgroup.size > 0 && (subgroup.stages & VK_SHADER_STAGE_COMPUTE_BIT) &&
                       ((subgroup.ops & requiredOps) == requiredOps);
    }

    mDecodeSubgroupSize = vkBn->getDevice().getSubgroupSize();
    if (mDecodeSubgroupSize == 0u) {
        mDecodeSubgroupSize = 64u;
    }

    {
        std::vector<VkDescriptorType> types = {
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        };
        const char* shader = nullptr;
        if (mUseSubgroup) {
            std::vector<uint32_t> spec = {static_cast<uint32_t>(activation)};
            switch (mQuantBits) {
                case 2:
                    shader = useFP16 ? "glsl_gemv_dequant_int2_FP16_comp" : "glsl_gemv_dequant_int2_comp";
                    break;
                case 3:
                    shader = useFP16 ? "glsl_gemv_dequant_int3_FP16_comp" : "glsl_gemv_dequant_int3_comp";
                    break;
                case 4:
                    shader = useFP16 ? "glsl_gemv_dequant_int4_FP16_comp" : "glsl_gemv_dequant_int4_comp";
                    mDecodeRowsPerGroup = (mBlockSize == 64u) ? 6u : 1u;
                    spec.push_back(mBlockSize);
                    spec.push_back(mBlockStride);
                    spec.push_back((mQuantCommon != nullptr && !mQuantCommon->asymmetric) ? 1u : 0u);
                    spec.push_back(static_cast<uint32_t>(mCi));
                    spec.push_back(mDecodeWeightStrideWords);
                    spec.push_back(static_cast<uint32_t>(mCo));
                    break;
                default:
                    shader = useFP16 ? "glsl_gemv_dequant_int8_FP16_comp" : "glsl_gemv_dequant_int8_comp";
                    break;
            }
            if (mQuantBits != 4) {
                mDecodeRowsPerGroup = 1u;
            }
            mDecodePipeline = vkBn->getPipeline(shader, types, {mDecodeSubgroupSize * mDecodeRowsPerGroup, 1, 1}, spec);
        } else {
            uint32_t localSize = 64u;
            std::vector<uint32_t> spec = {static_cast<uint32_t>(activation), localSize};
            switch (mQuantBits) {
                case 2:
                    shader = useFP16 ? "glsl_gemv_dequant_int2_nosubgroup_FP16_comp"
                                     : "glsl_gemv_dequant_int2_nosubgroup_comp";
                    break;
                case 3:
                    shader = useFP16 ? "glsl_gemv_dequant_int3_nosubgroup_FP16_comp"
                                     : "glsl_gemv_dequant_int3_nosubgroup_comp";
                    break;
                case 4:
                    shader = useFP16 ? "glsl_gemv_dequant_int4_nosubgroup_FP16_comp"
                                     : "glsl_gemv_dequant_int4_nosubgroup_comp";
                    break;
                default:
                    shader = useFP16 ? "glsl_gemv_dequant_int8_nosubgroup_FP16_comp"
                                     : "glsl_gemv_dequant_int8_nosubgroup_comp";
                    break;
            }
            mDecodePipeline = vkBn->getPipeline(shader, types, {localSize, 1, 1}, spec);
        }
        if (nullptr == mDecodePipeline) {
            return false;
        }
        mDecodeSet.reset(mDecodePipeline->createSet());
    }

    {
        std::vector<VkDescriptorType> types = {
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        };
        const char* shader = useFP16 ? "glsl_pack_a_k4m4_to_m64k4_FP16_comp" : "glsl_pack_a_k4m4_to_m64k4_comp";
        mPackAPipeline = vkBn->getPipeline(shader, types);
        if (nullptr == mPackAPipeline) {
            return false;
        }
        mPackASet.reset(mPackAPipeline->createSet());
    }

    {
        std::vector<VkDescriptorType> types = {
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        };
        const char* shader = nullptr;
        switch (mQuantBits) {
            case 2:
                shader = useFP16 ? "glsl_int2_weight_to_pack_FP16_comp" : "glsl_int2_weight_to_pack_comp";
                break;
            case 3:
                shader = useFP16 ? "glsl_int3_weight_to_pack_FP16_comp" : "glsl_int3_weight_to_pack_comp";
                break;
            case 4:
                shader = useFP16 ? "glsl_int4_weight_to_pack_FP16_comp" : "glsl_int4_weight_to_pack_comp";
                break;
            default:
                shader = useFP16 ? "glsl_int8_weight_to_pack_FP16_comp" : "glsl_int8_weight_to_pack_comp";
                break;
        }
        mWeightToPackPipeline = vkBn->getPipeline(shader, types);
        if (nullptr == mWeightToPackPipeline) {
            return false;
        }
        mWeightToPackSet.reset(mWeightToPackPipeline->createSet());
    }

    {
        std::vector<VkDescriptorType> types = {
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        };
        std::vector<uint32_t> spec = {static_cast<uint32_t>(activation)};
        const char* shader = useFP16 ? "glsl_gemm_m8n4_FP16_comp" : "glsl_gemm_m8n4_comp";
        mGemmPipeline = vkBn->getPipeline(shader, types, {}, spec);
        if (nullptr == mGemmPipeline) {
            return false;
        }
        mGemmSet.reset(mGemmPipeline->createSet());
    }

    return mDecodeSet != nullptr && mPackASet != nullptr && mWeightToPackSet != nullptr && mGemmSet != nullptr;
}

bool VulkanConv1x1General::onClone(Backend* bn, const Op* op, VulkanBasicExecution** dst) {
    if (nullptr == dst) {
        return true;
    }
    auto vkBn = static_cast<VulkanBackend*>(bn);
    if (nullptr != op && op->type() == OpType_GatherV2) {
        const bool offsetZero = (mQuantCommon != nullptr && !mQuantCommon->asymmetric);
        *dst = new VulkanSharedGather(vkBn, mCi, mCo, mQuantBits, mPadN, mBlockSize, mBlockStride,
                                      mDecodeWeightStrideWords, offsetZero, mQuantWeightBuffer, mQuantMetaBuffer);
        return true;
    }
    auto conv2D = op->main_as_Convolution2D();
    if (nullptr == conv2D || nullptr == conv2D->common()) {
        return false;
    }
    auto res = new VulkanConv1x1General(vkBn, conv2D->common(), mCi, mCo, mQuantCommon, false);
    res->mQuantBits = mQuantBits;
    res->mPadK = mPadK;
    res->mPadN = mPadN;
    res->mBlockSize = mBlockSize;
    res->mBlockStride = mBlockStride;
    res->mDecodeWeightStrideWords = mDecodeWeightStrideWords;
    res->mDecodeSubgroupSize = mDecodeSubgroupSize;
    res->mDecodeRowsPerGroup = mDecodeRowsPerGroup;
    res->mUseSubgroup = mUseSubgroup;
    res->mQuantWeightBuffer = mQuantWeightBuffer;
    res->mQuantMetaBuffer = mQuantMetaBuffer;
    res->mBiasBuffer = mBiasBuffer;
    *dst = res;
    return true;
}

ErrorCode VulkanConv1x1General::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                         const VulkanCommandPool::Buffer* cmdBuffer) {
    if (inputs.empty() || outputs.empty() || nullptr == mQuantWeightBuffer.get() || nullptr == mQuantMetaBuffer.get() ||
        nullptr == mBiasBuffer.get()) {
        return INVALID_VALUE;
    }

    auto input = inputs[0];
    auto output = outputs[0];
    const int M = output->batch() * output->height() * output->width();
    if (M <= 0 || mCi <= 0 || mCo <= 0) {
        return NO_ERROR;
    }

    auto vkBn = static_cast<VulkanBackend*>(backend());
    auto srcBuffer = vkBn->getTensorBuffer(input);
    auto dstBuffer = vkBn->getTensorBuffer(output);
    const bool useFP16 = vkBn->useFP16();

    auto dispatchWithProfile = [&](const char* name, const VulkanPipeline* pipeline,
                                   const std::shared_ptr<VulkanLayout::DescriptorSet>& set, uint32_t gx, uint32_t gy,
                                   uint32_t gz, const void* pc, uint32_t pcSize) {
#ifdef ENABLE_VULKAN_TIME_PROFILE
        auto* profiler = vkBn->timeProfiler();
        if (nullptr != profiler) {
            VulkanTimeProfileScope scope(profiler, cmdBuffer->get(), name, VulkanTimeProfiler::Kind::Shader);
            pipeline->bind(cmdBuffer->get(), set->get());
            if (nullptr != pc) {
                vkCmdPushConstants(cmdBuffer->get(), pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, pcSize, pc);
            }
            vkCmdDispatch(cmdBuffer->get(), gx, gy, gz);
            return;
        }
#else
        (void)name;
#endif
        pipeline->bind(cmdBuffer->get(), set->get());
        if (nullptr != pc) {
            vkCmdPushConstants(cmdBuffer->get(), pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, pcSize, pc);
        }
        vkCmdDispatch(cmdBuffer->get(), gx, gy, gz);
    };

    if (M == 1) {
        struct DecodeParams {
            uint32_t K;
            uint32_t N;
            uint32_t blockSize;
            uint32_t blockStride;
            uint32_t weightStride;
        } pc;
        pc.K = static_cast<uint32_t>(mCi);
        pc.N = static_cast<uint32_t>(mCo);
        pc.blockSize = mBlockSize;
        pc.blockStride = mBlockStride;
        pc.weightStride = mDecodeWeightStrideWords;

        mDecodeSet->writeBuffer(srcBuffer.first->buffer(), 0, vkBn->getTensorSize(input), srcBuffer.second);
        mDecodeSet->writeBuffer(mQuantWeightBuffer->buffer(), 1, mQuantWeightBuffer->size());
        mDecodeSet->writeBuffer(mQuantMetaBuffer->buffer(), 2, mQuantMetaBuffer->size());
        mDecodeSet->writeBuffer(mBiasBuffer->buffer(), 3, mBiasBuffer->size());
        mDecodeSet->writeBuffer(dstBuffer.first->buffer(), 4, vkBn->getTensorSize(output), dstBuffer.second);
        const char* decodeName = nullptr;
        if (mUseSubgroup) {
            switch (mQuantBits) {
                case 2:
                    decodeName = useFP16 ? "glsl_gemv_dequant_int2_FP16_comp" : "glsl_gemv_dequant_int2_comp";
                    break;
                case 3:
                    decodeName = useFP16 ? "glsl_gemv_dequant_int3_FP16_comp" : "glsl_gemv_dequant_int3_comp";
                    break;
                case 4:
                    decodeName = useFP16 ? "glsl_gemv_dequant_int4_FP16_comp" : "glsl_gemv_dequant_int4_comp";
                    break;
                default:
                    decodeName = useFP16 ? "glsl_gemv_dequant_int8_FP16_comp" : "glsl_gemv_dequant_int8_comp";
                    break;
            }
        } else {
            switch (mQuantBits) {
                case 2:
                    decodeName = useFP16 ? "glsl_gemv_dequant_int2_nosubgroup_FP16_comp"
                                         : "glsl_gemv_dequant_int2_nosubgroup_comp";
                    break;
                case 3:
                    decodeName = useFP16 ? "glsl_gemv_dequant_int3_nosubgroup_FP16_comp"
                                         : "glsl_gemv_dequant_int3_nosubgroup_comp";
                    break;
                case 4:
                    decodeName = useFP16 ? "glsl_gemv_dequant_int4_nosubgroup_FP16_comp"
                                         : "glsl_gemv_dequant_int4_nosubgroup_comp";
                    break;
                default:
                    decodeName = useFP16 ? "glsl_gemv_dequant_int8_nosubgroup_FP16_comp"
                                         : "glsl_gemv_dequant_int8_nosubgroup_comp";
                    break;
            }
        }
        const uint32_t decodeRowsPerGroup = mUseSubgroup ? mDecodeRowsPerGroup : 1u;
        dispatchWithProfile(decodeName, mDecodePipeline, mDecodeSet,
                            UP_DIV(static_cast<uint32_t>(mCo), decodeRowsPerGroup), 1, 1, &pc, sizeof(pc));
        return NO_ERROR;
    }

    const uint32_t padM = ROUND_UP(static_cast<uint32_t>(M), 64u);
    if ((mPadK & 3u) != 0u || (mPadN & 31u) != 0u || padM == 0u) {
        return INVALID_VALUE;
    }

    if (vkBn->useFP16()) {
        mTempInputPacked.reset(Tensor::createDevice<int16_t>({static_cast<int>(padM), static_cast<int>(mPadK)}));
        mTempWeightPacked.reset(Tensor::createDevice<int16_t>({static_cast<int>(mPadK), static_cast<int>(mPadN)}));
    } else {
        mTempInputPacked.reset(Tensor::createDevice<float>({static_cast<int>(padM), static_cast<int>(mPadK)}));
        mTempWeightPacked.reset(Tensor::createDevice<float>({static_cast<int>(mPadK), static_cast<int>(mPadN)}));
    }

    bool acquiredTempA = false;
    bool acquiredTempB = false;
    auto releaseTemp = [&]() {
        if (acquiredTempA && nullptr != mTempInputPacked.get()) {
            vkBn->onReleaseBuffer(mTempInputPacked.get(), Backend::DYNAMIC);
            acquiredTempA = false;
        }
        if (acquiredTempB && nullptr != mTempWeightPacked.get()) {
            vkBn->onReleaseBuffer(mTempWeightPacked.get(), Backend::DYNAMIC);
            acquiredTempB = false;
        }
    };

    if (!vkBn->onAcquireBuffer(mTempInputPacked.get(), Backend::DYNAMIC)) {
        return OUT_OF_MEMORY;
    }
    acquiredTempA = true;

    if (!vkBn->onAcquireBuffer(mTempWeightPacked.get(), Backend::DYNAMIC)) {
        releaseTemp();
        return OUT_OF_MEMORY;
    }
    acquiredTempB = true;

    auto packedABuffer = vkBn->getTensorBuffer(mTempInputPacked.get());
    auto packedBBuffer = vkBn->getTensorBuffer(mTempWeightPacked.get());
    const size_t packedASize = vkBn->getTensorSize(mTempInputPacked.get());
    const size_t packedBSize = vkBn->getTensorSize(mTempWeightPacked.get());

    {
        struct PackAParams {
            uint32_t M;
            uint32_t K;
        } pc;
        pc.M = static_cast<uint32_t>(M);
        pc.K = mPadK;

        mPackASet->writeBuffer(srcBuffer.first->buffer(), 0, vkBn->getTensorSize(input), srcBuffer.second);
        mPackASet->writeBuffer(packedABuffer.first->buffer(), 1, packedASize, packedABuffer.second);
        dispatchWithProfile(useFP16 ? "glsl_pack_a_k4m4_to_m64k4_FP16_comp" : "glsl_pack_a_k4m4_to_m64k4_comp",
                            mPackAPipeline, mPackASet, mPadK / 4u, padM / 64u, 1, &pc, sizeof(pc));
        cmdBuffer->barrierSource(packedABuffer.first->buffer(), packedABuffer.second, packedASize);
    }

    {
        struct WeightToPackParams {
            uint32_t N;
            uint32_t K;
            uint32_t blockSize;
            uint32_t KBlocks;
        } pc;
        pc.N = mPadN;
        pc.K = (mQuantBits == 2 || mQuantBits == 3) ? mPadK : static_cast<uint32_t>(mCi);
        pc.blockSize = mBlockSize;
        pc.KBlocks = mBlockStride;

        mWeightToPackSet->writeBuffer(mQuantWeightBuffer->buffer(), 0, mQuantWeightBuffer->size());
        mWeightToPackSet->writeBuffer(mQuantMetaBuffer->buffer(), 1, mQuantMetaBuffer->size());
        mWeightToPackSet->writeBuffer(packedBBuffer.first->buffer(), 2, packedBSize, packedBBuffer.second);
        const char* weightPackName = nullptr;
        switch (mQuantBits) {
            case 2:
                weightPackName = useFP16 ? "glsl_int2_weight_to_pack_FP16_comp" : "glsl_int2_weight_to_pack_comp";
                break;
            case 3:
                weightPackName = useFP16 ? "glsl_int3_weight_to_pack_FP16_comp" : "glsl_int3_weight_to_pack_comp";
                break;
            case 4:
                weightPackName = useFP16 ? "glsl_int4_weight_to_pack_FP16_comp" : "glsl_int4_weight_to_pack_comp";
                break;
            default:
                weightPackName = useFP16 ? "glsl_int8_weight_to_pack_FP16_comp" : "glsl_int8_weight_to_pack_comp";
                break;
        }
        dispatchWithProfile(weightPackName, mWeightToPackPipeline, mWeightToPackSet, UP_DIV(mPadN / 4u, 16u),
                            UP_DIV(mPadK / 4u, 8u), 1, &pc, sizeof(pc));
        cmdBuffer->barrierSource(packedBBuffer.first->buffer(), packedBBuffer.second, packedBSize);
    }

    {
        struct GemmParams {
            uint32_t M;
            uint32_t N;
            uint32_t K;
            uint32_t padN;
        } pc;
        pc.M = static_cast<uint32_t>(M);
        pc.N = static_cast<uint32_t>(mCo);
        pc.K = mPadK;
        pc.padN = mPadN;

        mGemmSet->writeBuffer(packedABuffer.first->buffer(), 0, packedASize, packedABuffer.second);
        mGemmSet->writeBuffer(packedBBuffer.first->buffer(), 1, packedBSize, packedBBuffer.second);
        mGemmSet->writeBuffer(mBiasBuffer->buffer(), 2, mBiasBuffer->size());
        mGemmSet->writeBuffer(dstBuffer.first->buffer(), 3, vkBn->getTensorSize(output), dstBuffer.second);
        dispatchWithProfile(useFP16 ? "glsl_gemm_m8n4_FP16_comp" : "glsl_gemm_m8n4_comp", mGemmPipeline, mGemmSet,
                            mPadN / 32u, UP_DIV(static_cast<uint32_t>(M), 8u), 1, &pc, sizeof(pc));
    }

    releaseTemp();
    return NO_ERROR;
}

} // namespace MNN
