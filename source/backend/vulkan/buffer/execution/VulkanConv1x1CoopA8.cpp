#include "VulkanConv1x1CoopA8.hpp"
#include "core/TensorUtils.hpp"
#include "core/Macro.h"
#include "VulkanBackend.hpp"
#include "VulkanSharedGather.hpp"
#include "backend/vulkan/vulkan/vulkan_wrapper.h"
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

struct SumKPrepareParams {
    uint32_t padN;
    uint32_t weightStride;
};

struct RequantMetaPrepareParams {
    uint32_t K;
    uint32_t N;
    uint32_t padK;
    uint32_t padN;
    uint32_t blockSize;
    uint32_t blockStride;
    uint32_t weightStride;
};

// Builds the static GPU buffers shared by decode + prefill paths:
//   - quantWeightBuffer: INT8: [padN, padK/4] uint32 (4x int8 packed).
//                        INT4: [padN, padK/8] uint32 (8x int4 packed,
//                              unsigned 0..15 with host +8 offset; padding
//                              nibbles are 0x8 -> decoded 0).
//   - quantMetaBuffer:   [padN, 2] FP interleaved (scale, offset).
//   - sumWqBuffer:       [padN] int32 = sum_k Wq[n, k], padding rows zero.
// Per-channel weights use quantMetaBuffer directly. INT4 block weights build a
// per-row requant meta buffer while keeping the persistent weight compressed.
static bool _prepareStaticBuffersGPU(VulkanBackend* vkBn, const ConvolutionCommon::Int8Common* quantCommon,
                                     bool useFP16, int ci, int co, uint32_t padN, uint32_t padK,
                                     std::shared_ptr<VulkanBuffer>& quantWeightBuffer,
                                     std::shared_ptr<VulkanBuffer>& quantMetaBuffer,
                                     std::shared_ptr<VulkanBuffer>& requantMetaBuffer,
                                     std::shared_ptr<VulkanBuffer>& sumWqBuffer) {
    if (nullptr == vkBn || nullptr == quantCommon || nullptr == quantCommon->weight.get()) {
        return false;
    }
    const bool isInt4 = quantCommon->canUseInt4;
    if (isInt4) {
        MNN_ASSERT(padK % 8u == 0u);
    }

    const int soSize = quantCommon->asymmetric ? 2 : 1;
    const int alphaSize = quantCommon->alpha.size();
    const int blockCount = std::max(1, alphaSize / std::max(1, co * soSize));
    const uint32_t blockSize = std::max<uint32_t>(1u, static_cast<uint32_t>(UP_DIV(ci, blockCount)));
    // Meta is stored for real quant blocks only; padK tail guards live in weight-to-coop shaders.
    const uint32_t blockStride = static_cast<uint32_t>(blockCount);
    const uint32_t decodeWeightStrideWords = isInt4 ? (padK / 8u) : (padK / 4u);

    const int8_t* qWeight = quantCommon->weight.get();
    const size_t rawWeightBytes = static_cast<size_t>(quantCommon->weight.size());
    const size_t alignedWeightBytes = std::max<size_t>(4u, ALIGN_UP4(rawWeightBytes));
    const size_t decodeWeightBytes =
        static_cast<size_t>(padN) * static_cast<size_t>(decodeWeightStrideWords) * sizeof(uint32_t);
    const size_t metaBytes =
        static_cast<size_t>(padN) * static_cast<size_t>(blockStride) * 2u * (useFP16 ? sizeof(int16_t) : sizeof(float));
    const size_t requantMetaBytes = static_cast<size_t>(padN) * 2u * (useFP16 ? sizeof(int16_t) : sizeof(float));
    const size_t sumWqBytes = static_cast<size_t>(padN) * sizeof(int32_t);

    const void* rawWeightSrc = qWeight;
    std::vector<uint8_t> weightAlignedHost;
    if (alignedWeightBytes != rawWeightBytes) {
        weightAlignedHost.resize(alignedWeightBytes, 0);
        if (rawWeightBytes > 0u) {
            ::memcpy(weightAlignedHost.data(), qWeight, rawWeightBytes);
        }
        rawWeightSrc = weightAlignedHost.data();
    }

    std::shared_ptr<VulkanBuffer> rawWeightBuffer(new VulkanBuffer(vkBn->getMemoryPool(), false, alignedWeightBytes,
                                                                   rawWeightSrc, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));

    quantWeightBuffer.reset(new VulkanBuffer(vkBn->getMemoryPool(), false, decodeWeightBytes, nullptr,
                                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_SHARING_MODE_EXCLUSIVE, 0));

    const float* alphaPtr = quantCommon->alpha.get();
    const size_t rawAlphaBytes = static_cast<size_t>(std::max(alphaSize, 1)) * sizeof(float);
    const float zero = 0.0f;
    const void* rawAlphaSrc = (alphaSize > 0 && nullptr != alphaPtr) ? alphaPtr : &zero;
    std::shared_ptr<VulkanBuffer> rawAlphaBuffer(
        new VulkanBuffer(vkBn->getMemoryPool(), false, rawAlphaBytes, rawAlphaSrc, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));

    quantMetaBuffer.reset(new VulkanBuffer(vkBn->getMemoryPool(), false, metaBytes, nullptr,
                                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_SHARING_MODE_EXCLUSIVE, 0));
    sumWqBuffer.reset(new VulkanBuffer(vkBn->getMemoryPool(), false, sumWqBytes, nullptr,
                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_SHARING_MODE_EXCLUSIVE, 0));
    if (isInt4 && blockCount > 1) {
        requantMetaBuffer.reset(new VulkanBuffer(vkBn->getMemoryPool(), false, requantMetaBytes, nullptr,
                                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_SHARING_MODE_EXCLUSIVE, 0));
    } else {
        requantMetaBuffer = quantMetaBuffer;
    }

    const char* weightShader =
        isInt4 ? "glsl_conv1x1_int4_weight_prepare_comp" : "glsl_conv1x1_int8_weight_prepare_comp";
    const char* metaShader =
        useFP16 ? "glsl_conv1x1_quant_meta_prepare_FP16_comp" : "glsl_conv1x1_quant_meta_prepare_comp";
    const char* sumKShader = isInt4 ? "glsl_conv1x1_int4_weight_sumK_comp" : "glsl_conv1x1_int8_weight_sumK_comp";

    std::vector<VkDescriptorType> twoBufTypes = {
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    };
    auto weightPipeline = vkBn->getPipeline(weightShader, twoBufTypes);
    auto metaPipeline = vkBn->getPipeline(metaShader, twoBufTypes);
    auto sumKPipeline = vkBn->getPipeline(sumKShader, twoBufTypes);
    const VulkanPipeline* requantMetaPipeline = nullptr;
    if (isInt4 && blockCount > 1) {
        const char* requantMetaShader = useFP16 ? "glsl_conv1x1_int4_requant_meta_prepare_FP16_comp"
                                                : "glsl_conv1x1_int4_requant_meta_prepare_comp";
        requantMetaPipeline = vkBn->getPipeline(requantMetaShader,
                                                {
                                                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                },
                                                {256u, 1u, 1u}, {});
    }
    if (nullptr == weightPipeline || nullptr == metaPipeline || nullptr == sumKPipeline ||
        (isInt4 && blockCount > 1 && nullptr == requantMetaPipeline)) {
        return false;
    }

    std::shared_ptr<VulkanLayout::DescriptorSet> weightSet(weightPipeline->createSet());
    std::shared_ptr<VulkanLayout::DescriptorSet> metaSet(metaPipeline->createSet());
    std::shared_ptr<VulkanLayout::DescriptorSet> sumKSet(sumKPipeline->createSet());
    std::shared_ptr<VulkanLayout::DescriptorSet> requantMetaSet;
    if (nullptr != requantMetaPipeline) {
        requantMetaSet.reset(requantMetaPipeline->createSet());
    }
    if (nullptr == weightSet.get() || nullptr == metaSet.get() || nullptr == sumKSet.get() ||
        (nullptr != requantMetaPipeline && nullptr == requantMetaSet.get())) {
        return false;
    }

    std::shared_ptr<VulkanCommandPool::Buffer> prepareCmd(vkBn->getPool().allocBuffer());
    prepareCmd->begin(0);

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

    if (nullptr != requantMetaPipeline) {
        RequantMetaPrepareParams pc;
        pc.K = static_cast<uint32_t>(ci);
        pc.N = static_cast<uint32_t>(co);
        pc.padK = padK;
        pc.padN = padN;
        pc.blockSize = blockSize;
        pc.blockStride = blockStride;
        pc.weightStride = decodeWeightStrideWords;

        requantMetaSet->writeBuffer(quantWeightBuffer->buffer(), 0, quantWeightBuffer->size());
        requantMetaSet->writeBuffer(quantMetaBuffer->buffer(), 1, quantMetaBuffer->size());
        requantMetaSet->writeBuffer(requantMetaBuffer->buffer(), 2, requantMetaBuffer->size());
        requantMetaPipeline->bind(prepareCmd->get(), requantMetaSet->get());
        vkCmdPushConstants(prepareCmd->get(), requantMetaPipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc),
                           &pc);
        vkCmdDispatch(prepareCmd->get(), padN, 1, 1);
        prepareCmd->barrierSource(requantMetaBuffer->buffer(), 0, requantMetaBuffer->size());
    }

    {
        SumKPrepareParams pc;
        pc.padN = padN;
        pc.weightStride = decodeWeightStrideWords;

        sumKSet->writeBuffer(quantWeightBuffer->buffer(), 0, quantWeightBuffer->size());
        sumKSet->writeBuffer(sumWqBuffer->buffer(), 1, sumWqBuffer->size());
        sumKPipeline->bind(prepareCmd->get(), sumKSet->get());
        vkCmdPushConstants(prepareCmd->get(), sumKPipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(prepareCmd->get(), UP_DIV(padN, 64u), 1, 1);
    }

    prepareCmd->end();
    std::vector<std::shared_ptr<VulkanBuffer>> keepBuffers = {rawWeightBuffer, rawAlphaBuffer};
    std::vector<std::shared_ptr<VulkanLayout::DescriptorSet>> keepSets = {weightSet, metaSet, sumKSet};
    if (nullptr != requantMetaSet) {
        keepSets.emplace_back(requantMetaSet);
    }
    vkBn->submitCommand(prepareCmd, std::move(keepBuffers), std::move(keepSets));
    return true;
}

} // namespace

VulkanConv1x1CoopA8::VulkanConv1x1CoopA8(VulkanBackend* backend, const Convolution2DCommon* convOption,
                                         const float* biasPtr, int ci, int co, VulkanDevice::CoopMatInfo coopMatInfo,
                                         std::shared_ptr<ConvolutionCommon::Int8Common> quantInfo)
    : VulkanBasicExecution(backend), mCommon(convOption), mCi(ci), mCo(co), mQuantCommon(std::move(quantInfo)) {
    MNN_ASSERT(coopMatInfo.supportS8S8S32);
    MNN_ASSERT(coopMatInfo.selectedS8CoopMatShape.size() == 3);
    mCoopM = coopMatInfo.selectedS8CoopMatShape[0];
    mCoopN = coopMatInfo.selectedS8CoopMatShape[1];
    mCoopK = coopMatInfo.selectedS8CoopMatShape[2];
    uint32_t subgroupSize = backend->getDevice().getSubgroupSize();
    MNN_ASSERT(subgroupSize > 0);
    mSubgroupSize = subgroupSize;
    _init(biasPtr, true);
}

VulkanConv1x1CoopA8::VulkanConv1x1CoopA8(VulkanBackend* backend, const Convolution2DCommon* convOption, int ci, int co,
                                         uint32_t coopM, uint32_t coopN, uint32_t coopK, uint32_t subgroupSize,
                                         std::shared_ptr<ConvolutionCommon::Int8Common> quantInfo,
                                         bool initStaticResource)
    : VulkanBasicExecution(backend), mCommon(convOption), mCi(ci), mCo(co), mQuantCommon(std::move(quantInfo)) {
    mCoopM = coopM;
    mCoopN = coopN;
    mCoopK = coopK;
    MNN_ASSERT(subgroupSize > 0);
    mSubgroupSize = subgroupSize;
    _init(nullptr, initStaticResource);
}

VulkanConv1x1CoopA8::~VulkanConv1x1CoopA8() {}

bool VulkanConv1x1CoopA8::onClone(Backend* bn, const Op* op, VulkanBasicExecution** dst) {
    if (nullptr == dst) {
        return true;
    }
    auto vkBn = static_cast<VulkanBackend*>(bn);
    if (nullptr != op && op->type() == OpType_GatherV2) {
        if (nullptr == mQuantCommon.get() || nullptr == mQuantWeightBuffer.get() || nullptr == mQuantMetaBuffer.get()) {
            return false;
        }
        const int quantBits = mIsInt4 ? 4 : 8;
        const uint32_t weightStride = (quantBits == 4) ? (mPadK / 8u) : (mPadK / 4u);
        const bool offsetZero = !mQuantCommon->asymmetric;
        *dst = new VulkanSharedGather(vkBn, mCi, mCo, quantBits, mPadN, mBlockSize, mBlockStride, weightStride,
                                      offsetZero, mQuantWeightBuffer, mQuantMetaBuffer);
        return true;
    }
    if (nullptr == op) {
        return false;
    }
    auto conv2D = op->main_as_Convolution2D();
    if (nullptr == conv2D || nullptr == conv2D->common()) {
        return false;
    }
    auto res = new VulkanConv1x1CoopA8(vkBn, conv2D->common(), mCi, mCo, mCoopM, mCoopN, mCoopK, mSubgroupSize,
                                       mQuantCommon, false);
    res->mPadK = mPadK;
    res->mPadN = mPadN;
    res->mBiasBuffer = mBiasBuffer;
    res->mQuantWeightBuffer = mQuantWeightBuffer;
    res->mQuantMetaBuffer = mQuantMetaBuffer;
    res->mRequantMetaBuffer = mRequantMetaBuffer;
    res->mSumWqBuffer = mSumWqBuffer;
    res->mUseRequantWeight = mUseRequantWeight;
    res->mDecodeRowsPerGroup = mDecodeRowsPerGroup;
    res->mBlockSize = mBlockSize;
    res->mBlockStride = mBlockStride;
    *dst = res;
    return true;
}

bool VulkanConv1x1CoopA8::_init(const float* biasPtr, bool initStaticResource) {
    auto vkBn = static_cast<VulkanBackend*>(backend());
    const bool useFP16 = vkBn->useFP16();
    const size_t fpElem = useFP16 ? sizeof(int16_t) : sizeof(float);
    mIsInt4 = mQuantCommon != nullptr && mQuantCommon->canUseInt4;

    const uint32_t K = mCi;
    const uint32_t N = mCo;
    mPadK = ROUND_UP(K, mCoopK);
    mPadN = ROUND_UP(N, mCoopN);
    const int soSize = mQuantCommon != nullptr && mQuantCommon->asymmetric ? 2 : 1;
    const int alphaSize = mQuantCommon != nullptr ? mQuantCommon->alpha.size() : 0;
    const int blockCount = std::max(1, alphaSize / std::max(1, mCo * soSize));
    mBlockSize = std::max<uint32_t>(1u, static_cast<uint32_t>(UP_DIV(mCi, blockCount)));
    // Keep the runtime stride aligned with quant meta layout; shaders avoid padK tail OOB reads.
    mBlockStride = static_cast<uint32_t>(blockCount);
    mUseRequantWeight = mIsInt4 && blockCount > 1;

    if (initStaticResource) {
        mBiasBuffer = std::make_shared<VulkanBuffer>(vkBn->getMemoryPool(), false, fpElem * mPadN, nullptr,
                                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        auto biasMap = mBiasBuffer->map();
        if (nullptr == biasMap) {
            return false;
        }
        ::memset(biasMap, 0, mPadN * fpElem);
        if (biasPtr) {
            if (useFP16) {
                std::vector<int16_t> biasFP16(N);
                FLOAT_TO_HALF(biasPtr, biasFP16.data(), N);
                ::memcpy(biasMap, biasFP16.data(), N * sizeof(int16_t));
            } else {
                ::memcpy(biasMap, biasPtr, N * sizeof(float));
            }
        }
        mBiasBuffer->unmap();

        if (!_prepareStaticBuffersGPU(vkBn, mQuantCommon.get(), useFP16, mCi, mCo, mPadN, mPadK, mQuantWeightBuffer,
                                      mQuantMetaBuffer, mRequantMetaBuffer, mSumWqBuffer)) {
            return false;
        }
    }
    if (!mUseRequantWeight) {
        mRequantMetaBuffer = mQuantMetaBuffer;
    }

    int activation = 0;
    if (mCommon->relu()) {
        activation = 1;
    } else if (mCommon->relu6()) {
        activation = 2;
    }

    // Decode (M == 1): reuse Coop-A16's gemv_dequant_int{4,8}; INT4
    // per-channel is the single-block subset of block quant.
    {
        std::vector<VkDescriptorType> types(5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        mDecodeRowsPerGroup = (mIsInt4 && (mBlockStride == 1u || mBlockSize == 64u)) ? 6u : 1u;
        std::vector<uint32_t> localSize = {mSubgroupSize * mDecodeRowsPerGroup, 1, 1};
        std::vector<uint32_t> spec = {(uint32_t)activation};
        const char* shader;
        if (mIsInt4) {
            shader = useFP16 ? "glsl_gemv_dequant_int4_FP16_comp" : "glsl_gemv_dequant_int4_comp";
            const uint32_t offsetMode = (mQuantCommon != nullptr && !mQuantCommon->asymmetric) ? 1u : 0u;
            spec.push_back(mBlockSize);
            spec.push_back(mBlockStride);
            spec.push_back(offsetMode);
            spec.push_back(static_cast<uint32_t>(mCi));
            spec.push_back((mPadK / 8u));
            spec.push_back(static_cast<uint32_t>(mCo));
        } else {
            shader = useFP16 ? "glsl_gemv_dequant_int8_FP16_comp" : "glsl_gemv_dequant_int8_comp";
        }
        mDecodePipeline = vkBn->getPipeline(shader, types, localSize, spec);
        if (nullptr == mDecodePipeline) {
            return false;
        }
        mDecodeSet.reset(mDecodePipeline->createSet());
    }
    // INT4-only: runtime nibble unpack pipeline. Idempotent across INT8 mode
    // (skipped on dispatch side); registered here only when needed.
    if (mIsInt4) {
        std::vector<VkDescriptorType> types(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        std::vector<uint32_t> localSize = {16u, 16u, 1u};
        mInt4UnpackPipeline = vkBn->getPipeline("glsl_dynamic_int4_to_int8_unpack_comp", types, localSize, {});
        mInt4UnpackSet.reset(mInt4UnpackPipeline->createSet());
    }
    if (mUseRequantWeight) {
        std::vector<VkDescriptorType> types(5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        std::vector<uint32_t> localSize = {256u, 1u, 1u};
        const char* shader =
            useFP16 ? "glsl_dynamic_int4_requant_weight_FP16_comp" : "glsl_dynamic_int4_requant_weight_comp";
        mInt4RequantWeightPipeline = vkBn->getPipeline(shader, types, localSize, {});
        mInt4RequantWeightSet.reset(mInt4RequantWeightPipeline->createSet());
    }

    // Prefill stages — spec constant_id starts at 3 (after local_size_x/y/z).
    {
        std::vector<VkDescriptorType> types(5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        std::vector<uint32_t> localSize = {256u, 1u, 1u};
        std::vector<uint32_t> spec = {mCoopM, mCoopK};
        const char* shader = useFP16 ? "glsl_dynamic_quant_all_pack_FP16_comp" : "glsl_dynamic_quant_all_pack_comp";
        mQuantAllPackPipeline = vkBn->getPipeline(shader, types, localSize, spec);
        mQuantAllPackSet.reset(mQuantAllPackPipeline->createSet());
    }
    {
        std::vector<VkDescriptorType> types(9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        std::vector<uint32_t> localSize = {mSubgroupSize, 1u, 1u};
        const uint32_t bBlockLinear = (mIsInt4 && !mUseRequantWeight) ? 1u : 0u;
        const uint32_t weightAsymmetric =
            (mUseRequantWeight || (mQuantCommon != nullptr && mQuantCommon->asymmetric)) ? 1u : 0u;
        std::vector<uint32_t> spec = {(uint32_t)activation, mCoopM, mCoopN, mCoopK, bBlockLinear, weightAsymmetric};
        const char* shader =
            useFP16 ? "glsl_dynamic_w8a8_coop_gemm_dequant_FP16_comp" : "glsl_dynamic_w8a8_coop_gemm_dequant_comp";
        mGemmDequantPipeline = vkBn->getPipeline(shader, types, localSize, spec);
        mGemmDequantSet.reset(mGemmDequantPipeline->createSet());
    }

    return true;
}

ErrorCode VulkanConv1x1CoopA8::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                        const VulkanCommandPool::Buffer* cmdBuffer) {
    auto input = inputs[0];
    auto output = outputs[0];
    auto vkBn = static_cast<VulkanBackend*>(backend());

    const int M = input->batch() * input->width() * input->height();
    const int K = mCi;
    const int N = mCo;
    const uint32_t padK = mPadK;
    const uint32_t padN = mPadN;

    auto srcBuffer = vkBn->getTensorBuffer(input);
    auto dstBuffer = vkBn->getTensorBuffer(output);
    const bool useFP16 = vkBn->useFP16();

    // Per-shader timing scope (no-op when ENABLE_VULKAN_TIME_PROFILE is off).
    // The label MUST match the shader registry key passed to getPipeline so
    // the profile output and the shader registry are 1:1 traceable.
    auto dispatchWithProfile = [&](const char* name, const VulkanPipeline* pipeline,
                                   const std::shared_ptr<VulkanLayout::DescriptorSet>& set, uint32_t gx, uint32_t gy,
                                   uint32_t gz, const void* pc, uint32_t pcSize) {
#ifdef ENABLE_VULKAN_TIME_PROFILE
        auto* profiler = vkBn->timeProfiler();
        if (nullptr != profiler) {
            VulkanTimeProfileScope scope(profiler, cmdBuffer->get(), name, VulkanTimeProfiler::Kind::Shader);
            pipeline->bind(cmdBuffer->get(), set->get());
            if (pc != nullptr) {
                vkCmdPushConstants(cmdBuffer->get(), pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, pcSize, pc);
            }
            vkCmdDispatch(cmdBuffer->get(), gx, gy, gz);
            return;
        }
#else
        (void)name;
#endif
        pipeline->bind(cmdBuffer->get(), set->get());
        if (pc != nullptr) {
            vkCmdPushConstants(cmdBuffer->get(), pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, pcSize, pc);
        }
        vkCmdDispatch(cmdBuffer->get(), gx, gy, gz);
    };
    if (M == 1 || (mIsInt4 && M <= 4)) {
        struct DecodeParams {
            uint32_t K;
            uint32_t N;
            uint32_t blockSize;
            uint32_t blockStride;
            uint32_t weightStride;
        } pc;
        pc.K = static_cast<uint32_t>(K);
        pc.N = static_cast<uint32_t>(N);
        pc.blockSize = mBlockSize;
        pc.blockStride = mBlockStride;
        pc.weightStride = mIsInt4 ? (padK / 8u) : (padK / 4u);

        mDecodeSet->writeBuffer(srcBuffer.first->buffer(), 0, vkBn->getTensorSize(input), srcBuffer.second);
        mDecodeSet->writeBuffer(mQuantWeightBuffer->buffer(), 1, mQuantWeightBuffer->size());
        mDecodeSet->writeBuffer(mQuantMetaBuffer->buffer(), 2, mQuantMetaBuffer->size());
        mDecodeSet->writeBuffer(mBiasBuffer->buffer(), 3, mBiasBuffer->size());
        mDecodeSet->writeBuffer(dstBuffer.first->buffer(), 4, vkBn->getTensorSize(output), dstBuffer.second);
        const char* decodeName;
        if (mIsInt4) {
            decodeName = useFP16 ? "glsl_gemv_dequant_int4_FP16_comp" : "glsl_gemv_dequant_int4_comp";
        } else {
            decodeName = useFP16 ? "glsl_gemv_dequant_int8_FP16_comp" : "glsl_gemv_dequant_int8_comp";
        }
        dispatchWithProfile(decodeName, mDecodePipeline, mDecodeSet,
                            UP_DIV(static_cast<uint32_t>(N), mDecodeRowsPerGroup), static_cast<uint32_t>(M), 1, &pc,
                            sizeof(pc));
        return NO_ERROR;
    }

    const uint32_t padM = ROUND_UP(M, mCoopM);
    const halide_type_t fpType = useFP16 ? halide_type_of<int16_t>() : halide_type_of<float>();
    std::shared_ptr<Tensor> tScaleA(Tensor::createDevice({(int)padM}, fpType));
    std::shared_ptr<Tensor> tOffsetA(Tensor::createDevice({(int)padM}, fpType));
    std::shared_ptr<Tensor> tAq(Tensor::createDevice<int8_t>({(int)padM, (int)padK}));
    std::shared_ptr<Tensor> tSumAq(Tensor::createDevice<int32_t>({(int)padM}));
    // INT4 prefill only: unpacked/requantized int8 weight buffer that the GEMM
    // reads in place of mQuantWeightBuffer. Small-M decode uses direct INT4
    // block GEMV and returns before this path.
    std::shared_ptr<Tensor> tWqInt8;
    std::shared_ptr<Tensor> tDynamicSumWq;
    if (mIsInt4) {
        tWqInt8.reset(Tensor::createDevice<int8_t>({(int)padN, (int)padK}));
        if (mUseRequantWeight) {
            tDynamicSumWq.reset(Tensor::createDevice<int32_t>({(int)padN}));
        }
    }

    std::vector<Tensor*> dyns = {tScaleA.get(), tOffsetA.get(), tAq.get(), tSumAq.get()};
    if (mIsInt4) {
        dyns.push_back(tWqInt8.get());
        if (mUseRequantWeight) {
            dyns.push_back(tDynamicSumWq.get());
        }
    }
    size_t acquired = 0;
    for (Tensor* t : dyns) {
        if (!vkBn->onAcquireBuffer(t, Backend::DYNAMIC)) {
            for (size_t j = 0; j < acquired; ++j) {
                vkBn->onReleaseBuffer(dyns[j], Backend::DYNAMIC);
            }
            return OUT_OF_MEMORY;
        }
        ++acquired;
    }

    auto bScaleA = vkBn->getTensorBuffer(tScaleA.get());
    auto bOffsetA = vkBn->getTensorBuffer(tOffsetA.get());
    auto bAq = vkBn->getTensorBuffer(tAq.get());
    auto bSumAq = vkBn->getTensorBuffer(tSumAq.get());
    const size_t szScaleA = vkBn->getTensorSize(tScaleA.get());
    const size_t szOffsetA = vkBn->getTensorSize(tOffsetA.get());
    const size_t szAq = vkBn->getTensorSize(tAq.get());
    const size_t szSumAq = vkBn->getTensorSize(tSumAq.get());
    {
        struct PC {
            uint32_t M, K, padM, padK;
        } pc;
        pc.M = (uint32_t)M;
        pc.K = (uint32_t)K;
        pc.padM = padM;
        pc.padK = padK;

        mQuantAllPackSet->writeBuffer(srcBuffer.first->buffer(), 0, vkBn->getTensorSize(input), srcBuffer.second);
        mQuantAllPackSet->writeBuffer(bScaleA.first->buffer(), 1, szScaleA, bScaleA.second);
        mQuantAllPackSet->writeBuffer(bOffsetA.first->buffer(), 2, szOffsetA, bOffsetA.second);
        mQuantAllPackSet->writeBuffer(bAq.first->buffer(), 3, szAq, bAq.second);
        mQuantAllPackSet->writeBuffer(bSumAq.first->buffer(), 4, szSumAq, bSumAq.second);
        dispatchWithProfile(useFP16 ? "glsl_dynamic_quant_all_pack_FP16_comp" : "glsl_dynamic_quant_all_pack_comp",
                            mQuantAllPackPipeline, mQuantAllPackSet, padM, 1, 1, &pc, sizeof(pc));
        cmdBuffer->barrierSource(bScaleA.first->buffer(), bScaleA.second, szScaleA);
        cmdBuffer->barrierSource(bOffsetA.first->buffer(), bOffsetA.second, szOffsetA);
        cmdBuffer->barrierSource(bAq.first->buffer(), bAq.second, szAq);
        cmdBuffer->barrierSource(bSumAq.first->buffer(), bSumAq.second, szSumAq);
    }

    // INT4 prefill only: prepare transient int8 weights before GEMM.
    if (mIsInt4) {
        auto bWqInt8 = vkBn->getTensorBuffer(tWqInt8.get());
        const size_t szWqInt8 = vkBn->getTensorSize(tWqInt8.get());
        if (mUseRequantWeight) {
            struct PC {
                uint32_t K;
                uint32_t N;
                uint32_t padK;
                uint32_t padN;
                uint32_t blockSize;
                uint32_t blockStride;
                uint32_t weightStride;
            } pc;
            pc.K = static_cast<uint32_t>(K);
            pc.N = static_cast<uint32_t>(N);
            pc.padK = padK;
            pc.padN = padN;
            pc.blockSize = mBlockSize;
            pc.blockStride = mBlockStride;
            pc.weightStride = padK / 8u;

            auto bDynamicSumWq = vkBn->getTensorBuffer(tDynamicSumWq.get());
            const size_t szDynamicSumWq = vkBn->getTensorSize(tDynamicSumWq.get());
            mInt4RequantWeightSet->writeBuffer(mQuantWeightBuffer->buffer(), 0, mQuantWeightBuffer->size());
            mInt4RequantWeightSet->writeBuffer(mQuantMetaBuffer->buffer(), 1, mQuantMetaBuffer->size());
            mInt4RequantWeightSet->writeBuffer(mRequantMetaBuffer->buffer(), 2, mRequantMetaBuffer->size());
            mInt4RequantWeightSet->writeBuffer(bWqInt8.first->buffer(), 3, szWqInt8, bWqInt8.second);
            mInt4RequantWeightSet->writeBuffer(bDynamicSumWq.first->buffer(), 4, szDynamicSumWq, bDynamicSumWq.second);
            dispatchWithProfile(useFP16 ? "glsl_dynamic_int4_requant_weight_FP16_comp"
                                        : "glsl_dynamic_int4_requant_weight_comp",
                                mInt4RequantWeightPipeline, mInt4RequantWeightSet, padN, 1, 1, &pc, sizeof(pc));
            cmdBuffer->barrierSource(bWqInt8.first->buffer(), bWqInt8.second, szWqInt8);
            cmdBuffer->barrierSource(bDynamicSumWq.first->buffer(), bDynamicSumWq.second, szDynamicSumWq);
        } else {
            struct PC {
                uint32_t padN, padK, wordsK, coopN, coopK;
            } pc;
            pc.padN = padN;
            pc.padK = padK;
            pc.wordsK = padK / 8u;
            pc.coopN = mCoopN;
            pc.coopK = mCoopK;

            mInt4UnpackSet->writeBuffer(mQuantWeightBuffer->buffer(), 0, mQuantWeightBuffer->size());
            mInt4UnpackSet->writeBuffer(bWqInt8.first->buffer(), 1, szWqInt8, bWqInt8.second);
            dispatchWithProfile("glsl_dynamic_int4_to_int8_unpack_comp", mInt4UnpackPipeline, mInt4UnpackSet,
                                UP_DIV(pc.wordsK, 16u), UP_DIV(padN, 16u), 1, &pc, sizeof(pc));
            cmdBuffer->barrierSource(bWqInt8.first->buffer(), bWqInt8.second, szWqInt8);
        }
    }

    {
        struct PC {
            uint32_t M, N, K, padM, padN, padK;
        } pc;
        pc.M = (uint32_t)M;
        pc.N = (uint32_t)N;
        pc.K = (uint32_t)K;
        pc.padM = padM;
        pc.padN = padN;
        pc.padK = padK;

        mGemmDequantSet->writeBuffer(bAq.first->buffer(), 0, szAq, bAq.second);
        if (mIsInt4) {
            auto bWqInt8 = vkBn->getTensorBuffer(tWqInt8.get());
            const size_t szWqInt8 = vkBn->getTensorSize(tWqInt8.get());
            mGemmDequantSet->writeBuffer(bWqInt8.first->buffer(), 1, szWqInt8, bWqInt8.second);
        } else {
            mGemmDequantSet->writeBuffer(mQuantWeightBuffer->buffer(), 1, mQuantWeightBuffer->size());
        }
        mGemmDequantSet->writeBuffer(bScaleA.first->buffer(), 2, szScaleA, bScaleA.second);
        mGemmDequantSet->writeBuffer(bOffsetA.first->buffer(), 3, szOffsetA, bOffsetA.second);
        mGemmDequantSet->writeBuffer(bSumAq.first->buffer(), 4, szSumAq, bSumAq.second);
        const auto& dequantMetaBuffer = mUseRequantWeight ? mRequantMetaBuffer : mQuantMetaBuffer;
        mGemmDequantSet->writeBuffer(dequantMetaBuffer->buffer(), 5, dequantMetaBuffer->size());
        if (mUseRequantWeight) {
            auto bDynamicSumWq = vkBn->getTensorBuffer(tDynamicSumWq.get());
            const size_t szDynamicSumWq = vkBn->getTensorSize(tDynamicSumWq.get());
            mGemmDequantSet->writeBuffer(bDynamicSumWq.first->buffer(), 6, szDynamicSumWq, bDynamicSumWq.second);
        } else {
            mGemmDequantSet->writeBuffer(mSumWqBuffer->buffer(), 6, mSumWqBuffer->size());
        }
        mGemmDequantSet->writeBuffer(mBiasBuffer->buffer(), 7, mBiasBuffer->size());
        mGemmDequantSet->writeBuffer(dstBuffer.first->buffer(), 8, vkBn->getTensorSize(output), dstBuffer.second);
        dispatchWithProfile(useFP16 ? "glsl_dynamic_w8a8_coop_gemm_dequant_FP16_comp"
                                    : "glsl_dynamic_w8a8_coop_gemm_dequant_comp",
                            mGemmDequantPipeline, mGemmDequantSet, padN / mCoopN, padM / mCoopM, 1, &pc, sizeof(pc));
    }

    for (Tensor* t : dyns) {
        vkBn->onReleaseBuffer(t, Backend::DYNAMIC);
    }

    return NO_ERROR;
}

} // namespace MNN
