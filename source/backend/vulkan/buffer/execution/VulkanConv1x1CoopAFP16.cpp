#include "VulkanConv1x1CoopAFP16.hpp"
#include "core/TensorUtils.hpp"
#include "core/Macro.h"
#include "VulkanBackend.hpp"
#include "VulkanSharedGather.hpp"
#include "backend/vulkan/vulkan/vulkan_wrapper.h"
#include <algorithm>
#include <cstdint>
#include <cstring>

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
                                    uint32_t decodeWeightStrideWords, bool isInt4,
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
    const size_t decodeWeightBytes =
        static_cast<size_t>(padN) * static_cast<size_t>(decodeWeightStrideWords) * sizeof(uint32_t);
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

    const char* weightShader =
        isInt4 ? "glsl_conv1x1_int4_weight_prepare_comp" : "glsl_conv1x1_int8_weight_prepare_comp";
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

VulkanConv1x1Coop::VulkanConv1x1Coop(VulkanBackend* backend, const Convolution2DCommon* convOption,
                                     const float* weightPtr, const float* biasPtr, int ci, int co,
                                     VulkanDevice::CoopMatInfo coopMatInfo,
                                     std::shared_ptr<ConvolutionCommon::Int8Common> quantInfo)
    : VulkanBasicExecution(backend),
      mCommon(convOption),
      mCi(ci),
      mCo(co),
      mIsQuant(quantInfo != nullptr),
      mQuantCommon(std::move(quantInfo)) {
    const std::vector<uint32_t>& selectedShape =
        backend->useFP16() ? coopMatInfo.selectedFP16CoopMatShape : coopMatInfo.selectedFP32CoopMatShape;
    mCoopM = selectedShape[0];
    mCoopN = selectedShape[1];
    mCoopK = selectedShape[2];
    uint32_t subgroupSize = backend->getDevice().getSubgroupSize();
    if (subgroupSize == 0) {
        subgroupSize = 64;
    }
    mSubgroupSize = subgroupSize;
    _init(weightPtr, biasPtr, true);
}

VulkanConv1x1Coop::VulkanConv1x1Coop(VulkanBackend* backend, const Convolution2DCommon* convOption, int ci, int co,
                                     uint32_t coopM, uint32_t coopN, uint32_t coopK, uint32_t subgroupSize,
                                     std::shared_ptr<ConvolutionCommon::Int8Common> quantInfo, bool initStaticResource)
    : VulkanBasicExecution(backend),
      mCommon(convOption),
      mCi(ci),
      mCo(co),
      mIsQuant(quantInfo != nullptr),
      mQuantCommon(std::move(quantInfo)) {
    mCoopM = coopM;
    mCoopN = coopN;
    mCoopK = coopK;
    if (subgroupSize == 0) {
        subgroupSize = 64;
    }
    mSubgroupSize = subgroupSize;
    _init(nullptr, nullptr, initStaticResource);
}

VulkanConv1x1Coop::~VulkanConv1x1Coop() {}

bool VulkanConv1x1Coop::onClone(Backend* bn, const Op* op, VulkanBasicExecution** dst) {
    if (nullptr == dst) {
        return true;
    }
    auto vkBn = static_cast<VulkanBackend*>(bn);
    if (nullptr != op && op->type() == OpType_GatherV2) {
        if (!mIsQuant || nullptr == mQuantCommon.get() || nullptr == mQuantWeightBuffer.get() ||
            nullptr == mQuantMetaBuffer.get() || mBlockSize == 0) {
            return false;
        }
        const int quantBits = mQuantCommon->canUseInt4 ? 4 : 8;
        const uint32_t blockStride = UP_DIV(mPadK, mBlockSize);
        const uint32_t weightStride = (quantBits == 4) ? (mPadK / 8u) : (mPadK / 4u);
        const bool offsetZero = !mQuantCommon->asymmetric;
        *dst = new VulkanSharedGather(vkBn, mCi, mCo, quantBits, mPadN, mBlockSize, blockStride, weightStride,
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
    auto res = new VulkanConv1x1Coop(vkBn, conv2D->common(), mCi, mCo, mCoopM, mCoopN, mCoopK, mSubgroupSize,
                                     mQuantCommon, false);
    res->mPadK = mPadK;
    res->mPadN = mPadN;
    res->mBlockSize = mBlockSize;
    res->mDecodeRowsPerGroup = mDecodeRowsPerGroup;
    res->mQuantConverted = mQuantConverted;
    res->mWeightBuffer = mWeightBuffer;
    res->mBiasBuffer = mBiasBuffer;
    res->mQuantWeightBuffer = mQuantWeightBuffer;
    res->mQuantMetaBuffer = mQuantMetaBuffer;
    *dst = res;
    return true;
}

bool VulkanConv1x1Coop::_init(const float* weightPtr, const float* biasPtr, bool initStaticResource) {
    auto vkBn = static_cast<VulkanBackend*>(backend());
    const bool useFP16 = vkBn->useFP16();

    const uint32_t K = mCi;
    const uint32_t N = mCo;
    mPadK = ROUND_UP(K, mCoopK);
    mPadN = ROUND_UP(N, mCoopN);

    const size_t elementSize = useFP16 ? sizeof(int16_t) : sizeof(float);
    const size_t weightSize = mPadK * mPadN;

    if (initStaticResource && !mIsQuant) {
        // [N, K] -> coop packed [Kt, Nt, mCoopK, mCoopN]
        mWeightBuffer = std::make_shared<VulkanBuffer>(vkBn->getMemoryPool(), false, elementSize * weightSize, nullptr,
                                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        auto weightMap = mWeightBuffer->map();
        std::vector<uint8_t> hostWeights;
        if (useFP16) {
            hostWeights.resize(weightSize * elementSize);
        }
        auto ptrFP16 = reinterpret_cast<int16_t*>(hostWeights.data());
        auto ptrFP32 = reinterpret_cast<float*>(weightMap);

        const uint32_t tilesN = mPadN / mCoopN;
        for (uint32_t n = 0; n < mPadN; ++n) {
            const uint32_t tn = n / mCoopN;
            const uint32_t col = n % mCoopN;
            for (uint32_t k = 0; k < mPadK; ++k) {
                const uint32_t tk = k / mCoopK;
                const uint32_t row = k % mCoopK;
                float val = 0.0f;
                if (nullptr != weightPtr && k < K && n < N) {
                    val = weightPtr[n * K + k];
                }
                const uint32_t dstIdx = (tk * tilesN + tn) * (mCoopK * mCoopN) + row * mCoopN + col;
                if (useFP16) {
                    ((half_float::half*)ptrFP16)[dstIdx] = (half_float::half)val;
                } else {
                    ptrFP32[dstIdx] = val;
                }
            }
        }
        if (useFP16) {
            ::memcpy(weightMap, hostWeights.data(), weightSize * elementSize);
        }
        mWeightBuffer->unmap();
    }

    if (initStaticResource) {
        // [N] -> [padN]
        mBiasBuffer = std::make_shared<VulkanBuffer>(vkBn->getMemoryPool(), false, elementSize * mPadN, nullptr,
                                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        auto biasMap = mBiasBuffer->map();
        ::memset(biasMap, 0, mPadN * elementSize);
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
    }

    if (mIsQuant) {
        const int soSize = mQuantCommon->asymmetric ? 2 : 1;
        const int alphaSize = mQuantCommon->alpha.size();
        const int blockCount = std::max(1, alphaSize / (mCo * soSize));
        mBlockSize = UP_DIV(mCi, blockCount);
        const uint32_t kBlockStride = UP_DIV(mPadK, mBlockSize);

        MNN_ASSERT(mBlockSize > 0);
        MNN_ASSERT((mBlockSize % mCoopK) == 0);

        if (initStaticResource) {
            const bool isInt4 = mQuantCommon->canUseInt4;
            const uint32_t decodeWeightStrideWords = isInt4 ? (mPadK / 8u) : (mPadK / 4u);
            if (!_prepareQuantBuffersGPU(vkBn, mQuantCommon.get(), useFP16, mCi, mCo, mPadN, kBlockStride,
                                         decodeWeightStrideWords, isInt4, mQuantWeightBuffer, mQuantMetaBuffer)) {
                return false;
            }
        }

        // Prefill dequant pipeline: Q + Meta -> coop-packed weight
        {
            std::vector<VkDescriptorType> types = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                   VK_DESCRIPTOR_TYPE_STORAGE_BUFFER};
            std::vector<uint32_t> localSize = {mQuantCommon->canUseInt4 ? std::max(mSubgroupSize, 128u) : mSubgroupSize,
                                               1, 1};
            std::vector<uint32_t> spec = {mCoopK, mCoopN};
            const char* shader = nullptr;
            if (mQuantCommon->canUseInt4) {
                shader = useFP16 ? "glsl_int4_weight_to_coop_FP16_comp" : "glsl_int4_weight_to_coop_comp";
            } else {
                shader = useFP16 ? "glsl_int8_weight_to_coop_FP16_comp" : "glsl_int8_weight_to_coop_comp";
            }
            mPrefillDequantPipeline = vkBn->getPipeline(shader, types, localSize, spec);
            mPrefillDequantSet.reset(mPrefillDequantPipeline->createSet());
        }

        // Decode pipeline: fused dequant + gemv (M == 1)
        {
            int activation = 0;
            if (mCommon->relu()) {
                activation = 1;
            }
            if (mCommon->relu6()) {
                activation = 2;
            }
            std::vector<VkDescriptorType> types = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                   VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                   VK_DESCRIPTOR_TYPE_STORAGE_BUFFER};
            mDecodeRowsPerGroup = (mQuantCommon->canUseInt4 && mBlockSize == 64u) ? 6u : 1u;
            std::vector<uint32_t> localSize = {mSubgroupSize * mDecodeRowsPerGroup, 1, 1};
            std::vector<uint32_t> spec = {(uint32_t)activation};
            const char* shader = nullptr;
            if (mQuantCommon->canUseInt4) {
                shader = useFP16 ? "glsl_gemv_dequant_int4_FP16_comp" : "glsl_gemv_dequant_int4_comp";
                const uint32_t offsetMode = !mQuantCommon->asymmetric ? 1u : 0u;
                spec.push_back(mBlockSize);
                spec.push_back(kBlockStride);
                spec.push_back(offsetMode);
                spec.push_back(static_cast<uint32_t>(mCi));
                spec.push_back((mPadK / 8u));
                spec.push_back(static_cast<uint32_t>(mCo));
            } else {
                shader = useFP16 ? "glsl_gemv_dequant_int8_FP16_comp" : "glsl_gemv_dequant_int8_comp";
            }
            mDecodePipeline = vkBn->getPipeline(shader, types, localSize, spec);
            mDecodeSet.reset(mDecodePipeline->createSet());
        }
    }

    // Pack: C4 -> coop A
    {
        std::vector<VkDescriptorType> types = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                               VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
        std::vector<uint32_t> localSize = {mSubgroupSize * 4, 1, 1};
        std::vector<uint32_t> packSpec = {mCoopM, mCoopK};
        std::string shader = useFP16 ? "glsl_C4_to_COOP_FP16_comp" : "glsl_C4_to_COOP_comp";
        mPackPipeline = vkBn->getPipeline(shader, types, localSize, packSpec);
        mPackSet.reset(mPackPipeline->createSet());
    }

    // Coop matmul
    {
        std::vector<uint32_t> localSize = {mSubgroupSize, 1, 1};
        std::vector<uint32_t> matmulSpec = {mCoopM, mCoopN, mCoopK};
        std::vector<VkDescriptorType> types = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                               VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                               VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
        std::string shader = useFP16 ? "glsl_matmul_coop_FP16_comp" : "glsl_matmul_coop_comp";
        mMatMulPipeline = vkBn->getPipeline(shader, types, localSize, matmulSpec);
        mMatMulSet.reset(mMatMulPipeline->createSet());
    }

    // Unpack: coop C -> C4
    {
        std::vector<VkDescriptorType> types = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                               VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
        std::vector<uint32_t> localSize = {mSubgroupSize, 4, 1};
        int activation = 0;
        if (mCommon->relu()) {
            activation = 1;
        }
        if (mCommon->relu6()) {
            activation = 2;
        }
        std::vector<uint32_t> unpackSpec = {(uint32_t)activation};
        std::string shader = useFP16 ? "glsl_COOP_to_C4_FP16_comp" : "glsl_COOP_to_C4_comp";
        mUnpackPipeline = vkBn->getPipeline(shader, types, localSize, unpackSpec);
        mUnpackSet.reset(mUnpackPipeline->createSet());
    }

    return true;
}

ErrorCode VulkanConv1x1Coop::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                      const VulkanCommandPool::Buffer* cmdBuffer) {
    auto input = inputs[0];
    auto output = outputs[0];
    auto vkBn = static_cast<VulkanBackend*>(backend());

    const int batch = input->batch();
    const int width = input->width();
    const int height = input->height();
    const int M = batch * width * height;
    const int K = mCi;
    const int N = mCo;

    const uint32_t padM = ROUND_UP(M, mCoopM);
    const uint32_t padK = mPadK;
    const uint32_t padN = mPadN;

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

    if (mIsQuant && M == 1) {
        // Decode path: fused dequant + gemv, write output directly.
        MNN_ASSERT((mBlockSize % mCoopK) == 0);

        struct DecodeParams {
            uint32_t K;
            uint32_t N;
            uint32_t blockSize;
            uint32_t blockStride;
            uint32_t weightStride;
        } pc;
        pc.K = (uint32_t)K;
        pc.N = (uint32_t)N;
        pc.blockSize = mBlockSize;
        pc.blockStride = UP_DIV(mPadK, mBlockSize);
        pc.weightStride = mQuantCommon->canUseInt4 ? (mPadK / 8) : (mPadK / 4);

        mDecodeSet->writeBuffer(srcBuffer.first->buffer(), 0, vkBn->getTensorSize(input), srcBuffer.second);
        mDecodeSet->writeBuffer(mQuantWeightBuffer->buffer(), 1, mQuantWeightBuffer->size());
        mDecodeSet->writeBuffer(mQuantMetaBuffer->buffer(), 2, mQuantMetaBuffer->size());
        mDecodeSet->writeBuffer(mBiasBuffer->buffer(), 3, mBiasBuffer->size());
        mDecodeSet->writeBuffer(dstBuffer.first->buffer(), 4, vkBn->getTensorSize(output), dstBuffer.second);
        const char* decodeName = nullptr;
        if (mQuantCommon->canUseInt4) {
            decodeName = useFP16 ? "glsl_gemv_dequant_int4_FP16_comp" : "glsl_gemv_dequant_int4_comp";
        } else {
            decodeName = useFP16 ? "glsl_gemv_dequant_int8_FP16_comp" : "glsl_gemv_dequant_int8_comp";
        }
        dispatchWithProfile(decodeName, mDecodePipeline, mDecodeSet, UP_DIV((uint32_t)N, mDecodeRowsPerGroup), 1, 1,
                            &pc, sizeof(pc));
        return NO_ERROR;
    }

    if (vkBn->useFP16()) {
        mTempInput.reset(Tensor::createDevice<int16_t>({(int)padM, (int)padK}));
        mTempOutput.reset(Tensor::createDevice<int16_t>({(int)padM, (int)padN}));
    } else {
        mTempInput.reset(Tensor::createDevice<float>({(int)padM, (int)padK}));
        mTempOutput.reset(Tensor::createDevice<float>({(int)padM, (int)padN}));
    }
    auto res = vkBn->onAcquireBuffer(mTempInput.get(), Backend::DYNAMIC);
    if (!res) {
        return OUT_OF_MEMORY;
    }
    res = vkBn->onAcquireBuffer(mTempOutput.get(), Backend::DYNAMIC);
    if (!res) {
        return OUT_OF_MEMORY;
    }

    std::pair<const VulkanBuffer*, size_t> weightBufferPair;
    size_t weightBufferSize = 0;
    if (mIsQuant) {
        if (!mTempWeight) {
            if (useFP16) {
                mTempWeight.reset(Tensor::createDevice<int16_t>({(int)padK, (int)padN}));
            } else {
                mTempWeight.reset(Tensor::createDevice<float>({(int)padK, (int)padN}));
            }
        }
        res = vkBn->onAcquireBuffer(mTempWeight.get(), Backend::DYNAMIC);
        if (!res) {
            return OUT_OF_MEMORY;
        }
        weightBufferPair = vkBn->getTensorBuffer(mTempWeight.get());
        weightBufferSize = vkBn->getTensorSize(mTempWeight.get());
    } else {
        weightBufferPair = {mWeightBuffer.get(), 0};
        weightBufferSize = mWeightBuffer->size();
    }

    auto tempInBuffer = vkBn->getTensorBuffer(mTempInput.get());
    auto tempOutBuffer = vkBn->getTensorBuffer(mTempOutput.get());

    if (mIsQuant) {
        struct DequantParams {
            uint32_t K;
            uint32_t N;
            uint32_t padK;
            uint32_t padN;
            uint32_t blockSize;
            uint32_t blockStride;
        } pc;
        pc.K = (uint32_t)K;
        pc.N = (uint32_t)N;
        pc.padK = padK;
        pc.padN = padN;
        pc.blockSize = mBlockSize;
        pc.blockStride = UP_DIV(mPadK, mBlockSize);

        mPrefillDequantSet->writeBuffer(mQuantWeightBuffer->buffer(), 0, mQuantWeightBuffer->size());
        mPrefillDequantSet->writeBuffer(mQuantMetaBuffer->buffer(), 1, mQuantMetaBuffer->size());
        mPrefillDequantSet->writeBuffer(weightBufferPair.first->buffer(), 2, weightBufferSize, weightBufferPair.second);
        const char* dequantName = nullptr;
        if (mQuantCommon->canUseInt4) {
            dequantName = useFP16 ? "glsl_int4_weight_to_coop_FP16_comp" : "glsl_int4_weight_to_coop_comp";
        } else {
            dequantName = useFP16 ? "glsl_int8_weight_to_coop_FP16_comp" : "glsl_int8_weight_to_coop_comp";
        }
        dispatchWithProfile(dequantName, mPrefillDequantPipeline, mPrefillDequantSet, padN / mCoopN, padK / mCoopK, 1,
                            &pc, sizeof(pc));
        cmdBuffer->barrierSource(weightBufferPair.first->buffer(), weightBufferPair.second, weightBufferSize);
    }

    {
        struct PackParams {
            uint32_t M;
            uint32_t K;
            uint32_t padM;
            uint32_t padK;
        } pc;
        pc.M = M;
        pc.K = K;
        pc.padM = padM;
        pc.padK = padK;
        mPackConst = vkBn->allocUniform(&pc, sizeof(pc));

        mPackSet->writeBuffer(srcBuffer.first->buffer(), 0, vkBn->getTensorSize(input), srcBuffer.second);
        mPackSet->writeBuffer(tempInBuffer.first->buffer(), 1, vkBn->getTensorSize(mTempInput.get()),
                              tempInBuffer.second);
        mPackSet->writeBuffer(mPackConst->buffer(), 2, mPackConst->size());
        dispatchWithProfile(useFP16 ? "glsl_C4_to_COOP_FP16_comp" : "glsl_C4_to_COOP_comp", mPackPipeline, mPackSet,
                            padK / mCoopK, padM / mCoopM, 1, nullptr, 0);
        cmdBuffer->barrierSource(tempInBuffer.first->buffer(), tempInBuffer.second,
                                 vkBn->getTensorSize(mTempInput.get()));
    }

    {
        struct MatMulParams {
            uint32_t M;
            uint32_t N;
            uint32_t K;
            uint32_t padding;
        } pc;
        pc.M = padM;
        pc.N = padN;
        pc.K = padK;
        pc.padding = 0;

        mMatMulConst = vkBn->allocUniform(&pc, sizeof(pc));
        mMatMulSet->writeBuffer(tempInBuffer.first->buffer(), 0, vkBn->getTensorSize(mTempInput.get()),
                                tempInBuffer.second);
        mMatMulSet->writeBuffer(weightBufferPair.first->buffer(), 1, weightBufferSize, weightBufferPair.second);
        mMatMulSet->writeBuffer(mBiasBuffer->buffer(), 2, mBiasBuffer->size());
        mMatMulSet->writeBuffer(tempOutBuffer.first->buffer(), 3, vkBn->getTensorSize(mTempOutput.get()),
                                tempOutBuffer.second);
        mMatMulSet->writeBuffer(mMatMulConst->buffer(), 4, mMatMulConst->size());
        dispatchWithProfile(useFP16 ? "glsl_matmul_coop_FP16_comp" : "glsl_matmul_coop_comp", mMatMulPipeline,
                            mMatMulSet, padN / mCoopN, padM / mCoopM, 1, nullptr, 0);
        cmdBuffer->barrierSource(tempOutBuffer.first->buffer(), tempOutBuffer.second,
                                 vkBn->getTensorSize(mTempOutput.get()));
    }

    {
        struct UnpackParams {
            uint32_t M;
            uint32_t N;
            uint32_t padM;
            uint32_t padN;
        } pc;
        pc.M = M;
        pc.N = N;
        pc.padM = padM;
        pc.padN = padN;

        mUnpackConst = vkBn->allocUniform(&pc, sizeof(pc));
        mUnpackSet->writeBuffer(tempOutBuffer.first->buffer(), 0, vkBn->getTensorSize(mTempOutput.get()),
                                tempOutBuffer.second);
        mUnpackSet->writeBuffer(dstBuffer.first->buffer(), 1, vkBn->getTensorSize(output), dstBuffer.second);
        mUnpackSet->writeBuffer(mUnpackConst->buffer(), 2, mUnpackConst->size());
        dispatchWithProfile(useFP16 ? "glsl_COOP_to_C4_FP16_comp" : "glsl_COOP_to_C4_comp", mUnpackPipeline, mUnpackSet,
                            ROUND_UP(padN, 32) / 32, ROUND_UP(padM, 32) / 32, 1, nullptr, 0);
    }

    vkBn->onReleaseBuffer(mTempInput.get(), Backend::DYNAMIC);
    vkBn->onReleaseBuffer(mTempOutput.get(), Backend::DYNAMIC);
    if (mIsQuant && mTempWeight.get()) {
        vkBn->onReleaseBuffer(mTempWeight.get(), Backend::DYNAMIC);
    }

    return NO_ERROR;
}

} // namespace MNN
