#include "VulkanConv1x1CoopA8.hpp"
#include "core/TensorUtils.hpp"
#include "core/Macro.h"
#include "VulkanBackend.hpp"
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

struct SumKPrepareParams {
    uint32_t padN;
    uint32_t weightStride;
};

// Builds the static GPU buffers shared by decode + prefill paths:
//   - quantWeightBuffer: INT8: [padN, padK/4] uint32 (4x int8 packed).
//                        INT4: [padN, padK/8] uint32 (8x int4 packed,
//                              unsigned 0..15 with host +8 offset; padding
//                              nibbles are 0x8 -> decoded 0).
//   - quantMetaBuffer:   [padN, 2] FP interleaved (scale, offset).
//   - sumWqBuffer:       [padN] int32 = sum_k Wq[n, k], padding rows zero.
// Per-channel asym only — caller must guard.
static bool _prepareStaticBuffersGPU(VulkanBackend* vkBn, const ConvolutionCommon::Int8Common* quantCommon,
                                     bool useFP16, int ci, int co, uint32_t padN, uint32_t padK,
                                     std::shared_ptr<VulkanBuffer>& quantWeightBuffer,
                                     std::shared_ptr<VulkanBuffer>& quantMetaBuffer,
                                     std::shared_ptr<VulkanBuffer>& sumWqBuffer) {
    if (nullptr == vkBn || nullptr == quantCommon || nullptr == quantCommon->weight.get()) {
        return false;
    }
    MNN_ASSERT(quantCommon->asymmetric);
    const bool isInt4 = quantCommon->canUseInt4;
    if (isInt4) {
        MNN_ASSERT(padK % 8u == 0u);
    }

    const int soSize = 2;
    const int alphaSize = quantCommon->alpha.size();
    const int blockCount = std::max(1, alphaSize / std::max(1, co * soSize));
    if (blockCount != 1) {
        return false;
    }
    const uint32_t blockStride = 1u;
    const uint32_t decodeWeightStrideWords = isInt4 ? (padK / 8u) : (padK / 4u);

    const int8_t* qWeight = quantCommon->weight.get();
    const size_t rawWeightBytes = static_cast<size_t>(quantCommon->weight.size());
    const size_t alignedWeightBytes = std::max<size_t>(4u, ALIGN_UP4(rawWeightBytes));
    const size_t decodeWeightBytes =
        static_cast<size_t>(padN) * static_cast<size_t>(decodeWeightStrideWords) * sizeof(uint32_t);
    const size_t metaBytes = static_cast<size_t>(padN) * static_cast<size_t>(blockStride) * 2u
                             * (useFP16 ? sizeof(int16_t) : sizeof(float));
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

    std::shared_ptr<VulkanBuffer> rawWeightBuffer(new VulkanBuffer(
        vkBn->getMemoryPool(), false, alignedWeightBytes, nullptr,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE, 0));
    vkBn->copyToGPUBuffer(rawWeightSrc, rawWeightBuffer->buffer(), alignedWeightBytes, 0);

    quantWeightBuffer.reset(new VulkanBuffer(vkBn->getMemoryPool(), false, decodeWeightBytes, nullptr,
                                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_SHARING_MODE_EXCLUSIVE, 0));

    const float* alphaPtr = quantCommon->alpha.get();
    const size_t rawAlphaBytes = static_cast<size_t>(std::max(alphaSize, 1)) * sizeof(float);
    std::shared_ptr<VulkanBuffer> rawAlphaBuffer(new VulkanBuffer(
        vkBn->getMemoryPool(), false, rawAlphaBytes, nullptr,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE, 0));
    if (alphaSize > 0 && nullptr != alphaPtr) {
        vkBn->copyToGPUBuffer(alphaPtr, rawAlphaBuffer->buffer(), static_cast<size_t>(alphaSize) * sizeof(float), 0);
    } else {
        const float zero = 0.0f;
        vkBn->copyToGPUBuffer(&zero, rawAlphaBuffer->buffer(), sizeof(float), 0);
    }

    quantMetaBuffer.reset(new VulkanBuffer(vkBn->getMemoryPool(), false, metaBytes, nullptr,
                                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_SHARING_MODE_EXCLUSIVE, 0));
    sumWqBuffer.reset(new VulkanBuffer(vkBn->getMemoryPool(), false, sumWqBytes, nullptr,
                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_SHARING_MODE_EXCLUSIVE, 0));

    const char* weightShader = isInt4 ? "glsl_conv1x1_int4_weight_prepare_comp"
                                       : "glsl_conv1x1_int8_weight_prepare_comp";
    const char* metaShader = useFP16 ? "glsl_conv1x1_quant_meta_prepare_FP16_comp"
                                     : "glsl_conv1x1_quant_meta_prepare_comp";
    const char* sumKShader = isInt4 ? "glsl_conv1x1_int4_weight_sumK_comp"
                                    : "glsl_conv1x1_int8_weight_sumK_comp";

    std::vector<VkDescriptorType> twoBufTypes = {
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    };
    auto weightPipeline = vkBn->getPipeline(weightShader, twoBufTypes);
    auto metaPipeline = vkBn->getPipeline(metaShader, twoBufTypes);
    auto sumKPipeline = vkBn->getPipeline(sumKShader, twoBufTypes);
    if (nullptr == weightPipeline || nullptr == metaPipeline || nullptr == sumKPipeline) {
        return false;
    }

    std::shared_ptr<VulkanLayout::DescriptorSet> weightSet(weightPipeline->createSet());
    std::shared_ptr<VulkanLayout::DescriptorSet> metaSet(metaPipeline->createSet());
    std::shared_ptr<VulkanLayout::DescriptorSet> sumKSet(sumKPipeline->createSet());
    if (nullptr == weightSet.get() || nullptr == metaSet.get() || nullptr == sumKSet.get()) {
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
        vkCmdPushConstants(prepareCmd->get(), weightPipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(pc), &pc);
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
        vkCmdPushConstants(prepareCmd->get(), metaPipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(pc), &pc);
        vkCmdDispatch(prepareCmd->get(), UP_DIV(blockStride, 16u), UP_DIV(padN, 16u), 1);
        prepareCmd->barrierSource(quantMetaBuffer->buffer(), 0, quantMetaBuffer->size());
    }

    {
        SumKPrepareParams pc;
        pc.padN = padN;
        pc.weightStride = decodeWeightStrideWords;

        sumKSet->writeBuffer(quantWeightBuffer->buffer(), 0, quantWeightBuffer->size());
        sumKSet->writeBuffer(sumWqBuffer->buffer(), 1, sumWqBuffer->size());
        sumKPipeline->bind(prepareCmd->get(), sumKSet->get());
        vkCmdPushConstants(prepareCmd->get(), sumKPipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(pc), &pc);
        vkCmdDispatch(prepareCmd->get(), UP_DIV(padN, 64u), 1, 1);
    }

    prepareCmd->end();
    vkBn->getPool().submitAndWait(prepareCmd->get());
    return true;
}

} // namespace

VulkanConv1x1CoopA8::VulkanConv1x1CoopA8(VulkanBackend* backend, const Convolution2DCommon* convOption,
                                         const float* biasPtr, int ci, int co,
                                         VulkanDevice::CoopMatInfo coopMatInfo,
                                         std::shared_ptr<ConvolutionCommon::Int8Common> quantInfo)
    : VulkanBasicExecution(backend),
      mCommon(convOption),
      mCi(ci),
      mCo(co),
      mQuantCommon(std::move(quantInfo)) {
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

VulkanConv1x1CoopA8::VulkanConv1x1CoopA8(VulkanBackend* backend, const Convolution2DCommon* convOption,
                                         int ci, int co,
                                         uint32_t coopM, uint32_t coopN, uint32_t coopK, uint32_t subgroupSize,
                                         std::shared_ptr<ConvolutionCommon::Int8Common> quantInfo,
                                         bool initStaticResource)
    : VulkanBasicExecution(backend),
      mCommon(convOption),
      mCi(ci),
      mCo(co),
      mQuantCommon(std::move(quantInfo)) {
    mCoopM = coopM;
    mCoopN = coopN;
    mCoopK = coopK;
    MNN_ASSERT(subgroupSize > 0);
    mSubgroupSize = subgroupSize;
    _init(nullptr, initStaticResource);
}

VulkanConv1x1CoopA8::~VulkanConv1x1CoopA8() {
}

bool VulkanConv1x1CoopA8::onClone(Backend* bn, const Op* op, VulkanBasicExecution** dst) {
    if (nullptr == dst) {
        return true;
    }
    auto vkBn = static_cast<VulkanBackend*>(bn);
    auto conv2D = op->main_as_Convolution2D();
    if (nullptr == conv2D || nullptr == conv2D->common()) {
        return false;
    }
    auto res = new VulkanConv1x1CoopA8(vkBn, conv2D->common(), mCi, mCo,
                                       mCoopM, mCoopN, mCoopK, mSubgroupSize,
                                       mQuantCommon, false);
    res->mPadK = mPadK;
    res->mPadN = mPadN;
    res->mBiasBuffer = mBiasBuffer;
    res->mQuantWeightBuffer = mQuantWeightBuffer;
    res->mQuantMetaBuffer = mQuantMetaBuffer;
    res->mSumWqBuffer = mSumWqBuffer;
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

    if (initStaticResource) {
        mBiasBuffer = std::make_shared<VulkanBuffer>(vkBn->getMemoryPool(), false, fpElem * mPadN, nullptr,
                                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        auto biasMap = mBiasBuffer->map();
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

        if (!_prepareStaticBuffersGPU(vkBn, mQuantCommon.get(), useFP16, mCi, mCo, mPadN, mPadK,
                                      mQuantWeightBuffer, mQuantMetaBuffer, mSumWqBuffer)) {
            return false;
        }
    }

    int activation = 0;
    if (mCommon->relu()) {
        activation = 1;
    } else if (mCommon->relu6()) {
        activation = 2;
    }

    // Decode (M == 1): reuse Coop-A16's gemv_dequant_int{4,8} — per-channel is
    // the single-block subset of block-quant. INT4 / INT8 share the binding
    // layout and push-constant struct; only weightStride and shader name vary
    // (handled in onEncode).
    {
        std::vector<VkDescriptorType> types(5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        std::vector<uint32_t> localSize = {mSubgroupSize, 1, 1};
        std::vector<uint32_t> spec = {(uint32_t)activation};
        const char* shader;
        if (mIsInt4) {
            shader = useFP16 ? "glsl_gemv_dequant_int4_FP16_comp" : "glsl_gemv_dequant_int4_comp";
        } else {
            shader = useFP16 ? "glsl_gemv_dequant_int8_FP16_comp" : "glsl_gemv_dequant_int8_comp";
        }
        mDecodePipeline = vkBn->getPipeline(shader, types, localSize, spec);
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

    // Prefill stages — spec constant_id starts at 3 (after local_size_x/y/z).
    {
        std::vector<VkDescriptorType> types(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        std::vector<uint32_t> localSize = {mCoopM, 1, 1};
        std::vector<uint32_t> spec = {mCoopM};
        const char* shader = useFP16 ? "glsl_dynamic_quant_minmax_FP16_comp"
                                     : "glsl_dynamic_quant_minmax_comp";
        mQuantMinMaxPipeline = vkBn->getPipeline(shader, types, localSize, spec);
        mQuantMinMaxSet.reset(mQuantMinMaxPipeline->createSet());
    }
    {
        std::vector<VkDescriptorType> types(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        std::vector<uint32_t> localSize = {64u, 1u, 1u};
        const char* shader = useFP16 ? "glsl_dynamic_quant_reduce_minmax_FP16_comp"
                                     : "glsl_dynamic_quant_reduce_minmax_comp";
        mQuantReduceMinMaxPipeline = vkBn->getPipeline(shader, types, localSize, {});
        mQuantReduceMinMaxSet.reset(mQuantReduceMinMaxPipeline->createSet());
    }
    {
        std::vector<VkDescriptorType> types(5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        const uint32_t packLocalX = std::max(mCoopM, mCoopK);
        std::vector<uint32_t> localSize = {packLocalX, 1u, 1u};
        std::vector<uint32_t> spec = {mCoopM, mCoopK};
        const char* shader = useFP16 ? "glsl_dynamic_quant_pack_FP16_comp"
                                     : "glsl_dynamic_quant_pack_comp";
        mQuantPackPipeline = vkBn->getPipeline(shader, types, localSize, spec);
        mQuantPackSet.reset(mQuantPackPipeline->createSet());
    }
    {
        std::vector<VkDescriptorType> types(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        std::vector<uint32_t> localSize = {64u, 1u, 1u};
        mQuantReduceSumPipeline = vkBn->getPipeline("glsl_dynamic_quant_reduce_sum_comp", types, localSize, {});
        mQuantReduceSumSet.reset(mQuantReduceSumPipeline->createSet());
    }
    {
        // GEMM spec constants 3..9: COOP_M, COOP_N, COOP_K, A_COL_MAJOR=0,
        // B_COL_MAJOR=1, A_BLOCK_LINEAR=1, B_BLOCK_LINEAR=0.
        std::vector<VkDescriptorType> types(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        std::vector<uint32_t> localSize = {mSubgroupSize, 1u, 1u};
        std::vector<uint32_t> spec = {mCoopM, mCoopN, mCoopK, 0u, 1u, 1u, 0u};
        mGemmS8Pipeline = vkBn->getPipeline("glsl_dynamic_w8a8_coop_gemm_comp", types, localSize, spec);
        mGemmSet.reset(mGemmS8Pipeline->createSet());
    }
    {
        std::vector<VkDescriptorType> types(8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        std::vector<uint32_t> localSize = {16u, 16u, 1u};
        std::vector<uint32_t> spec = {(uint32_t)activation};
        const char* shader = useFP16 ? "glsl_dynamic_w8a8_dequant_correction_FP16_comp"
                                     : "glsl_dynamic_w8a8_dequant_correction_comp";
        mDequantPipeline = vkBn->getPipeline(shader, types, localSize, spec);
        mDequantSet.reset(mDequantPipeline->createSet());
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
                                   const std::shared_ptr<VulkanLayout::DescriptorSet>& set,
                                   uint32_t gx, uint32_t gy, uint32_t gz,
                                   const void* pc, uint32_t pcSize) {
#ifdef ENABLE_VULKAN_TIME_PROFILE
        auto* profiler = vkBn->timeProfiler();
        if (nullptr != profiler) {
            VulkanTimeProfileScope scope(profiler, cmdBuffer->get(), name,
                                         VulkanTimeProfiler::Kind::Shader);
            pipeline->bind(cmdBuffer->get(), set->get());
            if (pc != nullptr) {
                vkCmdPushConstants(cmdBuffer->get(), pipeline->layout(),
                                   VK_SHADER_STAGE_COMPUTE_BIT, 0, pcSize, pc);
            }
            vkCmdDispatch(cmdBuffer->get(), gx, gy, gz);
            return;
        }
#else
        (void)name;
#endif
        pipeline->bind(cmdBuffer->get(), set->get());
        if (pc != nullptr) {
            vkCmdPushConstants(cmdBuffer->get(), pipeline->layout(),
                               VK_SHADER_STAGE_COMPUTE_BIT, 0, pcSize, pc);
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
        pc.K = (uint32_t)K;
        pc.N = (uint32_t)N;
        pc.blockSize = (uint32_t)K;
        pc.blockStride = 1u;
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
                            (uint32_t)N, 1, 1, &pc, sizeof(pc));
        return NO_ERROR;
    }

    const uint32_t padM = ROUND_UP(M, mCoopM);
    const uint32_t k4Count = UP_DIV((uint32_t)K, 4u);
    // 64 below mirrors K4_BLOCK in dynamic_quant_minmax.comp — keep in sync.
    const uint32_t partialBlocks = std::max(1u, UP_DIV(k4Count, 64u));
    const uint32_t tilesK = padK / mCoopK;
    const halide_type_t fpType = useFP16 ? halide_type_of<int16_t>() : halide_type_of<float>();

    std::shared_ptr<Tensor> tPartialMin(Tensor::createDevice({(int)partialBlocks, (int)padM}, fpType));
    std::shared_ptr<Tensor> tPartialMax(Tensor::createDevice({(int)partialBlocks, (int)padM}, fpType));
    std::shared_ptr<Tensor> tScaleA(Tensor::createDevice({(int)padM}, fpType));
    std::shared_ptr<Tensor> tOffsetA(Tensor::createDevice({(int)padM}, fpType));
    std::shared_ptr<Tensor> tAq(Tensor::createDevice<int8_t>({(int)padM, (int)padK}));
    std::shared_ptr<Tensor> tPartialSumAq(Tensor::createDevice<int32_t>({(int)tilesK, (int)padM}));
    std::shared_ptr<Tensor> tSumAq(Tensor::createDevice<int32_t>({(int)padM}));
    std::shared_ptr<Tensor> tAcc(Tensor::createDevice<int32_t>({(int)padM, (int)padN}));
    // INT4 only: unpacked int8 weight buffer that the GEMM reads in place of
    // mQuantWeightBuffer. Same lifetime as the other DYNAMIC tensors.
    std::shared_ptr<Tensor> tWqInt8;
    if (mIsInt4) {
        tWqInt8.reset(Tensor::createDevice<int8_t>({(int)padN, (int)padK}));
    }

    std::vector<Tensor*> dyns = {tPartialMin.get(), tPartialMax.get(), tScaleA.get(), tOffsetA.get(),
                                 tAq.get(), tPartialSumAq.get(), tSumAq.get(), tAcc.get()};
    if (mIsInt4) {
        dyns.push_back(tWqInt8.get());
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

    auto bPartialMin = vkBn->getTensorBuffer(tPartialMin.get());
    auto bPartialMax = vkBn->getTensorBuffer(tPartialMax.get());
    auto bScaleA     = vkBn->getTensorBuffer(tScaleA.get());
    auto bOffsetA    = vkBn->getTensorBuffer(tOffsetA.get());
    auto bAq         = vkBn->getTensorBuffer(tAq.get());
    auto bPartialSum = vkBn->getTensorBuffer(tPartialSumAq.get());
    auto bSumAq      = vkBn->getTensorBuffer(tSumAq.get());
    auto bAcc        = vkBn->getTensorBuffer(tAcc.get());
    const size_t szPartialMin = vkBn->getTensorSize(tPartialMin.get());
    const size_t szPartialMax = vkBn->getTensorSize(tPartialMax.get());
    const size_t szScaleA     = vkBn->getTensorSize(tScaleA.get());
    const size_t szOffsetA    = vkBn->getTensorSize(tOffsetA.get());
    const size_t szAq         = vkBn->getTensorSize(tAq.get());
    const size_t szPartialSum = vkBn->getTensorSize(tPartialSumAq.get());
    const size_t szSumAq      = vkBn->getTensorSize(tSumAq.get());
    const size_t szAcc        = vkBn->getTensorSize(tAcc.get());

    {
        struct PC { uint32_t M, K, padM, partialBlocks; } pc;
        pc.M = (uint32_t)M; pc.K = (uint32_t)K; pc.padM = padM; pc.partialBlocks = partialBlocks;

        mQuantMinMaxSet->writeBuffer(srcBuffer.first->buffer(), 0, vkBn->getTensorSize(input), srcBuffer.second);
        mQuantMinMaxSet->writeBuffer(bPartialMin.first->buffer(), 1, szPartialMin, bPartialMin.second);
        mQuantMinMaxSet->writeBuffer(bPartialMax.first->buffer(), 2, szPartialMax, bPartialMax.second);
        dispatchWithProfile(useFP16 ? "glsl_dynamic_quant_minmax_FP16_comp"
                                    : "glsl_dynamic_quant_minmax_comp",
                            mQuantMinMaxPipeline, mQuantMinMaxSet,
                            padM / mCoopM, partialBlocks, 1, &pc, sizeof(pc));
        cmdBuffer->barrierSource(bPartialMin.first->buffer(), bPartialMin.second, szPartialMin);
        cmdBuffer->barrierSource(bPartialMax.first->buffer(), bPartialMax.second, szPartialMax);
    }

    {
        struct PC { uint32_t M, padM, partialBlocks, reserved; } pc;
        pc.M = (uint32_t)M; pc.padM = padM; pc.partialBlocks = partialBlocks; pc.reserved = 0;

        mQuantReduceMinMaxSet->writeBuffer(bPartialMin.first->buffer(), 0, szPartialMin, bPartialMin.second);
        mQuantReduceMinMaxSet->writeBuffer(bPartialMax.first->buffer(), 1, szPartialMax, bPartialMax.second);
        mQuantReduceMinMaxSet->writeBuffer(bScaleA.first->buffer(), 2, szScaleA, bScaleA.second);
        mQuantReduceMinMaxSet->writeBuffer(bOffsetA.first->buffer(), 3, szOffsetA, bOffsetA.second);
        dispatchWithProfile(useFP16 ? "glsl_dynamic_quant_reduce_minmax_FP16_comp"
                                    : "glsl_dynamic_quant_reduce_minmax_comp",
                            mQuantReduceMinMaxPipeline, mQuantReduceMinMaxSet,
                            UP_DIV(padM, 64u), 1, 1, &pc, sizeof(pc));
        cmdBuffer->barrierSource(bScaleA.first->buffer(), bScaleA.second, szScaleA);
        cmdBuffer->barrierSource(bOffsetA.first->buffer(), bOffsetA.second, szOffsetA);
    }

    {
        struct PC { uint32_t M, K, padM, padK; } pc;
        pc.M = (uint32_t)M; pc.K = (uint32_t)K; pc.padM = padM; pc.padK = padK;

        mQuantPackSet->writeBuffer(srcBuffer.first->buffer(), 0, vkBn->getTensorSize(input), srcBuffer.second);
        mQuantPackSet->writeBuffer(bScaleA.first->buffer(), 1, szScaleA, bScaleA.second);
        mQuantPackSet->writeBuffer(bOffsetA.first->buffer(), 2, szOffsetA, bOffsetA.second);
        mQuantPackSet->writeBuffer(bAq.first->buffer(), 3, szAq, bAq.second);
        mQuantPackSet->writeBuffer(bPartialSum.first->buffer(), 4, szPartialSum, bPartialSum.second);
        dispatchWithProfile(useFP16 ? "glsl_dynamic_quant_pack_FP16_comp"
                                    : "glsl_dynamic_quant_pack_comp",
                            mQuantPackPipeline, mQuantPackSet,
                            padM / mCoopM, tilesK, 1, &pc, sizeof(pc));
        cmdBuffer->barrierSource(bAq.first->buffer(), bAq.second, szAq);
        cmdBuffer->barrierSource(bPartialSum.first->buffer(), bPartialSum.second, szPartialSum);
    }

    {
        struct PC { uint32_t M, padM, tilesK, reserved; } pc;
        pc.M = (uint32_t)M; pc.padM = padM; pc.tilesK = tilesK; pc.reserved = 0;

        mQuantReduceSumSet->writeBuffer(bPartialSum.first->buffer(), 0, szPartialSum, bPartialSum.second);
        mQuantReduceSumSet->writeBuffer(bSumAq.first->buffer(), 1, szSumAq, bSumAq.second);
        dispatchWithProfile("glsl_dynamic_quant_reduce_sum_comp",
                            mQuantReduceSumPipeline, mQuantReduceSumSet,
                            UP_DIV(padM, 64u), 1, 1, &pc, sizeof(pc));
        cmdBuffer->barrierSource(bSumAq.first->buffer(), bSumAq.second, szSumAq);
    }

    // INT4 only: nibble-packed weight -> int8 [padN, padK] before GEMM. The
    // O(padN*padK) write/read here trades INT4 runtime bandwidth back to INT8
    // levels, in exchange for fully reusing the W8A8 GEMM body.
    if (mIsInt4) {
        struct PC { uint32_t padN, padK, halfK; } pc;
        pc.padN = padN;
        pc.padK = padK;
        pc.halfK = padK / 2u;

        auto bWqInt8 = vkBn->getTensorBuffer(tWqInt8.get());
        const size_t szWqInt8 = vkBn->getTensorSize(tWqInt8.get());
        mInt4UnpackSet->writeBuffer(mQuantWeightBuffer->buffer(), 0, mQuantWeightBuffer->size());
        mInt4UnpackSet->writeBuffer(bWqInt8.first->buffer(), 1, szWqInt8, bWqInt8.second);
        dispatchWithProfile("glsl_dynamic_int4_to_int8_unpack_comp",
                            mInt4UnpackPipeline, mInt4UnpackSet,
                            UP_DIV(pc.halfK, 16u), UP_DIV(padN, 16u), 1, &pc, sizeof(pc));
        cmdBuffer->barrierSource(bWqInt8.first->buffer(), bWqInt8.second, szWqInt8);
    }

    {
        struct PC { uint32_t M, N, K; } pc;
        pc.M = padM; pc.N = padN; pc.K = padK;

        mGemmSet->writeBuffer(bAq.first->buffer(), 0, szAq, bAq.second);
        if (mIsInt4) {
            auto bWqInt8 = vkBn->getTensorBuffer(tWqInt8.get());
            const size_t szWqInt8 = vkBn->getTensorSize(tWqInt8.get());
            mGemmSet->writeBuffer(bWqInt8.first->buffer(), 1, szWqInt8, bWqInt8.second);
        } else {
            mGemmSet->writeBuffer(mQuantWeightBuffer->buffer(), 1, mQuantWeightBuffer->size());
        }
        mGemmSet->writeBuffer(bAcc.first->buffer(), 2, szAcc, bAcc.second);
        dispatchWithProfile("glsl_dynamic_w8a8_coop_gemm_comp",
                            mGemmS8Pipeline, mGemmSet,
                            padN / mCoopN, padM / mCoopM, 1, &pc, sizeof(pc));
        cmdBuffer->barrierSource(bAcc.first->buffer(), bAcc.second, szAcc);
    }

    {
        struct PC { uint32_t M, N, K, padM, padN; } pc;
        pc.M = (uint32_t)M; pc.N = (uint32_t)N; pc.K = (uint32_t)K; pc.padM = padM; pc.padN = padN;

        const uint32_t n4_valid = UP_DIV((uint32_t)N, 4u);
        mDequantSet->writeBuffer(bAcc.first->buffer(), 0, szAcc, bAcc.second);
        mDequantSet->writeBuffer(bScaleA.first->buffer(), 1, szScaleA, bScaleA.second);
        mDequantSet->writeBuffer(bOffsetA.first->buffer(), 2, szOffsetA, bOffsetA.second);
        mDequantSet->writeBuffer(bSumAq.first->buffer(), 3, szSumAq, bSumAq.second);
        mDequantSet->writeBuffer(mQuantMetaBuffer->buffer(), 4, mQuantMetaBuffer->size());
        mDequantSet->writeBuffer(mSumWqBuffer->buffer(), 5, mSumWqBuffer->size());
        mDequantSet->writeBuffer(mBiasBuffer->buffer(), 6, mBiasBuffer->size());
        mDequantSet->writeBuffer(dstBuffer.first->buffer(), 7, vkBn->getTensorSize(output), dstBuffer.second);
        dispatchWithProfile(useFP16 ? "glsl_dynamic_w8a8_dequant_correction_FP16_comp"
                                    : "glsl_dynamic_w8a8_dequant_correction_comp",
                            mDequantPipeline, mDequantSet,
                            UP_DIV((uint32_t)M, 16u), UP_DIV(n4_valid, 16u), 1, &pc, sizeof(pc));
    }

    for (Tensor* t : dyns) {
        vkBn->onReleaseBuffer(t, Backend::DYNAMIC);
    }

    return NO_ERROR;
}

} // namespace MNN
