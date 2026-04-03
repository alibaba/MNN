#include "VulkanConv1x1General.hpp"
#include "VulkanBackend.hpp"
#include "core/Macro.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

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

static bool _prepareQuantBuffersGPU(VulkanBackend* vkBn, const ConvolutionCommon::Int8Common* quantCommon,
                                    bool useFP16, int ci, int co, uint32_t padN, uint32_t blockStride,
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

    const char* weightShader = isInt4 ? "glsl_conv1x1_int4_weight_prepare_comp" : "glsl_conv1x1_int8_weight_prepare_comp";
    const char* metaShader = useFP16 ? "glsl_conv1x1_quant_meta_prepare_FP16_comp"
                                     : "glsl_conv1x1_quant_meta_prepare_comp";

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
        QuantWeightPrepareParams pc;
        pc.ci = static_cast<uint32_t>(ci);
        pc.co = static_cast<uint32_t>(co);
        pc.padN = padN;
        pc.weightStride = decodeWeightStrideWords;
        pc.srcBytes = static_cast<uint32_t>(rawWeightBytes);

        weightSet->writeBuffer(rawWeightBuffer->buffer(), 0, rawWeightBuffer->size());
        weightSet->writeBuffer(quantWeightBuffer->buffer(), 1, quantWeightBuffer->size());
        weightPipeline->bind(prepareCmd->get(), weightSet->get());
        vkCmdPushConstants(prepareCmd->get(), weightPipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
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
    vkBn->getPool().submitAndWait(prepareCmd->get());
    return true;
}

} // namespace

VulkanConv1x1General::VulkanConv1x1General(VulkanBackend* backend, const Convolution2DCommon* convOption,
                                           const float* biasPtr, int ci, int co,
                                           std::shared_ptr<ConvolutionCommon::Int8Common> quantInfo)
    : VulkanBasicExecution(backend), mCommon(convOption), mCi(ci), mCo(co), mQuantCommon(std::move(quantInfo)) {
    if (!_init(biasPtr, true)) {
        MNN_ERROR("VulkanConv1x1General init failed\n");
    }
}

VulkanConv1x1General::VulkanConv1x1General(VulkanBackend* backend, const Convolution2DCommon* convOption, int ci,
                                           int co, std::shared_ptr<ConvolutionCommon::Int8Common> quantInfo,
                                           bool initStaticResource)
    : VulkanBasicExecution(backend), mCommon(convOption), mCi(ci), mCo(co), mQuantCommon(std::move(quantInfo)) {
    if (!_init(nullptr, initStaticResource)) {
        MNN_ERROR("VulkanConv1x1General clone init failed\n");
    }
}

VulkanConv1x1General::~VulkanConv1x1General() {
}

bool VulkanConv1x1General::_init(const float* biasPtr, bool initStaticResource) {
    auto vkBn = static_cast<VulkanBackend*>(backend());
    if (nullptr == vkBn || nullptr == mQuantCommon.get() || nullptr == mQuantCommon->weight.get()) {
        return false;
    }

    const bool useFP16 = vkBn->useFP16();
    mIsInt4 = mQuantCommon->canUseInt4;
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

    if ((mBlockSize & 3u) != 0u) {
        MNN_ERROR("VulkanConv1x1General requires blockSize %% 4 == 0, blockSize=%u\n", mBlockSize);
        return false;
    }
    if (mCi % static_cast<int>(mBlockSize) != 0) {
        MNN_ERROR("VulkanConv1x1General requires K %% blockSize == 0, K=%d, blockSize=%u\n", mCi, mBlockSize);
        return false;
    }
    if ((mPadK % mBlockSize) != 0u) {
        MNN_ERROR("VulkanConv1x1General requires padK %% blockSize == 0, padK=%u, blockSize=%u\n", mPadK, mBlockSize);
        return false;
    }
    mBlockStride = mPadK / mBlockSize;
    mDecodeWeightStrideWords = mIsInt4 ? UP_DIV(mPadK, 8u) : (mPadK / 4u);

    if (initStaticResource) {
        if (!_prepareQuantBuffersGPU(vkBn, mQuantCommon.get(), useFP16, mCi, mCo, mPadN, mBlockStride,
                                     mDecodeWeightStrideWords, mIsInt4, mQuantWeightBuffer, mQuantMetaBuffer)) {
            return false;
        }

        std::vector<float> biasHost(mPadN, 0.0f);
        if (nullptr != biasPtr) {
            ::memcpy(biasHost.data(), biasPtr, static_cast<size_t>(mCo) * sizeof(float));
        }
        if (useFP16) {
            std::vector<int16_t> biasHalf(mPadN);
            FLOAT_TO_HALF(biasHost.data(), biasHalf.data(), static_cast<int>(mPadN));
            mBiasBuffer.reset(new VulkanBuffer(vkBn->getMemoryPool(), false, mPadN * sizeof(int16_t), nullptr,
                                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                   VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                               VK_SHARING_MODE_EXCLUSIVE, 0));
            vkBn->copyToGPUBuffer(biasHalf.data(), mBiasBuffer->buffer(), mPadN * sizeof(int16_t), 0);
        } else {
            mBiasBuffer.reset(new VulkanBuffer(vkBn->getMemoryPool(), false, mPadN * sizeof(float), nullptr,
                                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                   VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                               VK_SHARING_MODE_EXCLUSIVE, 0));
            vkBn->copyToGPUBuffer(biasHost.data(), mBiasBuffer->buffer(), mPadN * sizeof(float), 0);
        }
    }

    int activation = 0;
    if (mCommon->relu()) {
        activation = 1;
    }
    if (mCommon->relu6()) {
        activation = 2;
    }

    mDecodeSubgroupSize = vkBn->getDevice().getSubgroupSize();
    if (mDecodeSubgroupSize == 0u) {
        mDecodeSubgroupSize = 64u;
    }

    {
        std::vector<VkDescriptorType> types = {
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        };
        std::vector<uint32_t> spec = {static_cast<uint32_t>(activation)};
        const char* shader = nullptr;
        if (mIsInt4) {
            shader = useFP16 ? "glsl_gemv_dequant_int4_FP16_comp" : "glsl_gemv_dequant_int4_comp";
        } else {
            shader = useFP16 ? "glsl_gemv_dequant_int8_FP16_comp" : "glsl_gemv_dequant_int8_comp";
        }
        mDecodePipeline = vkBn->getPipeline(shader, types, {mDecodeSubgroupSize, 1, 1}, spec);
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
        if (mIsInt4) {
            shader = useFP16 ? "glsl_int4_weight_to_pack_FP16_comp" : "glsl_int4_weight_to_pack_comp";
        } else {
            shader = useFP16 ? "glsl_int8_weight_to_pack_FP16_comp" : "glsl_int8_weight_to_pack_comp";
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
    auto conv2D = op->main_as_Convolution2D();
    if (nullptr == conv2D || nullptr == conv2D->common()) {
        return false;
    }
    auto res = new VulkanConv1x1General(vkBn, conv2D->common(), mCi, mCo, mQuantCommon, false);
    res->mIsInt4 = mIsInt4;
    res->mPadK = mPadK;
    res->mPadN = mPadN;
    res->mBlockSize = mBlockSize;
    res->mBlockStride = mBlockStride;
    res->mDecodeWeightStrideWords = mDecodeWeightStrideWords;
    res->mDecodeSubgroupSize = mDecodeSubgroupSize;
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
        mDecodePipeline->bind(cmdBuffer->get(), mDecodeSet->get());
        vkCmdPushConstants(cmdBuffer->get(), mDecodePipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(DecodeParams), &pc);
        vkCmdDispatch(cmdBuffer->get(), static_cast<uint32_t>(mCo), 1, 1);
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
        mPackAPipeline->bind(cmdBuffer->get(), mPackASet->get());
        vkCmdPushConstants(cmdBuffer->get(), mPackAPipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(PackAParams), &pc);
        vkCmdDispatch(cmdBuffer->get(), mPadK / 4u, padM / 64u, 1);
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
        pc.K = mPadK;
        pc.blockSize = mBlockSize;
        pc.KBlocks = mBlockStride;

        mWeightToPackSet->writeBuffer(mQuantWeightBuffer->buffer(), 0, mQuantWeightBuffer->size());
        mWeightToPackSet->writeBuffer(mQuantMetaBuffer->buffer(), 1, mQuantMetaBuffer->size());
        mWeightToPackSet->writeBuffer(packedBBuffer.first->buffer(), 2, packedBSize, packedBBuffer.second);
        mWeightToPackPipeline->bind(cmdBuffer->get(), mWeightToPackSet->get());
        vkCmdPushConstants(cmdBuffer->get(), mWeightToPackPipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(WeightToPackParams), &pc);
        vkCmdDispatch(cmdBuffer->get(), UP_DIV(mPadN / 4u, 16u), UP_DIV(mPadK / 4u, 8u), 1);
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
        mGemmPipeline->bind(cmdBuffer->get(), mGemmSet->get());
        vkCmdPushConstants(cmdBuffer->get(), mGemmPipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(GemmParams), &pc);
        vkCmdDispatch(cmdBuffer->get(), mPadN / 32u, padM / 64u, 1);
    }

    releaseTemp();
    return NO_ERROR;
}

} // namespace MNN
