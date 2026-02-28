#include "VulkanConv1x1Coop.hpp"
#include "core/TensorUtils.hpp"
#include "core/Macro.h"
#include "VulkanBackend.hpp"
#include "backend/vulkan/vulkan/vulkan_wrapper.h"
#include <algorithm>

namespace MNN {

VulkanConv1x1Coop::VulkanConv1x1Coop(VulkanBackend* backend, const Convolution2DCommon* convOption, const float* weightPtr, const float* biasPtr, int ci, int co, VulkanDevice::CoopMatInfo coopMatInfo, std::shared_ptr<ConvolutionCommon::Int8Common> quantInfo, std::shared_ptr<ConvolutionCommon::Int8Common> weightHolder)
    : VulkanBasicExecution(backend), mCommon(convOption), mCi(ci), mCo(co), mIsQuant(quantInfo != nullptr), mQuantCommon(std::move(quantInfo)), mWeightFloatHolder(std::move(weightHolder)) {
    const std::vector<uint32_t> & selectedShape = backend->useFP16() ? (coopMatInfo.selectedFP16CoopMatShape) : (coopMatInfo.selectedFP32CoopMatShape);
    COOP_M = selectedShape[0];
    COOP_N = selectedShape[1];
    COOP_K = selectedShape[2];
    uint32_t subgroupSize = backend->getDevice().getSubgroupSize();
    mSubgroupSize = subgroupSize;
    _init(weightPtr, biasPtr, true);
}

VulkanConv1x1Coop::VulkanConv1x1Coop(VulkanBackend* backend, const Convolution2DCommon* convOption, int ci, int co,
                                     uint32_t coopM, uint32_t coopN, uint32_t coopK, uint32_t subgroupSize,
                                     std::shared_ptr<ConvolutionCommon::Int8Common> quantInfo,
                                     std::shared_ptr<ConvolutionCommon::Int8Common> weightHolder,
                                     bool initStaticResource)
    : VulkanBasicExecution(backend),
      mCommon(convOption),
      mCi(ci),
      mCo(co),
      mIsQuant(quantInfo != nullptr),
      mQuantCommon(std::move(quantInfo)),
      mWeightFloatHolder(std::move(weightHolder)) {
    COOP_M = coopM;
    COOP_N = coopN;
    COOP_K = coopK;
    mSubgroupSize = subgroupSize;
    _init(nullptr, nullptr, initStaticResource);
}

VulkanConv1x1Coop::~VulkanConv1x1Coop() {
}

bool VulkanConv1x1Coop::onClone(Backend* bn, const Op* op, VulkanBasicExecution** dst) {
    if (nullptr == dst) {
        return true;
    }
    auto vkBn = static_cast<VulkanBackend*>(bn);
    auto conv2D = op->main_as_Convolution2D();
    if (nullptr == conv2D || nullptr == conv2D->common()) {
        return false;
    }
    auto res = new VulkanConv1x1Coop(vkBn, conv2D->common(), mCi, mCo, COOP_M, COOP_N, COOP_K, mSubgroupSize, mQuantCommon,
                                     mWeightFloatHolder, false);
    res->mPadK = mPadK;
    res->mPadN = mPadN;
    res->mBlockSize = mBlockSize;
    res->mQuantConverted = mQuantConverted;
    res->mWeightBuffer = mWeightBuffer;
    res->mBiasBuffer = mBiasBuffer;
    res->mQuantWeightBuffer = mQuantWeightBuffer;
    res->mQuantScaleBuffer = mQuantScaleBuffer;
    res->mQuantOffsetBuffer = mQuantOffsetBuffer;
    *dst = res;
    return true;
}

bool VulkanConv1x1Coop::_init(const float* weightPtr, const float* biasPtr, bool initStaticResource) {
    auto vkBn = static_cast<VulkanBackend*>(backend());
    bool useFP16 = vkBn->useFP16();

    // Prepare Weights (Reorder to Block-Linear)
    // [N, K] ---> [K/COOP_K, N/COOP_N, COOP_K, COOP_N]
    uint32_t K = mCi;
    uint32_t N = mCo;
    mPadK = ROUND_UP(K, COOP_K);
    mPadN = ROUND_UP(N, COOP_N);
    
    size_t elementSize = useFP16 ? sizeof(int16_t) : sizeof(float);
    size_t weightSize = mPadK * mPadN;

    if (initStaticResource && !mIsQuant) {
        mWeightBuffer = std::make_shared<VulkanBuffer>(vkBn->getMemoryPool(), false, elementSize * weightSize, nullptr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        auto weightMap = mWeightBuffer->map();
        std::vector<uint8_t> hostWeights;
        if (useFP16) {
            hostWeights.resize(weightSize * elementSize);
        }
        auto ptrFP16 = reinterpret_cast<int16_t*>(hostWeights.data());
        auto ptrFP32 = reinterpret_cast<float*>(weightMap);

        uint32_t tilesK = mPadK / COOP_K;
        uint32_t tilesN = mPadN / COOP_N;

        for (uint32_t n = 0; n < mPadN; ++n) {
            uint32_t tn = n / COOP_N;
            uint32_t col = n % COOP_N;
            for (uint32_t k = 0; k < mPadK; ++k) {
                uint32_t tk = k / COOP_K;
                uint32_t row = k % COOP_K;
                float val = 0.0f;
                if (nullptr != weightPtr && k < K && n < N) {
                    val = weightPtr[n * K + k];
                }
                uint32_t dstIdx = (tk * tilesN + tn) * (COOP_K * COOP_N) + row * COOP_N + col;
                if (useFP16) {
                    ((half_float::half *)ptrFP16)[dstIdx] = (half_float::half) val;
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

    // Prepare Bias
    // [N] -> [PadN]
    if (initStaticResource) {
        mBiasBuffer = std::make_shared<VulkanBuffer>(vkBn->getMemoryPool(), false, elementSize * mPadN, nullptr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
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

    // Prepare quant buffers (static) and compute pipeline
    if (mIsQuant) {
        int soSize = mQuantCommon->asymmetric ? 2 : 1;
        int alphaSize = mQuantCommon->alpha.size();
        int blockCount = std::max(1, alphaSize / (mCo * soSize));
        mBlockSize = UP_DIV(mCi, blockCount);
        uint32_t blocksPerRow = UP_DIV(mPadK, mBlockSize);

        if (initStaticResource) {
            // Reorder quant weight to [K-major, padN]
            if (mQuantCommon->canUseInt4) {
                std::vector<int8_t> tempInt8(mPadK * mPadN, 0);
                const int8_t* src = mQuantCommon->weight.get();
                int weightCount = mQuantCommon->weight.size() * 2; // 2 int4 per byte
                for (uint32_t n = 0; n < (uint32_t)mCo; ++n) {
                    for (uint32_t k = 0; k < (uint32_t)mCi; ++k) {
                        int idx = n * mCi + k;
                        if (idx >= weightCount) { continue; }
                        int byteIdx = idx >> 1;
                        int nibbleShift = (idx & 1) == 0 ? 4 : 0;
                        int v = (src[byteIdx] >> nibbleShift) & 0xF;
                        v -= 8;
                        tempInt8[k * mPadN + n] = (int8_t)v;
                    }
                }
                size_t packedBytes = (mPadK * mPadN + 1) / 2;
                std::vector<uint8_t> packed(packedBytes, 0);
                for (uint32_t idx = 0; idx < mPadK * mPadN; idx += 2) {
                    int8_t v0 = tempInt8[idx];
                    int8_t v1 = (idx + 1 < mPadK * mPadN) ? tempInt8[idx + 1] : 0;
                    uint8_t b = (uint8_t)((v0 + 8) << 4) | (uint8_t)(v1 + 8);
                    packed[idx >> 1] = b;
                }
                mQuantWeightBuffer.reset(new VulkanBuffer(vkBn->getMemoryPool(), false, packedBytes, nullptr,
                                                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                          VK_SHARING_MODE_EXCLUSIVE, 0));
                vkBn->copyToGPUBuffer(packed.data(), mQuantWeightBuffer->buffer(), packedBytes, 0);
            } else {
                std::vector<int8_t> reordered(mPadK * mPadN, 0);
                const int8_t* src = mQuantCommon->weight.get();
                for (uint32_t n = 0; n < (uint32_t)mCo; ++n) {
                    for (uint32_t k = 0; k < (uint32_t)mCi; ++k) {
                        int idx = n * mCi + k;
                        reordered[k * mPadN + n] = src[idx];
                    }
                }
                mQuantWeightBuffer.reset(new VulkanBuffer(vkBn->getMemoryPool(), false, reordered.size(), nullptr,
                                                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                          VK_SHARING_MODE_EXCLUSIVE, 0));
                vkBn->copyToGPUBuffer(reordered.data(), mQuantWeightBuffer->buffer(), reordered.size(), 0);
            }

            // Reorder scale / offset to [blocksPerRow, padN]
            size_t soElem = blocksPerRow * mPadN;
            std::vector<float> scaleHost(soElem, 0.0f);
            std::vector<float> offsetHost(soElem, 0.0f);
            const float* alphaPtr = mQuantCommon->alpha.get();
            for (int n = 0; n < mCo; ++n) {
                for (int b = 0; b < blockCount; ++b) {
                    int alphaBase = soSize * (n * blockCount + b);
                    float offset = mQuantCommon->asymmetric ? alphaPtr[alphaBase] : 0.0f;
                    float scale  = alphaPtr[alphaBase + (mQuantCommon->asymmetric ? 1 : 0)];
                    int dstIdx = b * mPadN + n;
                    scaleHost[dstIdx] = scale;
                    offsetHost[dstIdx] = offset;
                }
            }
            // Convert to target precision and upload
            if (useFP16) {
                std::vector<int16_t> scaleHalf(soElem), offsetHalf(soElem);
                FLOAT_TO_HALF(scaleHost.data(), scaleHalf.data(), soElem);
                FLOAT_TO_HALF(offsetHost.data(), offsetHalf.data(), soElem);
                mQuantScaleBuffer.reset(new VulkanBuffer(vkBn->getMemoryPool(), false, soElem * sizeof(int16_t), nullptr,
                                                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                         VK_SHARING_MODE_EXCLUSIVE, 0));
                mQuantOffsetBuffer.reset(new VulkanBuffer(vkBn->getMemoryPool(), false, soElem * sizeof(int16_t), nullptr,
                                                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                          VK_SHARING_MODE_EXCLUSIVE, 0));
                vkBn->copyToGPUBuffer(scaleHalf.data(), mQuantScaleBuffer->buffer(), soElem * sizeof(int16_t), 0);
                vkBn->copyToGPUBuffer(offsetHalf.data(), mQuantOffsetBuffer->buffer(), soElem * sizeof(int16_t), 0);
            } else {
                mQuantScaleBuffer.reset(new VulkanBuffer(vkBn->getMemoryPool(), false, soElem * sizeof(float), nullptr,
                                                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                         VK_SHARING_MODE_EXCLUSIVE, 0));
                mQuantOffsetBuffer.reset(new VulkanBuffer(vkBn->getMemoryPool(), false, soElem * sizeof(float), nullptr,
                                                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                          VK_SHARING_MODE_EXCLUSIVE, 0));
                vkBn->copyToGPUBuffer(scaleHost.data(), mQuantScaleBuffer->buffer(), soElem * sizeof(float), 0);
                vkBn->copyToGPUBuffer(offsetHost.data(), mQuantOffsetBuffer->buffer(), soElem * sizeof(float), 0);
            }
        }

        // Dequant pipeline
        std::vector<VkDescriptorType> types = {
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
        };
        std::vector<uint32_t> localSize = {mSubgroupSize * 4, 1, 1};
        std::vector<uint32_t> spec = {COOP_K, COOP_N};
        const char* shader = nullptr;
        if (mQuantCommon->canUseInt4) {
            shader = useFP16 ? "glsl_int4_weight_to_coop_FP16_comp" : "glsl_int4_weight_to_coop_comp";
        } else {
            shader = useFP16 ? "glsl_int8_weight_to_coop_FP16_comp" : "glsl_int8_weight_to_coop_comp";
        }
        mDequantPipeline = vkBn->getPipeline(shader, types, localSize, spec);
        mDequantSet.reset(mDequantPipeline->createSet());
    }

    // Create Pipelines
    // A. Pack Pipeline (C4 -> Coop)
    {
        std::vector<VkDescriptorType> types = {
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // Src
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // Dst
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER  // Const
        };
        std::vector<uint32_t> localSize = { mSubgroupSize * 4, 1, 1 };
        std::vector<uint32_t> packSpec = { COOP_M, COOP_K };
        std::string shader = useFP16 ? "glsl_C4_to_COOP_FP16_comp" : "glsl_C4_to_COOP_comp";
        mPackPipeline = vkBn->getPipeline(shader, types, localSize, packSpec);
        mPackSet.reset(mPackPipeline->createSet());
    }

    // MatMul Pipeline
    {
        std::vector<VkDescriptorType> types = {
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // A
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // B
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // Bias
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // Output
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER  // Const
        };
        std::vector<uint32_t> localSize = { mSubgroupSize, 1, 1 };
        std::vector<uint32_t> matmulSpec = { COOP_M, COOP_N, COOP_K };
        std::string shader = useFP16 ? "glsl_matmul_coop_FP16_comp" : "glsl_matmul_coop_comp";
        mMatMulPipeline = vkBn->getPipeline(shader, types, localSize, matmulSpec);
        mMatMulSet.reset(mMatMulPipeline->createSet());
    }

    // Unpack Pipeline (Coop -> C4)
    {
        std::vector<VkDescriptorType> types = {
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // Src
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // Dst
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER  // Const
        };
        
        // Calculate Local Size
        std::vector<uint32_t> localSize = { mSubgroupSize, 4, 1 };
        
        // Activation: 0=None, 1=ReLU, 2=ReLU6
        int activation = 0;
        if (mCommon->relu()) activation = 1;
        if (mCommon->relu6()) activation = 2;
        
        // Spec Constants: ID 3 (ACTIVATION)
        std::vector<uint32_t> unpackSpec = { (uint32_t)activation };

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
    
    // Dimensions
    int batch = input->batch();
    int width = input->width();
    int height = input->height();
    int M = batch * width * height;
    int K = mCi;
    int N = mCo;
    
    uint32_t padM = ROUND_UP(M, COOP_M);
    uint32_t padK = mPadK;
    uint32_t padN = mPadN;
    
    if (vkBn->useFP16()) {
        mTempInput.reset(Tensor::createDevice<int16_t>({(int)padM, (int)padK}));
        mTempOutput.reset(Tensor::createDevice<int16_t>({(int)padM, (int)padN}));
    } else {
        mTempInput.reset(Tensor::createDevice<float>({(int)padM, (int)padK}));
        mTempOutput.reset(Tensor::createDevice<float>({(int)padM, (int)padN}));
    }
    auto res = vkBn->onAcquireBuffer(mTempInput.get(), Backend::DYNAMIC);
    if (!res) return OUT_OF_MEMORY;
    res = vkBn->onAcquireBuffer(mTempOutput.get(), Backend::DYNAMIC);
    if (!res) return OUT_OF_MEMORY;

    // Quant temp weight
    std::pair<const VulkanBuffer*, size_t> weightBufferPair;
    size_t weightBufferSize = 0;
    if (mIsQuant) {
        if (!mTempWeight) {
            if (vkBn->useFP16()) {
                mTempWeight.reset(Tensor::createDevice<int16_t>({(int)padK, (int)padN}));
            } else {
                mTempWeight.reset(Tensor::createDevice<float>({(int)padK, (int)padN}));
            }
        }
        res = vkBn->onAcquireBuffer(mTempWeight.get(), Backend::DYNAMIC);
        if (!res) return OUT_OF_MEMORY;
        weightBufferPair = vkBn->getTensorBuffer(mTempWeight.get());
        weightBufferSize = vkBn->getTensorSize(mTempWeight.get());
    } else {
        weightBufferPair = {mWeightBuffer.get(), 0};
        weightBufferSize = mWeightBuffer->size();
    }

    auto srcBuffer = vkBn->getTensorBuffer(input);
    auto dstBuffer = vkBn->getTensorBuffer(output);
    auto tempInBuffer = vkBn->getTensorBuffer(mTempInput.get());
    auto tempOutBuffer = vkBn->getTensorBuffer(mTempOutput.get());

    // 0. Dequantize weight to coop layout when using quantized weights
    if (mIsQuant) {
        struct DequantParams { uint32_t K; uint32_t N; uint32_t padK; uint32_t padN; uint32_t blockSize; } pc;
        pc.K = (uint32_t)K;
        pc.N = padN; // stride for Q/scale/offset is padded N
        pc.padK = padK;
        pc.padN = padN;
        pc.blockSize = mBlockSize;
        mDequantSet->writeBuffer(mQuantWeightBuffer->buffer(), 0, mQuantWeightBuffer->size());
        mDequantSet->writeBuffer(mQuantScaleBuffer->buffer(), 1, mQuantScaleBuffer->size());
        mDequantSet->writeBuffer(mQuantOffsetBuffer->buffer(), 2, mQuantOffsetBuffer->size());
        mDequantSet->writeBuffer(weightBufferPair.first->buffer(), 3, weightBufferSize, weightBufferPair.second);
        mDequantPipeline->bind(cmdBuffer->get(), mDequantSet->get());
        vkCmdPushConstants(cmdBuffer->get(), mDequantPipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(DequantParams), &pc);
        vkCmdDispatch(cmdBuffer->get(), padN / COOP_N, padK / COOP_K, 1);
        cmdBuffer->barrierSource(weightBufferPair.first->buffer(), weightBufferPair.second, weightBufferSize);
    }

    // 1. Pack (C4 -> Coop)
    {
        struct PackParams { uint32_t M; uint32_t K; uint32_t padM; uint32_t padK; } pc;
        pc.M = M; pc.K = K; pc.padM = padM; pc.padK = padK;
        mPackConst = vkBn->allocUniform(&pc, sizeof(pc));
        mPackSet->writeBuffer(srcBuffer.first->buffer(), 0, vkBn->getTensorSize(input), srcBuffer.second);
        mPackSet->writeBuffer(tempInBuffer.first->buffer(), 1, vkBn->getTensorSize(mTempInput.get()), tempInBuffer.second);
        mPackSet->writeBuffer(mPackConst->buffer(), 2, mPackConst->size());
        mPackPipeline->bind(cmdBuffer->get(), mPackSet->get());
        vkCmdDispatch(cmdBuffer->get(), padK / COOP_K, padM / COOP_M, 1);
        cmdBuffer->barrierSource(tempInBuffer.first->buffer(), tempInBuffer.second, vkBn->getTensorSize(mTempInput.get()));
    }

    // 2. MatMul (Coop)
    {
        struct MatMulParams { uint32_t M; uint32_t N; uint32_t K; uint32_t padding; } pc;
        pc.M = padM; pc.N = padN; pc.K = padK; pc.padding = 0;
        
        mMatMulConst = vkBn->allocUniform(&pc, sizeof(pc));

        mMatMulSet->writeBuffer(tempInBuffer.first->buffer(), 0, vkBn->getTensorSize(mTempInput.get()), tempInBuffer.second);
        mMatMulSet->writeBuffer(weightBufferPair.first->buffer(), 1, weightBufferSize, weightBufferPair.second);
        mMatMulSet->writeBuffer(mBiasBuffer->buffer(), 2, mBiasBuffer->size());
        mMatMulSet->writeBuffer(tempOutBuffer.first->buffer(), 3, vkBn->getTensorSize(mTempOutput.get()), tempOutBuffer.second);
        mMatMulSet->writeBuffer(mMatMulConst->buffer(), 4, mMatMulConst->size());
        mMatMulPipeline->bind(cmdBuffer->get(), mMatMulSet->get());
        vkCmdDispatch(cmdBuffer->get(), padN / COOP_N, padM / COOP_M, 1);
        cmdBuffer->barrierSource(tempOutBuffer.first->buffer(), tempOutBuffer.second, vkBn->getTensorSize(mTempOutput.get()));
    }

    // 3. Unpack (Coop -> C4)
    {
        struct UnpackParams { uint32_t M; uint32_t N; uint32_t padM; uint32_t padN; } pc;
        pc.M = M; pc.N = N; pc.padM = padM; pc.padN = padN;
        
        mUnpackConst = vkBn->allocUniform(&pc, sizeof(pc));

        mUnpackSet->writeBuffer(tempOutBuffer.first->buffer(), 0, vkBn->getTensorSize(mTempOutput.get()), tempOutBuffer.second);
        mUnpackSet->writeBuffer(dstBuffer.first->buffer(), 1, vkBn->getTensorSize(output), dstBuffer.second);
        mUnpackSet->writeBuffer(mUnpackConst->buffer(), 2, mUnpackConst->size());
        mUnpackPipeline->bind(cmdBuffer->get(), mUnpackSet->get());
        vkCmdDispatch(cmdBuffer->get(), ROUND_UP(padN, 32) / 32, ROUND_UP(padM, 32) / 32, 1);
    }

    vkBn->onReleaseBuffer(mTempInput.get(), Backend::DYNAMIC);
    vkBn->onReleaseBuffer(mTempOutput.get(), Backend::DYNAMIC);
    if (mIsQuant) {
        vkBn->onReleaseBuffer(mTempWeight.get(), Backend::DYNAMIC);
    }

    return NO_ERROR;
}

} // namespace MNN
