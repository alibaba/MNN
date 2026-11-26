//
//  CPUIndexedLinear.cpp
//  MNN
//
//  Created by MNN on 2026/09/29.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include "CPUIndexedLinear.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "backend/cpu/compute/ConvInt8TiledExecutor.hpp"
#include "core/Concurrency.h"
#include "core/IndexedLinearCommon.hpp"
#include "core/SimdHeader.h"
#include "core/TensorUtils.hpp"
#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <limits>

namespace MNN {
namespace {
const CoreFunctions* indexedLinearCore(Backend* backend) {
    auto cpu = static_cast<CPUBackend*>(backend);
    // IndexedLinear computes FP32 tensors in BF16 precision; it does not use the dense BF16 packers.
    return cpu->precisionMode() == BackendConfig::Precision_Low_BF16 ? MNNGetCoreFunctions() : cpu->functions();
}

#ifdef MNN_USE_NEON
// vuzp1q/vuzp2q are AArch64-only; the portable vuzpq pair deinterleaves identically on both ABIs.
static inline void splitGateUp(const uint16_t* src, uint16_t* gate, uint16_t* up) {
    auto v = vld1q_u16(src);
    auto p = vuzpq_u16(v, v);
    vst1_u16(gate, vget_low_u16(p.val[0]));
    vst1_u16(up, vget_low_u16(p.val[1]));
}
static inline void splitGateUp(const uint32_t* src, uint32_t* gate, uint32_t* up) {
    auto v = vld1q_u32(src);
    auto p = vuzpq_u32(v, v);
    vst1_u32(gate, vget_low_u32(p.val[0]));
    vst1_u32(up, vget_low_u32(p.val[1]));
}
#endif

template <typename T>
void unpackGateUpTile(const uint8_t* source, uint8_t* gates, uint8_t* ups, int channels, int rows, int pack) {
    auto src = reinterpret_cast<const T*>(source);
    auto gate = reinterpret_cast<T*>(gates);
    auto up = reinterpret_cast<T*>(ups);
    const int halfPack = pack / 2;
#ifdef MNN_USE_NEON
    const int chunk = 8 / sizeof(T);
    if (halfPack % chunk == 0) {
        for (int c = 0; c < channels; c += pack) {
            for (int half = 0; half < 2; ++half) {
                auto base = src + (2 * c + half * pack) * rows;
                auto gb = gate + c * rows + half * halfPack;
                auto ub = up + c * rows + half * halfPack;
                int r = 0;
                for (; r + 1 < rows; r += 2) {
                    for (int k = 0; k < halfPack; k += chunk) {
                        splitGateUp(base + r * pack + 2 * k, gb + r * pack + k, ub + r * pack + k);
                        splitGateUp(base + (r + 1) * pack + 2 * k, gb + (r + 1) * pack + k,
                                    ub + (r + 1) * pack + k);
                    }
                }
                for (; r < rows; ++r) {
                    for (int k = 0; k < halfPack; k += chunk) {
                        splitGateUp(base + r * pack + 2 * k, gb + r * pack + k, ub + r * pack + k);
                    }
                }
            }
        }
        return;
    }
#endif
    for (int c = 0; c < channels; c += pack) {
        for (int r = 0; r < rows; ++r) {
            for (int half = 0; half < 2; ++half) {
                const auto input = src + (2 * c + half * pack) * rows + r * pack;
                const int dst = c * rows + r * pack + half * halfPack;
                for (int i = 0; i < halfPack; ++i) {
                    gate[dst + i] = input[2 * i];
                    up[dst + i] = input[2 * i + 1];
                }
            }
        }
    }
}
} // namespace

struct CPUIndexedLinear::Resource {
    struct Expert {
        std::shared_ptr<Tensor> weight;
        std::shared_ptr<Tensor> bias;
        std::shared_ptr<Tensor> weightSum;
        std::shared_ptr<Tensor> denseBranchWeight;
        std::shared_ptr<Tensor> denseBranchSum;
        std::shared_ptr<Tensor> denseDecodeWeight;
        std::shared_ptr<Tensor> denseDecodeSum;
        int denseDecodeTileBytes = 0;
        int weightTileBytes = 0;
        int branchTileBytes = 0;
        int weightQuantInfoMode = 0;
    };
    const CoreFunctions* core = nullptr;
    std::vector<Expert> experts;
    int inputChannels = 0;
    int outputChannels = 0;
    int bytes = 0;
    int pack = 0;
    int ePack = 0;
    int lPack = 0;
    int hPack = 0;
    bool quantized = false;
    int weightBits = 0;
    int blocks = 1;
    int blockChannels = 0;
    int mainChannels = 0;
    int branchHPack = 0;
    int smeCores = 0;
    int alignedWeightOutput = 0;
    int denseMainDecode = 0;
    int denseMainPrefill = 0;
    int denseThreads = 0;
    int denseDivisionRatio = 0;
    int denseDynamicOption = 0;
    BackendConfig::MemoryMode denseMemory = BackendConfig::Memory_Normal;
    MatmulRelatedFunctions quantFunctions;
    decltype(MatmulRelatedFunctions::Int8GemmKernel) quantGemm = nullptr;
    decltype(MatmulRelatedFunctions::Int8GemmKernel) branchGemm = nullptr;
    decltype(MatmulRelatedFunctions::Int8GemmKernel) denseDecodeGemm = nullptr;
    decltype(CoreInt8Functions::MNNFloat2Int8) quantInput = nullptr;
    MNNUnaryExecute siluFunction = nullptr;
    MNNBinaryExecute mulFunction = nullptr;
    float clamp[2] = {-std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
};

CPUIndexedLinear::CPUIndexedLinear(Backend* backend, const Op* op) : Execution(backend) {
    mDenseGateUp = op->main_type() == OpParameter_FusedLinearParam;
    const auto param = mDenseGateUp ? nullptr : op->main_as_IndexedLinearParam();
    mRouteWeight = param ? param->routeWeight() : 0;
    mRouteEps = param ? param->routeEps() : 0.0f;
    mRouteScale = param ? param->routeScale() : 1.0f;
    auto core = indexedLinearCore(backend);
    mResource.reset(new Resource);
    auto& resource = *mResource;
    bool forceFloat = mDenseGateUp && static_cast<CPUBackend*>(backend)->memoryMode() != BackendConfig::Memory_Low;
#ifndef MNN_LOW_MEMORY
    forceFloat = true;
#endif
    resource.core = core;
    resource.bytes = core->bytes;
    resource.pack = core->pack;
    mInputMode = param ? param->inputMode() : 0;
    mFuseSwiGLU = mDenseGateUp || param->fusedActivation() == UnaryOpOperation_SILU;
    if (mFuseSwiGLU) {
        const int precision = static_cast<CPUBackend*>(backend)->precisionMode();
        resource.siluFunction = core->MNNSelectUnaryFunctionForFloat(UnaryOpOperation_SILU, precision);
        resource.mulFunction = core->MNNSelectBinaryFunctionForFloat(BinaryOpOperation_MUL);
        if (!resource.siluFunction || !resource.mulFunction) {
            mValid = false;
            return;
        }
    }
    int experts;
    if (!IndexedLinearCommon::dimensions(op, experts, resource.inputChannels, resource.outputChannels)) {
        mValid = false;
        return;
    }
    const int inputChannels = resource.inputChannels;
    const int outputChannels = resource.outputChannels;
    if (mDenseGateUp) {
        auto cpu = static_cast<CPUBackend*>(backend);
        resource.denseThreads = cpu->threadNumber();
        resource.denseDivisionRatio = backend->getRuntime()->hint().divisionRatio;
        resource.denseDynamicOption = backend->getRuntime()->hint().dynamicQuantOption;
        resource.denseMemory = cpu->memoryMode();
    }
    auto firstWeight = IndexedLinearCommon::loadLinear(op, 0, backend, forceFloat);
    if (!firstWeight) {
        mValid = false;
        return;
    }
    resource.quantized = !firstWeight->weightFloat.get();
    resource.mainChannels = outputChannels;
    if (resource.quantized) {
#ifdef MNN_LOW_MEMORY
        // The lowp integer GEMM entry points write FP16, not BF16.
        if (resource.bytes == 2 && resource.pack != 8) {
            mValid = false;
            return;
        }
        resource.quantFunctions = *static_cast<CPUBackend*>(backend)->int8GemmFunctions();
        const auto int8Core = static_cast<CPUBackend*>(backend)->int8Functions();
        resource.quantInput = resource.bytes == 2 ? int8Core->DynamicQuanInput_ARM82 : int8Core->MNNFloat2Int8;
        resource.quantFunctions.MNNGetGemmUnit(&resource.hPack, &resource.lPack, &resource.ePack);
        resource.weightBits = firstWeight->canUseInt4 ? 4 : 8;
        const int scaleCount = firstWeight->alphaSize / (firstWeight->asymmetric ? 2 : 1);
        if (scaleCount <= 0 || scaleCount % outputChannels != 0) {
            mValid = false;
            return;
        }
        resource.blocks = scaleCount / outputChannels;
        resource.blockChannels = inputChannels / resource.blocks;
        // A quantization block must have the same LP boundary in the activation
        // packer and weight packer. Reject unsupported block tails explicitly.
        if (inputChannels % resource.blocks) {
            mValid = false;
            return;
        }
        if (resource.blockChannels % resource.lPack) {
            if (mDenseGateUp) {
                mValid = false;
                return;
            }
            forceFloat = true;
            firstWeight = IndexedLinearCommon::loadLinear(op, 0, backend, forceFloat);
            if (!firstWeight || !firstWeight->weightFloat.get()) {
                mValid = false;
                return;
            }
            resource.quantized = false;
            resource.blocks = 1;
            resource.blockChannels = inputChannels;
        }
#else
        mValid = false;
        return;
#endif
    }
    if (resource.quantized) {
#ifdef MNN_LOW_MEMORY
        if (resource.bytes == 2) {
            resource.quantGemm = resource.weightBits == 4 ? resource.quantFunctions.MNNGemmInt8AddBiasScale_w4_Unit_FP16
                                                          : resource.quantFunctions.MNNGemmInt8AddBiasScale_Unit_FP16;
            core->MNNFp32ToLowp(resource.clamp, reinterpret_cast<int16_t*>(resource.clamp), 2);
        } else {
            resource.quantGemm = resource.weightBits == 4 ? resource.quantFunctions.Int8GemmKernel_W4
                                                          : resource.quantFunctions.Int8GemmKernel;
        }
        if ((int64_t)resource.blocks * resource.hPack *
                ((int64_t)resource.blockChannels * resource.weightBits / 8 + 2 * sizeof(float)) >
            std::numeric_limits<int>::max()) {
            mValid = false;
            return;
        }
        const auto& branch = core->arm82MatmulRelatedFunctions;
        const int threads = static_cast<CPUBackend*>(backend)->threadNumber();
        const int smeCores = core->smeCoreNumber;
        const int branchMinThreads = smeCores <= 1 ? core->perfCoreNumber : 2 * smeCores;
        const int divisionRatio = static_cast<CPUBackend*>(backend)->getRuntime()->hint().divisionRatio % 64;
        const int prefillRatio = divisionRatio / 8;
        int ratio = divisionRatio % 8;
        if (ratio == 0 || (prefillRatio > 0 && ratio > prefillRatio)) {
            ratio = 1;
        }
        if (resource.ePack == GEMM_INT8_DST_XUNIT_SME2 && divisionRatio > 0 && threads >= 4 && smeCores > 0 &&
            (mDenseGateUp || threads > branchMinThreads) && branch.MNNGetGemmUnit) {
            int branchHPack, branchLPack, branchEPack;
            branch.MNNGetGemmUnit(&branchHPack, &branchLPack, &branchEPack);
            const bool w4 = resource.weightBits == 4;
            auto branchGemm = resource.bytes == 2 ? (w4 ? branch.MNNGemmInt8AddBiasScale_w4_Unit_FP16
                                                        : branch.MNNGemmInt8AddBiasScale_Unit_FP16)
                                                  : (w4 ? branch.Int8GemmKernel_W4 : branch.Int8GemmKernel);
            int mainChannels = ROUND_UP(
                UP_DIV(outputChannels * smeCores * ratio, smeCores * ratio + threads - smeCores), resource.hPack);
            if (mDenseGateUp) {
                auto split = [&](int r, int workers) {
                    const int width = outputChannels / 2;
                    int unit = UP_DIV(UP_DIV(width, resource.pack), r * smeCores + workers - smeCores);
                    return 2 * std::min(width, ROUND_UP(unit * resource.pack * smeCores * r, GEMM_INT8_UNIT_SME2_128));
                };
                resource.denseMainDecode = split(ratio, threads);
                const int prefillThreads = std::max(4, static_cast<CPUBackend*>(backend)->computeThreadNumber(2));
                resource.denseMainPrefill = std::max(resource.denseMainDecode,
                                                    split(prefillRatio ? prefillRatio : 6, prefillThreads));
                resource.denseDecodeGemm = resource.bytes == 2 ?
                    (w4 ? resource.quantFunctions.MNNGemmInt8AddBiasScale_w4_Unit_FP16_DecodeMax :
                          resource.quantFunctions.MNNGemmInt8AddBiasScale_Unit_FP16_DecodeMax) :
                    (w4 ? resource.quantFunctions.MNNGemmInt8AddBiasScale_w4_Unit_FP32_DecodeMax :
                          resource.quantFunctions.MNNGemmInt8AddBiasScale_Unit_FP32_DecodeMax);
                mainChannels = resource.denseMainDecode;
            }
            if (branchGemm && branchLPack == resource.lPack && resource.hPack % branchHPack == 0 &&
                branchHPack % resource.pack == 0 && (mainChannels < outputChannels || mDenseGateUp)) {
                resource.mainChannels = mDenseGateUp ? outputChannels : mainChannels;
                resource.branchHPack = branchHPack;
                resource.branchGemm = branchGemm;
                resource.smeCores = smeCores;
            } else if (mDenseGateUp) {
                resource.denseDecodeGemm = nullptr;
            }
        }
        if (!resource.quantGemm || !resource.quantInput || !core->MNNAsyQuantInfo || !core->MNNAsyQuantFunc ||
            !core->MNNAbsMax || !core->MNNQuantScale || !core->MNNDynamicQuant ||
            !core->MNNDynamicUpdateConvBiasScale) {
            mValid = false;
            return;
        }
#else
        mValid = false;
        return;
#endif
    } else {
        core->MNNGetMatMulPackMode(&resource.ePack, &resource.lPack, &resource.hPack);
    }
    if (!resource.branchGemm) {
        resource.branchHPack = resource.hPack;
    }
    const int mainChannels = resource.mainChannels;
    const int packedMainChannels = mDenseGateUp && resource.branchGemm ? resource.denseMainPrefill : mainChannels;
    auto weightTileBytes = [&](int hPack, int quantInfoBytes) {
        return (int64_t)resource.blocks * hPack *
               ((int64_t)resource.blockChannels * resource.weightBits / 8 + 2 * quantInfoBytes);
    };
    const int64_t alignedInput = ROUND_UP((int64_t)inputChannels, resource.lPack);
    const int outputAlignment = mDenseGateUp && resource.denseDecodeGemm ? GEMM_INT8_UNIT_SME2_128 :
                                                                 std::max(resource.pack, resource.hPack);
    const int64_t alignedOutput64 = ROUND_UP((int64_t)outputChannels, outputAlignment);
    const int mainTiles = UP_DIV(packedMainChannels, resource.hPack);
    const int branchTiles = UP_DIV(outputChannels - mainChannels, resource.branchHPack);
    const int64_t alignedWeightOutput64 = mDenseGateUp ? alignedOutput64 :
        (int64_t)mainTiles * resource.hPack + (int64_t)branchTiles * resource.branchHPack;
    const int64_t weightBytes64 = resource.quantized
                                      ? mainTiles * weightTileBytes(resource.hPack, sizeof(float)) +
                                            branchTiles * weightTileBytes(resource.branchHPack, sizeof(float))
                                      : alignedInput * alignedWeightOutput64 * resource.bytes;
    const int64_t biasBytes64 = alignedOutput64 * (resource.quantized ? (int)sizeof(float) : resource.bytes);
    // Tensor lengths and the packer strides use int. Reject unrepresentable
    // buffers before narrowing, including temporary quantization metadata.
    const int64_t maxSize = std::numeric_limits<int>::max();
    if (weightBytes64 > maxSize || biasBytes64 > maxSize || alignedWeightOutput64 * inputChannels > maxSize ||
        (resource.quantized && ((int64_t)2 * resource.blocks * alignedWeightOutput64 * sizeof(float) > maxSize ||
                                alignedWeightOutput64 * (resource.blocks + 1LL) * sizeof(float) > maxSize))) {
        mValid = false;
        return;
    }
    const int alignedOutput = static_cast<int>(alignedOutput64);
    const int weightBytes = static_cast<int>(weightBytes64);
    resource.alignedWeightOutput = static_cast<int>(alignedWeightOutput64);
    resource.experts.resize(experts);
    for (int index = 0; index < resource.experts.size(); ++index) {
        auto decoded = index == 0 ? firstWeight : IndexedLinearCommon::loadLinear(op, index, backend, forceFloat);
        std::vector<float> bias;
        if (!decoded || !IndexedLinearCommon::loadBias(op, index, bias) ||
            resource.quantized == static_cast<bool>(decoded->weightFloat.get()) ||
            (!resource.quantized && decoded->weightFloat.size() != inputChannels * outputChannels)) {
            mValid = false;
            return;
        }
        auto& expert = resource.experts[index];
        // FP16-output SDOT, I8MM and SME2 kernels accept FP16 scale/source-bias metadata.
        // Keep FP32 sources (including corrected alpha) exact instead of narrowing them on load.
        expert.weightQuantInfoMode = resource.quantized && resource.bytes == 2 && core->supportSDot &&
                                     decoded->alphaIsFp16 && decoded->alphaHalf.get() && !decoded->alpha.get();
        const int quantInfoBytes = expert.weightQuantInfoMode ? sizeof(int16_t) : sizeof(float);
        expert.weightTileBytes =
            resource.quantized ? static_cast<int>(weightTileBytes(resource.hPack, quantInfoBytes)) : 0;
        expert.branchTileBytes =
            resource.quantized ? static_cast<int>(weightTileBytes(resource.branchHPack, quantInfoBytes)) : 0;
        const int expertWeightBytes = resource.quantized
                                          ? mainTiles * expert.weightTileBytes + branchTiles * expert.branchTileBytes
                                          : weightBytes;
        expert.weight.reset(Tensor::createDevice<uint8_t>({expertWeightBytes}));
        expert.bias.reset(Tensor::createDevice<uint8_t>({static_cast<int>(biasBytes64)}));
        mValid = backend->onAcquireBuffer(expert.weight.get(), Backend::STATIC) &&
                 backend->onAcquireBuffer(expert.bias.get(), Backend::STATIC);
        if (!mValid) {
            return;
        }
        bias.resize(alignedOutput, 0.0f);
        if (resource.quantized) {
#ifdef MNN_LOW_MEMORY
            const int bits = decoded->canUseInt4 ? 4 : 8;
            const int scaleCount = decoded->alphaSize / (decoded->asymmetric ? 2 : 1);
            if (!decoded->weight.get() ||
                decoded->weight.size() != (int64_t)inputChannels * outputChannels * bits / 8 ||
                bits != resource.weightBits || scaleCount != resource.blocks * outputChannels || decoded->canUseInt2 ||
                decoded->canUseInt3) {
                mValid = false;
                return;
            }
            const int lPack = resource.lPack;
            const int blockChannels = resource.blockChannels;
            const int blockDepth = blockChannels / lPack;
            const int alignedWeightOutput = static_cast<int>(alignedWeightOutput64);
            std::vector<uint8_t> unpacked(outputChannels * inputChannels);
            if (bits == 4) {
                const auto packed = reinterpret_cast<const uint8_t*>(decoded->weight.get());
                for (int element = 0; element < unpacked.size(); ++element) {
                    unpacked[element] = element % 2 ? packed[element / 2] & 15 : packed[element / 2] >> 4;
                }
            } else {
                ::memcpy(unpacked.data(), decoded->weight.get(), unpacked.size());
            }
            // Kernel sums remain FP32 even on Arm82; an int32 storage tensor
            // prevents the backend from allocating only two bytes per element.
            expert.weightSum.reset(Tensor::createDevice<int32_t>({alignedWeightOutput * (resource.blocks + 1)}));
            if (!backend->onAcquireBuffer(expert.weightSum.get(), Backend::STATIC)) {
                mValid = false;
                return;
            }
            ::memset(expert.weightSum->host<float>(), 0, expert.weightSum->size());
            const float* alpha = decoded->getAlphaFloat();
            if (!alpha) {
                mValid = false;
                return;
            }
            auto packRegion = [&](int firstChannel, int regionChannels, int hPack, bool smeLayout, int8_t* target,
                                  float* sums = nullptr) {
                const int weightTiles = UP_DIV(regionChannels, hPack);
                const int regionAligned = weightTiles * hPack;
                std::vector<uint8_t> reordered(regionAligned * inputChannels);
                int32_t reorderInfo[6] = {resource.blocks, regionChannels, inputChannels, 1, hPack, lPack};
                ConvInt8TiledExecutor::reorderWeight(reordered.data(), unpacked.data() + firstChannel * inputChannels,
                                                     reorderInfo);
                if (bits == 4) {
                    const int cellElements = hPack * lPack;
                    const int cellBytes = cellElements / 2;
                    // Match the existing W4 kernel ABI: SME stores adjacent nibbles;
                    // the other kernels store the two halves of an HP x LP cell.
                    for (int cell = 0; cell < reordered.size() / cellElements; ++cell) {
                        auto source = reordered.data() + cell * cellElements;
                        auto packedCell = reordered.data() + cell * cellBytes;
                        for (int byte = 0; byte < cellBytes; ++byte) {
                            if (smeLayout) {
                                packedCell[byte] = source[byte * 2] | (source[byte * 2 + 1] << 4);
                            } else {
                                packedCell[byte] = (source[byte] << 4) | source[byte + cellBytes];
                            }
                        }
                    }
                }
                std::vector<float> quantInfo(2 * resource.blocks * regionAligned, 0.0f);
                std::vector<int16_t> halfQuantInfo;
                if (expert.weightQuantInfoMode) {
                    halfQuantInfo.resize(quantInfo.size(), 0);
                }
                auto sumBase = sums ? sums : expert.weightSum->host<float>();
                auto regionSum = sumBase + firstChannel * resource.blocks;
                for (int local = 0; local < regionChannels; ++local) {
                    const int channel = firstChannel + local;
                    for (int block = 0; block < resource.blocks; ++block) {
                        const int index = channel * resource.blocks + block;
                        const float scale =
                            alpha[index * (decoded->asymmetric ? 2 : 1) + (decoded->asymmetric ? 1 : 0)];
                        const float offset =
                            (decoded->asymmetric ? alpha[index * 2] : 0.0f) - (bits == 4 ? 8.0f * scale : 0.0f);
                        quantInfo[block * regionAligned + local] = scale;
                        quantInfo[(resource.blocks + block) * regionAligned + local] = offset;
                        if (expert.weightQuantInfoMode) {
                            const int components = decoded->asymmetric ? 2 : 1;
                            halfQuantInfo[block * regionAligned + local] =
                                decoded->alphaHalf.get()[index * components + components - 1];
                            // The kernel reconstructs the W4 offset in FP32. Rounding the folded offset
                            // to FP16 here would change the represented weights.
                            halfQuantInfo[(resource.blocks + block) * regionAligned + local] =
                                decoded->asymmetric ? decoded->alphaHalf.get()[index * 2] : 0;
                        }
                        int sum = 0;
                        for (int inner = 0; inner < blockChannels; ++inner) {
                            const auto value = unpacked[channel * inputChannels + block * blockChannels + inner];
                            sum += bits == 4 ? static_cast<int>(value) : static_cast<int>(static_cast<int8_t>(value));
                        }
                        const float blockSum = sum * scale + blockChannels * offset;
                        regionSum[(local / hPack) * resource.blocks * hPack + block * hPack + local % hPack] = blockSum;
                        sumBase[alignedWeightOutput * resource.blocks + channel] += blockSum;
                    }
                }
                int32_t packInfo[6] = {resource.blocks, weightTiles,      blockDepth,
                                       hPack,           lPack * bits / 8, regionAligned};
                ConvInt8TiledExecutor::packWeightAndQuantInfo(
                    target, reinterpret_cast<const int8_t*>(reordered.data()),
                    expert.weightQuantInfoMode ? reinterpret_cast<const int8_t*>(halfQuantInfo.data())
                                               : reinterpret_cast<const int8_t*>(quantInfo.data()),
                    packInfo, quantInfoBytes);
            };
            packRegion(0, packedMainChannels, resource.hPack,
                       resource.ePack == GEMM_INT8_DST_XUNIT_SME2 && resource.hPack == GEMM_INT8_UNIT_SME2,
                       expert.weight->host<int8_t>());
            if (mainChannels < outputChannels) {
                packRegion(mainChannels, outputChannels - mainChannels, resource.branchHPack, false,
                           expert.weight->host<int8_t>() + mainTiles * expert.weightTileBytes);
            }
            if (mDenseGateUp && resource.branchGemm && resource.denseMainDecode < outputChannels) {
                const int64_t branchBytes = (int64_t)UP_DIV(outputChannels - resource.denseMainDecode,
                                                           resource.branchHPack) * expert.branchTileBytes;
                if (branchBytes > maxSize) {
                    mValid = false;
                    return;
                }
                expert.denseBranchWeight.reset(Tensor::createDevice<uint8_t>({static_cast<int>(branchBytes)}));
                expert.denseBranchSum.reset(Tensor::createDevice<int32_t>({alignedWeightOutput * (resource.blocks + 1)}));
                if (!backend->onAcquireBuffer(expert.denseBranchWeight.get(), Backend::STATIC) ||
                    !backend->onAcquireBuffer(expert.denseBranchSum.get(), Backend::STATIC)) {
                    mValid = false;
                    return;
                }
                ::memset(expert.denseBranchSum->host<float>(), 0, expert.denseBranchSum->size());
                packRegion(resource.denseMainDecode, outputChannels - resource.denseMainDecode, resource.branchHPack, false,
                           expert.denseBranchWeight->host<int8_t>(), expert.denseBranchSum->host<float>());
            }
            if (mDenseGateUp && resource.denseDecodeGemm) {
                const auto decodeTileBytes = weightTileBytes(GEMM_INT8_UNIT_SME2_128, quantInfoBytes);
                if (decodeTileBytes > maxSize) {
                    mValid = false;
                    return;
                }
                expert.denseDecodeTileBytes = static_cast<int>(decodeTileBytes);
                const int64_t decodeBytes = (int64_t)UP_DIV(resource.denseMainDecode, GEMM_INT8_UNIT_SME2_128) *
                                           expert.denseDecodeTileBytes;
                if (decodeBytes > maxSize) {
                    mValid = false;
                    return;
                }
                expert.denseDecodeWeight.reset(Tensor::createDevice<uint8_t>({static_cast<int>(decodeBytes)}));
                if (!backend->onAcquireBuffer(expert.denseDecodeWeight.get(), Backend::STATIC)) {
                    mValid = false;
                    return;
                }
                expert.denseDecodeSum.reset(Tensor::createDevice<int32_t>({alignedWeightOutput * (resource.blocks + 1)}));
                if (!backend->onAcquireBuffer(expert.denseDecodeSum.get(), Backend::STATIC)) {
                    mValid = false;
                    return;
                }
                ::memset(expert.denseDecodeSum->host<float>(), 0, expert.denseDecodeSum->size());
                packRegion(0, resource.denseMainDecode, GEMM_INT8_UNIT_SME2_128, true,
                           expert.denseDecodeWeight->host<int8_t>(), expert.denseDecodeSum->host<float>());
            }
            ::memcpy(expert.bias->host<float>(), bias.data(), bias.size() * sizeof(float));
#endif
        } else if (resource.bytes == 2) {
            std::vector<int16_t> lowpWeight(inputChannels * outputChannels);
            core->MNNFp32ToLowp(decoded->weightFloat.get(), lowpWeight.data(), lowpWeight.size());
            core->MNNPackForMatMul_B(expert.weight->host<float>(), reinterpret_cast<const float*>(lowpWeight.data()),
                                     outputChannels, 1, inputChannels, true);
            core->MNNFp32ToLowp(bias.data(), expert.bias->host<int16_t>(), bias.size());
        } else {
            core->MNNPackForMatMul_B(expert.weight->host<float>(), decoded->weightFloat.get(), outputChannels, 1,
                                     inputChannels, true);
            ::memcpy(expert.bias->host<float>(), bias.data(), bias.size() * sizeof(float));
        }
    }
}

CPUIndexedLinear::CPUIndexedLinear(Backend* backend, std::shared_ptr<Resource> resource, int inputMode, bool fuseSwiGLU,
                                   int routeWeight, float routeEps, float routeScale)
    : Execution(backend), mResource(std::move(resource)), mInputMode(inputMode), mFuseSwiGLU(fuseSwiGLU),
      mRouteWeight(routeWeight), mRouteEps(routeEps), mRouteScale(routeScale) {}

bool CPUIndexedLinear::onClone(Backend* backend, const Op* op, Execution** dst) {
    if (!dst && !mDenseGateUp) {
        return mValid;
    }
    if (!backend || (backend->type() != MNN_FORWARD_CPU && backend->type() != MNN_FORWARD_CPU_EXTENSION)) {
        return false;
    }
    int experts, inputChannels, outputChannels;
    const bool dense = op->main_type() == OpParameter_FusedLinearParam;
    if (!IndexedLinearCommon::dimensions(op, experts, inputChannels, outputChannels) ||
        experts != mResource->experts.size() || inputChannels != mResource->inputChannels ||
        outputChannels != mResource->outputChannels || dense != mDenseGateUp ||
        (!dense && (op->main_as_IndexedLinearParam()->inputMode() != mInputMode ||
        (op->main_as_IndexedLinearParam()->fusedActivation() == UnaryOpOperation_SILU) != mFuseSwiGLU))) {
        return false;
    }
    if (!dense && (op->main_as_IndexedLinearParam()->routeWeight() != mRouteWeight ||
        op->main_as_IndexedLinearParam()->routeEps() != mRouteEps ||
        op->main_as_IndexedLinearParam()->routeScale() != mRouteScale)) {
        return false;
    }
    auto core = indexedLinearCore(backend);
    if (!mValid || core != mResource->core || core->bytes != mResource->bytes || core->pack != mResource->pack) {
        return false;
    }
    if (mDenseGateUp && (static_cast<CPUBackend*>(backend)->threadNumber() != mResource->denseThreads ||
        static_cast<CPUBackend*>(backend)->memoryMode() != mResource->denseMemory ||
        backend->getRuntime()->hint().divisionRatio != mResource->denseDivisionRatio ||
        backend->getRuntime()->hint().dynamicQuantOption != mResource->denseDynamicOption ||
        (backend->getRuntime()->hint().useCachedMmap > 0 && !backend->getRuntime()->hint().weightMemoryPath.empty()))) {
        return false;
    }
    int ePack, lPack, hPack;
    if (mResource->quantized) {
        static_cast<CPUBackend*>(backend)->int8GemmFunctions()->MNNGetGemmUnit(&hPack, &lPack, &ePack);
    } else {
        core->MNNGetMatMulPackMode(&ePack, &lPack, &hPack);
    }
    if (ePack != mResource->ePack || lPack != mResource->lPack || hPack != mResource->hPack) {
        return false;
    }
    if (dst) {
        auto clone = new CPUIndexedLinear(backend, mResource, mInputMode, mFuseSwiGLU, mRouteWeight, mRouteEps, mRouteScale);
        clone->mDenseGateUp = mDenseGateUp;
        *dst = clone;
    }
    return true;
}

ErrorCode CPUIndexedLinear::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    if ((int)inputs.size() != (mDenseGateUp ? 1 : (mRouteWeight ? 3 : 2)) || outputs.size() != 1) {
        return INPUT_DATA_ERROR;
    }
    const auto& resource = *mResource;
    const auto x = inputs[0];
    const auto y = outputs[0];
    if (mDenseGateUp) {
        for (auto tensor : {x, y}) {
            if (tensor->dimensions() != 4 || tensor->length(2) != 1 || tensor->length(3) != 1 ||
                tensor->getType() != halide_type_of<float>() || tensor->length(0) <= 0 ||
                TensorUtils::getDescribe(tensor)->dimensionFormat != MNN_DATA_FORMAT_NC4HW4 ||
                CPUBackend::getBytes(backend(), tensor) != resource.bytes) {
                return INPUT_DATA_ERROR;
            }
        }
        if (x->length(0) != y->length(0) || x->length(1) != resource.inputChannels ||
            y->length(1) * 2 != resource.outputChannels || y->length(1) % resource.pack) {
            return INPUT_DATA_ERROR;
        }
        mTokens = x->length(0);
        mTopK = 1;
    } else {
        const auto ids = inputs[1];
        const Tensor* w = mRouteWeight ? inputs[2] : nullptr;
        auto contiguous = [](const Tensor* tensor) {
            if (TensorUtils::getDescribe(tensor)->dimensionFormat != MNN_DATA_FORMAT_NCHW) {
                return false;
            }
            int64_t stride = 1;
            for (int dim = tensor->dimensions() - 1; dim >= 0; --dim) {
                if (tensor->length(dim) < 0 || tensor->stride(dim) != stride) {
                    return false;
                }
                stride *= tensor->length(dim);
                if (stride > std::numeric_limits<int>::max()) {
                    return false;
                }
            }
            return true;
        };
        auto xDes = TensorUtils::getDescribe(x);
        bool xLayout = false;
        int xChannels = 0;
        if (xDes->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
            xLayout = mInputMode == 0 && x->dimensions() == 4 && x->length(2) == 1 && x->length(3) == 1 &&
                      x->length(1) % 4 == 0;
            const int inputPack = resource.quantized ? resource.lPack : resource.pack;
            if (xLayout && (inputPack > resource.pack || resource.pack % inputPack != 0)) {
                xLayout = false;
            }
            xChannels = xLayout ? x->length(1) : 0;
        } else if (x->dimensions() == (mInputMode == 0 ? 2 : 3) && contiguous(x)) {
            xLayout = true;
            xChannels = x->length(x->dimensions() - 1);
        }
        bool yOk;
        if (mRouteWeight) {
            auto yDes = TensorUtils::getDescribe(y);
            yOk = mInputMode == 1 && !mFuseSwiGLU && yDes->dimensionFormat == MNN_DATA_FORMAT_NC4HW4 &&
                  y->dimensions() == 4 && y->length(2) == 1 && y->length(3) == 1 &&
                  y->length(1) == resource.outputChannels && y->length(1) % resource.pack == 0;
        } else {
            yOk = contiguous(y) && y->dimensions() == 3 && y->length(1) == ids->length(1) &&
                  y->length(2) == resource.outputChannels / (mFuseSwiGLU ? 2 : 1);
        }
        if (x->getType() != halide_type_of<float>() || y->getType() != halide_type_of<float>() ||
            ids->getType() != halide_type_of<int32_t>() || !xLayout || !contiguous(ids) || !yOk ||
            (mRouteWeight == 1 && (w->getType() != halide_type_of<float>() || !contiguous(w) || w->dimensions() != 2 ||
                                  w->length(0) != ids->length(0) || w->length(1) != ids->length(1))) ||
            (mRouteWeight >= 2 && (w->getType() != halide_type_of<float>() || !contiguous(w) || w->dimensions() < 2 ||
                                  w->length(w->dimensions() - 1) != (int)resource.experts.size() ||
                                  w->elementSize() != (int64_t)ids->length(0) * (int64_t)resource.experts.size())) ||
            CPUBackend::getBytes(backend(), x) != resource.bytes || CPUBackend::getBytes(backend(), y) != resource.bytes ||
            ids->dimensions() != 2 || x->length(0) != ids->length(0) || xChannels != resource.inputChannels ||
            ids->length(1) <= 0 || (mInputMode == 1 && x->length(1) != ids->length(1)) || y->length(0) != ids->length(0)) {
            return INPUT_DATA_ERROR;
        }
        mTokens = ids->length(0);
        mTopK = ids->length(1);
    }
    const int routes = mTokens * mTopK;
    const int64_t maxTiles64 = mTokens == 1
                                   ? routes
                                   : std::min<int64_t>(routes, UP_DIV((int64_t)routes, resource.ePack) +
                                                                   std::min<int64_t>(routes, resource.experts.size()));
    mThreads = static_cast<CPUBackend*>(backend())->computeThreadNumber(mTokens);
    const int channelTile = mDenseGateUp ? 4096 / resource.bytes : 512;
    mChannelTile = ROUND_UP(channelTile, std::max(resource.pack, resource.hPack));
    mRouteParallel = mTokens == 1 && mThreads > 0 && !resource.branchGemm && mTopK % mThreads == 0;
    if (mRouteParallel) {
        const int unit = std::max(mFuseSwiGLU ? std::max(16, resource.pack) : resource.pack, resource.hPack);
        mChannelTile = ROUND_UP(std::min(resource.outputChannels, 2048), unit);
    }
    const int64_t inputTileBytes = (int64_t)resource.ePack * ROUND_UP((int64_t)resource.inputChannels, resource.lPack) *
                                   (resource.quantized ? 1 : resource.bytes);
    int64_t scratchBytes = (int64_t)resource.ePack *
                           ROUND_UP((int64_t)resource.inputChannels, std::max(resource.pack, resource.lPack)) *
                           resource.bytes;
    const int64_t quantInfoBytes = (int64_t)3 * resource.blocks * resource.ePack * sizeof(float);
    if (resource.quantized) {
        // qscale/qbias and min/max scratch; min/max may use the backend's lowp type.
        scratchBytes += (int64_t)4 * resource.blocks * resource.ePack * sizeof(float);
        // Input preparation and GEMM run in separate parallel regions, so the
        // same per-thread scratch can hold single-row zero-point-fused bias.
        scratchBytes = std::max(scratchBytes, (int64_t)mChannelTile * (int)sizeof(float));
    }
    if (mFuseSwiGLU) {
        scratchBytes = std::max(scratchBytes, (int64_t)2 * resource.ePack * mChannelTile * (int)sizeof(float));
    }
    const int64_t outputTileBytes = (int64_t)resource.ePack * mChannelTile * resource.bytes;
    const int accumulatorH = mDenseGateUp && resource.denseDecodeGemm ? GEMM_INT8_UNIT_SME2_128 : resource.hPack;
    const int64_t accumulatorBytes = (int64_t)mThreads * resource.ePack * accumulatorH * sizeof(float);
    const int foldThreads = mTokens >= 128 ? std::min(mThreads, UP_DIV(mTokens, 32)) : 1;
    const int64_t routeScratchBytes = foldThreads * (2LL * resource.outputChannels + mTopK) * sizeof(float);
    const int64_t maxSize = std::numeric_limits<int>::max();
    if (mThreads <= 0 || maxTiles64 > maxSize || inputTileBytes > maxSize || scratchBytes > maxSize ||
        quantInfoBytes > maxSize || outputTileBytes > maxSize || maxTiles64 * inputTileBytes > maxSize ||
        mThreads * scratchBytes > maxSize || mThreads * outputTileBytes > maxSize ||
        (resource.quantized && (maxTiles64 * quantInfoBytes > maxSize || accumulatorBytes > maxSize)) ||
        (mRouteWeight && routeScratchBytes > maxSize) ||
        (int64_t)routes * resource.outputChannels * resource.bytes > maxSize ||
        (int64_t)(mInputMode == 0 ? mTokens : routes) * resource.inputChannels * resource.bytes > maxSize) {
        return OUT_OF_MEMORY;
    }
    const int maxTiles = static_cast<int>(maxTiles64);
    mInputTileBytes = static_cast<int>(inputTileBytes);
    mScratchBytes = static_cast<int>(scratchBytes);
    mQuantInfoBytes = static_cast<int>(quantInfoBytes);
    mOutputTileBytes = static_cast<int>(outputTileBytes);
    auto allocate = [&](Tensor& tensor, int bytes) {
        tensor.buffer().type = halide_type_of<uint8_t>();
        tensor.buffer().dimensions = 1;
        tensor.setLength(0, std::max(1, bytes));
        TensorUtils::setLinearLayout(&tensor);
        return backend()->onAcquireBuffer(&tensor, Backend::DYNAMIC);
    };
    if (!allocate(mPackedInput, maxTiles * mInputTileBytes) || !allocate(mScratch, mThreads * mScratchBytes) ||
        !allocate(mPackedOutput, mThreads * mOutputTileBytes) ||
        (resource.quantized && (!allocate(mQuantInfo, maxTiles * mQuantInfoBytes) ||
                                !allocate(mAccumulator, static_cast<int>(accumulatorBytes))))) {
        return OUT_OF_MEMORY;
    }
    if (mRouteWeight && (!allocate(mRouteTmp, routes * resource.outputChannels * resource.bytes) ||
                         !allocate(mRouteScratch, static_cast<int>(routeScratchBytes)))) {
        return OUT_OF_MEMORY;
    }
    backend()->onReleaseBuffer(&mPackedInput, Backend::DYNAMIC);
    backend()->onReleaseBuffer(&mScratch, Backend::DYNAMIC);
    backend()->onReleaseBuffer(&mPackedOutput, Backend::DYNAMIC);
    if (mRouteWeight) {
        backend()->onReleaseBuffer(&mRouteTmp, Backend::DYNAMIC);
        backend()->onReleaseBuffer(&mRouteScratch, Backend::DYNAMIC);
    }
    if (resource.quantized) {
        backend()->onReleaseBuffer(&mQuantInfo, Backend::DYNAMIC);
        backend()->onReleaseBuffer(&mAccumulator, Backend::DYNAMIC);
    }
    mOffsets.resize(resource.experts.size() + 1);
    mRouteOrder.resize(routes);
    mTiles.reserve(maxTiles);
    return NO_ERROR;
}

ErrorCode CPUIndexedLinear::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    const auto& resource = *mResource;
    const auto core = resource.core;
    const int routes = mTokens * mTopK;
    const int channels = resource.inputChannels;
    const int outputChannels = resource.outputChannels;
    const int pack = resource.pack;
    const int bytes = resource.bytes;
    const auto ids = mDenseGateUp ? nullptr : inputs[1]->host<int32_t>();
    mTiles.clear();
    // Decode uses the K route entries directly, without touching all E experts.
    if (mDenseGateUp) {
        mOffsets[0] = 0;
        mOffsets[1] = mTokens;
        for (int t = 0; t < mTokens; ++t) {
            mRouteOrder[t] = t;
        }
        for (int t = 0; t < mTokens; t += resource.ePack) {
            mTiles.push_back({0, t, std::min(resource.ePack, mTokens - t)});
        }
    } else if (mTokens == 1) {
        for (int route = 0; route < routes; ++route) {
            if (ids[route] < 0 || ids[route] >= resource.experts.size()) {
                return INPUT_DATA_ERROR;
            }
            mRouteOrder[route] = route;
            mTiles.push_back({ids[route], route, 1});
        }
    } else {
        std::fill(mOffsets.begin(), mOffsets.end(), 0);
        for (int route = 0; route < routes; ++route) {
            if (ids[route] < 0 || ids[route] >= resource.experts.size()) {
                return INPUT_DATA_ERROR;
            }
            ++mOffsets[ids[route] + 1];
        }
        for (int expert = 1; expert < mOffsets.size(); ++expert) {
            mOffsets[expert] += mOffsets[expert - 1];
        }
        auto cursor = mOffsets;
        for (int route = 0; route < routes; ++route) {
            mRouteOrder[cursor[ids[route]]++] = route;
        }
        for (int expert = 0; expert < resource.experts.size(); ++expert) {
            for (int begin = mOffsets[expert]; begin < mOffsets[expert + 1]; begin += resource.ePack) {
                mTiles.push_back({expert, begin, std::min(resource.ePack, mOffsets[expert + 1] - begin)});
            }
        }
    }
    const auto input = inputs[0]->host<uint8_t>();
    auto output = mRouteWeight ? mRouteTmp.host<uint8_t>() : outputs[0]->host<uint8_t>();
    const bool xC4 = TensorUtils::getDescribe(inputs[0])->dimensionFormat == MNN_DATA_FORMAT_NC4HW4;
    const bool sharedDecodeInput = mTokens == 1 && mInputMode == 0;
    const int inputTiles = sharedDecodeInput ? std::min<size_t>(1, mTiles.size()) : mTiles.size();
    // Like DenseConvInt8TiledExecutor's 1x1 fastway, options 0 and 1 use
    // per-row symmetric prefill and asymmetric single-row decode. Option 1's
    // tensor-wise override applies only to general convolutions, not linears.
    const int dynamicQuantOption = static_cast<CPUBackend*>(backend())->getRuntime()->hint().dynamicQuantOption % 8;
    auto prepareTile = [&](int tileIndex, uint8_t* scratch) {
        const auto& tile = mTiles[tileIndex];
        const int inputPack = resource.quantized ? resource.lPack : pack;
        const int inputBytes = resource.ePack * ROUND_UP(channels, std::max(pack, inputPack)) * bytes;
        ::memset(scratch, 0, mDenseGateUp ? inputBytes : mScratchBytes);
        for (int row = 0; row < tile.rows; ++row) {
            int route = mRouteOrder[tile.firstRoute + row];
            int sourceRow = mInputMode == 0 ? route / mTopK : route;
            if (tile.rows == 1 && (!xC4 || mTokens == 1)) {
                ::memcpy(scratch, input + (int64_t)sourceRow * channels * bytes, channels * bytes);
            } else if (xC4) {
                const int blockStride = mTokens * pack;
                for (int channel = 0; channel < channels; channel += inputPack) {
                    auto dst = scratch + ((channel / inputPack) * tile.rows * inputPack + row * inputPack) * bytes;
                    for (int inner = 0; inner < std::min(inputPack, channels - channel);) {
                        const int c = channel + inner;
                        const int count = std::min(pack - c % pack, std::min(inputPack - inner, channels - c));
                        auto src = input + ((c / pack) * blockStride + sourceRow * pack + c % pack) * bytes;
                        ::memcpy(dst + inner * bytes, src, count * bytes);
                        inner += count;
                    }
                }
            } else {
                for (int channel = 0; channel < channels; channel += inputPack) {
                    auto dst = scratch + ((channel / inputPack) * tile.rows * inputPack + row * inputPack) * bytes;
                    auto src = input + (sourceRow * channels + channel) * bytes;
                    ::memcpy(dst, src, std::min(inputPack, channels - channel) * bytes);
                }
            }
        }
        if (resource.quantized) {
            auto packed = reinterpret_cast<int8_t*>(mPackedInput.host<uint8_t>() + tileIndex * mInputTileBytes);
            auto scale = reinterpret_cast<float*>(mQuantInfo.host<uint8_t>() + tileIndex * mQuantInfoBytes);
            auto bias = scale + resource.blocks * resource.ePack;
            auto sum = bias + resource.blocks * resource.ePack;
            auto auxiliary = reinterpret_cast<float*>(
                scratch + resource.ePack * ROUND_UP(channels, std::max(pack, inputPack)) * bytes);
            auto qscale = auxiliary;
            auto qbias = qscale + resource.blocks * resource.ePack;
            auto minimum = qbias + resource.blocks * resource.ePack;
            auto maximum = minimum + resource.blocks * resource.ePack;
            const int expertRows = mTokens == 1 ? 1 : mOffsets[tile.expert + 1] - mOffsets[tile.expert];
            const bool asymmetric = dynamicQuantOption == 2 || expertRows == 1;
            const int quantBlocks = dynamicQuantOption == 2 ? resource.blocks : 1;
            if (asymmetric) {
                size_t info[9] = {static_cast<size_t>(quantBlocks),
                                  static_cast<size_t>(tile.rows),
                                  static_cast<size_t>(inputPack),
                                  static_cast<size_t>(resource.ePack),
                                  static_cast<size_t>(inputPack),
                                  1,
                                  static_cast<size_t>(channels / quantBlocks / inputPack),
                                  static_cast<size_t>(expertRows == 1 && dynamicQuantOption != 2),
                                  0};
                core->MNNAsyQuantInfo(scale, bias, qscale, qbias, minimum, maximum,
                                      reinterpret_cast<const float*>(scratch), info);
                if (expertRows == 1 && dynamicQuantOption != 2) {
                    // Match ordinary Linear's decode quantization and its
                    // zero-point arithmetic, including FP16 rounding.
                    resource.quantInput(reinterpret_cast<const float*>(scratch), packed, UP_DIV(channels, pack), qscale,
                                        -128, 127, qbias, 0);
                    bias[0] = -qbias[0] * scale[0];
                } else {
                    core->MNNAsyQuantFunc(packed, reinterpret_cast<const float*>(scratch), qscale, qbias, info);
                }
            } else {
                core->MNNAbsMax(reinterpret_cast<const float*>(scratch), minimum, channels / inputPack, tile.rows,
                                inputPack);
                core->MNNQuantScale(minimum, qscale, scale, 1, tile.rows);
                core->MNNDynamicQuant(reinterpret_cast<const float*>(scratch), packed, qscale, channels / inputPack,
                                      tile.rows, inputPack, nullptr);
                for (int row = 0; row < tile.rows; ++row) {
#ifdef MNN_USE_SSE
                    bias[row] = -128.0f * scale[row];
#else
                    bias[row] = 0.0f;
#endif
                }
            }
            if (quantBlocks == 1) {
                for (int block = 1; block < resource.blocks; ++block) {
                    ::memcpy(scale + block * tile.rows, scale, tile.rows * sizeof(float));
                    ::memcpy(bias + block * tile.rows, bias, tile.rows * sizeof(float));
                }
            }
            if (mDenseGateUp && resource.quantFunctions.MNNSumByAxisLForMatmul_A) {
                SumByAxisParams sumParams{};
                sumParams.kernelCountUnitDouble = channels / inputPack;
                sumParams.unitColBufferSize = resource.ePack * channels;
                sumParams.DST_XUNIT = resource.ePack;
                sumParams.SRC_UNIT = inputPack;
                sumParams.blockNum = resource.blocks;
                sumParams.kernelxy = 1;
                sumParams.LU = channels / inputPack;
                sumParams.inputBlock = 1;
                resource.quantFunctions.MNNSumByAxisLForMatmul_A(sum, packed, scale, tile.rows, sumParams);
                return;
            }
            for (int block = 0; block < resource.blocks; ++block) {
                for (int row = 0; row < tile.rows; ++row) {
                    int integerSum = 0;
                    const auto source = packed + block * resource.blockChannels * tile.rows + row * inputPack;
                    for (int inner = 0; inner < resource.blockChannels / inputPack; ++inner) {
                        for (int lane = 0; lane < inputPack; ++lane) {
#ifdef MNN_USE_SSE
                            integerSum +=
                                reinterpret_cast<const uint8_t*>(source)[inner * tile.rows * inputPack + lane];
#else
                            integerSum += source[inner * tile.rows * inputPack + lane];
#endif
                        }
                    }
                    sum[block * tile.rows + row] = integerSum * scale[block * tile.rows + row];
                }
            }
            return;
        }
        const float* source = reinterpret_cast<const float*>(scratch);
        int32_t info[4] = {1, tile.rows, tile.rows, 1};
        // GEMM reads the aligned L extent, including the zero-filled scratch tail.
        int32_t extent[4] = {tile.rows, ROUND_UP(channels, resource.lPack), 0, 0};
        core->MNNPackC4ForMatMul_A(reinterpret_cast<float*>(mPackedInput.host<uint8_t>() + tileIndex * mInputTileBytes),
                                   &source, info, extent);
    };
    const int prepThreads = std::min(mThreads, inputTiles);
    const bool denseTileParallel = mDenseGateUp && mTiles.size() >= mThreads && mTokens > 1;
    if ((!mRouteParallel || sharedDecodeInput) && !denseTileParallel) {
        if (prepThreads <= 1) {
            for (int tileIndex = 0; tileIndex < inputTiles; ++tileIndex) {
                prepareTile(tileIndex, mScratch.host<uint8_t>());
            }
        } else {
            MNN_CONCURRENCY_BEGIN(thread, prepThreads) {
                for (int tileIndex = thread; tileIndex < inputTiles; tileIndex += prepThreads) {
                    prepareTile(tileIndex, mScratch.host<uint8_t>() + thread * mScratchBytes);
                }
            }
            MNN_CONCURRENCY_END();
        }
    }
    const int tiles = mTiles.size();
    const int mainChannels = mDenseGateUp && resource.branchGemm ?
        (mTokens == 1 ? resource.denseMainDecode : resource.denseMainPrefill) : resource.mainChannels;
    const int branchChannels = outputChannels - mainChannels;
    const int mainWeightTiles = UP_DIV(mainChannels, resource.hPack);
    const int mainWorkers = resource.branchGemm ? std::min(mThreads, resource.smeCores) : mThreads;
    auto channelTile = [&](int channels, int workers, int unit) {
        const int splits = UP_DIV(4 * workers, std::max(tiles, 1));
        return std::min(mChannelTile, ROUND_UP(UP_DIV(channels, splits), unit));
    };
    const bool denseDecode = mDenseGateUp && mTokens == 1 && resource.denseDecodeGemm;
    const int mainH = denseDecode ? GEMM_INT8_UNIT_SME2_128 : resource.hPack;
    const int mainUnit = mFuseSwiGLU ? std::max(mDenseGateUp ? 2 * pack : 16, std::max(16, mainH))
                                   : std::max(pack, resource.hPack);
    const int mainTile = mainChannels > 0 ? channelTile(mainChannels, mainWorkers, mainUnit) : 1;
    const int branchTile = branchChannels > 0 ? channelTile(branchChannels, mThreads,
        mDenseGateUp ? std::max(16, 2 * pack) : resource.branchHPack) : 1;
    const int mainSplits = UP_DIV(mainChannels, mainTile);
    const int branchSplits = UP_DIV(branchChannels, branchTile);
    const int mainTasks = tiles * mainSplits;
    const int branchTasks = tiles * branchSplits;
    std::atomic<int> nextMainTask{0};
    std::atomic<int> nextBranchTask{0};
    auto mainCursor = &nextMainTask;
    auto branchCursor = &nextBranchTask;
    const float postClamp[4] = {1.0f, 1.0f, -std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
    const float* post = postClamp;
    auto runTask = [&](int thread, int tileIndex, int channel, int channelCount, bool branch) {
        auto temporaryOutput = mPackedOutput.host<uint8_t>() + thread * mOutputTileBytes;
        const auto& tile = mTiles[tileIndex];
        const auto& expert = resource.experts[tile.expert];
        const int packedTileIndex = sharedDecodeInput ? 0 : tileIndex;
        if (resource.quantized) {
            auto info = reinterpret_cast<const float*>(mQuantInfo.host<uint8_t>() + packedTileIndex * mQuantInfoBytes);
            const int accumulatorH = mDenseGateUp && resource.denseDecodeGemm ? GEMM_INT8_UNIT_SME2_128 : resource.hPack;
            auto accumulator = mAccumulator.host<float>() + thread * resource.ePack * accumulatorH;
            QuanPostTreatParameters post{};
            post.useInt8 = 0;
            post.blockNum = resource.blocks;
            post.weightQuantInfoMode = expert.weightQuantInfoMode;
            post.inputScale = info;
            post.inputBias = info + resource.blocks * resource.ePack;
            post.srcKernelSum = const_cast<float*>(info + 2 * resource.blocks * resource.ePack);
            auto sums = mDenseGateUp && branch ? expert.denseBranchSum.get() :
                        (denseDecode ? expert.denseDecodeSum.get() : expert.weightSum.get());
            post.weightKernelSum = sums->host<float>() + channel * resource.blocks;
            post.biasFloat = expert.bias->host<float>() + channel;
            const int expertRows = mTokens == 1 ? 1 : mOffsets[tile.expert + 1] - mOffsets[tile.expert];
            if (expertRows == 1 && dynamicQuantOption != 2) {
                // Ordinary Linear combines the one input zero point with
                // bias once, after summing weight blocks. Preserve that
                // order so tiny differences do not change later MoE routes.
                auto fusedBias = reinterpret_cast<float*>(mScratch.host<uint8_t>() + thread * mScratchBytes);
                auto weightSum =
                    sums->host<float>() + resource.alignedWeightOutput * resource.blocks + channel;
                auto inputZero = info[resource.blocks * resource.ePack];
                core->MNNDynamicUpdateConvBiasScale(fusedBias, expert.bias->host<float>() + channel, weightSum,
                                                    &inputZero, UP_DIV(channelCount, 4));
                post.inputBias = nullptr;
                post.weightKernelSum = weightSum;
                post.biasFloat = fusedBias;
            }
            post.accumBuffer = accumulator;
            post.fp32minmax = const_cast<float*>(resource.clamp);
            int32_t indices[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
            post.indices = indices;
            const int8_t* weight;
            if (mDenseGateUp && branch) {
                weight = expert.denseBranchWeight->host<int8_t>() +
                    (channel - resource.denseMainDecode) / resource.branchHPack * expert.branchTileBytes;
            } else if (denseDecode) {
                weight = expert.denseDecodeWeight->host<int8_t>() +
                         channel / GEMM_INT8_UNIT_SME2_128 * expert.denseDecodeTileBytes;
            } else {
                weight = expert.weight->host<int8_t>() +
                         (branch ? mainWeightTiles * expert.weightTileBytes +
                                       (channel - mainChannels) / resource.branchHPack * expert.branchTileBytes
                                 : channel / resource.hPack * expert.weightTileBytes);
            }
            (branch ? resource.branchGemm : (denseDecode ? resource.denseDecodeGemm : resource.quantGemm))(
                reinterpret_cast<int8_t*>(temporaryOutput),
                reinterpret_cast<const int8_t*>(mPackedInput.host<uint8_t>() + packedTileIndex * mInputTileBytes),
                weight, resource.blockChannels / resource.lPack, tile.rows * pack * bytes, UP_DIV(channelCount, pack),
                &post, tile.rows);
        } else {
            size_t parameter[7] = {static_cast<size_t>(tile.rows * resource.lPack * bytes),
                                   static_cast<size_t>(ROUND_UP(channels, resource.lPack)),
                                   static_cast<size_t>(channelCount),
                                   static_cast<size_t>(tile.rows * pack * bytes),
                                   0,
                                   0,
                                   0};
            auto weight = expert.weight->host<uint8_t>() + channel * ROUND_UP(channels, resource.lPack) * bytes;
            auto bias = expert.bias->host<uint8_t>() + channel * bytes;
            core->MNNPackedMatMulRemain(
                reinterpret_cast<float*>(temporaryOutput),
                reinterpret_cast<const float*>(mPackedInput.host<uint8_t>() + packedTileIndex * mInputTileBytes),
                reinterpret_cast<const float*>(weight), tile.rows, parameter, post,
                reinterpret_cast<const float*>(bias), nullptr, nullptr);
        }
        if (mFuseSwiGLU) {
            const int halfPack = pack / 2;
            const int halfChannels = channelCount / 2;
            const int halfTotal = tile.rows * halfChannels;
            const int packs = UP_DIV(channelCount, pack);
            auto scratch = mScratch.host<uint8_t>() + thread * mScratchBytes;
            auto gatesBytes = scratch;
            auto upsBytes = scratch + halfTotal * bytes;
            auto siluBytes = scratch + 2 * halfTotal * bytes;
            if (mDenseGateUp) {
                if (bytes == 2) {
                    unpackGateUpTile<uint16_t>(temporaryOutput, gatesBytes, upsBytes, halfChannels, tile.rows, pack);
                } else {
                    unpackGateUpTile<uint32_t>(temporaryOutput, gatesBytes, upsBytes, halfChannels, tile.rows, pack);
                }
                resource.siluFunction(siluBytes, gatesBytes, halfTotal);
                resource.mulFunction(upsBytes, siluBytes, upsBytes, halfTotal, -1);
                for (int c = 0; c < halfChannels; c += pack) {
                    auto dst = output + ((int64_t)(channel / 2 + c) / pack * mTokens + tile.firstRoute) * pack * bytes;
                    ::memcpy(dst, upsBytes + c * tile.rows * bytes, tile.rows * pack * bytes);
                }
                return;
            }
            if (bytes == 2) {
                auto srcAll = reinterpret_cast<const uint16_t*>(temporaryOutput);
                auto gateAll = reinterpret_cast<uint16_t*>(gatesBytes);
                auto upAll = reinterpret_cast<uint16_t*>(upsBytes);
                for (int packIndex = 0; packIndex < packs; ++packIndex) {
                    const int inner = std::min(pack, channelCount - packIndex * pack);
                    for (int row = 0; row < tile.rows; ++row) {
                        auto src = srcAll + (packIndex * tile.rows + row) * pack;
                        auto gate = gateAll + row * halfChannels + packIndex * halfPack;
                        auto up = upAll + row * halfChannels + packIndex * halfPack;
                        for (int i = 0; i < inner; i += 2) {
                            gate[i / 2] = src[i];
                            up[i / 2] = src[i + 1];
                        }
                    }
                }
            } else {
                auto srcAll = reinterpret_cast<const uint32_t*>(temporaryOutput);
                auto gateAll = reinterpret_cast<uint32_t*>(gatesBytes);
                auto upAll = reinterpret_cast<uint32_t*>(upsBytes);
                for (int packIndex = 0; packIndex < packs; ++packIndex) {
                    const int inner = std::min(pack, channelCount - packIndex * pack);
                    for (int row = 0; row < tile.rows; ++row) {
                        auto src = srcAll + (packIndex * tile.rows + row) * pack;
                        auto gate = gateAll + row * halfChannels + packIndex * halfPack;
                        auto up = upAll + row * halfChannels + packIndex * halfPack;
                        for (int i = 0; i < inner; i += 2) {
                            gate[i / 2] = src[i];
                            up[i / 2] = src[i + 1];
                        }
                    }
                }
            }
            resource.siluFunction(siluBytes, gatesBytes, halfTotal);
            resource.mulFunction(upsBytes, siluBytes, upsBytes, halfTotal, -1);
            for (int row = 0; row < tile.rows; ++row) {
                const int route = mRouteOrder[tile.firstRoute + row];
                ::memcpy(output + (route * outputChannels / 2 + channel / 2) * bytes,
                         upsBytes + row * halfChannels * bytes, halfChannels * bytes);
            }
            return;
        }
        for (int row = 0; row < tile.rows; ++row) {
            const int route = mRouteOrder[tile.firstRoute + row];
            for (int offset = 0; offset < channelCount; offset += pack) {
                auto src = temporaryOutput + ((offset / pack) * tile.rows * pack + row * pack) * bytes;
                auto dst = output + (route * outputChannels + channel + offset) * bytes;
                ::memcpy(dst, src, std::min(pack, channelCount - offset) * bytes);
            }
        }
    };
    if (denseTileParallel || mRouteParallel) {
        MNN_CONCURRENCY_BEGIN(thread, mThreads) {
            for (int tileIndex = thread; tileIndex < tiles; tileIndex += mThreads) {
                if (denseTileParallel || !sharedDecodeInput) {
                    prepareTile(tileIndex, mScratch.host<uint8_t>() + thread * mScratchBytes);
                }
                const int endChannel = denseTileParallel ? mainChannels : outputChannels;
                for (int channel = 0; channel < endChannel; channel += mChannelTile) {
                    runTask(thread, tileIndex, channel,
                            std::min(mChannelTile, endChannel - channel), false);
                }
                if (denseTileParallel) {
                    for (int channel = mainChannels; channel < outputChannels; channel += mChannelTile) {
                        runTask(thread, tileIndex, channel, std::min(mChannelTile, outputChannels - channel), true);
                    }
                }
            }
        }
        MNN_CONCURRENCY_END();
    } else {
        MNN_CONCURRENCY_BEGIN(thread, mThreads) {
            if (thread < mainWorkers) {
                for (int task = mainCursor->fetch_add(1); task < mainTasks; task = mainCursor->fetch_add(1)) {
                    const int channel = task % mainSplits * mainTile;
                    runTask(thread, task / mainSplits, channel, std::min(mainTile, mainChannels - channel), false);
                }
            }
            for (int task = branchCursor->fetch_add(1); task < branchTasks; task = branchCursor->fetch_add(1)) {
                const int channel = mainChannels + task % branchSplits * branchTile;
                runTask(thread, task / branchSplits, channel, std::min(branchTile, outputChannels - channel), true);
            }
        }
        MNN_CONCURRENCY_END();
    }
    if (mRouteWeight) {
        const int C = resource.outputChannels;
        const int pack = resource.pack;
        const int bytes = resource.bytes;
        const auto tmp = mRouteTmp.host<uint8_t>();
        const auto yPtr = outputs[0]->host<uint8_t>();
        const auto wPtr = inputs[2]->host<uint8_t>();
        const bool lowpW = CPUBackend::getBytes(backend(), inputs[2]) == 2;
        const int nExperts = (int)resource.experts.size();
        const auto idsPtr = inputs[1]->host<int32_t>();
        const int foldThreads = mTokens >= 128 ? std::min(mThreads, UP_DIV(mTokens, 32)) : 1;
        auto fold = [&](int thread) {
            auto row = mRouteScratch.host<float>() + thread * (2 * C + mTopK);
            auto tmpF = row + C;
            auto wNorm = tmpF + C;
            const int begin = static_cast<int>((int64_t)mTokens * thread / foldThreads);
            const int end = static_cast<int>((int64_t)mTokens * (thread + 1) / foldThreads);
            for (int t = begin; t < end; ++t) {
                ::memset(row, 0, C * sizeof(float));
                if (mRouteWeight >= 2) {
                    float sum = mRouteEps;
                    for (int k = 0; k < mTopK; ++k) {
                        const int e = idsPtr[t * mTopK + k];
                        float g = 0.0f;
                        if (e >= 0 && e < nExperts) {
                            if (lowpW) {
                                int16_t one;
                                ::memcpy(&one, wPtr + ((int64_t)t * nExperts + e) * 2, 2);
                                core->MNNLowpToFp32(&one, &g, 1);
                            } else {
                                ::memcpy(&g, wPtr + ((int64_t)t * nExperts + e) * 4, 4);
                            }
                        }
                        wNorm[k] = g;
                        sum += g;
                    }
                    if (mRouteWeight == 3) {
                        float mx = -FLT_MAX;
                        for (int k = 0; k < mTopK; ++k) {
                            mx = std::max(mx, wNorm[k]);
                        }
                        sum = mRouteEps;
                        for (int k = 0; k < mTopK; ++k) {
                            wNorm[k] = std::exp(wNorm[k] - mx);
                            sum += wNorm[k];
                        }
                    }
                    for (int k = 0; k < mTopK; ++k) {
                        wNorm[k] = wNorm[k] / sum * mRouteScale;
                    }
                }
                for (int k = 0; k < mTopK; ++k) {
                    const auto srcRow = tmp + (int64_t)(t * mTopK + k) * C * bytes;
                    float wk;
                    if (mRouteWeight >= 2) {
                        wk = wNorm[k];
                    } else if (lowpW) {
                        int16_t one;
                        ::memcpy(&one, wPtr + (t * mTopK + k) * 2, 2);
                        core->MNNLowpToFp32(&one, &wk, 1);
                    } else {
                        ::memcpy(&wk, wPtr + (t * mTopK + k) * 4, 4);
                    }
                    const float* values = reinterpret_cast<const float*>(srcRow);
                    if (bytes == 2) {
                        core->MNNLowpToFp32(reinterpret_cast<const int16_t*>(srcRow), tmpF, C);
                        values = tmpF;
                    }
                    for (int c = 0; c < C; ++c) {
                        row[c] += wk * values[c];
                    }
                }
                if (mTokens == 1) {
                    if (bytes == 2) {
                        core->MNNFp32ToLowp(row, reinterpret_cast<int16_t*>(yPtr), C);
                    } else {
                        ::memcpy(yPtr, row, C * sizeof(float));
                    }
                    continue;
                }
                for (int c0 = 0; c0 < C; c0 += pack) {
                    auto dst = yPtr + ((int64_t)(c0 / pack) * mTokens * pack + (int64_t)t * pack) * bytes;
                    if (bytes == 2) {
                        core->MNNFp32ToLowp(row + c0, reinterpret_cast<int16_t*>(dst), pack);
                    } else {
                        ::memcpy(dst, row + c0, pack * sizeof(float));
                    }
                }
            }
        };
        if (foldThreads == 1) {
            fold(0);
        } else {
            MNN_CONCURRENCY_BEGIN(thread, foldThreads) {
                fold(thread);
            }
            MNN_CONCURRENCY_END();
        }
    }
    return NO_ERROR;
}

class CPUIndexedLinearCreator : public CPUBackend::Creator {
public:
    Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const Op* op,
                        Backend* backend) const override {
        const auto param = op->main_as_IndexedLinearParam();
        if (!param || !param->linears() || param->linears()->size() == 0 ||
            param->linears()->size() >= std::numeric_limits<int>::max() ||
            (param->inputMode() != 0 && param->inputMode() != 1)) {
            return nullptr;
        }
        int experts, inputChannels, outputChannels;
        if (!IndexedLinearCommon::dimensions(op, experts, inputChannels, outputChannels)) {
            return nullptr;
        }
        auto execution = new CPUIndexedLinear(backend, op);
        if (!execution->valid()) {
            delete execution;
            return nullptr;
        }
        return execution;
    }
};
REGISTER_CPU_OP_CREATOR_TRANSFORMER(CPUIndexedLinearCreator, OpType_IndexedLinear);
} // namespace MNN

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
