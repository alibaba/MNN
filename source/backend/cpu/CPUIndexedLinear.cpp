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
#include "core/TensorUtils.hpp"
#include <algorithm>
#include <atomic>
#include <cstring>
#include <limits>

namespace MNN {
namespace {
const CoreFunctions* indexedLinearCore(Backend* backend) {
    auto cpu = static_cast<CPUBackend*>(backend);
    // IndexedLinear computes FP32 tensors in BF16 precision; it does not use the dense BF16 packers.
    return cpu->precisionMode() == BackendConfig::Precision_Low_BF16 ? MNNGetCoreFunctions() : cpu->functions();
}
} // namespace

struct CPUIndexedLinear::Resource {
    struct Expert {
        std::shared_ptr<Tensor> weight;
        std::shared_ptr<Tensor> bias;
        std::shared_ptr<Tensor> weightSum;
        int weightTileBytes = 0;
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
    MatmulRelatedFunctions quantFunctions;
    decltype(MatmulRelatedFunctions::Int8GemmKernel) quantGemm = nullptr;
    decltype(CoreInt8Functions::MNNFloat2Int8) quantInput = nullptr;
    float clamp[2] = {-std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
};

CPUIndexedLinear::CPUIndexedLinear(Backend* backend, const Op* op) : Execution(backend) {
    const auto param = op->main_as_IndexedLinearParam();
    auto core = indexedLinearCore(backend);
    mResource.reset(new Resource);
    auto& resource = *mResource;
    bool forceFloat = false;
#ifndef MNN_LOW_MEMORY
    forceFloat = true;
#endif
    resource.core = core;
    resource.bytes = core->bytes;
    resource.pack = core->pack;
    mInputMode = param->inputMode();
    mFuseSwiGLU = param->fusedActivation() == UnaryOpOperation_SILU;
    int experts;
    if (!IndexedLinearCommon::dimensions(op, experts, resource.inputChannels, resource.outputChannels)) {
        mValid = false;
        return;
    }
    const int inputChannels = resource.inputChannels;
    const int outputChannels = resource.outputChannels;
    int weightTileBytes = 0;
    auto firstWeight = IndexedLinearCommon::loadLinear(op, 0, backend, forceFloat);
    if (!firstWeight) {
        mValid = false;
        return;
    }
    resource.quantized = !firstWeight->weightFloat.get();
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
        if (inputChannels % resource.blocks || resource.blockChannels % resource.lPack) {
            mValid = false;
            return;
        }
        if (resource.bytes == 2) {
            resource.quantGemm = resource.weightBits == 4 ? resource.quantFunctions.MNNGemmInt8AddBiasScale_w4_Unit_FP16
                                                          : resource.quantFunctions.MNNGemmInt8AddBiasScale_Unit_FP16;
            core->MNNFp32ToLowp(resource.clamp, reinterpret_cast<int16_t*>(resource.clamp), 2);
        } else {
            resource.quantGemm = resource.weightBits == 4 ? resource.quantFunctions.Int8GemmKernel_W4
                                                          : resource.quantFunctions.Int8GemmKernel;
        }
        const int64_t tileBytes = (int64_t)resource.blocks * resource.hPack *
                                  ((int64_t)resource.blockChannels * resource.weightBits / 8 + 2 * sizeof(float));
        if (tileBytes > std::numeric_limits<int>::max()) {
            mValid = false;
            return;
        }
        weightTileBytes = static_cast<int>(tileBytes);
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
    const int64_t alignedInput = ROUND_UP((int64_t)inputChannels, resource.lPack);
    const int64_t alignedOutput64 = ROUND_UP((int64_t)outputChannels, std::max(resource.pack, resource.hPack));
    const int64_t alignedWeightOutput64 = ROUND_UP((int64_t)outputChannels, resource.hPack);
    const int64_t weightBytes64 = resource.quantized ? alignedWeightOutput64 / resource.hPack * weightTileBytes
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
    resource.experts.resize(param->linears()->size());
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
            resource.quantized
                ? static_cast<int>((int64_t)resource.blocks * resource.hPack *
                                   ((int64_t)resource.blockChannels * resource.weightBits / 8 + 2 * quantInfoBytes))
                : 0;
        const int expertWeightBytes =
            resource.quantized ? static_cast<int>(alignedWeightOutput64 / resource.hPack) * expert.weightTileBytes
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
            const int hPack = resource.hPack;
            const int lPack = resource.lPack;
            const int blockChannels = resource.blockChannels;
            const int blockDepth = blockChannels / lPack;
            const int weightTiles = UP_DIV(outputChannels, hPack);
            const int alignedWeightOutput = weightTiles * hPack;
            std::vector<uint8_t> unpacked(outputChannels * inputChannels);
            if (bits == 4) {
                const auto packed = reinterpret_cast<const uint8_t*>(decoded->weight.get());
                for (int element = 0; element < unpacked.size(); ++element) {
                    unpacked[element] = element % 2 ? packed[element / 2] & 15 : packed[element / 2] >> 4;
                }
            } else {
                ::memcpy(unpacked.data(), decoded->weight.get(), unpacked.size());
            }
            std::vector<uint8_t> reordered(alignedWeightOutput * inputChannels);
            int32_t reorderInfo[6] = {resource.blocks, outputChannels, inputChannels, 1, hPack, lPack};
            ConvInt8TiledExecutor::reorderWeight(reordered.data(), unpacked.data(), reorderInfo);
            if (bits == 4) {
                const int cellElements = hPack * lPack;
                const int cellBytes = cellElements / 2;
                // Match the existing W4 kernel ABI: SME stores adjacent nibbles;
                // the other kernels store the two halves of an HP x LP cell.
                for (int cell = 0; cell < reordered.size() / cellElements; ++cell) {
                    auto source = reordered.data() + cell * cellElements;
                    auto target = reordered.data() + cell * cellBytes;
                    for (int byte = 0; byte < cellBytes; ++byte) {
                        if (resource.ePack == GEMM_INT8_DST_XUNIT_SME2 && hPack == GEMM_INT8_UNIT_SME2) {
                            target[byte] = source[byte * 2] | (source[byte * 2 + 1] << 4);
                        } else {
                            target[byte] = (source[byte] << 4) | source[byte + cellBytes];
                        }
                    }
                }
            }
            std::vector<float> quantInfo(2 * resource.blocks * alignedWeightOutput, 0.0f);
            std::vector<int16_t> halfQuantInfo;
            if (expert.weightQuantInfoMode) {
                halfQuantInfo.resize(quantInfo.size(), 0);
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
            for (int channel = 0; channel < outputChannels; ++channel) {
                for (int block = 0; block < resource.blocks; ++block) {
                    const int index = channel * resource.blocks + block;
                    const float scale = alpha[index * (decoded->asymmetric ? 2 : 1) + (decoded->asymmetric ? 1 : 0)];
                    const float offset =
                        (decoded->asymmetric ? alpha[index * 2] : 0.0f) - (bits == 4 ? 8.0f * scale : 0.0f);
                    quantInfo[block * alignedWeightOutput + channel] = scale;
                    quantInfo[(resource.blocks + block) * alignedWeightOutput + channel] = offset;
                    if (expert.weightQuantInfoMode) {
                        const int components = decoded->asymmetric ? 2 : 1;
                        halfQuantInfo[block * alignedWeightOutput + channel] =
                            decoded->alphaHalf.get()[index * components + components - 1];
                        // The kernel reconstructs the W4 offset in FP32. Rounding the folded offset
                        // to FP16 here would change the represented weights.
                        halfQuantInfo[(resource.blocks + block) * alignedWeightOutput + channel] =
                            decoded->asymmetric ? decoded->alphaHalf.get()[index * 2] : 0;
                    }
                    int sum = 0;
                    for (int inner = 0; inner < blockChannels; ++inner) {
                        const auto value = unpacked[channel * inputChannels + block * blockChannels + inner];
                        sum += bits == 4 ? static_cast<int>(value) : static_cast<int>(static_cast<int8_t>(value));
                    }
                    const float blockSum = sum * scale + blockChannels * offset;
                    expert.weightSum
                        ->host<float>()[(channel / hPack) * resource.blocks * hPack + block * hPack + channel % hPack] =
                        blockSum;
                    expert.weightSum->host<float>()[alignedWeightOutput * resource.blocks + channel] += blockSum;
                }
            }
            int32_t packInfo[6] = {resource.blocks, weightTiles,      blockDepth,
                                   hPack,           lPack * bits / 8, alignedWeightOutput};
            ConvInt8TiledExecutor::packWeightAndQuantInfo(
                expert.weight->host<int8_t>(), reinterpret_cast<const int8_t*>(reordered.data()),
                expert.weightQuantInfoMode ? reinterpret_cast<const int8_t*>(halfQuantInfo.data())
                                           : reinterpret_cast<const int8_t*>(quantInfo.data()),
                packInfo, quantInfoBytes);
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

CPUIndexedLinear::CPUIndexedLinear(Backend* backend, std::shared_ptr<Resource> resource, int inputMode, bool fuseSwiGLU)
    : Execution(backend), mResource(std::move(resource)), mInputMode(inputMode), mFuseSwiGLU(fuseSwiGLU) {}

bool CPUIndexedLinear::onClone(Backend* backend, const Op* op, Execution** dst) {
    if (!dst) {
        return mValid;
    }
    if (!backend || (backend->type() != MNN_FORWARD_CPU && backend->type() != MNN_FORWARD_CPU_EXTENSION)) {
        return false;
    }
    int experts, inputChannels, outputChannels;
    if (!IndexedLinearCommon::dimensions(op, experts, inputChannels, outputChannels) ||
        experts != mResource->experts.size() || inputChannels != mResource->inputChannels ||
        outputChannels != mResource->outputChannels || op->main_as_IndexedLinearParam()->inputMode() != mInputMode ||
        (op->main_as_IndexedLinearParam()->fusedActivation() == UnaryOpOperation_SILU) != mFuseSwiGLU) {
        return false;
    }
    auto core = indexedLinearCore(backend);
    if (!mValid || core != mResource->core || core->bytes != mResource->bytes || core->pack != mResource->pack) {
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
        *dst = new CPUIndexedLinear(backend, mResource, mInputMode, mFuseSwiGLU);
    }
    return true;
}

ErrorCode CPUIndexedLinear::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    if (inputs.size() != 2 || outputs.size() != 1) {
        return INPUT_DATA_ERROR;
    }
    const auto& resource = *mResource;
    const auto x = inputs[0];
    const auto ids = inputs[1];
    const auto y = outputs[0];
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
    // Low precision CPU backends keep a logical float32 tensor type while
    // storing core->bytes per element. Packed formats are not row-major.
    if (x->getType() != halide_type_of<float>() || y->getType() != halide_type_of<float>() ||
        ids->getType() != halide_type_of<int32_t>() || !contiguous(x) || !contiguous(ids) || !contiguous(y) ||
        CPUBackend::getBytes(backend(), x) != resource.bytes || CPUBackend::getBytes(backend(), y) != resource.bytes ||
        x->dimensions() != (mInputMode == 0 ? 2 : 3) || ids->dimensions() != 2 || y->dimensions() != 3 ||
        x->length(0) != ids->length(0) || x->length(x->dimensions() - 1) != resource.inputChannels ||
        ids->length(1) <= 0 || (mInputMode == 1 && x->length(1) != ids->length(1)) || y->length(0) != ids->length(0) ||
        y->length(1) != ids->length(1) || y->length(2) != resource.outputChannels / (mFuseSwiGLU ? 2 : 1)) {
        return INPUT_DATA_ERROR;
    }
    mTokens = inputs[1]->length(0);
    mTopK = inputs[1]->length(1);
    const int routes = mTokens * mTopK;
    const int64_t maxTiles64 = mTokens == 1
                                   ? routes
                                   : std::min<int64_t>(routes, UP_DIV((int64_t)routes, resource.ePack) +
                                                                   std::min<int64_t>(routes, resource.experts.size()));
    mThreads = static_cast<CPUBackend*>(backend())->threadNumber();
    mChannelTile = ROUND_UP(64, std::max(resource.pack, resource.hPack));
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
    const int64_t accumulatorBytes = (int64_t)mThreads * resource.ePack * resource.hPack * sizeof(float);
    const int64_t maxSize = std::numeric_limits<int>::max();
    if (mThreads <= 0 || maxTiles64 > maxSize || inputTileBytes > maxSize || scratchBytes > maxSize ||
        quantInfoBytes > maxSize || outputTileBytes > maxSize || maxTiles64 * inputTileBytes > maxSize ||
        mThreads * scratchBytes > maxSize || mThreads * outputTileBytes > maxSize ||
        (resource.quantized && (maxTiles64 * quantInfoBytes > maxSize || accumulatorBytes > maxSize)) ||
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
    backend()->onReleaseBuffer(&mPackedInput, Backend::DYNAMIC);
    backend()->onReleaseBuffer(&mScratch, Backend::DYNAMIC);
    backend()->onReleaseBuffer(&mPackedOutput, Backend::DYNAMIC);
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
    const auto ids = inputs[1]->host<int32_t>();
    mTiles.clear();
    // Decode uses the K route entries directly, without touching all E experts.
    if (mTokens == 1) {
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
    const auto output = outputs[0]->host<uint8_t>();
    const bool sharedDecodeInput = mTokens == 1 && mInputMode == 0;
    const int inputTiles = sharedDecodeInput ? std::min<size_t>(1, mTiles.size()) : mTiles.size();
    // Like DenseConvInt8TiledExecutor's 1x1 fastway, options 0 and 1 use
    // per-row symmetric prefill and asymmetric single-row decode. Option 1's
    // tensor-wise override applies only to general convolutions, not linears.
    const int dynamicQuantOption = static_cast<CPUBackend*>(backend())->getRuntime()->hint().dynamicQuantOption % 8;
    MNN_CONCURRENCY_BEGIN(thread, mThreads) {
        auto scratch = mScratch.host<uint8_t>() + thread * mScratchBytes;
        for (int tileIndex = thread; tileIndex < inputTiles; tileIndex += mThreads) {
            const auto& tile = mTiles[tileIndex];
            ::memset(scratch, 0, mScratchBytes);
            const int inputPack = resource.quantized ? resource.lPack : pack;
            for (int row = 0; row < tile.rows; ++row) {
                int route = mRouteOrder[tile.firstRoute + row];
                int sourceRow = mInputMode == 0 ? route / mTopK : route;
                for (int channel = 0; channel < channels; channel += inputPack) {
                    auto dst = scratch + ((channel / inputPack) * tile.rows * inputPack + row * inputPack) * bytes;
                    auto src = input + (sourceRow * channels + channel) * bytes;
                    ::memcpy(dst, src, std::min(inputPack, channels - channel) * bytes);
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
                        resource.quantInput(reinterpret_cast<const float*>(scratch), packed, UP_DIV(channels, pack),
                                            qscale, -128, 127, qbias, 0);
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
                continue;
            }
            const float* source = reinterpret_cast<const float*>(scratch);
            int32_t info[4] = {1, tile.rows, tile.rows, 1};
            // GEMM reads the aligned L extent, including the zero-filled scratch tail.
            int32_t extent[4] = {tile.rows, ROUND_UP(channels, resource.lPack), 0, 0};
            core->MNNPackC4ForMatMul_A(
                reinterpret_cast<float*>(mPackedInput.host<uint8_t>() + tileIndex * mInputTileBytes), &source, info,
                extent);
        }
    }
    MNN_CONCURRENCY_END();
    const int channelTiles = UP_DIV(outputChannels, mChannelTile);
    const int taskCount = mTiles.size() * channelTiles;
    std::atomic<int> nextTask{0};
    auto taskCursor = &nextTask;
    const float postClamp[4] = {1.0f, 1.0f, -std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
    const float* post        = postClamp;
    MNN_CONCURRENCY_BEGIN(thread, mThreads) {
        auto temporaryOutput = mPackedOutput.host<uint8_t>() + thread * mOutputTileBytes;
        for (int task = taskCursor->fetch_add(1); task < taskCount; task = taskCursor->fetch_add(1)) {
            const int tileIndex = task / channelTiles;
            const int channel = task % channelTiles * mChannelTile;
            const int channelCount = std::min(mChannelTile, outputChannels - channel);
            const auto& tile = mTiles[tileIndex];
            const auto& expert = resource.experts[tile.expert];
            const int packedTileIndex = sharedDecodeInput ? 0 : tileIndex;
            if (resource.quantized) {
                auto info =
                    reinterpret_cast<const float*>(mQuantInfo.host<uint8_t>() + packedTileIndex * mQuantInfoBytes);
                auto accumulator = mAccumulator.host<float>() + thread * resource.ePack * resource.hPack;
                QuanPostTreatParameters post{};
                post.useInt8 = 0;
                post.blockNum = resource.blocks;
                post.weightQuantInfoMode = expert.weightQuantInfoMode;
                post.inputScale = info;
                post.inputBias = info + resource.blocks * resource.ePack;
                post.srcKernelSum = const_cast<float*>(info + 2 * resource.blocks * resource.ePack);
                post.weightKernelSum = expert.weightSum->host<float>() + channel * resource.blocks;
                post.biasFloat = expert.bias->host<float>() + channel;
                const int expertRows = mTokens == 1 ? 1 : mOffsets[tile.expert + 1] - mOffsets[tile.expert];
                if (expertRows == 1 && dynamicQuantOption != 2) {
                    // Ordinary Linear combines the one input zero point with
                    // bias once, after summing weight blocks. Preserve that
                    // order so tiny differences do not change later MoE routes.
                    auto fusedBias = reinterpret_cast<float*>(mScratch.host<uint8_t>() + thread * mScratchBytes);
                    auto weightSum = expert.weightSum->host<float>() +
                                     ROUND_UP(outputChannels, resource.hPack) * resource.blocks + channel;
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
                auto weight = expert.weight->host<int8_t>() + (channel / resource.hPack) * expert.weightTileBytes;
                resource.quantGemm(
                    reinterpret_cast<int8_t*>(temporaryOutput),
                    reinterpret_cast<const int8_t*>(mPackedInput.host<uint8_t>() + packedTileIndex * mInputTileBytes),
                    weight, resource.blockChannels / resource.lPack, tile.rows * pack * bytes,
                    UP_DIV(channelCount, pack), &post, tile.rows);
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
                auto values = reinterpret_cast<float*>(mScratch.host<uint8_t>() + thread * mScratchBytes);
                const int physicalCount = tile.rows * ROUND_UP(channelCount, pack);
                if (bytes == 2) {
                    core->MNNLowpToFp32(reinterpret_cast<const int16_t*>(temporaryOutput), values, physicalCount);
                } else {
                    ::memcpy(values, temporaryOutput, physicalCount * sizeof(float));
                }
                const int gateCount = tile.rows * channelCount / 2;
                auto gates = values + physicalCount;
                auto activated = gates + gateCount;
                for (int row = 0; row < tile.rows; ++row) {
                    for (int offset = 0; offset < channelCount; offset += 2) {
                        const int index = (offset / pack) * tile.rows * pack + row * pack + offset % pack;
                        gates[row * channelCount / 2 + offset / 2] = values[index];
                    }
                }
                const bool lowPrecision =
                    bytes == 2 || static_cast<CPUBackend*>(backend())->precisionMode() == BackendConfig::Precision_Low;
                (lowPrecision ? MNNSiLuLowp : MNNSiLu)(activated, gates, gateCount);
                if (bytes == 2) {
                    core->MNNFp32ToLowp(activated, reinterpret_cast<int16_t*>(temporaryOutput), gateCount);
                    core->MNNLowpToFp32(reinterpret_cast<const int16_t*>(temporaryOutput), activated, gateCount);
                }
                for (int row = 0; row < tile.rows; ++row) {
                    for (int offset = 0; offset < channelCount; offset += 2) {
                        const int index = (offset / pack) * tile.rows * pack + row * pack + offset % pack;
                        activated[row * channelCount / 2 + offset / 2] *= values[index + 1];
                    }
                }
                if (bytes == 2)
                    core->MNNFp32ToLowp(activated, reinterpret_cast<int16_t*>(temporaryOutput), gateCount);
                for (int row = 0; row < tile.rows; ++row) {
                    const int route = mRouteOrder[tile.firstRoute + row];
                    auto src = bytes == 2 ? temporaryOutput + row * channelCount / 2 * bytes
                                          : reinterpret_cast<uint8_t*>(activated + row * channelCount / 2);
                    auto dst = output + (route * outputChannels / 2 + channel / 2) * bytes;
                    ::memcpy(dst, src, channelCount / 2 * bytes);
                }
                continue;
            }
            for (int row = 0; row < tile.rows; ++row) {
                const int route = mRouteOrder[tile.firstRoute + row];
                for (int offset = 0; offset < channelCount; offset += pack) {
                    auto src = temporaryOutput + ((offset / pack) * tile.rows * pack + row * pack) * bytes;
                    auto dst = output + (route * outputChannels + channel + offset) * bytes;
                    ::memcpy(dst, src, std::min(pack, channelCount - offset) * bytes);
                }
            }
        }
    }
    MNN_CONCURRENCY_END();
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
