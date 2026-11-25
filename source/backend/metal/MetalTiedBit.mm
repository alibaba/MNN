// Copyright (c) Alibaba, Inc. and its affiliates.
#include "MetalTiedBit.hpp"
#include "MetalTiedBitShader.hpp"
#include "ConvSimdGroupShader.hpp"
#include "MetalEnv.hpp"
#include "MetalSharedGather.hpp"
#include "MNNMetalContext.h"
#include <cassert>
#include <climits>
#include <cstddef>
#include <cstdint>

#if MNN_METAL_ENABLED
namespace MNN {

// These are kernel requirements: 32 lanes per reduction and one thread per
// radix bucket. SIMD groups per threadgroup are selected at resize instead.
static constexpr int kSimdWidth = 32;
static constexpr int kAuxThreads = 256;

static int tiedBitScanLanesPerBlock(int blockQuads, int blockCount, bool fp16, const MetalGpuInfo& gpu) {
    const int chunks = blockQuads / 4; // The scan reads 16 channels per lane at a time.
    const bool proClassMac = !gpu.aSeries && gpu.tier >= MetalGpuInfo::TierPro;
    if (!fp16 && proClassMac && chunks <= kSimdWidth && (chunks & (chunks - 1)) == 0) {
        return chunks; // One FP32 input chunk per lane without a partial block.
    }

    int bestLanes = 1;
    int64_t bestSteps = INT64_MAX, bestTieBreak = INT64_MAX;
    for (int lanes = 1; lanes <= chunks && lanes <= kSimdWidth; lanes *= 2) {
        const int blocksPerPass = kSimdWidth / lanes;
        const int blockPasses = 1 + (blockCount - 1) / blocksPerPass;
        const int chunkSteps = 1 + (chunks - 1) / lanes;
        // With several block passes, the conservative FP32 mapping reuses each scale/center across chunks.
        if (!fp16 && !proClassMac && lanes > 1 && chunkSteps == 1 && blockPasses > 1) {
            continue;
        }
        const int64_t steps = int64_t(blockPasses) * chunkSteps;
        // A row that exactly fills one pass has no idle block slots or further block passes.
        const int passOverhead = blockCount == blocksPerPass ? 0 : blockPasses;
        const int64_t tieBreak = fp16 ? blockPasses : int64_t(chunkSteps) + passOverhead;
        if (steps < bestSteps || (steps == bestSteps && tieBreak < bestTieBreak)) {
            bestLanes = lanes;
            bestSteps = steps;
            bestTieBreak = tieBreak;
        }
    }
    return bestLanes;
}

// Keep the Q1 prefix in C4 scan order. Store the Q234 suffix by individual
// output row: three bytes per eight inputs (one upper-bit byte, two low-bit
// bytes). Together the planes occupy the original four bits per weight, with
// no padding between rows or additional quantization parameters.
static void metalTiedBitPackQ1Q234(const uint16_t* codes, uint8_t* planes, size_t inputSlices, size_t outputSlices) {
    const int shifts[] = {4, 0, 12, 8};
    assert(inputSlices % 4 == 0);
    const size_t prefixBytes = inputSlices * outputSlices * 2;
    const size_t residualRowBytes = inputSlices / 2 * 3;
    for (size_t row = 0; row < outputSlices; ++row) {
        for (size_t z = 0; z < inputSlices; ++z) {
            const size_t tile = row * inputSlices + z;
            auto* high = planes + tile * 2;
            for (int lane = 0; lane < 4; ++lane) {
                const uint16_t word = codes[tile * 4 + lane];
                uint8_t a = 0, b = 0, c = 0;
                for (int k = 0; k < 4; ++k) {
                    const uint8_t q = (word >> shifts[k]) & 15;
                    const int bit = 3 - k;
                    a |= (q >> 3) << bit;
                    b |= ((q >> 2) & 1) << bit;
                    c |= (q & 3) << (bit * 2);
                }
                const int byte = lane / 2;
                if (lane % 2 == 0)
                    high[byte] = a << 4;
                else
                    high[byte] |= a;
                auto* residual = planes + prefixBytes + (row * 4 + lane) * residualRowBytes + z / 2 * 3;
                if (z % 2 == 0)
                    residual[0] = b << 4;
                else
                    residual[0] |= b;
                residual[1 + z % 2] = c;
            }
        }
    }
}

MetalTiedBit::MetalTiedBit(Backend* backend, const Op* op, std::shared_ptr<Tensor> bias)
    : MetalConvolutionCommon(backend, op, std::move(bias)) {}

Execution* MetalTiedBit::create(MetalBackend* backend, const Op* op) {
    auto reject = [op](const char* reason) -> Execution* {
        MNN_PRINT("Metal TiedBit: skipping %s (%s), trying standard convolution\n", op->name()->c_str(), reason);
        return nullptr;
    };
    const auto rt = static_cast<const MetalRuntime*>(backend->runtime());
    const auto conv = op->main_as_Convolution2D();
    const auto common = conv->common();
    const auto quant = conv->quanParameter();
    const int ic = common->inputCount(), oc = common->outputCount();
    if (!rt->supportSimdGroupReduce()) {
        return reject("device does not support SIMD-group reduction");
    }
    if (rt->maxThreadSize() < kAuxThreads) {
        return reject("device cannot run the 256-bucket selector");
    }
    if (backend->getMemoryMode() != BackendConfig::Memory_Low) {
        return reject("requires low-memory mode");
    }
    if (!quant || (!conv->external() && !quant->buffer()) || quant->type() == 3 || quant->type() == 8 ||
        quant->has_scaleInt()) {
        return reject("requires weight-only INT4 quantization");
    }
    if (common->group() != 1 || common->kernelX() != 1 || common->kernelY() != 1 ||
        common->strideX() != 1 || common->strideY() != 1 || common->dilateX() != 1 || common->dilateY() != 1 ||
        common->padX() != 0 || common->padY() != 0 || common->relu() || common->relu6()) {
        return reject("requires an ungrouped 1x1 projection without padding or activation");
    }
    if (ic <= 0 || ic % 16 != 0 || oc <= 0) {
        return reject("requires positive dimensions and input channels divisible by 16");
    }
    std::unique_ptr<MetalTiedBit> result(new MetalTiedBit(backend, op));
    if (!result->valid()) {
        return reject("failed to allocate bias");
    }
    result->loadWeight(op, true);
    if (!result->valid() || !result->mWeight || !result->mDequantScaleBias) {
        return reject("failed to load quantized weights or parameters");
    }
    if (result->mDequantBits != 4) {
        return reject("loaded weights are not INT4");
    }
    const auto& weight = result->mWeight;
    const auto& dequant = result->mDequantScaleBias;
    const size_t inputSlices = ic / 4, outputSlices = UP_DIV(size_t(oc), 4);
    // Canonical weights are padded to OC16, but dequant parameters to OC4.
    // Infer the quantization group from the original metadata, not the model.
    const size_t weightRows = UP_DIV(size_t(oc), 16) * 16 / 4;
    const size_t weightBytes = inputSlices * weightRows * 8;               // 4 x 4 INT4 codes per tile.
    const size_t scalarBytes = backend->useFp16InsteadFp32() ? sizeof(uint16_t) : sizeof(float);
    const size_t metadataStride = outputSlices * 2 * 4 * scalarBytes;
    if (weightBytes > INT_MAX || weight->usize() != weightBytes || dequant->usize() <= sizeof(float) ||
        (dequant->usize() - sizeof(float)) % metadataStride != 0) {
        return reject("weight or scale/center layout is incompatible");
    }
    const size_t blocks = (dequant->usize() - sizeof(float)) / metadataStride;
    // SharedGather and the scan consume aligned groups of 16 input channels.
    if (blocks == 0 || blocks > inputSlices || inputSlices % blocks != 0 || (inputSlices / blocks) % 4 != 0) {
        return reject("quantization groups must evenly divide the input and align to 16 channels");
    }
    // loadWeight can upload/transform asynchronously. This fence occurs once,
    // before CPU reads; it is not part of the per-token path.
    backend->wait();
    std::shared_ptr<Tensor> split(Tensor::createDevice<uint8_t>({int(weightBytes)}));
    if (!backend->onAcquireBuffer(split.get(), Backend::STATIC)) {
        return reject("failed to allocate packed weights");
    }
    auto srcWeight = MetalBackend::getBuffer(weight.get());
    auto dstWeight = MetalBackend::getBuffer(split.get());
    const auto* codes =
        reinterpret_cast<const uint16_t*>(static_cast<const uint8_t*>(srcWeight.first.contents) + srcWeight.second);
    auto* planes = static_cast<uint8_t*>(dstWeight.first.contents) + dstWeight.second;
    metalTiedBitPackQ1Q234(codes, planes, inputSlices, outputSlices);
    const size_t packedBytes = inputSlices * outputSlices * 8;
    memset(planes + packedBytes, 0, weightBytes - packedBytes);
    result->mInputChannels = ic;
    result->mVocab = oc;
    result->mBlockQuads = int(inputSlices / blocks);
    result->mWeight = std::move(split); // Release canonical Q4; clones share only the packed allocation.
    return result.release();
}

bool MetalTiedBit::onClone(Backend* backend, const Op* op, Execution** dst) {
    if (!mValid || (op->type() != OpType_Convolution && op->type() != OpType_GatherV2)) {
        return false;
    }
    if (dst == nullptr) {
        return true;
    }
    if (op->type() == OpType_GatherV2) {
        *dst = new MetalSharedGather(backend, mVocab, mWeight, mDequantScaleBias, mDequantBits, mScaleCoef, true);
    } else {
        auto result = new MetalTiedBit(backend, op, mBias);
        result->mWeight = mWeight;
        result->mDequantScaleBias = mDequantScaleBias;
        result->mDequantBits = mDequantBits;
        result->mScaleCoef = mScaleCoef;
        result->mInputChannels = mInputChannels;
        result->mVocab = mVocab;
        result->mBlockQuads = mBlockQuads;
        *dst = result;
    }
    MNN_METAL_PROFILE_REGISTER_CLONE(backend, op, *dst);
    return true;
}

ErrorCode MetalTiedBit::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    if (inputs.size() != 1 || outputs.size() != 1) {
        MNN_ERROR("Metal TiedBit requires one input and one output\n");
        return NOT_SUPPORT;
    }
    const auto input = inputs[0];
    const auto output = outputs[0];
    // Packed weights cannot use the canonical Q4 GEMM for multi-vector inputs.
    auto singleVector = [](const Tensor* tensor) {
        int64_t vectors = tensor->batch();
        for (int i = 2; i < tensor->dimensions(); ++i)
            vectors *= tensor->length(i);
        return vectors == 1;
    };
    // elementSize includes NC4HW4 channel padding; compare logical channels.
    if (input->channel() != mInputChannels || output->channel() != mVocab || !singleVector(input) ||
        !singleVector(output)) {
        MNN_ERROR("Metal TiedBit requires a single last-token projection\n");
        return NOT_SUPPORT;
    }
    if (mParam != nil) {
        return NO_ERROR;
    }
    auto backend = static_cast<MetalBackend*>(this->backend());
    auto rt = static_cast<const MetalRuntime*>(backend->runtime());
    if (mConstBuffer == nil) {
        mConstBuffer = backend->getConstBuffer(sizeof(Param));
    }
    if (mConstBuffer == nil) {
        return OUT_OF_MEMORY;
    }
    const int blockCount = mInputChannels / 4 / mBlockQuads;
    const Param constants = {1, mInputChannels / 4, 1, 1, 1, UP_DIV(mVocab, 4), mVocab, 1,
                             blockCount, mActivationType, mScaleCoef};
    memcpy(mConstBuffer.contents, &constants, sizeof(constants));
    const char* names[StageCount] = {"tied_bit_prepare_c4_sg",         "tied_bit_c4_sg",
                                    "tied_bit_global_histogram_high", "tied_bit_global_select_high_sg",
                                    "tied_bit_global_histogram_low",  "tied_bit_global_select_low_sg",
                                    "tied_bit_global_compact",        "tied_bit_compact_rerank"};
    const bool fp16 = backend->useFp16InsteadFp32();
    NSMutableDictionary* macros = [@{
        @"ftype" : fp16 ? @"half" : @"float",
        @"ftype2" : fp16 ? @"half2" : @"float2",
        @"ftype4" : fp16 ? @"half4" : @"float4",
        @"ftype2x4" : fp16 ? @"half2x4" : @"float2x4",
        @"ftype4x4" : fp16 ? @"half4x4" : @"float4x4",
        @"MNN_METAL_FLOAT32_COMPUTER" : @"1"
    } mutableCopy];
    if (fp16) {
        macros[@"MNN_METAL_FLOAT16_STORAGE"] = @"1";
    }
    const auto& gpu = rt->gpuInfo();
    const int scanLanes = tiedBitScanLanesPerBlock(mBlockQuads, blockCount, fp16, gpu);
    macros[@"TIED_BIT_SCAN_LANES_PER_BLOCK"] = @(scanLanes);
    macros[@"TIED_BIT_BLOCK_QUADS"] = @(mBlockQuads);
    macros[@"TIED_BIT_SIMD_WIDTH"] = @(kSimdWidth);
    macros[@"TIED_BIT_C4_SCAN"] = @"1";
    auto option = [[MTLCompileOptions alloc] init];
    const std::string source = std::string(gBasicConvPrefix) + gMetalTiedBitShader;
    if (rt->maxThreadSize() < kAuxThreads) {
        MNN_ERROR("Metal TiedBit: GPU gen=%d tier=%d A-series=%d cannot run the 256-bucket selector\n",
                  gpu.gen, int(gpu.tier), int(gpu.aSeries));
        return NOT_SUPPORT;
    }
    mThreads = ThreadConfig();
    const int topM = std::min(mVocab, MetalEnv::get().tiedBitTopM);
    const bool splitCandidate = topM <= 32 && blockCount >= 2;
    const bool directDecode = topM >= 1024;
    if (directDecode) {
        macros[@"TIED_BIT_DIRECT_DECODE"] = @"1";
    }
    // Estimate the grid from the cost of refining an unsplit row.
    const int channelsPerIteration = kSimdWidth * 8;
    const int rowIterations = 1 + (mInputChannels - 1) / channelsPerIteration;
    const int targetThreadgroups = 2 * rowIterations;
    while (mThreads.rerankSimdGroups > (splitCandidate ? 2 : 1) &&
           1 + (topM - 1) / mThreads.rerankSimdGroups < targetThreadgroups) {
        mThreads.rerankSimdGroups /= 2;
    }
    bool useSplit;
    bool retry;
    do {
        retry = false;
        useSplit = splitCandidate && mThreads.rerankSimdGroups >= 2;
        macros[@"TIED_BIT_SPLIT_K"] = @(int(useSplit));
        macros[@"TIED_BIT_SCAN_SIMD_GROUPS"] = @(mThreads.scanSimdGroups);
        macros[@"TIED_BIT_RERANK_SIMD_GROUPS"] = @(mThreads.rerankSimdGroups);
        option.preprocessorMacros = macros;
        for (int i = 0; i < StageCount; ++i) {
            std::vector<std::string> key = {names[i], fp16 ? "half" : "float", "TIED_BIT_Q234_ROW_V1", "MNN_METAL_FLOAT32_COMPUTER",
                                           "C4_PREPARED_SUMS"};
            key.emplace_back("BLOCK_QUADS_" + std::to_string(mBlockQuads));
            key.emplace_back("SIMD_WIDTH_" + std::to_string(kSimdWidth));
            key.emplace_back("SCAN_SIMD_GROUPS_" + std::to_string(mThreads.scanSimdGroups));
            key.emplace_back("RERANK_SIMD_GROUPS_" + std::to_string(mThreads.rerankSimdGroups));
            if (i == Scan) {
                key.emplace_back("SCAN_LANES_" + std::to_string(scanLanes));
            }
            if (useSplit && (i == SelectLow || i == Exact)) {
                key.emplace_back("SPLIT_K2");
            }
            if (directDecode && i == Exact) {
                key.emplace_back("DIRECT_DECODE");
            }
            auto pipeline = rt->findPipeline(key);
            if (pipeline == nil && !rt->pipelineCompileFailed(key)) {
                pipeline = backend->makeComputePipelineWithSourceOption(source.c_str(), names[i], option);
                rt->insertPipeline(key, pipeline);
            }
            int* groups = i == Scan ? &mThreads.scanSimdGroups : i == Exact ? &mThreads.rerankSimdGroups : nullptr;
            const int threads = groups ? *groups * kSimdWidth : kAuxThreads;
            if (pipeline == nil || pipeline.maxTotalThreadsPerThreadgroup < threads ||
                pipeline.threadExecutionWidth != kSimdWidth) {
                if (groups && *groups > 1 && (pipeline == nil || pipeline.threadExecutionWidth == kSimdWidth)) {
                    *groups /= 2;
                    // Rebuild SelectLow as well: it writes Exact's indirect grid.
                    retry = true;
                    break;
                }
                MNN_ERROR("Metal TiedBit: kernel %s unsupported on GPU gen=%d tier=%d A-series=%d "
                          "(scan=%d, rerank=%d SIMD groups)\n", names[i], gpu.gen, int(gpu.tier), int(gpu.aSeries),
                          mThreads.scanSimdGroups, mThreads.rerankSimdGroups);
                return NOT_SUPPORT;
            }
            mPipelines[i] = pipeline;
        }
    } while (retry);
    if (MetalEnv::get().pipelineInfo) {
        MNN_PRINT("Metal TiedBit: gen=%d tier=%d A-series=%d, IC=%d OC=%d group=%d M=%d, "
                  "scan=%d rerank=%d threads, refine=%s, scanLanesPerBlock=%d\n",
                  gpu.gen, int(gpu.tier), int(gpu.aSeries), mInputChannels, mVocab, mBlockQuads * 4, topM,
                  mThreads.scanSimdGroups * kSimdWidth, mThreads.rerankSimdGroups * kSimdWidth,
                  useSplit ? "split_k2" : directDecode ? "direct" : "packed", scanLanes);
    }
    auto context = (__bridge MNNMetalContext*)backend->context();
    mSums = [context newDeviceBuffer:size_t(mInputChannels / 16) * sizeof(float) access:CPUTransparent];
    mIndices = [context newDeviceBuffer:size_t(mVocab) * sizeof(int32_t) access:CPUTransparent];
    mCount = [context newDeviceBuffer:sizeof(uint32_t) access:CPUTransparent];
    mArgs = [context newDeviceBuffer:sizeof(MTLDispatchThreadgroupsIndirectArguments) access:CPUTransparent];
    mHistogram = [context newDeviceBuffer:256 * sizeof(uint32_t) access:CPUTransparent];
    mThreshold = [context newDeviceBuffer:4 * sizeof(uint32_t) access:CPUTransparent];
    if (!mSums || !mIndices || !mCount || !mArgs || !mHistogram || !mThreshold)
        return OUT_OF_MEMORY;
    struct {
        int count, topk, reserved, chunk;
        float reservedFloat;
    } param = {mVocab, topM, 0, mThreads.histogramChunk, -1.0f};
    mParam = [context newDeviceBuffer:sizeof(param) bytes:&param access:CPUWriteOnly];
    return mParam ? NO_ERROR : OUT_OF_MEMORY;
}

void MetalTiedBit::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                            id<MTLComputeCommandEncoder> encoder) {
    const auto input = inputs[0];
    const auto output = outputs[0];
#if MNN_METAL_OP_PROFILE
    static_cast<MetalBackend*>(backend())->setProfileSubtag("tied_bit_global_rerank");
#endif
    auto bind = [&](id<MTLBuffer> buffer, int index) { [encoder setBuffer:buffer offset:0 atIndex:index]; };
    auto dispatch = [&](Stage stage, int groups, int threads) {
        [encoder setComputePipelineState:mPipelines[stage]];
        [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    };
    auto common = [&]() {
        MetalBackend::setTensor(input, encoder, 0);
        MetalBackend::setTensor(output, encoder, 1);
        bind(mConstBuffer, 2);
        MetalBackend::setTensor(mWeight.get(), encoder, 3);
        MetalBackend::setTensor(mBias.get(), encoder, 4);
        MetalBackend::setTensor(mDequantScaleBias.get(), encoder, 5);
    };
    MetalBackend::setTensor(input, encoder, 0);
    bind(mSums, 1);
    bind(mHistogram, 2);
    bind(mThreshold, 3);
    MetalBackend::setTensor(output, encoder, 4);
    bind(mConstBuffer, 5);
    dispatch(Prepare, 1, kAuxThreads);
    common();
    bind(mSums, 6);
    // Each scan SIMD group produces two C4 output rows.
    dispatch(Scan, UP_DIV(mVocab, 2 * 4 * mThreads.scanSimdGroups), mThreads.scanSimdGroups * kSimdWidth);
    MetalBackend::setTensor(output, encoder, 0);
    bind(mHistogram, 1);
    bind(mParam, 2);
    dispatch(High, UP_DIV(mVocab, mThreads.histogramChunk), kAuxThreads);
    bind(mHistogram, 0);
    bind(mThreshold, 1);
    bind(mParam, 2);
    dispatch(SelectHigh, 1, kAuxThreads);
    // Resolve the remaining bytes of the key from most to least significant.
    const bool fp16 = static_cast<MetalBackend*>(backend())->useFp16InsteadFp32();
    for (int shift = fp16 ? 0 : 16; shift >= 0; shift -= 8) {
        MetalBackend::setTensor(output, encoder, 0);
        bind(mHistogram, 1);
        bind(mThreshold, 2);
        bind(mParam, 3);
        if (!fp16) {
            [encoder setBytes:&shift length:sizeof(shift) atIndex:4];
        }
        dispatch(Low, UP_DIV(mVocab, mThreads.histogramChunk), kAuxThreads);
        if (shift > 0) {
            bind(mHistogram, 0);
            bind(mThreshold, 1);
            bind(mParam, 2);
            dispatch(SelectHigh, 1, kAuxThreads);
        }
    }
    bind(mHistogram, 0);
    bind(mThreshold, 1);
    bind(mCount, 2);
    bind(mArgs, 3);
    bind(mParam, 4);
    dispatch(SelectLow, 1, kAuxThreads);
    MetalBackend::setTensor(output, encoder, 0);
    bind(mThreshold, 1);
    bind(mCount, 2);
    bind(mIndices, 3);
    bind(mParam, 4);
    [encoder setComputePipelineState:mPipelines[Compact]];
    [encoder dispatchThreads:MTLSizeMake(mVocab, 1, 1) threadsPerThreadgroup:MTLSizeMake(kAuxThreads, 1, 1)];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    // SelectLow already knows the inclusive candidate count. Compact's buffer
    // barrier above still protects both the IDs and the count consumed by Exact.
    common();
    bind(mIndices, 6);
    bind(mCount, 7);
    [encoder setComputePipelineState:mPipelines[Exact]];
    [encoder dispatchThreadgroupsWithIndirectBuffer:mArgs
                               indirectBufferOffset:0
                              threadsPerThreadgroup:MTLSizeMake(mThreads.rerankSimdGroups * kSimdWidth, 1, 1)];
}

}  // namespace MNN
#endif
