//
//  MetalIndexedLinearShader.hpp
//  MNN
//
//  Created by MNN on 2026/09/29.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalIndexedLinearShader_hpp
#define MetalIndexedLinearShader_hpp

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#if MNN_METAL_ENABLED
static const char* gIndexedLinear = R"metal(
#include <metal_stdlib>
using namespace metal;

struct indexed_linear_params {
    int inputChannels;
    int outputChannels;
    int experts;
    int topK;
    int routes;
    int inputMode;
    int blockSize;
    int blockCount;
    int weightStride;
};

inline float indexed_linear_weight(const device uchar* weights, const device float2* scales,
                                   constant indexed_linear_params& p, int expert, int output, int input) {
    const ulong index = ulong(output) * p.inputChannels + input;
    const device uchar* bank = weights + ulong(expert) * p.weightStride;
#if WEIGHT_BITS == 4
    const uchar packed = bank[index / 2];
    const int q = int((index & 1) ? (packed & 15) : (packed >> 4)) - 8;
#elif WEIGHT_BITS == 8
    const int q = int(((const device char*)bank)[index]);
#else
    return float(((const device ftype*)bank)[index]);
#endif
#if WEIGHT_BITS != 0
    const float2 scale = scales[(ulong(expert) * p.outputChannels + output) * p.blockCount + input / p.blockSize];
    return float(q) * scale.x + scale.y;
#endif
}

inline ftype indexed_linear_swiglu(float gate, float up) {
    const float g = float(ftype(gate));
    const ftype activated = ftype(g / (1.0f + exp(-g)));
    return ftype(float(activated) * float(ftype(up)));
}

// One simdgroup produces four output channels of one selected expert. The
// expert index stays on the device, including the single-token decode case.
kernel void indexed_linear_gemv(const device ftype* input [[buffer(0)]],
                                const device int* ids [[buffer(1)]],
                                device ftype* output [[buffer(2)]],
                                constant indexed_linear_params& p [[buffer(3)]],
                                const device uchar* weights [[buffer(4)]],
                                const device float2* scales [[buffer(5)]],
                                const device float* bias [[buffer(6)]],
                                uint3 group [[threadgroup_position_in_grid]],
                                uint lane [[thread_index_in_simdgroup]],
                                uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    const int route = int(group.y);
    const int firstOutput = (int(group.x) * 4 + int(simdgroup)) * 4;
    const int resultChannels = p.outputChannels / (FUSE_SILU_MUL ? 2 : 1);
    if (route >= p.routes || firstOutput >= resultChannels) {
        return;
    }
    const int expert = ids[route];
    const bool validExpert = expert >= 0 && expert < p.experts;
    const int sourceRow = p.inputMode == 0 ? route / p.topK : route;
    const device ftype* row = input + ulong(sourceRow) * p.inputChannels;
    float4 sum = 0.0f;
#if FUSE_SILU_MUL
    float4 upSum = 0.0f;
#endif
    if (validExpert) {
        for (int i = int(lane) * 4; i < p.inputChannels; i += 128) {
            float4 x = 0.0f;
            for (int j = 0; j < 4; ++j) {
                if (i + j < p.inputChannels) x[j] = float(row[i + j]);
            }
            for (int o = 0; o < 4; ++o) {
                if (firstOutput + o >= resultChannels) break;
                float4 w = 0.0f;
#if FUSE_SILU_MUL
                float4 upWeight = 0.0f;
#endif
                for (int j = 0; j < 4; ++j) {
                    if (i + j < p.inputChannels) {
                        const int channel = (firstOutput + o) * (FUSE_SILU_MUL ? 2 : 1);
                        w[j] = indexed_linear_weight(weights, scales, p, expert, channel, i + j);
#if FUSE_SILU_MUL
                        upWeight[j] = indexed_linear_weight(weights, scales, p, expert, channel + 1, i + j);
#endif
                    }
                }
                sum[o] += dot(x, w);
#if FUSE_SILU_MUL
                upSum[o] += dot(x, upWeight);
#endif
            }
        }
    }
    sum = float4(simd_sum(sum.x), simd_sum(sum.y), simd_sum(sum.z), simd_sum(sum.w));
#if FUSE_SILU_MUL
    upSum = float4(simd_sum(upSum.x), simd_sum(upSum.y), simd_sum(upSum.z), simd_sum(upSum.w));
#endif
    if (lane == 0) {
        for (int o = 0; o < 4 && firstOutput + o < resultChannels; ++o) {
            const ulong channel = ulong(expert) * p.outputChannels + (firstOutput + o) * (FUSE_SILU_MUL ? 2 : 1);
            const float b = validExpert ? bias[channel] : 0.0f;
#if FUSE_SILU_MUL
            const float upBias = validExpert ? bias[channel + 1] : 0.0f;
            output[ulong(route) * resultChannels + firstOutput + o] = indexed_linear_swiglu(sum[o] + b, upSum[o] + upBias);
#else
            output[ulong(route) * resultChannels + firstOutput + o] = ftype(sum[o] + b);
#endif
        }
    }
}

kernel void indexed_linear_clear(device atomic_uint* counts [[buffer(0)]],
                                  constant indexed_linear_params& p [[buffer(1)]],
                                  uint index [[thread_position_in_grid]]) {
    if (index < uint(p.experts)) atomic_store_explicit(counts + index, 0u, memory_order_relaxed);
}

kernel void indexed_linear_histogram(const device int* ids [[buffer(0)]],
                                      device atomic_uint* counts [[buffer(1)]],
                                      device int* ranks [[buffer(2)]],
                                      constant indexed_linear_params& p [[buffer(3)]],
                                      uint route [[thread_position_in_grid]]) {
    if (route >= uint(p.routes)) return;
    const int expert = ids[route];
    ranks[route] = expert >= 0 && expert < p.experts
        ? int(atomic_fetch_add_explicit(counts + expert, 1u, memory_order_relaxed)) : -1;
}

// Only O(E) work. Both route offsets and tile offsets are GPU-resident; the
// host dispatches a capacity bound, and inactive tile groups return uniformly.
kernel void indexed_linear_prefix(const device uint* counts [[buffer(0)]],
                                   device int* offsets [[buffer(1)]],
                                   device int* tileOffsets [[buffer(2)]],
                                   constant indexed_linear_params& p [[buffer(3)]]) {
    int routes = 0, tiles = 0;
    for (int e = 0; e < p.experts; ++e) {
        offsets[e] = routes;
        tileOffsets[e] = tiles;
        routes += int(counts[e]);
        tiles += (int(counts[e]) + 15) / 16;
    }
    offsets[p.experts] = routes;
    tileOffsets[p.experts] = tiles;
}

kernel void indexed_linear_scatter(const device int* ids [[buffer(0)]],
                                    const device int* ranks [[buffer(1)]],
                                    const device int* offsets [[buffer(2)]],
                                    device int* order [[buffer(3)]],
                                    device ftype* output [[buffer(4)]],
                                    constant indexed_linear_params& p [[buffer(5)]],
                                    uint route [[thread_position_in_grid]]) {
    if (route >= uint(p.routes)) return;
    if (ranks[route] < 0) {
        // Malformed IDs must never index outside the resident weight bank.
        for (int o = 0; o < p.outputChannels / (FUSE_SILU_MUL ? 2 : 1); ++o)
            output[ulong(route) * (p.outputChannels / (FUSE_SILU_MUL ? 2 : 1)) + o] = ftype(0);
        return;
    }
    order[offsets[ids[route]] + ranks[route]] = int(route);
}

kernel void indexed_linear_grouped(const device ftype* input [[buffer(0)]],
                                   device ftype* output [[buffer(1)]],
                                   constant indexed_linear_params& p [[buffer(2)]],
                                   const device uchar* weights [[buffer(3)]],
                                   const device float2* scales [[buffer(4)]],
                                   const device float* bias [[buffer(5)]],
                                   const device int* offsets [[buffer(6)]],
                                   const device int* tileOffsets [[buffer(7)]],
                                   const device int* order [[buffer(8)]],
                                   uint3 group [[threadgroup_position_in_grid]],
                                   uint threadIndex [[thread_index_in_threadgroup]],
                                   uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    const int tile = int(group.y);
    if (tile >= tileOffsets[p.experts]) return; // Threadgroup-uniform, before barriers.
    int low = 0, high = p.experts;
    while (low + 1 < high) {
        const int mid = (low + high) / 2;
        if (tileOffsets[mid] <= tile) low = mid; else high = mid;
    }
    const int expert = low;
    const int firstRow = offsets[expert] + (tile - tileOffsets[expert]) * 16;
    const int endRow = offsets[expert + 1];
    const int firstOutput = int(group.x) * 16;
    threadgroup ftype aTile[16 * 32];
    threadgroup ftype bTile[32 * 16];
#if USE_SIMDGROUP_MATRIX || FUSE_SILU_MUL
    threadgroup float cTile[16 * 16];
#endif
#if USE_SIMDGROUP_MATRIX
    simdgroup_float8x8 accum = make_filled_simdgroup_matrix<float, 8>(0.0f);
#else
    float accum0 = 0.0f, accum1 = 0.0f;
#endif
    for (int firstInput = 0; firstInput < p.inputChannels; firstInput += 32) {
        for (uint i = threadIndex; i < 512; i += 128) {
            const int rowIndex = firstRow + int(i / 32);
            const int inputChannel = firstInput + int(i % 32);
            ftype a = ftype(0);
            if (rowIndex < endRow && inputChannel < p.inputChannels) {
                const int route = order[rowIndex];
                const int sourceRow = p.inputMode == 0 ? route / p.topK : route;
                a = input[ulong(sourceRow) * p.inputChannels + inputChannel];
            }
            aTile[i] = a;
            // Stage B transposed: adjacent lanes unpack adjacent input channels
            // instead of striding across full output-channel weight rows.
            const int k = firstInput + int(i % 32);
            const int o = firstOutput + int(i / 32);
            bTile[i] = k < p.inputChannels && o < p.outputChannels
                ? ftype(indexed_linear_weight(weights, scales, p, expert, o, k)) : ftype(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
#if USE_SIMDGROUP_MATRIX
        for (int k = 0; k < 32; k += 8) {
            simdgroup_half8x8 a, b;
            simdgroup_load(a, aTile + (simdgroup / 2) * 8 * 32 + k, 32);
            simdgroup_load(b, bTile + (simdgroup % 2) * 8 * 32 + k, 32, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(accum, a, b, accum);
        }
#else
        const uint row = threadIndex / 16;
        const uint col = threadIndex % 16;
        for (int k = 0; k < 32; ++k) {
            const float b = float(bTile[col * 32 + k]);
            accum0 += float(aTile[row * 32 + k]) * b;
            accum1 += float(aTile[(row + 8) * 32 + k]) * b;
        }
#endif
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
#if USE_SIMDGROUP_MATRIX
    simdgroup_store(accum, cTile + (simdgroup / 2) * 8 * 16 + (simdgroup % 2) * 8, 16);
    threadgroup_barrier(mem_flags::mem_threadgroup);
#endif
#if FUSE_SILU_MUL
#if !USE_SIMDGROUP_MATRIX
    cTile[threadIndex] = accum0;
    cTile[threadIndex + 128] = accum1;
    threadgroup_barrier(mem_flags::mem_threadgroup);
#endif
    const int resultChannels = p.outputChannels / 2;
    const int rowIndex = firstRow + int(threadIndex / 8);
    const int channel = firstOutput + int(threadIndex % 8) * 2;
    if (rowIndex < endRow && channel < p.outputChannels) {
        const int index = int(threadIndex / 8) * 16 + int(threadIndex % 8) * 2;
        const ulong b = ulong(expert) * p.outputChannels + channel;
        output[ulong(order[rowIndex]) * resultChannels + channel / 2] =
            indexed_linear_swiglu(cTile[index] + bias[b], cTile[index + 1] + bias[b + 1]);
    }
#else
    for (uint i = threadIndex; i < 256; i += 128) {
        const int rowIndex = firstRow + int(i / 16);
        const int o = firstOutput + int(i % 16);
        if (rowIndex < endRow && o < p.outputChannels) {
#if USE_SIMDGROUP_MATRIX
            const float value = cTile[i];
#else
            const float value = i < 128 ? accum0 : accum1;
#endif
            output[ulong(order[rowIndex]) * p.outputChannels + o] = ftype(value + bias[ulong(expert) * p.outputChannels + o]);
        }
    }
#endif
}
)metal";
#endif
#endif // MNN_SUPPORT_TRANSFORMER_FUSE
#endif
