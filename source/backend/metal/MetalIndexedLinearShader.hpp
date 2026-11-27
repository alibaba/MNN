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

#ifndef IDX_M
#define IDX_M 16
#endif
#ifndef IDX_N
#define IDX_N 64
#endif
#ifndef IDX_E
#define IDX_E 256
#endif
#ifndef RW_ATOMIC
#define RW_ATOMIC 0
#endif
#ifndef RW_LOGIT
#define RW_LOGIT 0
#endif

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
    int tokens;
    int xC4;
    float routeEps;
    float routeScale;
};

inline float indexed_linear_weight(const device uchar* weights, constant indexed_linear_params& p, int expert,
                                   int output, int input) {
    const device uchar* bank = weights + ulong(expert) * p.weightStride;
    return float(((const device ftype*)bank)[ulong(output) * p.inputChannels + input]);
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
                        w[j] = indexed_linear_weight(weights, p, expert, channel, i + j);
#if FUSE_SILU_MUL
                        upWeight[j] = indexed_linear_weight(weights, p, expert, channel + 1, i + j);
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

#if WEIGHT_BITS == 4 || WEIGHT_BITS == 8
struct indexed_char4x4 {
    char4 rows[4];
};

kernel void indexed_linear_gemv_packed(const device ftype* input [[buffer(0)]],
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
    const int firstResult = (int(group.x) * 2 + int(simdgroup)) * 4;
    const int resultChannels = p.outputChannels / (FUSE_SILU_MUL ? 2 : 1);
    const int resultQuads = (resultChannels + 3) / 4;
    const int inputQuads = (p.inputChannels + 3) / 4;
    const bool alignedInput = (p.inputChannels % 4) == 0;
    if (route >= p.routes || firstResult >= resultChannels) {
        return;
    }

    const int expert = ids[route];
    const bool validExpert = expert >= 0 && expert < p.experts;
    const int sourceRow = p.inputMode == 0 ? route / p.topK : route;
    const device ftype4* row = reinterpret_cast<const device ftype4*>(input + ulong(sourceRow) * p.inputChannels);
    const uint residentExpert = validExpert ? uint(expert) : 0u;
#if WEIGHT_BITS == 4
    const device ushort4* bank = reinterpret_cast<const device ushort4*>(weights) +
                                 residentExpert * uint(p.weightStride / 8);
#if FUSE_SILU_MUL
    const device ushort4* upBank = bank + resultQuads * inputQuads;
#endif
#else
    const device indexed_char4x4* bank = reinterpret_cast<const device indexed_char4x4*>(weights) +
                                         residentExpert * uint(p.weightStride / 16);
#if FUSE_SILU_MUL
    const device indexed_char4x4* upBank = bank + resultQuads * inputQuads;
#endif
#endif
    const int quad = firstResult / 4;
    const uint tileBase = uint(quad) * uint(inputQuads);

    float4 gateSum = 0.0f;
#if FUSE_SILU_MUL
    float4 upSum = 0.0f;
#endif
    if (validExpert) {
        const int quadsPerBlock = p.blockSize / 4;
        int lanesPerBlock = min(32, max(quadsPerBlock / 4, 1));
        const int blocksInFlight = 32 / lanesPerBlock;
        const int laneInBlock = int(lane) % lanesPerBlock;
        const int blockSlot = int(lane) / lanesPerBlock;
        for (int block = blockSlot; block < p.blockCount; block += blocksInFlight) {
            const int blockInputEnd = min((block + 1) * p.blockSize, p.inputChannels);
            const int zEnd = min((block + 1) * quadsPerBlock, (blockInputEnd + 3) / 4);
            float4 gateRaw = 0.0f;
#if FUSE_SILU_MUL
            float4 upRaw = 0.0f;
#endif
            float inputSum = 0.0f;
            for (int z = block * quadsPerBlock + laneInBlock; z < zEnd; z += lanesPerBlock) {
                float4 x;
                if (alignedInput) {
                    x = float4(row[z]);
                } else {
                    const int c = z * 4;
                    x = float4(c < p.inputChannels ? float(input[ulong(sourceRow) * p.inputChannels + c]) : 0.0f,
                               c + 1 < p.inputChannels ? float(input[ulong(sourceRow) * p.inputChannels + c + 1]) : 0.0f,
                               c + 2 < p.inputChannels ? float(input[ulong(sourceRow) * p.inputChannels + c + 2]) : 0.0f,
                               c + 3 < p.inputChannels ? float(input[ulong(sourceRow) * p.inputChannels + c + 3]) : 0.0f);
                }
                inputSum += dot(x, float4(1.0f));
#if WEIGHT_BITS == 4
                const float4 in0 = x.xxxx * (1.0f / 16.0f);
                const float4 in2 = x.zzzz * (1.0f / 4096.0f);
                const float4 in3 = x.wwww * (1.0f / 256.0f);
                const ushort4 gateWeights = bank[tileBase + uint(z)];
                gateRaw += in0 * float4(gateWeights & ushort4(0x00F0u)) +
                           x.yyyy * float4(gateWeights & ushort4(0x000Fu)) +
                           in2 * float4(gateWeights & ushort4(0xF000u)) +
                           in3 * float4(gateWeights & ushort4(0x0F00u));
#if FUSE_SILU_MUL
                const ushort4 upWeights = upBank[tileBase + uint(z)];
                upRaw += in0 * float4(upWeights & ushort4(0x00F0u)) +
                         x.yyyy * float4(upWeights & ushort4(0x000Fu)) +
                         in2 * float4(upWeights & ushort4(0xF000u)) +
                         in3 * float4(upWeights & ushort4(0x0F00u));
#endif
#else
                const indexed_char4x4 gateWeights = bank[tileBase + uint(z)];
                gateRaw += float4(dot(x, float4(gateWeights.rows[0])), dot(x, float4(gateWeights.rows[1])),
                                  dot(x, float4(gateWeights.rows[2])), dot(x, float4(gateWeights.rows[3])));
#if FUSE_SILU_MUL
                const indexed_char4x4 upWeights = upBank[tileBase + uint(z)];
                upRaw += float4(dot(x, float4(upWeights.rows[0])), dot(x, float4(upWeights.rows[1])),
                                dot(x, float4(upWeights.rows[2])), dot(x, float4(upWeights.rows[3])));
#endif
#endif
            }

            float4 scale = 0.0f, offset = 0.0f;
#if FUSE_SILU_MUL
            float4 upScale = 0.0f, upOffset = 0.0f;
#endif
            for (int o = 0; o < 4; ++o) {
                if (firstResult + o >= resultChannels) {
                    break;
                }
#if FUSE_SILU_MUL
                const int channel = (firstResult + o) * 2;
                const ulong gateIndex = (ulong(expert) * p.outputChannels + channel) * p.blockCount + block;
                const ulong upIndex = gateIndex + p.blockCount;
                scale[o] = scales[gateIndex].x;
                offset[o] = scales[gateIndex].y;
                upScale[o] = scales[upIndex].x;
                upOffset[o] = scales[upIndex].y;
#else
                const ulong index = (ulong(expert) * p.outputChannels + firstResult + o) * p.blockCount + block;
                scale[o] = scales[index].x;
                offset[o] = scales[index].y;
#endif
            }
#if WEIGHT_BITS == 4
            offset -= 8.0f * scale;
#if FUSE_SILU_MUL
            upOffset -= 8.0f * upScale;
#endif
#endif
            gateSum += gateRaw * scale + inputSum * offset;
#if FUSE_SILU_MUL
            upSum += upRaw * upScale + inputSum * upOffset;
#endif
        }
    }
    gateSum = float4(simd_sum(gateSum.x), simd_sum(gateSum.y), simd_sum(gateSum.z), simd_sum(gateSum.w));
#if FUSE_SILU_MUL
    upSum = float4(simd_sum(upSum.x), simd_sum(upSum.y), simd_sum(upSum.z), simd_sum(upSum.w));
#endif
    if (lane == 0) {
        for (int o = 0; o < 4; ++o) {
            const int resultChannel = firstResult + o;
            if (resultChannel >= resultChannels) {
                break;
            }
            const int weightChannel =
                FUSE_SILU_MUL ? resultChannel * 2 : resultChannel;
            const ulong biasBase = ulong(expert) * p.outputChannels + weightChannel;
            const float gate = validExpert ? gateSum[o] + bias[biasBase] : 0.0f;
#if FUSE_SILU_MUL
            const float up = validExpert ? upSum[o] + bias[biasBase + 1] : 0.0f;
            output[ulong(route) * resultChannels + resultChannel] = indexed_linear_swiglu(gate, up);
#else
            output[ulong(route) * resultChannels + resultChannel] = ftype(gate);
#endif
        }
    }
}
#endif

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
        tiles += (int(counts[e]) + IDX_M - 1) / IDX_M;
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

kernel void indexed_linear_route(const device int* ids [[buffer(0)]],
                                 device int* ranks [[buffer(1)]],
                                 device int* offsets [[buffer(2)]],
                                 device int* tileOffsets [[buffer(3)]],
                                 device int* order [[buffer(4)]],
                                 device ftype* zeroFill [[buffer(5)]],
                                 constant indexed_linear_params& p [[buffer(6)]],
                                 uint tid [[thread_index_in_threadgroup]]) {
    threadgroup atomic_uint counts[IDX_E];
    for (uint e = tid; e < uint(p.experts); e += 1024) {
        atomic_store_explicit(counts + e, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint r = tid; r < uint(p.routes); r += 1024) {
        const int e = ids[r];
        ranks[r] = e >= 0 && e < p.experts
            ? int(atomic_fetch_add_explicit(counts + e, 1u, memory_order_relaxed)) : -1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        int routes = 0, tiles = 0;
        for (int e = 0; e < p.experts; ++e) {
            offsets[e] = routes;
            tileOffsets[e] = tiles;
            const int c = int(atomic_load_explicit(counts + e, memory_order_relaxed));
            routes += c;
            tiles += (c + IDX_M - 1) / IDX_M;
        }
        offsets[p.experts] = routes;
        tileOffsets[p.experts] = tiles;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
#if !RW_ATOMIC
    const int resultChannels = p.outputChannels / (FUSE_SILU_MUL ? 2 : 1);
#endif
    for (uint r = tid; r < uint(p.routes); r += 1024) {
        const int rank = ranks[r];
        if (rank >= 0) {
            order[offsets[ids[r]] + rank] = int(r);
        }
#if !RW_ATOMIC
        else {
            for (int o = 0; o < resultChannels; ++o) {
                zeroFill[ulong(r) * resultChannels + o] = ftype(0);
            }
        }
#endif
    }
}

#if RW_ATOMIC
kernel void indexed_linear_zero_y(device uint* output [[buffer(0)]],
                                  constant indexed_linear_params& p [[buffer(1)]],
                                  uint index [[thread_position_in_grid]]) {
    const uint words = uint(p.tokens) * uint(p.outputChannels) / 2;
    if (index < words) {
        output[index] = 0;
    }
}

inline void indexed_atomic_add_half2(device atomic_uint* word, half2 add) {
    uint old = atomic_load_explicit(word, memory_order_relaxed);
    for (;;) {
        const uint next = as_type<uint>(as_type<half2>(old) + add);
        if (atomic_compare_exchange_weak_explicit(word, &old, next,
                                                  memory_order_relaxed, memory_order_relaxed)) {
            break;
        }
    }
}
#endif

kernel void indexed_linear_grouped_simd(
    const device ftype* input [[buffer(0)]], device ftype* output [[buffer(1)]],
    constant indexed_linear_params& p [[buffer(2)]], const device uchar* weights [[buffer(3)]],
    const device float2* scales [[buffer(4)]], const device float* bias [[buffer(5)]],
    const device int* offsets [[buffer(6)]], const device int* tileOffsets [[buffer(7)]],
    const device int* order [[buffer(8)]], uint3 group [[threadgroup_position_in_grid]],
    uint threadIndex [[thread_index_in_threadgroup]], uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    const int tile = int(group.y);
    if (tile >= tileOffsets[p.experts])
        return; // Threadgroup-uniform, before barriers.
    int low = 0, high = p.experts;
    while (low + 1 < high) {
        const int mid = (low + high) / 2;
        if (tileOffsets[mid] <= tile)
            low = mid;
        else
            high = mid;
    }
    const int expert = low;
    const int firstRow = offsets[expert] + (tile - tileOffsets[expert]) * 16;
    const int endRow = offsets[expert + 1];
    const int firstOutput = int(group.x) * 64;
    const int resultChannels = p.outputChannels / (FUSE_SILU_MUL ? 2 : 1);
#if WEIGHT_BITS == 4 || WEIGHT_BITS == 8
    const int resultQuads = (resultChannels + 3) / 4;
    const int inputQuads = (p.inputChannels + 3) / 4;
#if WEIGHT_BITS == 4
    const device ushort* packedRows =
        reinterpret_cast<const device ushort*>(weights) + uint(expert) * uint(p.weightStride / 2);
#else
    const device char4* packedRows =
        reinterpret_cast<const device char4*>(weights) + uint(expert) * uint(p.weightStride / 4);
#endif
#endif
    threadgroup ftype aTile[16 * 32];
    threadgroup ftype bTile[32 * 64];
    threadgroup float cTile[16 * 64];
#if USE_SIMDGROUP_MATRIX
    simdgroup_float8x8 accum[4];
    for (int n = 0; n < 4; ++n) {
        accum[n] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }
#else
    float accum[8] = {0.0f};
#endif
    for (int firstInput = 0; firstInput < p.inputChannels; firstInput += 32) {
#if WEIGHT_BITS == 4 || WEIGHT_BITS == 8
        {
            for (uint i = threadIndex; i < 512; i += 128) {
                const int rowIndex = firstRow + int(i / 32);
                const int inputChannel = firstInput + int(i % 32);
                ftype a = ftype(0);
                if (rowIndex < endRow && inputChannel < p.inputChannels) {
                    const int route = order[rowIndex];
                    const int sourceRow = p.inputMode == 0 ? route / p.topK : route;
                    a = p.xC4 ? input[(ulong(inputChannel / 4) * p.tokens + sourceRow) * 4 + (inputChannel & 3)]
                              : input[ulong(sourceRow) * p.inputChannels + inputChannel];
                }
                aTile[i] = a;
            }
            {
                for (uint bi = threadIndex; bi < 64 * 8; bi += 128) {
                    const int bRow = int(bi / 8u);
                    const int inputGroup = int(bi % 8u);
                    const int outputChannel = firstOutput + bRow;
                    const int inputQuad = firstInput / 4 + inputGroup;
                    const int bBase = bRow * 32 + inputGroup * 4;
                    if (outputChannel >= p.outputChannels || inputQuad * 4 >= p.inputChannels) {
                        for (int inner = 0; inner < 4; ++inner) {
                            bTile[bBase + inner] = ftype(0);
                        }
                    } else {
                        int stream = 0;
                        int resultChannel = outputChannel;
#if FUSE_SILU_MUL
                        stream = outputChannel & 1;
                        resultChannel = outputChannel / 2;
#endif
                        const int tileIndex = (stream * resultQuads + resultChannel / 4) * inputQuads + inputQuad;
                        const int row = resultChannel % 4;
                        const int block = (firstInput + inputGroup * 4) / p.blockSize;
                        const float2 dequant =
                            scales[(ulong(expert) * p.outputChannels + outputChannel) * p.blockCount + block];
#if WEIGHT_BITS == 4
                        const ushort rowPacked = packedRows[tileIndex * 4 + row];
                        const uchar low = uchar(rowPacked & 0xFFu), high = uchar(rowPacked >> 8);
                        const uchar nibbles[4] = {uchar(low >> 4), uchar(low & 15), uchar(high >> 4), uchar(high & 15)};
                        bTile[bBase + 0] = ftype(float(int(nibbles[0]) - 8) * dequant.x + dequant.y);
                        bTile[bBase + 1] = ftype(float(int(nibbles[1]) - 8) * dequant.x + dequant.y);
                        bTile[bBase + 2] = ftype(float(int(nibbles[2]) - 8) * dequant.x + dequant.y);
                        bTile[bBase + 3] = ftype(float(int(nibbles[3]) - 8) * dequant.x + dequant.y);
#else
                        const char4 packed = packedRows[tileIndex * 4 + row];
                        bTile[bBase + 0] = ftype(float(packed.x) * dequant.x + dequant.y);
                        bTile[bBase + 1] = ftype(float(packed.y) * dequant.x + dequant.y);
                        bTile[bBase + 2] = ftype(float(packed.z) * dequant.x + dequant.y);
                        bTile[bBase + 3] = ftype(float(packed.w) * dequant.x + dequant.y);
#endif
                    }
                }
            }
        }
#else
        {
            for (uint i = threadIndex; i < 512; i += 128) {
                const int rowIndex = firstRow + int(i / 32);
                const int inputChannel = firstInput + int(i % 32);
                ftype a = ftype(0);
                if (rowIndex < endRow && inputChannel < p.inputChannels) {
                    const int route = order[rowIndex];
                    const int sourceRow = p.inputMode == 0 ? route / p.topK : route;
                    a = p.xC4 ? input[(ulong(inputChannel / 4) * p.tokens + sourceRow) * 4 + (inputChannel & 3)]
                              : input[ulong(sourceRow) * p.inputChannels + inputChannel];
                }
                aTile[i] = a;
            }
            // Stage B transposed: adjacent lanes unpack adjacent input channels
            // instead of striding across full output-channel weight rows.
            for (uint i = threadIndex; i < 64 * 32; i += 128) {
                const int k = firstInput + int(i % 32);
                const int o = firstOutput + int(i / 32);
                bTile[i] = k < p.inputChannels && o < p.outputChannels
                               ? ftype(indexed_linear_weight(weights, p, expert, o, k))
                               : ftype(0);
            }
        }
#endif
        threadgroup_barrier(mem_flags::mem_threadgroup);
#if USE_SIMDGROUP_MATRIX
        for (int k = 0; k < 32; k += 8) {
            simdgroup_matrix<ftype, 8, 8> a, b;
            simdgroup_load(a, aTile + (simdgroup / 2) * 8 * 32 + k, 32);
            for (int n = 0; n < 4; ++n) {
                simdgroup_load(b, bTile + ((simdgroup % 2) * 8 + n * 16) * 32 + k, 32, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(accum[n], a, b, accum[n]);
            }
        }
#else
        const uint col = threadIndex % 64;
        for (int k = 0; k < 32; ++k) {
            const float b = float(bTile[col * 32 + k]);
            for (int i = 0; i < 8; ++i) {
                const uint row = (threadIndex + i * 128) / 64;
                accum[i] += float(aTile[row * 32 + k]) * b;
            }
        }
#endif
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
#if USE_SIMDGROUP_MATRIX
    for (int n = 0; n < 4; ++n) {
        simdgroup_store(accum[n], cTile + (simdgroup / 2) * 8 * 64 + (simdgroup % 2) * 8 + n * 16,
                       64);
    }
#else
    for (int i = 0; i < 8; ++i) {
        cTile[threadIndex + i * 128] = accum[i];
    }
#endif
    threadgroup_barrier(mem_flags::mem_threadgroup);
#if FUSE_SILU_MUL
    for (uint i = threadIndex; i < 16 * 32; i += 128) {
        const int rowIndex = firstRow + int(i / 32);
        const int channel = firstOutput + int(i % 32) * 2;
        if (rowIndex < endRow && channel < p.outputChannels) {
            const int index = int(i) * 2;
            const ulong b = ulong(expert) * p.outputChannels + channel;
            output[ulong(order[rowIndex]) * resultChannels + channel / 2] =
                indexed_linear_swiglu(cTile[index] + bias[b], cTile[index + 1] + bias[b + 1]);
        }
    }
#else
    for (uint i = threadIndex; i < 16 * 64; i += 128) {
        const int rowIndex = firstRow + int(i / 64);
        const int o = firstOutput + int(i % 64);
        if (rowIndex < endRow && o < p.outputChannels) {
            const float value = cTile[i];
            output[ulong(order[rowIndex]) * p.outputChannels + o] =
                ftype(value + bias[ulong(expert) * p.outputChannels + o]);
        }
    }
#endif
}

kernel void indexed_linear_gemv_rw(const device ftype* input [[buffer(0)]],
                                   const device int* ids [[buffer(1)]],
                                   device ftype* output [[buffer(2)]],
                                   constant indexed_linear_params& p [[buffer(3)]],
                                   const device uchar* weights [[buffer(4)]],
                                   const device float2* scales [[buffer(5)]],
                                   const device float* bias [[buffer(6)]],
                                   const device ftype* routeWeights [[buffer(7)]],
                                   uint3 group [[threadgroup_position_in_grid]],
                                   uint lane [[thread_index_in_simdgroup]],
                                   uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    const int token = int(group.y);
    const int firstResult = (int(group.x) * 2 + int(simdgroup)) * 4;
    const int resultChannels = p.outputChannels;
    const int resultQuads = (resultChannels + 3) / 4;
    const int inputQuads = (p.inputChannels + 3) / 4;
    const bool alignedInput = (p.inputChannels % 4) == 0;
    if (token >= p.tokens || firstResult >= resultChannels) {
        return;
    }
    const int quad = firstResult / 4;
    const uint tileBase = uint(quad) * uint(inputQuads);
    float4 weightedExpertSum = 0.0f;
#if RW_RAW
    const device ftype* rwRow = routeWeights + ulong(token) * p.experts;
    float rsum = p.routeEps;
#if RW_LOGIT
    float rmax = -FLT_MAX;
    for (int k = 0; k < p.topK; ++k) {
        const int expertid = ids[token * p.topK + k];
        if (expertid >= 0 && expertid < p.experts) {
            rmax = max(rmax, float(rwRow[expertid]));
        }
    }
    for (int k = 0; k < p.topK; ++k) {
        const int expertid = ids[token * p.topK + k];
        if (expertid >= 0 && expertid < p.experts) {
            rsum += exp(float(rwRow[expertid]) - rmax);
        }
    }
#else
    for (int k = 0; k < p.topK; ++k) {
        const int expertid = ids[token * p.topK + k];
        if (expertid >= 0 && expertid < p.experts) {
            rsum += float(rwRow[expertid]);
        }
    }
#endif
#endif
    for (int k = 0; k < p.topK; ++k) {
        const int route = token * p.topK + k;
        const int expertid = ids[route];
        if (expertid < 0 || expertid >= p.experts) {
            continue;
        }
#if RW_LOGIT
        const float expertRoutingWeight = exp(float(rwRow[expertid]) - rmax) / rsum * p.routeScale;
#elif RW_RAW
        const float expertRoutingWeight = float(rwRow[expertid]) / rsum * p.routeScale;
#else
        const float expertRoutingWeight = float(routeWeights[route]);
#endif
        const device ftype* rowRaw = input + ulong(route) * p.inputChannels;
        float4 projectionSum = 0.0f;
#if WEIGHT_BITS == 4 || WEIGHT_BITS == 8
        const device ftype4* row = reinterpret_cast<const device ftype4*>(rowRaw);
#if WEIGHT_BITS == 4
        const device ushort4* bank = reinterpret_cast<const device ushort4*>(weights) +
                                     uint(expertid) * uint(p.weightStride / 8);
#else
        const device indexed_char4x4* bank = reinterpret_cast<const device indexed_char4x4*>(weights) +
                                             uint(expertid) * uint(p.weightStride / 16);
#endif
        const int quadsPerBlock = p.blockSize / 4;
        int lanesPerBlock = min(32, max(quadsPerBlock / 4, 1));
        const int blocksInFlight = 32 / lanesPerBlock;
        const int laneInBlock = int(lane) % lanesPerBlock;
        const int blockSlot = int(lane) / lanesPerBlock;
        for (int block = blockSlot; block < p.blockCount; block += blocksInFlight) {
            const int blockInputEnd = min((block + 1) * p.blockSize, p.inputChannels);
            const int zEnd = min((block + 1) * quadsPerBlock, (blockInputEnd + 3) / 4);
            float4 gateRaw = 0.0f;
            float inputSum = 0.0f;
            for (int z = block * quadsPerBlock + laneInBlock; z < zEnd; z += lanesPerBlock) {
                float4 x;
                if (alignedInput) {
                    x = float4(row[z]);
                } else {
                    const int c = z * 4;
                    x = float4(c < p.inputChannels ? float(rowRaw[c]) : 0.0f,
                               c + 1 < p.inputChannels ? float(rowRaw[c + 1]) : 0.0f,
                               c + 2 < p.inputChannels ? float(rowRaw[c + 2]) : 0.0f,
                               c + 3 < p.inputChannels ? float(rowRaw[c + 3]) : 0.0f);
                }
                inputSum += dot(x, float4(1.0f));
#if WEIGHT_BITS == 4
                const float4 in0 = x.xxxx * (1.0f / 16.0f);
                const float4 in2 = x.zzzz * (1.0f / 4096.0f);
                const float4 in3 = x.wwww * (1.0f / 256.0f);
                const ushort4 gateWeights = bank[tileBase + uint(z)];
                gateRaw += in0 * float4(gateWeights & ushort4(0x00F0u)) +
                           x.yyyy * float4(gateWeights & ushort4(0x000Fu)) +
                           in2 * float4(gateWeights & ushort4(0xF000u)) +
                           in3 * float4(gateWeights & ushort4(0x0F00u));
#else
                const indexed_char4x4 gateWeights = bank[tileBase + uint(z)];
                gateRaw += float4(dot(x, float4(gateWeights.rows[0])), dot(x, float4(gateWeights.rows[1])),
                                  dot(x, float4(gateWeights.rows[2])), dot(x, float4(gateWeights.rows[3])));
#endif
            }
            float4 scale = 0.0f, offset = 0.0f;
            for (int o = 0; o < 4; ++o) {
                const ulong index = (ulong(expertid) * p.outputChannels + firstResult + o) * p.blockCount + block;
                scale[o] = scales[index].x;
                offset[o] = scales[index].y;
            }
#if WEIGHT_BITS == 4
            offset -= 8.0f * scale;
#endif
            projectionSum += gateRaw * scale + inputSum * offset;
        }
#else
        for (int i = int(lane) * 4; i < p.inputChannels; i += 128) {
            float4 x = 0.0f;
            for (int j = 0; j < 4; ++j) {
                if (i + j < p.inputChannels) x[j] = float(rowRaw[i + j]);
            }
            for (int o = 0; o < 4; ++o) {
                if (firstResult + o >= resultChannels) break;
                float4 w = 0.0f;
                for (int j = 0; j < 4; ++j) {
                    if (i + j < p.inputChannels) {
                        w[j] = indexed_linear_weight(weights, p, expertid, firstResult + o, i + j);
                    }
                }
                projectionSum[o] += dot(x, w);
            }
        }
#endif
        projectionSum = float4(simd_sum(projectionSum.x), simd_sum(projectionSum.y),
                               simd_sum(projectionSum.z), simd_sum(projectionSum.w));
        if (lane == 0) {
            for (int o = 0; o < 4; ++o) {
                const int c = firstResult + o;
                if (c < resultChannels) {
                    weightedExpertSum[o] +=
                        expertRoutingWeight * (projectionSum[o] + bias[ulong(expertid) * p.outputChannels + c]);
                }
            }
        }
    }
    if (lane == 0) {
        for (int o = 0; o < 4; ++o) {
            const int c = firstResult + o;
            if (c < resultChannels) {
                output[ulong(token) * resultChannels + c] = ftype(weightedExpertSum[o]);
            }
        }
    }
}

kernel void indexed_linear_rw_reduce(const device ftype* tmp [[buffer(0)]],
                                     const device ftype* routeWeights [[buffer(1)]],
                                     device ftype* output [[buffer(2)]],
                                     constant indexed_linear_params& p [[buffer(3)]],
                                     const device int* ids [[buffer(4)]],
                                     uint2 gid [[thread_position_in_grid]]) {
    const int c = int(gid.x);
    const int token = int(gid.y);
    if (c >= p.outputChannels || token >= p.tokens) {
        return;
    }
#if RW_RAW
    const device ftype* rwRow = routeWeights + ulong(token) * p.experts;
    float rsum = p.routeEps;
#if RW_LOGIT
    float rmax = -FLT_MAX;
    for (int k = 0; k < p.topK; ++k) {
        const int e = ids[token * p.topK + k];
        if (e >= 0 && e < p.experts) {
            rmax = max(rmax, float(rwRow[e]));
        }
    }
    for (int k = 0; k < p.topK; ++k) {
        const int e = ids[token * p.topK + k];
        if (e >= 0 && e < p.experts) {
            rsum += exp(float(rwRow[e]) - rmax);
        }
    }
#else
    for (int k = 0; k < p.topK; ++k) {
        const int e = ids[token * p.topK + k];
        if (e >= 0 && e < p.experts) {
            rsum += float(rwRow[e]);
        }
    }
#endif
    const float rscale = p.routeScale;
#endif
    float acc = 0.0f;
    for (int k = 0; k < p.topK; ++k) {
#if RW_RAW
        const int e = ids[token * p.topK + k];
        if (e < 0 || e >= p.experts) {
            continue;
        }
#if RW_LOGIT
        const float w = exp(float(rwRow[e]) - rmax) / rsum * rscale;
#else
        const float w = float(rwRow[e]) / rsum * rscale;
#endif
        acc += w * float(tmp[ulong(token * p.topK + k) * p.outputChannels + c]);
#else
        acc += float(routeWeights[token * p.topK + k]) * float(tmp[ulong(token * p.topK + k) * p.outputChannels + c]);
#endif
    }
    output[ulong(c / 4 * p.tokens + token) * 4 + (c & 3)] = ftype(acc);
}

kernel void indexed_linear_permute_a(
    const device ftype* input [[buffer(0)]], const device int* order [[buffer(1)]],
    device ftype* out [[buffer(2)]], constant indexed_linear_params& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    const int row = int(gid);
    if (row >= p.routes) {
        return;
    }
    const int route = order[row];
    const int src = p.inputMode == 0 ? route / p.topK : route;
    const int quads = p.inputChannels / 4;
    device ftype4* out4 = reinterpret_cast<device ftype4*>(out + ulong(row) * p.inputChannels);
    if (p.xC4) {
        const device ftype4* in4 = reinterpret_cast<const device ftype4*>(input);
        for (int q = 0; q < quads; ++q) {
            out4[q] = in4[ulong(q) * p.tokens + src];
        }
    } else {
        const device ftype4* in4 = reinterpret_cast<const device ftype4*>(input + ulong(src) * p.inputChannels);
        for (int q = 0; q < quads; ++q) {
            out4[q] = in4[q];
        }
    }
}
)metal";

static const char* gIndexedLinearTensor = R"metal(
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

#ifndef IDX_VEC
#define IDX_VEC 1
#endif
#ifndef IDX_ADEV
#define IDX_ADEV 0
#endif
#ifndef IDX_K
#if IDX_VEC
#define IDX_K 64
#else
#define IDX_K 32
#endif
#endif

#define INDEXED_GROUP_N IDX_N
kernel void indexed_linear_grouped_tensor(
    const device ftype* input [[buffer(0)]], device ftype* output [[buffer(1)]],
    constant indexed_linear_params& p [[buffer(2)]], const device uchar* weights [[buffer(3)]],
    const device float2* scales [[buffer(4)]], const device float* bias [[buffer(5)]],
    const device int* offsets [[buffer(6)]], const device int* tileOffsets [[buffer(7)]],
    const device int* order [[buffer(8)]],
#if RW_ATOMIC
    const device ftype* routeWeights [[buffer(9)]], const device int* ids [[buffer(10)]],
#endif
    uint3 group [[threadgroup_position_in_grid]],
    uint threadIndex [[thread_index_in_threadgroup]], uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    const int tile = int(group.y);
    if (tile >= tileOffsets[p.experts])
        return;
    int low = 0, high = p.experts;
    while (low + 1 < high) {
        const int mid = (low + high) / 2;
        if (tileOffsets[mid] <= tile)
            low = mid;
        else
            high = mid;
    }
    const int expert = low;
    const int firstRow = offsets[expert] + (tile - tileOffsets[expert]) * IDX_M;
    const int endRow = offsets[expert + 1];
    const int firstOutput = int(group.x) * INDEXED_GROUP_N;
    const int resultChannels = p.outputChannels / (FUSE_SILU_MUL ? 2 : 1);
#if WEIGHT_BITS == 4 || WEIGHT_BITS == 8
    const int resultQuads = (resultChannels + 3) / 4;
    const int inputQuads = (p.inputChannels + 3) / 4;
#if WEIGHT_BITS == 4
    const device ushort* packedRows =
        reinterpret_cast<const device ushort*>(weights) + uint(expert) * uint(p.weightStride / 2);
#else
    const device char4* packedRows =
        reinterpret_cast<const device char4*>(weights) + uint(expert) * uint(p.weightStride / 4);
#endif
#endif
#if IDX_VEC && (WEIGHT_BITS == 4 || WEIGHT_BITS == 8)
#define IDX_SCALE_BYTES ((IDX_K / 16) * INDEXED_GROUP_N * 8)
#else
#define IDX_SCALE_BYTES 0
#endif
#if IDX_ADEV
#define IDX_A_BYTES 0
#else
#define IDX_A_BYTES (IDX_M * IDX_K * sizeof(ftype))
#endif
#define IDX_STAGE_BYTES (IDX_A_BYTES + IDX_K * INDEXED_GROUP_N * sizeof(ftype) + IDX_SCALE_BYTES)
#define IDX_C_BYTES (IDX_M * INDEXED_GROUP_N * sizeof(float))
#define IDX_RW_BYTES (RW_ATOMIC ? IDX_M * 4 : 0)
#define IDX_SMEM_BYTES ((IDX_STAGE_BYTES > IDX_C_BYTES ? IDX_STAGE_BYTES : IDX_C_BYTES) + IDX_RW_BYTES)
    threadgroup uchar smemRaw[IDX_SMEM_BYTES];
    threadgroup ftype* aTile = (threadgroup ftype*)smemRaw;
    threadgroup ftype* bTile = (threadgroup ftype*)(smemRaw + IDX_A_BYTES);
#if IDX_VEC && (WEIGHT_BITS == 4 || WEIGHT_BITS == 8)
    threadgroup float2* scaleTile =
        (threadgroup float2*)(smemRaw + IDX_A_BYTES + IDX_K * INDEXED_GROUP_N * sizeof(ftype));
#endif
    threadgroup float* cTile = (threadgroup float*)smemRaw;
#if RW_ATOMIC
    threadgroup float* wTile = (threadgroup float*)(smemRaw + IDX_SMEM_BYTES - IDX_RW_BYTES);
#endif
    auto ta = tensor<threadgroup ftype, dextents<int32_t, 2>, tensor_inline>(aTile, dextents<int32_t, 2>(IDX_K, IDX_M));
    auto tb = tensor<threadgroup ftype, dextents<int32_t, 2>, tensor_inline>(bTile,
        dextents<int32_t, 2>(IDX_K, INDEXED_GROUP_N));
    mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(
                                  IDX_M, INDEXED_GROUP_N, IDX_K, false, true, false,
                                  mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
                              execution_simdgroups<4>>
        mm;
#if IDX_ADEV
    const array<int, 2> aStrides = {1, p.inputChannels};
    auto t_a = tensor<device ftype, dextents<int32_t, 2>, tensor_inline>(
        (device ftype*)input, dextents<int32_t, 2>(p.inputChannels, p.routes), aStrides);
    auto sa_ref = t_a.slice(0, 0);
    auto sb_ref = tb.slice(0, 0);
    auto accum = mm.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(sa_ref)>,
                                                       metal::remove_addrspace_t<decltype(sb_ref)>, float>();
#else
    auto accum = mm.get_destination_cooperative_tensor<decltype(ta), decltype(tb), float>();
#endif
#if IDX_VEC && (WEIGHT_BITS == 4 || WEIGHT_BITS == 8) && !IDX_ADEV
    const int threadsPerRow = 128 / IDX_M;
    const int quadsPerThread = (IDX_K / 4) / threadsPerRow;
    const int aRow = int(threadIndex) / threadsPerRow;
    const int aQuad0 = int(threadIndex % uint(threadsPerRow)) * quadsPerThread;
    const int aRowIndex = firstRow + aRow;
    const bool aRowValid = aRowIndex < endRow;
    const int aRoute = aRowValid ? order[aRowIndex] : 0;
    const int aSource = p.inputMode == 0 ? aRoute / p.topK : aRoute;
#endif
    for (int firstInput = 0; firstInput < p.inputChannels; firstInput += IDX_K) {
#if WEIGHT_BITS == 4 || WEIGHT_BITS == 8
#if IDX_VEC
        {
#if !IDX_ADEV
            threadgroup ftype4* aTile4 = reinterpret_cast<threadgroup ftype4*>(aTile);
            for (int j = 0; j < quadsPerThread; ++j) {
                const int quad = aQuad0 + j;
                ftype4 a = ftype4(0);
                if (aRowValid && firstInput + quad * 4 < p.inputChannels) {
                    a = p.xC4
                            ? reinterpret_cast<const device ftype4*>(input)[ulong(firstInput / 4 + quad) * p.tokens + aSource]
                            : reinterpret_cast<const device ftype4*>(input + ulong(aSource) * p.inputChannels)[firstInput / 4 + quad];
                }
                aTile4[aRow * (IDX_K / 4) + quad] = a;
            }
#endif
            const int subBlocks = p.blockSize >= IDX_K ? 1 : IDX_K / p.blockSize;
            for (uint s = threadIndex; s < uint(subBlocks * INDEXED_GROUP_N); s += 128) {
                const int sub = int(s / INDEXED_GROUP_N);
                const int oc = firstOutput + int(s % INDEXED_GROUP_N);
                const int block = firstInput / p.blockSize + sub;
                scaleTile[sub * INDEXED_GROUP_N + s % INDEXED_GROUP_N] =
                    oc < p.outputChannels && block < p.blockCount
                        ? scales[(ulong(expert) * p.outputChannels + oc) * p.blockCount + block]
                        : float2(0, 0);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const int igCount = IDX_K / 4;
            const int quadCount = INDEXED_GROUP_N / 4;
            const int quadsPerStream = FUSE_SILU_MUL ? INDEXED_GROUP_N / 8 : quadCount;
            threadgroup ftype4* bTile4 = reinterpret_cast<threadgroup ftype4*>(bTile);
            for (uint item = threadIndex; item < uint(quadCount * igCount); item += 128) {
                const int ig = int(item % uint(igCount));
                const int quadTotal = int(item / uint(igCount));
                const int inputQuad = firstInput / 4 + ig;
                const int stream = quadTotal / quadsPerStream;
                const int quad = quadTotal % quadsPerStream;
                const int bRowBase = quad * (FUSE_SILU_MUL ? 8 : 4) + stream;
                const int resultQuad = (FUSE_SILU_MUL ? firstOutput / 8 : firstOutput / 4) + quad;
                const bool tileValid = inputQuad * 4 < p.inputChannels && resultQuad < resultQuads;
                const int tileIndex = (stream * resultQuads + resultQuad) * inputQuads + inputQuad;
#if WEIGHT_BITS == 4
                const ushort4 packed =
                    tileValid ? reinterpret_cast<const device ushort4*>(packedRows)[tileIndex] : ushort4(0);
#else
                const int4 packedRaw =
                    tileValid ? reinterpret_cast<const device int4*>(packedRows)[tileIndex] : int4(0);
#endif
                const int sub = p.blockSize >= IDX_K ? 0 : (ig * 4) / p.blockSize;
                for (int row = 0; row < 4; ++row) {
                    const int bRow = bRowBase + row * (FUSE_SILU_MUL ? 2 : 1);
                    ftype4 v = ftype4(0);
                    if (tileValid) {
                        const float2 dequant = scaleTile[sub * INDEXED_GROUP_N + bRow];
#if WEIGHT_BITS == 4
                        const ushort rowPacked = packed[row];
                        const uint low = uint(rowPacked) & 0xFFu, high = uint(rowPacked) >> 8;
                        v.x = ftype((float(int(low >> 4)) - 8.0f) * dequant.x + dequant.y);
                        v.y = ftype((float(int(low & 15u)) - 8.0f) * dequant.x + dequant.y);
                        v.z = ftype((float(int(high >> 4)) - 8.0f) * dequant.x + dequant.y);
                        v.w = ftype((float(int(high & 15u)) - 8.0f) * dequant.x + dequant.y);
#else
                        const char4 cv = as_type<char4>(packedRaw[row]);
                        v.x = ftype(float(cv.x) * dequant.x + dequant.y);
                        v.y = ftype(float(cv.y) * dequant.x + dequant.y);
                        v.z = ftype(float(cv.z) * dequant.x + dequant.y);
                        v.w = ftype(float(cv.w) * dequant.x + dequant.y);
#endif
                    }
                    bTile4[bRow * igCount + ig] = v;
                }
            }
        }
#else
        {
            for (uint i = threadIndex; i < uint(IDX_M) * IDX_K; i += 128) {
                const int rowIndex = firstRow + int(i / IDX_K);
                const int inputChannel = firstInput + int(i % IDX_K);
                ftype a = ftype(0);
                if (rowIndex < endRow && inputChannel < p.inputChannels) {
                    const int route = order[rowIndex];
                    const int sourceRow = p.inputMode == 0 ? route / p.topK : route;
                    a = p.xC4 ? input[(ulong(inputChannel / 4) * p.tokens + sourceRow) * 4 + (inputChannel & 3)]
                              : input[ulong(sourceRow) * p.inputChannels + inputChannel];
                }
                aTile[i] = a;
            }
            {
                for (uint bi = threadIndex; bi < INDEXED_GROUP_N * (IDX_K / 4); bi += 128) {
                    const int bRow = int(bi / (IDX_K / 4));
                    const int inputGroup = int(bi % (IDX_K / 4));
                    const int outputChannel = firstOutput + bRow;
                    const int inputQuad = firstInput / 4 + inputGroup;
                    const int bBase = bRow * IDX_K + inputGroup * 4;
                    if (outputChannel >= p.outputChannels || inputQuad * 4 >= p.inputChannels) {
                        for (int inner = 0; inner < 4; ++inner) {
                            bTile[bBase + inner] = ftype(0);
                        }
                    } else {
                        int stream = 0;
                        int resultChannel = outputChannel;
#if FUSE_SILU_MUL
                        stream = outputChannel & 1;
                        resultChannel = outputChannel / 2;
#endif
                        const int tileIndex = (stream * resultQuads + resultChannel / 4) * inputQuads + inputQuad;
                        const int row = resultChannel % 4;
                        const int block = (firstInput + inputGroup * 4) / p.blockSize;
                        const float2 dequant =
                            scales[(ulong(expert) * p.outputChannels + outputChannel) * p.blockCount + block];
#if WEIGHT_BITS == 4
                        const ushort rowPacked = packedRows[tileIndex * 4 + row];
                        const uchar low = uchar(rowPacked & 0xFFu), high = uchar(rowPacked >> 8);
                        const uchar nibbles[4] = {uchar(low >> 4), uchar(low & 15), uchar(high >> 4), uchar(high & 15)};
                        bTile[bBase + 0] = ftype(float(int(nibbles[0]) - 8) * dequant.x + dequant.y);
                        bTile[bBase + 1] = ftype(float(int(nibbles[1]) - 8) * dequant.x + dequant.y);
                        bTile[bBase + 2] = ftype(float(int(nibbles[2]) - 8) * dequant.x + dequant.y);
                        bTile[bBase + 3] = ftype(float(int(nibbles[3]) - 8) * dequant.x + dequant.y);
#else
                        const char4 packed = packedRows[tileIndex * 4 + row];
                        bTile[bBase + 0] = ftype(float(packed.x) * dequant.x + dequant.y);
                        bTile[bBase + 1] = ftype(float(packed.y) * dequant.x + dequant.y);
                        bTile[bBase + 2] = ftype(float(packed.z) * dequant.x + dequant.y);
                        bTile[bBase + 3] = ftype(float(packed.w) * dequant.x + dequant.y);
#endif
                    }
                }
            }
        }
#endif
#else
        {
            for (uint i = threadIndex; i < uint(IDX_M) * IDX_K; i += 128) {
                const int rowIndex = firstRow + int(i / IDX_K);
                const int inputChannel = firstInput + int(i % IDX_K);
                ftype a = ftype(0);
                if (rowIndex < endRow && inputChannel < p.inputChannels) {
                    const int route = order[rowIndex];
                    const int sourceRow = p.inputMode == 0 ? route / p.topK : route;
                    a = p.xC4 ? input[(ulong(inputChannel / 4) * p.tokens + sourceRow) * 4 + (inputChannel & 3)]
                              : input[ulong(sourceRow) * p.inputChannels + inputChannel];
                }
                aTile[i] = a;
            }
            for (uint i = threadIndex; i < INDEXED_GROUP_N * IDX_K; i += 128) {
                const int k = firstInput + int(i % IDX_K);
                const int o = firstOutput + int(i / IDX_K);
                bTile[i] = k < p.inputChannels && o < p.outputChannels
                               ? ftype(indexed_linear_weight(weights, p, expert, o, k))
                               : ftype(0);
            }
        }
#endif
        threadgroup_barrier(mem_flags::mem_threadgroup);
#if IDX_ADEV
        auto sa = t_a.slice(firstInput, firstRow);
#else
        auto sa = ta.slice(0, 0);
#endif
        auto sb = tb.slice(0, 0);
        mm.run(sa, sb, accum);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto tc = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(cTile,
        dextents<int32_t, 2>(INDEXED_GROUP_N, IDX_M));
    accum.store(tc);
    threadgroup_barrier(mem_flags::mem_threadgroup);
#if FUSE_SILU_MUL
    for (uint i = threadIndex; i < uint(IDX_M) * (INDEXED_GROUP_N / 2); i += 128) {
        const int rowIndex = firstRow + int(i / (INDEXED_GROUP_N / 2));
        const int channel = firstOutput + int(i % (INDEXED_GROUP_N / 2)) * 2;
        if (rowIndex < endRow && channel < p.outputChannels) {
            const int index = int(i) * 2;
            const ulong b = ulong(expert) * p.outputChannels + channel;
            output[ulong(order[rowIndex]) * resultChannels + channel / 2] =
                indexed_linear_swiglu(cTile[index] + bias[b], cTile[index + 1] + bias[b + 1]);
        }
    }
#else
#if RW_ATOMIC
    for (uint r = threadIndex; r < uint(IDX_M); r += 128) {
        const int rowIndex = firstRow + int(r);
        float w = 0.0f;
        if (rowIndex < endRow) {
            const int route = order[rowIndex];
            const int token = route / p.topK;
#if RW_RAW
            const device ftype* rwRow = routeWeights + ulong(token) * p.experts;
            float rsum = p.routeEps;
#if RW_LOGIT
            float rmax = -FLT_MAX;
            for (int k = 0; k < p.topK; ++k) {
                const int e = ids[token * p.topK + k];
                if (e >= 0 && e < p.experts) {
                    rmax = max(rmax, float(rwRow[e]));
                }
            }
            for (int k = 0; k < p.topK; ++k) {
                const int e = ids[token * p.topK + k];
                if (e >= 0 && e < p.experts) {
                    rsum += exp(float(rwRow[e]) - rmax);
                }
            }
            w = exp(float(rwRow[expert]) - rmax) / rsum * p.routeScale;
#else
            for (int k = 0; k < p.topK; ++k) {
                const int e = ids[token * p.topK + k];
                if (e >= 0 && e < p.experts) {
                    rsum += float(rwRow[e]);
                }
            }
            w = float(rwRow[expert]) / rsum * p.routeScale;
#endif
#else
            w = float(routeWeights[route]);
#endif
        }
        wTile[r] = w;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = threadIndex; i < uint(IDX_M) * (INDEXED_GROUP_N / 2); i += 128) {
        const int rowLocal = int(i / (INDEXED_GROUP_N / 2));
        const int rowIndex = firstRow + rowLocal;
        const int o = firstOutput + int(i % (INDEXED_GROUP_N / 2)) * 2;
        if (rowIndex >= endRow || o >= p.outputChannels) {
            continue;
        }
        const int route = order[rowIndex];
        const int token = route / p.topK;
        const float w = wTile[rowLocal];
        const int index = int(i) * 2;
        const ulong biasBase = ulong(expert) * p.outputChannels + o;
        const half2 add = half2(half(w * (cTile[index] + bias[biasBase])),
                                half(w * (cTile[index + 1] + bias[biasBase + 1])));
        const ulong word = (ulong(o / 4) * p.tokens + token) * 2 + (o & 3) / 2;
        indexed_atomic_add_half2(reinterpret_cast<device atomic_uint*>(output) + word, add);
    }
#else
    for (uint i = threadIndex; i < uint(IDX_M) * INDEXED_GROUP_N; i += 128) {
        const int rowIndex = firstRow + int(i / INDEXED_GROUP_N);
        const int o = firstOutput + int(i % INDEXED_GROUP_N);
        if (rowIndex < endRow && o < p.outputChannels) {
            const float value = cTile[i];
            output[ulong(order[rowIndex]) * p.outputChannels + o] =
                ftype(value + bias[ulong(expert) * p.outputChannels + o]);
        }
    }
#endif
#endif
}

)metal";

#endif
#endif // MNN_SUPPORT_TRANSFORMER_FUSE
#endif
