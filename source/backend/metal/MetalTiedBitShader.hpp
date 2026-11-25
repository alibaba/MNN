// Copyright (c) Alibaba, Inc. and its affiliates.
#ifndef MetalTiedBitShader_hpp
#define MetalTiedBitShader_hpp
#if MNN_METAL_ENABLED

// TiedBit: Q1 scan + global inclusive Top-M + original Q4 code reconstruction.
// Compile after gBasicConvPrefix, with FLOAT32 computation and the backend storage type.
static const char* gMetalTiedBitShader = R"metal(
#ifndef TIED_BIT_BLOCK_QUADS
#error TIED_BIT_BLOCK_QUADS must match the original quantization metadata
#endif

#if !defined(TIED_BIT_SIMD_WIDTH) || !defined(TIED_BIT_SCAN_SIMD_GROUPS) || !defined(TIED_BIT_RERANK_SIMD_GROUPS)
#error TiedBit SIMD geometry must match the host dispatch configuration
#endif

// Threadgroup geometry is supplied by MetalTiedBit.mm, not model dimensions.
constant constexpr int kSimdWidth = TIED_BIT_SIMD_WIDTH;
constant constexpr int kScanSimdGroups = TIED_BIT_SCAN_SIMD_GROUPS;
constant constexpr int kRerankSimdGroups = TIED_BIT_RERANK_SIMD_GROUPS;
#if TIED_BIT_SPLIT_K
constant constexpr int kRerankRowsPerGroup = kRerankSimdGroups / 2;
#else
constant constexpr int kRerankRowsPerGroup = kRerankSimdGroups;
#endif
constant constexpr int kHistogramThreads = 256; // One thread per byte-sized radix bucket.

// Four words in the selector's threshold buffer.
constant constexpr int kThresholdPrefix = 0;
constant constexpr int kThresholdAbovePrefix = 1;
constant constexpr int kThresholdKey = 2;
constant constexpr int kThresholdSelected = 3;

// High nibble first in each byte; C4 words use shifts {4, 0, 12, 8}.
constexpr int packedCodeShift(int channel) {
    return (channel / 2) * 8 + (1 - channel % 2) * 4;
}

constexpr int floorPowerOfTwo(int value) {
    return value < 2 ? 1 : 2 * floorPowerOfTwo(value / 2);
}

// Divide the SIMD lanes between input chunks and quantization groups.
constexpr int inputLanes(int quads, int chunkQuads) {
    return quads / chunkQuads >= kSimdWidth ? kSimdWidth : floorPowerOfTwo(quads / chunkQuads);
}

inline FLOAT4 tied_bit_mask_input(FLOAT value, ushort4 code, int inputLane) {
    const ushort mask = ushort(1u << (3 - inputLane));
    return select(FLOAT4(0), FLOAT4(value), (code & ushort4(mask)) != ushort4(0));
}

inline FLOAT4 tied_bit_dot4(FLOAT4 inputValue, ushort packedWeight) {
    const ushort4 shifts(packedCodeShift(0), packedCodeShift(1), packedCodeShift(2), packedCodeShift(3));
    const ushort4 code = (ushort4(packedWeight) >> shifts) & ushort4(0xf);
    return tied_bit_mask_input(inputValue[0], code, 0) + tied_bit_mask_input(inputValue[1], code, 1) +
           tied_bit_mask_input(inputValue[2], code, 2) + tied_bit_mask_input(inputValue[3], code, 3);
}

// One-bit lm_head proxy. Four Q1 codes (one per output lane, each covering
// four input channels) occupy two bytes per input C4 slice. All proxy logits
// are preserved for the global inclusive Top-M selector.
#ifdef TIED_BIT_C4_SCAN
kernel void tied_bit_prepare_c4_sg(const device ftype4 *in [[buffer(0)]],
                                    device float *inputSums [[buffer(1)]],
                                    device atomic_uint *histogram [[buffer(2)]],
                                    device uint *threshold [[buffer(3)]],
                                    device ftype *out [[buffer(4)]],
                                    constant conv1x1_constants& cst [[buffer(5)]],
                                    uint tid [[thread_index_in_threadgroup]]) {
    // One sum per scan input chunk; loop for dimensions wider than one TG.
    for (uint i = tid; i < uint(cst.inputDepthQuad / 4); i += kHistogramThreads) {
        const uint z = i * 4;
        const FLOAT4 a = FLOAT4(in[z]), b = FLOAT4(in[z + 1]);
        const FLOAT4 c = FLOAT4(in[z + 2]), d = FLOAT4(in[z + 3]);
        inputSums[i] = a[0] + a[1] + a[2] + a[3] + b[0] + b[1] + b[2] + b[3] +
                         c[0] + c[1] + c[2] + c[3] + d[0] + d[1] + d[2] + d[3];
    }
    atomic_store_explicit(histogram + tid, 0u, memory_order_relaxed);
    if (tid < 4) {
        threshold[tid] = 0u;
    }
    // The scan overwrites this sentinel. It also orders preparation against a
    // previous use of the same output buffer without another dispatch.
    if (tid == 0u) {
        out[0] = ftype(0);
    }
}
kernel void tied_bit_c4_sg(const device ftype4 *in                       [[buffer(0)]],
#else
kernel void tied_bit_g16(const device ftype4 *in                         [[buffer(0)]],
#endif
                          device ftype4 *out                              [[buffer(1)]],
                          constant conv1x1_constants& cst                 [[buffer(2)]],
                          const device uchar *wt                          [[buffer(3)]],
                          const device ftype4 *biasTerms                  [[buffer(4)]],
                          const device ftype4 *dequantScale               [[buffer(5)]],
#ifdef TIED_BIT_C4_SCAN
                          const device float *inputSums                   [[buffer(6)]],
#endif
                          uint3 gid [[threadgroup_position_in_grid]],
                          uint tiisg [[thread_index_in_simdgroup]],
                          uint sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr int blockSlices = TIED_BIT_BLOCK_QUADS;
    constexpr int middleStep = TIED_BIT_SCAN_LANES_PER_BLOCK;
    constexpr int outerStep = kSimdWidth / middleStep;
    const int uz = 2 * (int(gid.x) * kScanSimdGroups + int(sgitg));
    if (uz >= cst.outputDepthQuad) {
        return;
    }
    const int areaSize = cst.outputSize * cst.batch;
    const int rx = int(gid.y);
    // Q1 retains C4 tiles for the scan; Q234 is contiguous per vocabulary row.
    const int q1RowBytes = cst.inputDepthQuad * 2;
    const bool hasSecondRow = uz + 1 < cst.outputDepthQuad;
    const device uchar *rowWeight0 = wt + uz * q1RowBytes;
    const device uchar *rowWeight1 = rowWeight0 + (hasSecondRow ? q1RowBytes : 0);
    const int middleIndex = int(tiisg) & (middleStep - 1);
    const int outerIndex = int(tiisg) / middleStep;
    constexpr FLOAT q1Span = FLOAT(8), q1Offset = FLOAT(4.5);
    FLOAT4 result0 = FLOAT4(0);
    FLOAT4 result1 = FLOAT4(0);

    for (int bi = outerIndex; bi < cst.blockCount; bi += outerStep) {
        // Original grouped parameters, shared with Q4 refinement and embedding.
        // Restore the loader's range compensation before applying c + s*(8*bit - 4.5).
        const int sb0 = 2 * (uz * cst.blockCount + bi);
        const int sb1 = sb0 + (hasSecondRow ? 2 * cst.blockCount : 0);
        const FLOAT4 scale0 = FLOAT4(dequantScale[sb0]) / FLOAT(cst.scaleCoef);
        const FLOAT4 center0 = FLOAT4(dequantScale[sb0 + 1]) / FLOAT(cst.scaleCoef);
        const FLOAT4 scale1 = FLOAT4(dequantScale[sb1]) / FLOAT(cst.scaleCoef);
        const FLOAT4 center1 = FLOAT4(dequantScale[sb1 + 1]) / FLOAT(cst.scaleCoef);
        FLOAT blockInputSum = FLOAT(0);
        FLOAT4 blockRaw0 = FLOAT4(0), blockRaw1 = FLOAT4(0);
        for (int offset = middleIndex * 4; offset < blockSlices; offset += middleStep * 4) {
            const int z = bi * blockSlices + offset;
            FLOAT inputSum = FLOAT(0);
            FLOAT4 raw0 = FLOAT4(0), raw1 = FLOAT4(0);
            const uint2 packedWeight0 = *reinterpret_cast<const device uint2 *>(rowWeight0 + z * 2);
            const uint2 packedWeight1 = *reinterpret_cast<const device uint2 *>(rowWeight1 + z * 2);
            const FLOAT4 in0 = FLOAT4(in[(z + 0) * areaSize + rx]);
            const FLOAT4 in1 = FLOAT4(in[(z + 1) * areaSize + rx]);
            const FLOAT4 in2 = FLOAT4(in[(z + 2) * areaSize + rx]);
            const FLOAT4 in3 = FLOAT4(in[(z + 3) * areaSize + rx]);
#ifdef TIED_BIT_C4_SCAN
            inputSum = inputSums[z / 4];
#else
            inputSum += in0[0] + in0[1] + in0[2] + in0[3] +
                        in1[0] + in1[1] + in1[2] + in1[3] +
                        in2[0] + in2[1] + in2[2] + in2[3] +
                        in3[0] + in3[1] + in3[2] + in3[3];
#endif
            raw0 += tied_bit_dot4(in0, ushort(packedWeight0.x)) +
                    tied_bit_dot4(in1, ushort(packedWeight0.x >> 16)) +
                    tied_bit_dot4(in2, ushort(packedWeight0.y)) +
                    tied_bit_dot4(in3, ushort(packedWeight0.y >> 16));
            raw1 += tied_bit_dot4(in0, ushort(packedWeight1.x)) +
                    tied_bit_dot4(in1, ushort(packedWeight1.x >> 16)) +
                    tied_bit_dot4(in2, ushort(packedWeight1.y)) +
                    tied_bit_dot4(in3, ushort(packedWeight1.y >> 16));
            blockRaw0 += raw0;
            blockRaw1 += raw1;
            blockInputSum += inputSum;
        }
        result0 += blockRaw0 * (q1Span * scale0) + blockInputSum * (center0 - q1Offset * scale0);
        result1 += blockRaw1 * (q1Span * scale1) + blockInputSum * (center1 - q1Offset * scale1);
    }

    const FLOAT4 res0 = simd_sum(result0);
    const FLOAT4 res1 = simd_sum(result1);
    if (tiisg == 0) {
        const ftype4 value0 = ftype4(res0 + FLOAT4(biasTerms[uz]));
        out[uz] = value0;
        if (hasSecondRow) out[uz + 1] = ftype4(res1 + FLOAT4(biasTerms[uz + 1]));
    }
}

struct tied_bit_global_param { int count; int topk; int reserved; int chunk_size; float reserved_float; };
inline uint tied_bit_ordered_key(ftype value) {
#ifdef MNN_METAL_FLOAT16_STORAGE
    const ushort bits = as_type<ushort>(value);
    return (bits & 0x8000) != 0 ? ushort(~bits) : ushort(bits ^ 0x8000);
#else
    const uint bits = as_type<uint>(value);
    return (bits & 0x80000000u) != 0 ? ~bits : (bits ^ 0x80000000u);
#endif
}

kernel void tied_bit_global_histogram_reset(device atomic_uint *histogram [[buffer(0)]],
                                              device uint *threshold        [[buffer(1)]],
                                              uint tid [[thread_index_in_threadgroup]]) {
    atomic_store_explicit(histogram + tid, 0u, memory_order_relaxed);
    if (tid < 4) {
        threshold[tid] = 0u;
    }
}

kernel void tied_bit_global_histogram_high(const device ftype *logits             [[buffer(0)]],
                                             device atomic_uint *histogram          [[buffer(1)]],
                                             constant tied_bit_global_param& p      [[buffer(2)]],
                                             uint tid [[thread_index_in_threadgroup]],
                                             uint3 tgp [[threadgroup_position_in_grid]]) {
    threadgroup atomic_uint localHistogram[256];
    atomic_store_explicit(localHistogram + tid, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int begin = int(tgp.x) * p.chunk_size;
    const int end = min(begin + p.chunk_size, p.count);
    for (int i = begin + int(tid); i < end; i += kHistogramThreads) {
        const uint key = uint(tied_bit_ordered_key(logits[i]));
        const uint bucket = key >> ((sizeof(ftype) - 1) * 8);
        atomic_fetch_add_explicit(localHistogram + bucket, 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint count = atomic_load_explicit(localHistogram + tid, memory_order_relaxed);
    if (count != 0u) {
        atomic_fetch_add_explicit(histogram + tid, count, memory_order_relaxed);
    }
}

kernel void tied_bit_global_select_high_sg(device atomic_uint *histogram [[buffer(0)]],
                                          device uint *threshold        [[buffer(1)]],
                                          constant tied_bit_global_param& p [[buffer(2)]],
                                          uint tid [[thread_index_in_threadgroup]]) {
    // One thread per descending radix bucket, with a two-level SIMD prefix sum.
    threadgroup uint totals[kHistogramThreads / kSimdWidth];
    const uint bucket = uint(0xff) - tid;
    const uint count = atomic_load_explicit(histogram + bucket, memory_order_relaxed);
    const uint inclusive = simd_prefix_inclusive_sum(count);
    const uint total = simd_sum(count);
    if ((tid % kSimdWidth) == 0u) {
        totals[tid / kSimdWidth] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint above = inclusive - count;
    for (uint sg = 0u; sg < tid / kSimdWidth; ++sg) {
        above += totals[sg];
    }
#ifndef MNN_METAL_FLOAT16_STORAGE
    above += threshold[kThresholdAbovePrefix];
    // All lanes must read the old rank before the winning lane advances it.
    threadgroup_barrier(mem_flags::mem_device);
#endif
    if (above < uint(p.topk) && above + count >= uint(p.topk)) {
        threshold[kThresholdPrefix] = (threshold[kThresholdPrefix] << 8) | bucket;
        threshold[kThresholdAbovePrefix] = above;
    }
    threadgroup_barrier(mem_flags::mem_device);
    atomic_store_explicit(histogram + tid, 0u, memory_order_relaxed);
}

kernel void tied_bit_global_histogram_low(const device ftype *logits             [[buffer(0)]],
                                            device atomic_uint *histogram          [[buffer(1)]],
                                            const device uint *threshold           [[buffer(2)]],
                                            constant tied_bit_global_param& p      [[buffer(3)]],
#ifndef MNN_METAL_FLOAT16_STORAGE
                                            constant uint& shift                   [[buffer(4)]],
#endif
                                            uint tid [[thread_index_in_threadgroup]],
                                            uint3 tgp [[threadgroup_position_in_grid]]) {
    threadgroup atomic_uint localHistogram[256];
    atomic_store_explicit(localHistogram + tid, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
#ifdef MNN_METAL_FLOAT16_STORAGE
    constexpr uint shift = 0;
#endif
    const uint prefix = threshold[kThresholdPrefix];
    const int begin = int(tgp.x) * p.chunk_size;
    const int end = min(begin + p.chunk_size, p.count);
    for (int i = begin + int(tid); i < end; i += kHistogramThreads) {
        const uint key = uint(tied_bit_ordered_key(logits[i]));
        if ((key >> (shift + 8)) == prefix) {
            atomic_fetch_add_explicit(localHistogram + ((key >> shift) & 0xff), 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint count = atomic_load_explicit(localHistogram + tid, memory_order_relaxed);
    if (count != 0u) {
        atomic_fetch_add_explicit(histogram + tid, count, memory_order_relaxed);
    }
}

// Metal's indirect dispatch ABI: exactly one (x, y, z) threadgroup grid.
struct tied_bit_dispatch_args { uint x; uint y; uint z; };
inline void tied_bit_write_dispatch(device tied_bit_dispatch_args& args, uint candidates) {
    // Include the partial final threadgroup without overflowing the candidate count.
    args.x = candidates / kRerankRowsPerGroup + uint(candidates % kRerankRowsPerGroup != 0);
    args.y = 1u;
    args.z = 1u;
}

kernel void tied_bit_global_select_low_sg(const device atomic_uint *histogram [[buffer(0)]],
                                         device uint *threshold              [[buffer(1)]],
                                         device atomic_uint *candidateCount  [[buffer(2)]],
                                         device tied_bit_dispatch_args& indirectArgs           [[buffer(3)]],
                                         constant tied_bit_global_param& p   [[buffer(4)]],
                                         uint tid [[thread_index_in_threadgroup]]) {
    threadgroup uint totals[kHistogramThreads / kSimdWidth];
    const uint bucket = uint(0xff) - tid;
    const uint count = atomic_load_explicit(histogram + bucket, memory_order_relaxed);
    const uint inclusive = simd_prefix_inclusive_sum(count);
    const uint total = simd_sum(count);
    if ((tid % kSimdWidth) == 0u) {
        totals[tid / kSimdWidth] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint rankInBucket = uint(p.topk) - threshold[kThresholdAbovePrefix];
    uint seen = inclusive - count;
    for (uint sg = 0u; sg < tid / kSimdWidth; ++sg) {
        seen += totals[sg];
    }
    if (seen < rankInBucket && seen + count >= rankInBucket) {
        threshold[kThresholdKey] = (threshold[kThresholdPrefix] << 8) | bucket;
        // Include every tie, exactly as compact's ordered_key >= threshold.
        // This counts scalar candidates, not deduplicated C4 weight rows.
        threshold[kThresholdSelected] = threshold[kThresholdAbovePrefix] + seen + count;
    }
    // The winning bucket need not be lane zero; publish its count before lane zero reads it.
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0) {
        atomic_store_explicit(candidateCount, 0u, memory_order_relaxed);
        const uint selected = threshold[kThresholdSelected];
        tied_bit_write_dispatch(indirectArgs, selected);
    }
}

kernel void tied_bit_global_compact(device ftype *logits                       [[buffer(0)]],
                                      const device uint *threshold               [[buffer(1)]],
                                      device atomic_uint *candidateCount          [[buffer(2)]],
                                      device int *compactIndex                    [[buffer(3)]],
                                      constant tied_bit_global_param& p           [[buffer(4)]],
                                      uint tid [[thread_position_in_grid]]) {
    if (int(tid) >= p.count) return;
    const ftype value = logits[tid];
    if (uint(tied_bit_ordered_key(value)) >= threshold[kThresholdKey]) {
        const uint dst = atomic_fetch_add_explicit(candidateCount, 1u, memory_order_relaxed);
        compactIndex[dst] = int(tid);
    }
#ifdef MNN_METAL_FLOAT16_STORAGE
    logits[tid] = -65504.0h;
#else
    logits[tid] = -FLT_MAX;
#endif
}

inline ushort tied_bit_q234_unpack_lane(const device uchar *q1, const device uchar *residual,
                                        int tileInPair, int outputLane) {
    // Each row has one upper-bit byte and two low-bit bytes per eight inputs.
    const int byte = outputLane / 2;
    const int shift = (1 - outputLane % 2) * 4;
    const uint msb = (uint(q1[byte]) >> shift) & 0xf;
    const uint high1 = (uint(residual[0]) >> ((1 - tileInPair) * 4)) & 0xf;
    const uint low = uint(residual[1 + tileInPair]);
    ushort packed = 0;
    #pragma unroll
    for (int channel = 0; channel < 4; ++channel) {
        const int bit = 3 - channel;
        const uint code = ((low >> (bit * 2)) & 3) |
                          (((high1 >> bit) & 1u) << 2) |
                          (((msb >> bit) & 1u) << 3);
        packed |= ushort(code << packedCodeShift(channel));
    }
    return packed;
}

#ifdef TIED_BIT_DIRECT_DECODE
template <int channel>
inline FLOAT tied_bit_code(uint hi, uint mid, uint lo) {
    constexpr uint bit = 3 - channel;
    return FLOAT((((hi >> bit) & 1u) << 3) | (((mid >> bit) & 1u) << 2) | ((lo >> (2 * bit)) & 3u));
}
#endif

kernel void tied_bit_compact_rerank(const device ftype4 *in               [[buffer(0)]],
                                      device ftype *out                      [[buffer(1)]],
                                      constant conv1x1_constants& cst        [[buffer(2)]],
                                      const device uchar *wt                 [[buffer(3)]],
                                      const device ftype4 *biasTerms         [[buffer(4)]],
                                      const device ftype4 *dequantScale      [[buffer(5)]],
                                      const device int *compactIndex         [[buffer(6)]],
                                      device atomic_uint *candidateCount      [[buffer(7)]],
                                      uint3 gid [[threadgroup_position_in_grid]],
                                      uint tiisg [[thread_index_in_simdgroup]],
                                      uint sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr int blockSlices = TIED_BIT_BLOCK_QUADS;
    constexpr int middleStep = inputLanes(blockSlices, 2);
    constexpr int outerStep = kSimdWidth / middleStep;
#if TIED_BIT_SPLIT_K
    const int candidate = int(gid.x) * kRerankRowsPerGroup + int(sgitg) / 2;
    const bool upper = (sgitg & 1u) != 0;
    const int count = int(atomic_load_explicit(candidateCount, memory_order_relaxed));
    const int token = candidate < count ? compactIndex[candidate] : -1;
    const bool valid = uint(token) < uint(cst.outputChannel);
    const int outputIndex = valid ? token : 0;
    threadgroup FLOAT upperPartial[kRerankRowsPerGroup];
#else
    const int candidate = int(gid.x) * kRerankSimdGroups + int(sgitg);
    const int count = int(atomic_load_explicit(candidateCount, memory_order_relaxed));
    if (candidate >= count) return;
    const int outputIndex = compactIndex[candidate];
    if (outputIndex < 0 || outputIndex >= cst.outputChannel) return;
#endif
    const int outputRow = outputIndex / 4;
    const int outputLane = outputIndex % 4;
    // Q1 retains C4 tiles for the scan; Q234 is contiguous per vocabulary row.
    const int q1RowBytes = cst.inputDepthQuad * 2;
    const int q1PlaneBytes = cst.outputDepthQuad * q1RowBytes;
    const device uchar *rowQ1 = wt + outputRow * q1RowBytes;
    const device uchar *rowQ234 = wt + q1PlaneBytes + outputIndex * (cst.inputDepthQuad / 2 * 3);
    const int middleIndex = int(tiisg) & (middleStep - 1);
    const int outerIndex = int(tiisg) / middleStep;
    // Candidates are individual vocabulary rows. Decode and accumulate only
    // their selected C4 lane; computing the other three rows discards work and
    // can repeat that work when several candidates share a packed C4 row.
    FLOAT result = FLOAT(0);
#if TIED_BIT_SPLIT_K
    // Tail groups must reach the barrier even when their row has no candidate.
    const int firstBlock = upper ? cst.blockCount / 2 : 0;
    const int endBlock = valid ? (upper ? cst.blockCount : cst.blockCount / 2) : 0;
#else
    const int firstBlock = 0;
    const int endBlock = cst.blockCount;
#endif
    for (int bi = firstBlock + outerIndex; bi < endBlock; bi += outerStep) {
        const int sb = 2 * (outputRow * cst.blockCount + bi);
        const FLOAT originalScale = FLOAT(dequantScale[sb][outputLane]) / FLOAT(cst.scaleCoef);
        const FLOAT originalBias = FLOAT(dequantScale[sb + 1][outputLane]) / FLOAT(cst.scaleCoef);
        const FLOAT adjustedBias = originalBias - FLOAT(8) * originalScale;
        for (int offset = middleIndex * 2; offset < blockSlices; offset += middleStep * 2) {
            const int z = bi * blockSlices + offset;
            const device uchar *residual = rowQ234 + (z / 2) * 3;
#ifdef TIED_BIT_DIRECT_DECODE
            const uint pair = *(const device uint*)(rowQ1 + 2 * z);
            const uint shift = uint((outputLane / 2) * 8 + (1 - outputLane % 2) * 4);
            const uint hi0 = (pair >> shift) & 15u;
            const uint hi1 = (pair >> (16u + shift)) & 15u;
            const uint mid = residual[0], lo0 = residual[1], lo1 = residual[2];
#else
            const ushort w0 = tied_bit_q234_unpack_lane(rowQ1 + z * 2, residual, 0, outputLane);
            const ushort w1 = tied_bit_q234_unpack_lane(rowQ1 + (z + 1) * 2, residual, 1, outputLane);
#endif
            const FLOAT4 in0 = FLOAT4(in[z]);
            const FLOAT4 in1 = FLOAT4(in[z + 1]);
            const FLOAT inputSum = in0[0] + in0[1] + in0[2] + in0[3] +
                                   in1[0] + in1[1] + in1[2] + in1[3];
#ifdef TIED_BIT_DIRECT_DECODE
            const FLOAT raw = in0[0] * tied_bit_code<0>(hi0, mid >> 4, lo0) +
                              in0[1] * tied_bit_code<1>(hi0, mid >> 4, lo0) +
                              in0[2] * tied_bit_code<2>(hi0, mid >> 4, lo0) +
                              in0[3] * tied_bit_code<3>(hi0, mid >> 4, lo0) +
                              in1[0] * tied_bit_code<0>(hi1, mid, lo1) +
                              in1[1] * tied_bit_code<1>(hi1, mid, lo1) +
                              in1[2] * tied_bit_code<2>(hi1, mid, lo1) +
                              in1[3] * tied_bit_code<3>(hi1, mid, lo1);
#else
            const FLOAT raw = in0[0] * FLOAT((w0 >> packedCodeShift(0)) & 0xf) +
                              in0[1] * FLOAT((w0 >> packedCodeShift(1)) & 0xf) +
                              in0[2] * FLOAT((w0 >> packedCodeShift(2)) & 0xf) +
                              in0[3] * FLOAT((w0 >> packedCodeShift(3)) & 0xf) +
                              in1[0] * FLOAT((w1 >> packedCodeShift(0)) & 0xf) +
                              in1[1] * FLOAT((w1 >> packedCodeShift(1)) & 0xf) +
                              in1[2] * FLOAT((w1 >> packedCodeShift(2)) & 0xf) +
                              in1[3] * FLOAT((w1 >> packedCodeShift(3)) & 0xf);
#endif
            result += raw * originalScale + inputSum * adjustedBias;
        }
    }
#if TIED_BIT_SPLIT_K
    const FLOAT partial = simd_sum(result);
    if (upper && tiisg == 0) {
        upperPartial[sgitg / 2] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (valid && !upper && tiisg == 0) {
        const FLOAT combined = partial + upperPartial[sgitg / 2];
        out[outputIndex] = ftype(combined + FLOAT(biasTerms[outputRow][outputLane]));
    }
#else
    const FLOAT finalValue = simd_sum(result);
    if (tiisg == 0) {
        out[outputIndex] = ftype(finalValue + FLOAT(biasTerms[outputRow][outputLane]));
    }
#endif
}

)metal";

#endif
#endif
