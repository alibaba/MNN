//
//  TurboQuant.hpp
//  MNN
//
//  TurboQuant KV Cache Quantization (3.5 bits per value)
//  Based on: "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
//  Algorithm 1 (TurboQuant_mse): WHT rotation + Lloyd-Max 3-bit scalar quantization
//

#ifndef TURBOQUANT_HPP
#define TURBOQUANT_HPP

#include <cstdint>
#include <cmath>
#include <cstring>

#define TQ3_BLOCK_SIZE 32
#define TQ3_PACKED_INDICES_BYTES 12 // 32 * 3 bits = 96 bits = 12 bytes
#define TQ3_BYTES_PER_BLOCK 14      // 2 (fp16 scale) + 12 (packed indices)

// Lloyd-Max optimal 3-bit codebook centroids for N(0,1) distribution
// After RMS normalization + WHT on block_size=32, each coordinate ~ N(0,1)
// Pre-computed via iterative Lloyd-Max algorithm (178 iterations convergence)
static const float TQ3_CODEBOOK[8] = {-2.1519f, -1.3439f, -0.7560f, -0.2451f, 0.2451f, 0.7560f, 1.3439f, 2.1519f};

// Decision boundaries (midpoints between consecutive centroids)
static const float TQ3_BOUNDARIES[7] = {-1.7479f, -1.0500f, -0.5005f, 0.0f, 0.5005f, 1.0500f, 1.7479f};

// Deterministic sign pattern for WHT randomization (golden ratio hash)
// signs[i] = ((i * 0x9E3779B9) >> 31) ? -1.0f : 1.0f
static const float TQ3_SIGNS[TQ3_BLOCK_SIZE] = {
    1.0f,  -1.0f, 1.0f, -1.0f, 1.0f, 1.0f,  -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f,  -1.0f, 1.0f,
    -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f,  1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f,
};

// fp16 conversion helpers
static inline uint16_t tq3_float_to_fp16(float val) {
    uint32_t f32;
    memcpy(&f32, &val, sizeof(f32));
    uint32_t sign = (f32 >> 16) & 0x8000;
    int32_t exponent = ((f32 >> 23) & 0xFF) - 127 + 15;
    uint32_t mantissa = (f32 >> 13) & 0x3FF;
    if (exponent <= 0) {
        return (uint16_t)sign;
    } else if (exponent >= 31) {
        return (uint16_t)(sign | 0x7C00);
    }
    return (uint16_t)(sign | (exponent << 10) | mantissa);
}

static inline float tq3_fp16_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;
    if (exponent == 0) {
        if (mantissa == 0) {
            uint32_t result = sign;
            float ret;
            memcpy(&ret, &result, sizeof(ret));
            return ret;
        }
        // Subnormal
        exponent = 1;
        while (!(mantissa & 0x400)) {
            mantissa <<= 1;
            exponent--;
        }
        mantissa &= 0x3FF;
        exponent = exponent + 127 - 15;
    } else if (exponent == 31) {
        exponent = 255;
    } else {
        exponent = exponent + 127 - 15;
    }
    uint32_t result = sign | (exponent << 23) | (mantissa << 13);
    float ret;
    memcpy(&ret, &result, sizeof(ret));
    return ret;
}

// Walsh-Hadamard Transform forward (in-place, block size 32)
// Applies: sign flip → butterfly stages → normalize by 1/sqrt(32)
static inline void tq3_wht_forward_32(float* out, const float* in) {
    // Step 1: Apply sign flips
    for (int i = 0; i < TQ3_BLOCK_SIZE; i++) {
        out[i] = in[i] * TQ3_SIGNS[i];
    }
    // Step 2: Butterfly stages (log2(32) = 5 stages)
    for (int step = 1; step < TQ3_BLOCK_SIZE; step <<= 1) {
        for (int i = 0; i < TQ3_BLOCK_SIZE; i += step << 1) {
            for (int j = i; j < i + step; j++) {
                float a = out[j];
                float b = out[j + step];
                out[j] = a + b;
                out[j + step] = a - b;
            }
        }
    }
    // Step 3: Normalize
    const float norm = 1.0f / sqrtf((float)TQ3_BLOCK_SIZE);
    for (int i = 0; i < TQ3_BLOCK_SIZE; i++) {
        out[i] *= norm;
    }
}

// Walsh-Hadamard Transform inverse (in-place, block size 32)
// Applies: butterfly stages → normalize → undo sign flips
static inline void tq3_wht_inverse_32(float* out, const float* in) {
    // Step 1: Copy input
    for (int i = 0; i < TQ3_BLOCK_SIZE; i++) {
        out[i] = in[i];
    }
    // Step 2: Butterfly stages (WHT is self-inverse up to scaling)
    for (int step = 1; step < TQ3_BLOCK_SIZE; step <<= 1) {
        for (int i = 0; i < TQ3_BLOCK_SIZE; i += step << 1) {
            for (int j = i; j < i + step; j++) {
                float a = out[j];
                float b = out[j + step];
                out[j] = a + b;
                out[j + step] = a - b;
            }
        }
    }
    // Step 3: Normalize and undo sign flips
    const float norm = 1.0f / sqrtf((float)TQ3_BLOCK_SIZE);
    for (int i = 0; i < TQ3_BLOCK_SIZE; i++) {
        out[i] *= norm * TQ3_SIGNS[i];
    }
}

// Find nearest codebook index for a rotated coordinate value
static inline uint8_t tq3_find_nearest(float val) {
    uint8_t idx = 0;
    for (int b = 0; b < 7; b++) {
        if (val > TQ3_BOUNDARIES[b]) {
            idx = b + 1;
        }
    }
    return idx;
}

// Pack 8 3-bit indices into 3 bytes
static inline void tq3_pack_3bit_8(uint8_t* dst, const uint8_t* idx) {
    dst[0] = (idx[0]) | (idx[1] << 3) | (idx[2] << 6);
    dst[1] = (idx[2] >> 2) | (idx[3] << 1) | (idx[4] << 4) | (idx[5] << 7);
    dst[2] = (idx[5] >> 1) | (idx[6] << 2) | (idx[7] << 5);
}

// Unpack 3 bytes into 8 3-bit indices
static inline void tq3_unpack_3bit_8(uint8_t* idx, const uint8_t* src) {
    idx[0] = src[0] & 7;
    idx[1] = (src[0] >> 3) & 7;
    idx[2] = ((src[0] >> 6) | (src[1] << 2)) & 7;
    idx[3] = (src[1] >> 1) & 7;
    idx[4] = (src[1] >> 4) & 7;
    idx[5] = ((src[1] >> 7) | (src[2] << 1)) & 7;
    idx[6] = (src[2] >> 2) & 7;
    idx[7] = (src[2] >> 5) & 7;
}

// Quantize a block of 32 float values into 14 bytes TQ3 format
// Layout: [2 bytes fp16 scale] [12 bytes packed 3-bit indices]
static inline void tq3_quantize_block(uint8_t* dst, const float* src) {
    // Step 1: Compute RMS scale
    float sumSq = 0.0f;
    for (int i = 0; i < TQ3_BLOCK_SIZE; i++) {
        sumSq += src[i] * src[i];
    }
    float rms = sqrtf(sumSq / TQ3_BLOCK_SIZE);
    if (rms < 1e-10f) {
        rms = 1e-10f;
    }

    // Store scale as fp16
    uint16_t scaleFp16 = tq3_float_to_fp16(rms);
    memcpy(dst, &scaleFp16, 2);

    // Step 2: Normalize
    float normalized[TQ3_BLOCK_SIZE];
    float invRms = 1.0f / rms;
    for (int i = 0; i < TQ3_BLOCK_SIZE; i++) {
        normalized[i] = src[i] * invRms;
    }

    // Step 3: Apply WHT forward
    float rotated[TQ3_BLOCK_SIZE];
    tq3_wht_forward_32(rotated, normalized);

    // Step 4: Find nearest codebook index for each coordinate
    uint8_t indices[TQ3_BLOCK_SIZE];
    for (int i = 0; i < TQ3_BLOCK_SIZE; i++) {
        indices[i] = tq3_find_nearest(rotated[i]);
    }

    // Step 5: Pack 3-bit indices (4 groups of 8 → 12 bytes)
    for (int g = 0; g < 4; g++) {
        tq3_pack_3bit_8(dst + 2 + g * 3, indices + g * 8);
    }
}

// Dequantize a 14-byte TQ3 block into 32 float values
static inline void tq3_dequantize_block(float* dst, const uint8_t* src) {
    // Step 1: Read scale
    uint16_t scaleFp16;
    memcpy(&scaleFp16, src, 2);
    float scale = tq3_fp16_to_float(scaleFp16);

    // Step 2: Unpack 3-bit indices and look up centroids
    float rotated[TQ3_BLOCK_SIZE];
    for (int g = 0; g < 4; g++) {
        uint8_t indices[8];
        tq3_unpack_3bit_8(indices, src + 2 + g * 3);
        for (int k = 0; k < 8; k++) {
            rotated[g * 8 + k] = TQ3_CODEBOOK[indices[k]];
        }
    }

    // Step 3: Apply inverse WHT
    float reconstructed[TQ3_BLOCK_SIZE];
    tq3_wht_inverse_32(reconstructed, rotated);

    // Step 4: Scale back
    for (int i = 0; i < TQ3_BLOCK_SIZE; i++) {
        dst[i] = reconstructed[i] * scale;
    }
}

// Dequantize a 14-byte TQ3 block and write 32 values to strided destination
// dstStride: number of elements (not bytes) between consecutive output values
// Fuses: unpack → lookup → inverse WHT → scale → strided write
template <typename T>
static inline void tq3_dequantize_block_strided(T* dst, int dstStride, const uint8_t* src) {
    // Step 1: Read scale
    uint16_t scaleFp16;
    memcpy(&scaleFp16, src, 2);
    float scale = tq3_fp16_to_float(scaleFp16);

    // Step 2: Unpack 3-bit indices and look up centroids
    float rotated[TQ3_BLOCK_SIZE];
    for (int g = 0; g < 4; g++) {
        uint8_t indices[8];
        tq3_unpack_3bit_8(indices, src + 2 + g * 3);
        for (int k = 0; k < 8; k++) {
            rotated[g * 8 + k] = TQ3_CODEBOOK[indices[k]];
        }
    }

    // Step 3: Apply inverse WHT
    float reconstructed[TQ3_BLOCK_SIZE];
    tq3_wht_inverse_32(reconstructed, rotated);

    // Step 4: Scale and write to strided destination
    for (int i = 0; i < TQ3_BLOCK_SIZE; i++) {
        dst[i * dstStride] = (T)(reconstructed[i] * scale);
    }
}

// Compute dot product of pre-rotated query with a TQ3 block (32 values)
// q_rotated: WHT_forward(Q) for the corresponding 32-dim slice, already scaled by 1/sqrt(headDim)
// src: 14-byte TQ3 block
// Returns: scale * Σ(q_rotated[i] * codebook[idx[i]])
// This avoids the full dequant (no inverse WHT, no temp buffer)
static inline float tq3_vec_dot_block(const float* q_rotated, const uint8_t* src) {
    // Read scale
    uint16_t scaleFp16;
    memcpy(&scaleFp16, src, 2);
    float scale = tq3_fp16_to_float(scaleFp16);

    // Unpack indices and accumulate dot product with codebook values
    float dot = 0.0f;
    for (int g = 0; g < 4; g++) {
        uint8_t indices[8];
        tq3_unpack_3bit_8(indices, src + 2 + g * 3);
        for (int k = 0; k < 8; k++) {
            dot += q_rotated[g * 8 + k] * TQ3_CODEBOOK[indices[k]];
        }
    }
    return dot * scale;
}

// Accumulate one TQ3 block's codebook values weighted by w into acc_rotated[32]
// w should be softmax_weight * scale (caller computes this)
static inline void tq3_weighted_acc_block(float* acc_rotated, float w, const uint8_t* packed12) {
    for (int g = 0; g < 4; g++) {
        uint8_t indices[8];
        tq3_unpack_3bit_8(indices, packed12 + g * 3);
        for (int k = 0; k < 8; k++) {
            acc_rotated[g * 8 + k] += w * TQ3_CODEBOOK[indices[k]];
        }
    }
}

// ============================================================================
// TQ4: 4-bit TurboQuant (4.5 bits per value)
// Same WHT rotation as TQ3, but 16-entry Lloyd-Max codebook + nibble packing
// ============================================================================

#define TQ4_BLOCK_SIZE 32
#define TQ4_PACKED_INDICES_BYTES 16 // 32 * 4 bits = 128 bits = 16 bytes
#define TQ4_BYTES_PER_BLOCK 18      // 2 (fp16 scale) + 16 (packed indices)

// Lloyd-Max optimal 16-entry codebook centroids for N(0,1)
// D_mse ≈ 0.0095 (vs TQ3 D_mse ≈ 0.032)
static const float TQ4_CODEBOOK[16] = {
    -2.7326f, -2.0690f, -1.6180f, -1.2562f, -0.9423f, -0.6568f, -0.3880f, -0.1284f,
    0.1284f,  0.3880f,  0.6568f,  0.9423f,  1.2562f,  1.6180f,  2.0690f,  2.7326f,
};

static const float TQ4_BOUNDARIES[15] = {
    -2.4008f, -1.8435f, -1.4371f, -1.0993f, -0.7995f, -0.5224f, -0.2582f, 0.0000f,
    0.2582f,  0.5224f,  0.7995f,  1.0993f,  1.4371f,  1.8435f,  2.4008f,
};

static inline uint8_t tq4_find_nearest(float val) {
    uint8_t idx = 0;
    for (int b = 0; b < 15; b++) {
        if (val > TQ4_BOUNDARIES[b])
            idx = b + 1;
    }
    return idx;
}

// Pack 8 4-bit indices into 4 bytes (simple nibble packing)
static inline void tq4_pack_4bit_8(uint8_t* dst, const uint8_t* idx) {
    dst[0] = (idx[0]) | (idx[1] << 4);
    dst[1] = (idx[2]) | (idx[3] << 4);
    dst[2] = (idx[4]) | (idx[5] << 4);
    dst[3] = (idx[6]) | (idx[7] << 4);
}

// Unpack 4 bytes into 8 4-bit indices
static inline void tq4_unpack_4bit_8(uint8_t* idx, const uint8_t* src) {
    idx[0] = src[0] & 0xF;
    idx[1] = src[0] >> 4;
    idx[2] = src[1] & 0xF;
    idx[3] = src[1] >> 4;
    idx[4] = src[2] & 0xF;
    idx[5] = src[2] >> 4;
    idx[6] = src[3] & 0xF;
    idx[7] = src[3] >> 4;
}

// Quantize a block of 32 float values into 18 bytes TQ4 format
static inline void tq4_quantize_block(uint8_t* dst, const float* src) {
    float sumSq = 0.0f;
    for (int i = 0; i < TQ4_BLOCK_SIZE; i++)
        sumSq += src[i] * src[i];
    float rms = sqrtf(sumSq / TQ4_BLOCK_SIZE);
    if (rms < 1e-10f)
        rms = 1e-10f;

    uint16_t scaleFp16 = tq3_float_to_fp16(rms);
    memcpy(dst, &scaleFp16, 2);

    float normalized[TQ4_BLOCK_SIZE];
    float invRms = 1.0f / rms;
    for (int i = 0; i < TQ4_BLOCK_SIZE; i++)
        normalized[i] = src[i] * invRms;

    float rotated[TQ4_BLOCK_SIZE];
    tq3_wht_forward_32(rotated, normalized);

    uint8_t indices[TQ4_BLOCK_SIZE];
    for (int i = 0; i < TQ4_BLOCK_SIZE; i++)
        indices[i] = tq4_find_nearest(rotated[i]);

    // Pack 4-bit indices (4 groups of 8 → 16 bytes)
    for (int g = 0; g < 4; g++) {
        tq4_pack_4bit_8(dst + 2 + g * 4, indices + g * 8);
    }
}

// Dequantize a 18-byte TQ4 block into 32 float values
static inline void tq4_dequantize_block(float* dst, const uint8_t* src) {
    uint16_t scaleFp16;
    memcpy(&scaleFp16, src, 2);
    float scale = tq3_fp16_to_float(scaleFp16);

    float rotated[TQ4_BLOCK_SIZE];
    for (int g = 0; g < 4; g++) {
        uint8_t indices[8];
        tq4_unpack_4bit_8(indices, src + 2 + g * 4);
        for (int k = 0; k < 8; k++)
            rotated[g * 8 + k] = TQ4_CODEBOOK[indices[k]];
    }

    float reconstructed[TQ4_BLOCK_SIZE];
    tq3_wht_inverse_32(reconstructed, rotated);
    for (int i = 0; i < TQ4_BLOCK_SIZE; i++)
        dst[i] = reconstructed[i] * scale;
}

static inline float tq4_vec_dot_block(const float* q_rotated, const uint8_t* src) {
    uint16_t scaleFp16;
    memcpy(&scaleFp16, src, 2);
    float scale = tq3_fp16_to_float(scaleFp16);

    float dot = 0.0f;
    for (int g = 0; g < 4; g++) {
        uint8_t indices[8];
        tq4_unpack_4bit_8(indices, src + 2 + g * 4);
        for (int k = 0; k < 8; k++)
            dot += q_rotated[g * 8 + k] * TQ4_CODEBOOK[indices[k]];
    }
    return dot * scale;
}

static inline void tq4_weighted_acc_block(float* acc_rotated, float w, const uint8_t* packed16) {
    for (int g = 0; g < 4; g++) {
        uint8_t indices[8];
        tq4_unpack_4bit_8(indices, packed16 + g * 4);
        for (int k = 0; k < 8; k++)
            acc_rotated[g * 8 + k] += w * TQ4_CODEBOOK[indices[k]];
    }
}

// ============================================================================
// NEON SIMD optimized versions (aarch64)
// ============================================================================
#if defined(__aarch64__)
#include <arm_neon.h>

// Helper: unpack 3 bytes → 8 codebook float values via NEON vtbl on fp16 codebook
// cb_bytes: codebook as 8 fp16 values reinterpreted as uint8x16_t
// Returns two float32x4_t (lo: indices 0-3, hi: indices 4-7)
static inline void tq3_neon_unpack_lookup(float32x4_t& lo, float32x4_t& hi, const uint8_t* packed3,
                                          uint8x16_t cb_bytes) {
    // Load 3 bytes as 24-bit integer
    uint32_t w = packed3[0] | ((uint32_t)packed3[1] << 8) | ((uint32_t)packed3[2] << 16);

    // Extract 8 3-bit indices using NEON variable shifts
    uint32x4_t wv = vdupq_n_u32(w);
    static const int32_t shifts_lo[4] = {0, -3, -6, -9};
    static const int32_t shifts_hi[4] = {-12, -15, -18, -21};
    uint32x4_t idx_lo = vandq_u32(vshlq_u32(wv, vld1q_s32(shifts_lo)), vdupq_n_u32(7));
    uint32x4_t idx_hi = vandq_u32(vshlq_u32(wv, vld1q_s32(shifts_hi)), vdupq_n_u32(7));

    // Narrow uint32x4 → uint16x4 → uint8x8 (8 indices)
    uint8x8_t idx8 = vmovn_u16(vcombine_u16(vmovn_u32(idx_lo), vmovn_u32(idx_hi)));

    // Build fp16 byte-level lookup: index i → bytes [2*i, 2*i+1]
    uint8x8_t idx2 = vshl_n_u8(idx8, 1);
    uint8x8_t idx2p1 = vadd_u8(idx2, vdup_n_u8(1));
    uint8x8x2_t z = vzip_u8(idx2, idx2p1);
    uint8x16_t lookup = vcombine_u8(z.val[0], z.val[1]);

    // Gather fp16 codebook values via table lookup, convert to fp32
    uint8x16_t gathered = vqtbl1q_u8(cb_bytes, lookup);
    float16x8_t fp16v = vreinterpretq_f16_u8(gathered);
    lo = vcvt_f32_f16(vget_low_f16(fp16v));
    hi = vcvt_f32_f16(vget_high_f16(fp16v));
}

// Prepare codebook as fp16 bytes for NEON vtbl lookup
static inline uint8x16_t tq3_neon_codebook_fp16() {
    float32x4_t cb_lo = vld1q_f32(TQ3_CODEBOOK);
    float32x4_t cb_hi = vld1q_f32(TQ3_CODEBOOK + 4);
    float16x4_t fp16_lo = vcvt_f16_f32(cb_lo);
    float16x4_t fp16_hi = vcvt_f16_f32(cb_hi);
    return vreinterpretq_u8_f16(vcombine_f16(fp16_lo, fp16_hi));
}

// NEON optimized vec_dot: dot product of pre-rotated query with TQ3 block
static inline float tq3_vec_dot_block_neon(const float* q_rotated, const uint8_t* src) {
    uint16_t scaleFp16;
    memcpy(&scaleFp16, src, 2);
    float scale = tq3_fp16_to_float(scaleFp16);

    uint8x16_t cb = tq3_neon_codebook_fp16();
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);

    for (int g = 0; g < 4; g++) {
        float32x4_t lo, hi;
        tq3_neon_unpack_lookup(lo, hi, src + 2 + g * 3, cb);
        acc0 = vfmaq_f32(acc0, vld1q_f32(q_rotated + g * 8), lo);
        acc1 = vfmaq_f32(acc1, vld1q_f32(q_rotated + g * 8 + 4), hi);
    }

    acc0 = vaddq_f32(acc0, acc1);
    return vaddvq_f32(acc0) * scale;
}

// NEON optimized weighted accumulation for Value path
// acc[32] += w * codebook[indices], where w = softmax_weight * tq3_scale
static inline void tq3_weighted_acc_block_neon(float* acc, float w, const uint8_t* packed12) {
    uint8x16_t cb = tq3_neon_codebook_fp16();
    float32x4_t wv = vdupq_n_f32(w);

    for (int g = 0; g < 4; g++) {
        float32x4_t lo, hi;
        tq3_neon_unpack_lookup(lo, hi, packed12 + g * 3, cb);
        float32x4_t a0 = vld1q_f32(acc + g * 8);
        float32x4_t a1 = vld1q_f32(acc + g * 8 + 4);
        a0 = vfmaq_f32(a0, wv, lo);
        a1 = vfmaq_f32(a1, wv, hi);
        vst1q_f32(acc + g * 8, a0);
        vst1q_f32(acc + g * 8 + 4, a1);
    }
}

// Override TQ3 scalar versions with NEON
#undef tq3_vec_dot_block
#define tq3_vec_dot_block tq3_vec_dot_block_neon
#undef tq3_weighted_acc_block
#define tq3_weighted_acc_block tq3_weighted_acc_block_neon

// --- TQ4 NEON ---

// Helper: unpack 4 bytes (8 nibbles) → 8 codebook float values via NEON vtbl
// 16-entry fp16 codebook = 32 bytes → uint8x16x2_t for vqtbl2q
static inline void tq4_neon_unpack_lookup(float32x4_t& lo, float32x4_t& hi, const uint8_t* packed4,
                                          uint8x16x2_t cb_bytes) {
    // Load 4 bytes, extract 8 nibbles
    uint8x8_t raw = vld1_u8(packed4); // only first 4 bytes used
    // Even nibbles (low): raw & 0x0F; Odd nibbles (high): raw >> 4
    uint8x8_t lo_nib = vand_u8(raw, vdup_n_u8(0x0F));
    uint8x8_t hi_nib = vshr_n_u8(raw, 4);
    // Interleave: {lo[0], hi[0], lo[1], hi[1], ...} = {idx0,idx1,idx2,idx3,...}
    uint8x8x2_t z = vzip_u8(lo_nib, hi_nib);
    uint8x8_t idx8 = z.val[0]; // first 8 indices (from 4 bytes)

    // Build fp16 byte-level lookup: index i → bytes [2*i, 2*i+1]
    uint8x8_t idx2 = vshl_n_u8(idx8, 1);
    uint8x8_t idx2p1 = vadd_u8(idx2, vdup_n_u8(1));
    uint8x8x2_t zz = vzip_u8(idx2, idx2p1);
    uint8x16_t lookup = vcombine_u8(zz.val[0], zz.val[1]);

    // Gather from 32-byte codebook via 2-table lookup
    uint8x16_t gathered = vqtbl2q_u8(cb_bytes, lookup);
    float16x8_t fp16v = vreinterpretq_f16_u8(gathered);
    lo = vcvt_f32_f16(vget_low_f16(fp16v));
    hi = vcvt_f32_f16(vget_high_f16(fp16v));
}

static inline uint8x16x2_t tq4_neon_codebook_fp16() {
    uint8x16x2_t result;
    float32x4_t c0 = vld1q_f32(TQ4_CODEBOOK);
    float32x4_t c1 = vld1q_f32(TQ4_CODEBOOK + 4);
    float32x4_t c2 = vld1q_f32(TQ4_CODEBOOK + 8);
    float32x4_t c3 = vld1q_f32(TQ4_CODEBOOK + 12);
    float16x4_t h0 = vcvt_f16_f32(c0);
    float16x4_t h1 = vcvt_f16_f32(c1);
    float16x4_t h2 = vcvt_f16_f32(c2);
    float16x4_t h3 = vcvt_f16_f32(c3);
    result.val[0] = vreinterpretq_u8_f16(vcombine_f16(h0, h1));
    result.val[1] = vreinterpretq_u8_f16(vcombine_f16(h2, h3));
    return result;
}

static inline float tq4_vec_dot_block_neon(const float* q_rotated, const uint8_t* src) {
    uint16_t scaleFp16;
    memcpy(&scaleFp16, src, 2);
    float scale = tq3_fp16_to_float(scaleFp16);

    uint8x16x2_t cb = tq4_neon_codebook_fp16();
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);

    for (int g = 0; g < 4; g++) {
        float32x4_t lo, hi;
        tq4_neon_unpack_lookup(lo, hi, src + 2 + g * 4, cb);
        acc0 = vfmaq_f32(acc0, vld1q_f32(q_rotated + g * 8), lo);
        acc1 = vfmaq_f32(acc1, vld1q_f32(q_rotated + g * 8 + 4), hi);
    }

    acc0 = vaddq_f32(acc0, acc1);
    return vaddvq_f32(acc0) * scale;
}

static inline void tq4_weighted_acc_block_neon(float* acc, float w, const uint8_t* packed16) {
    uint8x16x2_t cb = tq4_neon_codebook_fp16();
    float32x4_t wv = vdupq_n_f32(w);

    for (int g = 0; g < 4; g++) {
        float32x4_t lo, hi;
        tq4_neon_unpack_lookup(lo, hi, packed16 + g * 4, cb);
        float32x4_t a0 = vld1q_f32(acc + g * 8);
        float32x4_t a1 = vld1q_f32(acc + g * 8 + 4);
        a0 = vfmaq_f32(a0, wv, lo);
        a1 = vfmaq_f32(a1, wv, hi);
        vst1q_f32(acc + g * 8, a0);
        vst1q_f32(acc + g * 8 + 4, a1);
    }
}

// Override TQ4 scalar versions with NEON
#undef tq4_vec_dot_block
#define tq4_vec_dot_block tq4_vec_dot_block_neon
#undef tq4_weighted_acc_block
#define tq4_weighted_acc_block tq4_weighted_acc_block_neon

// --- NEON WHT Transform ---

// WHT forward: sign flip + 5 butterfly stages + normalize
// NEON accelerates: sign flip+normalize fused, butterfly stages 3-5 (step>=4) vectorized
static inline void wht_forward_32_neon(float* out, const float* in) {
    // Fused sign flip + normalize: out[i] = in[i] * signs[i] * (1/sqrt(32))
    const float norm = 1.0f / sqrtf(32.0f);
    for (int i = 0; i < 32; i += 4) {
        float32x4_t v = vld1q_f32(in + i);
        float32x4_t s = vmulq_n_f32(vld1q_f32(TQ3_SIGNS + i), norm);
        vst1q_f32(out + i, vmulq_f32(v, s));
    }
    // Butterfly stages
    // Stage 1 (step=1) and Stage 2 (step=2): scalar (step < 4, hard to vectorize within registers)
    for (int step = 1; step <= 2; step <<= 1) {
        for (int i = 0; i < 32; i += step << 1) {
            for (int j = i; j < i + step; j++) {
                float a = out[j], b = out[j + step];
                out[j] = a + b;
                out[j + step] = a - b;
            }
        }
    }
    // Stage 3-5 (step=4,8,16): NEON vectorized
    for (int step = 4; step < 32; step <<= 1) {
        for (int i = 0; i < 32; i += step << 1) {
            for (int j = i; j < i + step; j += 4) {
                float32x4_t a = vld1q_f32(out + j);
                float32x4_t b = vld1q_f32(out + j + step);
                vst1q_f32(out + j, vaddq_f32(a, b));
                vst1q_f32(out + j + step, vsubq_f32(a, b));
            }
        }
    }
}

// WHT inverse: butterfly stages + normalize + sign flip
static inline void wht_inverse_32_neon(float* out, const float* in) {
    memcpy(out, in, 32 * sizeof(float));
    // Stages 1-2: scalar
    for (int step = 1; step <= 2; step <<= 1) {
        for (int i = 0; i < 32; i += step << 1) {
            for (int j = i; j < i + step; j++) {
                float a = out[j], b = out[j + step];
                out[j] = a + b;
                out[j + step] = a - b;
            }
        }
    }
    // Stages 3-5: NEON
    for (int step = 4; step < 32; step <<= 1) {
        for (int i = 0; i < 32; i += step << 1) {
            for (int j = i; j < i + step; j += 4) {
                float32x4_t a = vld1q_f32(out + j);
                float32x4_t b = vld1q_f32(out + j + step);
                vst1q_f32(out + j, vaddq_f32(a, b));
                vst1q_f32(out + j + step, vsubq_f32(a, b));
            }
        }
    }
    // Normalize + sign flip
    const float norm = 1.0f / sqrtf(32.0f);
    for (int i = 0; i < 32; i += 4) {
        float32x4_t v = vld1q_f32(out + i);
        float32x4_t s = vld1q_f32(TQ3_SIGNS + i);
        vst1q_f32(out + i, vmulq_n_f32(vmulq_f32(v, s), norm));
    }
}

// --- NEON quantize block ---

static inline void tq3_quantize_block_neon(uint8_t* dst, const float* src) {
    // RMS via NEON
    float32x4_t sumSq = vdupq_n_f32(0.0f);
    for (int i = 0; i < 32; i += 4) {
        float32x4_t v = vld1q_f32(src + i);
        sumSq = vfmaq_f32(sumSq, v, v);
    }
    float rms = sqrtf(vaddvq_f32(sumSq) / 32.0f);
    if (rms < 1e-10f)
        rms = 1e-10f;

    uint16_t scaleFp16 = tq3_float_to_fp16(rms);
    memcpy(dst, &scaleFp16, 2);

    // Normalize via NEON
    float normalized[32];
    float32x4_t invRmsV = vdupq_n_f32(1.0f / rms);
    for (int i = 0; i < 32; i += 4) {
        vst1q_f32(normalized + i, vmulq_f32(vld1q_f32(src + i), invRmsV));
    }

    // WHT forward
    float rotated[32];
    wht_forward_32_neon(rotated, normalized);

    // Codebook search: for TQ3 (7 boundaries), use vectorized comparison
    uint8_t indices[32];
    float32x4_t b0 = vdupq_n_f32(TQ3_BOUNDARIES[0]);
    float32x4_t b1 = vdupq_n_f32(TQ3_BOUNDARIES[1]);
    float32x4_t b2 = vdupq_n_f32(TQ3_BOUNDARIES[2]);
    float32x4_t b3 = vdupq_n_f32(TQ3_BOUNDARIES[3]);
    float32x4_t b4 = vdupq_n_f32(TQ3_BOUNDARIES[4]);
    float32x4_t b5 = vdupq_n_f32(TQ3_BOUNDARIES[5]);
    float32x4_t b6 = vdupq_n_f32(TQ3_BOUNDARIES[6]);
    for (int i = 0; i < 32; i += 4) {
        float32x4_t v = vld1q_f32(rotated + i);
        // Each vcgtq returns 0xFFFFFFFF (-1 as uint32) if v > boundary, else 0
        // Sum of negated masks = count of boundaries exceeded = index
        uint32x4_t idx = vdupq_n_u32(0);
        idx = vsubq_u32(idx, vcgtq_f32(v, b0));
        idx = vsubq_u32(idx, vcgtq_f32(v, b1));
        idx = vsubq_u32(idx, vcgtq_f32(v, b2));
        idx = vsubq_u32(idx, vcgtq_f32(v, b3));
        idx = vsubq_u32(idx, vcgtq_f32(v, b4));
        idx = vsubq_u32(idx, vcgtq_f32(v, b5));
        idx = vsubq_u32(idx, vcgtq_f32(v, b6));
        indices[i] = (uint8_t)vgetq_lane_u32(idx, 0);
        indices[i + 1] = (uint8_t)vgetq_lane_u32(idx, 1);
        indices[i + 2] = (uint8_t)vgetq_lane_u32(idx, 2);
        indices[i + 3] = (uint8_t)vgetq_lane_u32(idx, 3);
    }

    // Pack 3-bit indices
    for (int g = 0; g < 4; g++) {
        tq3_pack_3bit_8(dst + 2 + g * 3, indices + g * 8);
    }
}

static inline void tq4_quantize_block_neon(uint8_t* dst, const float* src) {
    // RMS
    float32x4_t sumSq = vdupq_n_f32(0.0f);
    for (int i = 0; i < 32; i += 4) {
        float32x4_t v = vld1q_f32(src + i);
        sumSq = vfmaq_f32(sumSq, v, v);
    }
    float rms = sqrtf(vaddvq_f32(sumSq) / 32.0f);
    if (rms < 1e-10f)
        rms = 1e-10f;

    uint16_t scaleFp16 = tq3_float_to_fp16(rms);
    memcpy(dst, &scaleFp16, 2);

    // Normalize
    float normalized[32];
    float32x4_t invRmsV = vdupq_n_f32(1.0f / rms);
    for (int i = 0; i < 32; i += 4) {
        vst1q_f32(normalized + i, vmulq_f32(vld1q_f32(src + i), invRmsV));
    }

    // WHT forward
    float rotated[32];
    wht_forward_32_neon(rotated, normalized);

    // TQ4 codebook search: 15 boundaries, binary search approach
    // Or just use linear comparison (15 vcgtq is still fast)
    uint8_t indices[32];
    float32x4_t bd[15];
    for (int b = 0; b < 15; b++)
        bd[b] = vdupq_n_f32(TQ4_BOUNDARIES[b]);
    for (int i = 0; i < 32; i += 4) {
        float32x4_t v = vld1q_f32(rotated + i);
        uint32x4_t idx = vdupq_n_u32(0);
        for (int b = 0; b < 15; b++) {
            idx = vsubq_u32(idx, vcgtq_f32(v, bd[b]));
        }
        indices[i] = (uint8_t)vgetq_lane_u32(idx, 0);
        indices[i + 1] = (uint8_t)vgetq_lane_u32(idx, 1);
        indices[i + 2] = (uint8_t)vgetq_lane_u32(idx, 2);
        indices[i + 3] = (uint8_t)vgetq_lane_u32(idx, 3);
    }

    for (int g = 0; g < 4; g++) {
        tq4_pack_4bit_8(dst + 2 + g * 4, indices + g * 8);
    }
}

// Override scalar versions with NEON
#undef tq3_wht_forward_32
#define tq3_wht_forward_32 wht_forward_32_neon
#undef tq3_wht_inverse_32
#define tq3_wht_inverse_32 wht_inverse_32_neon
#undef tq3_quantize_block
#define tq3_quantize_block tq3_quantize_block_neon
#undef tq4_quantize_block
#define tq4_quantize_block tq4_quantize_block_neon

#endif // __aarch64__

#endif // TURBOQUANT_HPP
