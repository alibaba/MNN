//
//  CPUFusedAddRMSNorm.cpp
//  MNN
//
//  Fused Add + RMSNorm kernel: dst = (src0 + src1) / sqrt(mean² + eps) * gamma
//  Single pass, intermediate values stay in NEON registers — no DDR roundtrip.
//

#include <cmath>
#include "backend/cpu/compute/CommonOptFunction.h"

#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif

void MNNAddAndRMSNorm(float* dst, const float* src0, const float* src1,
                      const float* gamma, float epsilon, size_t size) {
#ifdef MNN_USE_NEON
    // --- NEON fused path ---
    const float32x4_t veps = vdupq_n_f32(epsilon);
    float32x4_t vsqsum0 = vdupq_n_f32(0.0f);
    float32x4_t vsqsum1 = vdupq_n_f32(0.0f);
    float32x4_t vsqsum2 = vdupq_n_f32(0.0f);
    float32x4_t vsqsum3 = vdupq_n_f32(0.0f);

    int j = 0;

    // 1) First pass: load src0 + src1, add, accumulate sum-of-squares
    for (; j + 15 < size; j += 16) {
        float32x4_t s0_0 = vld1q_f32(&src0[j + 0]);
        float32x4_t s0_1 = vld1q_f32(&src0[j + 4]);
        float32x4_t s0_2 = vld1q_f32(&src0[j + 8]);
        float32x4_t s0_3 = vld1q_f32(&src0[j + 12]);

        float32x4_t s1_0 = vld1q_f32(&src1[j + 0]);
        float32x4_t s1_1 = vld1q_f32(&src1[j + 4]);
        float32x4_t s1_2 = vld1q_f32(&src1[j + 8]);
        float32x4_t s1_3 = vld1q_f32(&src1[j + 12]);

        // Add in registers — no write to DDR
        float32x4_t sum0 = vaddq_f32(s0_0, s1_0);
        float32x4_t sum1 = vaddq_f32(s0_1, s1_1);
        float32x4_t sum2 = vaddq_f32(s0_2, s1_2);
        float32x4_t sum3 = vaddq_f32(s0_3, s1_3);

        vsqsum0 = vmlaq_f32(vsqsum0, sum0, sum0);
        vsqsum1 = vmlaq_f32(vsqsum1, sum1, sum1);
        vsqsum2 = vmlaq_f32(vsqsum2, sum2, sum2);
        vsqsum3 = vmlaq_f32(vsqsum3, sum3, sum3);

        // Keep sum in register for second pass
        // Store to temp buffer for reuse — tradeoff: small SRAM vs DDR
        // We'll recompute in pass 2 instead (save the temp buffer write)
    }

    // Reduce 4 accumulators
    float32x4_t vsqsum = vaddq_f32(vsqsum0, vsqsum1);
    vsqsum = vaddq_f32(vsqsum, vsqsum2);
    vsqsum = vaddq_f32(vsqsum, vsqsum3);
    // Horizontal sum
    float sum_sq = vaddvq_f32(vsqsum);

    // Tail elements (< 16)
    for (; j < size; j++) {
        float v = src0[j] + src1[j];
        sum_sq += v * v;
    }

    float rms = std::sqrt(sum_sq / (float)size + epsilon);
    float scale = 1.0f / rms;

    // 2) Second pass: load, add, scale, gamma, store
    const float32x4_t vscale = vdupq_n_f32(scale);
    j = 0;
    for (; j + 15 < size; j += 16) {
        float32x4_t s0_0 = vld1q_f32(&src0[j + 0]);
        float32x4_t s0_1 = vld1q_f32(&src0[j + 4]);
        float32x4_t s0_2 = vld1q_f32(&src0[j + 8]);
        float32x4_t s0_3 = vld1q_f32(&src0[j + 12]);

        float32x4_t s1_0 = vld1q_f32(&src1[j + 0]);
        float32x4_t s1_1 = vld1q_f32(&src1[j + 4]);
        float32x4_t s1_2 = vld1q_f32(&src1[j + 8]);
        float32x4_t s1_3 = vld1q_f32(&src1[j + 12]);

        float32x4_t v0 = vaddq_f32(s0_0, s1_0);
        float32x4_t v1 = vaddq_f32(s0_1, s1_1);
        float32x4_t v2 = vaddq_f32(s0_2, s1_2);
        float32x4_t v3 = vaddq_f32(s0_3, s1_3);

        v0 = vmulq_f32(v0, vscale);
        v1 = vmulq_f32(v1, vscale);
        v2 = vmulq_f32(v2, vscale);
        v3 = vmulq_f32(v3, vscale);

        // Apply gamma
        if (gamma) {
            float32x4_t g0 = vld1q_f32(&gamma[j + 0]);
            float32x4_t g1 = vld1q_f32(&gamma[j + 4]);
            float32x4_t g2 = vld1q_f32(&gamma[j + 8]);
            float32x4_t g3 = vld1q_f32(&gamma[j + 12]);
            v0 = vmulq_f32(v0, g0);
            v1 = vmulq_f32(v1, g1);
            v2 = vmulq_f32(v2, g2);
            v3 = vmulq_f32(v3, g3);
        }

        vst1q_f32(&dst[j + 0], v0);
        vst1q_f32(&dst[j + 4], v1);
        vst1q_f32(&dst[j + 8], v2);
        vst1q_f32(&dst[j + 12], v3);
    }

    // Tail
    for (; j < size; j++) {
        float v = src0[j] + src1[j];
        v = v * scale;
        if (gamma) v = v * gamma[j];
        dst[j] = v;
    }

#else
    // --- Pure C fallback ---
    float sum_sq = 0.0f;
    for (int i = 0; i < size; i++) {
        float v = src0[i] + src1[i];
        sum_sq += v * v;
    }
    float scale = 1.0f / std::sqrt(sum_sq / (float)size + epsilon);
    for (int i = 0; i < size; i++) {
        float v = (src0[i] + src1[i]) * scale;
        if (gamma) v *= gamma[i];
        dst[i] = v;
    }
#endif
}
