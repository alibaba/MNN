// Spacemit IME2 fused Attention kernels and TCM task runtime.
#include <riscv_vector.h>

#include <algorithm>
#include <cfloat>
#include <cstddef>
#include <cmath>
#include <cstring>

#include "backend/cpu/compute/CommonOptFunction.h"

static inline bool MNNSpacemitIme2AttentionSoftmaxEnabled() {
#if defined(MNN_USE_SPACEMIT_IME2)
    return true;
#else
    return false;
#endif
}

static inline vfloat32m1_t MNNRvvAttentionExp(vfloat32m1_t x, size_t vl) {
    constexpr float ln2 = 0.6931471805599453f;
    constexpr float invLn2 = 1.4426950408889634f;
    x = __riscv_vfmax_vf_f32m1(x, -87.0f, vl);
    x = __riscv_vfmin_vf_f32m1(x, 87.0f, vl);

    const vfloat32m1_t divFloat = __riscv_vfmul_vf_f32m1(x, invLn2, vl);
    const vint32m1_t divInt = __riscv_vfcvt_x_f_v_i32m1(divFloat, vl);
    const vfloat32m1_t div = __riscv_vfcvt_f_x_v_f32m1(divInt, vl);
    const vfloat32m1_t expBasic =
        __riscv_vreinterpret_v_i32m1_f32m1(__riscv_vsll_vx_i32m1(__riscv_vadd_vx_i32m1(divInt, 127, vl), 23, vl));

    const vfloat32m1_t remain = __riscv_vfnmsub_vf_f32m1(div, ln2, x, vl);
    const vfloat32m1_t t = __riscv_vfmul_vf_f32m1(remain, 0.25f, vl);
    vfloat32m1_t poly = __riscv_vfmv_v_f_f32m1(1.0f / 120.0f, vl);
    poly = __riscv_vfadd_vf_f32m1(__riscv_vfmul_vv_f32m1(poly, t, vl), 1.0f / 24.0f, vl);
    poly = __riscv_vfadd_vf_f32m1(__riscv_vfmul_vv_f32m1(poly, t, vl), 1.0f / 6.0f, vl);
    poly = __riscv_vfadd_vf_f32m1(__riscv_vfmul_vv_f32m1(poly, t, vl), 0.5f, vl);
    poly = __riscv_vfadd_vf_f32m1(__riscv_vfmul_vv_f32m1(poly, t, vl), 1.0f, vl);
    poly = __riscv_vfadd_vf_f32m1(__riscv_vfmul_vv_f32m1(poly, t, vl), 1.0f, vl);
    poly = __riscv_vfmul_vv_f32m1(poly, poly, vl);
    poly = __riscv_vfmul_vv_f32m1(poly, poly, vl);
    return __riscv_vfmul_vv_f32m1(expBasic, poly, vl);
}

static inline float MNNRvvAttentionExpScalar(float x) {
    constexpr float ln2 = 0.6931471805599453f;
    constexpr float invLn2 = 1.4426950408889634f;
    x = std::fmax(x, -87.0f);
    x = std::fmin(x, 87.0f);
    const int div = static_cast<int>(std::lrintf(x * invLn2));
    const int bits = (div + 127) << 23;
    float expBasic;
    std::memcpy(&expBasic, &bits, sizeof(expBasic));
    const float t = (x - div * ln2) * 0.25f;
    float poly = 1.0f / 120.0f;
    poly = poly * t + 1.0f / 24.0f;
    poly = poly * t + 1.0f / 6.0f;
    poly = poly * t + 0.5f;
    poly = poly * t + 1.0f;
    poly = poly * t + 1.0f;
    poly *= poly;
    poly *= poly;
    return expBasic * poly;
}

static inline void MNNRvvAttentionZeroGroups(float* dst, ptrdiff_t strideBytes, int groups) {
    if (groups <= 0) {
        return;
    }
    const size_t vl = __riscv_vsetvl_e32m1(groups);
    const vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    const vfloat32m1x4_t zeros = __riscv_vcreate_v_f32m1x4(zero, zero, zero, zero);
    __riscv_vssseg4e32_v_f32m1x4(dst, strideBytes, zeros, vl);
}

void MNNSpacemitIme2AttentionSoftmax(float* softmaxDst, const float* softmaxSrc, float* runningMax, float* runningSum,
                                     float* updateScale, int outside, int reduceSize, int kvSeqOffset, int validOffset,
                                     int pack, bool mask) {
    const int totalGroups = (reduceSize + 3) / 4;
    if (!MNNSpacemitIme2AttentionSoftmaxEnabled() || __riscv_vlenb() != 128 || pack != 4 || reduceSize <= 0 ||
        (reduceSize % 4) != 0 || runningMax == nullptr || runningSum == nullptr || updateScale == nullptr ||
        __riscv_vsetvl_e32m1(totalGroups) != (size_t)totalGroups) {
        MNNSoftmax(softmaxDst, softmaxSrc, runningMax, runningSum, updateScale, outside, reduceSize, kvSeqOffset,
                   validOffset, pack, mask);
        return;
    }

    const int stride = outside * 4;
    const ptrdiff_t strideBytes = (ptrdiff_t)stride * sizeof(float);
    for (int row = 0; row < outside; ++row) {
        float* dstBase = softmaxDst + row * 4;
        const float* srcBase = softmaxSrc + row * 4;
        if (mask && kvSeqOffset > row + validOffset) {
            updateScale[row] = 1.0f;
            MNNRvvAttentionZeroGroups(dstBase, strideBytes, totalGroups);
            continue;
        }

        const int validSize = mask ? std::min(reduceSize, row + validOffset + 1 - kvSeqOffset) : reduceSize;
        const int fullGroups = validSize / 4;
        const int tail = validSize % 4;
        float newMax = -FLT_MAX;

        vfloat32m1x4_t source;
        if (fullGroups > 0) {
            const size_t vl = fullGroups;
            source = __riscv_vlsseg4e32_v_f32m1x4(srcBase, strideBytes, vl);
            vfloat32m1_t maximum = __riscv_vget_v_f32m1x4_f32m1(source, 0);
            maximum = __riscv_vfmax_vv_f32m1(maximum, __riscv_vget_v_f32m1x4_f32m1(source, 1), vl);
            maximum = __riscv_vfmax_vv_f32m1(maximum, __riscv_vget_v_f32m1x4_f32m1(source, 2), vl);
            maximum = __riscv_vfmax_vv_f32m1(maximum, __riscv_vget_v_f32m1x4_f32m1(source, 3), vl);
            const vfloat32m1_t seed = __riscv_vfmv_s_f_f32m1(-FLT_MAX, 1);
            const vfloat32m1_t reduced = __riscv_vfredmax_vs_f32m1_f32m1(maximum, seed, vl);
            newMax = __riscv_vfmv_f_s_f32m1_f32(reduced);
        }
        const float* tailSrc = srcBase + fullGroups * stride;
        for (int i = 0; i < tail; ++i) {
            newMax = std::fmax(newMax, tailSrc[i]);
        }

        const float oldMax = runningMax[row];
        const float finalMax = std::fmax(oldMax, newMax);
        float sum = 0.0f;
        if (fullGroups > 0) {
            const size_t vl = fullGroups;
            const vfloat32m1_t exp0 =
                MNNRvvAttentionExp(__riscv_vfsub_vf_f32m1(__riscv_vget_v_f32m1x4_f32m1(source, 0), finalMax, vl), vl);
            const vfloat32m1_t exp1 =
                MNNRvvAttentionExp(__riscv_vfsub_vf_f32m1(__riscv_vget_v_f32m1x4_f32m1(source, 1), finalMax, vl), vl);
            const vfloat32m1_t exp2 =
                MNNRvvAttentionExp(__riscv_vfsub_vf_f32m1(__riscv_vget_v_f32m1x4_f32m1(source, 2), finalMax, vl), vl);
            const vfloat32m1_t exp3 =
                MNNRvvAttentionExp(__riscv_vfsub_vf_f32m1(__riscv_vget_v_f32m1x4_f32m1(source, 3), finalMax, vl), vl);
            const vfloat32m1x4_t result = __riscv_vcreate_v_f32m1x4(exp0, exp1, exp2, exp3);
            __riscv_vssseg4e32_v_f32m1x4(dstBase, strideBytes, result, vl);

            vfloat32m1_t laneSum = __riscv_vfadd_vv_f32m1(exp0, exp1, vl);
            laneSum = __riscv_vfadd_vv_f32m1(laneSum, exp2, vl);
            laneSum = __riscv_vfadd_vv_f32m1(laneSum, exp3, vl);
            const vfloat32m1_t seed = __riscv_vfmv_s_f_f32m1(0.0f, 1);
            const vfloat32m1_t reduced = __riscv_vfredusum_vs_f32m1_f32m1(laneSum, seed, vl);
            sum = __riscv_vfmv_f_s_f32m1_f32(reduced);
        }

        float* tailDst = dstBase + fullGroups * stride;
        for (int i = 0; i < tail; ++i) {
            const float value = MNNRvvAttentionExpScalar(tailSrc[i] - finalMax);
            tailDst[i] = value;
            sum += value;
        }
        for (int i = tail; i < 4 && tail > 0; ++i) {
            tailDst[i] = 0.0f;
        }

        const int validGroups = fullGroups + (tail > 0 ? 1 : 0);
        MNNRvvAttentionZeroGroups(dstBase + validGroups * stride, strideBytes, totalGroups - validGroups);

        const float scaleForSum = std::exp(oldMax - finalMax);
        runningSum[row] = runningSum[row] * scaleForSum + sum;
        runningMax[row] = finalMax;
        updateScale[row] = scaleForSum;
    }
}

#if defined(__riscv_zvfh)
extern "C" int MNNSpacemitIme2FlashAttentionFp32C4Supported() {
    return 1;
}

static inline vfloat32m2_t MNNRvvFlashExp(vfloat32m2_t x, size_t vl) {
    x = __riscv_vfmax_vf_f32m2(x, -87.0f, vl);
    const vfloat32m2_t r = __riscv_vfmv_v_f_f32m2(0x1.8p23f, vl);
    const vfloat32m2_t z = __riscv_vfmacc_vf_f32m2(r, 0x1.715476p+0f, x, vl);
    const vfloat32m2_t n = __riscv_vfsub_vv_f32m2(z, r, vl);
    const vfloat32m2_t b =
        __riscv_vfnmsac_vf_f32m2(__riscv_vfnmsac_vf_f32m2(x, 0x1.62e4p-1f, n, vl), 0x1.7f7d1cp-20f, n, vl);
    const vuint32m2_t e = __riscv_vsll_vx_u32m2(__riscv_vreinterpret_v_f32m2_u32m2(z), 23, vl);
    const vfloat32m2_t k = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vadd_vx_u32m2(e, 0x3f800000, vl));
    const vfloat32m2_t u = __riscv_vfmul_vv_f32m2(b, b, vl);
    const vfloat32m2_t j = __riscv_vfmacc_vv_f32m2(
        __riscv_vfmul_vf_f32m2(b, 0x1.ffffecp-1f, vl),
        __riscv_vfmacc_vv_f32m2(
            __riscv_vfmacc_vf_f32m2(__riscv_vfmv_v_f_f32m2(0x1.fffdb6p-2f, vl), 0x1.555e66p-3f, b, vl),
            __riscv_vfmacc_vf_f32m2(__riscv_vfmv_v_f_f32m2(0x1.573e2ep-5f, vl), 0x1.0e4020p-7f, b, vl), u, vl),
        u, vl);
    return __riscv_vfmacc_vv_f32m2(k, j, k, vl);
}

static inline float MNNRvvFlashMax(const float* src, int size) {
    vfloat32m1_t maximum = __riscv_vfmv_s_f_f32m1(-FLT_MAX, 1);
    int offset = 0;
    while (offset < size) {
        const size_t vl = __riscv_vsetvl_e32m4(size - offset);
        const vfloat32m4_t values = __riscv_vle32_v_f32m4(src + offset, vl);
        maximum = __riscv_vfredmax_vs_f32m4_f32m1(values, maximum, vl);
        offset += vl;
    }
    return __riscv_vfmv_f_s_f32m1_f32(maximum);
}

static inline float MNNRvvFlashSoftmax(float* data, int validSize, int fullSize, float maximum) {
    float sum = 0.0f;
    int offset = 0;
    while (offset < validSize) {
        const size_t vl = __riscv_vsetvl_e32m2(validSize - offset);
        vfloat32m2_t values = __riscv_vle32_v_f32m2(data + offset, vl);
        values = __riscv_vfsub_vf_f32m2(values, maximum, vl);
        values = MNNRvvFlashExp(values, vl);
        __riscv_vse32_v_f32m2(data + offset, values, vl);
        const vfloat32m1_t seed = __riscv_vfmv_s_f_f32m1(0.0f, 1);
        const vfloat32m1_t reduced = __riscv_vfredusum_vs_f32m2_f32m1(values, seed, vl);
        sum += __riscv_vfmv_f_s_f32m1_f32(reduced);
        offset += vl;
    }
    if (validSize < fullSize) {
        std::memset(data + validSize, 0, (fullSize - validSize) * sizeof(float));
    }
    return sum;
}

static inline void MNNRvvFlashScale(float* data, float scale, int size) {
    int offset = 0;
    while (offset < size) {
        const size_t vl = __riscv_vsetvl_e32m4(size - offset);
        vfloat32m4_t values = __riscv_vle32_v_f32m4(data + offset, vl);
        values = __riscv_vfmul_vf_f32m4(values, scale, vl);
        __riscv_vse32_v_f32m4(data + offset, values, vl);
        offset += vl;
    }
}

static inline void MNNRvvFlashPackQ(_Float16* dst, const float* src, int srcStride, int rows, int dim, float scale) {
    for (int row = 0; row < rows; ++row) {
        int offset = 0;
        while (offset < dim) {
            const size_t vl = __riscv_vsetvl_e32m4(dim - offset);
            vfloat32m4_t values = __riscv_vle32_v_f32m4(src + row * srcStride + offset, vl);
            values = __riscv_vfmul_vf_f32m4(values, scale, vl);
            const vfloat16m2_t half = __riscv_vfncvt_f_f_w_f16m2(values, vl);
            __riscv_vse16_v_f16m2(dst + row * dim + offset, half, vl);
            offset += vl;
        }
    }
}

static inline void MNNRvvFlashPackK(_Float16* dst, const float* src, int dim) {
    constexpr int kvTile = 64;
    constexpr int tokenGroups = kvTile / 4;
    const size_t vl = __riscv_vsetvl_e32m1(tokenGroups);
    const ptrdiff_t srcStride = (ptrdiff_t)dim * 4 * sizeof(float);
    for (int d = 0; d < dim; ++d) {
        const vfloat32m1x4_t values = __riscv_vlsseg4e32_v_f32m1x4(src + d * 4, srcStride, vl);
        const vfloat16mf2_t half0 = __riscv_vfncvt_f_f_w_f16mf2(__riscv_vget_v_f32m1x4_f32m1(values, 0), vl);
        const vfloat16mf2_t half1 = __riscv_vfncvt_f_f_w_f16mf2(__riscv_vget_v_f32m1x4_f32m1(values, 1), vl);
        const vfloat16mf2_t half2 = __riscv_vfncvt_f_f_w_f16mf2(__riscv_vget_v_f32m1x4_f32m1(values, 2), vl);
        const vfloat16mf2_t half3 = __riscv_vfncvt_f_f_w_f16mf2(__riscv_vget_v_f32m1x4_f32m1(values, 3), vl);
        __riscv_vsse16_v_f16mf2(dst + d * kvTile + 0, 4 * sizeof(_Float16), half0, vl);
        __riscv_vsse16_v_f16mf2(dst + d * kvTile + 1, 4 * sizeof(_Float16), half1, vl);
        __riscv_vsse16_v_f16mf2(dst + d * kvTile + 2, 4 * sizeof(_Float16), half2, vl);
        __riscv_vsse16_v_f16mf2(dst + d * kvTile + 3, 4 * sizeof(_Float16), half3, vl);
    }
}

static inline void MNNRvvFlashPackV(_Float16* dst, const float* src, int dim) {
    constexpr int kvTile = 64;
    const size_t vl = __riscv_vsetvl_e32m2(kvTile);
    for (int d = 0; d < dim; d += 4) {
        const vfloat32m2x4_t values = __riscv_vlseg4e32_v_f32m2x4(src + d * kvTile, vl);
        const vfloat16m1_t half0 = __riscv_vfncvt_f_f_w_f16m1(__riscv_vget_v_f32m2x4_f32m2(values, 0), vl);
        const vfloat16m1_t half1 = __riscv_vfncvt_f_f_w_f16m1(__riscv_vget_v_f32m2x4_f32m2(values, 1), vl);
        const vfloat16m1_t half2 = __riscv_vfncvt_f_f_w_f16m1(__riscv_vget_v_f32m2x4_f32m2(values, 2), vl);
        const vfloat16m1_t half3 = __riscv_vfncvt_f_f_w_f16m1(__riscv_vget_v_f32m2x4_f32m2(values, 3), vl);
        __riscv_vsse16_v_f16m1(dst + d + 0, dim * sizeof(_Float16), half0, vl);
        __riscv_vsse16_v_f16m1(dst + d + 1, dim * sizeof(_Float16), half1, vl);
        __riscv_vsse16_v_f16m1(dst + d + 2, dim * sizeof(_Float16), half2, vl);
        __riscv_vsse16_v_f16m1(dst + d + 3, dim * sizeof(_Float16), half3, vl);
    }
}

static inline void MNNRvvFlashQK4(float* dst0, float* dst1, float* dst2, float* dst3, const _Float16* q0,
                                  const _Float16* q1, const _Float16* q2, const _Float16* q3, const _Float16* key,
                                  int dim, int tokenCount) {
    constexpr int kvTile = 64;
    const size_t vl = __riscv_vsetvl_e16m1(tokenCount);
    vfloat32m2_t acc0 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
    vfloat32m2_t acc1 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
    vfloat32m2_t acc2 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
    vfloat32m2_t acc3 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
    for (int d = 0; d < dim; ++d) {
        const vfloat16m1_t keyValues = __riscv_vle16_v_f16m1(key + d * kvTile, vl);
        acc0 = __riscv_vfwmacc_vf_f32m2(acc0, q0[d], keyValues, vl);
        acc1 = __riscv_vfwmacc_vf_f32m2(acc1, q1[d], keyValues, vl);
        acc2 = __riscv_vfwmacc_vf_f32m2(acc2, q2[d], keyValues, vl);
        acc3 = __riscv_vfwmacc_vf_f32m2(acc3, q3[d], keyValues, vl);
    }
    __riscv_vse32_v_f32m2(dst0, acc0, vl);
    __riscv_vse32_v_f32m2(dst1, acc1, vl);
    __riscv_vse32_v_f32m2(dst2, acc2, vl);
    __riscv_vse32_v_f32m2(dst3, acc3, vl);
}

static inline void MNNRvvFlashPV4(float* dst0, float* dst1, float* dst2, float* dst3, const float* prob0,
                                  const float* prob1, const float* prob2, const float* prob3, const _Float16* value,
                                  int dim, int tokenCount) {
    int d = 0;
    while (d < dim) {
        const size_t vl = __riscv_vsetvl_e16m2(dim - d);
        vfloat32m4_t acc0 = __riscv_vle32_v_f32m4(dst0 + d, vl);
        vfloat32m4_t acc1 = __riscv_vle32_v_f32m4(dst1 + d, vl);
        vfloat32m4_t acc2 = __riscv_vle32_v_f32m4(dst2 + d, vl);
        vfloat32m4_t acc3 = __riscv_vle32_v_f32m4(dst3 + d, vl);
        for (int token = 0; token < tokenCount; ++token) {
            const vfloat16m2_t value16 = __riscv_vle16_v_f16m2(value + token * dim + d, vl);
            const vfloat32m4_t value32 = __riscv_vfwcvt_f_f_v_f32m4(value16, vl);
            acc0 = __riscv_vfmacc_vf_f32m4(acc0, prob0[token], value32, vl);
            acc1 = __riscv_vfmacc_vf_f32m4(acc1, prob1[token], value32, vl);
            acc2 = __riscv_vfmacc_vf_f32m4(acc2, prob2[token], value32, vl);
            acc3 = __riscv_vfmacc_vf_f32m4(acc3, prob3[token], value32, vl);
        }
        __riscv_vse32_v_f32m4(dst0 + d, acc0, vl);
        __riscv_vse32_v_f32m4(dst1 + d, acc1, vl);
        __riscv_vse32_v_f32m4(dst2 + d, acc2, vl);
        __riscv_vse32_v_f32m4(dst3 + d, acc3, vl);
        d += vl;
    }
}

static inline void MNNRvvFlashStoreC4(float* dst, const float* src, const float* reciprocal, int queryBase, int seqLen,
                                      int dim) {
    constexpr int queryTile = 64;
    const size_t vl = __riscv_vsetvl_e32m2(queryTile);
    const ptrdiff_t srcStride = (ptrdiff_t)dim * sizeof(float);
    const vfloat32m2_t reciprocalValue = __riscv_vle32_v_f32m2(reciprocal, vl);
    for (int d = 0; d < dim; d += 4) {
        const vfloat32m2_t value0 =
            __riscv_vfmul_vv_f32m2(__riscv_vlse32_v_f32m2(src + d + 0, srcStride, vl), reciprocalValue, vl);
        const vfloat32m2_t value1 =
            __riscv_vfmul_vv_f32m2(__riscv_vlse32_v_f32m2(src + d + 1, srcStride, vl), reciprocalValue, vl);
        const vfloat32m2_t value2 =
            __riscv_vfmul_vv_f32m2(__riscv_vlse32_v_f32m2(src + d + 2, srcStride, vl), reciprocalValue, vl);
        const vfloat32m2_t value3 =
            __riscv_vfmul_vv_f32m2(__riscv_vlse32_v_f32m2(src + d + 3, srcStride, vl), reciprocalValue, vl);
        const vfloat32m2x4_t values = __riscv_vcreate_v_f32m2x4(value0, value1, value2, value3);
        __riscv_vsseg4e32_v_f32m2x4(dst + (d / 4) * seqLen * 4 + queryBase * 4, values, vl);
    }
}

static inline void MNNRvvFlashRunCausalBlock(const _Float16* query, const _Float16* key, const _Float16* value,
                                             float* score, float* output, float* maximum, float* sum, int queryBase,
                                             int kvBase, int dim) {
    constexpr int queryTile = 64;
    constexpr int kvTile = 64;
    for (int row = 0; row < queryTile; row += 4) {
        const int tokenCount = std::min(kvTile, queryBase + row + 4 - kvBase);
        MNNRvvFlashQK4(score + (row + 0) * kvTile, score + (row + 1) * kvTile, score + (row + 2) * kvTile,
                       score + (row + 3) * kvTile, query + (row + 0) * dim, query + (row + 1) * dim,
                       query + (row + 2) * dim, query + (row + 3) * dim, key, dim, tokenCount);
    }

    for (int row = 0; row < queryTile; ++row) {
        const int validSize = std::min(kvTile, queryBase + row - kvBase + 1);
        const int tokenCount = std::min(kvTile, queryBase + (row & ~3) + 4 - kvBase);
        if (validSize <= 0) {
            continue;
        }
        float* rowScore = score + row * kvTile;
        const float tileMaximum = MNNRvvFlashMax(rowScore, validSize);
        const float newMaximum = std::fmax(maximum[row], tileMaximum);
        if (sum[row] > 0.0f && newMaximum > maximum[row]) {
            const float update = std::exp(maximum[row] - newMaximum);
            MNNRvvFlashScale(output + row * dim, update, dim);
            sum[row] *= update;
        }
        maximum[row] = newMaximum;
        sum[row] += MNNRvvFlashSoftmax(rowScore, validSize, tokenCount, newMaximum);
    }

    for (int row = 0; row < queryTile; row += 4) {
        const int tokenCount = std::min(kvTile, queryBase + row + 4 - kvBase);
        MNNRvvFlashPV4(output + (row + 0) * dim, output + (row + 1) * dim, output + (row + 2) * dim,
                       output + (row + 3) * dim, score + (row + 0) * kvTile, score + (row + 1) * kvTile,
                       score + (row + 2) * kvTile, score + (row + 3) * kvTile, value, dim, tokenCount);
    }
}

extern "C" int MNNSpacemitIme2FlashAttentionFp32C4(float* dst, const float* query, const float* key, const float* value,
                                                   int seqLen, int numHeads, int headDim, float scale, void* qkvScratch,
                                                   size_t qkvScratchBytes, float* scoreScratch, size_t scoreCount,
                                                   float* outputScratch, size_t outputCount) {
    constexpr int queryTile = 64;
    constexpr int kvTile = 64;
    const size_t packedTileBytes = queryTile * headDim * sizeof(_Float16);
    if (dst == nullptr || query == nullptr || key == nullptr || value == nullptr || qkvScratch == nullptr ||
        scoreScratch == nullptr || outputScratch == nullptr) {
        return -1;
    }
    if (__riscv_vlenb() != 128) {
        return -2;
    }
    if (seqLen != 128 || headDim != 128) {
        return -3;
    }
    if (qkvScratchBytes < 3 * packedTileBytes) {
        return -4;
    }
    if (scoreCount < queryTile * kvTile) {
        return -5;
    }
    if (outputCount < queryTile * headDim) {
        return -6;
    }

    _Float16* query16 = reinterpret_cast<_Float16*>(qkvScratch);
    _Float16* key16 = query16 + queryTile * headDim;
    _Float16* value16 = key16 + kvTile * headDim;
    float maximum[queryTile];
    float sum[queryTile];

    for (int queryBase = 0; queryBase < seqLen; queryBase += queryTile) {
        MNNRvvFlashPackQ(query16, query + queryBase * numHeads * headDim, numHeads * headDim, queryTile, headDim,
                         scale);
        std::memset(outputScratch, 0, queryTile * headDim * sizeof(float));
        for (int row = 0; row < queryTile; ++row) {
            maximum[row] = -FLT_MAX;
            sum[row] = 0.0f;
        }

        for (int kvBase = 0; kvBase < seqLen; kvBase += kvTile) {
            if (kvBase > queryBase + queryTile - 1) {
                continue;
            }
            MNNRvvFlashPackK(key16, key + (kvBase / 4) * headDim * 4, headDim);
            MNNRvvFlashPackV(value16, value + (kvBase / kvTile) * headDim * kvTile, headDim);
            MNNRvvFlashRunCausalBlock(query16, key16, value16, scoreScratch, outputScratch, maximum, sum, queryBase,
                                      kvBase, headDim);
        }

        for (int row = 0; row < queryTile; ++row) {
            sum[row] = sum[row] > 0.0f ? 1.0f / sum[row] : 0.0f;
        }
        MNNRvvFlashStoreC4(dst, outputScratch, sum, queryBase, seqLen, headDim);
    }
    return 1;
}

// Keep the K3 pair-head hot loop isolated and cache-line aligned to stabilize its instruction layout.
static __attribute__((hot, noinline, aligned(64))) int
MNNSpacemitIme2FlashAttentionFp32C4PairBaseline(float* dst, const float* query, const float* key, const float* value,
                                                int seqLen, int numHeads, int headDim, float scale, void* scratch,
                                                size_t scratchBytes) {
    constexpr int queryTile = 64;
    constexpr int kvTile = 64;
    constexpr int pairCount = 2;
    const size_t halfTile = queryTile * headDim;
    const size_t kvBlockNums = (seqLen + kvTile - 1) / kvTile;
    const size_t requiredBytes = ((2 * kvBlockNums + pairCount) * halfTile) * sizeof(_Float16) +
                                 (queryTile * kvTile + pairCount * queryTile * headDim) * sizeof(float);
    if (dst == nullptr || query == nullptr || key == nullptr || value == nullptr || scratch == nullptr) {
        return -1;
    }
    if (__riscv_vlenb() != 128) {
        return -2;
    }
    if (seqLen < queryTile || seqLen > 512 || (seqLen % queryTile) != 0 || headDim != 128 ||
        scratchBytes < requiredBytes) {
        return -3;
    }

    _Float16* key16 = reinterpret_cast<_Float16*>(scratch);
    _Float16* value16 = key16 + kvBlockNums * halfTile;
    _Float16* query16 = value16 + kvBlockNums * halfTile;
    float* score = reinterpret_cast<float*>(query16 + pairCount * halfTile);
    float* output = score + queryTile * kvTile;
    float maximum[pairCount][queryTile];
    float sum[pairCount][queryTile];

    for (int kvBase = 0; kvBase < seqLen; kvBase += kvTile) {
        const int block = kvBase / kvTile;
        MNNRvvFlashPackK(key16 + block * halfTile, key + (kvBase / 4) * headDim * 4, headDim);
        MNNRvvFlashPackV(value16 + block * halfTile, value + block * headDim * kvTile, headDim);
    }

    for (int queryBase = 0; queryBase < seqLen; queryBase += queryTile) {
        for (int head = 0; head < pairCount; ++head) {
            MNNRvvFlashPackQ(query16 + head * halfTile, query + head * headDim + queryBase * numHeads * headDim,
                             numHeads * headDim, queryTile, headDim, scale);
            std::memset(output + head * queryTile * headDim, 0, queryTile * headDim * sizeof(float));
            for (int row = 0; row < queryTile; ++row) {
                maximum[head][row] = -FLT_MAX;
                sum[head][row] = 0.0f;
            }
        }

        for (int kvBase = 0; kvBase < seqLen; kvBase += kvTile) {
            if (kvBase > queryBase + queryTile - 1) {
                continue;
            }
            const int block = kvBase / kvTile;
            for (int head = 0; head < pairCount; ++head) {
                MNNRvvFlashRunCausalBlock(query16 + head * halfTile, key16 + block * halfTile,
                                          value16 + block * halfTile, score, output + head * queryTile * headDim,
                                          maximum[head], sum[head], queryBase, kvBase, headDim);
            }
        }

        for (int head = 0; head < pairCount; ++head) {
            for (int row = 0; row < queryTile; ++row) {
                sum[head][row] = sum[head][row] > 0.0f ? 1.0f / sum[head][row] : 0.0f;
            }
            MNNRvvFlashStoreC4(dst + head * seqLen * headDim, output + head * queryTile * headDim, sum[head], queryBase,
                               seqLen, headDim);
        }
    }
    return 1;
}

extern "C" int MNNSpacemitIme2FlashAttentionFp32C4Pair(float* dst, const float* query, const float* key,
                                                       const float* value, int seqLen, int numHeads, int headDim,
                                                       float scale, void* scratch, size_t scratchBytes) {
    return MNNSpacemitIme2FlashAttentionFp32C4PairBaseline(dst, query, key, value, seqLen, numHeads, headDim, scale,
                                                           scratch, scratchBytes);
}
#else
extern "C" int MNNSpacemitIme2FlashAttentionFp32C4Supported() {
    return 0;
}

extern "C" int MNNSpacemitIme2FlashAttentionFp32C4(float*, const float*, const float*, const float*, int, int, int,
                                                   float, void*, size_t, float*, size_t, float*, size_t) {
    return 0;
}

extern "C" int MNNSpacemitIme2FlashAttentionFp32C4Pair(float*, const float*, const float*, const float*, int, int, int,
                                                       float, void*, size_t) {
    return 0;
}
#endif
