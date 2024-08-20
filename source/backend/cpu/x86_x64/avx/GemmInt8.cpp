//
//  GemmInt8.cpp
//  MNN
//
//  Created by MNN on 2020/09/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "GemmCommon.hpp"
#include "FunctionSummary.hpp"
#include "core/Macro.h"
#include <math.h>
#define AVX2_PACKINT8 8
#define GEMMINT8_AVX2_E 4
#define GEMMINT8_AVX2_L 4
#define GEMMINT8_AVX2_H 8
namespace {
static inline __m128i mm_loadu_si128(const void* addr) {
    return _mm_loadu_si128((__m128i const*)addr);
}
static inline void MNN__mm_storeu_si64(void* add, __m128i value) {
    float temp[4];
    _mm_storeu_ps(temp, _mm_castsi128_ps(value));
    ::memcpy(add, temp, sizeof(int64_t));
}
}  // namespace

#define POSTTREAT(N) \
f##N = _mm256_min_ps(f##N, maxValue);\
f##N = _mm256_max_ps(f##N, minValue);\
auto m##N = _mm256_cmp_ps(f##N, zero128, 1);\
m##N = _mm256_blendv_ps(plus, minus, m##N);\
f##N = _mm256_add_ps(f##N, m##N);\
D##N = _mm256_cvtps_epi32(_mm256_round_ps(f##N, 3));\
D##N = _mm256_add_epi32(D##N, offset);\
D##N = _mm256_packs_epi32(D##N, _mm256_castps_si256(_mm256_permute2f128_ps(_mm256_castsi256_ps(D##N), _mm256_castsi256_ps(D##N), 1)));\
auto d##N = _mm_packus_epi16(_mm256_castsi256_si128(D##N), _mm256_castsi256_si128(_mm256_castps_si256(zero128)));\
MNN__mm_storeu_si64(dst_x + N * 8, d##N);


inline __m256i NORMAL_HADD(__m256i x, __m256i y) {
auto c0 = _mm256_castps_si256(_mm256_permute2f128_ps(_mm256_castsi256_ps(x), _mm256_castsi256_ps(y), 32));
auto c1 = _mm256_castps_si256(_mm256_permute2f128_ps(_mm256_castsi256_ps(x), _mm256_castsi256_ps(y), 49));
return _mm256_hadd_epi32(c0, c1);
}

#define EXTRACT_ADD(i)\
auto d##i##0 = _mm_castps_si128(_mm256_extractf128_ps(_mm256_castsi256_ps(D##i), 0));\
auto d##i##1 = _mm_castps_si128(_mm256_extractf128_ps(_mm256_castsi256_ps(D##i), 1));\
auto d##i = _mm_add_epi32(d##i##0, d##i##1);
#define COMPUTE(u, v)\
D##u##v = _mm256_add_epi32(D##u##v, _mm256_madd_epi16(W##u, S##v));

#define LOAD_INT4_TO_INT8 \
auto w_int4 = _mm_loadu_si128((__m128i const*)weight_sz);\
auto w_int4_high = _mm_and_si128(mask, _mm_srli_epi16(w_int4, 4));\
auto w_int4_low = _mm_and_si128(mask, w_int4);\
auto w_0 = _mm_unpacklo_epi8(w_int4_high, w_int4_low);\
auto w_1 = _mm_unpackhi_epi8(w_int4_high, w_int4_low);

void _AVX_MNNGemmInt8AddBiasScale_16x4_w4(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad, const QuanPostTreatParameters* post, size_t realDst) {
    MNN_ASSERT(post->useInt8==0);
    const auto dst_step_tmp = dst_step / sizeof(int8_t);
    auto zero128 = _mm256_set1_ps(0.0f);
    auto minValue = _mm256_set1_ps(post->minValue);
    auto maxValue = _mm256_set1_ps(post->maxValue);
    auto offset = _mm256_set1_epi32(128);
    __m256 fp32min, fp32max;
    if (post->fp32minmax) {
        fp32min = _mm256_set1_ps((post->fp32minmax)[0]);
        fp32max = _mm256_set1_ps((post->fp32minmax)[1]);
    }
    int blockNum = post->blockNum;
    const float* biasPtr = nullptr;
    if (post->biasFloat) {
        biasPtr = post->biasFloat;
    }

    int weight_step_Z = 0.5 * blockNum * src_depth_quad * (GEMMINT8_AVX2_L * GEMMINT8_AVX2_H);
    int weight_step_Y = 0.5 * (GEMMINT8_AVX2_L * GEMMINT8_AVX2_H);
    const __m128i mask = _mm_set1_epi8(0xf);
    
    auto srcKernelSumPtr = post->srcKernelSum;
    __m256 kernelSum0 = _mm256_setzero_ps();
    __m256 kernelSum1 = _mm256_setzero_ps();
    __m256 kernelSum2 = _mm256_setzero_ps();
    __m256 kernelSum3 = _mm256_setzero_ps();
    if (GEMMINT8_AVX2_E == realDst) {
        kernelSum0 = _mm256_set1_ps(post->srcKernelSum[0]);
        kernelSum1 = _mm256_set1_ps(post->srcKernelSum[1]);
        kernelSum2 = _mm256_set1_ps(post->srcKernelSum[2]);
        kernelSum3 = _mm256_set1_ps(post->srcKernelSum[3]);
    } else {
        kernelSum0 = _mm256_set1_ps(post->srcKernelSum[0]);
        if (realDst > 1) {
            kernelSum1 = _mm256_set1_ps(post->srcKernelSum[1]);
        }
        if (realDst > 2) {
            kernelSum2 = _mm256_set1_ps(post->srcKernelSum[2]);
        }
    }
    auto f128   = _mm256_set1_ps(128.f);
    __m256 extrascale0 = _mm256_setzero_ps();
    __m256 extrascale1 = _mm256_setzero_ps();
    __m256 extrascale2 = _mm256_setzero_ps();
    __m256 extrascale3 = _mm256_setzero_ps();
    if (post->extraScale) {
        if (GEMMINT8_AVX2_E == realDst) {
            extrascale0 = _mm256_set1_ps(post->extraScale[0]);
            extrascale1 = _mm256_set1_ps(post->extraScale[1]);
            extrascale2 = _mm256_set1_ps(post->extraScale[2]);
            extrascale3 = _mm256_set1_ps(post->extraScale[3]);
        } else {
            extrascale0 = _mm256_set1_ps(post->extraScale[0]);
            if (realDst > 1) {
                extrascale1 = _mm256_set1_ps(post->extraScale[1]);
            }
            if (realDst > 2) {
                extrascale2 = _mm256_set1_ps(post->extraScale[2]);
            }
        }
    }
    //printf("e=%d, sz=%d, dz=%d\n", realDst, src_depth_quad, dst_depth_quad);
    if (GEMMINT8_AVX2_E == realDst) {
        for (int dz = 0; dz < dst_depth_quad; ++dz) {
            const auto weight_dz = weight + dz * weight_step_Z;
            const auto weightBias_dz = post->weightQuanBias + dz * AVX2_PACKINT8;
            const float* scale_dz = post->scale + dz * AVX2_PACKINT8;
            auto dst_z           = dst + dz * dst_step_tmp;
            const auto src_x   = src;
            auto dst_x         = dst_z;
            __m256i D00 = _mm256_set1_epi32(0);
            __m256i D01 = _mm256_set1_epi32(0);
            __m256i D02 = _mm256_set1_epi32(0);
            __m256i D03 = _mm256_set1_epi32(0);
            __m256i D10 = _mm256_set1_epi32(0);
            __m256i D11 = _mm256_set1_epi32(0);
            __m256i D12 = _mm256_set1_epi32(0);
            __m256i D13 = _mm256_set1_epi32(0);

            for (int sz = 0; sz < src_depth_quad; ++sz) {
                const auto weight_sz = weight_dz + sz * weight_step_Y;
                const auto src_z     = src_x + sz * GEMMINT8_AVX2_L * GEMMINT8_AVX2_E;
                LOAD_INT4_TO_INT8;
                auto W0 = _mm256_cvtepi8_epi16(w_0);
                auto W1 = _mm256_cvtepi8_epi16(w_1);

                auto s0 = _mm_castps_si128(_mm_broadcast_ss((float*)src_z + 0));
                auto s1 = _mm_castps_si128(_mm_broadcast_ss((float*)src_z + 1));
                auto s2 = _mm_castps_si128(_mm_broadcast_ss((float*)src_z + 2));
                auto s3 = _mm_castps_si128(_mm_broadcast_ss((float*)src_z + 3));
                auto S0 = _mm256_cvtepu8_epi16(s0);
                auto S1 = _mm256_cvtepu8_epi16(s1);
                auto S2 = _mm256_cvtepu8_epi16(s2);
                auto S3 = _mm256_cvtepu8_epi16(s3);

                COMPUTE(0, 0);
                COMPUTE(1, 0);
                COMPUTE(0, 1);
                COMPUTE(1, 1);
                COMPUTE(0, 2);
                COMPUTE(1, 2);
                COMPUTE(0, 3);
                COMPUTE(1, 3);
            }
            auto D0 = NORMAL_HADD(D00, D10);
            auto D1 = NORMAL_HADD(D01, D11);
            auto D2 = NORMAL_HADD(D02, D12);
            auto D3 = NORMAL_HADD(D03, D13);
            auto scaleValue = _mm256_loadu_ps(scale_dz);         
            auto weightBiasValue = _mm256_loadu_ps((float*)weightBias_dz);

            auto f0 = _mm256_cvtepi32_ps(D0);
            auto f1 = _mm256_cvtepi32_ps(D1);
            auto f2 = _mm256_cvtepi32_ps(D2);
            auto f3 = _mm256_cvtepi32_ps(D3);
            // x_kernelSum x w_quantZero
            auto xy0_0 = _mm256_mul_ps(kernelSum0, weightBiasValue); // x dimemsion first
            auto xy0_1 = _mm256_mul_ps(kernelSum1, weightBiasValue); // ..second
            auto xy0_2 = _mm256_mul_ps(kernelSum2, weightBiasValue); // .. third
            auto xy0_3 = _mm256_mul_ps(kernelSum3, weightBiasValue); // ..fourth
            f0 = _mm256_mul_ps(f0, scaleValue);
            f1 = _mm256_mul_ps(f1, scaleValue);
            f2 = _mm256_mul_ps(f2, scaleValue);
            f3 = _mm256_mul_ps(f3, scaleValue);
            if (post->extraScale) {
                f0 = _mm256_mul_ps(f0, extrascale0);
                f1 = _mm256_mul_ps(f1, extrascale1);
                f2 = _mm256_mul_ps(f2, extrascale2);
                f3 = _mm256_mul_ps(f3, extrascale3);
                if (post->extraBias && nullptr != biasPtr) {
                    auto extraB = post->extraBias + dz * AVX2_PACKINT8;
                    auto extrabias = _mm256_loadu_ps(extraB);
                    extrabias = _mm256_mul_ps(f128, extrabias);
                    auto extrabias0 = _mm256_mul_ps(extrabias, extrascale0);
                    auto extrabias1 = _mm256_mul_ps(extrabias, extrascale1);
                    auto extrabias2 = _mm256_mul_ps(extrabias, extrascale2);
                    auto extrabias3 = _mm256_mul_ps(extrabias, extrascale3);
                    f0 = _mm256_sub_ps(f0, extrabias0);
                    f1 = _mm256_sub_ps(f1, extrabias1);
                    f2 = _mm256_sub_ps(f2, extrabias2);
                    f3 = _mm256_sub_ps(f3, extrabias3);
                }
            }
            f0 = _mm256_add_ps(f0, xy0_0);
            f1 = _mm256_add_ps(f1, xy0_1);
            f2 = _mm256_add_ps(f2, xy0_2);
            f3 = _mm256_add_ps(f3, xy0_3);

            if (nullptr != biasPtr) {
                const auto bias_dz   = biasPtr + dz * AVX2_PACKINT8;
                auto biasValue       = _mm256_loadu_ps(bias_dz);
                f0 = _mm256_add_ps(f0, biasValue);
                f1 = _mm256_add_ps(f1, biasValue);
                f2 = _mm256_add_ps(f2, biasValue);
                f3 = _mm256_add_ps(f3, biasValue);
            } else {
                auto dstv0 = _mm256_loadu_ps(((float*)dst_x) + 0 * AVX2_PACKINT8);
                auto dstv1 = _mm256_loadu_ps(((float*)dst_x) + 1 * AVX2_PACKINT8);
                auto dstv2 = _mm256_loadu_ps(((float*)dst_x) + 2 * AVX2_PACKINT8);
                auto dstv3 = _mm256_loadu_ps(((float*)dst_x) + 3 * AVX2_PACKINT8);
                f0 = _mm256_add_ps(f0, dstv0);
                f1 = _mm256_add_ps(f1, dstv1);
                f2 = _mm256_add_ps(f2, dstv2);
                f3 = _mm256_add_ps(f3, dstv3);
            }
            if (post->fp32minmax) {
                f0 = _mm256_min_ps(f0, fp32max);
                f1 = _mm256_min_ps(f1, fp32max);
                f2 = _mm256_min_ps(f2, fp32max);
                f3 = _mm256_min_ps(f3, fp32max);
                f0 = _mm256_max_ps(f0, fp32min);
                f1 = _mm256_max_ps(f1, fp32min);
                f2 = _mm256_max_ps(f2, fp32min);
                f3 = _mm256_max_ps(f3, fp32min);
            }
            _mm256_storeu_ps(((float*)dst_x) + 0 * AVX2_PACKINT8, f0);
            _mm256_storeu_ps(((float*)dst_x) + 1 * AVX2_PACKINT8, f1);
            _mm256_storeu_ps(((float*)dst_x) + 2 * AVX2_PACKINT8, f2);
            _mm256_storeu_ps(((float*)dst_x) + 3 * AVX2_PACKINT8, f3);
            
        }
        return;
    }
    if (3 == realDst) {
        for (int dz = 0; dz < dst_depth_quad; ++dz) {
            const auto weight_dz = weight + dz * weight_step_Z;
            const auto weightBias_dz = post->weightQuanBias + dz * AVX2_PACKINT8;
            const float* scale_dz = post->scale + dz * AVX2_PACKINT8;
            auto dst_z           = dst + dz * dst_step_tmp;
            const auto src_x   = src;
            auto dst_x         = dst_z;
            __m256i D00 = _mm256_set1_epi32(0);
            __m256i D01 = _mm256_set1_epi32(0);
            __m256i D02 = _mm256_set1_epi32(0);

            __m256i D10 = _mm256_set1_epi32(0);
            __m256i D11 = _mm256_set1_epi32(0);
            __m256i D12 = _mm256_set1_epi32(0);

            for (int sz = 0; sz < src_depth_quad; ++sz) {
                const auto weight_sz = weight_dz + sz * weight_step_Y;
                const auto src_z     = src_x + sz * GEMMINT8_AVX2_L * GEMMINT8_AVX2_E;
                LOAD_INT4_TO_INT8;

                auto W0 = _mm256_cvtepi8_epi16(w_0);
                auto W1 = _mm256_cvtepi8_epi16(w_1);

                auto s0 = _mm_castps_si128(_mm_broadcast_ss((float*)src_z + 0));
                auto s1 = _mm_castps_si128(_mm_broadcast_ss((float*)src_z + 1));
                auto s2 = _mm_castps_si128(_mm_broadcast_ss((float*)src_z + 2));
                auto S0 = _mm256_cvtepu8_epi16(s0);
                auto S1 = _mm256_cvtepu8_epi16(s1);
                auto S2 = _mm256_cvtepu8_epi16(s2);

                COMPUTE(0, 0);
                COMPUTE(1, 0);
                COMPUTE(0, 1);
                COMPUTE(1, 1);
                COMPUTE(0, 2);
                COMPUTE(1, 2);
            }
            auto D0 = NORMAL_HADD(D00, D10);
            auto D1 = NORMAL_HADD(D01, D11);
            auto D2 = NORMAL_HADD(D02, D12);
            auto scaleValue = _mm256_loadu_ps(scale_dz);
            auto weightBiasValue = _mm256_loadu_ps((float*)weightBias_dz);

            auto f0 = _mm256_cvtepi32_ps(D0);
            auto f1 = _mm256_cvtepi32_ps(D1);
            auto f2 = _mm256_cvtepi32_ps(D2);
            // x_kernelSum x w_quantZero
            auto xy0_0 = _mm256_mul_ps(kernelSum0, weightBiasValue); // x dimemsion first
            auto xy0_1 = _mm256_mul_ps(kernelSum1, weightBiasValue); // ..second
            auto xy0_2 = _mm256_mul_ps(kernelSum2, weightBiasValue); // .. third
            f0 = _mm256_mul_ps(f0, scaleValue);
            f1 = _mm256_mul_ps(f1, scaleValue);
            f2 = _mm256_mul_ps(f2, scaleValue);
            if (post->extraScale) {
                f0 = _mm256_mul_ps(f0, extrascale0);
                f1 = _mm256_mul_ps(f1, extrascale1);
                f2 = _mm256_mul_ps(f2, extrascale2);
                if (post->extraBias && nullptr != biasPtr) {
                    auto extraB = post->extraBias + dz * AVX2_PACKINT8;
                    auto extrabias = _mm256_loadu_ps(extraB);
                    extrabias = _mm256_mul_ps(f128, extrabias);
                    auto extrabias0 = _mm256_mul_ps(extrabias, extrascale0);
                    auto extrabias1 = _mm256_mul_ps(extrabias, extrascale1);
                    auto extrabias2 = _mm256_mul_ps(extrabias, extrascale2);
                    f0 = _mm256_sub_ps(f0, extrabias0);
                    f1 = _mm256_sub_ps(f1, extrabias1);
                    f2 = _mm256_sub_ps(f2, extrabias2);
                }
            }
            f0 = _mm256_add_ps(f0, xy0_0);
            f1 = _mm256_add_ps(f1, xy0_1);
            f2 = _mm256_add_ps(f2, xy0_2);

            if (nullptr != biasPtr) {
                const auto bias_dz   = biasPtr + dz * AVX2_PACKINT8;
                auto biasValue       = _mm256_loadu_ps(bias_dz);
                f0 = _mm256_add_ps(f0, biasValue);
                f1 = _mm256_add_ps(f1, biasValue);
                f2 = _mm256_add_ps(f2, biasValue);
            } else {
                auto dstv0 = _mm256_loadu_ps(((float*)dst_x) + 0 * AVX2_PACKINT8);
                auto dstv1 = _mm256_loadu_ps(((float*)dst_x) + 1 * AVX2_PACKINT8);
                auto dstv2 = _mm256_loadu_ps(((float*)dst_x) + 2 * AVX2_PACKINT8);
                f0 = _mm256_add_ps(f0, dstv0);
                f1 = _mm256_add_ps(f1, dstv1);
                f2 = _mm256_add_ps(f2, dstv2);
            }
            if (post->fp32minmax) {
                f0 = _mm256_min_ps(f0, fp32max);
                f1 = _mm256_min_ps(f1, fp32max);
                f2 = _mm256_min_ps(f2, fp32max);
                f0 = _mm256_max_ps(f0, fp32min);
                f1 = _mm256_max_ps(f1, fp32min);
                f2 = _mm256_max_ps(f2, fp32min);
            }
            _mm256_storeu_ps(((float*)dst_x) + 0 * AVX2_PACKINT8, f0);
            _mm256_storeu_ps(((float*)dst_x) + 1 * AVX2_PACKINT8, f1);
            _mm256_storeu_ps(((float*)dst_x) + 2 * AVX2_PACKINT8, f2);
            
        }
        return;
    }    
    if (2 == realDst) {
        for (int dz = 0; dz < dst_depth_quad; ++dz) {
            const auto weight_dz = weight + dz * weight_step_Z;
            const auto weightBias_dz = post->weightQuanBias + dz * AVX2_PACKINT8;
            const float* scale_dz = post->scale + dz * AVX2_PACKINT8;
            auto dst_z           = dst + dz * dst_step_tmp;
            const auto src_x   = src;
            auto dst_x         = dst_z;
            __m256i D00 = _mm256_set1_epi32(0);
            __m256i D01 = _mm256_set1_epi32(0);

            __m256i D10 = _mm256_set1_epi32(0);
            __m256i D11 = _mm256_set1_epi32(0);

            for (int sz = 0; sz < src_depth_quad; ++sz) {
                const auto weight_sz = weight_dz + sz * weight_step_Y;
                const auto src_z     = src_x + sz * GEMMINT8_AVX2_L * GEMMINT8_AVX2_E;
                LOAD_INT4_TO_INT8;
                auto W0 = _mm256_cvtepi8_epi16(w_0);
                auto W1 = _mm256_cvtepi8_epi16(w_1);

                auto s0 = _mm_castps_si128(_mm_broadcast_ss((float*)src_z + 0));
                auto s1 = _mm_castps_si128(_mm_broadcast_ss((float*)src_z + 1));
                auto S0 = _mm256_cvtepu8_epi16(s0);
                auto S1 = _mm256_cvtepu8_epi16(s1);

                COMPUTE(0, 0);
                COMPUTE(1, 0);
                COMPUTE(0, 1);
                COMPUTE(1, 1);
            }
            auto D0 = NORMAL_HADD(D00, D10);
            auto D1 = NORMAL_HADD(D01, D11);
            auto scaleValue = _mm256_loadu_ps(scale_dz);
            auto weightBiasValue = _mm256_loadu_ps((float*)weightBias_dz);

            auto f0 = _mm256_cvtepi32_ps(D0);
            auto f1 = _mm256_cvtepi32_ps(D1);
            // x_kernelSum x w_quantZero
            auto xy0_0 = _mm256_mul_ps(kernelSum0, weightBiasValue); // x dimemsion first
            auto xy0_1 = _mm256_mul_ps(kernelSum1, weightBiasValue); // ..second
            f0 = _mm256_mul_ps(f0, scaleValue);
            f1 = _mm256_mul_ps(f1, scaleValue);
            if (post->extraScale) {
                f0 = _mm256_mul_ps(f0, extrascale0);
                f1 = _mm256_mul_ps(f1, extrascale1);
                if (post->extraBias && nullptr != biasPtr) {
                    auto extraB = post->extraBias + dz * AVX2_PACKINT8;
                    auto extrabias = _mm256_loadu_ps(extraB);
                    extrabias = _mm256_mul_ps(f128, extrabias);
                    auto extrabias0 = _mm256_mul_ps(extrabias, extrascale0);
                    auto extrabias1 = _mm256_mul_ps(extrabias, extrascale1);
                    auto extrabias2 = _mm256_mul_ps(extrabias, extrascale2);
                    f0 = _mm256_sub_ps(f0, extrabias0);
                    f1 = _mm256_sub_ps(f1, extrabias1);
                }
            }
            f0 = _mm256_add_ps(f0, xy0_0);
            f1 = _mm256_add_ps(f1, xy0_1);

            if (nullptr != biasPtr) {
                const auto bias_dz   = biasPtr + dz * AVX2_PACKINT8;
                auto biasValue       = _mm256_loadu_ps(bias_dz);
                f0 = _mm256_add_ps(f0, biasValue);
                f1 = _mm256_add_ps(f1, biasValue);
            } else {
                auto dstv0 = _mm256_loadu_ps(((float*)dst_x) + 0 * AVX2_PACKINT8);
                auto dstv1 = _mm256_loadu_ps(((float*)dst_x) + 1 * AVX2_PACKINT8);
                f0         = _mm256_add_ps(f0, dstv0);
                f1         = _mm256_add_ps(f1, dstv1);
            }
            if (post->fp32minmax) {
                f0 = _mm256_min_ps(f0, fp32max);
                f1 = _mm256_min_ps(f1, fp32max);
                f0 = _mm256_max_ps(f0, fp32min);
                f1 = _mm256_max_ps(f1, fp32min);
            }
            _mm256_storeu_ps(((float*)dst_x) + 0 * AVX2_PACKINT8, f0);
            _mm256_storeu_ps(((float*)dst_x) + 1 * AVX2_PACKINT8, f1);
            
        }
        return;
    }    
    if (1 == realDst) {
        for (int dz = 0; dz < dst_depth_quad; ++dz) {
            const auto weight_dz = weight + dz * weight_step_Z;
            const auto weightBias_dz = post->weightQuanBias + dz * AVX2_PACKINT8;
            const float* scale_dz = post->scale + dz * AVX2_PACKINT8;
            auto dst_z           = dst + dz * dst_step_tmp;
            const auto src_x   = src;
            auto dst_x         = dst_z;
            __m256i D00 = _mm256_set1_epi32(0);
            __m256i D10 = _mm256_set1_epi32(0);

            for (int sz = 0; sz < src_depth_quad; ++sz) {
                const auto weight_sz = weight_dz + sz * weight_step_Y;
                const auto src_z     = src_x + sz * GEMMINT8_AVX2_L * GEMMINT8_AVX2_E;
                LOAD_INT4_TO_INT8;
                auto W0 = _mm256_cvtepi8_epi16(w_0);
                auto W1 = _mm256_cvtepi8_epi16(w_1);

                auto s0 = _mm_castps_si128(_mm_broadcast_ss((float*)src_z + 0));
                auto S0 = _mm256_cvtepu8_epi16(s0);

                COMPUTE(0, 0);
                COMPUTE(1, 0);
            }
            auto D0 = NORMAL_HADD(D00, D10);
            auto scaleValue = _mm256_loadu_ps(scale_dz);
            auto weightBiasValue = _mm256_loadu_ps((float*)weightBias_dz);

            auto f0 = _mm256_cvtepi32_ps(D0);
            // x_kernelSum x w_quantZero
            auto xy0_0 = _mm256_mul_ps(kernelSum0, weightBiasValue); // x dimemsion first
            f0 = _mm256_mul_ps(f0, scaleValue);
            if (post->extraScale) {
                f0 = _mm256_mul_ps(f0, extrascale0);
                if (post->extraBias && nullptr != biasPtr) {
                    auto extraB = post->extraBias + dz * AVX2_PACKINT8;
                    auto extrabias = _mm256_loadu_ps(extraB);
                    extrabias = _mm256_mul_ps(f128, extrabias);
                    auto extrabias0 = _mm256_mul_ps(extrabias, extrascale0);
                    auto extrabias1 = _mm256_mul_ps(extrabias, extrascale1);
                    auto extrabias2 = _mm256_mul_ps(extrabias, extrascale2);
                    f0 = _mm256_sub_ps(f0, extrabias0);
                }
            }
            f0 = _mm256_add_ps(f0, xy0_0);

            if (nullptr != biasPtr) {
                const auto bias_dz   = biasPtr + dz * AVX2_PACKINT8;
                auto biasValue       = _mm256_loadu_ps(bias_dz);
                f0 = _mm256_add_ps(f0, biasValue);
            } else {
                auto dstv = _mm256_loadu_ps(((float*)dst_x) + 0 * AVX2_PACKINT8);
                f0        = _mm256_add_ps(f0, dstv);
            }
            if (post->fp32minmax) {
                f0 = _mm256_min_ps(f0, fp32max);
                f0 = _mm256_max_ps(f0, fp32min);
            }
            
            _mm256_storeu_ps(((float*)dst_x) + 0 * AVX2_PACKINT8, f0);
            
        }
        return;
    }    

}

void _AVX_MNNGemmInt8AddBiasScale_16x4_Unit(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad, const QuanPostTreatParameters* post, size_t realDst) {
    const auto dst_step_tmp = dst_step / sizeof(int8_t);
    auto zero128 = _mm256_set1_ps(0.0f);
    auto minValue = _mm256_set1_ps(post->minValue);
    auto maxValue = _mm256_set1_ps(post->maxValue);
    auto plus = _mm256_set1_ps(0.5f);
    auto minus = _mm256_set1_ps(-0.5f);
    auto offset = _mm256_set1_epi32(128);
    __m256 fp32min, fp32max;
    if (0 == post->useInt8 && post->fp32minmax) {
        fp32min = _mm256_set1_ps((post->fp32minmax)[0]);
        fp32max = _mm256_set1_ps((post->fp32minmax)[1]);
    }
    int blockNum = post->blockNum;
    const float* biasPtr = nullptr;
    if (post->biasFloat) {
        biasPtr = post->biasFloat;
    }
    auto srcKernelSumPtr = post->srcKernelSum;
    __m256 kernelSum0 = _mm256_setzero_ps();
    __m256 kernelSum1 = _mm256_setzero_ps();
    __m256 kernelSum2 = _mm256_setzero_ps();
    __m256 kernelSum3 = _mm256_setzero_ps();
    if (GEMMINT8_AVX2_E == realDst) {
        kernelSum0 = _mm256_set1_ps(post->srcKernelSum[0]);
        kernelSum1 = _mm256_set1_ps(post->srcKernelSum[1]);
        kernelSum2 = _mm256_set1_ps(post->srcKernelSum[2]);
        kernelSum3 = _mm256_set1_ps(post->srcKernelSum[3]);
    } else {
        kernelSum0 = _mm256_set1_ps(post->srcKernelSum[0]);
        if (realDst > 1) {
            kernelSum1 = _mm256_set1_ps(post->srcKernelSum[1]);
        }
        if (realDst > 2) {
            kernelSum2 = _mm256_set1_ps(post->srcKernelSum[2]);
        }
    }
    auto f128   = _mm256_set1_ps(128.f);
    __m256 extrascale0 = _mm256_setzero_ps();
    __m256 extrascale1 = _mm256_setzero_ps();
    __m256 extrascale2 = _mm256_setzero_ps();
    __m256 extrascale3 = _mm256_setzero_ps();
    if (post->extraScale) {
        if (GEMMINT8_AVX2_E == realDst) {
            extrascale0 = _mm256_set1_ps(post->extraScale[0]);
            extrascale1 = _mm256_set1_ps(post->extraScale[1]);
            extrascale2 = _mm256_set1_ps(post->extraScale[2]);
            extrascale3 = _mm256_set1_ps(post->extraScale[3]);
        } else {
            extrascale0 = _mm256_set1_ps(post->extraScale[0]);
            if (realDst > 1) {
                extrascale1 = _mm256_set1_ps(post->extraScale[1]);
            }
            if (realDst > 2) {
                extrascale2 = _mm256_set1_ps(post->extraScale[2]);
            }
        }
    }
    //printf("e=%d, sz=%d, dz=%d\n", realDst, src_depth_quad, dst_depth_quad);
    if (GEMMINT8_AVX2_E == realDst) {
        for (int dz = 0; dz < dst_depth_quad; ++dz) {
            const auto weight_dz = weight + dz * blockNum * src_depth_quad * (GEMMINT8_AVX2_L * GEMMINT8_AVX2_H);
            const auto weightBias_dz = post->weightQuanBias + dz * AVX2_PACKINT8;
            const float* scale_dz = post->scale + dz * AVX2_PACKINT8;
            auto dst_z           = dst + dz * dst_step_tmp;
            const auto src_x   = src;
            auto dst_x         = dst_z;
            __m256i D00 = _mm256_set1_epi32(0);
            __m256i D01 = _mm256_set1_epi32(0);
            __m256i D02 = _mm256_set1_epi32(0);
            __m256i D03 = _mm256_set1_epi32(0);
            __m256i D10 = _mm256_set1_epi32(0);
            __m256i D11 = _mm256_set1_epi32(0);
            __m256i D12 = _mm256_set1_epi32(0);
            __m256i D13 = _mm256_set1_epi32(0);

            for (int sz = 0; sz < src_depth_quad; ++sz) {
                const auto weight_sz = weight_dz + sz * (GEMMINT8_AVX2_L * GEMMINT8_AVX2_H);
                const auto src_z     = src_x + sz * GEMMINT8_AVX2_L * GEMMINT8_AVX2_E;
                auto w0 = mm_loadu_si128(weight_sz + 16 * 0);
                auto w1 = mm_loadu_si128(weight_sz + 16 * 1);
                auto W0 = _mm256_cvtepi8_epi16(w0);
                auto W1 = _mm256_cvtepi8_epi16(w1);

                auto s0 = _mm_castps_si128(_mm_broadcast_ss((float*)src_z + 0));
                auto s1 = _mm_castps_si128(_mm_broadcast_ss((float*)src_z + 1));
                auto s2 = _mm_castps_si128(_mm_broadcast_ss((float*)src_z + 2));
                auto s3 = _mm_castps_si128(_mm_broadcast_ss((float*)src_z + 3));
                auto S0 = _mm256_cvtepu8_epi16(s0);
                auto S1 = _mm256_cvtepu8_epi16(s1);
                auto S2 = _mm256_cvtepu8_epi16(s2);
                auto S3 = _mm256_cvtepu8_epi16(s3);

                COMPUTE(0, 0);
                COMPUTE(1, 0);
                COMPUTE(0, 1);
                COMPUTE(1, 1);
                COMPUTE(0, 2);
                COMPUTE(1, 2);
                COMPUTE(0, 3);
                COMPUTE(1, 3);
            }
            auto D0 = NORMAL_HADD(D00, D10);
            auto D1 = NORMAL_HADD(D01, D11);
            auto D2 = NORMAL_HADD(D02, D12);
            auto D3 = NORMAL_HADD(D03, D13);
            auto scaleValue = _mm256_loadu_ps(scale_dz);
            auto weightBiasValue = _mm256_loadu_ps((float*)weightBias_dz);

            auto f0 = _mm256_cvtepi32_ps(D0);
            auto f1 = _mm256_cvtepi32_ps(D1);
            auto f2 = _mm256_cvtepi32_ps(D2);
            auto f3 = _mm256_cvtepi32_ps(D3);
            // x_kernelSum x w_quantZero
            auto xy0_0 = _mm256_mul_ps(kernelSum0, weightBiasValue); // x dimemsion first
            auto xy0_1 = _mm256_mul_ps(kernelSum1, weightBiasValue); // ..second
            auto xy0_2 = _mm256_mul_ps(kernelSum2, weightBiasValue); // .. third
            auto xy0_3 = _mm256_mul_ps(kernelSum3, weightBiasValue); // ..fourth
            f0 = _mm256_mul_ps(f0, scaleValue);
            f1 = _mm256_mul_ps(f1, scaleValue);
            f2 = _mm256_mul_ps(f2, scaleValue);
            f3 = _mm256_mul_ps(f3, scaleValue);
            if (post->extraScale) {
                f0 = _mm256_mul_ps(f0, extrascale0);
                f1 = _mm256_mul_ps(f1, extrascale1);
                f2 = _mm256_mul_ps(f2, extrascale2);
                f3 = _mm256_mul_ps(f3, extrascale3);
                if (post->extraBias && nullptr != biasPtr) {
                    auto extraB = post->extraBias + dz * AVX2_PACKINT8;
                    auto extrabias = _mm256_loadu_ps(extraB);
                    extrabias = _mm256_mul_ps(f128, extrabias);
                    auto extrabias0 = _mm256_mul_ps(extrabias, extrascale0);
                    auto extrabias1 = _mm256_mul_ps(extrabias, extrascale1);
                    auto extrabias2 = _mm256_mul_ps(extrabias, extrascale2);
                    auto extrabias3 = _mm256_mul_ps(extrabias, extrascale3);
                    f0 = _mm256_sub_ps(f0, extrabias0);
                    f1 = _mm256_sub_ps(f1, extrabias1);
                    f2 = _mm256_sub_ps(f2, extrabias2);
                    f3 = _mm256_sub_ps(f3, extrabias3);
                }
            }
            f0 = _mm256_add_ps(f0, xy0_0);
            f1 = _mm256_add_ps(f1, xy0_1);
            f2 = _mm256_add_ps(f2, xy0_2);
            f3 = _mm256_add_ps(f3, xy0_3);
            if (nullptr != biasPtr) {
                const auto bias_dz   = biasPtr + dz * AVX2_PACKINT8;
                auto biasValue       = _mm256_loadu_ps(bias_dz);
                f0 = _mm256_add_ps(f0, biasValue);
                f1 = _mm256_add_ps(f1, biasValue);
                f2 = _mm256_add_ps(f2, biasValue);
                f3 = _mm256_add_ps(f3, biasValue);
            }
            if (post->useInt8 == 1) {
                POSTTREAT(0);
                POSTTREAT(1);
                POSTTREAT(2);
                POSTTREAT(3);
            } else {
                if (nullptr == biasPtr) {
                    auto dstv0 = _mm256_loadu_ps(((float*)dst_x) + 0 * AVX2_PACKINT8);
                    auto dstv1 = _mm256_loadu_ps(((float*)dst_x) + 1 * AVX2_PACKINT8);
                    auto dstv2 = _mm256_loadu_ps(((float*)dst_x) + 2 * AVX2_PACKINT8);
                    auto dstv3 = _mm256_loadu_ps(((float*)dst_x) + 3 * AVX2_PACKINT8);

                    f0 = _mm256_add_ps(f0, dstv0);
                    f1 = _mm256_add_ps(f1, dstv1);
                    f2 = _mm256_add_ps(f2, dstv2);
                    f3 = _mm256_add_ps(f3, dstv3);
                }
                if (post->fp32minmax) {
                    f0 = _mm256_min_ps(f0, fp32max);
                    f1 = _mm256_min_ps(f1, fp32max);
                    f2 = _mm256_min_ps(f2, fp32max);
                    f3 = _mm256_min_ps(f3, fp32max);
                    f0 = _mm256_max_ps(f0, fp32min);
                    f1 = _mm256_max_ps(f1, fp32min);
                    f2 = _mm256_max_ps(f2, fp32min);
                    f3 = _mm256_max_ps(f3, fp32min);
                }
                _mm256_storeu_ps(((float*)dst_x) + 0 * AVX2_PACKINT8, f0);
                _mm256_storeu_ps(((float*)dst_x) + 1 * AVX2_PACKINT8, f1);
                _mm256_storeu_ps(((float*)dst_x) + 2 * AVX2_PACKINT8, f2);
                _mm256_storeu_ps(((float*)dst_x) + 3 * AVX2_PACKINT8, f3);
            }
        }
        return;
    }
    if (3 == realDst) {
        for (int dz = 0; dz < dst_depth_quad; ++dz) {
            const auto weight_dz = weight + dz * blockNum * src_depth_quad * (GEMMINT8_AVX2_L * GEMMINT8_AVX2_H);
            const auto weightBias_dz = post->weightQuanBias + dz * AVX2_PACKINT8;
            const float* scale_dz = post->scale + dz * AVX2_PACKINT8;
            auto dst_z           = dst + dz * dst_step_tmp;
            const auto src_x   = src;
            auto dst_x         = dst_z;
            __m256i D00 = _mm256_set1_epi32(0);
            __m256i D01 = _mm256_set1_epi32(0);
            __m256i D02 = _mm256_set1_epi32(0);

            __m256i D10 = _mm256_set1_epi32(0);
            __m256i D11 = _mm256_set1_epi32(0);
            __m256i D12 = _mm256_set1_epi32(0);

            for (int sz = 0; sz < src_depth_quad; ++sz) {
                const auto weight_sz = weight_dz + sz * (GEMMINT8_AVX2_L * GEMMINT8_AVX2_H);
                const auto src_z     = src_x + sz * GEMMINT8_AVX2_L * GEMMINT8_AVX2_E;
                auto w0 = mm_loadu_si128(weight_sz + 16 * 0);
                auto w1 = mm_loadu_si128(weight_sz + 16 * 1);
                auto W0 = _mm256_cvtepi8_epi16(w0);
                auto W1 = _mm256_cvtepi8_epi16(w1);

                auto s0 = _mm_castps_si128(_mm_broadcast_ss((float*)src_z + 0));
                auto s1 = _mm_castps_si128(_mm_broadcast_ss((float*)src_z + 1));
                auto s2 = _mm_castps_si128(_mm_broadcast_ss((float*)src_z + 2));
                auto S0 = _mm256_cvtepu8_epi16(s0);
                auto S1 = _mm256_cvtepu8_epi16(s1);
                auto S2 = _mm256_cvtepu8_epi16(s2);

                COMPUTE(0, 0);
                COMPUTE(1, 0);
                COMPUTE(0, 1);
                COMPUTE(1, 1);
                COMPUTE(0, 2);
                COMPUTE(1, 2);
            }
            auto D0 = NORMAL_HADD(D00, D10);
            auto D1 = NORMAL_HADD(D01, D11);
            auto D2 = NORMAL_HADD(D02, D12);
            auto scaleValue = _mm256_loadu_ps(scale_dz);
            auto weightBiasValue = _mm256_loadu_ps((float*)weightBias_dz);

            auto f0 = _mm256_cvtepi32_ps(D0);
            auto f1 = _mm256_cvtepi32_ps(D1);
            auto f2 = _mm256_cvtepi32_ps(D2);
            // x_kernelSum x w_quantZero
            auto xy0_0 = _mm256_mul_ps(kernelSum0, weightBiasValue); // x dimemsion first
            auto xy0_1 = _mm256_mul_ps(kernelSum1, weightBiasValue); // ..second
            auto xy0_2 = _mm256_mul_ps(kernelSum2, weightBiasValue); // .. third
            f0 = _mm256_mul_ps(f0, scaleValue);
            f1 = _mm256_mul_ps(f1, scaleValue);
            f2 = _mm256_mul_ps(f2, scaleValue);
            if (post->extraScale) {
                f0 = _mm256_mul_ps(f0, extrascale0);
                f1 = _mm256_mul_ps(f1, extrascale1);
                f2 = _mm256_mul_ps(f2, extrascale2);
                if (post->extraBias && nullptr != biasPtr) {
                    auto extraB = post->extraBias + dz * AVX2_PACKINT8;
                    auto extrabias = _mm256_loadu_ps(extraB);
                    extrabias = _mm256_mul_ps(f128, extrabias);
                    auto extrabias0 = _mm256_mul_ps(extrabias, extrascale0);
                    auto extrabias1 = _mm256_mul_ps(extrabias, extrascale1);
                    auto extrabias2 = _mm256_mul_ps(extrabias, extrascale2);
                    f0 = _mm256_sub_ps(f0, extrabias0);
                    f1 = _mm256_sub_ps(f1, extrabias1);
                    f2 = _mm256_sub_ps(f2, extrabias2);
                }
            }
            f0 = _mm256_add_ps(f0, xy0_0);
            f1 = _mm256_add_ps(f1, xy0_1);
            f2 = _mm256_add_ps(f2, xy0_2);
            if (nullptr != biasPtr) {
                const auto bias_dz   = biasPtr + dz * AVX2_PACKINT8;
                auto biasValue       = _mm256_loadu_ps(bias_dz);
                f0 = _mm256_add_ps(f0, biasValue);
                f1 = _mm256_add_ps(f1, biasValue);
                f2 = _mm256_add_ps(f2, biasValue);
            }
            if (post->useInt8 == 1) {
                POSTTREAT(0);
                POSTTREAT(1);
                POSTTREAT(2);
            } else {
                if (nullptr == biasPtr) {
                    auto dstv0 = _mm256_loadu_ps(((float*)dst_x) + 0 * AVX2_PACKINT8);
                    auto dstv1 = _mm256_loadu_ps(((float*)dst_x) + 1 * AVX2_PACKINT8);
                    auto dstv2 = _mm256_loadu_ps(((float*)dst_x) + 2 * AVX2_PACKINT8);
                    f0 = _mm256_add_ps(f0, dstv0);
                    f1 = _mm256_add_ps(f1, dstv1);
                    f2 = _mm256_add_ps(f2, dstv2);
                }
                if (post->fp32minmax) {
                    f0 = _mm256_min_ps(f0, fp32max);
                    f1 = _mm256_min_ps(f1, fp32max);
                    f2 = _mm256_min_ps(f2, fp32max);
                    f0 = _mm256_max_ps(f0, fp32min);
                    f1 = _mm256_max_ps(f1, fp32min);
                    f2 = _mm256_max_ps(f2, fp32min);
                }
                _mm256_storeu_ps(((float*)dst_x) + 0 * AVX2_PACKINT8, f0);
                _mm256_storeu_ps(((float*)dst_x) + 1 * AVX2_PACKINT8, f1);
                _mm256_storeu_ps(((float*)dst_x) + 2 * AVX2_PACKINT8, f2);
            }
        }
        return;
    }    
    if (2 == realDst) {
        for (int dz = 0; dz < dst_depth_quad; ++dz) {
            const auto weight_dz = weight + dz * blockNum * src_depth_quad * (GEMMINT8_AVX2_L * GEMMINT8_AVX2_H);
            const auto weightBias_dz = post->weightQuanBias + dz * AVX2_PACKINT8;
            const float* scale_dz = post->scale + dz * AVX2_PACKINT8;
            auto dst_z           = dst + dz * dst_step_tmp;
            const auto src_x   = src;
            auto dst_x         = dst_z;
            __m256i D00 = _mm256_set1_epi32(0);
            __m256i D01 = _mm256_set1_epi32(0);

            __m256i D10 = _mm256_set1_epi32(0);
            __m256i D11 = _mm256_set1_epi32(0);

            for (int sz = 0; sz < src_depth_quad; ++sz) {
                const auto weight_sz = weight_dz + sz * (GEMMINT8_AVX2_L * GEMMINT8_AVX2_H);
                const auto src_z     = src_x + sz * GEMMINT8_AVX2_L * GEMMINT8_AVX2_E;
                auto w0 = mm_loadu_si128(weight_sz + 16 * 0);
                auto w1 = mm_loadu_si128(weight_sz + 16 * 1);
                auto W0 = _mm256_cvtepi8_epi16(w0);
                auto W1 = _mm256_cvtepi8_epi16(w1);

                auto s0 = _mm_castps_si128(_mm_broadcast_ss((float*)src_z + 0));
                auto s1 = _mm_castps_si128(_mm_broadcast_ss((float*)src_z + 1));
                auto S0 = _mm256_cvtepu8_epi16(s0);
                auto S1 = _mm256_cvtepu8_epi16(s1);

                COMPUTE(0, 0);
                COMPUTE(1, 0);
                COMPUTE(0, 1);
                COMPUTE(1, 1);
            }
            auto D0 = NORMAL_HADD(D00, D10);
            auto D1 = NORMAL_HADD(D01, D11);
            auto scaleValue = _mm256_loadu_ps(scale_dz);
            auto weightBiasValue = _mm256_loadu_ps((float*)weightBias_dz);

            auto f0 = _mm256_cvtepi32_ps(D0);
            auto f1 = _mm256_cvtepi32_ps(D1);
            // x_kernelSum x w_quantZero
            auto xy0_0 = _mm256_mul_ps(kernelSum0, weightBiasValue); // x dimemsion first
            auto xy0_1 = _mm256_mul_ps(kernelSum1, weightBiasValue); // ..second
            f0 = _mm256_mul_ps(f0, scaleValue);
            f1 = _mm256_mul_ps(f1, scaleValue);
            if (post->extraScale) {
                f0 = _mm256_mul_ps(f0, extrascale0);
                f1 = _mm256_mul_ps(f1, extrascale1);
                if (post->extraBias && nullptr != biasPtr) {
                    auto extraB = post->extraBias + dz * AVX2_PACKINT8;
                    auto extrabias = _mm256_loadu_ps(extraB);
                    extrabias = _mm256_mul_ps(f128, extrabias);
                    auto extrabias0 = _mm256_mul_ps(extrabias, extrascale0);
                    auto extrabias1 = _mm256_mul_ps(extrabias, extrascale1);
                    auto extrabias2 = _mm256_mul_ps(extrabias, extrascale2);
                    f0 = _mm256_sub_ps(f0, extrabias0);
                    f1 = _mm256_sub_ps(f1, extrabias1);
                }
            }
            f0 = _mm256_add_ps(f0, xy0_0);
            f1 = _mm256_add_ps(f1, xy0_1);
            if (nullptr != biasPtr) {
                const auto bias_dz   = biasPtr + dz * AVX2_PACKINT8;
                auto biasValue       = _mm256_loadu_ps(bias_dz);
                f0 = _mm256_add_ps(f0, biasValue);
                f1 = _mm256_add_ps(f1, biasValue);
            }
            if (post->useInt8 == 1) {
                POSTTREAT(0);
                POSTTREAT(1);
            } else {
                if (nullptr == biasPtr) {
                    auto dstv0 = _mm256_loadu_ps(((float*)dst_x) + 0 * AVX2_PACKINT8);
                    auto dstv1 = _mm256_loadu_ps(((float*)dst_x) + 1 * AVX2_PACKINT8);
                    f0 = _mm256_add_ps(f0, dstv0);
                    f1 = _mm256_add_ps(f1, dstv1);
                }
                if (post->fp32minmax) {
                    f0 = _mm256_min_ps(f0, fp32max);
                    f1 = _mm256_min_ps(f1, fp32max);
                    f0 = _mm256_max_ps(f0, fp32min);
                    f1 = _mm256_max_ps(f1, fp32min);
                }
                _mm256_storeu_ps(((float*)dst_x) + 0 * AVX2_PACKINT8, f0);
                _mm256_storeu_ps(((float*)dst_x) + 1 * AVX2_PACKINT8, f1);
            }
        }
        return;
    }    
    if (1 == realDst) {
        for (int dz = 0; dz < dst_depth_quad; ++dz) {
            const auto weight_dz = weight + dz * blockNum * src_depth_quad * (GEMMINT8_AVX2_L * GEMMINT8_AVX2_H);
            const auto weightBias_dz = post->weightQuanBias + dz * AVX2_PACKINT8;
            const float* scale_dz = post->scale + dz * AVX2_PACKINT8;
            auto dst_z           = dst + dz * dst_step_tmp;
            const auto src_x   = src;
            auto dst_x         = dst_z;
            __m256i D00 = _mm256_set1_epi32(0);
            __m256i D10 = _mm256_set1_epi32(0);

            for (int sz = 0; sz < src_depth_quad; ++sz) {
                const auto weight_sz = weight_dz + sz * (GEMMINT8_AVX2_L * GEMMINT8_AVX2_H);
                const auto src_z     = src_x + sz * GEMMINT8_AVX2_L * GEMMINT8_AVX2_E;
                auto w0 = mm_loadu_si128(weight_sz + 16 * 0);
                auto w1 = mm_loadu_si128(weight_sz + 16 * 1);
                auto W0 = _mm256_cvtepi8_epi16(w0);
                auto W1 = _mm256_cvtepi8_epi16(w1);

                auto s0 = _mm_castps_si128(_mm_broadcast_ss((float*)src_z + 0));
                auto S0 = _mm256_cvtepu8_epi16(s0);

                COMPUTE(0, 0);
                COMPUTE(1, 0);
            }
            auto D0 = NORMAL_HADD(D00, D10);
            auto scaleValue = _mm256_loadu_ps(scale_dz);
            auto weightBiasValue = _mm256_loadu_ps((float*)weightBias_dz);

            auto f0 = _mm256_cvtepi32_ps(D0);
            // x_kernelSum x w_quantZero
            auto xy0_0 = _mm256_mul_ps(kernelSum0, weightBiasValue); // x dimemsion first
            f0 = _mm256_mul_ps(f0, scaleValue);
            if (post->extraScale) {
                f0 = _mm256_mul_ps(f0, extrascale0);
                if (post->extraBias && nullptr != biasPtr) {
                    auto extraB = post->extraBias + dz * AVX2_PACKINT8;
                    auto extrabias = _mm256_loadu_ps(extraB);
                    extrabias = _mm256_mul_ps(f128, extrabias);
                    auto extrabias0 = _mm256_mul_ps(extrabias, extrascale0);
                    auto extrabias1 = _mm256_mul_ps(extrabias, extrascale1);
                    auto extrabias2 = _mm256_mul_ps(extrabias, extrascale2);
                    f0 = _mm256_sub_ps(f0, extrabias0);
                }
            }
            f0 = _mm256_add_ps(f0, xy0_0);
            if (nullptr != biasPtr) {
                const auto bias_dz   = biasPtr + dz * AVX2_PACKINT8;
                auto biasValue       = _mm256_loadu_ps(bias_dz);
                f0 = _mm256_add_ps(f0, biasValue);
            }
            if (post->useInt8 == 1) {
                POSTTREAT(0);
            } else {
                if (nullptr == biasPtr) {
                    auto dstv0 = _mm256_loadu_ps(((float*)dst_x) + 0 * AVX2_PACKINT8);
                    f0 = _mm256_add_ps(f0, dstv0);
                }
                if (post->fp32minmax) {
                    f0 = _mm256_min_ps(f0, fp32max);
                    f0 = _mm256_max_ps(f0, fp32min);
                }
                _mm256_storeu_ps(((float*)dst_x) + 0 * AVX2_PACKINT8, f0);
            }
        }
        return;
    }    

}
void _AVX_MNNGemmInt8AddBiasScale_16x4_Unit_Fast(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad, const QuanPostTreatParameters* post, size_t realDst) {
    const auto dst_step_tmp = dst_step / sizeof(int8_t);
    auto zero128 = _mm256_set1_ps(0.0f);
    auto minValue = _mm256_set1_ps(post->minValue);
    auto maxValue = _mm256_set1_ps(post->maxValue);
    auto plus = _mm256_set1_ps(0.5f);
    auto minus = _mm256_set1_ps(-0.5f);
    auto oneValue = _mm256_set1_epi16(1);
    auto offset = _mm256_set1_epi32(128);
    __m256 fp32min, fp32max;
    if (0 == post->useInt8) {
        fp32min = _mm256_set1_ps((post->fp32minmax)[0]);
        fp32max = _mm256_set1_ps((post->fp32minmax)[1]);
    }
    auto srcKernelSumPtr = post->srcKernelSum;
    __m256 kernelSum0 = _mm256_setzero_ps();
    __m256 kernelSum1 = _mm256_setzero_ps();
    __m256 kernelSum2 = _mm256_setzero_ps();
    __m256 kernelSum3 = _mm256_setzero_ps();
    if (GEMMINT8_AVX2_E == realDst) {
        kernelSum0 = _mm256_set1_ps(post->srcKernelSum[0]);
        kernelSum1 = _mm256_set1_ps(post->srcKernelSum[1]);
        kernelSum2 = _mm256_set1_ps(post->srcKernelSum[2]);
        kernelSum3 = _mm256_set1_ps(post->srcKernelSum[3]);
    } else {
        kernelSum0 = _mm256_set1_ps(post->srcKernelSum[0]);
        if (realDst > 1) {
            kernelSum1 = _mm256_set1_ps(post->srcKernelSum[1]);
        }
        if (realDst > 2) {
            kernelSum2 = _mm256_set1_ps(post->srcKernelSum[2]);
        }
    }
    //printf("e=%d, sz=%d, dz=%d\n", realDst, src_depth_quad, dst_depth_quad);
    if (GEMMINT8_AVX2_E == realDst) {
        for (int dz = 0; dz < dst_depth_quad; ++dz) {
            const auto weight_dz = weight + dz * src_depth_quad * (GEMMINT8_AVX2_L * GEMMINT8_AVX2_H);
            const auto bias_dz = post->biasFloat + dz * AVX2_PACKINT8;
            const auto weightBias_dz = post->weightQuanBias + dz * AVX2_PACKINT8;
            const float* scale_dz = post->scale + dz * AVX2_PACKINT8;
            auto dst_z           = dst + dz * dst_step_tmp;
            const auto src_x   = src;
            auto dst_x         = dst_z;
            __m256i D00 = _mm256_set1_epi32(0);
            __m256i D01 = _mm256_set1_epi32(0);
            __m256i D02 = _mm256_set1_epi32(0);
            __m256i D03 = _mm256_set1_epi32(0);

            for (int sz = 0; sz < src_depth_quad; ++sz) {
                const auto weight_sz = weight_dz + sz * (GEMMINT8_AVX2_L * GEMMINT8_AVX2_H);
                const auto src_z     = src_x + sz * GEMMINT8_AVX2_L * GEMMINT8_AVX2_E;
                auto w0 = _mm256_loadu_si256((__m256i*)weight_sz);

                auto s0 = _mm256_castps_si256(_mm256_broadcast_ss((float*)src_z + 0));
                auto s1 = _mm256_castps_si256(_mm256_broadcast_ss((float*)src_z + 1));
                auto s2 = _mm256_castps_si256(_mm256_broadcast_ss((float*)src_z + 2));
                auto s3 = _mm256_castps_si256(_mm256_broadcast_ss((float*)src_z + 3));

                D00 = _mm256_add_epi32(D00, _mm256_madd_epi16(_mm256_maddubs_epi16(s0, w0), oneValue));
                D01 = _mm256_add_epi32(D01, _mm256_madd_epi16(_mm256_maddubs_epi16(s1, w0), oneValue));
                D02 = _mm256_add_epi32(D02, _mm256_madd_epi16(_mm256_maddubs_epi16(s2, w0), oneValue));
                D03 = _mm256_add_epi32(D03, _mm256_madd_epi16(_mm256_maddubs_epi16(s3, w0), oneValue));

            }
            auto D0 = D00;
            auto D1 = D01;
            auto D2 = D02;
            auto D3 = D03;

            // auto biasValue0 = _mm256_loadu_si256((__m256i*)(bias_dz));
            auto weightBiasValue = _mm256_loadu_ps((float*)weightBias_dz);
            // D0 = _mm256_add_epi32(D0, biasValue0);
            // D1 = _mm256_add_epi32(D1, biasValue0);
            // D2 = _mm256_add_epi32(D2, biasValue0);
            // D3 = _mm256_add_epi32(D3, biasValue0);

            auto scaleValue = _mm256_loadu_ps(scale_dz);
            auto f0 = _mm256_cvtepi32_ps(D0);
            auto f1 = _mm256_cvtepi32_ps(D1);
            auto f2 = _mm256_cvtepi32_ps(D2);
            auto f3 = _mm256_cvtepi32_ps(D3);
            // x_kernelSum x w_quantZero
            auto xy0_0 = _mm256_mul_ps(kernelSum0, weightBiasValue); // x dimemsion first
            auto xy0_1 = _mm256_mul_ps(kernelSum1, weightBiasValue); // ..second
            auto xy0_2 = _mm256_mul_ps(kernelSum2, weightBiasValue); // .. third
            auto xy0_3 = _mm256_mul_ps(kernelSum3, weightBiasValue); // ..fourth
            f0 = _mm256_mul_ps(f0, scaleValue);
            f1 = _mm256_mul_ps(f1, scaleValue);
            f2 = _mm256_mul_ps(f2, scaleValue);
            f3 = _mm256_mul_ps(f3, scaleValue);
            f0 = _mm256_add_ps(f0, xy0_0);
            f1 = _mm256_add_ps(f1, xy0_1);
            f2 = _mm256_add_ps(f2, xy0_2);
            f3 = _mm256_add_ps(f3, xy0_3);
            auto biasValue       = _mm256_loadu_ps(bias_dz);
            f0 = _mm256_add_ps(f0, biasValue);
            f1 = _mm256_add_ps(f1, biasValue);
            f2 = _mm256_add_ps(f2, biasValue);
            f3 = _mm256_add_ps(f3, biasValue);
            if (post->useInt8 == 0) {
                f0 = _mm256_min_ps(f0, fp32max);
                f1 = _mm256_min_ps(f1, fp32max);
                f2 = _mm256_min_ps(f2, fp32max);
                f3 = _mm256_min_ps(f3, fp32max);
                f0 = _mm256_max_ps(f0, fp32min);
                f1 = _mm256_max_ps(f1, fp32min);
                f2 = _mm256_max_ps(f2, fp32min);
                f3 = _mm256_max_ps(f3, fp32min);
                _mm256_storeu_ps(((float*)dst_x) + 0 * AVX2_PACKINT8, f0);
                _mm256_storeu_ps(((float*)dst_x) + 1 * AVX2_PACKINT8, f1);
                _mm256_storeu_ps(((float*)dst_x) + 2 * AVX2_PACKINT8, f2);
                _mm256_storeu_ps(((float*)dst_x) + 3 * AVX2_PACKINT8, f3);
            } else {
                POSTTREAT(0);
                POSTTREAT(1);
                POSTTREAT(2);
                POSTTREAT(3);
            }
        }
        return;
    }
    if (3 == realDst) {
        for (int dz = 0; dz < dst_depth_quad; ++dz) {
            const auto weight_dz = weight + dz * src_depth_quad * (GEMMINT8_AVX2_L * GEMMINT8_AVX2_H);
            const auto bias_dz = post->biasFloat + dz * AVX2_PACKINT8;
            const auto weightBias_dz = post->weightQuanBias + dz * AVX2_PACKINT8;
            const float* scale_dz = post->scale + dz * AVX2_PACKINT8;
            auto dst_z           = dst + dz * dst_step_tmp;
            const auto src_x   = src;
            auto dst_x         = dst_z;
            __m256i D00 = _mm256_set1_epi32(0);
            __m256i D01 = _mm256_set1_epi32(0);
            __m256i D02 = _mm256_set1_epi32(0);

            for (int sz = 0; sz < src_depth_quad; ++sz) {
                const auto weight_sz = weight_dz + sz * (GEMMINT8_AVX2_L * GEMMINT8_AVX2_H);
                const auto src_z     = src_x + sz * GEMMINT8_AVX2_L * GEMMINT8_AVX2_E;
                auto w0 = _mm256_loadu_si256((__m256i*)weight_sz);

                auto s0 = _mm256_castps_si256(_mm256_broadcast_ss((float*)src_z + 0));
                auto s1 = _mm256_castps_si256(_mm256_broadcast_ss((float*)src_z + 1));
                auto s2 = _mm256_castps_si256(_mm256_broadcast_ss((float*)src_z + 2));

                D00 = _mm256_add_epi32(D00, _mm256_madd_epi16(_mm256_maddubs_epi16(s0, w0), oneValue));
                D01 = _mm256_add_epi32(D01, _mm256_madd_epi16(_mm256_maddubs_epi16(s1, w0), oneValue));
                D02 = _mm256_add_epi32(D02, _mm256_madd_epi16(_mm256_maddubs_epi16(s2, w0), oneValue));
            }
            auto D0 = D00;
            auto D1 = D01;
            auto D2 = D02;

            // auto biasValue0 = _mm256_loadu_si256((__m256i*)(bias_dz));
            auto weightBiasValue = _mm256_loadu_ps((float*)weightBias_dz);
            // D0 = _mm256_add_epi32(D0, biasValue0);
            // D1 = _mm256_add_epi32(D1, biasValue0);
            // D2 = _mm256_add_epi32(D2, biasValue0);

            auto scaleValue = _mm256_loadu_ps(scale_dz);
            
            auto f0 = _mm256_cvtepi32_ps(D0);
            auto f1 = _mm256_cvtepi32_ps(D1);
            auto f2 = _mm256_cvtepi32_ps(D2);
            // x_kernelSum x w_quantZero
            auto xy0_0 = _mm256_mul_ps(kernelSum0, weightBiasValue); // x dimemsion first
            auto xy0_1 = _mm256_mul_ps(kernelSum1, weightBiasValue); // ..second
            auto xy0_2 = _mm256_mul_ps(kernelSum2, weightBiasValue); // .. third
            f0 = _mm256_mul_ps(f0, scaleValue);
            f1 = _mm256_mul_ps(f1, scaleValue);
            f2 = _mm256_mul_ps(f2, scaleValue);
            f0 = _mm256_add_ps(f0, xy0_0);
            f1 = _mm256_add_ps(f1, xy0_1);
            f2 = _mm256_add_ps(f2, xy0_2);
            auto biasValue       = _mm256_loadu_ps(bias_dz);
            f0 = _mm256_add_ps(f0, biasValue);
            f1 = _mm256_add_ps(f1, biasValue);
            f2 = _mm256_add_ps(f2, biasValue);
            if (post->useInt8 == 0) {
                f0 = _mm256_min_ps(f0, fp32max);
                f1 = _mm256_min_ps(f1, fp32max);
                f2 = _mm256_min_ps(f2, fp32max);
                f0 = _mm256_max_ps(f0, fp32min);
                f1 = _mm256_max_ps(f1, fp32min);
                f2 = _mm256_max_ps(f2, fp32min);
                _mm256_storeu_ps(((float*)dst_x) + 0 * AVX2_PACKINT8, f0);
                _mm256_storeu_ps(((float*)dst_x) + 1 * AVX2_PACKINT8, f1);
                _mm256_storeu_ps(((float*)dst_x) + 2 * AVX2_PACKINT8, f2);
            } else {
                POSTTREAT(0);
                POSTTREAT(1);
                POSTTREAT(2);
            }
        }
        return;
    }    
    if (2 == realDst) {
        for (int dz = 0; dz < dst_depth_quad; ++dz) {
            const auto weight_dz = weight + dz * src_depth_quad * (GEMMINT8_AVX2_L * GEMMINT8_AVX2_H);
            const auto bias_dz = post->biasFloat + dz * AVX2_PACKINT8;
            const auto weightBias_dz = post->weightQuanBias + dz * AVX2_PACKINT8;
            const float* scale_dz = post->scale + dz * AVX2_PACKINT8;
            auto dst_z           = dst + dz * dst_step_tmp;
            const auto src_x   = src;
            auto dst_x         = dst_z;
            __m256i D00 = _mm256_set1_epi32(0);
            __m256i D01 = _mm256_set1_epi32(0);

            for (int sz = 0; sz < src_depth_quad; ++sz) {
                const auto weight_sz = weight_dz + sz * (GEMMINT8_AVX2_L * GEMMINT8_AVX2_H);
                const auto src_z     = src_x + sz * GEMMINT8_AVX2_L * GEMMINT8_AVX2_E;
                auto w0 = _mm256_loadu_si256((__m256i*)weight_sz);

                auto s0 = _mm256_castps_si256(_mm256_broadcast_ss((float*)src_z + 0));
                auto s1 = _mm256_castps_si256(_mm256_broadcast_ss((float*)src_z + 1));

                D00 = _mm256_add_epi32(D00, _mm256_madd_epi16(_mm256_maddubs_epi16(s0, w0), oneValue));
                D01 = _mm256_add_epi32(D01, _mm256_madd_epi16(_mm256_maddubs_epi16(s1, w0), oneValue));
            }
            auto D0 = D00;
            auto D1 = D01;

            auto weightBiasValue = _mm256_loadu_ps((float*)weightBias_dz);

            auto scaleValue = _mm256_loadu_ps(scale_dz);
            auto f0 = _mm256_cvtepi32_ps(D0);
            auto f1 = _mm256_cvtepi32_ps(D1);
            // x_kernelSum x w_quantZero
            auto xy0_0 = _mm256_mul_ps(kernelSum0, weightBiasValue); // x dimemsion first
            auto xy0_1 = _mm256_mul_ps(kernelSum1, weightBiasValue); // ..second
            f0 = _mm256_mul_ps(f0, scaleValue);
            f1 = _mm256_mul_ps(f1, scaleValue);
            f0 = _mm256_add_ps(f0, xy0_0);
            f1 = _mm256_add_ps(f1, xy0_1);
            auto biasValue       = _mm256_loadu_ps(bias_dz);
            f0 = _mm256_add_ps(f0, biasValue);
            f1 = _mm256_add_ps(f1, biasValue);
            if (post->useInt8 == 0) {
                f0 = _mm256_min_ps(f0, fp32max);
                f1 = _mm256_min_ps(f1, fp32max);
                f0 = _mm256_max_ps(f0, fp32min);
                f1 = _mm256_max_ps(f1, fp32min);
                _mm256_storeu_ps(((float*)dst_x) + 0 * AVX2_PACKINT8, f0);
                _mm256_storeu_ps(((float*)dst_x) + 1 * AVX2_PACKINT8, f1);
            } else {
                POSTTREAT(0);
                POSTTREAT(1);
            }
        }
        return;
    }    
    if (1 == realDst) {
        for (int dz = 0; dz < dst_depth_quad; ++dz) {
            const auto weight_dz = weight + dz * src_depth_quad * (GEMMINT8_AVX2_L * GEMMINT8_AVX2_H);
            const auto bias_dz = post->biasFloat + dz * AVX2_PACKINT8;
            const auto weightBias_dz = post->weightQuanBias + dz * AVX2_PACKINT8;
            const float* scale_dz = post->scale + dz * AVX2_PACKINT8;
            auto dst_z           = dst + dz * dst_step_tmp;
            const auto src_x   = src;
            auto dst_x         = dst_z;
            __m256i D00 = _mm256_set1_epi32(0);

            for (int sz = 0; sz < src_depth_quad; ++sz) {
                const auto weight_sz = weight_dz + sz * (GEMMINT8_AVX2_L * GEMMINT8_AVX2_H);
                const auto src_z     = src_x + sz * GEMMINT8_AVX2_L * GEMMINT8_AVX2_E;
                auto w0 = _mm256_loadu_si256((__m256i*)weight_sz);

                auto s0 = _mm256_castps_si256(_mm256_broadcast_ss((float*)src_z + 0));

                D00 = _mm256_add_epi32(D00, _mm256_madd_epi16(_mm256_maddubs_epi16(s0, w0), oneValue));
            }
            auto D0 = D00;
            auto weightBiasValue = _mm256_loadu_ps((float*)weightBias_dz);

            auto scaleValue = _mm256_loadu_ps(scale_dz);
            auto f0 = _mm256_cvtepi32_ps(D0);
            // x_kernelSum x w_quantZero
            auto xy0_0 = _mm256_mul_ps(kernelSum0, weightBiasValue); // x dimemsion first
            f0 = _mm256_mul_ps(f0, scaleValue);
            f0 = _mm256_add_ps(f0, xy0_0);
            auto biasValue       = _mm256_loadu_ps(bias_dz);
            f0 = _mm256_add_ps(f0, biasValue);
            if (post->useInt8 == 0) {
                f0 = _mm256_min_ps(f0, fp32max);
                f0 = _mm256_max_ps(f0, fp32min);
                _mm256_storeu_ps(((float*)dst_x) + 0 * AVX2_PACKINT8, f0);
            } else {
                POSTTREAT(0);
            }
        }
        return;
    }
}

#undef MAIN_COMPUTE
#undef STORE_TEMP
void _AVX_MNNLineDepthWiseInt8AddBiasScaleUnit(int8_t* dstO, const int8_t* srcO, const int8_t* weightO, const QuanPostTreatParameters* parameters, size_t width, size_t src_w_step, size_t fw, size_t fh, size_t dilateX_step, size_t dilateY_step, int8_t* idxOrder) {
    int pack = 16;
    auto dst = dstO;
    auto src = (const int16_t*)srcO;
    auto weight = (const int16_t*)weightO;
    auto biasValue0 = _mm256_castps_si256(_mm256_loadu_ps((const float*)parameters->bias));
    auto biasValue1 = _mm256_castps_si256(_mm256_loadu_ps((const float*)parameters->bias + 8));

    auto scaleValue0 = _mm256_loadu_ps((const float*)parameters->scale);
    auto scaleValue1 = _mm256_loadu_ps((const float*)parameters->scale + 8);
    __m256i srcValue0;
    __m256i zero = _mm256_xor_si256(srcValue0, srcValue0);
    __m256i d0, d1;
    int dx, fx, fy;
    __m256 zero256 = _mm256_set1_ps(0.0f);
    auto minValue = _mm256_set1_epi16((int16_t)(parameters->minValue + 128));
    auto maxValue = _mm256_set1_epi16((int16_t)(parameters->maxValue + 128));
    __m256 plus = _mm256_set1_ps(0.5f);
    __m256 minus = _mm256_set1_ps(-0.5f);
    auto offset = _mm256_set1_epi32(128);

    for (dx = 0; dx < width; ++dx) {
        d0 = biasValue0;
        d1 = biasValue1;

        auto dst_x          = dst;
        const auto src_z    = src;
        for (fy = 0; fy < fh; ++fy) {
            const auto src_y    = src_z + fy * dilateY_step;
            const auto weight_y = weight + fy * fw * pack;
            for (fx = 0; fx < fw; ++fx) {
                const auto src_x    = src_y + fx * dilateX_step;
                auto s0_16 = _mm256_castps_si256(_mm256_loadu_ps((float*)src_x));
                s0_16 = _mm256_permute4x64_epi64(s0_16, 0xD8); // Reorder 0,1,2,3->0,2,1,3 to ensure s0_32 is 0,1 and s1_32 is 2,3.
                auto s0_32 = _mm256_unpacklo_epi16(s0_16, zero);
                auto s1_32 = _mm256_unpackhi_epi16(s0_16, zero);

                const auto weight_x = weight_y + pack * fx;
                auto w0_16 = _mm256_castps_si256(_mm256_loadu_ps((float*)weight_x));
                w0_16 = _mm256_permute4x64_epi64(w0_16, 0xD8);
                auto w0_32 = _mm256_unpacklo_epi16(w0_16, zero);
                auto w1_32 = _mm256_unpackhi_epi16(w0_16, zero);

                d0 = _mm256_add_epi32(d0, _mm256_madd_epi16(w0_32, s0_32));
                d1 = _mm256_add_epi32(d1, _mm256_madd_epi16(w1_32, s1_32));
            }
        }
        __m256 f0 = _mm256_cvtepi32_ps(d0);
        __m256 f1 = _mm256_cvtepi32_ps(d1);
        f0 = _mm256_mul_ps(f0, scaleValue0);
        f1 = _mm256_mul_ps(f1, scaleValue1);
        auto m0 = _mm256_cmp_ps(f0, zero256, 1);
        auto m1 = _mm256_cmp_ps(f1, zero256, 1);
        m0 = _mm256_blendv_ps(plus, minus, m0);
        m1 = _mm256_blendv_ps(plus, minus, m1);
        f0 = _mm256_add_ps(f0, m0);
        f1 = _mm256_add_ps(f1, m1);
        // _MM_FROUND_TO_ZERO
        d0 = _mm256_cvtps_epi32(_mm256_round_ps(f0, 3));
        d1 = _mm256_cvtps_epi32(_mm256_round_ps(f1, 3));
        d0 = _mm256_add_epi32(d0, offset);
        d1 = _mm256_add_epi32(d1, offset);
        
        d0 = _mm256_permute4x64_epi64(_mm256_packs_epi32(d0, d1), 0xD8);
        d0 = _mm256_min_epi16(d0, maxValue);
        d0 = _mm256_max_epi16(d0, minValue);
        auto y256i = _mm256_permute4x64_epi64(_mm256_packus_epi16(d0, _mm256_setzero_si256()), 0xD8);
        auto y128 = _mm_castsi128_ps(_mm256_extracti128_si256(y256i, 0));
        _mm_storeu_ps((float*)dst, y128);
        dst += 16;
        src += src_w_step;
    }
}

void _AVX_MNNFloat2Int8(const float* src, int8_t* dst, size_t sizeQuad, const float* scalep, ssize_t minV, ssize_t maxV, ssize_t zeroPoint) {
    auto zero = _mm256_set1_epi32(0);
    auto minValue = _mm256_set1_ps(minV);
    auto maxValue = _mm256_set1_ps(maxV);
    auto zeroPointValue = _mm256_set1_ps(zeroPoint);
    auto offset = _mm256_set1_epi32(128);
    auto plus = _mm256_set1_ps(0.5f);
    auto minus = _mm256_set1_ps(-0.5f);
    auto scaleValue = _mm256_loadu_ps(scalep);

    for (int i = 0; i < sizeQuad; ++i) {
        auto f0 = _mm256_loadu_ps(src + 8 * i);
        f0 = _mm256_mul_ps(f0, scaleValue);
        f0 = _mm256_add_ps(f0, zeroPointValue);
        f0 = _mm256_min_ps(f0, maxValue);
        f0 = _mm256_max_ps(f0, minValue);
        auto m0 = _mm256_cmp_ps(f0, _mm256_castsi256_ps(zero), 1);
        m0 = _mm256_blendv_ps(plus, minus, m0);
        f0 = _mm256_add_ps(f0, m0);
        // 3: _MM_FROUND_TO_ZERO
        auto d0 = _mm256_cvtps_epi32(_mm256_round_ps(f0, 3));
        d0 = _mm256_add_epi32(d0, offset);
        d0 = _mm256_packs_epi32(d0, _mm256_setzero_si256());
        d0 = _mm256_permute4x64_epi64(d0, 0xD8);
#if defined(_MSC_VER)
        __m256i x = static_cast<__m256i>(_mm256_packus_epi16(d0, _mm256_setzero_si256()));
        *((int64_t*)dst + i) = x.m256i_i64[0];
#else
         __v4di x = static_cast<__v4di>(_mm256_packus_epi16(d0, _mm256_setzero_si256()));
         *((int64_t*)dst + i) = x[0];
#endif
    }
}

void _AVX_MNNInt8ScaleToFloat(float* dst, const int8_t* src, const float* scale, size_t sizeQuad, ssize_t zeroPoint) {
    auto sizeC4 = sizeQuad / 4;
    auto sizeRemain = sizeQuad % 4;
    auto zero = _mm256_set1_epi32(0);
    auto scaleValue = _mm256_loadu_ps(scale);
    auto zeroPointValue = _mm256_set1_epi32(zeroPoint + 128);
    for (int i = 0; i < sizeC4; ++i) {
        auto s = _mm256_castps_si256(_mm256_loadu_ps((const float*)(src)));
        auto s0_16 = _mm256_permute4x64_epi64(_mm256_unpacklo_epi8(s, zero), 0XD8);
        auto s1_16 = _mm256_permute4x64_epi64(_mm256_unpackhi_epi8(s, zero), 0xD8);
        auto s0_32 = _mm256_unpacklo_epi16(s0_16, zero);
        auto s1_32 = _mm256_unpacklo_epi16(s1_16, zero);
        auto s2_32 = _mm256_unpackhi_epi16(s0_16, zero);
        auto s3_32 = _mm256_unpackhi_epi16(s1_16, zero);
        s0_32 = _mm256_sub_epi32(s0_32, zeroPointValue);
        s1_32 = _mm256_sub_epi32(s1_32, zeroPointValue);
        s2_32 = _mm256_sub_epi32(s2_32, zeroPointValue);
        s3_32 = _mm256_sub_epi32(s3_32, zeroPointValue);
        auto s0_f = _mm256_cvtepi32_ps(s0_32);
        auto s1_f = _mm256_cvtepi32_ps(s1_32);
        auto s2_f = _mm256_cvtepi32_ps(s2_32);
        auto s3_f = _mm256_cvtepi32_ps(s3_32);
        _mm256_storeu_ps(dst + 8 * 0, _mm256_mul_ps(s0_f, scaleValue));
        _mm256_storeu_ps(dst + 8 * 1, _mm256_mul_ps(s1_f, scaleValue));
        _mm256_storeu_ps(dst + 8 * 2, _mm256_mul_ps(s2_f, scaleValue));
        _mm256_storeu_ps(dst + 8 * 3, _mm256_mul_ps(s3_f, scaleValue));
        src += 32;
        dst += 32;
    }
    if (sizeRemain > 0) {
        int8_t srcTemp[256];
        ::memcpy(srcTemp, src, sizeRemain * 8);
        auto s = _mm256_castps_si256(_mm256_loadu_ps((const float*)(srcTemp)));
        auto s0_16 = _mm256_permute4x64_epi64(_mm256_unpacklo_epi8(s, zero), 0XD8);
        auto s1_16 = _mm256_permute4x64_epi64(_mm256_unpackhi_epi8(s, zero), 0xD8);
        auto s0_32 = _mm256_unpacklo_epi16(s0_16, zero);
        auto s1_32 = _mm256_unpacklo_epi16(s1_16, zero);
        auto s2_32 = _mm256_unpackhi_epi16(s0_16, zero);
        auto s3_32 = _mm256_unpackhi_epi16(s1_16, zero);
        s0_32 = _mm256_sub_epi32(s0_32, zeroPointValue);
        s1_32 = _mm256_sub_epi32(s1_32, zeroPointValue);
        s2_32 = _mm256_sub_epi32(s2_32, zeroPointValue);
        s3_32 = _mm256_sub_epi32(s3_32, zeroPointValue);
        auto s0_f = _mm256_cvtepi32_ps(s0_32);
        auto s1_f = _mm256_cvtepi32_ps(s1_32);
        auto s2_f = _mm256_cvtepi32_ps(s2_32);
        auto s3_f = _mm256_cvtepi32_ps(s3_32);
        switch (sizeRemain) {
            case 3:
                _mm256_storeu_ps(dst + 8 * 0, _mm256_mul_ps(s0_f, scaleValue));
                _mm256_storeu_ps(dst + 8 * 1, _mm256_mul_ps(s1_f, scaleValue));
                _mm256_storeu_ps(dst + 8 * 2, _mm256_mul_ps(s2_f, scaleValue));
                break;
            case 2:
                _mm256_storeu_ps(dst + 8 * 0, _mm256_mul_ps(s0_f, scaleValue));
                _mm256_storeu_ps(dst + 8 * 1, _mm256_mul_ps(s1_f, scaleValue));
                break;
            case 1:
                _mm256_storeu_ps(dst + 8 * 0, _mm256_mul_ps(s0_f, scaleValue));
                break;
            default:
                break;
        }
    }
}

static void _AVX2_MNNGetGemmUnit(int* UNIT, int* SRC_UNIT, int* DST_XUNIT) {
    *UNIT = GEMMINT8_AVX2_H;
    *SRC_UNIT = GEMMINT8_AVX2_L;
    *DST_XUNIT = GEMMINT8_AVX2_E;
}

static void _AVXMNNPackC4ForMatMul_A(int8_t* destOrigin, int8_t const** sourceGroup, const int32_t* info, const int32_t* el) {
    int number = info[0];
    int eReal = info[1];
    int xStride = info[3];
    int xS4 = xStride * AVX2_PACKINT8 / sizeof(int32_t);
    int PUNIT = AVX2_PACKINT8 / GEMMINT8_AVX2_L;
    int FLOATPACK = AVX2_PACKINT8 / sizeof(int32_t);
    int eOutsideStride = info[2] / sizeof(int32_t);
    const int EP = GEMMINT8_AVX2_E;
    int eDest = EP;
    const int LP = GEMMINT8_AVX2_L;
    for (int n=0; n<number; ++n) {
        int e = el[4 * n + 0];
        int l = el[4 * n + 1];
        int eOffset = el[4 * n + 2];
        int lOffset = el[4 * n + 3];
        int eC = eOffset / eDest;
        int eR = eOffset % eDest;
        auto source = (int32_t*)sourceGroup[n];
        auto dest = (int32_t*)(destOrigin + eC * info[2] + eR * LP + lOffset * EP);
        //printf("e=%d, l=%d, eOffset=%d, lOffset=%d, eDest=%d\n", e, l, eOffset, lOffset, eDest);
        l = l / 4; // Use float instead of int8 * 4
        int eS = eDest - eR;
        for (int x = 0; x < l; ++x) {
            int eRemain = e;
            auto xR                  = x % PUNIT;
            auto xC                  = x / PUNIT;
            auto d = dest + x * eDest;
            auto s = source + xC * eReal * FLOATPACK + xR;
            if (eR > 0) {
                int eStep = ALIMIN(eRemain, eS);
                for (int yi=0; yi<eStep; ++yi) {
                    d[yi] = s[yi * xS4];
                }
                eRemain-=eStep;
                d += (eOutsideStride - eR);
                s += eS * xS4;
            }
            while (eRemain > 0) {
                int eStep = ALIMIN(eDest, eRemain);
                for (int yi=0; yi<eStep; ++yi) {
                    d[yi] = s[yi * xS4];
                }
                eRemain-=eStep;
                d+= eOutsideStride;
                s+= eStep * xS4;
            }
        }
    }
}

void _AVX_MNNInt8FunctionInit(void* functions) {
    auto gAVX2CoreInt8Functions = (MNN::CoreInt8Functions*)functions;
    // MatMul
    gAVX2CoreInt8Functions->Int8GemmKernel = _AVX_MNNGemmInt8AddBiasScale_16x4_Unit;
    gAVX2CoreInt8Functions->Int8GemmKernelFast = _AVX_MNNGemmInt8AddBiasScale_16x4_Unit_Fast;
    gAVX2CoreInt8Functions->MNNGetGemmUnit = _AVX2_MNNGetGemmUnit;
    gAVX2CoreInt8Functions->MNNPackC4Int8ForMatMul_A = _AVXMNNPackC4ForMatMul_A;
#ifdef MNN_LOW_MEMORY
    gAVX2CoreInt8Functions->Int8GemmKernel_W4 = _AVX_MNNGemmInt8AddBiasScale_16x4_w4;
#endif

    // Int8 <-> Float
    gAVX2CoreInt8Functions->MNNFloat2Int8 = _AVX_MNNFloat2Int8;
    gAVX2CoreInt8Functions->MNNInt8ScaleToFloat = _AVX_MNNInt8ScaleToFloat;

    // conv depthwise
    gAVX2CoreInt8Functions->ConvDepthwiseLineInt8 = _AVX_MNNLineDepthWiseInt8AddBiasScaleUnit;
}
