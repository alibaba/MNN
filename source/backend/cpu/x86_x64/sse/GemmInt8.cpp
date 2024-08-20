//
//  GemmInt8.cpp
//  MNN
//
//  Created by MNN on b'2021/07/09'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "GemmCommon.hpp"
#include "FunctionSummary.hpp"
#include "core/Macro.h"
#include "backend/cpu/compute/CommonOptFunction.h"
#include <algorithm>
#include <cmath>

// require SSE 4.1
void _SSE_MNNGemmInt8AddBiasScale_16x4_Unit(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step,
                                            size_t dst_depth_quad, const QuanPostTreatParameters* post, size_t realDst) {
    const auto dst_step_tmp = dst_step / sizeof(int8_t);
    __m128i zero = _mm_set1_epi32(0);
    __m128 minValue = _mm_set1_ps(post->minValue);
    __m128 maxValue = _mm_set1_ps(post->maxValue);
    __m128 plus = _mm_set1_ps(0.5f);
    __m128 minus = _mm_set1_ps(-0.5f);
    __m128 fp32min, fp32max;
    if (0 == post->useInt8 && post->fp32minmax) {
        fp32min = _mm_set1_ps((post->fp32minmax)[0]);
        fp32max = _mm_set1_ps((post->fp32minmax)[1]);
    }
    auto oneValue = _mm_set1_epi16(1);
    auto offset = _mm_set1_epi32(128);
    auto f128 = _mm_set1_ps(128.f);
    auto srcKernelSumPtr = post->srcKernelSum;
    __m128 kernelSum0 = _mm_setzero_ps();
    __m128 kernelSum1 = _mm_setzero_ps();
    __m128 kernelSum2 = _mm_setzero_ps();
    __m128 kernelSum3 = _mm_setzero_ps();
    if (GEMM_INT8_DST_XUNIT == realDst) {
        kernelSum0 = _mm_load_ps1(post->srcKernelSum);
        kernelSum1 = _mm_load_ps1(post->srcKernelSum + 1);
        kernelSum2 = _mm_load_ps1(post->srcKernelSum + 2);
        kernelSum3 = _mm_load_ps1(post->srcKernelSum + 3);
    } else {
        kernelSum0 = _mm_load_ps1(post->srcKernelSum);
        if (realDst > 1) {
            kernelSum1 = _mm_load_ps1(post->srcKernelSum + 1);
        }
        if (realDst > 2) {
            kernelSum2 = _mm_load_ps1(post->srcKernelSum + 2);
        }
    }
    __m128 extrascale0 = _mm_setzero_ps();
    __m128 extrascale1 = _mm_setzero_ps();
    __m128 extrascale2 = _mm_setzero_ps();
    __m128 extrascale3 = _mm_setzero_ps();
    if (post->extraScale) {
        if (GEMM_INT8_DST_XUNIT == realDst) {
            extrascale0 = _mm_load_ps1(post->extraScale);
            extrascale1 = _mm_load_ps1(post->extraScale + 1);
            extrascale2 = _mm_load_ps1(post->extraScale + 2);
            extrascale3 = _mm_load_ps1(post->extraScale + 3);
        } else {
            extrascale0 = _mm_load_ps1(post->extraScale);
            if (realDst > 1) {
                extrascale1 = _mm_load_ps1(post->extraScale + 1);
            }
            if (realDst > 2) {
                extrascale2 = _mm_load_ps1(post->extraScale + 2);
            }
        }
    }
    const float* biasPtr = nullptr;
    if (post->biasFloat) {
        biasPtr = post->biasFloat;
    }
    auto blockNum = post->blockNum;
    for (int dz = 0; dz < dst_depth_quad; ++dz) {
        const auto weight_dz = weight + dz * (src_depth_quad * blockNum) * (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT);
        const auto weightBias_dz = post->weightQuanBias + dz * GEMM_INT8_UNIT;
        const float* scale_dz = nullptr;
        scale_dz  = post->scale + dz * GEMM_INT8_UNIT;
        auto dst_z           = dst + dz * dst_step_tmp;
        const auto src_x   = src;
        auto dst_x         = dst_z;
        __m128i d0 = _mm_set1_epi32(0);
        __m128i d1 = _mm_set1_epi32(0);
        __m128i d2 = _mm_set1_epi32(0);
        __m128i d3 = _mm_set1_epi32(0);

        __m128i e0 = _mm_set1_epi32(0);
        __m128i e1 = _mm_set1_epi32(0);
        __m128i e2 = _mm_set1_epi32(0);
        __m128i e3 = _mm_set1_epi32(0);

        __m128i D0 = _mm_set1_epi32(0);
        __m128i D1 = _mm_set1_epi32(0);
        __m128i D2 = _mm_set1_epi32(0);
        __m128i D3 = _mm_set1_epi32(0);

        __m128i E0 = _mm_set1_epi32(0);
        __m128i E1 = _mm_set1_epi32(0);
        __m128i E2 = _mm_set1_epi32(0);
        __m128i E3 = _mm_set1_epi32(0);

        for (int sz = 0; sz < src_depth_quad; ++sz) {
            const auto weight_sz = weight_dz + (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT) * sz;
            const auto src_z     = src_x + sz * GEMM_INT8_DST_XUNIT * GEMM_INT8_SRC_UNIT;
            auto w0 = _mm_loadu_si128((__m128i*)(weight_sz + GEMM_INT8_SRC_UNIT * 0));
            auto w1 = _mm_loadu_si128((__m128i*)(weight_sz + GEMM_INT8_SRC_UNIT * 1));
            auto w2 = _mm_loadu_si128((__m128i*)(weight_sz + GEMM_INT8_SRC_UNIT * 2));
            auto w3 = _mm_loadu_si128((__m128i*)(weight_sz + GEMM_INT8_SRC_UNIT * 3));

            auto s0 = _mm_loadu_si128((__m128i*)(src_z + GEMM_INT8_SRC_UNIT * 0));
            auto s1 = _mm_loadu_si128((__m128i*)(src_z + GEMM_INT8_SRC_UNIT * 1));
            auto s2 = _mm_loadu_si128((__m128i*)(src_z + GEMM_INT8_SRC_UNIT * 2));
            auto s3 = _mm_loadu_si128((__m128i*)(src_z + GEMM_INT8_SRC_UNIT * 3));

//#define COMPUTE(i, j)\
//auto d##i##j = _mm_maddubs_epi16(s##i, w##j);\
//d##i##j = _mm_madd_epi16(d##i##j, oneValue);\

#define COMPUTE(i, j)\
auto W##i##j##0 = _mm_srai_epi16(_mm_unpacklo_epi8(zero, w##j), 8);\
auto W##i##j##1 = _mm_srai_epi16(_mm_unpackhi_epi8(zero, w##j), 8);\
auto S##i##j##0 = _mm_unpacklo_epi8(s##i, zero);\
auto S##i##j##1 = _mm_unpackhi_epi8(s##i, zero);\
auto d##i##j = _mm_add_epi32(_mm_madd_epi16(S##i##j##0, W##i##j##0), _mm_madd_epi16(S##i##j##1, W##i##j##1));\

            COMPUTE(0, 0);
            COMPUTE(0, 1);
            COMPUTE(0, 2);
            COMPUTE(0, 3);
            COMPUTE(1, 0);
            COMPUTE(1, 1);
            COMPUTE(1, 2);
            COMPUTE(1, 3);
            COMPUTE(2, 0);
            COMPUTE(2, 1);
            COMPUTE(2, 2);
            COMPUTE(2, 3);
            COMPUTE(3, 0);
            COMPUTE(3, 1);
            COMPUTE(3, 2);
            COMPUTE(3, 3);

            d0 = _mm_add_epi32(d0, d00);
            d1 = _mm_add_epi32(d1, d01);
            d2 = _mm_add_epi32(d2, d02);
            d3 = _mm_add_epi32(d3, d03);

            e0 = _mm_add_epi32(e0, d10);
            e1 = _mm_add_epi32(e1, d11);
            e2 = _mm_add_epi32(e2, d12);
            e3 = _mm_add_epi32(e3, d13);

            D0 = _mm_add_epi32(D0, d20);
            D1 = _mm_add_epi32(D1, d21);
            D2 = _mm_add_epi32(D2, d22);
            D3 = _mm_add_epi32(D3, d23);

            E0 = _mm_add_epi32(E0, d30);
            E1 = _mm_add_epi32(E1, d31);
            E2 = _mm_add_epi32(E2, d32);
            E3 = _mm_add_epi32(E3, d33);
        }
        d0 = _mm_hadd_epi32(d0, d1);
        d1 = _mm_hadd_epi32(d2, d3);
        d0 = _mm_hadd_epi32(d0, d1);

        e0 = _mm_hadd_epi32(e0, e1);
        e1 = _mm_hadd_epi32(e2, e3);
        d1 = _mm_hadd_epi32(e0, e1);

        D0 = _mm_hadd_epi32(D0, D1);
        D1 = _mm_hadd_epi32(D2, D3);
        d2 = _mm_hadd_epi32(D0, D1);

        E0 = _mm_hadd_epi32(E0, E1);
        E1 = _mm_hadd_epi32(E2, E3);
        d3 = _mm_hadd_epi32(E0, E1);
        auto scaleValue = _mm_loadu_ps(scale_dz);
        // auto biasValue = _mm_loadu_si128((__m128i*)(bias_dz));
        // d0 = _mm_add_epi32(d0, biasValue);
        // d1 = _mm_add_epi32(d1, biasValue);
        // d2 = _mm_add_epi32(d2, biasValue);
        // d3 = _mm_add_epi32(d3, biasValue);
        //auto biasValue = _mm_loadu_ps((float*)(bias_dz));
        auto weightBiasValue = _mm_loadu_ps((float*)weightBias_dz);
        __m128 f0 = _mm_cvtepi32_ps(d0);
        __m128 f1 = _mm_cvtepi32_ps(d1);
        __m128 f2 = _mm_cvtepi32_ps(d2);
        __m128 f3 = _mm_cvtepi32_ps(d3);
        // x_kernelSum x w_quantZero
        auto xy0_0 = _mm_mul_ps(kernelSum0, weightBiasValue); // x dimemsion first
        auto xy0_1 = _mm_mul_ps(kernelSum1, weightBiasValue); // ..second
        auto xy0_2 = _mm_mul_ps(kernelSum2, weightBiasValue); // .. third
        auto xy0_3 = _mm_mul_ps(kernelSum3, weightBiasValue); // ..fourth
        f0 = _mm_mul_ps(f0, scaleValue);
        f1 = _mm_mul_ps(f1, scaleValue);
        f2 = _mm_mul_ps(f2, scaleValue);
        f3 = _mm_mul_ps(f3, scaleValue);
        if (post->extraScale) {
            f0 = _mm_mul_ps(f0, extrascale0);
            f1 = _mm_mul_ps(f1, extrascale1);
            f2 = _mm_mul_ps(f2, extrascale2);
            f3 = _mm_mul_ps(f3, extrascale3);
            if (post->extraBias && nullptr != biasPtr) {
                auto extraB = post->extraBias + dz * GEMM_INT8_UNIT;
                auto extrabias = _mm_loadu_ps(extraB);
                extrabias = _mm_mul_ps(f128, extrabias);
                auto extrabias0 = _mm_mul_ps(extrabias, extrascale0);
                auto extrabias1 = _mm_mul_ps(extrabias, extrascale1);
                auto extrabias2 = _mm_mul_ps(extrabias, extrascale2);
                auto extrabias3 = _mm_mul_ps(extrabias, extrascale3);
                f0 = _mm_sub_ps(f0, extrabias0);
                f1 = _mm_sub_ps(f1, extrabias1);
                f2 = _mm_sub_ps(f2, extrabias2);
                f3 = _mm_sub_ps(f3, extrabias3);
            }
        }
        f0 = _mm_add_ps(f0, xy0_0);
        f1 = _mm_add_ps(f1, xy0_1);
        f2 = _mm_add_ps(f2, xy0_2);
        f3 = _mm_add_ps(f3, xy0_3);
        if (nullptr != biasPtr) {
            const auto bias_dz   = biasPtr + dz * GEMM_INT8_UNIT;
            auto biasValue = _mm_loadu_ps(bias_dz);
            f0 = _mm_add_ps(f0, biasValue);
            f1 = _mm_add_ps(f1, biasValue);
            f2 = _mm_add_ps(f2, biasValue);
            f3 = _mm_add_ps(f3, biasValue);
        }
        if (post->useInt8 == 1) {
            // for Relu Int8 activation
            f0 = _mm_min_ps(f0, maxValue);
            f1 = _mm_min_ps(f1, maxValue);
            f2 = _mm_min_ps(f2, maxValue);
            f3 = _mm_min_ps(f3, maxValue);
            f0 = _mm_max_ps(f0, minValue);
            f1 = _mm_max_ps(f1, minValue);
            f2 = _mm_max_ps(f2, minValue);
            f3 = _mm_max_ps(f3, minValue);
            auto m0 = _mm_cmplt_ps(f0, _mm_castsi128_ps(zero));
            auto m1 = _mm_cmplt_ps(f1, _mm_castsi128_ps(zero));
            auto m2 = _mm_cmplt_ps(f2, _mm_castsi128_ps(zero));
            auto m3 = _mm_cmplt_ps(f3, _mm_castsi128_ps(zero));
            m0 = _mm_blendv_ps(plus, minus, m0);
            m1 = _mm_blendv_ps(plus, minus, m1);
            m2 = _mm_blendv_ps(plus, minus, m2);
            m3 = _mm_blendv_ps(plus, minus, m3);
            f0 = _mm_add_ps(f0, m0);
            f1 = _mm_add_ps(f1, m1);
            f2 = _mm_add_ps(f2, m2);
            f3 = _mm_add_ps(f3, m3);
            // 3: _MM_FROUND_TO_ZERO
            d0 = _mm_cvtps_epi32(_mm_round_ps(f0, 3));
            d1 = _mm_cvtps_epi32(_mm_round_ps(f1, 3));
            d2 = _mm_cvtps_epi32(_mm_round_ps(f2, 3));
            d3 = _mm_cvtps_epi32(_mm_round_ps(f3, 3));
            d0 = _mm_add_epi32(d0, offset);
            d1 = _mm_add_epi32(d1, offset);
            d2 = _mm_add_epi32(d2, offset);
            d3 = _mm_add_epi32(d3, offset);

            // Int32 -> Int8
            d0 = _mm_packs_epi32(d0, d1);
            d2 = _mm_packs_epi32(d2, d3);
            d0 = _mm_packus_epi16(d0, d2);
            if (GEMM_INT8_DST_XUNIT == realDst) {
                _mm_storeu_ps((float*)dst_x, _mm_castsi128_ps(d0));
            } else {
                int32_t tempV[4];
                _mm_storeu_si128((__m128i*)tempV, d0);
                for (int j=0; j<realDst; ++j) {
                    ((int32_t*)dst_x)[j] = tempV[j];
                }
            }
        } else { // Store float values directly.
            // for Relu float activation.
            __m128 f[4] = {f0, f1, f2, f3};
            if (nullptr == biasPtr) {
                for (int j = 0; j < realDst; ++j) {
                    auto dstv = _mm_loadu_ps(((float*)dst_x) + j * 4);
                    f[j] = _mm_add_ps(dstv, f[j]);
                }
            }
            if (post->fp32minmax) {
                f[0] = _mm_min_ps(f[0], fp32max);
                f[1] = _mm_min_ps(f[1], fp32max);
                f[2] = _mm_min_ps(f[2], fp32max);
                f[3] = _mm_min_ps(f[3], fp32max);
                f[0] = _mm_max_ps(f[0], fp32min);
                f[1] = _mm_max_ps(f[1], fp32min);
                f[2] = _mm_max_ps(f[2], fp32min);
                f[3] = _mm_max_ps(f[3], fp32min);
            }
            for (int j = 0; j < realDst; ++j) {
                _mm_storeu_ps(((float*)dst_x) + j * 4, f[j]);
            }
        }
    }
}

static inline void _load_int4_to_int8(const uint8_t* src, int8_t* dst) {
    uint8_t c = 0xf;
    for (int i = 0; i < 32; ++i) {
        dst[2 * i] = (src[i] >> 4);
        dst[2 * i +1] = (src[i] & c);
    }
}
void _SSE_MNNGemmInt8AddBiasScale_16x4_w4(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step,
                                            size_t dst_depth_quad, const QuanPostTreatParameters* post, size_t realDst) {
    MNN_ASSERT(post->useInt8 == 0);
    const auto dst_step_tmp = dst_step / sizeof(int8_t);
    __m128i zero = _mm_set1_epi32(0);
    __m128 minValue = _mm_set1_ps(post->minValue);
    __m128 maxValue = _mm_set1_ps(post->maxValue);
    __m128 fp32min, fp32max;
    if (post->fp32minmax) {
        fp32min = _mm_set1_ps((post->fp32minmax)[0]);
        fp32max = _mm_set1_ps((post->fp32minmax)[1]);
    }
    const float* biasPtr = nullptr;
    if (post->biasFloat) {
        biasPtr = post->biasFloat;
    }
    int blockNum = post->blockNum;
    int weight_step_Z = 0.5 * (src_depth_quad * blockNum) * (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT);
    int weight_step_Y = 0.5 * (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT);

    auto oneValue = _mm_set1_epi16(1);
    auto offset = _mm_set1_epi32(128);
    auto srcKernelSumPtr = post->srcKernelSum;
    __m128 kernelSum0 = _mm_setzero_ps();
    __m128 kernelSum1 = _mm_setzero_ps();
    __m128 kernelSum2 = _mm_setzero_ps();
    __m128 kernelSum3 = _mm_setzero_ps();
    if (GEMM_INT8_DST_XUNIT == realDst) {
        kernelSum0 = _mm_load_ps1(post->srcKernelSum);
        kernelSum1 = _mm_load_ps1(post->srcKernelSum + 1);
        kernelSum2 = _mm_load_ps1(post->srcKernelSum + 2);
        kernelSum3 = _mm_load_ps1(post->srcKernelSum + 3);
    } else {
        kernelSum0 = _mm_load_ps1(post->srcKernelSum);
        if (realDst > 1) {
            kernelSum1 = _mm_load_ps1(post->srcKernelSum + 1);
        }
        if (realDst > 2) {
            kernelSum2 = _mm_load_ps1(post->srcKernelSum + 2);
        }
    }
    auto f128 = _mm_set1_ps(128.f);
    __m128 extrascale0 = _mm_setzero_ps();
    __m128 extrascale1 = _mm_setzero_ps();
    __m128 extrascale2 = _mm_setzero_ps();
    __m128 extrascale3 = _mm_setzero_ps();
    if (post->extraScale) {
        if (GEMM_INT8_DST_XUNIT == realDst) {
            extrascale0 = _mm_load_ps1(post->extraScale);
            extrascale1 = _mm_load_ps1(post->extraScale + 1);
            extrascale2 = _mm_load_ps1(post->extraScale + 2);
            extrascale3 = _mm_load_ps1(post->extraScale + 3);
        } else {
            extrascale0 = _mm_load_ps1(post->extraScale);
            if (realDst > 1) {
                extrascale1 = _mm_load_ps1(post->extraScale + 1);
            }
            if (realDst > 2) {
                extrascale2 = _mm_load_ps1(post->extraScale + 2);
            }
        }
    }
    for (int dz = 0; dz < dst_depth_quad; ++dz) {
        const auto weight_dz = weight + dz * weight_step_Z;
        const auto weightBias_dz = post->weightQuanBias + dz * GEMM_INT8_UNIT;
        const float* scale_dz = nullptr;
        scale_dz  = post->scale + dz * GEMM_INT8_UNIT;
        auto dst_z           = dst + dz * dst_step_tmp;
        const auto src_x   = src;
        auto dst_x         = dst_z;
        __m128i d0 = _mm_set1_epi32(0);
        __m128i d1 = _mm_set1_epi32(0);
        __m128i d2 = _mm_set1_epi32(0);
        __m128i d3 = _mm_set1_epi32(0);

        __m128i e0 = _mm_set1_epi32(0);
        __m128i e1 = _mm_set1_epi32(0);
        __m128i e2 = _mm_set1_epi32(0);
        __m128i e3 = _mm_set1_epi32(0);

        __m128i D0 = _mm_set1_epi32(0);
        __m128i D1 = _mm_set1_epi32(0);
        __m128i D2 = _mm_set1_epi32(0);
        __m128i D3 = _mm_set1_epi32(0);

        __m128i E0 = _mm_set1_epi32(0);
        __m128i E1 = _mm_set1_epi32(0);
        __m128i E2 = _mm_set1_epi32(0);
        __m128i E3 = _mm_set1_epi32(0);

        for (int sz = 0; sz < src_depth_quad; ++sz) {
            const auto weight_sz = weight_dz + weight_step_Y * sz;
            const auto src_z     = src_x + sz * GEMM_INT8_DST_XUNIT * GEMM_INT8_SRC_UNIT;

            int8_t tmp_w[64];
            _load_int4_to_int8((uint8_t*)weight_sz, tmp_w);

            auto w0 = _mm_loadu_si128((__m128i*)(tmp_w + GEMM_INT8_SRC_UNIT * 0));
            auto w1 = _mm_loadu_si128((__m128i*)(tmp_w + GEMM_INT8_SRC_UNIT * 1));
            auto w2 = _mm_loadu_si128((__m128i*)(tmp_w + GEMM_INT8_SRC_UNIT * 2));
            auto w3 = _mm_loadu_si128((__m128i*)(tmp_w + GEMM_INT8_SRC_UNIT * 3));

            auto s0 = _mm_loadu_si128((__m128i*)(src_z + GEMM_INT8_SRC_UNIT * 0));
            auto s1 = _mm_loadu_si128((__m128i*)(src_z + GEMM_INT8_SRC_UNIT * 1));
            auto s2 = _mm_loadu_si128((__m128i*)(src_z + GEMM_INT8_SRC_UNIT * 2));
            auto s3 = _mm_loadu_si128((__m128i*)(src_z + GEMM_INT8_SRC_UNIT * 3));


//#define COMPUTE(i, j)\
//auto d##i##j = _mm_maddubs_epi16(s##i, w##j);\
//d##i##j = _mm_madd_epi16(d##i##j, oneValue);\

#define COMPUTE(i, j)\
auto W##i##j##0 = _mm_srai_epi16(_mm_unpacklo_epi8(zero, w##j), 8);\
auto W##i##j##1 = _mm_srai_epi16(_mm_unpackhi_epi8(zero, w##j), 8);\
auto S##i##j##0 = _mm_unpacklo_epi8(s##i, zero);\
auto S##i##j##1 = _mm_unpackhi_epi8(s##i, zero);\
auto d##i##j = _mm_add_epi32(_mm_madd_epi16(S##i##j##0, W##i##j##0), _mm_madd_epi16(S##i##j##1, W##i##j##1));\

            COMPUTE(0, 0);
            COMPUTE(0, 1);
            COMPUTE(0, 2);
            COMPUTE(0, 3);
            COMPUTE(1, 0);
            COMPUTE(1, 1);
            COMPUTE(1, 2);
            COMPUTE(1, 3);
            COMPUTE(2, 0);
            COMPUTE(2, 1);
            COMPUTE(2, 2);
            COMPUTE(2, 3);
            COMPUTE(3, 0);
            COMPUTE(3, 1);
            COMPUTE(3, 2);
            COMPUTE(3, 3);

            d0 = _mm_add_epi32(d0, d00);
            d1 = _mm_add_epi32(d1, d01);
            d2 = _mm_add_epi32(d2, d02);
            d3 = _mm_add_epi32(d3, d03);

            e0 = _mm_add_epi32(e0, d10);
            e1 = _mm_add_epi32(e1, d11);
            e2 = _mm_add_epi32(e2, d12);
            e3 = _mm_add_epi32(e3, d13);

            D0 = _mm_add_epi32(D0, d20);
            D1 = _mm_add_epi32(D1, d21);
            D2 = _mm_add_epi32(D2, d22);
            D3 = _mm_add_epi32(D3, d23);

            E0 = _mm_add_epi32(E0, d30);
            E1 = _mm_add_epi32(E1, d31);
            E2 = _mm_add_epi32(E2, d32);
            E3 = _mm_add_epi32(E3, d33);
        }
        d0 = _mm_hadd_epi32(d0, d1);
        d1 = _mm_hadd_epi32(d2, d3);
        d0 = _mm_hadd_epi32(d0, d1);

        e0 = _mm_hadd_epi32(e0, e1);
        e1 = _mm_hadd_epi32(e2, e3);
        d1 = _mm_hadd_epi32(e0, e1);

        D0 = _mm_hadd_epi32(D0, D1);
        D1 = _mm_hadd_epi32(D2, D3);
        d2 = _mm_hadd_epi32(D0, D1);

        E0 = _mm_hadd_epi32(E0, E1);
        E1 = _mm_hadd_epi32(E2, E3);
        d3 = _mm_hadd_epi32(E0, E1);
        auto scaleValue = _mm_loadu_ps(scale_dz);
        // auto biasValue = _mm_loadu_si128((__m128i*)(bias_dz));
        // d0 = _mm_add_epi32(d0, biasValue);
        // d1 = _mm_add_epi32(d1, biasValue);
        // d2 = _mm_add_epi32(d2, biasValue);
        // d3 = _mm_add_epi32(d3, biasValue);
        //auto biasValue = _mm_loadu_ps((float*)(bias_dz));
        auto weightBiasValue = _mm_loadu_ps((float*)weightBias_dz);
        __m128 f0 = _mm_cvtepi32_ps(d0);
        __m128 f1 = _mm_cvtepi32_ps(d1);
        __m128 f2 = _mm_cvtepi32_ps(d2);
        __m128 f3 = _mm_cvtepi32_ps(d3);
        // x_kernelSum x w_quantZero
        auto xy0_0 = _mm_mul_ps(kernelSum0, weightBiasValue); // x dimemsion first
        auto xy0_1 = _mm_mul_ps(kernelSum1, weightBiasValue); // ..second
        auto xy0_2 = _mm_mul_ps(kernelSum2, weightBiasValue); // .. third
        auto xy0_3 = _mm_mul_ps(kernelSum3, weightBiasValue); // ..fourth
        f0 = _mm_mul_ps(f0, scaleValue);
        f1 = _mm_mul_ps(f1, scaleValue);
        f2 = _mm_mul_ps(f2, scaleValue);
        f3 = _mm_mul_ps(f3, scaleValue);
        if (post->extraScale) {
            f0 = _mm_mul_ps(f0, extrascale0);
            f1 = _mm_mul_ps(f1, extrascale1);
            f2 = _mm_mul_ps(f2, extrascale2);
            f3 = _mm_mul_ps(f3, extrascale3);
            if (post->extraBias && nullptr != biasPtr) {
                auto extraB = post->extraBias + dz * GEMM_INT8_UNIT;
                auto extrabias = _mm_loadu_ps(extraB);
                extrabias = _mm_mul_ps(f128, extrabias);
                auto extrabias0 = _mm_mul_ps(extrabias, extrascale0);
                auto extrabias1 = _mm_mul_ps(extrabias, extrascale1);
                auto extrabias2 = _mm_mul_ps(extrabias, extrascale2);
                auto extrabias3 = _mm_mul_ps(extrabias, extrascale3);
                f0 = _mm_sub_ps(f0, extrabias0);
                f1 = _mm_sub_ps(f1, extrabias1);
                f2 = _mm_sub_ps(f2, extrabias2);
                f3 = _mm_sub_ps(f3, extrabias3);
            }
        }
        f0 = _mm_add_ps(f0, xy0_0);
        f1 = _mm_add_ps(f1, xy0_1);
        f2 = _mm_add_ps(f2, xy0_2);
        f3 = _mm_add_ps(f3, xy0_3);

        if (nullptr != biasPtr) {
            const auto bias_dz   = biasPtr + dz * GEMM_INT8_UNIT;
            auto biasValue = _mm_loadu_ps(bias_dz);
            f0 = _mm_add_ps(f0, biasValue);
            f1 = _mm_add_ps(f1, biasValue);
            f2 = _mm_add_ps(f2, biasValue);
            f3 = _mm_add_ps(f3, biasValue);
        }
        __m128 f[4] = {f0, f1, f2, f3};
        if (nullptr == biasPtr) {
            for (int j = 0; j < realDst; ++j) {
                auto dstv = _mm_loadu_ps(((float*)dst_x) + j * 4);
                f[j] = _mm_add_ps(dstv, f[j]);
            }
        }
        if (post->fp32minmax) {
            f[0] = _mm_min_ps(f[0], fp32max);
            f[1] = _mm_min_ps(f[1], fp32max);
            f[2] = _mm_min_ps(f[2], fp32max);
            f[3] = _mm_min_ps(f[3], fp32max);
            f[0] = _mm_max_ps(f[0], fp32min);
            f[1] = _mm_max_ps(f[1], fp32min);
            f[2] = _mm_max_ps(f[2], fp32min);
            f[3] = _mm_max_ps(f[3], fp32min);
        }
        for (int j = 0; j < realDst; ++j) {
            _mm_storeu_ps(((float*)dst_x) + j * 4, f[j]);
        }
    }
}

void _SSE_MNNInt8ToInt16(int16_t* dest, const int8_t* sourceO, size_t count) {
    int countC16 = count / 16;
    int countR = count % 16;
    auto zero = _mm_set1_epi8(0);
    auto source = (const uint8_t*)sourceO;
    for (int i = 0; i < countC16; ++i) {
        auto s = _mm_castps_si128(_mm_loadu_ps((float*)source));
        auto d0 = _mm_unpacklo_epi8(s, zero);
        auto d1 = _mm_unpackhi_epi8(s, zero);
        _mm_storeu_ps((float*)dest, _mm_castsi128_ps(d0));
        _mm_storeu_ps((float*)dest + 4, _mm_castsi128_ps(d1));

        dest += 16;
        source += 16;
    }
    for (int i = 0; i < countR; ++i) {
        dest[i] = source[i];
    }
}

void _SSE_MNNReluInt8(int8_t* dst, const int8_t* src, size_t size, ssize_t zeroPoint) {
    auto zero = _mm_set1_epi8(zeroPoint - 128);// uint8 128
    for (int i = 0; i < size; i+=16) {
        auto x = _mm_castps_si128(_mm_loadu_ps((const float*)(src + i)));
        _mm_storeu_ps((float*)(dst + i), _mm_castsi128_ps(_mm_max_epu8(x, zero)));
    }
}

// require SSE 4.1
void _SSE_MNNFloat2Int8(const float* src, int8_t* dst, size_t sizeQuad, const float* scalep, ssize_t minV, ssize_t maxV, ssize_t zeroPoint) {
    __m128i zero = _mm_set1_epi32(0);
    __m128 minValue = _mm_set1_ps(minV);
    __m128 maxValue = _mm_set1_ps(maxV);
    __m128 zeroPointValue = _mm_set1_ps(zeroPoint);
    __m128 plus = _mm_set1_ps(0.5f);
    __m128 minus = _mm_set1_ps(-0.5f);
    __m128 scaleValue = _mm_loadu_ps(scalep);
    auto offset = _mm_set1_epi32(128);

    for (int i = 0; i < sizeQuad; ++i) {
        __m128 f0 = _mm_loadu_ps(src + 4 * i);
        f0 = _mm_mul_ps(f0, scaleValue);
        f0 = _mm_add_ps(f0, zeroPointValue);
        f0 = _mm_min_ps(f0, maxValue);
        f0 = _mm_max_ps(f0, minValue);
        auto m0 = _mm_cmplt_ps(f0, _mm_castsi128_ps(zero));
        m0 = _mm_blendv_ps(plus, minus, m0);
        f0 = _mm_add_ps(f0, m0);
        // 3: _MM_FROUND_TO_ZERO
        auto d0 = _mm_cvtps_epi32(_mm_round_ps(f0, 3));
        d0 = _mm_add_epi32(d0, offset);
        d0 = _mm_packs_epi32(d0, d0);
        d0 = _mm_packus_epi16(d0, d0);
        *((int*)dst + i) = _mm_cvtsi128_si32(d0);
    }
}

void _SSE_MNNInt8ScaleToFloat(float* dst, const int8_t* src, const float* scale, size_t sizeQuad, ssize_t zeroPoint) {
    auto sizeC4 = sizeQuad / 4;
    auto sizeRemain = sizeQuad % 4;
    __m128i zero = _mm_set1_epi32(0);
    __m128 scaleValue = _mm_loadu_ps(scale);
    __m128i zeroPointValue = _mm_set1_epi32(zeroPoint + 128);
    for (int i = 0; i < sizeC4; ++i) {
        auto s = _mm_castps_si128(_mm_loadu_ps((const float*)(src)));
        auto s0_16 = _mm_unpacklo_epi8(s, zero);
        auto s1_16 = _mm_unpackhi_epi8(s, zero);
        auto s0_32 = _mm_unpacklo_epi16(s0_16, zero);
        auto s1_32 = _mm_unpackhi_epi16(s0_16, zero);
        auto s2_32 = _mm_unpacklo_epi16(s1_16, zero);
        auto s3_32 = _mm_unpackhi_epi16(s1_16, zero);
        s0_32 = _mm_sub_epi32(s0_32, zeroPointValue);
        s1_32 = _mm_sub_epi32(s1_32, zeroPointValue);
        s2_32 = _mm_sub_epi32(s2_32, zeroPointValue);
        s3_32 = _mm_sub_epi32(s3_32, zeroPointValue);
        auto s0_f = _mm_cvtepi32_ps(s0_32);
        auto s1_f = _mm_cvtepi32_ps(s1_32);
        auto s2_f = _mm_cvtepi32_ps(s2_32);
        auto s3_f = _mm_cvtepi32_ps(s3_32);
        _mm_storeu_ps(dst + 4 * 0, _mm_mul_ps(s0_f, scaleValue));
        _mm_storeu_ps(dst + 4 * 1, _mm_mul_ps(s1_f, scaleValue));
        _mm_storeu_ps(dst + 4 * 2, _mm_mul_ps(s2_f, scaleValue));
        _mm_storeu_ps(dst + 4 * 3, _mm_mul_ps(s3_f, scaleValue));
        src += 16;
        dst += 16;
    }
    if (sizeRemain > 0) {
        int8_t srcTemp[128];
        ::memcpy(srcTemp, src, sizeRemain * 4);
        auto s = *(__m128i*)srcTemp;
        auto s0_16 = _mm_unpacklo_epi8(s, zero);
        auto s1_16 = _mm_unpackhi_epi8(s, zero);
        auto s0_32 = _mm_unpacklo_epi16(s0_16, zero);
        auto s1_32 = _mm_unpackhi_epi16(s0_16, zero);
        auto s2_32 = _mm_unpacklo_epi16(s1_16, zero);
        auto s3_32 = _mm_unpackhi_epi16(s1_16, zero);
        s0_32 = _mm_sub_epi32(s0_32, zeroPointValue);
        s1_32 = _mm_sub_epi32(s1_32, zeroPointValue);
        s2_32 = _mm_sub_epi32(s2_32, zeroPointValue);
        s3_32 = _mm_sub_epi32(s3_32, zeroPointValue);
        auto s0_f = _mm_cvtepi32_ps(s0_32);
        auto s1_f = _mm_cvtepi32_ps(s1_32);
        auto s2_f = _mm_cvtepi32_ps(s2_32);
        auto s3_f = _mm_cvtepi32_ps(s3_32);
        switch (sizeRemain) {
            case 3:
                _mm_storeu_ps(dst + 4 * 0, _mm_mul_ps(s0_f, scaleValue));
                _mm_storeu_ps(dst + 4 * 1, _mm_mul_ps(s1_f, scaleValue));
                _mm_storeu_ps(dst + 4 * 2, _mm_mul_ps(s2_f, scaleValue));
                break;
            case 2:
                _mm_storeu_ps(dst + 4 * 0, _mm_mul_ps(s0_f, scaleValue));
                _mm_storeu_ps(dst + 4 * 1, _mm_mul_ps(s1_f, scaleValue));
                break;
            case 1:
                _mm_storeu_ps(dst + 4 * 0, _mm_mul_ps(s0_f, scaleValue));
                break;
            default:
                break;
        }
    }
}

// require SSE 4.1
void _SSE_MNNLineDepthWiseInt8AddBiasScaleUnit(int8_t* dstO, const int8_t* srcO, const int8_t* weightO, const QuanPostTreatParameters* parameters, size_t width, size_t src_w_step, size_t fw, size_t fh, size_t dilateX_step, size_t dilateY_step, int8_t* idxOrder) {
    int pack = 16;
    auto dst = dstO;
    auto src = (const int16_t*)srcO;
    auto weight = (const int16_t*)weightO;
    auto biasValue0 = _mm_castps_si128(_mm_loadu_ps((const float*)parameters->bias));
    auto biasValue1 = _mm_castps_si128(_mm_loadu_ps((const float*)parameters->bias + 4));
    auto biasValue2 = _mm_castps_si128(_mm_loadu_ps((const float*)parameters->bias + 8));
    auto biasValue3 = _mm_castps_si128(_mm_loadu_ps((const float*)parameters->bias + 12));

    auto scaleValue0 = _mm_loadu_ps((const float*)parameters->scale);
    auto scaleValue1 = _mm_loadu_ps((const float*)parameters->scale + 4);
    auto scaleValue2 = _mm_loadu_ps((const float*)parameters->scale + 8);
    auto scaleValue3 = _mm_loadu_ps((const float*)parameters->scale + 12);
    __m128i d0, d1, d2, d3;
    int dx, fx, fy;
    __m128i srcValue0;
    auto srcTemp0 = (int64_t*)(&srcValue0);
    __m128i srcValue1;
    auto srcTemp1 = (int64_t*)(&srcValue1);
    __m128i zero = _mm_xor_si128(srcValue1, srcValue1);
    __m128 zero128 = _mm_set1_ps(0.0f);
    auto minValue = _mm_set1_epi16(parameters->minValue + 128);
    auto maxValue = _mm_set1_epi16(parameters->maxValue + 128);
    __m128 plus = _mm_set1_ps(0.5f);
    __m128 minus = _mm_set1_ps(-0.5f);
    auto offset = _mm_set1_epi32(128);
    // Stride = 1
    for (dx = 0; dx < width; ++dx) {
        d0 = biasValue0;
        d1 = biasValue1;
        d2 = biasValue2;
        d3 = biasValue3;

        auto dst_x          = dst;
        const auto src_z    = src;
        for (fy = 0; fy < fh; ++fy) {
            const auto src_y    = src_z + fy * dilateY_step;
            const auto weight_y = weight + fy * fw * pack;
            for (fx = 0; fx < fw; ++fx) {
                const auto src_x    = src_y + fx * dilateX_step;
                auto s0_16 = _mm_castps_si128(_mm_loadu_ps((float*)src_x));
                auto s1_16 = _mm_castps_si128(_mm_loadu_ps((float*)src_x + 4));
                auto s0_32 = _mm_unpacklo_epi16(s0_16, zero);
                auto s1_32 = _mm_unpackhi_epi16(s0_16, zero);
                auto s2_32 = _mm_unpacklo_epi16(s1_16, zero);
                auto s3_32 = _mm_unpackhi_epi16(s1_16, zero);
                
                const auto weight_x = weight_y + pack * fx;
                auto w0_16 = _mm_castps_si128(_mm_loadu_ps((float*)weight_x));
                auto w1_16 = _mm_castps_si128(_mm_loadu_ps((float*)weight_x + 4));
                auto w0_32 = _mm_unpacklo_epi16(w0_16, zero);
                auto w1_32 = _mm_unpackhi_epi16(w0_16, zero);
                auto w2_32 = _mm_unpacklo_epi16(w1_16, zero);
                auto w3_32 = _mm_unpackhi_epi16(w1_16, zero);
                
                d0 = _mm_add_epi32(d0, _mm_madd_epi16(w0_32, s0_32));
                d1 = _mm_add_epi32(d1, _mm_madd_epi16(w1_32, s1_32));
                d2 = _mm_add_epi32(d2, _mm_madd_epi16(w2_32, s2_32));
                d3 = _mm_add_epi32(d3, _mm_madd_epi16(w3_32, s3_32));
            }
        }
        __m128 f0 = _mm_cvtepi32_ps(d0);
        __m128 f1 = _mm_cvtepi32_ps(d1);
        __m128 f2 = _mm_cvtepi32_ps(d2);
        __m128 f3 = _mm_cvtepi32_ps(d3);
        f0 = _mm_mul_ps(f0, scaleValue0);
        f1 = _mm_mul_ps(f1, scaleValue1);
        f2 = _mm_mul_ps(f2, scaleValue2);
        f3 = _mm_mul_ps(f3, scaleValue3);
        auto m0 = _mm_cmplt_ps(f0, zero128);
        m0 = _mm_blendv_ps(plus, minus, m0);
        f0 = _mm_add_ps(f0, m0);
        auto m1 = _mm_cmplt_ps(f1, zero128);
        m1 = _mm_blendv_ps(plus, minus, m1);
        f1 = _mm_add_ps(f1, m1);
        auto m2 = _mm_cmplt_ps(f2, zero128);
        m2 = _mm_blendv_ps(plus, minus, m2);
        f2 = _mm_add_ps(f2, m2);
        auto m3 = _mm_cmplt_ps(f3, zero128);
        m3 = _mm_blendv_ps(plus, minus, m3);
        f3 = _mm_add_ps(f3, m3);
        // 3: _MM_FROUND_TO_ZERO
        d0 = _mm_cvtps_epi32(_mm_round_ps(f0, 3));
        d0 = _mm_add_epi32(d0, offset);
        d1 = _mm_cvtps_epi32(_mm_round_ps(f1, 3));
        d1 = _mm_add_epi32(d1, offset);
        d2 = _mm_cvtps_epi32(_mm_round_ps(f2, 3));
        d2 = _mm_add_epi32(d2, offset);
        d3 = _mm_cvtps_epi32(_mm_round_ps(f3, 3));
        d3 = _mm_add_epi32(d3, offset);

        // Int32 -> Int8
        d0 = _mm_packs_epi32(d0, d1);
        d2 = _mm_packs_epi32(d2, d3);
        d0 = _mm_min_epi16(d0, maxValue);
        d0 = _mm_max_epi16(d0, minValue);
        d2 = _mm_min_epi16(d2, maxValue);
        d2 = _mm_max_epi16(d2, minValue);
        d0 = _mm_packus_epi16(d0, d2);

        _mm_storeu_ps((float*)(dst), _mm_castsi128_ps(d0));
        dst += 16;
        src += src_w_step;
    }
}
extern "C" {
void MNNInt8ToUInt8(void* ptr, int count) {
    auto src = (int8_t*)ptr;
    auto dst = (uint8_t*)ptr;
    int c16 = count / 16;
    count = count % 16;
    auto zero = _mm_set1_epi8(0);
    auto offset = _mm_set1_epi16(128);
    for (int v = 0; v < c16; ++v) {
        auto i8Value = _mm_loadu_si128((__m128i*)(src));
        auto i16Value0 = _mm_srai_epi16(_mm_unpacklo_epi8(zero, i8Value), 8);
        auto i16Value1 = _mm_srai_epi16(_mm_unpackhi_epi8(zero, i8Value), 8);
        i16Value0 = _mm_add_epi16(i16Value0, offset);
        i16Value1 = _mm_add_epi16(i16Value1, offset);
        i8Value = _mm_packus_epi16(i16Value0, i16Value1);
        _mm_storeu_si128((__m128i*)dst, i8Value);
        dst += 16;
        src += 16;
    }
    for (int v = 0; v < count; ++v) {
        dst[v] = (int)src[v] + 128;
    }
}
void MNNUInt8ToInt8(void* ptr, int count) {
    auto src = (uint8_t*)ptr;
    auto dst = (int8_t*)ptr;
    int c16 = count / 16;
    count = count % 16;
    auto zero = _mm_set1_epi8(0);
    auto offset = _mm_set1_epi16(128);
    for (int v = 0; v < c16; ++v) {
        auto i8Value = _mm_loadu_si128((__m128i*)(src));
        auto i16Value0 = _mm_unpacklo_epi8(zero, i8Value);
        auto i16Value1 = _mm_unpackhi_epi8(zero, i8Value);
        i16Value0 = _mm_sub_epi16(i16Value0, offset);
        i16Value1 = _mm_sub_epi16(i16Value1, offset);
        i8Value = _mm_packus_epi16(i16Value0, i16Value1);
        _mm_storeu_si128((__m128i*)dst, i8Value);
        dst += 16;
        src += 16;
    }
    for (int v = 0; v < count; ++v) {
        dst[v] = (int)src[v] - 128;
    }
}
}
