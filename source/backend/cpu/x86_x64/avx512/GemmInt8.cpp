//
//  GemmInt8.cpp
//  MNN
//
//  Created by MNN on 2020/09/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "FunctionSummary.hpp"
#include "core/Macro.h"
#include "GemmInt8Macro.h"


#ifdef MNN_AVX512_VNNI
extern void _AVX512_MNNGemmInt8AddBiasScale_16x4_Unit_VNNI(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad, const QuanPostTreatParameters* post, size_t realDst);
extern void _AVX512_MNNLineDepthWiseInt8AddBiasScaleUnit_VNNI(int8_t* dstO, const int8_t* srcO, const int8_t* weightO, const QuanPostTreatParameters* parameters, size_t width, size_t src_w_step, size_t fw, size_t fh, size_t dilateX_step, size_t dilateY_step, int8_t* idxOrder=nullptr);
extern void _AVX512_MNNGemmInt8AddBiasScale_16x4_w4_Unit_VNNI(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad, const QuanPostTreatParameters* post, size_t realDst);
#endif

// Define in GemmInt8_4_4_64.cpp
extern void _AVX512_NO_VNNI_4_4_64(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad, const QuanPostTreatParameters* post, size_t realDst);
extern void _AVX512_NO_VNNI_4_4_64_w4(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad, const QuanPostTreatParameters* post, size_t realDst);

// Define in GemmInt8_4_4_64_7bit.cpp
extern void _AVX512_NO_VNNI_4_4_64_7bit(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad, const QuanPostTreatParameters* post, size_t realDst);


static void _AVX512BasicMNNPackC4ForMatMul_A(int8_t* destOrigin, int8_t const** sourceGroup, const int32_t* info, const int32_t* el) {
    int number = info[0];
    int eReal = info[1];
    int xStride = info[3];
    int xS4 = xStride * 16 / sizeof(float);
    int eOutsideStride = info[2] / sizeof(int32_t);
    const int EP = GEMMINT8_AVX512_E;
    int eDest = EP;
    const int LP = 4;
    for (int n=0; n<number; ++n) {
        int e = el[4 * n + 0];
        int l = el[4 * n + 1];
        int eOffset = el[4 * n + 2];
        int lOffset = el[4 * n + 3];
        int eC = eOffset / eDest;
        int eR = eOffset % eDest;
        int eS = eDest - eR;
        auto source = (float*)sourceGroup[n];
        auto dest = (float*)(destOrigin + eC * info[2] + eR * LP + lOffset * EP);
        l = l / 4; // Use float instead of int8 * 4
        if (eR > 0) {
            int eStep = ALIMIN(e, eS);
            for (int y = 0; y < eStep; ++y) {
                for (int x = 0; x < l; ++x) {
                    auto xR                  = x % 4;
                    auto xC                  = x / 4;
                    dest[x * eDest + y] = source[xC * eReal * 4 + y * xS4 + xR];
                }
            }
            e-= eStep;
            dest += (eOutsideStride - eR);
            source += eStep * xS4;
        }
        if (e <=0 ) {
            continue;
        }
        const int pack   = GEMMINT8_AVX512_E;
        auto ePack       = e / pack;
        auto lC4         = l / 4;
        auto lDiv        = UP_DIV(l, 4);
        auto eRemain     = ePack * pack;
        auto lRemain     = lC4 * 4;
        auto lRes        = l - lRemain;
        for (int y = 0; y < ePack; ++y) {
            auto dstY = dest + y * eOutsideStride;
            auto srcY = source + y * pack * xS4;
            for (int x = 0; x < lC4; ++x) {
                auto srcX = srcY + x * 4 * eReal;
                auto dstX = dstY + x * pack * 4;
                auto s00  = _mm_loadu_ps(srcX + 0 * xS4);
                auto s01  = _mm_loadu_ps(srcX + 1 * xS4);
                auto s02  = _mm_loadu_ps(srcX + 2 * xS4);
                auto s03  = _mm_loadu_ps(srcX + 3 * xS4);

                _MM_TRANSPOSE4_PS(s00, s01, s02, s03);

    #define STORE_TEMP(i)                               \
        _mm_storeu_ps(dstX + 4 * i, s##0##i); \

                STORE_TEMP(0);
                STORE_TEMP(1);
                STORE_TEMP(2);
                STORE_TEMP(3);
            }
            if (lRes == 0) {
                continue;
            }
            auto srcX = srcY + lC4 * 4 * eReal;
            auto dstX = dstY + lC4 * eDest * 4;
            auto s00  = _mm_loadu_ps(srcX + 0 * xS4);
            auto s01  = _mm_loadu_ps(srcX + 1 * xS4);
            auto s02  = _mm_loadu_ps(srcX + 2 * xS4);
            auto s03  = _mm_loadu_ps(srcX + 3 * xS4);

            _MM_TRANSPOSE4_PS(s00, s01, s02, s03);
            if (lRes == 3) {
                STORE_TEMP(0);
                STORE_TEMP(1);
                STORE_TEMP(2);
            } else if (lRes == 2) {
                STORE_TEMP(0);
                STORE_TEMP(1);
            } else {
                STORE_TEMP(0);
            }
        }
        // Down
        {
            auto eLast    = e - eRemain;
            auto lastDest = dest + ePack * eOutsideStride;
            for (int y = eRemain; y < e; ++y) {
                auto yR = y - eRemain;
                for (int x = 0; x < l; ++x) {
                    auto xR                  = x % 4;
                    auto xC                  = x / 4;
                    lastDest[x * eDest + yR] = source[xC * eReal * 4 + y * 4 * xStride + xR];
                }
            }
        }
    }
}


void _AVX512_MNNLineDepthWiseInt8AddBiasScaleUnit(int8_t* dstO, const int8_t* srcO, const int8_t* weightO, const QuanPostTreatParameters* parameters, size_t width, size_t src_w_step, size_t fw, size_t fh, size_t dilateX_step, size_t dilateY_step, int8_t* idxOrder) {
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
void _AVX512_MNNFloat2Int8(const float* src, int8_t* dst, size_t sizeQuad, const float* scalep, ssize_t minV, ssize_t maxV, ssize_t zeroPoint) {
    auto zero = _mm256_set1_epi32(0);
    auto minValue = _mm256_set1_ps(minV);
    auto maxValue = _mm256_set1_ps(maxV);
    auto zeroPointValue = _mm256_set1_ps(zeroPoint);
    auto offset = _mm256_set1_epi32(128);
    auto plus = _mm256_set1_ps(0.5f);
    auto minus = _mm256_set1_ps(-0.5f);
    auto scaleValue0 = _mm256_loadu_ps(scalep);
    auto scaleValue1 = _mm256_loadu_ps(scalep + 8);

    for (int i = 0; i < sizeQuad; ++i) {
        auto f0 = _mm256_loadu_ps(src + PACK_UNIT * i);
        auto f1 = _mm256_loadu_ps(src + PACK_UNIT * i + 8);
        f0 = _mm256_mul_ps(f0, scaleValue0);
        f1 = _mm256_mul_ps(f1, scaleValue1);
        f0 = _mm256_add_ps(f0, zeroPointValue);
        f1 = _mm256_add_ps(f1, zeroPointValue);
        f0 = _mm256_min_ps(f0, maxValue);
        f1 = _mm256_min_ps(f1, maxValue);
        f0 = _mm256_max_ps(f0, minValue);
        f1 = _mm256_max_ps(f1, minValue);
        auto m0 = _mm256_cmp_ps(f0, _mm256_castsi256_ps(zero), 1);
        auto m1 = _mm256_cmp_ps(f1, _mm256_castsi256_ps(zero), 1);
        m0 = _mm256_blendv_ps(plus, minus, m0);
        m1 = _mm256_blendv_ps(plus, minus, m1);
        f0 = _mm256_add_ps(f0, m0);
        f1 = _mm256_add_ps(f1, m1);
        // 3: _MM_FROUND_TO_ZERO
        auto d0 = _mm256_cvtps_epi32(_mm256_round_ps(f0, 3));
        auto d1 = _mm256_cvtps_epi32(_mm256_round_ps(f1, 3));
        d0 = _mm256_add_epi32(d0, offset);
        d1 = _mm256_add_epi32(d1, offset);
        d0 = _mm256_packs_epi32(d0, _mm256_setzero_si256());
        d1 = _mm256_packs_epi32(d1, _mm256_setzero_si256());
        d0 = _mm256_permute4x64_epi64(d0, 0xD8);
        d1 = _mm256_permute4x64_epi64(d1, 0xD8);
#if defined(_MSC_VER)
        __m256i x = static_cast<__m256i>(_mm256_packus_epi16(d0, _mm256_setzero_si256()));
        __m256i y = static_cast<__m256i>(_mm256_packus_epi16(d1, _mm256_setzero_si256()));
        *((int64_t*)dst + 2 * i + 0) = x.m256i_i64[0];
        *((int64_t*)dst + 2 * i + 1) = y.m256i_i64[0];
#else
        __v4di x = static_cast<__v4di>(_mm256_packus_epi16(d0, _mm256_setzero_si256()));
        __v4di y = static_cast<__v4di>(_mm256_packus_epi16(d1, _mm256_setzero_si256()));
        *((int64_t*)dst + 2 * i + 0) = x[0];
        *((int64_t*)dst + 2 * i + 1) = y[0];
#endif
    }
}

void _AVX512_MNNInt8ScaleToFloat(float* dst, const int8_t* src, const float* scale, size_t sizeQuad, ssize_t zeroPoint) {
    auto sizeC4 = sizeQuad / 2;
    auto sizeRemain = sizeQuad % 2;
    auto zero = _mm256_set1_epi32(0);
    auto scaleValue0 = _mm256_loadu_ps(scale);
    auto scaleValue1 = _mm256_loadu_ps(scale + 8);
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
        _mm256_storeu_ps(dst + 8 * 0, _mm256_mul_ps(s0_f, scaleValue0));
        _mm256_storeu_ps(dst + 8 * 1, _mm256_mul_ps(s1_f, scaleValue1));
        _mm256_storeu_ps(dst + 8 * 2, _mm256_mul_ps(s2_f, scaleValue0));
        _mm256_storeu_ps(dst + 8 * 3, _mm256_mul_ps(s3_f, scaleValue1));
        src += 32;
        dst += 32;
    }
    if (sizeRemain > 0) {
        auto s = _mm256_castsi128_si256(_mm_castps_si128(_mm_loadu_ps((const float*)src)));
        auto s0_16 = _mm256_permute4x64_epi64(_mm256_unpacklo_epi8(s, zero), 0XD8);
        auto s1_16 = _mm256_permute4x64_epi64(_mm256_unpackhi_epi8(s, zero), 0xD8);
        auto s0_32 = _mm256_unpacklo_epi16(s0_16, zero);
        auto s1_32 = _mm256_unpacklo_epi16(s1_16, zero);
        auto s2_32 = _mm256_unpackhi_epi16(s0_16, zero);
        auto s3_32 = _mm256_unpackhi_epi16(s1_16, zero);
        s0_32 = _mm256_sub_epi32(s0_32, zeroPointValue);
        s1_32 = _mm256_sub_epi32(s1_32, zeroPointValue);
        auto s0_f = _mm256_cvtepi32_ps(s0_32);
        auto s1_f = _mm256_cvtepi32_ps(s1_32);
        _mm256_storeu_ps(dst + 8 * 0, _mm256_mul_ps(s0_f, scaleValue0));
        _mm256_storeu_ps(dst + 8 * 1, _mm256_mul_ps(s1_f, scaleValue1));
    }
}


static void _AVX512_MNNGetGemmUnit(int* UNIT, int* SRC_UNIT, int* DST_XUNIT) {
    *UNIT = GEMMINT8_AVX512_H_NOVNNI;
    *SRC_UNIT = GEMMINT8_AVX512_L;
    *DST_XUNIT = GEMMINT8_AVX512_E;
}

static void _AVX512_MNNGetGemmUnit_VNNI(int* UNIT, int* SRC_UNIT, int* DST_XUNIT) {
    *UNIT = GEMMINT8_AVX512_H_VNNI;
    *SRC_UNIT = GEMMINT8_AVX512_L;
    *DST_XUNIT = GEMMINT8_AVX512_E;
}

void _AVX512_MNNInt8FunctionInit(void* functions, bool supportVNNI) {
    auto gAVX2CoreInt8Functions = (MNN::CoreInt8Functions*)functions;
#ifdef MNN_AVX512_VNNI
    if (supportVNNI) {
        gAVX2CoreInt8Functions->Int8GemmKernel = _AVX512_MNNGemmInt8AddBiasScale_16x4_Unit_VNNI;
        gAVX2CoreInt8Functions->Int8GemmKernelFast = _AVX512_MNNGemmInt8AddBiasScale_16x4_Unit_VNNI;
        gAVX2CoreInt8Functions->Int8GemmKernel_W4 = _AVX512_MNNGemmInt8AddBiasScale_16x4_w4_Unit_VNNI;
        // conv depthwise
        gAVX2CoreInt8Functions->ConvDepthwiseLineInt8 = _AVX512_MNNLineDepthWiseInt8AddBiasScaleUnit_VNNI;
        // MatMul
        gAVX2CoreInt8Functions->MNNGetGemmUnit = _AVX512_MNNGetGemmUnit_VNNI;
        // Im2Col
        gAVX2CoreInt8Functions->MNNPackC4Int8ForMatMul_A = _AVX512BasicMNNPackC4ForMatMul_A;

        

    } else
#endif
    {
        gAVX2CoreInt8Functions->Int8GemmKernel = _AVX512_NO_VNNI_4_4_64;
        gAVX2CoreInt8Functions->Int8GemmKernelFast = _AVX512_NO_VNNI_4_4_64_7bit;
        gAVX2CoreInt8Functions->Int8GemmKernel_W4 = _AVX512_NO_VNNI_4_4_64_w4;
        // conv depthwise
        gAVX2CoreInt8Functions->ConvDepthwiseLineInt8 = _AVX512_MNNLineDepthWiseInt8AddBiasScaleUnit;
        // MatMul
        gAVX2CoreInt8Functions->MNNGetGemmUnit = _AVX512_MNNGetGemmUnit;
        // Im2Col
        gAVX2CoreInt8Functions->MNNPackC4Int8ForMatMul_A = _AVX512BasicMNNPackC4ForMatMul_A;
    }
    // Int8 <-> Float
    gAVX2CoreInt8Functions->MNNFloat2Int8 = _AVX512_MNNFloat2Int8;
    gAVX2CoreInt8Functions->MNNInt8ScaleToFloat = _AVX512_MNNInt8ScaleToFloat;
}
