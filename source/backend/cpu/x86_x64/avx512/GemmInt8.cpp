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
    int realDstCount = info[4];
    for (int n=0; n<number; ++n) {
        int e = el[4 * n + 0];
        int l = el[4 * n + 1];
        int eOffset = el[4 * n + 2];
        int lOffset = el[4 * n + 3];
        int eC = eOffset / EP;
        int eR = eOffset % EP;
        int eS = eDest - eR;
        // a bag: e_filled*LP, e_filled<=EP.
        bool willReachBag = false;// will fill last bag(EP*LP) but this bag has e_filled<EP, maybe not the first bag.
        int elast = EP;
        int eOutsideStride4LastBag = eOutsideStride;
        if (realDstCount % EP > 0) {
            int jobsE = realDstCount - eOffset - e;
            if (jobsE == 0 || (jobsE < (realDstCount % EP))) {
                willReachBag = true;
            }
        }
        auto source = (float*)sourceGroup[n];
        auto dest = (float*)(destOrigin + eC * info[2] + eR * LP + lOffset * EP);
        l = l / 4; // Use float instead of int8 * 4
        if (willReachBag && e + eR < EP) { // The first bag to fill has e_filled<EP.
            elast = ALIMAX(eR + e, realDstCount % EP); // maybe padding, so max().
            dest = (float*)(destOrigin + lOffset * elast + eC * info[2] + eR * LP);
        }
        int offsetLC_ = lOffset / 4;
        if (eR > 0) {
            int eStep = ALIMIN(e, eS);
            for (int y = 0; y < eStep; ++y) {
                for (int x = 0; x < l; ++x) {
                    auto xR                  = x % 4;
                    auto xC                  = x / 4;
                    dest[x * elast + y] = source[xC * eReal * 4 + y * xS4 + xR];
                }
            }
            e-= eStep;
            if (!willReachBag || e >= EP) {
                dest += (eOutsideStride - eR);
            } else { // The bag to fill: e_filled < EP
                int e_tofill = ALIMAX(e, realDstCount % EP); // maybe padding>0
                eOutsideStride4LastBag = eOutsideStride - (EP * offsetLC_);
                dest += (eOutsideStride4LastBag - eR + offsetLC_ * e_tofill);
            }
            source += eStep * xS4;
        }
        if (e <=0 ) {
            continue;
        }

        auto ePack       = e / EP;
        auto lC4         = l / LP;
        auto lDiv        = UP_DIV(l, 4);
        auto eRemain     = ePack * EP;
        auto lRemain     = lC4 * 4;
        auto lRes        = l - lRemain;
        for (int y = 0; y < ePack; ++y) {
            auto dstY = dest + y * eOutsideStride;
            auto srcY = source + y * EP * xS4;
            for (int x = 0; x < lC4; ++x) {
                auto srcX = srcY + x * 4 * eReal;
                auto dstX = dstY + x * EP * 4;
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
            auto srcX = srcY + lC4 * LP * eReal;
            auto dstX = dstY + lC4 * EP * LP;
            
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
            auto eLast    = e - eRemain; // e - ePack * EP
            auto lastDest = dest + ePack * eOutsideStride;
            int eFill = EP;
            if (eLast > 0 && willReachBag) {
                eFill = ALIMAX((realDstCount % EP), eLast);
                if (ePack > 0) {
                    lastDest = dest + ePack * eOutsideStride - offsetLC_ * (EP - eFill);
                }
            }
            for (int y = eRemain; y < e; ++y) {
                auto yR = y - eRemain;
                for (int x = 0; x < l; ++x) {
                    auto xR                  = x % 4;
                    auto xC                  = x / 4;
                    lastDest[x * eFill + yR] = source[xC * eReal * 4 + y * 4 * xStride + xR];
                }
            }
        }
    }
}

static void _AVX512_MNNTMacCompute(float* dst, const int8_t* table, const float* inputSum, const TMacResource* res, const PlaneInfo* plane) {
    // res->mHp=64
    auto ocC4 = plane->ocDiv;
    auto blC4 = res->mBlockSizeC4;
    const auto mask = _mm256_set1_epi8(0xf);
    __m512 sum0, sum1, sum2, sum3;
    int halfhp = res->mHp / 2;
    const auto offset = _mm512_set1_ps(plane->offset);
    const auto offset128 = _mm512_set1_ps(128.f * blC4);
    __m256i zero = _mm256_set1_epi32(0);
    const auto dequantscale = _mm512_set1_ps(plane->dequantscale);
    const auto f = _mm512_set1_ps(2.0f);
    const auto minv = _mm512_set1_ps(plane->minValue);
    const auto maxv = _mm512_set1_ps(plane->maxValue);
    float tmp0[16], tmp1[16], tmp2[16], tmp3[16];
    // Compute Dst: Loop up and sum
    for (int oz=0; oz<ocC4; ++oz) {
        auto dstZ = dst + oz * (res->mHp / 4) * plane->planeSize * 4;
        auto biasPtrZ = (float*)(plane->mBiasPtr) + res->mHp * oz;
        auto blockScaleZ = (const float*)plane->mWeightScalePtr + oz * res->mBlockNumber * 2 * res->mHp;
        auto blockBiasZ = blockScaleZ + res->mHp;
        auto csum0 = _mm512_loadu_ps(biasPtrZ);
        auto csum1 = _mm512_loadu_ps(biasPtrZ + 16);
        auto csum2 = _mm512_loadu_ps(biasPtrZ + 32);
        auto csum3 = _mm512_loadu_ps(biasPtrZ + 48);
        for (int ib=0; ib<res->mBlockNumber; ++ib) {
            auto tableIB = table + (ib) * (blC4) * 16;
            auto blockScale = blockScaleZ + ib * 2 * res->mHp;
            auto blockBias  = blockBiasZ + ib * 2 * res->mHp;
            auto scale0 = _mm512_loadu_ps(blockScale);
            auto scale1 = _mm512_loadu_ps(blockScale + 16);
            auto scale2 = _mm512_loadu_ps(blockScale + 32);
            auto scale3 = _mm512_loadu_ps(blockScale + 48);
            auto bias0 = _mm512_loadu_ps(blockBias);
            auto bias1 = _mm512_loadu_ps(blockBias + 16);
            auto bias2 = _mm512_loadu_ps(blockBias + 32);
            auto bias3 = _mm512_loadu_ps(blockBias + 48);
            sum0 = _mm512_setzero_ps();
            sum1 = _mm512_setzero_ps();
            sum2 = _mm512_setzero_ps();
            sum3 = _mm512_setzero_ps();
            auto inputSummer = _mm512_set1_ps(inputSum[ib]);
            for (int b=0; b<res->mBits; ++b) {
                auto bsum0_int16 = _mm256_setzero_si256();
                auto bsum1_int16 = _mm256_setzero_si256();
                auto bsum2_int16 = _mm256_setzero_si256();
                auto bsum3_int16 = _mm256_setzero_si256();
                auto weight = plane->mWeightPtr + res->mWeightInt8->stride(0) * oz + res->mWeightInt8->stride(1) * ib + b * blC4 * res->mHp / 2;
                for (int iz=0; iz<blC4; ++iz) {
                    auto tablePtr = (uint8_t*)tableIB + iz * 16;
                    auto srclut = _mm_loadu_si128(reinterpret_cast<__m128i*>(tablePtr));
                    auto weightZ = weight + iz * halfhp;
                    auto w0_int4 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(weightZ));
                    auto w1 = _mm256_and_si256(mask, _mm256_srli_epi16(w0_int4, 4));
                    auto w0 = _mm256_and_si256(mask, w0_int4);

                    __m256i lut = _mm256_set_m128i(srclut, srclut);
                    __m256i v0 = _mm256_shuffle_epi8(lut, w0);
                    __m256i v1 = _mm256_shuffle_epi8(lut, w1);
                    auto v00 = _mm256_unpacklo_epi8(v0, zero); // 0,1,2,3,4,5,6,7,16,17,18,19,20,21,22,23
                    auto v01 = _mm256_unpackhi_epi8(v0, zero); // 8,9,10,11,12,13,14,15,24,25,26,27,28,29,30,31
                    auto v10 = _mm256_unpacklo_epi8(v1, zero); // 0,1,2,3,4,5,6,7,16,17,18,19,20,21,22,23
                    auto v11 = _mm256_unpackhi_epi8(v1, zero); // 8,9,10,11,12,13,14,15,24,25,26,27,28,29,30,31
                    bsum0_int16 = _mm256_add_epi16(bsum0_int16, v00);
                    bsum1_int16 = _mm256_add_epi16(bsum1_int16, v01);
                    bsum2_int16 = _mm256_add_epi16(bsum2_int16, v10);
                    bsum3_int16 = _mm256_add_epi16(bsum3_int16, v11);
                }
                sum0 = _mm512_mul_ps(sum0, f);
                sum1 = _mm512_mul_ps(sum1, f);
                sum2 = _mm512_mul_ps(sum2, f);
                sum3 = _mm512_mul_ps(sum3, f);
                auto v0 = _mm256_unpacklo_epi16(bsum0_int16, zero); // 0,1,2,3,16,17,18,19
                auto v1 = _mm256_unpackhi_epi16(bsum0_int16, zero); // 4,5,6,7,20,21,22,23
                auto v2 = _mm256_unpacklo_epi16(bsum1_int16, zero); // 8,9,10,11,24,25,26,27
                auto v3 = _mm256_unpackhi_epi16(bsum1_int16, zero); // 12,13,14,15,28,29,30,31
                auto v4 = _mm256_unpacklo_epi16(bsum2_int16, zero); 
                auto v5 = _mm256_unpackhi_epi16(bsum2_int16, zero);
                auto v6 = _mm256_unpacklo_epi16(bsum3_int16, zero);
                auto v7 = _mm256_unpackhi_epi16(bsum3_int16, zero);

                auto v00 = _mm256_cvtepi32_ps(_mm256_set_m128i(_mm256_extracti128_si256(v1, 0), _mm256_extracti128_si256(v0, 0)));
                auto v01 = _mm256_cvtepi32_ps(_mm256_set_m128i(_mm256_extracti128_si256(v3, 0), _mm256_extracti128_si256(v2, 0)));
                auto v10 = _mm256_cvtepi32_ps(_mm256_set_m128i(_mm256_extracti128_si256(v5, 0), _mm256_extracti128_si256(v4, 0)));
                auto v11 = _mm256_cvtepi32_ps(_mm256_set_m128i(_mm256_extracti128_si256(v7, 0), _mm256_extracti128_si256(v6, 0)));
                auto v02 = _mm256_cvtepi32_ps(_mm256_set_m128i(_mm256_extracti128_si256(v1, 1), _mm256_extracti128_si256(v0, 1)));
                auto v03 = _mm256_cvtepi32_ps(_mm256_set_m128i(_mm256_extracti128_si256(v3, 1), _mm256_extracti128_si256(v2, 1)));
                auto v12 = _mm256_cvtepi32_ps(_mm256_set_m128i(_mm256_extracti128_si256(v5, 1), _mm256_extracti128_si256(v4, 1)));
                auto v13 = _mm256_cvtepi32_ps(_mm256_set_m128i(_mm256_extracti128_si256(v7, 1), _mm256_extracti128_si256(v6, 1)));

                _mm256_storeu_ps(tmp0, v00);
                _mm256_storeu_ps(tmp0 + 8, v01);
                _mm256_storeu_ps(tmp1, v02);
                _mm256_storeu_ps(tmp1 + 8, v03);
                _mm256_storeu_ps(tmp2, v10);
                _mm256_storeu_ps(tmp2 + 8, v11);
                _mm256_storeu_ps(tmp3, v12);
                _mm256_storeu_ps(tmp3 + 8, v13);
                
                auto v0_ = _mm512_loadu_ps(tmp0);
                auto v1_ = _mm512_loadu_ps(tmp1);
                auto v2_ = _mm512_loadu_ps(tmp2);
                auto v3_ = _mm512_loadu_ps(tmp3);
                v0_ = _mm512_sub_ps(v0_, offset128);
                v1_ = _mm512_sub_ps(v1_, offset128);
                v2_ = _mm512_sub_ps(v2_, offset128);
                v3_ = _mm512_sub_ps(v3_, offset128);

                sum0 = _mm512_add_ps(sum0, v0_);
                sum1 = _mm512_add_ps(sum1, v1_);
                sum2 = _mm512_add_ps(sum2, v2_);
                sum3 = _mm512_add_ps(sum3, v3_);
                
            }
            auto f0 = _mm512_mul_ps(inputSummer, bias0);
            auto f1 = _mm512_mul_ps(inputSummer, bias1);
            auto f2 = _mm512_mul_ps(inputSummer, bias2);
            auto f3 = _mm512_mul_ps(inputSummer, bias3);

            auto m0 = _mm512_mul_ps(dequantscale, scale0);
            auto m1 = _mm512_mul_ps(dequantscale, scale1);
            auto m2 = _mm512_mul_ps(dequantscale, scale2);
            auto m3 = _mm512_mul_ps(dequantscale, scale3);
            
            m0 = _mm512_mul_ps(m0, _mm512_sub_ps(sum0, offset));
            m1 = _mm512_mul_ps(m1, _mm512_sub_ps(sum1, offset));
            m2 = _mm512_mul_ps(m2, _mm512_sub_ps(sum2, offset));
            m3 = _mm512_mul_ps(m3, _mm512_sub_ps(sum3, offset));
            csum0 = _mm512_add_ps(csum0, _mm512_add_ps(m0, f0));
            csum1 = _mm512_add_ps(csum1, _mm512_add_ps(m1, f1));
            csum2 = _mm512_add_ps(csum2, _mm512_add_ps(m2, f2));
            csum3 = _mm512_add_ps(csum3, _mm512_add_ps(m3, f3));
        }
        csum0 = _mm512_max_ps(minv, _mm512_min_ps(maxv, csum0));
        csum1 = _mm512_max_ps(minv, _mm512_min_ps(maxv, csum1));
        csum2 = _mm512_max_ps(minv, _mm512_min_ps(maxv, csum2));
        csum3 = _mm512_max_ps(minv, _mm512_min_ps(maxv, csum3));
        _mm512_storeu_ps(dstZ, csum0);
        _mm512_storeu_ps(dstZ + 16 * plane->planeSize, csum1);
        _mm512_storeu_ps(dstZ + 32 * plane->planeSize, csum2);
        _mm512_storeu_ps(dstZ + 48 * plane->planeSize, csum3);
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
void _AVX512_MNNFloat2Int8(const float* src, int8_t* dst, size_t sizeQuad, const float* scalep, ssize_t minV, ssize_t maxV, const float* zeroPoint, ssize_t quanParamVec) {
    auto zero = _mm256_set1_epi32(0);
    auto minValue = _mm256_set1_ps(minV);
    auto maxValue = _mm256_set1_ps(maxV);
    auto zeroPointValue0 = _mm256_set1_ps(zeroPoint[0]);
    auto zeroPointValue1 = zeroPointValue0;
    auto offset = _mm256_set1_epi32(128);
    auto plus = _mm256_set1_ps(0.5f);
    auto minus = _mm256_set1_ps(-0.5f);
    auto scaleValue0 = _mm256_set1_ps(scalep[0]);
    auto scaleValue1 = scaleValue0;
    if (quanParamVec & 1) {
        scaleValue0 = _mm256_loadu_ps(scalep);
        scaleValue1 = _mm256_loadu_ps(scalep + 8);
    }
    if (quanParamVec >> 1) {
        zeroPointValue0 = _mm256_loadu_ps(zeroPoint);
        zeroPointValue1 = _mm256_loadu_ps(zeroPoint + 8);
    }

    for (int i = 0; i < sizeQuad; ++i) {
        auto f0 = _mm256_loadu_ps(src + PACK_UNIT * i);
        auto f1 = _mm256_loadu_ps(src + PACK_UNIT * i + 8);
        f0 = _mm256_mul_ps(f0, scaleValue0);
        f1 = _mm256_mul_ps(f1, scaleValue1);
        f0 = _mm256_add_ps(f0, zeroPointValue0);
        f1 = _mm256_add_ps(f1, zeroPointValue1);
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
        *((int64_t*)dst + 2 * i + 0) = _mm256_extract_epi64(_mm256_packus_epi16(d0, _mm256_setzero_si256()), 0);
        *((int64_t*)dst + 2 * i + 1) = _mm256_extract_epi64(_mm256_packus_epi16(d1, _mm256_setzero_si256()), 0);
    }
}

void _AVX512_MNNInt8ScaleToFloat(float* dst, const int8_t* src, const float* scale, size_t sizeQuad, const float* zeroPoint, ssize_t quanParamVec) {
    auto sizeC4 = sizeQuad / 2;
    auto sizeRemain = sizeQuad % 2;
    auto zero = _mm256_set1_epi32(0);
    auto offset = _mm256_set1_ps(128.f);
    
    auto scaleValue0 = _mm256_set1_ps(scale[0]);
    auto scaleValue1 = scaleValue0;
    if (quanParamVec & 1) {
        scaleValue0 = _mm256_loadu_ps(scale);
        scaleValue1 = _mm256_loadu_ps(scale + 8);
    }
    auto zeroPointValue0 = _mm256_add_ps(_mm256_set1_ps(zeroPoint[0]), offset);
    auto zeroPointValue1 = zeroPointValue0;
    if (quanParamVec >> 1) {
        zeroPointValue0 = _mm256_add_ps(_mm256_loadu_ps(zeroPoint), offset);
        zeroPointValue1 = _mm256_add_ps(_mm256_loadu_ps(zeroPoint + 8), offset);
    }

    for (int i = 0; i < sizeC4; ++i) {
        auto s = _mm256_castps_si256(_mm256_loadu_ps((const float*)(src)));
        auto s0_16 = _mm256_permute4x64_epi64(_mm256_unpacklo_epi8(s, zero), 0XD8);
        auto s1_16 = _mm256_permute4x64_epi64(_mm256_unpackhi_epi8(s, zero), 0xD8);
        auto s0_32 = _mm256_unpacklo_epi16(s0_16, zero);
        auto s1_32 = _mm256_unpacklo_epi16(s1_16, zero);
        auto s2_32 = _mm256_unpackhi_epi16(s0_16, zero);
        auto s3_32 = _mm256_unpackhi_epi16(s1_16, zero);
        auto s0_f = _mm256_cvtepi32_ps(s0_32);
        auto s1_f = _mm256_cvtepi32_ps(s1_32);
        auto s2_f = _mm256_cvtepi32_ps(s2_32);
        auto s3_f = _mm256_cvtepi32_ps(s3_32);
        s0_f = _mm256_sub_ps(s0_f, zeroPointValue0);
        s1_f = _mm256_sub_ps(s1_f, zeroPointValue1);
        s2_f = _mm256_sub_ps(s2_f, zeroPointValue0);
        s3_f = _mm256_sub_ps(s3_f, zeroPointValue1);
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
        auto s0_f = _mm256_cvtepi32_ps(s0_32);
        auto s1_f = _mm256_cvtepi32_ps(s1_32);
        s0_f = _mm256_sub_ps(s0_f, zeroPointValue0);
        s1_f = _mm256_sub_ps(s1_f, zeroPointValue1);
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
    // TMac
    gAVX2CoreInt8Functions->MNNTMacCompute = _AVX512_MNNTMacCompute;
}
