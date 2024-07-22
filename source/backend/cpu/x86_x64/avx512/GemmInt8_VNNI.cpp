//
//  GemmInt8_VNNI.cpp
//  MNN
//
//  Created by MNN on 2021/09/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef MNN_AVX512_VNNI

#include "FunctionSummary.hpp"
#include "GemmInt8Macro.h"
#define GEMMINT8_AVX512_H GEMMINT8_AVX512_H_VNNI
#define _MM256_SET_M128I(__H, __L) _mm256_insertf128_si256(_mm256_castsi128_si256(__L), __H, 1) // for compile compatiable
#define AVX512_BROADCAST_INT32(src) _mm512_castps_si512(_mm512_broadcastss_ps(_mm_load_ss(src)))

#define DEQUANT_VALUE(N) \
    auto f##N = _mm512_cvtepi32_ps(D##N);\
    f##N = _mm512_mul_ps(f##N, scaleValue);

#define SCALE_BIAS_VEC(N) \
    f##N = _mm512_add_ps(f##N, biasValue);

#define POSTTREAT(N, O) \
                f##N = _mm512_min_ps(f##N, maxValue);\
                f##N = _mm512_max_ps(f##N, minValue);\
                auto m##N = _mm512_cmp_ps_mask(f##N, zero512, 1);\
                auto b##N = _mm512_mask_blend_ps(m##N, plus, minus);\
                f##N = _mm512_add_ps(f##N, b##N);\
                auto d##N = _mm512_cvtps_epi32(_mm512_roundscale_ps(f##N, 3));\
                auto hd##N = _mm512_cvtsepi32_epi16(d##N); hd##N = _mm256_add_epi16(hd##N, offset);\
                auto h0##N = _mm256_extracti128_si256(hd##N, 0);\
                auto h1##N = _mm256_extracti128_si256(hd##N, 1);\
                h0##N = _mm_packus_epi16(h0##N, h1##N);\
                _mm_storeu_si128((__m128i*)dst_x + O, h0##N);

#define POST_TREAT_FLOAT(N,M,K,V) \
                f##N = _mm512_min_ps(f##N, fp32max);\
                f##N = _mm512_max_ps(f##N, fp32min);\
                f##M = _mm512_min_ps(f##M, fp32max);\
                f##M = _mm512_max_ps(f##M, fp32min);\
                f##K = _mm512_min_ps(f##K, fp32max);\
                f##K = _mm512_max_ps(f##K, fp32min);\
                f##V = _mm512_min_ps(f##V, fp32max);\
                f##V = _mm512_max_ps(f##V, fp32min);

#define SRCKERNELSUM_MUL_WEIGHTQUANBIAS \
                xy0_0 = _mm512_mul_ps(kernelSum0, weightBiasValue);\
                xy0_1 = _mm512_mul_ps(kernelSum1, weightBiasValue);\
                xy0_2 = _mm512_mul_ps(kernelSum2, weightBiasValue);\
                xy0_3 = _mm512_mul_ps(kernelSum3, weightBiasValue);

#define PLUS_TERM(N,M,K,V) \
                f##N = _mm512_add_ps(f##N, xy0_0);\
                f##M = _mm512_add_ps(f##M, xy0_1);\
                f##K = _mm512_add_ps(f##K, xy0_2);\
                f##V = _mm512_add_ps(f##V, xy0_3);

#define POST_TREAT_FLOAT_3(N,M,K) \
                f##N = _mm512_min_ps(f##N, fp32max);\
                f##N = _mm512_max_ps(f##N, fp32min);\
                f##M = _mm512_min_ps(f##M, fp32max);\
                f##M = _mm512_max_ps(f##M, fp32min);\
                f##K = _mm512_min_ps(f##K, fp32max);\
                f##K = _mm512_max_ps(f##K, fp32min);

#define SRCKERNELSUM_MUL_WEIGHTQUANBIAS_3 \
                xy0_0 = _mm512_mul_ps(kernelSum0, weightBiasValue);\
                xy0_1 = _mm512_mul_ps(kernelSum1, weightBiasValue);\
                xy0_2 = _mm512_mul_ps(kernelSum2, weightBiasValue);

#define PLUS_TERM_3(N,M,K) \
                f##N = _mm512_add_ps(f##N, xy0_0);\
                f##M = _mm512_add_ps(f##M, xy0_1);\
                f##K = _mm512_add_ps(f##K, xy0_2);

#define POST_TREAT_FLOAT_2(N,M) \
                f##N = _mm512_min_ps(f##N, fp32max);\
                f##N = _mm512_max_ps(f##N, fp32min);\
                f##M = _mm512_min_ps(f##M, fp32max);\
                f##M = _mm512_max_ps(f##M, fp32min);

#define SRCKERNELSUM_MUL_WEIGHTQUANBIAS_2 \
                xy0_0 = _mm512_mul_ps(kernelSum0, weightBiasValue);\
                xy0_1 = _mm512_mul_ps(kernelSum1, weightBiasValue);

#define PLUS_TERM_2(N,M) \
                f##N = _mm512_add_ps(f##N, xy0_0);\
                f##M = _mm512_add_ps(f##M, xy0_1);

#define POST_TREAT_FLOAT_1(N) \
                f##N = _mm512_min_ps(f##N, fp32max);\
                f##N = _mm512_max_ps(f##N, fp32min);

#define SRCKERNELSUM_MUL_WEIGHTQUANBIAS_1 \
                xy0_0 = _mm512_mul_ps(kernelSum0, weightBiasValue);

#define PLUS_TERM_1(N) \
                f##N = _mm512_add_ps(f##N, xy0_0);

// GemmInt8 with VNNI
void _AVX512_MNNGemmInt8AddBiasScale_16x4_Unit_VNNI(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad, const QuanPostTreatParameters* post, size_t realDst) {
    const auto dst_step_tmp = dst_step / sizeof(int8_t);
    auto zero512 = _mm512_set1_ps(0.0f);
    auto minValue = _mm512_set1_ps(post->minValue);
    auto maxValue = _mm512_set1_ps(post->maxValue);
    auto plus = _mm512_set1_ps(0.5f);
    auto minus = _mm512_set1_ps(-0.5f);
    auto offset = _mm256_set1_epi16(128);
    int dzUnit = GEMMINT8_AVX512_H / PACK_UNIT;
    int dzU = dst_depth_quad / dzUnit;
    int dzR = dst_depth_quad % dzUnit;
    __m512 fp32min, fp32max;
    if (0 == post->useInt8 && post->fp32minmax) {
        fp32min = _mm512_set1_ps((post->fp32minmax)[0]);
        fp32max = _mm512_set1_ps((post->fp32minmax)[1]);
    }
    auto blockNum = post->blockNum;
    const float* biasPtr = nullptr;
    const float* bias_dz = nullptr;
    const float* extraB_dz = nullptr;
    if (post->biasFloat) {
        biasPtr = post->biasFloat;
    }
    auto srcKernelSumPtr = post->srcKernelSum;
    __m512 kernelSum0 = _mm512_setzero_ps();
    __m512 kernelSum1 = _mm512_setzero_ps();
    __m512 kernelSum2 = _mm512_setzero_ps();
    __m512 kernelSum3 = _mm512_setzero_ps();
    if (GEMMINT8_AVX512_E == realDst) {
        kernelSum0 = _mm512_set1_ps(post->srcKernelSum[0]);
        kernelSum1 = _mm512_set1_ps(post->srcKernelSum[1]);
        kernelSum2 = _mm512_set1_ps(post->srcKernelSum[2]);
        kernelSum3 = _mm512_set1_ps(post->srcKernelSum[3]);
    } else {
        kernelSum0 = _mm512_set1_ps(post->srcKernelSum[0]);
        if (realDst > 1) {
            kernelSum1 = _mm512_set1_ps(post->srcKernelSum[1]);
        }
        if (realDst > 2) {
            kernelSum2 = _mm512_set1_ps(post->srcKernelSum[2]);
        }
    }
    auto f128   = _mm512_set1_ps(128.f);
    __m512 extrascale0 = _mm512_setzero_ps();
    __m512 extrascale1 = _mm512_setzero_ps();
    __m512 extrascale2 = _mm512_setzero_ps();
    __m512 extrascale3 = _mm512_setzero_ps();
    if (post->extraScale) {
        if (GEMMINT8_AVX512_E == realDst) {
            extrascale0 = _mm512_set1_ps(post->extraScale[0]);
            extrascale1 = _mm512_set1_ps(post->extraScale[1]);
            extrascale2 = _mm512_set1_ps(post->extraScale[2]);
            extrascale3 = _mm512_set1_ps(post->extraScale[3]);
        } else {
            extrascale0 = _mm512_set1_ps(post->extraScale[0]);
            if (realDst > 1) {
                extrascale1 = _mm512_set1_ps(post->extraScale[1]);
            }
            if (realDst > 2) {
                extrascale2 = _mm512_set1_ps(post->extraScale[2]);
            }
        }
    }
    int weightZStride = blockNum * src_depth_quad * (GEMMINT8_AVX512_L * GEMMINT8_AVX512_H);
    if (realDst == GEMMINT8_AVX512_E) {
        for (int dz = 0; dz < dzU; ++dz) {
            auto weight_dz = weight + dz * weightZStride;
            if (biasPtr) {
                bias_dz = biasPtr + dz * PACK_UNIT * dzUnit;
            }
            if (post->extraBias) {
                extraB_dz = post->extraBias + dz * PACK_UNIT * dzUnit;
            }
            const auto weightBias_dz = post->weightQuanBias + dz * PACK_UNIT * dzUnit;
            float* scale_dz = (float*)post->scale + dz * PACK_UNIT * dzUnit;
            auto dst_z = dst + dz * dst_step_tmp * dzUnit;
            const auto src_x   = src;
            auto dst_x         = dst_z;
            __m512i D0 = _mm512_set1_epi32(0);
            __m512i D1 = _mm512_set1_epi32(0);
            __m512i D2 = _mm512_set1_epi32(0);
            __m512i D3 = _mm512_set1_epi32(0);

            __m512i D4 = _mm512_set1_epi32(0);
            __m512i D5 = _mm512_set1_epi32(0);
            __m512i D6 = _mm512_set1_epi32(0);
            __m512i D7 = _mm512_set1_epi32(0);

            __m512i D8 = _mm512_set1_epi32(0);
            __m512i D9 = _mm512_set1_epi32(0);
            __m512i D10 = _mm512_set1_epi32(0);
            __m512i D11 = _mm512_set1_epi32(0);

            __m512i D12 = _mm512_set1_epi32(0);
            __m512i D13 = _mm512_set1_epi32(0);
            __m512i D14 = _mm512_set1_epi32(0);
            __m512i D15 = _mm512_set1_epi32(0);


            for (int sz = 0; sz < src_depth_quad; ++sz) {
                const auto weight_sz = weight_dz + (GEMMINT8_AVX512_L * GEMMINT8_AVX512_H) * sz;
                const auto src_z     = (const float*)(src_x + sz * GEMMINT8_AVX512_E * GEMMINT8_AVX512_L);
                auto w0 = _mm512_loadu_si512(weight_sz);
                auto w1 = _mm512_loadu_si512(weight_sz + 1 * PACK_UNIT * GEMMINT8_AVX512_L);
                auto w2 = _mm512_loadu_si512(weight_sz + 2 * PACK_UNIT * GEMMINT8_AVX512_L);
                auto w3 = _mm512_loadu_si512(weight_sz + 3 * PACK_UNIT * GEMMINT8_AVX512_L);

                auto s0 = AVX512_BROADCAST_INT32(src_z + 0);
                auto s1 = AVX512_BROADCAST_INT32(src_z + 1);
                auto s2 = AVX512_BROADCAST_INT32(src_z + 2);
                auto s3 = AVX512_BROADCAST_INT32(src_z + 3);

                D0 = _mm512_dpbusds_epi32(D0, s0, w0);
                D1 = _mm512_dpbusds_epi32(D1, s1, w0);
                D2 = _mm512_dpbusds_epi32(D2, s2, w0);
                D3 = _mm512_dpbusds_epi32(D3, s3, w0);

                D4 = _mm512_dpbusds_epi32(D4, s0, w1);
                D5 = _mm512_dpbusds_epi32(D5, s1, w1);
                D6 = _mm512_dpbusds_epi32(D6, s2, w1);
                D7 = _mm512_dpbusds_epi32(D7, s3, w1);

                D8 = _mm512_dpbusds_epi32(D8, s0, w2);
                D9 = _mm512_dpbusds_epi32(D9, s1, w2);
                D10 = _mm512_dpbusds_epi32(D10, s2, w2);
                D11 = _mm512_dpbusds_epi32(D11, s3, w2);

                D12 = _mm512_dpbusds_epi32(D12, s0, w3);
                D13 = _mm512_dpbusds_epi32(D13, s1, w3);
                D14 = _mm512_dpbusds_epi32(D14, s2, w3);
                D15 = _mm512_dpbusds_epi32(D15, s3, w3);
            }
            auto scaleValue = _mm512_loadu_ps(scale_dz);
            auto weightBiasValue = _mm512_loadu_ps(weightBias_dz);
            __m512 xy0_0, xy0_1, xy0_2, xy0_3;
            // x_kernelSum x w_quantZero
            SRCKERNELSUM_MUL_WEIGHTQUANBIAS;
            DEQUANT_VALUE(0);
            DEQUANT_VALUE(1);
            DEQUANT_VALUE(2);
            DEQUANT_VALUE(3);

            if (post->extraScale) { // Batch quant
                f0 = _mm512_mul_ps(f0, extrascale0);
                f1 = _mm512_mul_ps(f1, extrascale1);
                f2 = _mm512_mul_ps(f2, extrascale2);
                f3 = _mm512_mul_ps(f3, extrascale3);
                if (post->extraBias && nullptr != biasPtr) {
                    auto extrabias = _mm512_loadu_ps(extraB_dz);
                    extrabias = _mm512_mul_ps(f128, extrabias);
                    auto extrabias0 = _mm512_mul_ps(extrabias, extrascale0);
                    auto extrabias1 = _mm512_mul_ps(extrabias, extrascale1);
                    auto extrabias2 = _mm512_mul_ps(extrabias, extrascale2);
                    auto extrabias3 = _mm512_mul_ps(extrabias, extrascale3);
                    f0 = _mm512_sub_ps(f0, extrabias0);
                    f1 = _mm512_sub_ps(f1, extrabias1);
                    f2 = _mm512_sub_ps(f2, extrabias2);
                    f3 = _mm512_sub_ps(f3, extrabias3);
                }
            }

            PLUS_TERM(0,1,2,3);
            if (nullptr != biasPtr) {
                auto biasValue = _mm512_loadu_ps(bias_dz);
                SCALE_BIAS_VEC(0);
                SCALE_BIAS_VEC(1);
                SCALE_BIAS_VEC(2);
                SCALE_BIAS_VEC(3);
            }

            scaleValue = _mm512_loadu_ps(scale_dz + 1 * PACK_UNIT);
            weightBiasValue = _mm512_loadu_ps(weightBias_dz + 1 * PACK_UNIT);
            // x_kernelSum x w_quantZero
            SRCKERNELSUM_MUL_WEIGHTQUANBIAS;
            DEQUANT_VALUE(4);
            DEQUANT_VALUE(5);
            DEQUANT_VALUE(6);
            DEQUANT_VALUE(7);

            if (post->extraScale) { // Batch quant
                f4 = _mm512_mul_ps(f4, extrascale0);
                f5 = _mm512_mul_ps(f5, extrascale1);
                f6 = _mm512_mul_ps(f6, extrascale2);
                f7 = _mm512_mul_ps(f7, extrascale3);
                if (post->extraBias && nullptr != biasPtr) {
                    auto extrabias = _mm512_loadu_ps(extraB_dz + 1 * PACK_UNIT);
                    extrabias = _mm512_mul_ps(f128, extrabias);
                    auto extrabias0 = _mm512_mul_ps(extrabias, extrascale0);
                    auto extrabias1 = _mm512_mul_ps(extrabias, extrascale1);
                    auto extrabias2 = _mm512_mul_ps(extrabias, extrascale2);
                    auto extrabias3 = _mm512_mul_ps(extrabias, extrascale3);
                    f4 = _mm512_sub_ps(f4, extrabias0);
                    f5 = _mm512_sub_ps(f5, extrabias1);
                    f6 = _mm512_sub_ps(f6, extrabias2);
                    f7 = _mm512_sub_ps(f7, extrabias3);
                }
            }

            PLUS_TERM(4,5,6,7);
            if (nullptr != biasPtr) {
                auto biasValue = _mm512_loadu_ps(bias_dz + 1 * PACK_UNIT);
                SCALE_BIAS_VEC(4);
                SCALE_BIAS_VEC(5);
                SCALE_BIAS_VEC(6);
                SCALE_BIAS_VEC(7);
            }

            scaleValue = _mm512_loadu_ps(scale_dz + 2 * PACK_UNIT);
            weightBiasValue = _mm512_loadu_ps(weightBias_dz + 2 * PACK_UNIT);
            // x_kernelSum x w_quantZero
            SRCKERNELSUM_MUL_WEIGHTQUANBIAS;
            DEQUANT_VALUE(8);
            DEQUANT_VALUE(9);
            DEQUANT_VALUE(10);
            DEQUANT_VALUE(11);

            if (post->extraScale) { // Batch quant
                f8 = _mm512_mul_ps(f8, extrascale0);
                f9 = _mm512_mul_ps(f9, extrascale1);
                f10 = _mm512_mul_ps(f10, extrascale2);
                f11 = _mm512_mul_ps(f11, extrascale3);
                if (post->extraBias && nullptr != biasPtr) {
                    auto extrabias = _mm512_loadu_ps(extraB_dz + 2 * PACK_UNIT);
                    extrabias = _mm512_mul_ps(f128, extrabias);
                    auto extrabias0 = _mm512_mul_ps(extrabias, extrascale0);
                    auto extrabias1 = _mm512_mul_ps(extrabias, extrascale1);
                    auto extrabias2 = _mm512_mul_ps(extrabias, extrascale2);
                    auto extrabias3 = _mm512_mul_ps(extrabias, extrascale3);
                    f8 = _mm512_sub_ps(f8, extrabias0);
                    f9 = _mm512_sub_ps(f9, extrabias1);
                    f10 = _mm512_sub_ps(f10, extrabias2);
                    f11 = _mm512_sub_ps(f11, extrabias3);
                }
            }

            PLUS_TERM(8,9,10,11);
            if (nullptr != biasPtr) {
                auto biasValue = _mm512_loadu_ps(bias_dz + 2 * PACK_UNIT);
                SCALE_BIAS_VEC(8);
                SCALE_BIAS_VEC(9);
                SCALE_BIAS_VEC(10);
                SCALE_BIAS_VEC(11);
            }

            scaleValue = _mm512_loadu_ps(scale_dz + 3 * PACK_UNIT);
            weightBiasValue = _mm512_loadu_ps(weightBias_dz + 3 * PACK_UNIT);
            // x_kernelSum x w_quantZero
            SRCKERNELSUM_MUL_WEIGHTQUANBIAS;
            DEQUANT_VALUE(12);
            DEQUANT_VALUE(13);
            DEQUANT_VALUE(14);
            DEQUANT_VALUE(15);

            if (post->extraScale) { // Batch quant
                f12 = _mm512_mul_ps(f12, extrascale0);
                f13 = _mm512_mul_ps(f13, extrascale1);
                f14 = _mm512_mul_ps(f14, extrascale2);
                f15 = _mm512_mul_ps(f15, extrascale3);
                if (post->extraBias && nullptr != biasPtr) {
                    auto extrabias = _mm512_loadu_ps(extraB_dz + 3 * PACK_UNIT);
                    extrabias = _mm512_mul_ps(f128, extrabias);
                    auto extrabias0 = _mm512_mul_ps(extrabias, extrascale0);
                    auto extrabias1 = _mm512_mul_ps(extrabias, extrascale1);
                    auto extrabias2 = _mm512_mul_ps(extrabias, extrascale2);
                    auto extrabias3 = _mm512_mul_ps(extrabias, extrascale3);
                    f12 = _mm512_sub_ps(f12, extrabias0);
                    f13 = _mm512_sub_ps(f13, extrabias1);
                    f14 = _mm512_sub_ps(f14, extrabias2);
                    f15 = _mm512_sub_ps(f15, extrabias3);
                }
            }

            PLUS_TERM(12,13,14,15);
            if (nullptr != biasPtr) {
                auto biasValue = _mm512_loadu_ps(bias_dz + 3 * PACK_UNIT);
                SCALE_BIAS_VEC(12);
                SCALE_BIAS_VEC(13);
                SCALE_BIAS_VEC(14);
                SCALE_BIAS_VEC(15);
            }

            if (post->useInt8 == 0) {
                if (biasPtr == nullptr) {
                    auto destTmp = dst_x;
                    f0 = _mm512_add_ps(_mm512_loadu_ps((float*)destTmp), f0);
                    f1 = _mm512_add_ps(_mm512_loadu_ps(((float*)destTmp) + 16), f1);
                    f2 = _mm512_add_ps(_mm512_loadu_ps(((float*)destTmp) + 16 * 2), f2);
                    f3 = _mm512_add_ps(_mm512_loadu_ps(((float*)destTmp) + 16 * 3), f3);
                    destTmp += dst_step_tmp;
                    f4 = _mm512_add_ps(_mm512_loadu_ps(((float*)destTmp) + 16 * 0), f4);
                    f5 = _mm512_add_ps(_mm512_loadu_ps(((float*)destTmp) + 16 * 1), f5);
                    f6 = _mm512_add_ps(_mm512_loadu_ps(((float*)destTmp) + 16 * 2), f6);
                    f7 = _mm512_add_ps(_mm512_loadu_ps(((float*)destTmp) + 16 * 3), f7);
                    destTmp += dst_step_tmp;
                    f8 = _mm512_add_ps(_mm512_loadu_ps(((float*)destTmp) + 16 * 0), f8);
                    f9 = _mm512_add_ps(_mm512_loadu_ps(((float*)destTmp) + 16 * 1), f9);
                    f10 = _mm512_add_ps(_mm512_loadu_ps(((float*)destTmp) + 16 * 2), f10);
                    f11 = _mm512_add_ps(_mm512_loadu_ps(((float*)destTmp) + 16 * 3), f11);
                    destTmp += dst_step_tmp;
                    f12 = _mm512_add_ps(_mm512_loadu_ps(((float*)destTmp) + 16 * 0), f12);
                    f13 = _mm512_add_ps(_mm512_loadu_ps(((float*)destTmp) + 16 * 1), f13);
                    f14 = _mm512_add_ps(_mm512_loadu_ps(((float*)destTmp) + 16 * 2), f14);
                    f15 = _mm512_add_ps(_mm512_loadu_ps(((float*)destTmp) + 16 * 3), f15);
                }
                if (post->fp32minmax) {
                    POST_TREAT_FLOAT(0,1,2,3);
                    POST_TREAT_FLOAT(4,5,6,7);
                    POST_TREAT_FLOAT(8,9,10,11);
                    POST_TREAT_FLOAT(12,13,14,15);
                }

                _mm512_storeu_ps(((float*)dst_x), f0);
                _mm512_storeu_ps(((float*)dst_x) + 16, f1);
                _mm512_storeu_ps(((float*)dst_x) + 16 * 2, f2);
                _mm512_storeu_ps(((float*)dst_x) + 16 * 3, f3);
                dst_x += dst_step_tmp;
                _mm512_storeu_ps(((float*)dst_x) + 16 * 0, f4);
                _mm512_storeu_ps(((float*)dst_x) + 16 * 1, f5);
                _mm512_storeu_ps(((float*)dst_x) + 16 * 2, f6);
                _mm512_storeu_ps(((float*)dst_x) + 16 * 3, f7);
                dst_x += dst_step_tmp;
                _mm512_storeu_ps(((float*)dst_x) + 16 * 0, f8);
                _mm512_storeu_ps(((float*)dst_x) + 16 * 1, f9);
                _mm512_storeu_ps(((float*)dst_x) + 16 * 2, f10);
                _mm512_storeu_ps(((float*)dst_x) + 16 * 3, f11);
                dst_x += dst_step_tmp;
                _mm512_storeu_ps(((float*)dst_x) + 16 * 0, f12);
                _mm512_storeu_ps(((float*)dst_x) + 16 * 1, f13);
                _mm512_storeu_ps(((float*)dst_x) + 16 * 2, f14);
                _mm512_storeu_ps(((float*)dst_x) + 16 * 3, f15);
            } else {
                POSTTREAT(0, 0);
                POSTTREAT(1, 1);
                POSTTREAT(2, 2);
                POSTTREAT(3, 3);
                dst_x += dst_step_tmp;

                POSTTREAT(4, 0);
                POSTTREAT(5, 1);
                POSTTREAT(6, 2);
                POSTTREAT(7, 3);
                dst_x += dst_step_tmp;

                POSTTREAT(8, 0);
                POSTTREAT(9, 1);
                POSTTREAT(10, 2);
                POSTTREAT(11, 3);
                dst_x += dst_step_tmp;

                POSTTREAT(12, 0);
                POSTTREAT(13, 1);
                POSTTREAT(14, 2);
                POSTTREAT(15, 3);
            }
        }
        auto weight_dz = weight + dzU * weightZStride;
        if (biasPtr) {
            bias_dz = biasPtr + dzU * PACK_UNIT * dzUnit;
        }
        if (post->extraBias) {
            extraB_dz = post->extraBias + dzU * PACK_UNIT * dzUnit;
        }
        float* scale_dz = (float*)post->scale + dzU * PACK_UNIT * dzUnit;
        const auto weightBias_dz = post->weightQuanBias + dzU * PACK_UNIT * dzUnit;

        auto dst_z = dst + dzU * dst_step_tmp * dzUnit;
        const auto src_x   = src;
        auto dst_x         = dst_z;
        for (int i=0; i<dzR; ++i) {
            __m512i D0 = _mm512_set1_epi32(0);
            __m512i D1 = _mm512_set1_epi32(0);
            __m512i D2 = _mm512_set1_epi32(0);
            __m512i D3 = _mm512_set1_epi32(0);

            for (int sz = 0; sz < src_depth_quad; ++sz) {
                const auto weight_sz = weight_dz + (GEMMINT8_AVX512_L * GEMMINT8_AVX512_H) * sz;
                const auto src_z     = (const float*)(src_x + sz * GEMMINT8_AVX512_E * GEMMINT8_AVX512_L);
                auto w0 = _mm512_loadu_si512(weight_sz);

                auto s0 = AVX512_BROADCAST_INT32(src_z + 0);
                auto s1 = AVX512_BROADCAST_INT32(src_z + 1);
                auto s2 = AVX512_BROADCAST_INT32(src_z + 2);
                auto s3 = AVX512_BROADCAST_INT32(src_z + 3);

                D0 = _mm512_dpbusds_epi32(D0, s0, w0);
                D1 = _mm512_dpbusds_epi32(D1, s1, w0);
                D2 = _mm512_dpbusds_epi32(D2, s2, w0);
                D3 = _mm512_dpbusds_epi32(D3, s3, w0);
            }

            auto scaleValue = _mm512_loadu_ps(scale_dz);
            auto weightBiasValue = _mm512_loadu_ps(weightBias_dz);
            __m512 xy0_0, xy0_1, xy0_2, xy0_3;
            // x_kernelSum x w_quantZero
            SRCKERNELSUM_MUL_WEIGHTQUANBIAS;
            DEQUANT_VALUE(0);
            DEQUANT_VALUE(1);
            DEQUANT_VALUE(2);
            DEQUANT_VALUE(3);

            if (post->extraScale) { // Batch quant
                f0 = _mm512_mul_ps(f0, extrascale0);
                f1 = _mm512_mul_ps(f1, extrascale1);
                f2 = _mm512_mul_ps(f2, extrascale2);
                f3 = _mm512_mul_ps(f3, extrascale3);
                if (post->extraBias && nullptr != biasPtr) {
                    auto extrabias = _mm512_loadu_ps(extraB_dz);
                    extrabias = _mm512_mul_ps(f128, extrabias);
                    auto extrabias0 = _mm512_mul_ps(extrabias, extrascale0);
                    auto extrabias1 = _mm512_mul_ps(extrabias, extrascale1);
                    auto extrabias2 = _mm512_mul_ps(extrabias, extrascale2);
                    auto extrabias3 = _mm512_mul_ps(extrabias, extrascale3);
                    f0 = _mm512_sub_ps(f0, extrabias0);
                    f1 = _mm512_sub_ps(f1, extrabias1);
                    f2 = _mm512_sub_ps(f2, extrabias2);
                    f3 = _mm512_sub_ps(f3, extrabias3);
                }
            }

            PLUS_TERM(0,1,2,3);
            if (nullptr != biasPtr) {
                auto biasValue = _mm512_loadu_ps(bias_dz);
                SCALE_BIAS_VEC(0);
                SCALE_BIAS_VEC(1);
                SCALE_BIAS_VEC(2);
                SCALE_BIAS_VEC(3);
            }

            if (post->useInt8 == 0) {
                if (nullptr == biasPtr) {
                    f0 = _mm512_add_ps(_mm512_loadu_ps((float*)dst_x), f0);
                    f1 = _mm512_add_ps(_mm512_loadu_ps(((float*)dst_x) + 16), f1);
                    f2 = _mm512_add_ps(_mm512_loadu_ps(((float*)dst_x) + 16 * 2), f2);
                    f3 = _mm512_add_ps(_mm512_loadu_ps(((float*)dst_x) + 16 * 3), f3);
                }
                if (post->fp32minmax) {
                    POST_TREAT_FLOAT(0,1,2,3);
                }
                _mm512_storeu_ps(((float*)dst_x), f0);
                _mm512_storeu_ps(((float*)dst_x) + 16, f1);
                _mm512_storeu_ps(((float*)dst_x) + 16 * 2, f2);
                _mm512_storeu_ps(((float*)dst_x) + 16 * 3, f3);
            } else {
                POSTTREAT(0, 0);
                POSTTREAT(1, 1);
                POSTTREAT(2, 2);
                POSTTREAT(3, 3);
            }
            dst_x += dst_step_tmp;
            scale_dz += PACK_UNIT;
            if (biasPtr) {
                bias_dz += PACK_UNIT;
            }
            if (post->extraBias) {
                extraB_dz += PACK_UNIT;
            }
            weight_dz += PACK_UNIT * GEMMINT8_AVX512_L;
        }
        return;
    }
    // e = 3
    if (realDst == 3) {
        for (int dz = 0; dz < dzU; ++dz) {
            auto weight_dz = weight + dz * weightZStride;
            if (biasPtr) {
                bias_dz = biasPtr + dz * PACK_UNIT * dzUnit;
            }
            if (post->extraBias) {
                extraB_dz = post->extraBias + dz * PACK_UNIT * dzUnit;
            }
            float* scale_dz = (float*)post->scale + dz * PACK_UNIT * dzUnit;
            const auto weightBias_dz = post->weightQuanBias + dz * PACK_UNIT * dzUnit;
            auto dst_z = dst + dz * dst_step_tmp * dzUnit;
            const auto src_x   = src;
            auto dst_x         = dst_z;
            __m512i D0 = _mm512_set1_epi32(0);
            __m512i D1 = _mm512_set1_epi32(0);
            __m512i D2 = _mm512_set1_epi32(0);

            __m512i D4 = _mm512_set1_epi32(0);
            __m512i D5 = _mm512_set1_epi32(0);
            __m512i D6 = _mm512_set1_epi32(0);

            __m512i D8 = _mm512_set1_epi32(0);
            __m512i D9 = _mm512_set1_epi32(0);
            __m512i D10 = _mm512_set1_epi32(0);

            __m512i D12 = _mm512_set1_epi32(0);
            __m512i D13 = _mm512_set1_epi32(0);
            __m512i D14 = _mm512_set1_epi32(0);


            for (int sz = 0; sz < src_depth_quad; ++sz) {
                const auto weight_sz = weight_dz + (GEMMINT8_AVX512_L * GEMMINT8_AVX512_H) * sz;
                const auto src_z     = (const float*)(src_x + sz * GEMMINT8_AVX512_E * GEMMINT8_AVX512_L);
                auto w0 = _mm512_loadu_si512(weight_sz);
                auto w1 = _mm512_loadu_si512(weight_sz + 1 * PACK_UNIT * GEMMINT8_AVX512_L);
                auto w2 = _mm512_loadu_si512(weight_sz + 2 * PACK_UNIT * GEMMINT8_AVX512_L);
                auto w3 = _mm512_loadu_si512(weight_sz + 3 * PACK_UNIT * GEMMINT8_AVX512_L);

                auto s0 = AVX512_BROADCAST_INT32(src_z + 0);
                auto s1 = AVX512_BROADCAST_INT32(src_z + 1);
                auto s2 = AVX512_BROADCAST_INT32(src_z + 2);

                D0 = _mm512_dpbusds_epi32(D0, s0, w0);
                D1 = _mm512_dpbusds_epi32(D1, s1, w0);
                D2 = _mm512_dpbusds_epi32(D2, s2, w0);

                D4 = _mm512_dpbusds_epi32(D4, s0, w1);
                D5 = _mm512_dpbusds_epi32(D5, s1, w1);
                D6 = _mm512_dpbusds_epi32(D6, s2, w1);

                D8 = _mm512_dpbusds_epi32(D8, s0, w2);
                D9 = _mm512_dpbusds_epi32(D9, s1, w2);
                D10 = _mm512_dpbusds_epi32(D10, s2, w2);

                D12 = _mm512_dpbusds_epi32(D12, s0, w3);
                D13 = _mm512_dpbusds_epi32(D13, s1, w3);
                D14 = _mm512_dpbusds_epi32(D14, s2, w3);
            }

            auto scaleValue = _mm512_loadu_ps(scale_dz);
            auto weightBiasValue = _mm512_loadu_ps(weightBias_dz);
            __m512 xy0_0, xy0_1, xy0_2;
            // x_kernelSum x w_quantZero
            SRCKERNELSUM_MUL_WEIGHTQUANBIAS_3;
            DEQUANT_VALUE(0);
            DEQUANT_VALUE(1);
            DEQUANT_VALUE(2);

            if (post->extraScale) { // Batch quant
                f0 = _mm512_mul_ps(f0, extrascale0);
                f1 = _mm512_mul_ps(f1, extrascale1);
                f2 = _mm512_mul_ps(f2, extrascale2);
                if (post->extraBias && nullptr != biasPtr) {
                    auto extrabias = _mm512_loadu_ps(extraB_dz);
                    extrabias = _mm512_mul_ps(f128, extrabias);
                    auto extrabias0 = _mm512_mul_ps(extrabias, extrascale0);
                    auto extrabias1 = _mm512_mul_ps(extrabias, extrascale1);
                    auto extrabias2 = _mm512_mul_ps(extrabias, extrascale2);
                    f0 = _mm512_sub_ps(f0, extrabias0);
                    f1 = _mm512_sub_ps(f1, extrabias1);
                    f2 = _mm512_sub_ps(f2, extrabias2);
                }
            }

            PLUS_TERM_3(0,1,2);
            if (nullptr != biasPtr) {
                auto biasValue = _mm512_loadu_ps(bias_dz);
                SCALE_BIAS_VEC(0);
                SCALE_BIAS_VEC(1);
                SCALE_BIAS_VEC(2);
            }

            scaleValue = _mm512_loadu_ps(scale_dz + 1 * PACK_UNIT);
            weightBiasValue = _mm512_loadu_ps(weightBias_dz + 1 * PACK_UNIT);
            // x_kernelSum x w_quantZero
            SRCKERNELSUM_MUL_WEIGHTQUANBIAS_3;
            DEQUANT_VALUE(4);
            DEQUANT_VALUE(5);
            DEQUANT_VALUE(6);

            if (post->extraScale) { // Batch quant
                f4 = _mm512_mul_ps(f4, extrascale0);
                f5 = _mm512_mul_ps(f5, extrascale1);
                f6 = _mm512_mul_ps(f6, extrascale2);
                if (post->extraBias && nullptr != biasPtr) {
                    auto extrabias = _mm512_loadu_ps(extraB_dz + 1 * PACK_UNIT);
                    extrabias = _mm512_mul_ps(f128, extrabias);
                    auto extrabias0 = _mm512_mul_ps(extrabias, extrascale0);
                    auto extrabias1 = _mm512_mul_ps(extrabias, extrascale1);
                    auto extrabias2 = _mm512_mul_ps(extrabias, extrascale2);
                    f4 = _mm512_sub_ps(f4, extrabias0);
                    f5 = _mm512_sub_ps(f5, extrabias1);
                    f6 = _mm512_sub_ps(f6, extrabias2);
                }
            }

            PLUS_TERM_3(4,5,6);
            if (biasPtr) {
                auto biasValue = _mm512_loadu_ps(bias_dz + 1 * PACK_UNIT);
                SCALE_BIAS_VEC(4);
                SCALE_BIAS_VEC(5);
                SCALE_BIAS_VEC(6);
            }

            scaleValue = _mm512_loadu_ps(scale_dz + 2 * PACK_UNIT);
            weightBiasValue = _mm512_loadu_ps(weightBias_dz + 2 * PACK_UNIT);
            // x_kernelSum x w_quantZero
            SRCKERNELSUM_MUL_WEIGHTQUANBIAS_3;
            DEQUANT_VALUE(8);
            DEQUANT_VALUE(9);
            DEQUANT_VALUE(10);

            if (post->extraScale) { // Batch quant
                f8 = _mm512_mul_ps(f8, extrascale0);
                f9 = _mm512_mul_ps(f9, extrascale1);
                f10 = _mm512_mul_ps(f10, extrascale2);
                if (post->extraBias && nullptr != biasPtr) {
                    auto extrabias = _mm512_loadu_ps(extraB_dz + 2 * PACK_UNIT);
                    extrabias = _mm512_mul_ps(f128, extrabias);
                    auto extrabias0 = _mm512_mul_ps(extrabias, extrascale0);
                    auto extrabias1 = _mm512_mul_ps(extrabias, extrascale1);
                    auto extrabias2 = _mm512_mul_ps(extrabias, extrascale2);
                    f8 = _mm512_sub_ps(f8, extrabias0);
                    f9 = _mm512_sub_ps(f9, extrabias1);
                    f10 = _mm512_sub_ps(f10, extrabias2);
                }
            }

            PLUS_TERM_3(8,9,10);
            if (biasPtr) {
                auto biasValue = _mm512_loadu_ps(bias_dz + 2 * PACK_UNIT);
                SCALE_BIAS_VEC(8);
                SCALE_BIAS_VEC(9);
                SCALE_BIAS_VEC(10);
            }

            scaleValue = _mm512_loadu_ps(scale_dz + 3 * PACK_UNIT);
            weightBiasValue = _mm512_loadu_ps(weightBias_dz + 3 * PACK_UNIT);
            // x_kernelSum x w_quantZero
            SRCKERNELSUM_MUL_WEIGHTQUANBIAS_3;
            DEQUANT_VALUE(12);
            DEQUANT_VALUE(13);
            DEQUANT_VALUE(14);

            if (post->extraScale) { // Batch quant
                f12 = _mm512_mul_ps(f12, extrascale0);
                f13 = _mm512_mul_ps(f13, extrascale1);
                f14 = _mm512_mul_ps(f14, extrascale2);
                if (post->extraBias && nullptr != biasPtr) {
                    auto extrabias = _mm512_loadu_ps(extraB_dz + 3 * PACK_UNIT);
                    extrabias = _mm512_mul_ps(f128, extrabias);
                    auto extrabias0 = _mm512_mul_ps(extrabias, extrascale0);
                    auto extrabias1 = _mm512_mul_ps(extrabias, extrascale1);
                    auto extrabias2 = _mm512_mul_ps(extrabias, extrascale2);
                    f12 = _mm512_sub_ps(f12, extrabias0);
                    f13 = _mm512_sub_ps(f13, extrabias1);
                    f14 = _mm512_sub_ps(f14, extrabias2);
                }
            }

            PLUS_TERM_3(12,13,14);
            if (biasPtr) {
                auto biasValue = _mm512_loadu_ps(bias_dz + 3 * PACK_UNIT);
                SCALE_BIAS_VEC(12);
                SCALE_BIAS_VEC(13);
                SCALE_BIAS_VEC(14);
            }

            if (post->useInt8 == 0) {
                if (biasPtr == nullptr) {
                    auto dstTmp = dst_x;
                    f0 = _mm512_add_ps(_mm512_loadu_ps(((float*)dstTmp)), f0);
                    f1 = _mm512_add_ps(_mm512_loadu_ps(((float*)dstTmp) + 16), f1);
                    f2 = _mm512_add_ps(_mm512_loadu_ps(((float*)dstTmp) + 16 * 2), f2);
                    dstTmp += dst_step_tmp;
                    f4 = _mm512_add_ps(_mm512_loadu_ps(((float*)dstTmp) + 16 * 0), f4);
                    f5 = _mm512_add_ps(_mm512_loadu_ps(((float*)dstTmp) + 16 * 1), f5);
                    f6 = _mm512_add_ps(_mm512_loadu_ps(((float*)dstTmp) + 16 * 2), f6);
                    dstTmp += dst_step_tmp;
                    f8 = _mm512_add_ps(_mm512_loadu_ps(((float*)dstTmp) + 16 * 0), f8);
                    f9 = _mm512_add_ps(_mm512_loadu_ps(((float*)dstTmp) + 16 * 1), f9);
                    f10 = _mm512_add_ps(_mm512_loadu_ps(((float*)dstTmp) + 16 * 2), f10);
                    dstTmp += dst_step_tmp;
                    f12 = _mm512_add_ps(_mm512_loadu_ps(((float*)dstTmp) + 16 * 0), f12);
                    f13 = _mm512_add_ps(_mm512_loadu_ps(((float*)dstTmp) + 16 * 1), f13);
                    f14 = _mm512_add_ps(_mm512_loadu_ps(((float*)dstTmp) + 16 * 2), f14);
                }
                if (post->fp32minmax) {
                    POST_TREAT_FLOAT_3(0,1,2);
                    POST_TREAT_FLOAT_3(4,5,6);
                    POST_TREAT_FLOAT_3(8,9,10);
                    POST_TREAT_FLOAT_3(12,13,14);
                }
                _mm512_storeu_ps(((float*)dst_x), f0);
                _mm512_storeu_ps(((float*)dst_x) + 16, f1);
                _mm512_storeu_ps(((float*)dst_x) + 16 * 2, f2);
                dst_x += dst_step_tmp;
                _mm512_storeu_ps(((float*)dst_x) + 16 * 0, f4);
                _mm512_storeu_ps(((float*)dst_x) + 16 * 1, f5);
                _mm512_storeu_ps(((float*)dst_x) + 16 * 2, f6);
                dst_x += dst_step_tmp;
                _mm512_storeu_ps(((float*)dst_x) + 16 * 0, f8);
                _mm512_storeu_ps(((float*)dst_x) + 16 * 1, f9);
                _mm512_storeu_ps(((float*)dst_x) + 16 * 2, f10);
                dst_x += dst_step_tmp;
                _mm512_storeu_ps(((float*)dst_x) + 16 * 0, f12);
                _mm512_storeu_ps(((float*)dst_x) + 16 * 1, f13);
                _mm512_storeu_ps(((float*)dst_x) + 16 * 2, f14);
            } else {
                POSTTREAT(0, 0);
                POSTTREAT(1, 1);
                POSTTREAT(2, 2);
                dst_x += dst_step_tmp;

                POSTTREAT(4, 0);
                POSTTREAT(5, 1);
                POSTTREAT(6, 2);
                dst_x += dst_step_tmp;

                POSTTREAT(8, 0);
                POSTTREAT(9, 1);
                POSTTREAT(10, 2);
                dst_x += dst_step_tmp;

                POSTTREAT(12, 0);
                POSTTREAT(13, 1);
                POSTTREAT(14, 2);
            }
        }
        auto weight_dz = weight + dzU * weightZStride;
        if (biasPtr) {
            bias_dz = post->biasFloat + dzU * PACK_UNIT * dzUnit;
        }
        if (post->extraBias) {
            extraB_dz = post->extraBias + dzU * PACK_UNIT * dzUnit;
        }
        float* scale_dz = (float*)post->scale + dzU * PACK_UNIT * dzUnit;
        const auto weightBias_dz = post->weightQuanBias + dzU * PACK_UNIT * dzUnit;

        auto dst_z = dst + dzU * dst_step_tmp * dzUnit;
        const auto src_x   = src;
        auto dst_x         = dst_z;
        for (int i=0; i<dzR; ++i) {
            __m512i D0 = _mm512_set1_epi32(0);
            __m512i D1 = _mm512_set1_epi32(0);
            __m512i D2 = _mm512_set1_epi32(0);

            for (int sz = 0; sz < src_depth_quad; ++sz) {
                const auto weight_sz = weight_dz + (GEMMINT8_AVX512_L * GEMMINT8_AVX512_H) * sz;
                const auto src_z     = (const float*)(src_x + sz * GEMMINT8_AVX512_E * GEMMINT8_AVX512_L);
                auto w0 = _mm512_loadu_si512(weight_sz);

                auto s0 = AVX512_BROADCAST_INT32(src_z + 0);
                auto s1 = AVX512_BROADCAST_INT32(src_z + 1);
                auto s2 = AVX512_BROADCAST_INT32(src_z + 2);

                D0 = _mm512_dpbusds_epi32(D0, s0, w0);
                D1 = _mm512_dpbusds_epi32(D1, s1, w0);
                D2 = _mm512_dpbusds_epi32(D2, s2, w0);
            }

            
            auto scaleValue = _mm512_loadu_ps(scale_dz);
            auto weightBiasValue = _mm512_loadu_ps(weightBias_dz);
            __m512 xy0_0, xy0_1, xy0_2;
            // x_kernelSum x w_quantZero
            SRCKERNELSUM_MUL_WEIGHTQUANBIAS_3;
            DEQUANT_VALUE(0);
            DEQUANT_VALUE(1);
            DEQUANT_VALUE(2);

            if (post->extraScale) { // Batch quant
                f0 = _mm512_mul_ps(f0, extrascale0);
                f1 = _mm512_mul_ps(f1, extrascale1);
                f2 = _mm512_mul_ps(f2, extrascale2);
                if (post->extraBias && nullptr != biasPtr) {
                    auto extrabias = _mm512_loadu_ps(extraB_dz);
                    extrabias = _mm512_mul_ps(f128, extrabias);
                    auto extrabias0 = _mm512_mul_ps(extrabias, extrascale0);
                    auto extrabias1 = _mm512_mul_ps(extrabias, extrascale1);
                    auto extrabias2 = _mm512_mul_ps(extrabias, extrascale2);
                    f0 = _mm512_sub_ps(f0, extrabias0);
                    f1 = _mm512_sub_ps(f1, extrabias1);
                    f2 = _mm512_sub_ps(f2, extrabias2);
                }
            }

            PLUS_TERM_3(0,1,2);
            if (biasPtr) {
                auto biasValue = _mm512_loadu_ps(bias_dz);
                SCALE_BIAS_VEC(0);
                SCALE_BIAS_VEC(1);
                SCALE_BIAS_VEC(2);
            }
            
            if (post->useInt8 == 0) {
                if (biasPtr == nullptr) {
                    f0 = _mm512_add_ps(_mm512_loadu_ps(((float*)dst_x)), f0);
                    f1 = _mm512_add_ps(_mm512_loadu_ps(((float*)dst_x) + 16), f1);
                    f2 = _mm512_add_ps(_mm512_loadu_ps(((float*)dst_x) + 16 * 2), f2);
                }
                if (post->fp32minmax) {
                    POST_TREAT_FLOAT_3(0,1,2);
                }
                _mm512_storeu_ps(((float*)dst_x), f0);
                _mm512_storeu_ps(((float*)dst_x) + 16, f1);
                _mm512_storeu_ps(((float*)dst_x) + 16 * 2, f2);
            } else {
                POSTTREAT(0, 0);
                POSTTREAT(1, 1);
                POSTTREAT(2, 2);
            }
            dst_x += dst_step_tmp;
            scale_dz += PACK_UNIT;
            if (biasPtr) {
                bias_dz += PACK_UNIT;
            }
            if (post->extraBias) {
                extraB_dz += PACK_UNIT;
            }
            weight_dz += PACK_UNIT * GEMMINT8_AVX512_L;
        }
        return;
    }
    // e = 2
    if (realDst == 2) {
        for (int dz = 0; dz < dzU; ++dz) {
            auto weight_dz = weight + dz * weightZStride;
            if (biasPtr) {
                bias_dz = post->biasFloat + dz * PACK_UNIT * dzUnit;
            }
            if (post->extraBias) {
                extraB_dz = post->extraBias + dz * PACK_UNIT * dzUnit;
            }
            float* scale_dz = (float*)post->scale + dz * PACK_UNIT * dzUnit;
            const auto weightBias_dz = post->weightQuanBias + dz * PACK_UNIT * dzUnit;
            auto dst_z = dst + dz * dst_step_tmp * dzUnit;
            const auto src_x   = src;
            auto dst_x         = dst_z;
            __m512i D0 = _mm512_set1_epi32(0);
            __m512i D1 = _mm512_set1_epi32(0);

            __m512i D4 = _mm512_set1_epi32(0);
            __m512i D5 = _mm512_set1_epi32(0);

            __m512i D8 = _mm512_set1_epi32(0);
            __m512i D9 = _mm512_set1_epi32(0);

            __m512i D12 = _mm512_set1_epi32(0);
            __m512i D13 = _mm512_set1_epi32(0);


            for (int sz = 0; sz < src_depth_quad; ++sz) {
                const auto weight_sz = weight_dz + (GEMMINT8_AVX512_L * GEMMINT8_AVX512_H) * sz;
                const auto src_z     = (const float*)(src_x + sz * GEMMINT8_AVX512_E * GEMMINT8_AVX512_L);
                auto w0 = _mm512_loadu_si512(weight_sz);
                auto w1 = _mm512_loadu_si512(weight_sz + 1 * PACK_UNIT * GEMMINT8_AVX512_L);
                auto w2 = _mm512_loadu_si512(weight_sz + 2 * PACK_UNIT * GEMMINT8_AVX512_L);
                auto w3 = _mm512_loadu_si512(weight_sz + 3 * PACK_UNIT * GEMMINT8_AVX512_L);

                auto s0 = AVX512_BROADCAST_INT32(src_z + 0);
                auto s1 = AVX512_BROADCAST_INT32(src_z + 1);

                D0 = _mm512_dpbusds_epi32(D0, s0, w0);
                D1 = _mm512_dpbusds_epi32(D1, s1, w0);

                D4 = _mm512_dpbusds_epi32(D4, s0, w1);
                D5 = _mm512_dpbusds_epi32(D5, s1, w1);

                D8 = _mm512_dpbusds_epi32(D8, s0, w2);
                D9 = _mm512_dpbusds_epi32(D9, s1, w2);

                D12 = _mm512_dpbusds_epi32(D12, s0, w3);
                D13 = _mm512_dpbusds_epi32(D13, s1, w3);
            }

            auto scaleValue = _mm512_loadu_ps(scale_dz);
            auto weightBiasValue = _mm512_loadu_ps(weightBias_dz);
            __m512 xy0_0, xy0_1;

            // x_kernelSum x w_quantZero
            SRCKERNELSUM_MUL_WEIGHTQUANBIAS_2;
            DEQUANT_VALUE(0);
            DEQUANT_VALUE(1);

            if (post->extraScale) { // Batch quant
                f0 = _mm512_mul_ps(f0, extrascale0);
                f1 = _mm512_mul_ps(f1, extrascale1);
                if (post->extraBias && nullptr != biasPtr) {
                    auto extrabias = _mm512_loadu_ps(extraB_dz);
                    extrabias = _mm512_mul_ps(f128, extrabias);
                    auto extrabias0 = _mm512_mul_ps(extrabias, extrascale0);
                    auto extrabias1 = _mm512_mul_ps(extrabias, extrascale1);
                    f0 = _mm512_sub_ps(f0, extrabias0);
                    f1 = _mm512_sub_ps(f1, extrabias1);
                }
            }

            PLUS_TERM_2(0,1);
            if (biasPtr) {
                auto biasValue = _mm512_loadu_ps(bias_dz);
                SCALE_BIAS_VEC(0);
                SCALE_BIAS_VEC(1);
            }

            scaleValue = _mm512_loadu_ps(scale_dz + 1 * PACK_UNIT);
            weightBiasValue = _mm512_loadu_ps(weightBias_dz + 1 * PACK_UNIT);
            // x_kernelSum x w_quantZero
            SRCKERNELSUM_MUL_WEIGHTQUANBIAS_2;
            DEQUANT_VALUE(4);
            DEQUANT_VALUE(5);

            if (post->extraScale) { // Batch quant
                f4 = _mm512_mul_ps(f4, extrascale0);
                f5 = _mm512_mul_ps(f5, extrascale1);
                if (post->extraBias && nullptr != biasPtr) {
                    auto extrabias = _mm512_loadu_ps(extraB_dz + 1 * PACK_UNIT);
                    extrabias = _mm512_mul_ps(f128, extrabias);
                    auto extrabias0 = _mm512_mul_ps(extrabias, extrascale0);
                    auto extrabias1 = _mm512_mul_ps(extrabias, extrascale1);
                    f4 = _mm512_sub_ps(f4, extrabias0);
                    f5 = _mm512_sub_ps(f5, extrabias1);
                }
            }

            PLUS_TERM_2(4,5);
            if (biasPtr) {
                auto biasValue = _mm512_loadu_ps(bias_dz + 1 * PACK_UNIT);
                SCALE_BIAS_VEC(4);
                SCALE_BIAS_VEC(5);
            }

            scaleValue = _mm512_loadu_ps(scale_dz + 2 * PACK_UNIT);
            weightBiasValue = _mm512_loadu_ps(weightBias_dz + 2 * PACK_UNIT);
            SRCKERNELSUM_MUL_WEIGHTQUANBIAS_2;
            DEQUANT_VALUE(8);
            DEQUANT_VALUE(9);

            if (post->extraScale) { // Batch quant
                f8 = _mm512_mul_ps(f8, extrascale0);
                f9 = _mm512_mul_ps(f9, extrascale1);
                if (post->extraBias && nullptr != biasPtr) {
                    auto extrabias = _mm512_loadu_ps(extraB_dz + 2 * PACK_UNIT);
                    extrabias = _mm512_mul_ps(f128, extrabias);
                    auto extrabias0 = _mm512_mul_ps(extrabias, extrascale0);
                    auto extrabias1 = _mm512_mul_ps(extrabias, extrascale1);
                    f8 = _mm512_sub_ps(f8, extrabias0);
                    f9 = _mm512_sub_ps(f9, extrabias1);
                }
            }

            PLUS_TERM_2(8,9);
            if (biasPtr) {
                auto biasValue = _mm512_loadu_ps(bias_dz + 2 * PACK_UNIT);
                SCALE_BIAS_VEC(8);
                SCALE_BIAS_VEC(9);
            }

            scaleValue = _mm512_loadu_ps(scale_dz + 3 * PACK_UNIT);
            weightBiasValue = _mm512_loadu_ps(weightBias_dz + 3 * PACK_UNIT);
            SRCKERNELSUM_MUL_WEIGHTQUANBIAS_2;
            DEQUANT_VALUE(12);
            DEQUANT_VALUE(13);

            if (post->extraScale) { // Batch quant
                f12 = _mm512_mul_ps(f12, extrascale0);
                f13 = _mm512_mul_ps(f13, extrascale1);
                if (post->extraBias && nullptr != biasPtr) {
                    auto extrabias = _mm512_loadu_ps(extraB_dz + 3 * PACK_UNIT);
                    extrabias = _mm512_mul_ps(f128, extrabias);
                    auto extrabias0 = _mm512_mul_ps(extrabias, extrascale0);
                    auto extrabias1 = _mm512_mul_ps(extrabias, extrascale1);
                    f12 = _mm512_sub_ps(f12, extrabias0);
                    f13 = _mm512_sub_ps(f13, extrabias1);
                }
            }

            PLUS_TERM_2(12,13);
            if (biasPtr) {
                auto biasValue = _mm512_loadu_ps(bias_dz + 3 * PACK_UNIT);
                SCALE_BIAS_VEC(12);
                SCALE_BIAS_VEC(13);
            }

            if (post->useInt8 == 0) {
                if (nullptr == biasPtr) {
                    auto dstTmp = dst_x;
                    f0 = _mm512_add_ps(_mm512_loadu_ps(((float*)dstTmp)), f0);
                    f1 = _mm512_add_ps(_mm512_loadu_ps(((float*)dstTmp) + 16), f1);
                    dstTmp += dst_step_tmp;
                    f4 = _mm512_add_ps(_mm512_loadu_ps(((float*)dstTmp) + 16 * 0), f4);
                    f5 = _mm512_add_ps(_mm512_loadu_ps(((float*)dstTmp) + 16 * 1), f5);
                    dstTmp += dst_step_tmp;
                    f8 = _mm512_add_ps(_mm512_loadu_ps(((float*)dstTmp) + 16 * 0), f8);
                    f9 = _mm512_add_ps(_mm512_loadu_ps(((float*)dstTmp) + 16 * 1), f9);
                    dstTmp += dst_step_tmp;
                    f12 = _mm512_add_ps(_mm512_loadu_ps(((float*)dstTmp) + 16 * 0), f12);
                    f13 = _mm512_add_ps(_mm512_loadu_ps(((float*)dstTmp) + 16 * 1), f13);
                }
                if (post->fp32minmax) {
                    POST_TREAT_FLOAT_2(0,1);
                    POST_TREAT_FLOAT_2(4,5);
                    POST_TREAT_FLOAT_2(8,9);
                    POST_TREAT_FLOAT_2(12,13);
                }
                _mm512_storeu_ps(((float*)dst_x), f0);
                _mm512_storeu_ps(((float*)dst_x) + 16, f1);
                dst_x += dst_step_tmp;
                _mm512_storeu_ps(((float*)dst_x) + 16 * 0, f4);
                _mm512_storeu_ps(((float*)dst_x) + 16 * 1, f5);
                dst_x += dst_step_tmp;
                _mm512_storeu_ps(((float*)dst_x) + 16 * 0, f8);
                _mm512_storeu_ps(((float*)dst_x) + 16 * 1, f9);
                dst_x += dst_step_tmp;
                _mm512_storeu_ps(((float*)dst_x) + 16 * 0, f12);
                _mm512_storeu_ps(((float*)dst_x) + 16 * 1, f13);
            } else {
                POSTTREAT(0, 0);
                POSTTREAT(1, 1);
                dst_x += dst_step_tmp;

                POSTTREAT(4, 0);
                POSTTREAT(5, 1);
                dst_x += dst_step_tmp;

                POSTTREAT(8, 0);
                POSTTREAT(9, 1);
                dst_x += dst_step_tmp;

                POSTTREAT(12, 0);
                POSTTREAT(13, 1);
            }
        }
        auto weight_dz = weight + dzU * weightZStride;
        if (biasPtr) {
            bias_dz = post->biasFloat + dzU * PACK_UNIT * dzUnit;
        }
        if (post->extraBias) {
            extraB_dz = post->extraBias + dzU * PACK_UNIT * dzUnit;
        }
        float* scale_dz = (float*)post->scale + dzU * PACK_UNIT * dzUnit;
        const auto weightBias_dz = post->weightQuanBias + dzU * PACK_UNIT * dzUnit;

        auto dst_z = dst + dzU * dst_step_tmp * dzUnit;
        const auto src_x   = src;
        auto dst_x         = dst_z;
        for (int i=0; i<dzR; ++i) {
            __m512i D0 = _mm512_set1_epi32(0);
            __m512i D1 = _mm512_set1_epi32(0);

            for (int sz = 0; sz < src_depth_quad; ++sz) {
                const auto weight_sz = weight_dz + (GEMMINT8_AVX512_L * GEMMINT8_AVX512_H) * sz;
                const auto src_z     = (const float*)(src_x + sz * GEMMINT8_AVX512_E * GEMMINT8_AVX512_L);
                auto w0 = _mm512_loadu_si512(weight_sz);

                auto s0 = AVX512_BROADCAST_INT32(src_z + 0);
                auto s1 = AVX512_BROADCAST_INT32(src_z + 1);

                D0 = _mm512_dpbusds_epi32(D0, s0, w0);
                D1 = _mm512_dpbusds_epi32(D1, s1, w0);
            }

            auto scaleValue = _mm512_loadu_ps(scale_dz);
            auto weightBiasValue = _mm512_loadu_ps(weightBias_dz);
            __m512 xy0_0, xy0_1;
            // x_kernelSum x w_quantZero
            SRCKERNELSUM_MUL_WEIGHTQUANBIAS_2;
            DEQUANT_VALUE(0);
            DEQUANT_VALUE(1);

            if (post->extraScale) { // Batch quant
                f0 = _mm512_mul_ps(f0, extrascale0);
                f1 = _mm512_mul_ps(f1, extrascale1);
                if (post->extraBias && nullptr != biasPtr) {
                    auto extrabias = _mm512_loadu_ps(extraB_dz);
                    extrabias = _mm512_mul_ps(f128, extrabias);
                    auto extrabias0 = _mm512_mul_ps(extrabias, extrascale0);
                    auto extrabias1 = _mm512_mul_ps(extrabias, extrascale1);
                    f0 = _mm512_sub_ps(f0, extrabias0);
                    f1 = _mm512_sub_ps(f1, extrabias1);
                }
            }

            PLUS_TERM_2(0,1);
            if (biasPtr) {
                auto biasValue = _mm512_loadu_ps(bias_dz);
                SCALE_BIAS_VEC(0);
                SCALE_BIAS_VEC(1);
            }

            if (post->useInt8 == 0) {
                if (nullptr == biasPtr) {
                    f0 = _mm512_add_ps(_mm512_loadu_ps(((float*)dst_x)), f0);
                    f1 = _mm512_add_ps(_mm512_loadu_ps(((float*)dst_x) + 16), f1);
                }
                POST_TREAT_FLOAT_2(0,1);
                _mm512_storeu_ps(((float*)dst_x), f0);
                _mm512_storeu_ps(((float*)dst_x) + 16, f1);
            } else {
                POSTTREAT(0, 0);
                POSTTREAT(1, 1);
            }
            dst_x += dst_step_tmp;
            scale_dz += PACK_UNIT;
            if (biasPtr) {
                bias_dz += PACK_UNIT;
            }
            if (post->extraBias) {
                extraB_dz += PACK_UNIT;
            }
            weight_dz += PACK_UNIT * GEMMINT8_AVX512_L;
        }
        return;
    }
    if (realDst == 1) {
        for (int dz = 0; dz < dzU; ++dz) {
            auto weight_dz = weight + dz * weightZStride;
            if (biasPtr) {
                bias_dz = post->biasFloat + dz * PACK_UNIT * dzUnit;
            }
            if (post->extraBias) {
                extraB_dz = post->extraBias + dz * PACK_UNIT * dzUnit;
            }
            float* scale_dz = (float*)post->scale + dz * PACK_UNIT * dzUnit;
            const auto weightBias_dz = post->weightQuanBias + dz * PACK_UNIT * dzUnit;
            auto dst_z = dst + dz * dst_step_tmp * dzUnit;
            const auto src_x   = src;
            auto dst_x         = dst_z;
            __m512i D0 = _mm512_set1_epi32(0);

            __m512i D4 = _mm512_set1_epi32(0);

            __m512i D8 = _mm512_set1_epi32(0);

            __m512i D12 = _mm512_set1_epi32(0);

            for (int sz = 0; sz < src_depth_quad; ++sz) {
                const auto weight_sz = weight_dz + (GEMMINT8_AVX512_L * GEMMINT8_AVX512_H) * sz;
                const auto src_z     = (const float*)(src_x + sz * GEMMINT8_AVX512_E * GEMMINT8_AVX512_L);
                auto w0 = _mm512_loadu_si512(weight_sz);
                auto w1 = _mm512_loadu_si512(weight_sz + 1 * PACK_UNIT * GEMMINT8_AVX512_L);
                auto w2 = _mm512_loadu_si512(weight_sz + 2 * PACK_UNIT * GEMMINT8_AVX512_L);
                auto w3 = _mm512_loadu_si512(weight_sz + 3 * PACK_UNIT * GEMMINT8_AVX512_L);

                auto s0 = AVX512_BROADCAST_INT32(src_z + 0);

                D0 = _mm512_dpbusds_epi32(D0, s0, w0);

                D4 = _mm512_dpbusds_epi32(D4, s0, w1);

                D8 = _mm512_dpbusds_epi32(D8, s0, w2);

                D12 = _mm512_dpbusds_epi32(D12, s0, w3);
            }

            auto scaleValue = _mm512_loadu_ps(scale_dz);
            auto weightBiasValue = _mm512_loadu_ps(weightBias_dz);
            __m512 xy0_0;

            // x_kernelSum x w_quantZero
            SRCKERNELSUM_MUL_WEIGHTQUANBIAS_1;
            DEQUANT_VALUE(0);

            if (post->extraScale) { // Batch quant
                f0 = _mm512_mul_ps(f0, extrascale0);
                if (post->extraBias && nullptr != biasPtr) {
                    auto extrabias = _mm512_loadu_ps(extraB_dz);
                    extrabias = _mm512_mul_ps(f128, extrabias);
                    auto extrabias0 = _mm512_mul_ps(extrabias, extrascale0);
                    f0 = _mm512_sub_ps(f0, extrabias0);
                }
            }

            PLUS_TERM_1(0);
            if (biasPtr) {
                auto biasValue = _mm512_loadu_ps(bias_dz);
                SCALE_BIAS_VEC(0);
            }

            scaleValue = _mm512_loadu_ps(scale_dz + 1 * PACK_UNIT);
            weightBiasValue = _mm512_loadu_ps(weightBias_dz + 1 * PACK_UNIT);
            // x_kernelSum x w_quantZero
            SRCKERNELSUM_MUL_WEIGHTQUANBIAS_1;
            DEQUANT_VALUE(4);

            if (post->extraScale) { // Batch quant
                f4 = _mm512_mul_ps(f4, extrascale0);
                if (post->extraBias && nullptr != biasPtr) {
                    auto extrabias = _mm512_loadu_ps(extraB_dz + 1 * PACK_UNIT);
                    extrabias = _mm512_mul_ps(f128, extrabias);
                    auto extrabias0 = _mm512_mul_ps(extrabias, extrascale0);
                    f4 = _mm512_sub_ps(f4, extrabias0);
                }
            }

            PLUS_TERM_1(4);
            if (biasPtr) {
                auto biasValue = _mm512_loadu_ps(bias_dz + 1 * PACK_UNIT);
                SCALE_BIAS_VEC(4);
            }

            scaleValue = _mm512_loadu_ps(scale_dz + 2 * PACK_UNIT);
            weightBiasValue = _mm512_loadu_ps(weightBias_dz + 2 * PACK_UNIT);
            // x_kernelSum x w_quantZero
            SRCKERNELSUM_MUL_WEIGHTQUANBIAS_1;
            DEQUANT_VALUE(8);

            if (post->extraScale) { // Batch quant
                f8 = _mm512_mul_ps(f8, extrascale0);
                if (post->extraBias && nullptr != biasPtr) {
                    auto extrabias = _mm512_loadu_ps(extraB_dz + 2 * PACK_UNIT);
                    extrabias = _mm512_mul_ps(f128, extrabias);
                    auto extrabias0 = _mm512_mul_ps(extrabias, extrascale0);
                    f8 = _mm512_sub_ps(f8, extrabias0);
                }
            }

            PLUS_TERM_1(8);
            if (biasPtr) {
                auto biasValue = _mm512_loadu_ps(bias_dz + 2 * PACK_UNIT);
                SCALE_BIAS_VEC(8);
            }

            scaleValue = _mm512_loadu_ps(scale_dz + 3 * PACK_UNIT);
            weightBiasValue = _mm512_loadu_ps(weightBias_dz + 3 * PACK_UNIT);
            // x_kernelSum x w_quantZero
            SRCKERNELSUM_MUL_WEIGHTQUANBIAS_1;
            DEQUANT_VALUE(12);

            if (post->extraScale) { // Batch quant
                f12 = _mm512_mul_ps(f12, extrascale0);
                if (post->extraBias && nullptr != biasPtr) {
                    auto extrabias = _mm512_loadu_ps(extraB_dz + 3 * PACK_UNIT);
                    extrabias = _mm512_mul_ps(f128, extrabias);
                    auto extrabias0 = _mm512_mul_ps(extrabias, extrascale0);
                    f12 = _mm512_sub_ps(f12, extrabias0);
                }
            }

            PLUS_TERM_1(12);
            if (biasPtr) {
                auto biasValue = _mm512_loadu_ps(bias_dz + 3 * PACK_UNIT);
                SCALE_BIAS_VEC(12);
            }

            if (post->useInt8 == 0) {
                if (nullptr == biasPtr) {
                    auto dstTemp = dst_x;
                    f0 = _mm512_add_ps(_mm512_loadu_ps(((float*)dstTemp)), f0);
                    dstTemp += dst_step_tmp;
                    f4 = _mm512_add_ps(_mm512_loadu_ps(((float*)dstTemp) + 16 * 0), f4);
                    dstTemp += dst_step_tmp;
                    f8 = _mm512_add_ps(_mm512_loadu_ps(((float*)dstTemp) + 16 * 0), f8);
                    dstTemp += dst_step_tmp;
                    f12 = _mm512_add_ps(_mm512_loadu_ps(((float*)dstTemp) + 16 * 0), f12);
                }
                if (post->fp32minmax) {
                    POST_TREAT_FLOAT_1(0);
                    POST_TREAT_FLOAT_1(4);
                    POST_TREAT_FLOAT_1(8);
                    POST_TREAT_FLOAT_1(12);
                }
                _mm512_storeu_ps(((float*)dst_x), f0);
                dst_x += dst_step_tmp;
                _mm512_storeu_ps(((float*)dst_x) + 16 * 0, f4);
                dst_x += dst_step_tmp;
                _mm512_storeu_ps(((float*)dst_x) + 16 * 0, f8);
                dst_x += dst_step_tmp;
                _mm512_storeu_ps(((float*)dst_x) + 16 * 0, f12);
            } else {
                POSTTREAT(0, 0);
                dst_x += dst_step_tmp;

                POSTTREAT(4, 0);
                dst_x += dst_step_tmp;

                POSTTREAT(8, 0);
                dst_x += dst_step_tmp;

                POSTTREAT(12, 0);
            }
        }
        auto weight_dz = weight + dzU * weightZStride;
        if (biasPtr) {
            bias_dz = post->biasFloat + dzU * PACK_UNIT * dzUnit;
        }
        if (post->extraBias) {
            extraB_dz = post->extraBias + dzU * PACK_UNIT * dzUnit;
        }
        float* scale_dz = (float*)post->scale + dzU * PACK_UNIT * dzUnit;
        const auto weightBias_dz = post->weightQuanBias + dzU * PACK_UNIT * dzUnit;

        auto dst_z = dst + dzU * dst_step_tmp * dzUnit;
        const auto src_x   = src;
        auto dst_x         = dst_z;
        for (int i=0; i<dzR; ++i) {
            __m512i D0 = _mm512_set1_epi32(0);

            for (int sz = 0; sz < src_depth_quad; ++sz) {
                const auto weight_sz = weight_dz + (GEMMINT8_AVX512_L * GEMMINT8_AVX512_H) * sz;
                const auto src_z     = (const float*)(src_x + sz * GEMMINT8_AVX512_E * GEMMINT8_AVX512_L);
                auto w0 = _mm512_loadu_si512(weight_sz);

                auto s0 = AVX512_BROADCAST_INT32(src_z + 0);

                D0 = _mm512_dpbusds_epi32(D0, s0, w0);
            }

            auto scaleValue = _mm512_loadu_ps(scale_dz);

            auto weightBiasValue = _mm512_loadu_ps(weightBias_dz);
            __m512 xy0_0;
            // x_kernelSum x w_quantZero
            SRCKERNELSUM_MUL_WEIGHTQUANBIAS_1;
            DEQUANT_VALUE(0);

            if (post->extraScale) { // Batch quant
                f0 = _mm512_mul_ps(f0, extrascale0);
                if (post->extraBias && nullptr != biasPtr) {
                    auto extrabias = _mm512_loadu_ps(extraB_dz);
                    extrabias = _mm512_mul_ps(f128, extrabias);
                    auto extrabias0 = _mm512_mul_ps(extrabias, extrascale0);
                    f0 = _mm512_sub_ps(f0, extrabias0);
                }
            }

            PLUS_TERM_1(0);
            if (biasPtr) {
                auto biasValue = _mm512_loadu_ps(bias_dz);
                SCALE_BIAS_VEC(0);
            }

            if (post->useInt8 == 0) {
                if (nullptr == biasPtr) {
                    f0 = _mm512_add_ps(_mm512_loadu_ps(((float*)dst_x)), f0);
                }
                if (post->fp32minmax) {
                    POST_TREAT_FLOAT_1(0);
                }
                _mm512_storeu_ps(((float*)dst_x), f0);
            } else {
                POSTTREAT(0, 0);
            }
            dst_x += dst_step_tmp;
            scale_dz += PACK_UNIT;
            if (biasPtr) {
                bias_dz += PACK_UNIT;
            }
            if (post->extraBias) {
                extraB_dz += PACK_UNIT;
            }
            weight_dz += PACK_UNIT * GEMMINT8_AVX512_L;
        }
        return;
    }
}

// GemmInt8 with VNNI int4-weight fp32-output
void _AVX512_MNNGemmInt8AddBiasScale_16x4_w4_Unit_VNNI(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad, const QuanPostTreatParameters* post, size_t realDst) {
    MNN_ASSERT(post->useInt8 == 0);
    const auto dst_step_tmp = dst_step / sizeof(int8_t);
    auto zero512 = _mm512_set1_ps(0.0f);
    int dzUnit = GEMMINT8_AVX512_H / PACK_UNIT;
    int dzU = dst_depth_quad / dzUnit;
    int dzR = dst_depth_quad % dzUnit;
    const __m512i mask = _mm512_set1_epi8(0xf);
    __m512 fp32min, fp32max;
    if (post->fp32minmax) {
        fp32min = _mm512_set1_ps((post->fp32minmax)[0]);
        fp32max = _mm512_set1_ps((post->fp32minmax)[1]);
    }
    auto blockNum = post->blockNum;
    const float* biasPtr = nullptr;
    const float* bias_dz = nullptr;
    const float* extraB_dz = nullptr;
    if (post->biasFloat) {
        biasPtr = post->biasFloat;
    }
    
    auto srcKernelSumPtr = post->srcKernelSum;
    __m512 kernelSum0 = _mm512_setzero_ps();
    __m512 kernelSum1 = _mm512_setzero_ps();
    __m512 kernelSum2 = _mm512_setzero_ps();
    __m512 kernelSum3 = _mm512_setzero_ps();
    if (GEMMINT8_AVX512_E == realDst) {
        kernelSum0 = _mm512_set1_ps(post->srcKernelSum[0]);
        kernelSum1 = _mm512_set1_ps(post->srcKernelSum[1]);
        kernelSum2 = _mm512_set1_ps(post->srcKernelSum[2]);
        kernelSum3 = _mm512_set1_ps(post->srcKernelSum[3]);
    } else {
        kernelSum0 = _mm512_set1_ps(post->srcKernelSum[0]);
        if (realDst > 1) {
            kernelSum1 = _mm512_set1_ps(post->srcKernelSum[1]);
        }
        if (realDst > 2) {
            kernelSum2 = _mm512_set1_ps(post->srcKernelSum[2]);
        }
    }
    auto f128   = _mm512_set1_ps(128.f);
    __m512 extrascale0 = _mm512_setzero_ps();
    __m512 extrascale1 = _mm512_setzero_ps();
    __m512 extrascale2 = _mm512_setzero_ps();
    __m512 extrascale3 = _mm512_setzero_ps();
    if (post->extraScale) {
        if (GEMMINT8_AVX512_E == realDst) {
            extrascale0 = _mm512_set1_ps(post->extraScale[0]);
            extrascale1 = _mm512_set1_ps(post->extraScale[1]);
            extrascale2 = _mm512_set1_ps(post->extraScale[2]);
            extrascale3 = _mm512_set1_ps(post->extraScale[3]);
        } else {
            extrascale0 = _mm512_set1_ps(post->extraScale[0]);
            if (realDst > 1) {
                extrascale1 = _mm512_set1_ps(post->extraScale[1]);
            }
            if (realDst > 2) {
                extrascale2 = _mm512_set1_ps(post->extraScale[2]);
            }
        }
    }
    int weight_step_Z = static_cast<int32_t>(blockNum * src_depth_quad * (GEMMINT8_AVX512_L * GEMMINT8_AVX512_H) / 2); // sizeof(int4_t)
    int weight_step_Y = static_cast<int32_t>(GEMMINT8_AVX512_L * GEMMINT8_AVX512_H / 2); // sizeof(int4_t)

    if (realDst == GEMMINT8_AVX512_E) {
        for (int dz = 0; dz < dzU; ++dz) {
            auto weight_dz = weight + dz * weight_step_Z;
            if (post->biasFloat) {
                bias_dz = biasPtr + dz * PACK_UNIT * dzUnit;
            }
            if (post->extraBias) {
                extraB_dz = post->extraBias + dz * PACK_UNIT * dzUnit;
            }
            const auto weightBias_dz = post->weightQuanBias + dz * PACK_UNIT * dzUnit;
            float* scale_dz = (float*)post->scale + dz * PACK_UNIT * dzUnit;
            auto dst_z = dst + dz * dst_step_tmp * dzUnit;
            const auto src_x   = src;
            auto dst_x         = dst_z;
            __m512i D0 = _mm512_set1_epi32(0);
            __m512i D1 = _mm512_set1_epi32(0);
            __m512i D2 = _mm512_set1_epi32(0);
            __m512i D3 = _mm512_set1_epi32(0);

            __m512i D4 = _mm512_set1_epi32(0);
            __m512i D5 = _mm512_set1_epi32(0);
            __m512i D6 = _mm512_set1_epi32(0);
            __m512i D7 = _mm512_set1_epi32(0);

            __m512i D8 = _mm512_set1_epi32(0);
            __m512i D9 = _mm512_set1_epi32(0);
            __m512i D10 = _mm512_set1_epi32(0);
            __m512i D11 = _mm512_set1_epi32(0);

            __m512i D12 = _mm512_set1_epi32(0);
            __m512i D13 = _mm512_set1_epi32(0);
            __m512i D14 = _mm512_set1_epi32(0);
            __m512i D15 = _mm512_set1_epi32(0);


            for (int sz = 0; sz < src_depth_quad; ++sz) {
                const auto weight_sz = weight_dz + weight_step_Y * sz;
                const auto src_z     = (const float*)(src_x + sz * GEMMINT8_AVX512_E * GEMMINT8_AVX512_L);

                // int4->int8: total count=4*64(GEMMINT8_AVX512_L * GEMMINT8_AVX512_H)
                // Load 4*64 int4 weight
                auto w0_int4_64 = _mm512_loadu_si512(weight_sz); // 128xint4_t=64 byte
                auto w1_int4_64 = _mm512_loadu_si512(weight_sz + 64); // 128xint4_t
                // 256xint4_t->256xint8_t
                auto w0 = _mm512_and_si512(mask, _mm512_srli_epi16(w0_int4_64, 4)); // 64xint8_t
                auto w2 = _mm512_and_si512(mask, w0_int4_64); // 64xint8_t
                auto w1 = _mm512_and_si512(mask, _mm512_srli_epi16(w1_int4_64, 4));
                auto w3 = _mm512_and_si512(mask, w1_int4_64);

                auto s0 = AVX512_BROADCAST_INT32(src_z + 0);
                auto s1 = AVX512_BROADCAST_INT32(src_z + 1);
                auto s2 = AVX512_BROADCAST_INT32(src_z + 2);
                auto s3 = AVX512_BROADCAST_INT32(src_z + 3);

                D0 = _mm512_dpbusds_epi32(D0, s0, w0);
                D1 = _mm512_dpbusds_epi32(D1, s1, w0);
                D2 = _mm512_dpbusds_epi32(D2, s2, w0);
                D3 = _mm512_dpbusds_epi32(D3, s3, w0);

                D4 = _mm512_dpbusds_epi32(D4, s0, w1);
                D5 = _mm512_dpbusds_epi32(D5, s1, w1);
                D6 = _mm512_dpbusds_epi32(D6, s2, w1);
                D7 = _mm512_dpbusds_epi32(D7, s3, w1);

                D8 = _mm512_dpbusds_epi32(D8, s0, w2);
                D9 = _mm512_dpbusds_epi32(D9, s1, w2);
                D10 = _mm512_dpbusds_epi32(D10, s2, w2);
                D11 = _mm512_dpbusds_epi32(D11, s3, w2);

                D12 = _mm512_dpbusds_epi32(D12, s0, w3);
                D13 = _mm512_dpbusds_epi32(D13, s1, w3);
                D14 = _mm512_dpbusds_epi32(D14, s2, w3);
                D15 = _mm512_dpbusds_epi32(D15, s3, w3);
            }
            auto scaleValue = _mm512_loadu_ps(scale_dz);
            auto weightBiasValue = _mm512_loadu_ps(weightBias_dz);
            __m512 xy0_0, xy0_1, xy0_2, xy0_3;
            // x_kernelSum x w_quantZero
            SRCKERNELSUM_MUL_WEIGHTQUANBIAS;
            DEQUANT_VALUE(0);
            DEQUANT_VALUE(1);
            DEQUANT_VALUE(2);
            DEQUANT_VALUE(3);

            if (post->extraScale) { // Batch quant
                f0 = _mm512_mul_ps(f0, extrascale0);
                f1 = _mm512_mul_ps(f1, extrascale1);
                f2 = _mm512_mul_ps(f2, extrascale2);
                f3 = _mm512_mul_ps(f3, extrascale3);
                if (post->extraBias && nullptr != biasPtr) {
                    auto extrabias = _mm512_loadu_ps(extraB_dz);
                    extrabias = _mm512_mul_ps(f128, extrabias);
                    auto extrabias0 = _mm512_mul_ps(extrabias, extrascale0);
                    auto extrabias1 = _mm512_mul_ps(extrabias, extrascale1);
                    auto extrabias2 = _mm512_mul_ps(extrabias, extrascale2);
                    auto extrabias3 = _mm512_mul_ps(extrabias, extrascale3);
                    f0 = _mm512_sub_ps(f0, extrabias0);
                    f1 = _mm512_sub_ps(f1, extrabias1);
                    f2 = _mm512_sub_ps(f2, extrabias2);
                    f3 = _mm512_sub_ps(f3, extrabias3);
                }
            }

            PLUS_TERM(0,1,2,3);
            if (nullptr != biasPtr) {
                auto biasValue = _mm512_loadu_ps(bias_dz);
                SCALE_BIAS_VEC(0);
                SCALE_BIAS_VEC(1);
                SCALE_BIAS_VEC(2);
                SCALE_BIAS_VEC(3);
            }

            scaleValue = _mm512_loadu_ps(scale_dz + 1 * PACK_UNIT);
            weightBiasValue = _mm512_loadu_ps(weightBias_dz + 1 * PACK_UNIT);
            // x_kernelSum x w_quantZero
            SRCKERNELSUM_MUL_WEIGHTQUANBIAS;
            DEQUANT_VALUE(4);
            DEQUANT_VALUE(5);
            DEQUANT_VALUE(6);
            DEQUANT_VALUE(7);

            if (post->extraScale) { // Batch quant
                f4 = _mm512_mul_ps(f4, extrascale0);
                f5 = _mm512_mul_ps(f5, extrascale1);
                f6 = _mm512_mul_ps(f6, extrascale2);
                f7 = _mm512_mul_ps(f7, extrascale3);
                if (post->extraBias && nullptr != biasPtr) {
                    auto extrabias = _mm512_loadu_ps(extraB_dz + 1 * PACK_UNIT);
                    extrabias = _mm512_mul_ps(f128, extrabias);
                    auto extrabias0 = _mm512_mul_ps(extrabias, extrascale0);
                    auto extrabias1 = _mm512_mul_ps(extrabias, extrascale1);
                    auto extrabias2 = _mm512_mul_ps(extrabias, extrascale2);
                    auto extrabias3 = _mm512_mul_ps(extrabias, extrascale3);
                    f4 = _mm512_sub_ps(f4, extrabias0);
                    f5 = _mm512_sub_ps(f5, extrabias1);
                    f6 = _mm512_sub_ps(f6, extrabias2);
                    f7 = _mm512_sub_ps(f7, extrabias3);
                }
            }

            PLUS_TERM(4,5,6,7);
            if (nullptr != biasPtr) {
                auto biasValue = _mm512_loadu_ps(bias_dz + 1 * PACK_UNIT);
                SCALE_BIAS_VEC(4);
                SCALE_BIAS_VEC(5);
                SCALE_BIAS_VEC(6);
                SCALE_BIAS_VEC(7);
            }

            scaleValue = _mm512_loadu_ps(scale_dz + 2 * PACK_UNIT);
            weightBiasValue = _mm512_loadu_ps(weightBias_dz + 2 * PACK_UNIT);
            // x_kernelSum x w_quantZero
            SRCKERNELSUM_MUL_WEIGHTQUANBIAS;
            DEQUANT_VALUE(8);
            DEQUANT_VALUE(9);
            DEQUANT_VALUE(10);
            DEQUANT_VALUE(11);

            if (post->extraScale) { // Batch quant
                f8 = _mm512_mul_ps(f8, extrascale0);
                f9 = _mm512_mul_ps(f9, extrascale1);
                f10 = _mm512_mul_ps(f10, extrascale2);
                f11 = _mm512_mul_ps(f11, extrascale3);
                if (post->extraBias && nullptr != biasPtr) {
                    auto extrabias = _mm512_loadu_ps(extraB_dz + 2 * PACK_UNIT);
                    extrabias = _mm512_mul_ps(f128, extrabias);
                    auto extrabias0 = _mm512_mul_ps(extrabias, extrascale0);
                    auto extrabias1 = _mm512_mul_ps(extrabias, extrascale1);
                    auto extrabias2 = _mm512_mul_ps(extrabias, extrascale2);
                    auto extrabias3 = _mm512_mul_ps(extrabias, extrascale3);
                    f8 = _mm512_sub_ps(f8, extrabias0);
                    f9 = _mm512_sub_ps(f9, extrabias1);
                    f10 = _mm512_sub_ps(f10, extrabias2);
                    f11 = _mm512_sub_ps(f11, extrabias3);
                }
            }

            PLUS_TERM(8,9,10,11);
            if (nullptr != biasPtr) {
                auto biasValue = _mm512_loadu_ps(bias_dz + 2 * PACK_UNIT);
                SCALE_BIAS_VEC(8);
                SCALE_BIAS_VEC(9);
                SCALE_BIAS_VEC(10);
                SCALE_BIAS_VEC(11);
            }

            scaleValue = _mm512_loadu_ps(scale_dz + 3 * PACK_UNIT);
            weightBiasValue = _mm512_loadu_ps(weightBias_dz + 3 * PACK_UNIT);
            // x_kernelSum x w_quantZero
            SRCKERNELSUM_MUL_WEIGHTQUANBIAS;
            DEQUANT_VALUE(12);
            DEQUANT_VALUE(13);
            DEQUANT_VALUE(14);
            DEQUANT_VALUE(15);

            if (post->extraScale) { // Batch quant
                f12 = _mm512_mul_ps(f12, extrascale0);
                f13 = _mm512_mul_ps(f13, extrascale1);
                f14 = _mm512_mul_ps(f14, extrascale2);
                f15 = _mm512_mul_ps(f15, extrascale3);
                if (post->extraBias && nullptr != biasPtr) {
                    auto extrabias = _mm512_loadu_ps(extraB_dz + 3 * PACK_UNIT);
                    extrabias = _mm512_mul_ps(f128, extrabias);
                    auto extrabias0 = _mm512_mul_ps(extrabias, extrascale0);
                    auto extrabias1 = _mm512_mul_ps(extrabias, extrascale1);
                    auto extrabias2 = _mm512_mul_ps(extrabias, extrascale2);
                    auto extrabias3 = _mm512_mul_ps(extrabias, extrascale3);
                    f12 = _mm512_sub_ps(f12, extrabias0);
                    f13 = _mm512_sub_ps(f13, extrabias1);
                    f14 = _mm512_sub_ps(f14, extrabias2);
                    f15 = _mm512_sub_ps(f15, extrabias3);
                }
            }

            PLUS_TERM(12,13,14,15);
            if (nullptr != biasPtr) {
                auto biasValue = _mm512_loadu_ps(bias_dz + 3 * PACK_UNIT);
                SCALE_BIAS_VEC(12);
                SCALE_BIAS_VEC(13);
                SCALE_BIAS_VEC(14);
                SCALE_BIAS_VEC(15);
            }
            if (biasPtr == nullptr) {
                    auto destTmp = dst_x;
                    f0 = _mm512_add_ps(_mm512_loadu_ps((float*)destTmp), f0);
                    f1 = _mm512_add_ps(_mm512_loadu_ps(((float*)destTmp) + 16), f1);
                    f2 = _mm512_add_ps(_mm512_loadu_ps(((float*)destTmp) + 16 * 2), f2);
                    f3 = _mm512_add_ps(_mm512_loadu_ps(((float*)destTmp) + 16 * 3), f3);
                    destTmp += dst_step_tmp;
                    f4 = _mm512_add_ps(_mm512_loadu_ps(((float*)destTmp) + 16 * 0), f4);
                    f5 = _mm512_add_ps(_mm512_loadu_ps(((float*)destTmp) + 16 * 1), f5);
                    f6 = _mm512_add_ps(_mm512_loadu_ps(((float*)destTmp) + 16 * 2), f6);
                    f7 = _mm512_add_ps(_mm512_loadu_ps(((float*)destTmp) + 16 * 3), f7);
                    destTmp += dst_step_tmp;
                    f8 = _mm512_add_ps(_mm512_loadu_ps(((float*)destTmp) + 16 * 0), f8);
                    f9 = _mm512_add_ps(_mm512_loadu_ps(((float*)destTmp) + 16 * 1), f9);
                    f10 = _mm512_add_ps(_mm512_loadu_ps(((float*)destTmp) + 16 * 2), f10);
                    f11 = _mm512_add_ps(_mm512_loadu_ps(((float*)destTmp) + 16 * 3), f11);
                    destTmp += dst_step_tmp;
                    f12 = _mm512_add_ps(_mm512_loadu_ps(((float*)destTmp) + 16 * 0), f12);
                    f13 = _mm512_add_ps(_mm512_loadu_ps(((float*)destTmp) + 16 * 1), f13);
                    f14 = _mm512_add_ps(_mm512_loadu_ps(((float*)destTmp) + 16 * 2), f14);
                    f15 = _mm512_add_ps(_mm512_loadu_ps(((float*)destTmp) + 16 * 3), f15);
                }
            if (post->fp32minmax) {
                POST_TREAT_FLOAT(0,1,2,3);
                POST_TREAT_FLOAT(4,5,6,7);
                POST_TREAT_FLOAT(8,9,10,11);
                POST_TREAT_FLOAT(12,13,14,15);
            }

            _mm512_storeu_ps(((float*)dst_x), f0);
            _mm512_storeu_ps(((float*)dst_x) + 16, f1);
            _mm512_storeu_ps(((float*)dst_x) + 16 * 2, f2);
            _mm512_storeu_ps(((float*)dst_x) + 16 * 3, f3);
            dst_x += dst_step_tmp;
            _mm512_storeu_ps(((float*)dst_x) + 16 * 0, f4);
            _mm512_storeu_ps(((float*)dst_x) + 16 * 1, f5);
            _mm512_storeu_ps(((float*)dst_x) + 16 * 2, f6);
            _mm512_storeu_ps(((float*)dst_x) + 16 * 3, f7);
            dst_x += dst_step_tmp;
            _mm512_storeu_ps(((float*)dst_x) + 16 * 0, f8);
            _mm512_storeu_ps(((float*)dst_x) + 16 * 1, f9);
            _mm512_storeu_ps(((float*)dst_x) + 16 * 2, f10);
            _mm512_storeu_ps(((float*)dst_x) + 16 * 3, f11);
            dst_x += dst_step_tmp;
            _mm512_storeu_ps(((float*)dst_x) + 16 * 0, f12);
            _mm512_storeu_ps(((float*)dst_x) + 16 * 1, f13);
            _mm512_storeu_ps(((float*)dst_x) + 16 * 2, f14);
            _mm512_storeu_ps(((float*)dst_x) + 16 * 3, f15);
            
        }
        auto weight_dz = weight + dzU * weight_step_Z;
        if (biasPtr) {
            bias_dz = biasPtr + dzU * PACK_UNIT * dzUnit;
        }
        if (post->extraBias) {
            extraB_dz = post->extraBias + dzU * PACK_UNIT * dzUnit;
        }
        float* scale_dz = (float*)post->scale + dzU * PACK_UNIT * dzUnit;
        const auto weightBias_dz = post->weightQuanBias + dzU * PACK_UNIT * dzUnit;

        auto dst_z = dst + dzU * dst_step_tmp * dzUnit;
        const auto src_x   = src;
        auto dst_x         = dst_z;
        for (int i=0; i<dzR; ++i) {
            __m512i D0 = _mm512_set1_epi32(0);
            __m512i D1 = _mm512_set1_epi32(0);
            __m512i D2 = _mm512_set1_epi32(0);
            __m512i D3 = _mm512_set1_epi32(0);

            for (int sz = 0; sz < src_depth_quad; ++sz) {
                const auto weight_sz = weight_dz + weight_step_Y * sz;
                const auto src_z     = (const float*)(src_x + sz * GEMMINT8_AVX512_E * GEMMINT8_AVX512_L);
                auto w0_int4_64 = _mm512_loadu_si512(weight_sz); // 128xint4_t=64 byte
                // 256xint4_t->256xint8_t
                auto w0 = _mm512_and_si512(mask, _mm512_srli_epi16(w0_int4_64, 4)); // 64xint8_t

                auto s0 = AVX512_BROADCAST_INT32(src_z + 0);
                auto s1 = AVX512_BROADCAST_INT32(src_z + 1);
                auto s2 = AVX512_BROADCAST_INT32(src_z + 2);
                auto s3 = AVX512_BROADCAST_INT32(src_z + 3);

                D0 = _mm512_dpbusds_epi32(D0, s0, w0);
                D1 = _mm512_dpbusds_epi32(D1, s1, w0);
                D2 = _mm512_dpbusds_epi32(D2, s2, w0);
                D3 = _mm512_dpbusds_epi32(D3, s3, w0);
            }

            auto scaleValue = _mm512_loadu_ps(scale_dz);
            auto weightBiasValue = _mm512_loadu_ps(weightBias_dz);
            __m512 xy0_0, xy0_1, xy0_2, xy0_3;
            // x_kernelSum x w_quantZero
            SRCKERNELSUM_MUL_WEIGHTQUANBIAS;
            DEQUANT_VALUE(0);
            DEQUANT_VALUE(1);
            DEQUANT_VALUE(2);
            DEQUANT_VALUE(3);

            if (post->extraScale) { // Batch quant
                f0 = _mm512_mul_ps(f0, extrascale0);
                f1 = _mm512_mul_ps(f1, extrascale1);
                f2 = _mm512_mul_ps(f2, extrascale2);
                f3 = _mm512_mul_ps(f3, extrascale3);
                if (post->extraBias && nullptr != biasPtr) {
                    auto extrabias = _mm512_loadu_ps(extraB_dz);
                    extrabias = _mm512_mul_ps(f128, extrabias);
                    auto extrabias0 = _mm512_mul_ps(extrabias, extrascale0);
                    auto extrabias1 = _mm512_mul_ps(extrabias, extrascale1);
                    auto extrabias2 = _mm512_mul_ps(extrabias, extrascale2);
                    auto extrabias3 = _mm512_mul_ps(extrabias, extrascale3);
                    f0 = _mm512_sub_ps(f0, extrabias0);
                    f1 = _mm512_sub_ps(f1, extrabias1);
                    f2 = _mm512_sub_ps(f2, extrabias2);
                    f3 = _mm512_sub_ps(f3, extrabias3);
                }
            }

            PLUS_TERM(0,1,2,3);
            if (nullptr != biasPtr) {
                auto biasValue = _mm512_loadu_ps(bias_dz);
                SCALE_BIAS_VEC(0);
                SCALE_BIAS_VEC(1);
                SCALE_BIAS_VEC(2);
                SCALE_BIAS_VEC(3);
            }

            if (nullptr == biasPtr) {
                f0 = _mm512_add_ps(_mm512_loadu_ps((float*)dst_x), f0);
                f1 = _mm512_add_ps(_mm512_loadu_ps(((float*)dst_x) + 16), f1);
                f2 = _mm512_add_ps(_mm512_loadu_ps(((float*)dst_x) + 16 * 2), f2);
                f3 = _mm512_add_ps(_mm512_loadu_ps(((float*)dst_x) + 16 * 3), f3);
            }
            if (post->fp32minmax) {
                POST_TREAT_FLOAT(0,1,2,3);
            }
            _mm512_storeu_ps(((float*)dst_x), f0);
            _mm512_storeu_ps(((float*)dst_x) + 16, f1);
            _mm512_storeu_ps(((float*)dst_x) + 16 * 2, f2);
            _mm512_storeu_ps(((float*)dst_x) + 16 * 3, f3);
            
            dst_x += dst_step_tmp;
            scale_dz += PACK_UNIT;
            if (biasPtr) {
                bias_dz += PACK_UNIT;
            }
            if (post->extraBias) {
                extraB_dz += PACK_UNIT;
            }
            weight_dz += PACK_UNIT * GEMMINT8_AVX512_L;
        }
        return;
    }
    // e = 3
    if (realDst == 3) {
        for (int dz = 0; dz < dzU; ++dz) {
            auto weight_dz = weight + dz * weight_step_Z;
            if (biasPtr) {
                bias_dz = biasPtr + dz * PACK_UNIT * dzUnit;
            }
            if (post->extraBias) {
                extraB_dz = post->extraBias + dz * PACK_UNIT * dzUnit;
            }
            float* scale_dz = (float*)post->scale + dz * PACK_UNIT * dzUnit;
            const auto weightBias_dz = post->weightQuanBias + dz * PACK_UNIT * dzUnit;
            auto dst_z = dst + dz * dst_step_tmp * dzUnit;
            const auto src_x   = src;
            auto dst_x         = dst_z;
            __m512i D0 = _mm512_set1_epi32(0);
            __m512i D1 = _mm512_set1_epi32(0);
            __m512i D2 = _mm512_set1_epi32(0);

            __m512i D4 = _mm512_set1_epi32(0);
            __m512i D5 = _mm512_set1_epi32(0);
            __m512i D6 = _mm512_set1_epi32(0);

            __m512i D8 = _mm512_set1_epi32(0);
            __m512i D9 = _mm512_set1_epi32(0);
            __m512i D10 = _mm512_set1_epi32(0);

            __m512i D12 = _mm512_set1_epi32(0);
            __m512i D13 = _mm512_set1_epi32(0);
            __m512i D14 = _mm512_set1_epi32(0);


            for (int sz = 0; sz < src_depth_quad; ++sz) {
                const auto weight_sz = weight_dz + weight_step_Y * sz;
                const auto src_z     = (const float*)(src_x + sz * GEMMINT8_AVX512_E * GEMMINT8_AVX512_L);
                // int4->int8: total count=4*64(GEMMINT8_AVX512_L * GEMMINT8_AVX512_H)
                // Load 4*64 int4 weight
                auto w0_int4_64 = _mm512_loadu_si512(weight_sz); // 128xint4_t=64 byte
                auto w1_int4_64 = _mm512_loadu_si512(weight_sz + 64); // 128xint4_t
                // 256xint4_t->256xint8_t
                auto w0 = _mm512_and_si512(mask, _mm512_srli_epi16(w0_int4_64, 4)); // 64xint8_t
                auto w2 = _mm512_and_si512(mask, w0_int4_64); // 64xint8_t
                auto w1 = _mm512_and_si512(mask, _mm512_srli_epi16(w1_int4_64, 4));
                auto w3 = _mm512_and_si512(mask, w1_int4_64);

                auto s0 = AVX512_BROADCAST_INT32(src_z + 0);
                auto s1 = AVX512_BROADCAST_INT32(src_z + 1);
                auto s2 = AVX512_BROADCAST_INT32(src_z + 2);

                D0 = _mm512_dpbusds_epi32(D0, s0, w0);
                D1 = _mm512_dpbusds_epi32(D1, s1, w0);
                D2 = _mm512_dpbusds_epi32(D2, s2, w0);

                D4 = _mm512_dpbusds_epi32(D4, s0, w1);
                D5 = _mm512_dpbusds_epi32(D5, s1, w1);
                D6 = _mm512_dpbusds_epi32(D6, s2, w1);

                D8 = _mm512_dpbusds_epi32(D8, s0, w2);
                D9 = _mm512_dpbusds_epi32(D9, s1, w2);
                D10 = _mm512_dpbusds_epi32(D10, s2, w2);

                D12 = _mm512_dpbusds_epi32(D12, s0, w3);
                D13 = _mm512_dpbusds_epi32(D13, s1, w3);
                D14 = _mm512_dpbusds_epi32(D14, s2, w3);
            }

            auto scaleValue = _mm512_loadu_ps(scale_dz);
            auto weightBiasValue = _mm512_loadu_ps(weightBias_dz);
            __m512 xy0_0, xy0_1, xy0_2;
            // x_kernelSum x w_quantZero
            SRCKERNELSUM_MUL_WEIGHTQUANBIAS_3;
            DEQUANT_VALUE(0);
            DEQUANT_VALUE(1);
            DEQUANT_VALUE(2);

            if (post->extraScale) { // Batch quant
                f0 = _mm512_mul_ps(f0, extrascale0);
                f1 = _mm512_mul_ps(f1, extrascale1);
                f2 = _mm512_mul_ps(f2, extrascale2);
                if (post->extraBias && nullptr != biasPtr) {
                    auto extrabias = _mm512_loadu_ps(extraB_dz);
                    extrabias = _mm512_mul_ps(f128, extrabias);
                    auto extrabias0 = _mm512_mul_ps(extrabias, extrascale0);
                    auto extrabias1 = _mm512_mul_ps(extrabias, extrascale1);
                    auto extrabias2 = _mm512_mul_ps(extrabias, extrascale2);
                    f0 = _mm512_sub_ps(f0, extrabias0);
                    f1 = _mm512_sub_ps(f1, extrabias1);
                    f2 = _mm512_sub_ps(f2, extrabias2);
                }
            }

            PLUS_TERM_3(0,1,2);
            if (nullptr != biasPtr) {
                auto biasValue = _mm512_loadu_ps(bias_dz);
                SCALE_BIAS_VEC(0);
                SCALE_BIAS_VEC(1);
                SCALE_BIAS_VEC(2);
            }

            scaleValue = _mm512_loadu_ps(scale_dz + 1 * PACK_UNIT);
            weightBiasValue = _mm512_loadu_ps(weightBias_dz + 1 * PACK_UNIT);
            // x_kernelSum x w_quantZero
            SRCKERNELSUM_MUL_WEIGHTQUANBIAS_3;
            DEQUANT_VALUE(4);
            DEQUANT_VALUE(5);
            DEQUANT_VALUE(6);

            if (post->extraScale) { // Batch quant
                f4 = _mm512_mul_ps(f4, extrascale0);
                f5 = _mm512_mul_ps(f5, extrascale1);
                f6 = _mm512_mul_ps(f6, extrascale2);
                if (post->extraBias && nullptr != biasPtr) {
                    auto extrabias = _mm512_loadu_ps(extraB_dz + 1 * PACK_UNIT);
                    extrabias = _mm512_mul_ps(f128, extrabias);
                    auto extrabias0 = _mm512_mul_ps(extrabias, extrascale0);
                    auto extrabias1 = _mm512_mul_ps(extrabias, extrascale1);
                    auto extrabias2 = _mm512_mul_ps(extrabias, extrascale2);
                    f4 = _mm512_sub_ps(f4, extrabias0);
                    f5 = _mm512_sub_ps(f5, extrabias1);
                    f6 = _mm512_sub_ps(f6, extrabias2);
                }
            }

            PLUS_TERM_3(4,5,6);
            if (biasPtr) {
                auto biasValue = _mm512_loadu_ps(bias_dz + 1 * PACK_UNIT);
                SCALE_BIAS_VEC(4);
                SCALE_BIAS_VEC(5);
                SCALE_BIAS_VEC(6);
            }

            scaleValue = _mm512_loadu_ps(scale_dz + 2 * PACK_UNIT);
            weightBiasValue = _mm512_loadu_ps(weightBias_dz + 2 * PACK_UNIT);
            // x_kernelSum x w_quantZero
            SRCKERNELSUM_MUL_WEIGHTQUANBIAS_3;
            DEQUANT_VALUE(8);
            DEQUANT_VALUE(9);
            DEQUANT_VALUE(10);

            if (post->extraScale) { // Batch quant
                f8 = _mm512_mul_ps(f8, extrascale0);
                f9 = _mm512_mul_ps(f9, extrascale1);
                f10 = _mm512_mul_ps(f10, extrascale2);
                if (post->extraBias && nullptr != biasPtr) {
                    auto extrabias = _mm512_loadu_ps(extraB_dz + 2 * PACK_UNIT);
                    extrabias = _mm512_mul_ps(f128, extrabias);
                    auto extrabias0 = _mm512_mul_ps(extrabias, extrascale0);
                    auto extrabias1 = _mm512_mul_ps(extrabias, extrascale1);
                    auto extrabias2 = _mm512_mul_ps(extrabias, extrascale2);
                    f8 = _mm512_sub_ps(f8, extrabias0);
                    f9 = _mm512_sub_ps(f9, extrabias1);
                    f10 = _mm512_sub_ps(f10, extrabias2);
                }
            }

            PLUS_TERM_3(8,9,10);
            if (biasPtr) {
                auto biasValue = _mm512_loadu_ps(bias_dz + 2 * PACK_UNIT);
                SCALE_BIAS_VEC(8);
                SCALE_BIAS_VEC(9);
                SCALE_BIAS_VEC(10);
            }

            scaleValue = _mm512_loadu_ps(scale_dz + 3 * PACK_UNIT);
            weightBiasValue = _mm512_loadu_ps(weightBias_dz + 3 * PACK_UNIT);
            // x_kernelSum x w_quantZero
            SRCKERNELSUM_MUL_WEIGHTQUANBIAS_3;
            DEQUANT_VALUE(12);
            DEQUANT_VALUE(13);
            DEQUANT_VALUE(14);

            if (post->extraScale) { // Batch quant
                f12 = _mm512_mul_ps(f12, extrascale0);
                f13 = _mm512_mul_ps(f13, extrascale1);
                f14 = _mm512_mul_ps(f14, extrascale2);
                if (post->extraBias && nullptr != biasPtr) {
                    auto extrabias = _mm512_loadu_ps(extraB_dz + 3 * PACK_UNIT);
                    extrabias = _mm512_mul_ps(f128, extrabias);
                    auto extrabias0 = _mm512_mul_ps(extrabias, extrascale0);
                    auto extrabias1 = _mm512_mul_ps(extrabias, extrascale1);
                    auto extrabias2 = _mm512_mul_ps(extrabias, extrascale2);
                    f12 = _mm512_sub_ps(f12, extrabias0);
                    f13 = _mm512_sub_ps(f13, extrabias1);
                    f14 = _mm512_sub_ps(f14, extrabias2);
                }
            }

            PLUS_TERM_3(12,13,14);
            if (biasPtr) {
                auto biasValue = _mm512_loadu_ps(bias_dz + 3 * PACK_UNIT);
                SCALE_BIAS_VEC(12);
                SCALE_BIAS_VEC(13);
                SCALE_BIAS_VEC(14);
            }

            if (biasPtr == nullptr) {
                auto dstTmp = dst_x;
                f0 = _mm512_add_ps(_mm512_loadu_ps(((float*)dstTmp)), f0);
                f1 = _mm512_add_ps(_mm512_loadu_ps(((float*)dstTmp) + 16), f1);
                f2 = _mm512_add_ps(_mm512_loadu_ps(((float*)dstTmp) + 16 * 2), f2);
                dstTmp += dst_step_tmp;
                f4 = _mm512_add_ps(_mm512_loadu_ps(((float*)dstTmp) + 16 * 0), f4);
                f5 = _mm512_add_ps(_mm512_loadu_ps(((float*)dstTmp) + 16 * 1), f5);
                f6 = _mm512_add_ps(_mm512_loadu_ps(((float*)dstTmp) + 16 * 2), f6);
                dstTmp += dst_step_tmp;
                f8 = _mm512_add_ps(_mm512_loadu_ps(((float*)dstTmp) + 16 * 0), f8);
                f9 = _mm512_add_ps(_mm512_loadu_ps(((float*)dstTmp) + 16 * 1), f9);
                f10 = _mm512_add_ps(_mm512_loadu_ps(((float*)dstTmp) + 16 * 2), f10);
                dstTmp += dst_step_tmp;
                f12 = _mm512_add_ps(_mm512_loadu_ps(((float*)dstTmp) + 16 * 0), f12);
                f13 = _mm512_add_ps(_mm512_loadu_ps(((float*)dstTmp) + 16 * 1), f13);
                f14 = _mm512_add_ps(_mm512_loadu_ps(((float*)dstTmp) + 16 * 2), f14);
            }
            if (post->fp32minmax) {
                POST_TREAT_FLOAT_3(0,1,2);
                POST_TREAT_FLOAT_3(4,5,6);
                POST_TREAT_FLOAT_3(8,9,10);
                POST_TREAT_FLOAT_3(12,13,14);
            }
            _mm512_storeu_ps(((float*)dst_x), f0);
            _mm512_storeu_ps(((float*)dst_x) + 16, f1);
            _mm512_storeu_ps(((float*)dst_x) + 16 * 2, f2);
            dst_x += dst_step_tmp;
            _mm512_storeu_ps(((float*)dst_x) + 16 * 0, f4);
            _mm512_storeu_ps(((float*)dst_x) + 16 * 1, f5);
            _mm512_storeu_ps(((float*)dst_x) + 16 * 2, f6);
            dst_x += dst_step_tmp;
            _mm512_storeu_ps(((float*)dst_x) + 16 * 0, f8);
            _mm512_storeu_ps(((float*)dst_x) + 16 * 1, f9);
            _mm512_storeu_ps(((float*)dst_x) + 16 * 2, f10);
            dst_x += dst_step_tmp;
            _mm512_storeu_ps(((float*)dst_x) + 16 * 0, f12);
            _mm512_storeu_ps(((float*)dst_x) + 16 * 1, f13);
            _mm512_storeu_ps(((float*)dst_x) + 16 * 2, f14);
            
        }
        auto weight_dz = weight + dzU * weight_step_Z;
        if (biasPtr) {
            bias_dz = post->biasFloat + dzU * PACK_UNIT * dzUnit;
        }
        if (post->extraBias) {
            extraB_dz = post->extraBias + dzU * PACK_UNIT * dzUnit;
        }
        float* scale_dz = (float*)post->scale + dzU * PACK_UNIT * dzUnit;
        const auto weightBias_dz = post->weightQuanBias + dzU * PACK_UNIT * dzUnit;

        auto dst_z = dst + dzU * dst_step_tmp * dzUnit;
        const auto src_x   = src;
        auto dst_x         = dst_z;
        for (int i=0; i<dzR; ++i) {
            __m512i D0 = _mm512_set1_epi32(0);
            __m512i D1 = _mm512_set1_epi32(0);
            __m512i D2 = _mm512_set1_epi32(0);

            for (int sz = 0; sz < src_depth_quad; ++sz) {
                const auto weight_sz = weight_dz + weight_step_Y * sz;
                const auto src_z     = (const float*)(src_x + sz * GEMMINT8_AVX512_E * GEMMINT8_AVX512_L);
                auto w0_int4_64 = _mm512_loadu_si512(weight_sz); // 128xint4_t=64 byte
                auto w0 = _mm512_and_si512(mask, _mm512_srli_epi16(w0_int4_64, 4)); // 64xint8_t

                auto s0 = AVX512_BROADCAST_INT32(src_z + 0);
                auto s1 = AVX512_BROADCAST_INT32(src_z + 1);
                auto s2 = AVX512_BROADCAST_INT32(src_z + 2);

                D0 = _mm512_dpbusds_epi32(D0, s0, w0);
                D1 = _mm512_dpbusds_epi32(D1, s1, w0);
                D2 = _mm512_dpbusds_epi32(D2, s2, w0);
            }

            
            auto scaleValue = _mm512_loadu_ps(scale_dz);
            auto weightBiasValue = _mm512_loadu_ps(weightBias_dz);
            __m512 xy0_0, xy0_1, xy0_2;
            // x_kernelSum x w_quantZero
            SRCKERNELSUM_MUL_WEIGHTQUANBIAS_3;
            DEQUANT_VALUE(0);
            DEQUANT_VALUE(1);
            DEQUANT_VALUE(2);

            if (post->extraScale) { // Batch quant
                f0 = _mm512_mul_ps(f0, extrascale0);
                f1 = _mm512_mul_ps(f1, extrascale1);
                f2 = _mm512_mul_ps(f2, extrascale2);
                if (post->extraBias && nullptr != biasPtr) {
                    auto extrabias = _mm512_loadu_ps(extraB_dz);
                    extrabias = _mm512_mul_ps(f128, extrabias);
                    auto extrabias0 = _mm512_mul_ps(extrabias, extrascale0);
                    auto extrabias1 = _mm512_mul_ps(extrabias, extrascale1);
                    auto extrabias2 = _mm512_mul_ps(extrabias, extrascale2);
                    f0 = _mm512_sub_ps(f0, extrabias0);
                    f1 = _mm512_sub_ps(f1, extrabias1);
                    f2 = _mm512_sub_ps(f2, extrabias2);
                }
            }

            PLUS_TERM_3(0,1,2);
            if (biasPtr) {
                auto biasValue = _mm512_loadu_ps(bias_dz);
                SCALE_BIAS_VEC(0);
                SCALE_BIAS_VEC(1);
                SCALE_BIAS_VEC(2);
            }

            if (biasPtr == nullptr) {
                f0 = _mm512_add_ps(_mm512_loadu_ps(((float*)dst_x)), f0);
                f1 = _mm512_add_ps(_mm512_loadu_ps(((float*)dst_x) + 16), f1);
                f2 = _mm512_add_ps(_mm512_loadu_ps(((float*)dst_x) + 16 * 2), f2);
            }
            if (post->fp32minmax) {
                POST_TREAT_FLOAT_3(0,1,2);
            }
            _mm512_storeu_ps(((float*)dst_x), f0);
            _mm512_storeu_ps(((float*)dst_x) + 16, f1);
            _mm512_storeu_ps(((float*)dst_x) + 16 * 2, f2);
            
            dst_x += dst_step_tmp;
            scale_dz += PACK_UNIT;
            if (biasPtr) {
                bias_dz += PACK_UNIT;
            }
            if (post->extraBias) {
                extraB_dz += PACK_UNIT;
            }
            weight_dz += PACK_UNIT * GEMMINT8_AVX512_L;
        }
        return;
    }
    // e = 2
    if (realDst == 2) {
        for (int dz = 0; dz < dzU; ++dz) {
            auto weight_dz = weight + dz * weight_step_Z;
            if (biasPtr) {
                bias_dz = post->biasFloat + dz * PACK_UNIT * dzUnit;
            }
            if (post->extraBias) {
                extraB_dz = post->extraBias + dz * PACK_UNIT * dzUnit;
            }
            float* scale_dz = (float*)post->scale + dz * PACK_UNIT * dzUnit;
            const auto weightBias_dz = post->weightQuanBias + dz * PACK_UNIT * dzUnit;
            auto dst_z = dst + dz * dst_step_tmp * dzUnit;
            const auto src_x   = src;
            auto dst_x         = dst_z;
            __m512i D0 = _mm512_set1_epi32(0);
            __m512i D1 = _mm512_set1_epi32(0);

            __m512i D4 = _mm512_set1_epi32(0);
            __m512i D5 = _mm512_set1_epi32(0);

            __m512i D8 = _mm512_set1_epi32(0);
            __m512i D9 = _mm512_set1_epi32(0);

            __m512i D12 = _mm512_set1_epi32(0);
            __m512i D13 = _mm512_set1_epi32(0);


            for (int sz = 0; sz < src_depth_quad; ++sz) {
                const auto weight_sz = weight_dz + weight_step_Y * sz;
                const auto src_z     = (const float*)(src_x + sz * GEMMINT8_AVX512_E * GEMMINT8_AVX512_L);
                // int4->int8: total count=4*64(GEMMINT8_AVX512_L * GEMMINT8_AVX512_H)
                // Load 4*64 int4 weight
                auto w0_int4_64 = _mm512_loadu_si512(weight_sz); // 128xint4_t=64 byte
                auto w1_int4_64 = _mm512_loadu_si512(weight_sz + 64); // 128xint4_t
                // 256xint4_t->256xint8_t
                auto w0 = _mm512_and_si512(mask, _mm512_srli_epi16(w0_int4_64, 4)); // 64xint8_t
                auto w2 = _mm512_and_si512(mask, w0_int4_64); // 64xint8_t
                auto w1 = _mm512_and_si512(mask, _mm512_srli_epi16(w1_int4_64, 4));
                auto w3 = _mm512_and_si512(mask, w1_int4_64);

                auto s0 = AVX512_BROADCAST_INT32(src_z + 0);
                auto s1 = AVX512_BROADCAST_INT32(src_z + 1);

                D0 = _mm512_dpbusds_epi32(D0, s0, w0);
                D1 = _mm512_dpbusds_epi32(D1, s1, w0);

                D4 = _mm512_dpbusds_epi32(D4, s0, w1);
                D5 = _mm512_dpbusds_epi32(D5, s1, w1);

                D8 = _mm512_dpbusds_epi32(D8, s0, w2);
                D9 = _mm512_dpbusds_epi32(D9, s1, w2);

                D12 = _mm512_dpbusds_epi32(D12, s0, w3);
                D13 = _mm512_dpbusds_epi32(D13, s1, w3);
            }

            auto scaleValue = _mm512_loadu_ps(scale_dz);
            auto weightBiasValue = _mm512_loadu_ps(weightBias_dz);
            __m512 xy0_0, xy0_1;

            // x_kernelSum x w_quantZero
            SRCKERNELSUM_MUL_WEIGHTQUANBIAS_2;
            DEQUANT_VALUE(0);
            DEQUANT_VALUE(1);

            if (post->extraScale) { // Batch quant
                f0 = _mm512_mul_ps(f0, extrascale0);
                f1 = _mm512_mul_ps(f1, extrascale1);
                if (post->extraBias && nullptr != biasPtr) {
                    auto extrabias = _mm512_loadu_ps(extraB_dz);
                    extrabias = _mm512_mul_ps(f128, extrabias);
                    auto extrabias0 = _mm512_mul_ps(extrabias, extrascale0);
                    auto extrabias1 = _mm512_mul_ps(extrabias, extrascale1);
                    f0 = _mm512_sub_ps(f0, extrabias0);
                    f1 = _mm512_sub_ps(f1, extrabias1);
                }
            }

            PLUS_TERM_2(0,1);
            if (biasPtr) {
                auto biasValue = _mm512_loadu_ps(bias_dz);
                SCALE_BIAS_VEC(0);
                SCALE_BIAS_VEC(1);
            }

            scaleValue = _mm512_loadu_ps(scale_dz + 1 * PACK_UNIT);
            weightBiasValue = _mm512_loadu_ps(weightBias_dz + 1 * PACK_UNIT);
            // x_kernelSum x w_quantZero
            SRCKERNELSUM_MUL_WEIGHTQUANBIAS_2;
            DEQUANT_VALUE(4);
            DEQUANT_VALUE(5);

            if (post->extraScale) { // Batch quant
                f4 = _mm512_mul_ps(f4, extrascale0);
                f5 = _mm512_mul_ps(f5, extrascale1);
                if (post->extraBias && nullptr != biasPtr) {
                    auto extrabias = _mm512_loadu_ps(extraB_dz + 1 * PACK_UNIT);
                    extrabias = _mm512_mul_ps(f128, extrabias);
                    auto extrabias0 = _mm512_mul_ps(extrabias, extrascale0);
                    auto extrabias1 = _mm512_mul_ps(extrabias, extrascale1);
                    f4 = _mm512_sub_ps(f4, extrabias0);
                    f5 = _mm512_sub_ps(f5, extrabias1);
                }
            }

            PLUS_TERM_2(4,5);
            if (biasPtr) {
                auto biasValue = _mm512_loadu_ps(bias_dz + 1 * PACK_UNIT);
                SCALE_BIAS_VEC(4);
                SCALE_BIAS_VEC(5);
            }

            scaleValue = _mm512_loadu_ps(scale_dz + 2 * PACK_UNIT);
            weightBiasValue = _mm512_loadu_ps(weightBias_dz + 2 * PACK_UNIT);
            SRCKERNELSUM_MUL_WEIGHTQUANBIAS_2;
            DEQUANT_VALUE(8);
            DEQUANT_VALUE(9);

            if (post->extraScale) { // Batch quant
                f8 = _mm512_mul_ps(f8, extrascale0);
                f9 = _mm512_mul_ps(f9, extrascale1);
                if (post->extraBias && nullptr != biasPtr) {
                    auto extrabias = _mm512_loadu_ps(extraB_dz + 2 * PACK_UNIT);
                    extrabias = _mm512_mul_ps(f128, extrabias);
                    auto extrabias0 = _mm512_mul_ps(extrabias, extrascale0);
                    auto extrabias1 = _mm512_mul_ps(extrabias, extrascale1);
                    f8 = _mm512_sub_ps(f8, extrabias0);
                    f9 = _mm512_sub_ps(f9, extrabias1);
                }
            }

            PLUS_TERM_2(8,9);
            if (biasPtr) {
                auto biasValue = _mm512_loadu_ps(bias_dz + 2 * PACK_UNIT);
                SCALE_BIAS_VEC(8);
                SCALE_BIAS_VEC(9);
            }

            scaleValue = _mm512_loadu_ps(scale_dz + 3 * PACK_UNIT);
            weightBiasValue = _mm512_loadu_ps(weightBias_dz + 3 * PACK_UNIT);
            SRCKERNELSUM_MUL_WEIGHTQUANBIAS_2;
            DEQUANT_VALUE(12);
            DEQUANT_VALUE(13);

            if (post->extraScale) { // Batch quant
                f12 = _mm512_mul_ps(f12, extrascale0);
                f13 = _mm512_mul_ps(f13, extrascale1);
                if (post->extraBias && nullptr != biasPtr) {
                    auto extrabias = _mm512_loadu_ps(extraB_dz + 3 * PACK_UNIT);
                    extrabias = _mm512_mul_ps(f128, extrabias);
                    auto extrabias0 = _mm512_mul_ps(extrabias, extrascale0);
                    auto extrabias1 = _mm512_mul_ps(extrabias, extrascale1);
                    f12 = _mm512_sub_ps(f12, extrabias0);
                    f13 = _mm512_sub_ps(f13, extrabias1);
                }
            }

            PLUS_TERM_2(12,13);
            if (biasPtr) {
                auto biasValue = _mm512_loadu_ps(bias_dz + 3 * PACK_UNIT);
                SCALE_BIAS_VEC(12);
                SCALE_BIAS_VEC(13);
            }

            if (nullptr == biasPtr) {
                auto dstTmp = dst_x;
                f0 = _mm512_add_ps(_mm512_loadu_ps(((float*)dstTmp)), f0);
                f1 = _mm512_add_ps(_mm512_loadu_ps(((float*)dstTmp) + 16), f1);
                dstTmp += dst_step_tmp;
                f4 = _mm512_add_ps(_mm512_loadu_ps(((float*)dstTmp) + 16 * 0), f4);
                f5 = _mm512_add_ps(_mm512_loadu_ps(((float*)dstTmp) + 16 * 1), f5);
                dstTmp += dst_step_tmp;
                f8 = _mm512_add_ps(_mm512_loadu_ps(((float*)dstTmp) + 16 * 0), f8);
                f9 = _mm512_add_ps(_mm512_loadu_ps(((float*)dstTmp) + 16 * 1), f9);
                dstTmp += dst_step_tmp;
                f12 = _mm512_add_ps(_mm512_loadu_ps(((float*)dstTmp) + 16 * 0), f12);
                f13 = _mm512_add_ps(_mm512_loadu_ps(((float*)dstTmp) + 16 * 1), f13);
            }
            if (post->fp32minmax) {
                POST_TREAT_FLOAT_2(0,1);
                POST_TREAT_FLOAT_2(4,5);
                POST_TREAT_FLOAT_2(8,9);
                POST_TREAT_FLOAT_2(12,13);
            }
            _mm512_storeu_ps(((float*)dst_x), f0);
            _mm512_storeu_ps(((float*)dst_x) + 16, f1);
            dst_x += dst_step_tmp;
            _mm512_storeu_ps(((float*)dst_x) + 16 * 0, f4);
            _mm512_storeu_ps(((float*)dst_x) + 16 * 1, f5);
            dst_x += dst_step_tmp;
            _mm512_storeu_ps(((float*)dst_x) + 16 * 0, f8);
            _mm512_storeu_ps(((float*)dst_x) + 16 * 1, f9);
            dst_x += dst_step_tmp;
            _mm512_storeu_ps(((float*)dst_x) + 16 * 0, f12);
            _mm512_storeu_ps(((float*)dst_x) + 16 * 1, f13);

        }
        auto weight_dz = weight + dzU * weight_step_Z;
        if (biasPtr) {
            bias_dz = post->biasFloat + dzU * PACK_UNIT * dzUnit;
        }
        if (post->extraBias) {
            extraB_dz = post->extraBias + dzU * PACK_UNIT * dzUnit;
        }
        float* scale_dz = (float*)post->scale + dzU * PACK_UNIT * dzUnit;
        const auto weightBias_dz = post->weightQuanBias + dzU * PACK_UNIT * dzUnit;

        auto dst_z = dst + dzU * dst_step_tmp * dzUnit;
        const auto src_x   = src;
        auto dst_x         = dst_z;
        for (int i=0; i<dzR; ++i) {
            __m512i D0 = _mm512_set1_epi32(0);
            __m512i D1 = _mm512_set1_epi32(0);

            for (int sz = 0; sz < src_depth_quad; ++sz) {
                const auto weight_sz = weight_dz + weight_step_Y * sz;
                const auto src_z     = (const float*)(src_x + sz * GEMMINT8_AVX512_E * GEMMINT8_AVX512_L);
                auto w0_int4_64 = _mm512_loadu_si512(weight_sz); // 128xint4_t=64 byte
                auto w0 = _mm512_and_si512(mask, _mm512_srli_epi16(w0_int4_64, 4)); // 64xint8_t

                auto s0 = AVX512_BROADCAST_INT32(src_z + 0);
                auto s1 = AVX512_BROADCAST_INT32(src_z + 1);

                D0 = _mm512_dpbusds_epi32(D0, s0, w0);
                D1 = _mm512_dpbusds_epi32(D1, s1, w0);
            }

            auto scaleValue = _mm512_loadu_ps(scale_dz);
            auto weightBiasValue = _mm512_loadu_ps(weightBias_dz);
            __m512 xy0_0, xy0_1;
            // x_kernelSum x w_quantZero
            SRCKERNELSUM_MUL_WEIGHTQUANBIAS_2;
            DEQUANT_VALUE(0);
            DEQUANT_VALUE(1);

            if (post->extraScale) { // Batch quant
                f0 = _mm512_mul_ps(f0, extrascale0);
                f1 = _mm512_mul_ps(f1, extrascale1);
                if (post->extraBias && nullptr != biasPtr) {
                    auto extrabias = _mm512_loadu_ps(extraB_dz);
                    extrabias = _mm512_mul_ps(f128, extrabias);
                    auto extrabias0 = _mm512_mul_ps(extrabias, extrascale0);
                    auto extrabias1 = _mm512_mul_ps(extrabias, extrascale1);
                    f0 = _mm512_sub_ps(f0, extrabias0);
                    f1 = _mm512_sub_ps(f1, extrabias1);
                }
            }

            PLUS_TERM_2(0,1);
            if (biasPtr) {
                auto biasValue = _mm512_loadu_ps(bias_dz);
                SCALE_BIAS_VEC(0);
                SCALE_BIAS_VEC(1);
            }

            if (nullptr == biasPtr) {
                f0 = _mm512_add_ps(_mm512_loadu_ps(((float*)dst_x)), f0);
                f1 = _mm512_add_ps(_mm512_loadu_ps(((float*)dst_x) + 16), f1);
            }
            POST_TREAT_FLOAT_2(0,1);
            _mm512_storeu_ps(((float*)dst_x), f0);
            _mm512_storeu_ps(((float*)dst_x) + 16, f1);
            
            dst_x += dst_step_tmp;
            scale_dz += PACK_UNIT;
            if (biasPtr) {
                bias_dz += PACK_UNIT;
            }
            if (post->extraBias) {
                extraB_dz += PACK_UNIT;
            }
            weight_dz += PACK_UNIT * GEMMINT8_AVX512_L;
        }
        return;
    }
    if (realDst == 1) {
        for (int dz = 0; dz < dzU; ++dz) {
            auto weight_dz = weight + dz * weight_step_Z;
            if (biasPtr) {
                bias_dz = post->biasFloat + dz * PACK_UNIT * dzUnit;
            }
            if (post->extraBias) {
                extraB_dz = post->extraBias + dz * PACK_UNIT * dzUnit;
            }
            float* scale_dz = (float*)post->scale + dz * PACK_UNIT * dzUnit;
            const auto weightBias_dz = post->weightQuanBias + dz * PACK_UNIT * dzUnit;
            auto dst_z = dst + dz * dst_step_tmp * dzUnit;
            const auto src_x   = src;
            auto dst_x         = dst_z;
            __m512i D0 = _mm512_set1_epi32(0);

            __m512i D4 = _mm512_set1_epi32(0);

            __m512i D8 = _mm512_set1_epi32(0);

            __m512i D12 = _mm512_set1_epi32(0);

            for (int sz = 0; sz < src_depth_quad; ++sz) {
                const auto weight_sz = weight_dz + weight_step_Y * sz;
                const auto src_z     = (const float*)(src_x + sz * GEMMINT8_AVX512_E * GEMMINT8_AVX512_L);
                // int4->int8: total count=4*64(GEMMINT8_AVX512_L * GEMMINT8_AVX512_H)
                // Load 4*64 int4 weight
                auto w0_int4_64 = _mm512_loadu_si512(weight_sz); // 128xint4_t=64 byte
                auto w1_int4_64 = _mm512_loadu_si512(weight_sz + 64); // 128xint4_t
                // 256xint4_t->256xint8_t
                auto w0 = _mm512_and_si512(mask, _mm512_srli_epi16(w0_int4_64, 4)); // 64xint8_t
                auto w2 = _mm512_and_si512(mask, w0_int4_64); // 64xint8_t
                auto w1 = _mm512_and_si512(mask, _mm512_srli_epi16(w1_int4_64, 4));
                auto w3 = _mm512_and_si512(mask, w1_int4_64);

                auto s0 = AVX512_BROADCAST_INT32(src_z + 0);

                D0 = _mm512_dpbusds_epi32(D0, s0, w0);

                D4 = _mm512_dpbusds_epi32(D4, s0, w1);

                D8 = _mm512_dpbusds_epi32(D8, s0, w2);

                D12 = _mm512_dpbusds_epi32(D12, s0, w3);
            }

            auto scaleValue = _mm512_loadu_ps(scale_dz);
            auto weightBiasValue = _mm512_loadu_ps(weightBias_dz);
            __m512 xy0_0;

            // x_kernelSum x w_quantZero
            SRCKERNELSUM_MUL_WEIGHTQUANBIAS_1;
            DEQUANT_VALUE(0);

            if (post->extraScale) { // Batch quant
                f0 = _mm512_mul_ps(f0, extrascale0);
                if (post->extraBias && nullptr != biasPtr) {
                    auto extrabias = _mm512_loadu_ps(extraB_dz);
                    extrabias = _mm512_mul_ps(f128, extrabias);
                    auto extrabias0 = _mm512_mul_ps(extrabias, extrascale0);
                    f0 = _mm512_sub_ps(f0, extrabias0);
                }
            }

            PLUS_TERM_1(0);
            if (biasPtr) {
                auto biasValue = _mm512_loadu_ps(bias_dz);
                SCALE_BIAS_VEC(0);
            }

            scaleValue = _mm512_loadu_ps(scale_dz + 1 * PACK_UNIT);
            weightBiasValue = _mm512_loadu_ps(weightBias_dz + 1 * PACK_UNIT);
            // x_kernelSum x w_quantZero
            SRCKERNELSUM_MUL_WEIGHTQUANBIAS_1;
            DEQUANT_VALUE(4);

            if (post->extraScale) { // Batch quant
                f4 = _mm512_mul_ps(f4, extrascale0);
                if (post->extraBias && nullptr != biasPtr) {
                    auto extrabias = _mm512_loadu_ps(extraB_dz + 1 * PACK_UNIT);
                    extrabias = _mm512_mul_ps(f128, extrabias);
                    auto extrabias0 = _mm512_mul_ps(extrabias, extrascale0);
                    f4 = _mm512_sub_ps(f4, extrabias0);
                }
            }

            PLUS_TERM_1(4);
            if (biasPtr) {
                auto biasValue = _mm512_loadu_ps(bias_dz + 1 * PACK_UNIT);
                SCALE_BIAS_VEC(4);
            }

            scaleValue = _mm512_loadu_ps(scale_dz + 2 * PACK_UNIT);
            weightBiasValue = _mm512_loadu_ps(weightBias_dz + 2 * PACK_UNIT);
            // x_kernelSum x w_quantZero
            SRCKERNELSUM_MUL_WEIGHTQUANBIAS_1;
            DEQUANT_VALUE(8);

            if (post->extraScale) { // Batch quant
                f8 = _mm512_mul_ps(f8, extrascale0);
                if (post->extraBias && nullptr != biasPtr) {
                    auto extrabias = _mm512_loadu_ps(extraB_dz + 2 * PACK_UNIT);
                    extrabias = _mm512_mul_ps(f128, extrabias);
                    auto extrabias0 = _mm512_mul_ps(extrabias, extrascale0);
                    f8 = _mm512_sub_ps(f8, extrabias0);
                }
            }

            PLUS_TERM_1(8);
            if (biasPtr) {
                auto biasValue = _mm512_loadu_ps(bias_dz + 2 * PACK_UNIT);
                SCALE_BIAS_VEC(8);
            }

            scaleValue = _mm512_loadu_ps(scale_dz + 3 * PACK_UNIT);
            weightBiasValue = _mm512_loadu_ps(weightBias_dz + 3 * PACK_UNIT);
            // x_kernelSum x w_quantZero
            SRCKERNELSUM_MUL_WEIGHTQUANBIAS_1;
            DEQUANT_VALUE(12);

            if (post->extraScale) { // Batch quant
                f12 = _mm512_mul_ps(f12, extrascale0);
                if (post->extraBias && nullptr != biasPtr) {
                    auto extrabias = _mm512_loadu_ps(extraB_dz + 3 * PACK_UNIT);
                    extrabias = _mm512_mul_ps(f128, extrabias);
                    auto extrabias0 = _mm512_mul_ps(extrabias, extrascale0);
                    f12 = _mm512_sub_ps(f12, extrabias0);
                }
            }

            PLUS_TERM_1(12);
            if (biasPtr) {
                auto biasValue = _mm512_loadu_ps(bias_dz + 3 * PACK_UNIT);
                SCALE_BIAS_VEC(12);
            }

            if (nullptr == biasPtr) {
                auto dstTemp = dst_x;
                f0 = _mm512_add_ps(_mm512_loadu_ps(((float*)dstTemp)), f0);
                dstTemp += dst_step_tmp;
                f4 = _mm512_add_ps(_mm512_loadu_ps(((float*)dstTemp) + 16 * 0), f4);
                dstTemp += dst_step_tmp;
                f8 = _mm512_add_ps(_mm512_loadu_ps(((float*)dstTemp) + 16 * 0), f8);
                dstTemp += dst_step_tmp;
                f12 = _mm512_add_ps(_mm512_loadu_ps(((float*)dstTemp) + 16 * 0), f12);
            }
            if (post->fp32minmax) {
                POST_TREAT_FLOAT_1(0);
                POST_TREAT_FLOAT_1(4);
                POST_TREAT_FLOAT_1(8);
                POST_TREAT_FLOAT_1(12);
            }
            _mm512_storeu_ps(((float*)dst_x), f0);
            dst_x += dst_step_tmp;
            _mm512_storeu_ps(((float*)dst_x) + 16 * 0, f4);
            dst_x += dst_step_tmp;
            _mm512_storeu_ps(((float*)dst_x) + 16 * 0, f8);
            dst_x += dst_step_tmp;
            _mm512_storeu_ps(((float*)dst_x) + 16 * 0, f12);
            
        }
        auto weight_dz = weight + dzU * weight_step_Z;
        if (biasPtr) {
            bias_dz = post->biasFloat + dzU * PACK_UNIT * dzUnit;
        }
        if (post->extraBias) {
            extraB_dz = post->extraBias + dzU * PACK_UNIT * dzUnit;
        }
        float* scale_dz = (float*)post->scale + dzU * PACK_UNIT * dzUnit;
        const auto weightBias_dz = post->weightQuanBias + dzU * PACK_UNIT * dzUnit;

        auto dst_z = dst + dzU * dst_step_tmp * dzUnit;
        const auto src_x   = src;
        auto dst_x         = dst_z;
        for (int i=0; i<dzR; ++i) {
            __m512i D0 = _mm512_set1_epi32(0);

            for (int sz = 0; sz < src_depth_quad; ++sz) {
                const auto weight_sz = weight_dz + weight_step_Y * sz;
                const auto src_z     = (const float*)(src_x + sz * GEMMINT8_AVX512_E * GEMMINT8_AVX512_L);
                auto w0_int4_64 = _mm512_loadu_si512(weight_sz); // 128xint4_t=64 byte
                auto w0 = _mm512_and_si512(mask, _mm512_srli_epi16(w0_int4_64, 4)); // 64xint8_t

                auto s0 = AVX512_BROADCAST_INT32(src_z + 0);

                D0 = _mm512_dpbusds_epi32(D0, s0, w0);
            }

            auto scaleValue = _mm512_loadu_ps(scale_dz);

            auto weightBiasValue = _mm512_loadu_ps(weightBias_dz);
            __m512 xy0_0;
            // x_kernelSum x w_quantZero
            SRCKERNELSUM_MUL_WEIGHTQUANBIAS_1;
            DEQUANT_VALUE(0);

            if (post->extraScale) { // Batch quant
                f0 = _mm512_mul_ps(f0, extrascale0);
                if (post->extraBias && nullptr != biasPtr) {
                    auto extrabias = _mm512_loadu_ps(extraB_dz);
                    extrabias = _mm512_mul_ps(f128, extrabias);
                    auto extrabias0 = _mm512_mul_ps(extrabias, extrascale0);
                    f0 = _mm512_sub_ps(f0, extrabias0);
                }
            }

            PLUS_TERM_1(0);
            if (biasPtr) {
                auto biasValue = _mm512_loadu_ps(bias_dz);
                SCALE_BIAS_VEC(0);
            }

            if (nullptr == biasPtr) {
                f0 = _mm512_add_ps(_mm512_loadu_ps(((float*)dst_x)), f0);
            }
            if (post->fp32minmax) {
                POST_TREAT_FLOAT_1(0);
            }
            _mm512_storeu_ps(((float*)dst_x), f0);
            dst_x += dst_step_tmp;
            scale_dz += PACK_UNIT;
            if (biasPtr) {
                bias_dz += PACK_UNIT;
            }
            if (post->extraBias) {
                extraB_dz += PACK_UNIT;
            }
            weight_dz += PACK_UNIT * GEMMINT8_AVX512_L;
        }
        return;
    }
}

void _AVX512_MNNLineDepthWiseInt8AddBiasScaleUnit_VNNI(int8_t* dstO, const int8_t* srcO, const int8_t* weightO, const QuanPostTreatParameters* parameters, size_t width, size_t src_w_step, size_t fw, size_t fh, size_t dilateX_step, size_t dilateY_step, int8_t* idxOrder) {
    auto dst = dstO;
    auto src = (const int16_t*)srcO;
    int widthC4 = width / 2;
    int widthRemain = width % 2;
    auto weight = (const int16_t*)weightO;
    auto biasValue0 = _mm256_castps_si256(_mm256_loadu_ps((const float*)parameters->bias));
    auto scaleValue0 = _mm256_loadu_ps((const float*)parameters->scale);
    auto biasValue1 = _mm256_castps_si256(_mm256_loadu_ps((const float*)parameters->bias + 8));
    auto scaleValue1 = _mm256_loadu_ps((const float*)parameters->scale + 8);
    __m256i d0, d1, d2, d3;
    int dx, fx, fy;
    __m256i zero = _mm256_setzero_si256();
    __m256 zero128 = _mm256_set1_ps(0.0f);
    __m128i minValue = _mm_set1_epi16(parameters->minValue + 128);
    __m128i maxValue = _mm_set1_epi16(parameters->maxValue + 128);
    __m256 plus = _mm256_set1_ps(0.5f);
    __m256 minus = _mm256_set1_ps(-0.5f);
    for (dx = 0; dx < widthC4; ++dx) {
        d0 = biasValue0;
        d1 = biasValue1;
        d2 = biasValue0;
        d3 = biasValue1;

        auto dst_x          = dst;
        const auto src_z    = src;
        for (fy = 0; fy < fh; ++fy) {
            const auto src_y    = src_z + fy * dilateY_step;
            const auto weight_y = weight + fy * fw * PACK_UNIT;
            for (fx = 0; fx < fw; ++fx) {
                const auto src_x    = src_y + fx * dilateX_step;
                auto S0 = _mm256_castsi128_si256(_mm_castps_si128(_mm_loadu_ps((const float*)(src_x + 0 * src_w_step))));
                auto S1 = _mm256_castsi128_si256(_mm_castps_si128(_mm_loadu_ps((const float*)(src_x + 0 * src_w_step + 8))));
                auto S2 = _mm256_castsi128_si256(_mm_castps_si128(_mm_loadu_ps((const float*)(src_x + 1 * src_w_step))));
                auto S3 = _mm256_castsi128_si256(_mm_castps_si128(_mm_loadu_ps((const float*)(src_x + 1 * src_w_step + 8))));
                const auto weight_x = weight_y + PACK_UNIT * fx;
                auto W0 = _mm256_castsi128_si256(_mm_castps_si128(_mm_loadu_ps((const float*)weight_x)));
                auto W1 = _mm256_castsi128_si256(_mm_castps_si128(_mm_loadu_ps((const float*)(weight_x + 8))));
                auto s00 = _mm256_unpacklo_epi16(S0, zero);
                auto s10 = _mm256_unpacklo_epi16(S1, zero);
                auto s20 = _mm256_unpacklo_epi16(S2, zero);
                auto s30 = _mm256_unpacklo_epi16(S3, zero);
                auto s01 = _mm256_unpackhi_epi16(S0, zero);
                auto s11 = _mm256_unpackhi_epi16(S1, zero);
                auto s21 = _mm256_unpackhi_epi16(S2, zero);
                auto s31 = _mm256_unpackhi_epi16(S3, zero);
                auto w00 = _mm256_unpacklo_epi16(W0, zero);
                auto w01 = _mm256_unpackhi_epi16(W0, zero);
                auto w10 = _mm256_unpacklo_epi16(W1, zero);
                auto w11 = _mm256_unpackhi_epi16(W1, zero);

                S0 = _mm256_permute2f128_si256(s00, s01, 32);
                S1 = _mm256_permute2f128_si256(s10, s11, 32);
                S2 = _mm256_permute2f128_si256(s20, s21, 32);
                S3 = _mm256_permute2f128_si256(s30, s31, 32);
                W0 = _mm256_permute2f128_si256(w00, w01, 32);
                W1 = _mm256_permute2f128_si256(w10, w11, 32);
                d0 = _mm256_add_epi32(d0, _mm256_madd_epi16(W0, S0));
                d1 = _mm256_add_epi32(d1, _mm256_madd_epi16(W1, S1));
                d2 = _mm256_add_epi32(d2, _mm256_madd_epi16(W0, S2));
                d3 = _mm256_add_epi32(d3, _mm256_madd_epi16(W1, S3));
            }
        }
        __m256 f0 = _mm256_cvtepi32_ps(d0);
        __m256 f1 = _mm256_cvtepi32_ps(d1);
        __m256 f2 = _mm256_cvtepi32_ps(d2);
        __m256 f3 = _mm256_cvtepi32_ps(d3);
        f0 = _mm256_mul_ps(f0, scaleValue0);
        f1 = _mm256_mul_ps(f1, scaleValue1);
        f2 = _mm256_mul_ps(f2, scaleValue0);
        f3 = _mm256_mul_ps(f3, scaleValue1);
        auto m0 = _mm256_cmp_ps(f0, zero128, 1);
        auto m1 = _mm256_cmp_ps(f1, zero128, 1);
        auto m2 = _mm256_cmp_ps(f2, zero128, 1);
        auto m3 = _mm256_cmp_ps(f3, zero128, 1);
        m0 = _mm256_blendv_ps(plus, minus, m0);
        m1 = _mm256_blendv_ps(plus, minus, m1);
        m2 = _mm256_blendv_ps(plus, minus, m2);
        m3 = _mm256_blendv_ps(plus, minus, m3);
        f0 = _mm256_add_ps(f0, m0);
        f1 = _mm256_add_ps(f1, m1);
        f2 = _mm256_add_ps(f2, m2);
        f3 = _mm256_add_ps(f3, m3);
        // 3: _MM_FROUND_TO_ZERO
        d0 = _mm256_cvtps_epi32(_mm256_round_ps(f0, 3));
        d1 = _mm256_cvtps_epi32(_mm256_round_ps(f1, 3));
        d2 = _mm256_cvtps_epi32(_mm256_round_ps(f2, 3));
        d3 = _mm256_cvtps_epi32(_mm256_round_ps(f3, 3));
        auto offset = _mm256_set1_epi32(128);
        d0 = _mm256_add_epi32(d0, offset);
        d1 = _mm256_add_epi32(d1, offset);
        d2 = _mm256_add_epi32(d2, offset);
        d3 = _mm256_add_epi32(d3, offset);

        auto e0 = _mm256_permute2f128_si256(d0, d1, 32);
        auto e1 = _mm256_permute2f128_si256(d0, d1, 49);
        auto e2 = _mm256_permute2f128_si256(d2, d3, 32);
        auto e3 = _mm256_permute2f128_si256(d2, d3, 49);
        // Int32 -> Int8
        d0 = _mm256_packs_epi32(e0, e1);
        d2 = _mm256_packs_epi32(e2, e3);

        auto D0 = _mm256_extracti128_si256(d0, 0);
        auto D1 = _mm256_extracti128_si256(d0, 1);
        auto D2 = _mm256_extracti128_si256(d2, 0);
        auto D3 = _mm256_extracti128_si256(d2, 1);

        D0 = _mm_min_epi16(D0, maxValue);
        D1 = _mm_min_epi16(D1, maxValue);
        D0 = _mm_max_epi16(D0, minValue);
        D1 = _mm_max_epi16(D1, minValue);

        D2 = _mm_min_epi16(D2, maxValue);
        D3 = _mm_min_epi16(D3, maxValue);
        D2 = _mm_max_epi16(D2, minValue);
        D3 = _mm_max_epi16(D3, minValue);
        _mm_storeu_ps((float*)(dst), _mm_castsi128_ps(_mm_packus_epi16(D0, D1)));
        _mm_storeu_ps((float*)(dst + 16), _mm_castsi128_ps(_mm_packus_epi16(D2, D3)));
        dst += 32;
        src += src_w_step * 2;
    }
    if (widthRemain > 0) {
        d0 = biasValue0;
        d1 = biasValue1;

        auto dst_x          = dst;
        const auto src_z    = src;
        for (fy = 0; fy < fh; ++fy) {
            const auto src_y    = src_z + fy * dilateY_step;
            const auto weight_y = weight + fy * fw * PACK_UNIT;
            for (fx = 0; fx < fw; ++fx) {
                const auto src_x    = src_y + fx * dilateX_step;
                auto S0 = _mm256_castsi128_si256(_mm_castps_si128(_mm_loadu_ps((const float*)(src_x + 0 * src_w_step))));
                auto S1 = _mm256_castsi128_si256(_mm_castps_si128(_mm_loadu_ps((const float*)(src_x + 0 * src_w_step + 8))));
                const auto weight_x = weight_y + PACK_UNIT * fx;
                auto W0 = _mm256_castsi128_si256(_mm_castps_si128(_mm_loadu_ps((const float*)weight_x)));
                auto W1 = _mm256_castsi128_si256(_mm_castps_si128(_mm_loadu_ps((const float*)(weight_x + 8))));
                auto s00 = _mm256_unpacklo_epi16(S0, zero);
                auto s10 = _mm256_unpacklo_epi16(S1, zero);
                auto s01 = _mm256_unpackhi_epi16(S0, zero);
                auto s11 = _mm256_unpackhi_epi16(S1, zero);
                auto w00 = _mm256_unpacklo_epi16(W0, zero);
                auto w01 = _mm256_unpackhi_epi16(W0, zero);
                auto w10 = _mm256_unpacklo_epi16(W1, zero);
                auto w11 = _mm256_unpackhi_epi16(W1, zero);
                S0 = _mm256_permute2f128_si256(s00, s01, 32);
                S1 = _mm256_permute2f128_si256(s10, s11, 32);
                W0 = _mm256_permute2f128_si256(w00, w01, 32);
                W1 = _mm256_permute2f128_si256(w10, w11, 32);
                d0 = _mm256_add_epi32(d0, _mm256_madd_epi16(W0, S0));
                d1 = _mm256_add_epi32(d1, _mm256_madd_epi16(W1, S1));
            }
        }
        __m256 f0 = _mm256_cvtepi32_ps(d0);
        __m256 f1 = _mm256_cvtepi32_ps(d1);
        f0 = _mm256_mul_ps(f0, scaleValue0);
        f1 = _mm256_mul_ps(f1, scaleValue1);
        auto m0 = _mm256_cmp_ps(f0, zero128, 1);
        auto m1 = _mm256_cmp_ps(f1, zero128, 1);
        m0 = _mm256_blendv_ps(plus, minus, m0);
        m1 = _mm256_blendv_ps(plus, minus, m1);
        f0 = _mm256_add_ps(f0, m0);
        f1 = _mm256_add_ps(f1, m1);
        // 3: _MM_FROUND_TO_ZERO
        d0 = _mm256_cvtps_epi32(_mm256_round_ps(f0, 3));
        d1 = _mm256_cvtps_epi32(_mm256_round_ps(f1, 3));

        auto offset = _mm256_set1_epi32(128);
        d0 = _mm256_add_epi32(d0, offset);
        d1 = _mm256_add_epi32(d1, offset);

        auto e0 = _mm256_permute2f128_si256(d0, d1, 32);
        auto e1 = _mm256_permute2f128_si256(d0, d1, 49);
        // Int32 -> Int8
        d0 = _mm256_packs_epi32(e0, e1);
        auto D0 = _mm256_extracti128_si256(d0, 0);
        auto D1 = _mm256_extracti128_si256(d0, 1);

        D0 = _mm_min_epi16(D0, maxValue);
        D1 = _mm_min_epi16(D1, maxValue);
        D0 = _mm_max_epi16(D0, minValue);
        D1 = _mm_max_epi16(D1, minValue);

        _mm_storeu_ps((float*)(dst), _mm_castsi128_ps(_mm_packus_epi16(D0, D1)));
    }
}
#endif

#undef _MM256_SET_M128I
