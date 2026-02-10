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

#define MUL_WEIGHT_SCALE(N, P) \
    auto f##N = _mm512_cvtepi32_ps(D##N);\
    f##N = _mm512_mul_ps(f##N, scaleValue##P);

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
    // common
    int dzUnit = GEMMINT8_AVX512_H / PACK_UNIT;
    int dzU = dst_depth_quad / dzUnit;
    int dzR = dst_depth_quad % dzUnit;
    const float* biasPtr = nullptr;
    const float* bias_dz = nullptr;
    if (post->biasFloat) {
        biasPtr = post->biasFloat;
    }
    // int8 output relevant.
    auto zero512 = _mm512_set1_ps(0.0f);
    auto minValue = _mm512_set1_ps(post->minValue);
    auto maxValue = _mm512_set1_ps(post->maxValue);
    auto plus = _mm512_set1_ps(0.5f);
    auto minus = _mm512_set1_ps(-0.5f);
    auto offset = _mm256_set1_epi16(128);
    // float outout relevant
    auto neg128f   = _mm512_set1_ps(-128.f);
    __m512 bias00, bias10, bias20, bias30, bias01, bias02, bias03, bias11, bias12, bias13, bias21, bias22, bias23, bias31, bias32, bias33;
    __m512 fp32min, fp32max;
    if (0 == post->useInt8 && post->fp32minmax) {
        fp32min = _mm512_set1_ps((post->fp32minmax)[0]);
        fp32max = _mm512_set1_ps((post->fp32minmax)[1]);
    }
    auto blockNum = post->blockNum;
    const float* weightKernelSum_dz = nullptr;
    auto accumbuff = post->accumBuffer;

    __m512 kernelSum0, kernelSum1, kernelSum2, kernelSum3;
    __m512 inputbias0, inputbias1, inputbias2, inputbias3;
    __m512 inputscale0, inputscale1, inputscale2, inputscale3;

    if (post->inputScale) {
        inputscale0 = _mm512_set1_ps(post->inputScale[0]);
        if (realDst > 1) {
            inputscale1 = _mm512_set1_ps(post->inputScale[1]);    
        }
        if (realDst > 2) {
            inputscale2 = _mm512_set1_ps(post->inputScale[2]);
        }
        if (realDst > 3) {
            inputscale3 = _mm512_set1_ps(post->inputScale[3]);
        }
    }

    int weight_step_Z = static_cast<int32_t>(src_depth_quad * (GEMMINT8_AVX512_L * GEMMINT8_AVX512_H)) + (2 * sizeof(float) * GEMMINT8_AVX512_H);
    int weight_step_Y = static_cast<int32_t>(GEMMINT8_AVX512_L * GEMMINT8_AVX512_H);
    int weightPackStride = GEMMINT8_AVX512_L * PACK_UNIT;
    int source_step = realDst * PACK_UNIT;

    if (realDst == GEMMINT8_AVX512_E) {
        for (int dz = 0; dz < dzU; ++dz) {
            if (biasPtr) {
                bias_dz = post->biasFloat + dz * PACK_UNIT * dzUnit;
            }
            auto dst_x = dst + dz * dst_step_tmp * dzUnit;
            auto accum_x = accumbuff;

            for (int bk = 0; bk < blockNum; ++bk) {
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

                // block's weight&scale&bias
                const auto weight_dz = weight + dz * blockNum * weight_step_Z + bk *  weight_step_Z;
                const float* scale_dz = reinterpret_cast<const float*>(weight_dz + src_depth_quad * weight_step_Y);
                const auto weightBias_dz = scale_dz + GEMMINT8_AVX512_H;
                // block's input
                const auto src_x = src + bk * src_depth_quad * GEMMINT8_AVX512_L * realDst;
                for (int sz = 0; sz < src_depth_quad; ++sz) {
                    const auto weight_sz = weight_dz + weight_step_Y * sz;
                    const auto src_z     = (const float*)(src_x + sz * realDst * GEMMINT8_AVX512_L);
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
                // int32_t -> float
                auto scaleValue0 = _mm512_loadu_ps(scale_dz);
                auto scaleValue1 = _mm512_loadu_ps(scale_dz + 1 * PACK_UNIT);
                auto scaleValue2 = _mm512_loadu_ps(scale_dz + 2 * PACK_UNIT);
                auto scaleValue3 = _mm512_loadu_ps(scale_dz + 3 * PACK_UNIT);
                auto weightBiasValue0 = _mm512_loadu_ps(weightBias_dz);
                auto weightBiasValue1 = _mm512_loadu_ps(weightBias_dz + 1 * PACK_UNIT);
                auto weightBiasValue2 = _mm512_loadu_ps(weightBias_dz + 2 * PACK_UNIT);
                auto weightBiasValue3 = _mm512_loadu_ps(weightBias_dz + 3 * PACK_UNIT);
                // input info
                kernelSum0 = _mm512_set1_ps((post->srcKernelSum + bk * realDst)[0]);
                kernelSum1 = _mm512_set1_ps((post->srcKernelSum + bk * realDst)[1]);
                kernelSum2 = _mm512_set1_ps((post->srcKernelSum + bk * realDst)[2]);
                kernelSum3 = _mm512_set1_ps((post->srcKernelSum + bk * realDst)[3]);
                if (post->inputBias) {
                    inputscale0 = _mm512_set1_ps((post->inputScale + bk * realDst)[0]);
                    inputscale1 = _mm512_set1_ps((post->inputScale + bk * realDst)[1]);
                    inputscale2 = _mm512_set1_ps((post->inputScale + bk * realDst)[2]);
                    inputscale3 = _mm512_set1_ps((post->inputScale + bk * realDst)[3]);
                    inputbias0 = _mm512_set1_ps((post->inputBias + bk * realDst)[0]);
                    inputbias1 = _mm512_set1_ps((post->inputBias + bk * realDst)[1]);
                    inputbias2 = _mm512_set1_ps((post->inputBias + bk * realDst)[2]);
                    inputbias3 = _mm512_set1_ps((post->inputBias + bk * realDst)[3]);
                }

                MUL_WEIGHT_SCALE(0, 0);
                MUL_WEIGHT_SCALE(1, 0);
                MUL_WEIGHT_SCALE(2, 0);
                MUL_WEIGHT_SCALE(3, 0);
                MUL_WEIGHT_SCALE(4, 1);
                MUL_WEIGHT_SCALE(5, 1);
                MUL_WEIGHT_SCALE(6, 1);
                MUL_WEIGHT_SCALE(7, 1);
                MUL_WEIGHT_SCALE(8, 2);
                MUL_WEIGHT_SCALE(9, 2);
                MUL_WEIGHT_SCALE(10, 2);
                MUL_WEIGHT_SCALE(11, 2);
                MUL_WEIGHT_SCALE(12, 3);
                MUL_WEIGHT_SCALE(13, 3);
                MUL_WEIGHT_SCALE(14, 3);
                MUL_WEIGHT_SCALE(15, 3);

                if (post->inputScale) { // Batch quant
                    f0 = _mm512_mul_ps(f0, inputscale0);
                    f1 = _mm512_mul_ps(f1, inputscale1);
                    f2 = _mm512_mul_ps(f2, inputscale2);
                    f3 = _mm512_mul_ps(f3, inputscale3);
                    f4 = _mm512_mul_ps(f4, inputscale0);
                    f5 = _mm512_mul_ps(f5, inputscale1);
                    f6 = _mm512_mul_ps(f6, inputscale2);
                    f7 = _mm512_mul_ps(f7, inputscale3);
                    f8 = _mm512_mul_ps(f8, inputscale0);
                    f9 = _mm512_mul_ps(f9, inputscale1);
                    f10 = _mm512_mul_ps(f10, inputscale2);
                    f11 = _mm512_mul_ps(f11, inputscale3);
                    f12 = _mm512_mul_ps(f12, inputscale0);
                    f13 = _mm512_mul_ps(f13, inputscale1);
                    f14 = _mm512_mul_ps(f14, inputscale2);
                    f15 = _mm512_mul_ps(f15, inputscale3);
                    if ((post->useInt8 == 0) && post->weightKernelSum && (post->inputBias || (bk == 0))) {
                        if (post->inputBias) {
                            weightKernelSum_dz = post->weightKernelSum + bk * GEMMINT8_AVX512_H + dz * blockNum * GEMMINT8_AVX512_H;
                            auto wsum0 = _mm512_loadu_ps(weightKernelSum_dz);
                            auto wsum1 = _mm512_loadu_ps(weightKernelSum_dz + 1 * PACK_UNIT);
                            auto wsum2 = _mm512_loadu_ps(weightKernelSum_dz + 2 * PACK_UNIT);
                            auto wsum3 = _mm512_loadu_ps(weightKernelSum_dz + 3 * PACK_UNIT);
                            bias00 = _mm512_mul_ps(inputbias0, wsum0);
                            bias01 = _mm512_mul_ps(inputbias1, wsum0);
                            bias02 = _mm512_mul_ps(inputbias2, wsum0);
                            bias03 = _mm512_mul_ps(inputbias3, wsum0);
                            bias10 = _mm512_mul_ps(inputbias0, wsum1);
                            bias11 = _mm512_mul_ps(inputbias1, wsum1);
                            bias12 = _mm512_mul_ps(inputbias2, wsum1);
                            bias13 = _mm512_mul_ps(inputbias3, wsum1);
                            bias20 = _mm512_mul_ps(inputbias0, wsum2);
                            bias21 = _mm512_mul_ps(inputbias1, wsum2);
                            bias22 = _mm512_mul_ps(inputbias2, wsum2);
                            bias23 = _mm512_mul_ps(inputbias3, wsum2);
                            bias30 = _mm512_mul_ps(inputbias0, wsum3);
                            bias31 = _mm512_mul_ps(inputbias1, wsum3);
                            bias32 = _mm512_mul_ps(inputbias2, wsum3);
                            bias33 = _mm512_mul_ps(inputbias3, wsum3);
                        } else if (bk == 0) { // if input not block quant, only accum once!
                            weightKernelSum_dz = post->weightKernelSum + dz * GEMMINT8_AVX512_H;
                            auto wsum0 = _mm512_loadu_ps(weightKernelSum_dz);
                            auto wsum1 = _mm512_loadu_ps(weightKernelSum_dz + 1 * PACK_UNIT);
                            auto wsum2 = _mm512_loadu_ps(weightKernelSum_dz + 2 * PACK_UNIT);
                            auto wsum3 = _mm512_loadu_ps(weightKernelSum_dz + 3 * PACK_UNIT);
                            bias00 = _mm512_mul_ps(_mm512_mul_ps(inputscale0, neg128f), wsum0);
                            bias01 = _mm512_mul_ps(_mm512_mul_ps(inputscale1, neg128f), wsum0);
                            bias02 = _mm512_mul_ps(_mm512_mul_ps(inputscale2, neg128f), wsum0);
                            bias03 = _mm512_mul_ps(_mm512_mul_ps(inputscale3, neg128f), wsum0);
                            bias10 = _mm512_mul_ps(_mm512_mul_ps(inputscale0, neg128f), wsum1);
                            bias11 = _mm512_mul_ps(_mm512_mul_ps(inputscale1, neg128f), wsum1);
                            bias12 = _mm512_mul_ps(_mm512_mul_ps(inputscale2, neg128f), wsum1);
                            bias13 = _mm512_mul_ps(_mm512_mul_ps(inputscale3, neg128f), wsum1);
                            bias20 = _mm512_mul_ps(_mm512_mul_ps(inputscale0, neg128f), wsum2);
                            bias21 = _mm512_mul_ps(_mm512_mul_ps(inputscale1, neg128f), wsum2);
                            bias22 = _mm512_mul_ps(_mm512_mul_ps(inputscale2, neg128f), wsum2);
                            bias23 = _mm512_mul_ps(_mm512_mul_ps(inputscale3, neg128f), wsum2);
                            bias30 = _mm512_mul_ps(_mm512_mul_ps(inputscale0, neg128f), wsum3);
                            bias31 = _mm512_mul_ps(_mm512_mul_ps(inputscale1, neg128f), wsum3);
                            bias32 = _mm512_mul_ps(_mm512_mul_ps(inputscale2, neg128f), wsum3);
                            bias33 = _mm512_mul_ps(_mm512_mul_ps(inputscale3, neg128f), wsum3);
                        }
                        f0 = _mm512_add_ps(f0, bias00);
                        f1 = _mm512_add_ps(f1, bias01);
                        f2 = _mm512_add_ps(f2, bias02);
                        f3 = _mm512_add_ps(f3, bias03);
                        f4 = _mm512_add_ps(f4, bias10);
                        f5 = _mm512_add_ps(f5, bias11);
                        f6 = _mm512_add_ps(f6, bias12);
                        f7 = _mm512_add_ps(f7, bias13);
                        f8 = _mm512_add_ps(f8, bias20);
                        f9 = _mm512_add_ps(f9, bias21);
                        f10 = _mm512_add_ps(f10, bias22);
                        f11 = _mm512_add_ps(f11, bias23);
                        f12 = _mm512_add_ps(f12, bias30);
                        f13 = _mm512_add_ps(f13, bias31);
                        f14 = _mm512_add_ps(f14, bias32);
                        f15 = _mm512_add_ps(f15, bias33);
                    }
                }
                f0 = _mm512_add_ps(_mm512_mul_ps(kernelSum0, weightBiasValue0), f0);
                f1 = _mm512_add_ps(_mm512_mul_ps(kernelSum1, weightBiasValue0), f1);
                f2 = _mm512_add_ps(_mm512_mul_ps(kernelSum2, weightBiasValue0), f2);
                f3 = _mm512_add_ps(_mm512_mul_ps(kernelSum3, weightBiasValue0), f3);
                f4 = _mm512_add_ps(_mm512_mul_ps(kernelSum0, weightBiasValue1), f4);
                f5 = _mm512_add_ps(_mm512_mul_ps(kernelSum1, weightBiasValue1), f5);
                f6 = _mm512_add_ps(_mm512_mul_ps(kernelSum2, weightBiasValue1), f6);
                f7 = _mm512_add_ps(_mm512_mul_ps(kernelSum3, weightBiasValue1), f7);
                f8 = _mm512_add_ps(_mm512_mul_ps(kernelSum0, weightBiasValue2), f8);
                f9 = _mm512_add_ps(_mm512_mul_ps(kernelSum1, weightBiasValue2), f9);
                f10 = _mm512_add_ps(_mm512_mul_ps(kernelSum2, weightBiasValue2),f10);
                f11 = _mm512_add_ps(_mm512_mul_ps(kernelSum3, weightBiasValue2),f11);
                f12 = _mm512_add_ps(_mm512_mul_ps(kernelSum0, weightBiasValue3),f12);
                f13 = _mm512_add_ps(_mm512_mul_ps(kernelSum1, weightBiasValue3),f13);
                f14 = _mm512_add_ps(_mm512_mul_ps(kernelSum2, weightBiasValue3),f14);
                f15 = _mm512_add_ps(_mm512_mul_ps(kernelSum3, weightBiasValue3),f15);

                if (post->useInt8 == 1) {
                    if (biasPtr) {
                        auto biasValue0 = _mm512_loadu_ps(bias_dz);
                        auto biasValue4 = _mm512_loadu_ps(bias_dz + 1 * PACK_UNIT);
                        auto biasValue8 = _mm512_loadu_ps(bias_dz + 2 * PACK_UNIT);
                        auto biasValue12 = _mm512_loadu_ps(bias_dz + 3 * PACK_UNIT);
                        f0 = _mm512_add_ps(f0, biasValue0);
                        f1 = _mm512_add_ps(f1, biasValue0);
                        f2 = _mm512_add_ps(f2, biasValue0);
                        f3 = _mm512_add_ps(f3, biasValue0);
                        f4 = _mm512_add_ps(f4, biasValue4);
                        f5 = _mm512_add_ps(f5, biasValue4);
                        f6 = _mm512_add_ps(f6, biasValue4);
                        f7 = _mm512_add_ps(f7, biasValue4);
                        f8 = _mm512_add_ps(f8, biasValue8);
                        f9 = _mm512_add_ps(f9, biasValue8);
                        f10 = _mm512_add_ps(f10, biasValue8);
                        f11 = _mm512_add_ps(f11, biasValue8);
                        f12 = _mm512_add_ps(f12, biasValue12);
                        f13 = _mm512_add_ps(f13, biasValue12);
                        f14 = _mm512_add_ps(f14, biasValue12);
                        f15 = _mm512_add_ps(f15, biasValue12);
                    }
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
                    continue;
                }
                if (bk > 0) {
                    f0 = _mm512_add_ps(_mm512_loadu_ps(accum_x), f0);
                    f1 = _mm512_add_ps(_mm512_loadu_ps(accum_x + 16), f1);
                    f2 = _mm512_add_ps(_mm512_loadu_ps(accum_x + 16 * 2), f2);
                    f3 = _mm512_add_ps(_mm512_loadu_ps(accum_x + 16 * 3), f3);

                    f4 = _mm512_add_ps(_mm512_loadu_ps(accum_x + source_step), f4);
                    f5 = _mm512_add_ps(_mm512_loadu_ps(accum_x + source_step + 16 * 1), f5);
                    f6 = _mm512_add_ps(_mm512_loadu_ps(accum_x + source_step + 16 * 2), f6);
                    f7 = _mm512_add_ps(_mm512_loadu_ps(accum_x + source_step + 16 * 3), f7);

                    f8 = _mm512_add_ps(_mm512_loadu_ps(accum_x + 2 * source_step), f8);
                    f9 = _mm512_add_ps(_mm512_loadu_ps(accum_x + 2 * source_step + 16 * 1), f9);
                    f10 = _mm512_add_ps(_mm512_loadu_ps(accum_x + 2 * source_step + 16 * 2), f10);
                    f11 = _mm512_add_ps(_mm512_loadu_ps(accum_x + 2 * source_step + 16 * 3), f11);

                    f12 = _mm512_add_ps(_mm512_loadu_ps(accum_x + 3 * source_step), f12);
                    f13 = _mm512_add_ps(_mm512_loadu_ps(accum_x + 3 * source_step + 16 * 1), f13);
                    f14 = _mm512_add_ps(_mm512_loadu_ps(accum_x + 3 * source_step + 16 * 2), f14);
                    f15 = _mm512_add_ps(_mm512_loadu_ps(accum_x + 3 * source_step + 16 * 3), f15);
                }
                if (bk == blockNum - 1) {
                    if (biasPtr) {
                        auto biasValue0 = _mm512_loadu_ps(bias_dz);
                        auto biasValue4 = _mm512_loadu_ps(bias_dz + 1 * PACK_UNIT);
                        auto biasValue8 = _mm512_loadu_ps(bias_dz + 2 * PACK_UNIT);
                        auto biasValue12 = _mm512_loadu_ps(bias_dz + 3 * PACK_UNIT);
                        f0 = _mm512_add_ps(f0, biasValue0);
                        f1 = _mm512_add_ps(f1, biasValue0);
                        f2 = _mm512_add_ps(f2, biasValue0);
                        f3 = _mm512_add_ps(f3, biasValue0);
                        f4 = _mm512_add_ps(f4, biasValue4);
                        f5 = _mm512_add_ps(f5, biasValue4);
                        f6 = _mm512_add_ps(f6, biasValue4);
                        f7 = _mm512_add_ps(f7, biasValue4);
                        f8 = _mm512_add_ps(f8, biasValue8);
                        f9 = _mm512_add_ps(f9, biasValue8);
                        f10 = _mm512_add_ps(f10, biasValue8);
                        f11 = _mm512_add_ps(f11, biasValue8);
                        f12 = _mm512_add_ps(f12, biasValue12);
                        f13 = _mm512_add_ps(f13, biasValue12);
                        f14 = _mm512_add_ps(f14, biasValue12);
                        f15 = _mm512_add_ps(f15, biasValue12);
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
                    _mm512_storeu_ps(accum_x, f0);
                    _mm512_storeu_ps(accum_x + 16, f1);
                    _mm512_storeu_ps(accum_x + 16 * 2, f2);
                    _mm512_storeu_ps(accum_x + 16 * 3, f3);
                    _mm512_storeu_ps(accum_x + source_step, f4);
                    _mm512_storeu_ps(accum_x + source_step + 16 * 1, f5);
                    _mm512_storeu_ps(accum_x + source_step + 16 * 2, f6);
                    _mm512_storeu_ps(accum_x + source_step + 16 * 3, f7);
                    _mm512_storeu_ps(accum_x + 2 * source_step, f8);
                    _mm512_storeu_ps(accum_x + 2 * source_step + 16 * 1, f9);
                    _mm512_storeu_ps(accum_x + 2 * source_step + 16 * 2, f10);
                    _mm512_storeu_ps(accum_x + 2 * source_step + 16 * 3, f11);
                    _mm512_storeu_ps(accum_x + 3 * source_step, f12);
                    _mm512_storeu_ps(accum_x + 3 * source_step + 16 * 1, f13);
                    _mm512_storeu_ps(accum_x + 3 * source_step + 16 * 2, f14);
                    _mm512_storeu_ps(accum_x + 3 * source_step + 16 * 3, f15);
                }
            }
        } // dzU
        // the remaining ocDivPack
        if (dzR == 0) {
            return;
        }
        auto weight_dz = weight + dzU * blockNum * weight_step_Z;
        if (biasPtr) {
            bias_dz = post->biasFloat + dzU * PACK_UNIT * dzUnit;
        }

        auto dst_x = dst + dzU * dst_step_tmp * dzUnit;
        for (int i=0; i<dzR; ++i) {
            auto accum_x = accumbuff;
            for (int bk = 0; bk < blockNum; ++bk) {
                __m512i D0 = _mm512_set1_epi32(0);
                __m512i D1 = _mm512_set1_epi32(0);
                __m512i D2 = _mm512_set1_epi32(0);
                __m512i D3 = _mm512_set1_epi32(0);
                auto weightDzSub = weight_dz + bk * weight_step_Z + i * weightPackStride;
                auto scaleDz = (float*)(weight_dz + bk * weight_step_Z + src_depth_quad * weight_step_Y);
                 auto biasDz = scaleDz + GEMMINT8_AVX512_H;
                const auto src_x = src + bk * src_depth_quad * GEMMINT8_AVX512_L * realDst;

                for (int sz = 0; sz < src_depth_quad; ++sz) {
                    const auto weight_sz = weightDzSub + weight_step_Y * sz;
                    const auto src_z     = (const float*)(src_x + sz * realDst * GEMMINT8_AVX512_L);
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

                auto scaleValue0 = _mm512_loadu_ps(scaleDz + i * PACK_UNIT);
                auto weightBiasValue0 = _mm512_loadu_ps(biasDz + i * PACK_UNIT);
                MUL_WEIGHT_SCALE(0, 0);
                MUL_WEIGHT_SCALE(1, 0);
                MUL_WEIGHT_SCALE(2, 0);
                MUL_WEIGHT_SCALE(3, 0);

                // input info
                kernelSum0 = _mm512_set1_ps((post->srcKernelSum + bk * realDst)[0]);
                kernelSum1 = _mm512_set1_ps((post->srcKernelSum + bk * realDst)[1]);
                kernelSum2 = _mm512_set1_ps((post->srcKernelSum + bk * realDst)[2]);
                kernelSum3 = _mm512_set1_ps((post->srcKernelSum + bk * realDst)[3]);
                if (post->inputBias) {
                    inputscale0 = _mm512_set1_ps((post->inputScale + bk * realDst)[0]);
                    inputscale1 = _mm512_set1_ps((post->inputScale + bk * realDst)[1]);
                    inputscale2 = _mm512_set1_ps((post->inputScale + bk * realDst)[2]);
                    inputscale3 = _mm512_set1_ps((post->inputScale + bk * realDst)[3]);
                    inputbias0 = _mm512_set1_ps((post->inputBias + bk * realDst)[0]);
                    inputbias1 = _mm512_set1_ps((post->inputBias + bk * realDst)[1]);
                    inputbias2 = _mm512_set1_ps((post->inputBias + bk * realDst)[2]);
                    inputbias3 = _mm512_set1_ps((post->inputBias + bk * realDst)[3]);
                }
                if (post->inputScale) { // Batch quant
                    f0 = _mm512_mul_ps(f0, inputscale0);
                    f1 = _mm512_mul_ps(f1, inputscale1);
                    f2 = _mm512_mul_ps(f2, inputscale2);
                    f3 = _mm512_mul_ps(f3, inputscale3);

                    if ((post->useInt8 == 0) && post->weightKernelSum && (post->inputBias || (bk == 0))) {
                        if (post->inputBias) {
                            weightKernelSum_dz = post->weightKernelSum + dzU * blockNum * GEMMINT8_AVX512_H + bk * GEMMINT8_AVX512_H;
                            auto wsum0 = _mm512_loadu_ps(weightKernelSum_dz + i * PACK_UNIT);
                            bias00 = _mm512_mul_ps(inputbias0, wsum0);
                            bias01 = _mm512_mul_ps(inputbias1, wsum0);
                            bias02 = _mm512_mul_ps(inputbias2, wsum0);
                            bias03 = _mm512_mul_ps(inputbias3, wsum0);
                        } else if (bk == 0) { // if input not block quant, only accum once!
                            weightKernelSum_dz = post->weightKernelSum + dzU * PACK_UNIT * dzUnit;
                            auto wsum0 = _mm512_loadu_ps(weightKernelSum_dz + i * PACK_UNIT);
                            bias00 = _mm512_mul_ps(_mm512_mul_ps(inputscale0, neg128f), wsum0);
                            bias01 = _mm512_mul_ps(_mm512_mul_ps(inputscale1, neg128f), wsum0);
                            bias02 = _mm512_mul_ps(_mm512_mul_ps(inputscale2, neg128f), wsum0);
                            bias03 = _mm512_mul_ps(_mm512_mul_ps(inputscale3, neg128f), wsum0);
                        }
                        f0 = _mm512_add_ps(f0, bias00);
                        f1 = _mm512_add_ps(f1, bias01);
                        f2 = _mm512_add_ps(f2, bias02);
                        f3 = _mm512_add_ps(f3, bias03);
                    }
                }
                f0 = _mm512_add_ps(_mm512_mul_ps(kernelSum0, weightBiasValue0), f0);
                f1 = _mm512_add_ps(_mm512_mul_ps(kernelSum1, weightBiasValue0), f1);
                f2 = _mm512_add_ps(_mm512_mul_ps(kernelSum2, weightBiasValue0), f2);
                f3 = _mm512_add_ps(_mm512_mul_ps(kernelSum3, weightBiasValue0), f3);
                
                if (post->useInt8 == 1) {
                    if (nullptr != biasPtr) {
                        auto biasValue0 = _mm512_loadu_ps(bias_dz + i * PACK_UNIT);
                        f0 = _mm512_add_ps(f0, biasValue0);
                        f1 = _mm512_add_ps(f1, biasValue0);
                        f2 = _mm512_add_ps(f2, biasValue0);
                        f3 = _mm512_add_ps(f3, biasValue0);
                    }
                    POSTTREAT(0, 0);
                    POSTTREAT(1, 1);
                    POSTTREAT(2, 2);
                    POSTTREAT(3, 3);
                    dst_x += dst_step_tmp;
                    continue;
                }
                if (bk > 0) {
                    f0 = _mm512_add_ps(_mm512_loadu_ps((float*)accum_x), f0);
                    f1 = _mm512_add_ps(_mm512_loadu_ps(((float*)accum_x) + 16), f1);
                    f2 = _mm512_add_ps(_mm512_loadu_ps(((float*)accum_x) + 16 * 2), f2);
                    f3 = _mm512_add_ps(_mm512_loadu_ps(((float*)accum_x) + 16 * 3), f3);
                }
                if (bk == blockNum - 1) {
                    if (nullptr != biasPtr) {
                        auto biasValue0 = _mm512_loadu_ps(bias_dz + i * PACK_UNIT);
                        f0 = _mm512_add_ps(f0, biasValue0);
                        f1 = _mm512_add_ps(f1, biasValue0);
                        f2 = _mm512_add_ps(f2, biasValue0);
                        f3 = _mm512_add_ps(f3, biasValue0);
                    }
                    if (post->fp32minmax) {
                        POST_TREAT_FLOAT(0,1,2,3);
                    }
                    _mm512_storeu_ps((float*)(dst_x + i * dst_step_tmp), f0);
                    _mm512_storeu_ps((float*)(dst_x + i * dst_step_tmp) + 16, f1);
                    _mm512_storeu_ps((float*)(dst_x + i * dst_step_tmp) + 16 * 2, f2);
                    _mm512_storeu_ps((float*)(dst_x + i * dst_step_tmp) + 16 * 3, f3);
                } else {
                    _mm512_storeu_ps(((float*)accum_x), f0);
                    _mm512_storeu_ps(((float*)accum_x) + 16, f1);
                    _mm512_storeu_ps(((float*)accum_x) + 16 * 2, f2);
                    _mm512_storeu_ps(((float*)accum_x) + 16 * 3, f3);
                }
            }
        }
        return;
    }
    
    if (realDst == 3) {
        for (int dz = 0; dz < dzU; ++dz) {
            if (biasPtr) {
                bias_dz = post->biasFloat + dz * PACK_UNIT * dzUnit;
            }
            auto dst_x = dst + dz * dst_step_tmp * dzUnit;
            auto accum_x = accumbuff;

            for (int bk = 0; bk < blockNum; ++bk) {
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

                // block's weight&scale&bias
                const auto weight_dz = weight + dz * blockNum * weight_step_Z + bk *  weight_step_Z;
                const float* scale_dz = reinterpret_cast<const float*>(weight_dz + src_depth_quad * weight_step_Y);
                const auto weightBias_dz = scale_dz + GEMMINT8_AVX512_H;
                // block's input
                const auto src_x = src + bk * src_depth_quad * GEMMINT8_AVX512_L * realDst;
                for (int sz = 0; sz < src_depth_quad; ++sz) {
                    const auto weight_sz = weight_dz + weight_step_Y * sz;
                    const auto src_z     = (const float*)(src_x + sz * realDst * GEMMINT8_AVX512_L);

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
                // int32_t -> float
                auto scaleValue0 = _mm512_loadu_ps(scale_dz);
                auto scaleValue1 = _mm512_loadu_ps(scale_dz + 1 * PACK_UNIT);
                auto scaleValue2 = _mm512_loadu_ps(scale_dz + 2 * PACK_UNIT);
                auto scaleValue3 = _mm512_loadu_ps(scale_dz + 3 * PACK_UNIT);
                auto weightBiasValue0 = _mm512_loadu_ps(weightBias_dz);
                auto weightBiasValue1 = _mm512_loadu_ps(weightBias_dz + 1 * PACK_UNIT);
                auto weightBiasValue2 = _mm512_loadu_ps(weightBias_dz + 2 * PACK_UNIT);
                auto weightBiasValue3 = _mm512_loadu_ps(weightBias_dz + 3 * PACK_UNIT);

                // input info
                kernelSum0 = _mm512_set1_ps((post->srcKernelSum + bk * realDst)[0]);
                kernelSum1 = _mm512_set1_ps((post->srcKernelSum + bk * realDst)[1]);
                kernelSum2 = _mm512_set1_ps((post->srcKernelSum + bk * realDst)[2]);
                if (post->inputBias) {
                    inputscale0 = _mm512_set1_ps((post->inputScale + bk * realDst)[0]);
                    inputscale1 = _mm512_set1_ps((post->inputScale + bk * realDst)[1]);
                    inputscale2 = _mm512_set1_ps((post->inputScale + bk * realDst)[2]);
                    inputbias0 = _mm512_set1_ps((post->inputBias + bk * realDst)[0]);
                    inputbias1 = _mm512_set1_ps((post->inputBias + bk * realDst)[1]);
                    inputbias2 = _mm512_set1_ps((post->inputBias + bk * realDst)[2]);
                }

                MUL_WEIGHT_SCALE(0, 0);
                MUL_WEIGHT_SCALE(1, 0);
                MUL_WEIGHT_SCALE(2, 0);
                MUL_WEIGHT_SCALE(4, 1);
                MUL_WEIGHT_SCALE(5, 1);
                MUL_WEIGHT_SCALE(6, 1);
                MUL_WEIGHT_SCALE(8, 2);
                MUL_WEIGHT_SCALE(9, 2);
                MUL_WEIGHT_SCALE(10, 2);
                MUL_WEIGHT_SCALE(12, 3);
                MUL_WEIGHT_SCALE(13, 3);
                MUL_WEIGHT_SCALE(14, 3);

                if (post->inputScale) { // Batch quant
                    f0 = _mm512_mul_ps(f0, inputscale0);
                    f1 = _mm512_mul_ps(f1, inputscale1);
                    f2 = _mm512_mul_ps(f2, inputscale2);
                    f4 = _mm512_mul_ps(f4, inputscale0);
                    f5 = _mm512_mul_ps(f5, inputscale1);
                    f6 = _mm512_mul_ps(f6, inputscale2);
                    f8 = _mm512_mul_ps(f8, inputscale0);
                    f9 = _mm512_mul_ps(f9, inputscale1);
                    f10 = _mm512_mul_ps(f10, inputscale2);
                    f12 = _mm512_mul_ps(f12, inputscale0);
                    f13 = _mm512_mul_ps(f13, inputscale1);
                    f14 = _mm512_mul_ps(f14, inputscale2);
                    if ((post->useInt8 == 0) && post->weightKernelSum && (post->inputBias || (bk == 0))) {
                        if (post->inputBias) {
                            weightKernelSum_dz = post->weightKernelSum + bk * GEMMINT8_AVX512_H + dz * blockNum * GEMMINT8_AVX512_H;
                            auto wsum0 = _mm512_loadu_ps(weightKernelSum_dz);
                            auto wsum1 = _mm512_loadu_ps(weightKernelSum_dz + 1 * PACK_UNIT);
                            auto wsum2 = _mm512_loadu_ps(weightKernelSum_dz + 2 * PACK_UNIT);
                            auto wsum3 = _mm512_loadu_ps(weightKernelSum_dz + 3 * PACK_UNIT);
                            bias00 = _mm512_mul_ps(inputbias0, wsum0);
                            bias01 = _mm512_mul_ps(inputbias1, wsum0);
                            bias02 = _mm512_mul_ps(inputbias2, wsum0);
                            bias10 = _mm512_mul_ps(inputbias0, wsum1);
                            bias11 = _mm512_mul_ps(inputbias1, wsum1);
                            bias12 = _mm512_mul_ps(inputbias2, wsum1);
                            bias20 = _mm512_mul_ps(inputbias0, wsum2);
                            bias21 = _mm512_mul_ps(inputbias1, wsum2);
                            bias22 = _mm512_mul_ps(inputbias2, wsum2);
                            bias30 = _mm512_mul_ps(inputbias0, wsum3);
                            bias31 = _mm512_mul_ps(inputbias1, wsum3);
                            bias32 = _mm512_mul_ps(inputbias2, wsum3);
                        } else if (bk == 0) { // if input not block quant, only accum once!
                            weightKernelSum_dz = post->weightKernelSum + dz * GEMMINT8_AVX512_H;
                            auto wsum0 = _mm512_loadu_ps(weightKernelSum_dz);
                            auto wsum1 = _mm512_loadu_ps(weightKernelSum_dz + 1 * PACK_UNIT);
                            auto wsum2 = _mm512_loadu_ps(weightKernelSum_dz + 2 * PACK_UNIT);
                            auto wsum3 = _mm512_loadu_ps(weightKernelSum_dz + 3 * PACK_UNIT);
                            bias00 = _mm512_mul_ps(_mm512_mul_ps(inputscale0, neg128f), wsum0);
                            bias01 = _mm512_mul_ps(_mm512_mul_ps(inputscale1, neg128f), wsum0);
                            bias02 = _mm512_mul_ps(_mm512_mul_ps(inputscale2, neg128f), wsum0);
                            bias10 = _mm512_mul_ps(_mm512_mul_ps(inputscale0, neg128f), wsum1);
                            bias11 = _mm512_mul_ps(_mm512_mul_ps(inputscale1, neg128f), wsum1);
                            bias12 = _mm512_mul_ps(_mm512_mul_ps(inputscale2, neg128f), wsum1);
                            bias20 = _mm512_mul_ps(_mm512_mul_ps(inputscale0, neg128f), wsum2);
                            bias21 = _mm512_mul_ps(_mm512_mul_ps(inputscale1, neg128f), wsum2);
                            bias22 = _mm512_mul_ps(_mm512_mul_ps(inputscale2, neg128f), wsum2);
                            bias30 = _mm512_mul_ps(_mm512_mul_ps(inputscale0, neg128f), wsum3);
                            bias31 = _mm512_mul_ps(_mm512_mul_ps(inputscale1, neg128f), wsum3);
                            bias32 = _mm512_mul_ps(_mm512_mul_ps(inputscale2, neg128f), wsum3);
                        }
                        f0 = _mm512_add_ps(f0, bias00);
                        f1 = _mm512_add_ps(f1, bias01);
                        f2 = _mm512_add_ps(f2, bias02);
                        f4 = _mm512_add_ps(f4, bias10);
                        f5 = _mm512_add_ps(f5, bias11);
                        f6 = _mm512_add_ps(f6, bias12);
                        f8 = _mm512_add_ps(f8, bias20);
                        f9 = _mm512_add_ps(f9, bias21);
                        f10 = _mm512_add_ps(f10, bias22);
                        f12 = _mm512_add_ps(f12, bias30);
                        f13 = _mm512_add_ps(f13, bias31);
                        f14 = _mm512_add_ps(f14, bias32);
                    }
                }
                f0 = _mm512_add_ps(_mm512_mul_ps(kernelSum0, weightBiasValue0), f0);
                f1 = _mm512_add_ps(_mm512_mul_ps(kernelSum1, weightBiasValue0), f1);
                f2 = _mm512_add_ps(_mm512_mul_ps(kernelSum2, weightBiasValue0), f2);
                f4 = _mm512_add_ps(_mm512_mul_ps(kernelSum0, weightBiasValue1), f4);
                f5 = _mm512_add_ps(_mm512_mul_ps(kernelSum1, weightBiasValue1), f5);
                f6 = _mm512_add_ps(_mm512_mul_ps(kernelSum2, weightBiasValue1), f6);
                f8 = _mm512_add_ps(_mm512_mul_ps(kernelSum0, weightBiasValue2), f8);
                f9 = _mm512_add_ps(_mm512_mul_ps(kernelSum1, weightBiasValue2), f9);
                f10 = _mm512_add_ps(_mm512_mul_ps(kernelSum2, weightBiasValue2),f10);
                f12 = _mm512_add_ps(_mm512_mul_ps(kernelSum0, weightBiasValue3),f12);
                f13 = _mm512_add_ps(_mm512_mul_ps(kernelSum1, weightBiasValue3),f13);
                f14 = _mm512_add_ps(_mm512_mul_ps(kernelSum2, weightBiasValue3),f14);

                if (post->useInt8 == 1) {
                    if (biasPtr) {
                        auto biasValue0 = _mm512_loadu_ps(bias_dz);
                        auto biasValue4 = _mm512_loadu_ps(bias_dz + 1 * PACK_UNIT);
                        auto biasValue8 = _mm512_loadu_ps(bias_dz + 2 * PACK_UNIT);
                        auto biasValue12 = _mm512_loadu_ps(bias_dz + 3 * PACK_UNIT);
                        f0 = _mm512_add_ps(f0, biasValue0);
                        f1 = _mm512_add_ps(f1, biasValue0);
                        f2 = _mm512_add_ps(f2, biasValue0);
                        f4 = _mm512_add_ps(f4, biasValue4);
                        f5 = _mm512_add_ps(f5, biasValue4);
                        f6 = _mm512_add_ps(f6, biasValue4);
                        f8 = _mm512_add_ps(f8, biasValue8);
                        f9 = _mm512_add_ps(f9, biasValue8);
                        f10 = _mm512_add_ps(f10, biasValue8);
                        f12 = _mm512_add_ps(f12, biasValue12);
                        f13 = _mm512_add_ps(f13, biasValue12);
                        f14 = _mm512_add_ps(f14, biasValue12);
                    }
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
                    continue;
                }
                if (bk > 0) {
                    f0 = _mm512_add_ps(_mm512_loadu_ps(accum_x), f0);
                    f1 = _mm512_add_ps(_mm512_loadu_ps(accum_x + 16), f1);
                    f2 = _mm512_add_ps(_mm512_loadu_ps(accum_x + 16 * 2), f2);

                    f4 = _mm512_add_ps(_mm512_loadu_ps(accum_x + source_step), f4);
                    f5 = _mm512_add_ps(_mm512_loadu_ps(accum_x + source_step + 16 * 1), f5);
                    f6 = _mm512_add_ps(_mm512_loadu_ps(accum_x + source_step + 16 * 2), f6);

                    f8 = _mm512_add_ps(_mm512_loadu_ps(accum_x + 2 * source_step), f8);
                    f9 = _mm512_add_ps(_mm512_loadu_ps(accum_x + 2 * source_step + 16 * 1), f9);
                    f10 = _mm512_add_ps(_mm512_loadu_ps(accum_x + 2 * source_step + 16 * 2), f10);

                    f12 = _mm512_add_ps(_mm512_loadu_ps(accum_x + 3 * source_step), f12);
                    f13 = _mm512_add_ps(_mm512_loadu_ps(accum_x + 3 * source_step + 16 * 1), f13);
                    f14 = _mm512_add_ps(_mm512_loadu_ps(accum_x + 3 * source_step + 16 * 2), f14);
                }
                if (bk == blockNum - 1) {
                    if (biasPtr) {
                        auto biasValue0 = _mm512_loadu_ps(bias_dz);
                        auto biasValue4 = _mm512_loadu_ps(bias_dz + 1 * PACK_UNIT);
                        auto biasValue8 = _mm512_loadu_ps(bias_dz + 2 * PACK_UNIT);
                        auto biasValue12 = _mm512_loadu_ps(bias_dz + 3 * PACK_UNIT);
                        f0 = _mm512_add_ps(f0, biasValue0);
                        f1 = _mm512_add_ps(f1, biasValue0);
                        f2 = _mm512_add_ps(f2, biasValue0);
                        f4 = _mm512_add_ps(f4, biasValue4);
                        f5 = _mm512_add_ps(f5, biasValue4);
                        f6 = _mm512_add_ps(f6, biasValue4);
                        f8 = _mm512_add_ps(f8, biasValue8);
                        f9 = _mm512_add_ps(f9, biasValue8);
                        f10 = _mm512_add_ps(f10, biasValue8);
                        f12 = _mm512_add_ps(f12, biasValue12);
                        f13 = _mm512_add_ps(f13, biasValue12);
                        f14 = _mm512_add_ps(f14, biasValue12);
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
                    _mm512_storeu_ps(accum_x, f0);
                    _mm512_storeu_ps(accum_x + 16, f1);
                    _mm512_storeu_ps(accum_x + 16 * 2, f2);
                    _mm512_storeu_ps(accum_x + source_step, f4);
                    _mm512_storeu_ps(accum_x + source_step + 16 * 1, f5);
                    _mm512_storeu_ps(accum_x + source_step + 16 * 2, f6);
                    _mm512_storeu_ps(accum_x + 2 * source_step, f8);
                    _mm512_storeu_ps(accum_x + 2 * source_step + 16 * 1, f9);
                    _mm512_storeu_ps(accum_x + 2 * source_step + 16 * 2, f10);
                    _mm512_storeu_ps(accum_x + 3 * source_step, f12);
                    _mm512_storeu_ps(accum_x + 3 * source_step + 16 * 1, f13);
                    _mm512_storeu_ps(accum_x + 3 * source_step + 16 * 2, f14);
                }
            }
        } // dzU
        // the remaining ocDivPack
        auto weight_dz = weight + dzU * blockNum * weight_step_Z;                                            // weight address for remaining
        if (biasPtr) {
            bias_dz = post->biasFloat + dzU * PACK_UNIT * dzUnit;
        }

        auto dst_x = dst + dzU * dst_step_tmp * dzUnit;
        for (int i=0; i<dzR; ++i) {
            auto accum_x = accumbuff;
            for (int bk = 0; bk < blockNum; ++bk) {
                __m512i D0 = _mm512_set1_epi32(0);
                __m512i D1 = _mm512_set1_epi32(0);
                __m512i D2 = _mm512_set1_epi32(0);
                auto weightDzSub = weight_dz + bk * weight_step_Z + i * weightPackStride;
                auto scaleDz = (float*)(weight_dz + bk * weight_step_Z + src_depth_quad * weight_step_Y);
                 auto biasDz = scaleDz + GEMMINT8_AVX512_H;
                const auto src_x = src + bk * src_depth_quad * GEMMINT8_AVX512_L * realDst;

                for (int sz = 0; sz < src_depth_quad; ++sz) {
                    const auto weight_sz = weightDzSub + weight_step_Y * sz;
                    const auto src_z     = (const float*)(src_x + sz * realDst * GEMMINT8_AVX512_L);
                    auto w0 = _mm512_loadu_si512(weight_sz);

                    auto s0 = AVX512_BROADCAST_INT32(src_z + 0);
                    auto s1 = AVX512_BROADCAST_INT32(src_z + 1);
                    auto s2 = AVX512_BROADCAST_INT32(src_z + 2);

                    D0 = _mm512_dpbusds_epi32(D0, s0, w0);
                    D1 = _mm512_dpbusds_epi32(D1, s1, w0);
                    D2 = _mm512_dpbusds_epi32(D2, s2, w0);
                }

                auto scaleValue0 = _mm512_loadu_ps(scaleDz + i * PACK_UNIT);
                auto weightBiasValue0 = _mm512_loadu_ps(biasDz + i * PACK_UNIT);
                // input info
                kernelSum0 = _mm512_set1_ps((post->srcKernelSum + bk * realDst)[0]);
                kernelSum1 = _mm512_set1_ps((post->srcKernelSum + bk * realDst)[1]);
                kernelSum2 = _mm512_set1_ps((post->srcKernelSum + bk * realDst)[2]);
                if (post->inputBias) {
                    inputscale0 = _mm512_set1_ps((post->inputScale + bk * realDst)[0]);
                    inputscale1 = _mm512_set1_ps((post->inputScale + bk * realDst)[1]);
                    inputscale2 = _mm512_set1_ps((post->inputScale + bk * realDst)[2]);
                    inputbias0 = _mm512_set1_ps((post->inputBias + bk * realDst)[0]);
                    inputbias1 = _mm512_set1_ps((post->inputBias + bk * realDst)[1]);
                    inputbias2 = _mm512_set1_ps((post->inputBias + bk * realDst)[2]);
                }
                MUL_WEIGHT_SCALE(0, 0);
                MUL_WEIGHT_SCALE(1, 0);
                MUL_WEIGHT_SCALE(2, 0);

                if (post->inputScale) { // Batch quant
                    f0 = _mm512_mul_ps(f0, inputscale0);
                    f1 = _mm512_mul_ps(f1, inputscale1);
                    f2 = _mm512_mul_ps(f2, inputscale2);
                    if ((post->useInt8 == 0) && post->weightKernelSum && (post->inputBias || (bk == 0))) {
                        if (post->inputBias) {
                            weightKernelSum_dz = post->weightKernelSum + dzU * blockNum * GEMMINT8_AVX512_H + bk * GEMMINT8_AVX512_H;
                            auto wsum0 = _mm512_loadu_ps(weightKernelSum_dz + i * PACK_UNIT);
                            bias00 = _mm512_mul_ps(inputbias0, wsum0);
                            bias01 = _mm512_mul_ps(inputbias1, wsum0);
                            bias02 = _mm512_mul_ps(inputbias2, wsum0);
                        } else if (bk == 0) { // if input not block quant, only accum once!
                            weightKernelSum_dz = post->weightKernelSum + dzU * PACK_UNIT * dzUnit;
                            auto wsum0 = _mm512_loadu_ps(weightKernelSum_dz + i * PACK_UNIT);
                            bias00 = _mm512_mul_ps(_mm512_mul_ps(inputscale0, neg128f), wsum0);
                            bias01 = _mm512_mul_ps(_mm512_mul_ps(inputscale1, neg128f), wsum0);
                            bias02 = _mm512_mul_ps(_mm512_mul_ps(inputscale2, neg128f), wsum0);
                        }
                        f0 = _mm512_add_ps(f0, bias00);
                        f1 = _mm512_add_ps(f1, bias01);
                        f2 = _mm512_add_ps(f2, bias02);
                    }
                }
                f0 = _mm512_add_ps(_mm512_mul_ps(kernelSum0, weightBiasValue0), f0);
                f1 = _mm512_add_ps(_mm512_mul_ps(kernelSum1, weightBiasValue0), f1);
                f2 = _mm512_add_ps(_mm512_mul_ps(kernelSum2, weightBiasValue0), f2);

                if (post->useInt8 == 1) {
                    if (nullptr != biasPtr) {
                        auto biasValue0 = _mm512_loadu_ps(bias_dz + i * PACK_UNIT);
                        f0 = _mm512_add_ps(f0, biasValue0);
                        f1 = _mm512_add_ps(f1, biasValue0);
                        f2 = _mm512_add_ps(f2, biasValue0);
                    }
                    POSTTREAT(0, 0);
                    POSTTREAT(1, 1);
                    POSTTREAT(2, 2);
                    dst_x += dst_step_tmp;
                    continue;
                }
                if (bk > 0) {
                    f0 = _mm512_add_ps(_mm512_loadu_ps((float*)accum_x), f0);
                    f1 = _mm512_add_ps(_mm512_loadu_ps(((float*)accum_x) + 16), f1);
                    f2 = _mm512_add_ps(_mm512_loadu_ps(((float*)accum_x) + 16 * 2), f2);
                }
                if (bk == blockNum - 1) {
                    if (nullptr != biasPtr) {
                        auto biasValue0 = _mm512_loadu_ps(bias_dz + i * PACK_UNIT);
                        f0 = _mm512_add_ps(f0, biasValue0);
                        f1 = _mm512_add_ps(f1, biasValue0);
                        f2 = _mm512_add_ps(f2, biasValue0);
                    }
                    if (post->fp32minmax) {
                        POST_TREAT_FLOAT_3(0,1,2);
                    }
                    _mm512_storeu_ps((float*)(dst_x + i * dst_step_tmp), f0);
                    _mm512_storeu_ps((float*)(dst_x + i * dst_step_tmp) + 16, f1);
                    _mm512_storeu_ps((float*)(dst_x + i * dst_step_tmp) + 16 * 2, f2);
                } else {
                    _mm512_storeu_ps(((float*)accum_x), f0);
                    _mm512_storeu_ps(((float*)accum_x) + 16, f1);
                    _mm512_storeu_ps(((float*)accum_x) + 16 * 2, f2);
                }
            }
        }
        return;
    }

    if (realDst == 2) {
        for (int dz = 0; dz < dzU; ++dz) {
            if (biasPtr) {
                bias_dz = post->biasFloat + dz * PACK_UNIT * dzUnit;
            }
            auto dst_x = dst + dz * dst_step_tmp * dzUnit;
            auto accum_x = accumbuff;

            for (int bk = 0; bk < blockNum; ++bk) {
                __m512i D0 = _mm512_set1_epi32(0);
                __m512i D1 = _mm512_set1_epi32(0);

                __m512i D4 = _mm512_set1_epi32(0);
                __m512i D5 = _mm512_set1_epi32(0);

                __m512i D8 = _mm512_set1_epi32(0);
                __m512i D9 = _mm512_set1_epi32(0);

                __m512i D12 = _mm512_set1_epi32(0);
                __m512i D13 = _mm512_set1_epi32(0);

                // block's weight&scale&bias
                const auto weight_dz = weight + dz * blockNum * weight_step_Z + bk *  weight_step_Z;
                const float* scale_dz = reinterpret_cast<const float*>(weight_dz + src_depth_quad * weight_step_Y);
                const auto weightBias_dz = scale_dz + GEMMINT8_AVX512_H;
                // block's input
                const auto src_x = src + bk * src_depth_quad * GEMMINT8_AVX512_L * realDst;
                for (int sz = 0; sz < src_depth_quad; ++sz) {
                    const auto weight_sz = weight_dz + weight_step_Y * sz;
                    const auto src_z     = (const float*)(src_x + sz * realDst * GEMMINT8_AVX512_L);
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
                // int32_t -> float
                auto scaleValue0 = _mm512_loadu_ps(scale_dz);
                auto scaleValue1 = _mm512_loadu_ps(scale_dz + 1 * PACK_UNIT);
                auto scaleValue2 = _mm512_loadu_ps(scale_dz + 2 * PACK_UNIT);
                auto scaleValue3 = _mm512_loadu_ps(scale_dz + 3 * PACK_UNIT);
                auto weightBiasValue0 = _mm512_loadu_ps(weightBias_dz);
                auto weightBiasValue1 = _mm512_loadu_ps(weightBias_dz + 1 * PACK_UNIT);
                auto weightBiasValue2 = _mm512_loadu_ps(weightBias_dz + 2 * PACK_UNIT);
                auto weightBiasValue3 = _mm512_loadu_ps(weightBias_dz + 3 * PACK_UNIT);
                // input info
                kernelSum0 = _mm512_set1_ps((post->srcKernelSum + bk * realDst)[0]);
                kernelSum1 = _mm512_set1_ps((post->srcKernelSum + bk * realDst)[1]);
                if (post->inputBias) {
                    inputscale0 = _mm512_set1_ps((post->inputScale + bk * realDst)[0]);
                    inputscale1 = _mm512_set1_ps((post->inputScale + bk * realDst)[1]);
                    inputbias0 = _mm512_set1_ps((post->inputBias + bk * realDst)[0]);
                    inputbias1 = _mm512_set1_ps((post->inputBias + bk * realDst)[1]);
                }

                MUL_WEIGHT_SCALE(0, 0);
                MUL_WEIGHT_SCALE(1, 0);
                MUL_WEIGHT_SCALE(4, 1);
                MUL_WEIGHT_SCALE(5, 1);
                MUL_WEIGHT_SCALE(8, 2);
                MUL_WEIGHT_SCALE(9, 2);
                MUL_WEIGHT_SCALE(12, 3);
                MUL_WEIGHT_SCALE(13, 3);

                if (post->inputScale) { // Batch quant
                    f0 = _mm512_mul_ps(f0, inputscale0);
                    f1 = _mm512_mul_ps(f1, inputscale1);
                    f4 = _mm512_mul_ps(f4, inputscale0);
                    f5 = _mm512_mul_ps(f5, inputscale1);
                    f8 = _mm512_mul_ps(f8, inputscale0);
                    f9 = _mm512_mul_ps(f9, inputscale1);
                    f12 = _mm512_mul_ps(f12, inputscale0);
                    f13 = _mm512_mul_ps(f13, inputscale1);
                    if ((post->useInt8 == 0) && post->weightKernelSum && (post->inputBias || (bk == 0))) {
                        if (post->inputBias) {
                            weightKernelSum_dz = post->weightKernelSum + bk * GEMMINT8_AVX512_H + dz * blockNum * GEMMINT8_AVX512_H;
                            auto wsum0 = _mm512_loadu_ps(weightKernelSum_dz);
                            auto wsum1 = _mm512_loadu_ps(weightKernelSum_dz + 1 * PACK_UNIT);
                            auto wsum2 = _mm512_loadu_ps(weightKernelSum_dz + 2 * PACK_UNIT);
                            auto wsum3 = _mm512_loadu_ps(weightKernelSum_dz + 3 * PACK_UNIT);
                            bias00 = _mm512_mul_ps(inputbias0, wsum0);
                            bias01 = _mm512_mul_ps(inputbias1, wsum0);
                            bias10 = _mm512_mul_ps(inputbias0, wsum1);
                            bias11 = _mm512_mul_ps(inputbias1, wsum1);
                            bias20 = _mm512_mul_ps(inputbias0, wsum2);
                            bias21 = _mm512_mul_ps(inputbias1, wsum2);
                            bias30 = _mm512_mul_ps(inputbias0, wsum3);
                            bias31 = _mm512_mul_ps(inputbias1, wsum3);
                        } else if (bk == 0) { // if input not block quant, only accum once!
                            weightKernelSum_dz = post->weightKernelSum + dz * GEMMINT8_AVX512_H;
                            auto wsum0 = _mm512_loadu_ps(weightKernelSum_dz);
                            auto wsum1 = _mm512_loadu_ps(weightKernelSum_dz + 1 * PACK_UNIT);
                            auto wsum2 = _mm512_loadu_ps(weightKernelSum_dz + 2 * PACK_UNIT);
                            auto wsum3 = _mm512_loadu_ps(weightKernelSum_dz + 3 * PACK_UNIT);
                            bias00 = _mm512_mul_ps(_mm512_mul_ps(inputscale0, neg128f), wsum0);
                            bias01 = _mm512_mul_ps(_mm512_mul_ps(inputscale1, neg128f), wsum0);
                            bias10 = _mm512_mul_ps(_mm512_mul_ps(inputscale0, neg128f), wsum1);
                            bias11 = _mm512_mul_ps(_mm512_mul_ps(inputscale1, neg128f), wsum1);
                            bias20 = _mm512_mul_ps(_mm512_mul_ps(inputscale0, neg128f), wsum2);
                            bias21 = _mm512_mul_ps(_mm512_mul_ps(inputscale1, neg128f), wsum2);
                            bias30 = _mm512_mul_ps(_mm512_mul_ps(inputscale0, neg128f), wsum3);
                            bias31 = _mm512_mul_ps(_mm512_mul_ps(inputscale1, neg128f), wsum3);
                        }
                        f0 = _mm512_add_ps(f0, bias00);
                        f1 = _mm512_add_ps(f1, bias01);
                        f4 = _mm512_add_ps(f4, bias10);
                        f5 = _mm512_add_ps(f5, bias11);
                        f8 = _mm512_add_ps(f8, bias20);
                        f9 = _mm512_add_ps(f9, bias21);
                        f12 = _mm512_add_ps(f12, bias30);
                        f13 = _mm512_add_ps(f13, bias31);
                    }
                }
                f0 = _mm512_add_ps(_mm512_mul_ps(kernelSum0, weightBiasValue0), f0);
                f1 = _mm512_add_ps(_mm512_mul_ps(kernelSum1, weightBiasValue0), f1);
                f4 = _mm512_add_ps(_mm512_mul_ps(kernelSum0, weightBiasValue1), f4);
                f5 = _mm512_add_ps(_mm512_mul_ps(kernelSum1, weightBiasValue1), f5);
                f8 = _mm512_add_ps(_mm512_mul_ps(kernelSum0, weightBiasValue2), f8);
                f9 = _mm512_add_ps(_mm512_mul_ps(kernelSum1, weightBiasValue2), f9);
                f12 = _mm512_add_ps(_mm512_mul_ps(kernelSum0, weightBiasValue3),f12);
                f13 = _mm512_add_ps(_mm512_mul_ps(kernelSum1, weightBiasValue3),f13);

                if (post->useInt8 == 1) {
                    if (biasPtr) {
                        auto biasValue0 = _mm512_loadu_ps(bias_dz);
                        auto biasValue4 = _mm512_loadu_ps(bias_dz + 1 * PACK_UNIT);
                        auto biasValue8 = _mm512_loadu_ps(bias_dz + 2 * PACK_UNIT);
                        auto biasValue12 = _mm512_loadu_ps(bias_dz + 3 * PACK_UNIT);
                        f0 = _mm512_add_ps(f0, biasValue0);
                        f1 = _mm512_add_ps(f1, biasValue0);
                        f4 = _mm512_add_ps(f4, biasValue4);
                        f5 = _mm512_add_ps(f5, biasValue4);
                        f8 = _mm512_add_ps(f8, biasValue8);
                        f9 = _mm512_add_ps(f9, biasValue8);
                        f12 = _mm512_add_ps(f12, biasValue12);
                        f13 = _mm512_add_ps(f13, biasValue12);
                    }
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
                    continue;
                }
                if (bk > 0) {
                    f0 = _mm512_add_ps(_mm512_loadu_ps(accum_x), f0);
                    f1 = _mm512_add_ps(_mm512_loadu_ps(accum_x + 16), f1);

                    f4 = _mm512_add_ps(_mm512_loadu_ps(accum_x + source_step), f4);
                    f5 = _mm512_add_ps(_mm512_loadu_ps(accum_x + source_step + 16 * 1), f5);

                    f8 = _mm512_add_ps(_mm512_loadu_ps(accum_x + 2 * source_step), f8);
                    f9 = _mm512_add_ps(_mm512_loadu_ps(accum_x + 2 * source_step + 16 * 1), f9);

                    f12 = _mm512_add_ps(_mm512_loadu_ps(accum_x + 3 * source_step), f12);
                    f13 = _mm512_add_ps(_mm512_loadu_ps(accum_x + 3 * source_step + 16 * 1), f13);
                }
                if (bk == blockNum - 1) {
                    if (biasPtr) {
                        auto biasValue0 = _mm512_loadu_ps(bias_dz);
                        auto biasValue4 = _mm512_loadu_ps(bias_dz + 1 * PACK_UNIT);
                        auto biasValue8 = _mm512_loadu_ps(bias_dz + 2 * PACK_UNIT);
                        auto biasValue12 = _mm512_loadu_ps(bias_dz + 3 * PACK_UNIT);
                        f0 = _mm512_add_ps(f0, biasValue0);
                        f1 = _mm512_add_ps(f1, biasValue0);
                        f4 = _mm512_add_ps(f4, biasValue4);
                        f5 = _mm512_add_ps(f5, biasValue4);
                        f8 = _mm512_add_ps(f8, biasValue8);
                        f9 = _mm512_add_ps(f9, biasValue8);
                        f12 = _mm512_add_ps(f12, biasValue12);
                        f13 = _mm512_add_ps(f13, biasValue12);
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
                    _mm512_storeu_ps(accum_x, f0);
                    _mm512_storeu_ps(accum_x + 16, f1);
                    _mm512_storeu_ps(accum_x + source_step, f4);
                    _mm512_storeu_ps(accum_x + source_step + 16 * 1, f5);
                    _mm512_storeu_ps(accum_x + 2 * source_step, f8);
                    _mm512_storeu_ps(accum_x + 2 * source_step + 16 * 1, f9);
                    _mm512_storeu_ps(accum_x + 3 * source_step, f12);
                    _mm512_storeu_ps(accum_x + 3 * source_step + 16 * 1, f13);
                }
            }
        } // dzU
        // the remaining ocDivPack
        auto weight_dz = weight + dzU * blockNum * weight_step_Z;                                            // weight address for remaining

        if (biasPtr) {
            bias_dz = post->biasFloat + dzU * PACK_UNIT * dzUnit;
        }

        auto dst_x = dst + dzU * dst_step_tmp * dzUnit;
        for (int i=0; i<dzR; ++i) {
            auto accum_x = accumbuff;
            for (int bk = 0; bk < blockNum; ++bk) {
                __m512i D0 = _mm512_set1_epi32(0);
                __m512i D1 = _mm512_set1_epi32(0);
                auto weightDzSub = weight_dz + bk * weight_step_Z + i * weightPackStride;
                auto scaleDz = (float*)(weight_dz + bk * weight_step_Z + src_depth_quad * weight_step_Y);
                 auto biasDz = scaleDz + GEMMINT8_AVX512_H;
                const auto src_x = src + bk * src_depth_quad * GEMMINT8_AVX512_L * realDst;

                for (int sz = 0; sz < src_depth_quad; ++sz) {
                    const auto weight_sz = weightDzSub + weight_step_Y * sz;
                    const auto src_z     = (const float*)(src_x + sz * realDst * GEMMINT8_AVX512_L);
                    auto w0 = _mm512_loadu_si512(weight_sz);

                    auto s0 = AVX512_BROADCAST_INT32(src_z + 0);
                    auto s1 = AVX512_BROADCAST_INT32(src_z + 1);

                    D0 = _mm512_dpbusds_epi32(D0, s0, w0);
                    D1 = _mm512_dpbusds_epi32(D1, s1, w0);
                }

                auto scaleValue0 = _mm512_loadu_ps(scaleDz + i * PACK_UNIT);
                auto weightBiasValue0 = _mm512_loadu_ps(biasDz + i * PACK_UNIT);
                // input info
                kernelSum0 = _mm512_set1_ps((post->srcKernelSum + bk * realDst)[0]);
                kernelSum1 = _mm512_set1_ps((post->srcKernelSum + bk * realDst)[1]);
                if (post->inputBias) {
                    inputscale0 = _mm512_set1_ps((post->inputScale + bk * realDst)[0]);
                    inputscale1 = _mm512_set1_ps((post->inputScale + bk * realDst)[1]);
                    inputbias0 = _mm512_set1_ps((post->inputBias + bk * realDst)[0]);
                    inputbias1 = _mm512_set1_ps((post->inputBias + bk * realDst)[1]);
                }
                MUL_WEIGHT_SCALE(0, 0);
                MUL_WEIGHT_SCALE(1, 0);

                if (post->inputScale) { // Batch quant
                    f0 = _mm512_mul_ps(f0, inputscale0);
                    f1 = _mm512_mul_ps(f1, inputscale1);
                    if ((post->useInt8 == 0) && post->weightKernelSum && (post->inputBias || (bk == 0))) {
                        if (post->inputBias) {
                            weightKernelSum_dz = post->weightKernelSum + dzU * blockNum * GEMMINT8_AVX512_H + bk * GEMMINT8_AVX512_H;
                            auto wsum0 = _mm512_loadu_ps(weightKernelSum_dz + i * PACK_UNIT);
                            bias00 = _mm512_mul_ps(inputbias0, wsum0);
                            bias01 = _mm512_mul_ps(inputbias1, wsum0);
                        } else if (bk == 0) { // if input not block quant, only accum once!
                            weightKernelSum_dz = post->weightKernelSum + dzU * PACK_UNIT * dzUnit;
                            auto wsum0 = _mm512_loadu_ps(weightKernelSum_dz + i * PACK_UNIT);
                            bias00 = _mm512_mul_ps(_mm512_mul_ps(inputscale0, neg128f), wsum0);
                            bias01 = _mm512_mul_ps(_mm512_mul_ps(inputscale1, neg128f), wsum0);
                        }
                        f0 = _mm512_add_ps(f0, bias00);
                        f1 = _mm512_add_ps(f1, bias01);
                    }
                }
                f0 = _mm512_add_ps(_mm512_mul_ps(kernelSum0, weightBiasValue0), f0);
                f1 = _mm512_add_ps(_mm512_mul_ps(kernelSum1, weightBiasValue0), f1);

                if (post->useInt8 == 1) {
                    if (nullptr != biasPtr) {
                        auto biasValue0 = _mm512_loadu_ps(bias_dz + i * PACK_UNIT);
                        f0 = _mm512_add_ps(f0, biasValue0);
                        f1 = _mm512_add_ps(f1, biasValue0);
                    }
                    POSTTREAT(0, 0);
                    POSTTREAT(1, 1);
                    dst_x += dst_step_tmp;
                    continue;
                }
                if (bk > 0) {
                    f0 = _mm512_add_ps(_mm512_loadu_ps((float*)accum_x), f0);
                    f1 = _mm512_add_ps(_mm512_loadu_ps(((float*)accum_x) + 16), f1);
                }
                if (bk == blockNum - 1) {
                    if (nullptr != biasPtr) {
                        auto biasValue0 = _mm512_loadu_ps(bias_dz + i * PACK_UNIT);
                        f0 = _mm512_add_ps(f0, biasValue0);
                        f1 = _mm512_add_ps(f1, biasValue0);
                    }
                    if (post->fp32minmax) {
                        POST_TREAT_FLOAT_2(0,1);
                    }
                    _mm512_storeu_ps((float*)(dst_x + i * dst_step_tmp), f0);
                    _mm512_storeu_ps((float*)(dst_x + i * dst_step_tmp) + 16, f1);
                } else {
                    _mm512_storeu_ps(((float*)accum_x), f0);
                    _mm512_storeu_ps(((float*)accum_x) + 16, f1);
                }
            }
        }
        return;
    }
    if (realDst == 1) {
        for (int dz = 0; dz < dzU; ++dz) {
            if (biasPtr) {
                bias_dz = post->biasFloat + dz * PACK_UNIT * dzUnit;
            }
            auto dst_x = dst + dz * dst_step_tmp * dzUnit;
            auto accum_x = accumbuff;

            for (int bk = 0; bk < blockNum; ++bk) {
                __m512i D0 = _mm512_set1_epi32(0);
                __m512i D4 = _mm512_set1_epi32(0);
                __m512i D8 = _mm512_set1_epi32(0);
                __m512i D12 = _mm512_set1_epi32(0);

                // block's weight&scale&bias
                const auto weight_dz = weight + dz * blockNum * weight_step_Z + bk *  weight_step_Z;
                const float* scale_dz = reinterpret_cast<const float*>(weight_dz + src_depth_quad * weight_step_Y);
                const auto weightBias_dz = scale_dz + GEMMINT8_AVX512_H;
                // block's input
                const auto src_x = src + bk * src_depth_quad * GEMMINT8_AVX512_L * realDst;

                for (int sz = 0; sz < src_depth_quad; ++sz) {
                    const auto weight_sz = weight_dz + weight_step_Y * sz;
                    const auto src_z     = (const float*)(src_x + sz * realDst * GEMMINT8_AVX512_L);
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
                // int32_t -> float
                auto scaleValue0 = _mm512_loadu_ps(scale_dz);
                auto scaleValue1 = _mm512_loadu_ps(scale_dz + 1 * PACK_UNIT);
                auto scaleValue2 = _mm512_loadu_ps(scale_dz + 2 * PACK_UNIT);
                auto scaleValue3 = _mm512_loadu_ps(scale_dz + 3 * PACK_UNIT);
                auto weightBiasValue0 = _mm512_loadu_ps(weightBias_dz);
                auto weightBiasValue1 = _mm512_loadu_ps(weightBias_dz + 1 * PACK_UNIT);
                auto weightBiasValue2 = _mm512_loadu_ps(weightBias_dz + 2 * PACK_UNIT);
                auto weightBiasValue3 = _mm512_loadu_ps(weightBias_dz + 3 * PACK_UNIT);
                // input info
                kernelSum0 = _mm512_set1_ps((post->srcKernelSum + bk * realDst)[0]);
                if (post->inputBias) {
                    inputscale0 = _mm512_set1_ps((post->inputScale + bk * realDst)[0]);
                    inputbias0 = _mm512_set1_ps((post->inputBias + bk * realDst)[0]);
                }

                MUL_WEIGHT_SCALE(0, 0);
                MUL_WEIGHT_SCALE(4, 1);
                MUL_WEIGHT_SCALE(8, 2);
                MUL_WEIGHT_SCALE(12, 3);

                if (post->inputScale) { // Batch quant
                    f0 = _mm512_mul_ps(f0, inputscale0);
                    f4 = _mm512_mul_ps(f4, inputscale0);
                    f8 = _mm512_mul_ps(f8, inputscale0);
                    f12 = _mm512_mul_ps(f12, inputscale0);
                    if ((post->useInt8 == 0) && post->weightKernelSum && (post->inputBias || (bk == 0))) {
                        if (post->inputBias) {
                            weightKernelSum_dz = post->weightKernelSum + bk * GEMMINT8_AVX512_H + dz * blockNum * GEMMINT8_AVX512_H;
                            auto wsum0 = _mm512_loadu_ps(weightKernelSum_dz);
                            auto wsum1 = _mm512_loadu_ps(weightKernelSum_dz + 1 * PACK_UNIT);
                            auto wsum2 = _mm512_loadu_ps(weightKernelSum_dz + 2 * PACK_UNIT);
                            auto wsum3 = _mm512_loadu_ps(weightKernelSum_dz + 3 * PACK_UNIT);
                            bias00 = _mm512_mul_ps(inputbias0, wsum0);
                            bias01 = _mm512_mul_ps(inputbias0, wsum1);
                            bias02 = _mm512_mul_ps(inputbias0, wsum2);
                            bias03 = _mm512_mul_ps(inputbias0, wsum3);
                        } else if (bk == 0) { // if input not block quant, only accum once!
                            weightKernelSum_dz = post->weightKernelSum + dz * GEMMINT8_AVX512_H;
                            auto wsum0 = _mm512_loadu_ps(weightKernelSum_dz);
                            auto wsum1 = _mm512_loadu_ps(weightKernelSum_dz + 1 * PACK_UNIT);
                            auto wsum2 = _mm512_loadu_ps(weightKernelSum_dz + 2 * PACK_UNIT);
                            auto wsum3 = _mm512_loadu_ps(weightKernelSum_dz + 3 * PACK_UNIT);
                            bias00 = _mm512_mul_ps(_mm512_mul_ps(inputscale0, neg128f), wsum0);
                            bias01 = _mm512_mul_ps(_mm512_mul_ps(inputscale0, neg128f), wsum1);
                            bias02 = _mm512_mul_ps(_mm512_mul_ps(inputscale0, neg128f), wsum2);
                            bias03 = _mm512_mul_ps(_mm512_mul_ps(inputscale0, neg128f), wsum3);
                        }
                        f0 = _mm512_add_ps(f0, bias00);
                        f4 = _mm512_add_ps(f4, bias01);
                        f8 = _mm512_add_ps(f8, bias02);
                        f12 = _mm512_add_ps(f12, bias03);
                    }
                }
                f0 = _mm512_add_ps(_mm512_mul_ps(kernelSum0, weightBiasValue0), f0);
                f4 = _mm512_add_ps(_mm512_mul_ps(kernelSum0, weightBiasValue1), f4);
                f8 = _mm512_add_ps(_mm512_mul_ps(kernelSum0, weightBiasValue2), f8);
                f12 = _mm512_add_ps(_mm512_mul_ps(kernelSum0, weightBiasValue3), f12);

                if (post->useInt8 == 1) {
                    if (biasPtr) {
                        auto biasValue0 = _mm512_loadu_ps(bias_dz);
                        auto biasValue4 = _mm512_loadu_ps(bias_dz + 1 * PACK_UNIT);
                        auto biasValue8 = _mm512_loadu_ps(bias_dz + 2 * PACK_UNIT);
                        auto biasValue12 = _mm512_loadu_ps(bias_dz + 3 * PACK_UNIT);
                        f0 = _mm512_add_ps(f0, biasValue0);
                        f4 = _mm512_add_ps(f4, biasValue4);
                        f8 = _mm512_add_ps(f8, biasValue8);
                        f12 = _mm512_add_ps(f12, biasValue12);
                    }
                    POSTTREAT(0, 0);
                    dst_x += dst_step_tmp;
    
                    POSTTREAT(4, 0);
                    dst_x += dst_step_tmp;
    
                    POSTTREAT(8, 0);
                    dst_x += dst_step_tmp;
    
                    POSTTREAT(12, 0);
                    continue;
                }
                if (bk > 0) { // Add accumbuffer if blockNum>1
                    f0 = _mm512_add_ps(_mm512_loadu_ps(accum_x), f0);
                    f4 = _mm512_add_ps(_mm512_loadu_ps(accum_x + source_step), f4);
                    f8 = _mm512_add_ps(_mm512_loadu_ps(accum_x + 2 * source_step), f8);
                    f12 = _mm512_add_ps(_mm512_loadu_ps(accum_x + 3 * source_step), f12);
                }
                if (bk == blockNum - 1) { // If last block, post process before saving to dest address.
                    if (biasPtr) {
                        auto biasValue0 = _mm512_loadu_ps(bias_dz);
                        auto biasValue4 = _mm512_loadu_ps(bias_dz + 1 * PACK_UNIT);
                        auto biasValue8 = _mm512_loadu_ps(bias_dz + 2 * PACK_UNIT);
                        auto biasValue12 = _mm512_loadu_ps(bias_dz + 3 * PACK_UNIT);
                        f0 = _mm512_add_ps(f0, biasValue0);
                        f4 = _mm512_add_ps(f4, biasValue4);
                        f8 = _mm512_add_ps(f8, biasValue8);
                        f12 = _mm512_add_ps(f12, biasValue12);
                    }
                    if (post->fp32minmax) {
                        POST_TREAT_FLOAT_1(0);
                        POST_TREAT_FLOAT_1(4);
                        POST_TREAT_FLOAT_1(8);
                        POST_TREAT_FLOAT_1(12);
                    }
                    _mm512_storeu_ps((float*)dst_x, f0);
                    _mm512_storeu_ps((float*)(dst_x + dst_step_tmp), f4);
                    _mm512_storeu_ps((float*)(dst_x + 2 * dst_step_tmp), f8);
                    _mm512_storeu_ps((float*)(dst_x + 3 * dst_step_tmp), f12);
                } else { // save to accumbuffer to added to next block
                    _mm512_storeu_ps(accum_x, f0);
                    _mm512_storeu_ps(accum_x + source_step, f4);
                    _mm512_storeu_ps(accum_x + 2 * source_step, f8);
                    _mm512_storeu_ps(accum_x + 3 * source_step, f12);
                }
            }
        }
        // the remaining ocDivPack
        auto weight_dz = weight + dzU * blockNum * weight_step_Z;                                            // weight address for remaining
        if (biasPtr) {
            bias_dz = post->biasFloat + dzU * PACK_UNIT * dzUnit;
        }

        auto dst_x = dst + dzU * dst_step_tmp * dzUnit;
        for (int i=0; i<dzR; ++i) {
            auto accum_x = accumbuff;
            for (int bk = 0; bk < blockNum; ++bk) {
                auto weightDzSub = weight_dz + bk * weight_step_Z + i * weightPackStride;
                auto scaleDz = (float*)(weight_dz + bk * weight_step_Z + src_depth_quad * weight_step_Y);
                 auto biasDz = scaleDz + GEMMINT8_AVX512_H;
                const auto src_x = src + bk * src_depth_quad * GEMMINT8_AVX512_L * realDst;

                __m512i D0 = _mm512_set1_epi32(0);

                for (int sz = 0; sz < src_depth_quad; ++sz) {
                    const auto weight_sz = weightDzSub + weight_step_Y * sz;
                    const auto src_z     = (const float*)(src_x + sz * realDst * GEMMINT8_AVX512_L);
                    auto w0 = _mm512_loadu_si512(weight_sz);
                    auto s0 = AVX512_BROADCAST_INT32(src_z + 0);
                    D0 = _mm512_dpbusds_epi32(D0, s0, w0);
                }

                auto scaleValue0 = _mm512_loadu_ps(scaleDz + i * PACK_UNIT);
                auto weightBiasValue0 = _mm512_loadu_ps(biasDz + i * PACK_UNIT);
                // input info
                kernelSum0 = _mm512_set1_ps((post->srcKernelSum + bk * realDst)[0]);
                if (post->inputBias) {
                    inputscale0 = _mm512_set1_ps((post->inputScale + bk * realDst)[0]);
                    inputbias0 = _mm512_set1_ps((post->inputBias + bk * realDst)[0]);
                }
                MUL_WEIGHT_SCALE(0, 0);

                if (post->inputScale) { // Batch quant
                    f0 = _mm512_mul_ps(f0, inputscale0);
                    if ((post->useInt8 == 0) && post->weightKernelSum && (post->inputBias || (bk == 0))) {
                        if (post->inputBias) {
                            weightKernelSum_dz = post->weightKernelSum + dzU * blockNum * GEMMINT8_AVX512_H + bk * GEMMINT8_AVX512_H;
                            auto wsum0 = _mm512_loadu_ps(weightKernelSum_dz + i * PACK_UNIT);
                            bias00 = _mm512_mul_ps(inputbias0, wsum0);
                        } else if (bk == 0) { // if input not block quant, only accum once!
                            weightKernelSum_dz = post->weightKernelSum + dzU * PACK_UNIT * dzUnit;
                            auto wsum0 = _mm512_loadu_ps(weightKernelSum_dz + i * PACK_UNIT);
                            bias00 = _mm512_mul_ps(_mm512_mul_ps(inputscale0, neg128f), wsum0);
                        }
                        f0 = _mm512_add_ps(f0, bias00);
                    }
                }
                f0 = _mm512_add_ps(_mm512_mul_ps(kernelSum0, weightBiasValue0), f0);

                if (post->useInt8 == 1) {
                    if (biasPtr) {
                        auto biasValue = _mm512_loadu_ps(bias_dz + i * PACK_UNIT);
                        SCALE_BIAS_VEC(0);
                    }
                    POSTTREAT(0, 0);
                    dst_x += dst_step_tmp;
                    continue;
                }
                if (bk > 0) {
                    f0 = _mm512_add_ps(_mm512_loadu_ps(accum_x), f0);
                }
                if (bk == blockNum - 1) {
                    if (biasPtr) {
                        auto biasValue = _mm512_loadu_ps(bias_dz + i * PACK_UNIT);
                        SCALE_BIAS_VEC(0);
                    }
                    if (post->fp32minmax) {
                        POST_TREAT_FLOAT_1(0);
                    }
                    _mm512_storeu_ps((float*)(dst_x + i * dst_step_tmp), f0);
                } else {
                    _mm512_storeu_ps(((float*)accum_x), f0);
                }
            }
        }
        return;
    }
}

// GemmInt8 with VNNI int4-weight fp32-output
#define SUB_ORDER {0,2,1,3}

void _AVX512_MNNGemmInt8AddBiasScale_16x4_w4_Unit_VNNI(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad, const QuanPostTreatParameters* post, size_t realDst) {
    MNN_ASSERT(post->useInt8 == 0);
    int suborder[4] = SUB_ORDER;
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
    const float* weightKernelSum_dz = nullptr;
    if (post->biasFloat) {
        biasPtr = post->biasFloat;
    }
    auto accumbuff = post->accumBuffer;
    __m512 kernelSum0, kernelSum1, kernelSum2, kernelSum3;
    __m512 inputbias0, inputbias1, inputbias2, inputbias3;
    __m512 inputscale0, inputscale1, inputscale2, inputscale3;
    if (post->inputScale) {
        inputscale0 = _mm512_set1_ps(post->inputScale[0]);
        if (realDst > 1) {
            inputscale1 = _mm512_set1_ps(post->inputScale[1]);    
        }
        if (realDst > 2) {
            inputscale2 = _mm512_set1_ps(post->inputScale[2]);
        }
        if (realDst > 3) {
            inputscale3 = _mm512_set1_ps(post->inputScale[3]);
        }
    }
    auto neg128f   = _mm512_set1_ps(-128.f);
    __m512 bias00, bias10, bias20, bias30, bias01, bias02, bias03, bias11, bias12, bias13, bias21, bias22, bias23, bias31, bias32, bias33;

    int weight_step_Y = GEMMINT8_AVX512_L * GEMMINT8_AVX512_H / 2;
    int weight_step_Z = src_depth_quad * weight_step_Y + (2 * 4 * GEMMINT8_AVX512_H);
    int weightPackStride = GEMMINT8_AVX512_L / 2 * PACK_UNIT;
    int source_step = realDst * PACK_UNIT;
    if (realDst == GEMMINT8_AVX512_E) {
        for (int dz = 0; dz < dzU; ++dz) {
            if (biasPtr) {
                bias_dz = post->biasFloat + dz * GEMMINT8_AVX512_H;
            }
            auto dst_x = dst + dz * dst_step_tmp * dzUnit;
            auto accum_x = accumbuff;

            for (int bk = 0; bk < blockNum; ++bk) {
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

                // block's weight&scale&bias
                const auto weight_dz = weight + dz * (blockNum * weight_step_Z) + bk *  weight_step_Z;
                const float* scale_dz = reinterpret_cast<const float*>(weight_dz + src_depth_quad * weight_step_Y);
                const auto weightBias_dz = scale_dz + GEMMINT8_AVX512_H;
                // block's input
                const auto src_x = src + bk * src_depth_quad * GEMMINT8_AVX512_L * realDst;
                for (int sz = 0; sz < src_depth_quad; ++sz) {
                    const auto weight_sz = weight_dz + weight_step_Y * sz;
                    const auto src_z     = (const float*)(src_x + sz * realDst * GEMMINT8_AVX512_L);

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
                // int32_t -> float
                auto scaleValue0 = _mm512_loadu_ps(scale_dz);
                auto scaleValue1 = _mm512_loadu_ps(scale_dz + 1 * PACK_UNIT);
                auto scaleValue2 = _mm512_loadu_ps(scale_dz + 2 * PACK_UNIT);
                auto scaleValue3 = _mm512_loadu_ps(scale_dz + 3 * PACK_UNIT);
                auto weightBiasValue0 = _mm512_loadu_ps(weightBias_dz);
                auto weightBiasValue1 = _mm512_loadu_ps(weightBias_dz + 1 * PACK_UNIT);
                auto weightBiasValue2 = _mm512_loadu_ps(weightBias_dz + 2 * PACK_UNIT);
                auto weightBiasValue3 = _mm512_loadu_ps(weightBias_dz + 3 * PACK_UNIT);
                // input info
                kernelSum0 = _mm512_set1_ps((post->srcKernelSum + bk * realDst)[0]);
                kernelSum1 = _mm512_set1_ps((post->srcKernelSum + bk * realDst)[1]);
                kernelSum2 = _mm512_set1_ps((post->srcKernelSum + bk * realDst)[2]);
                kernelSum3 = _mm512_set1_ps((post->srcKernelSum + bk * realDst)[3]);
                if (post->inputBias) {
                    inputscale0 = _mm512_set1_ps((post->inputScale + bk * realDst)[0]);
                    inputscale1= _mm512_set1_ps((post->inputScale + bk * realDst)[1]);
                    inputscale2 = _mm512_set1_ps((post->inputScale + bk * realDst)[2]);
                    inputscale3 = _mm512_set1_ps((post->inputScale + bk * realDst)[3]);
                    inputbias0 = _mm512_set1_ps((post->inputBias + bk * realDst)[0]);
                    inputbias1 = _mm512_set1_ps((post->inputBias + bk * realDst)[1]);
                    inputbias2 = _mm512_set1_ps((post->inputBias + bk * realDst)[2]);
                    inputbias3 = _mm512_set1_ps((post->inputBias + bk * realDst)[3]);
                }

                MUL_WEIGHT_SCALE(0, 0);
                MUL_WEIGHT_SCALE(1, 0);
                MUL_WEIGHT_SCALE(2, 0);
                MUL_WEIGHT_SCALE(3, 0);
                MUL_WEIGHT_SCALE(4, 1);
                MUL_WEIGHT_SCALE(5, 1);
                MUL_WEIGHT_SCALE(6, 1);
                MUL_WEIGHT_SCALE(7, 1);
                MUL_WEIGHT_SCALE(8, 2);
                MUL_WEIGHT_SCALE(9, 2);
                MUL_WEIGHT_SCALE(10, 2);
                MUL_WEIGHT_SCALE(11, 2);
                MUL_WEIGHT_SCALE(12, 3);
                MUL_WEIGHT_SCALE(13, 3);
                MUL_WEIGHT_SCALE(14, 3);
                MUL_WEIGHT_SCALE(15, 3);

                if (post->inputScale) { // Batch quant
                    f0 = _mm512_mul_ps(f0, inputscale0);
                    f1 = _mm512_mul_ps(f1, inputscale1);
                    f2 = _mm512_mul_ps(f2, inputscale2);
                    f3 = _mm512_mul_ps(f3, inputscale3);
                    f4 = _mm512_mul_ps(f4, inputscale0);
                    f5 = _mm512_mul_ps(f5, inputscale1);
                    f6 = _mm512_mul_ps(f6, inputscale2);
                    f7 = _mm512_mul_ps(f7, inputscale3);
                    f8 = _mm512_mul_ps(f8, inputscale0);
                    f9 = _mm512_mul_ps(f9, inputscale1);
                    f10 = _mm512_mul_ps(f10, inputscale2);
                    f11 = _mm512_mul_ps(f11, inputscale3);
                    f12 = _mm512_mul_ps(f12, inputscale0);
                    f13 = _mm512_mul_ps(f13, inputscale1);
                    f14 = _mm512_mul_ps(f14, inputscale2);
                    f15 = _mm512_mul_ps(f15, inputscale3);
                    if ((post->useInt8 == 0) && post->weightKernelSum && (post->inputBias || (bk == 0))) {
                        if (post->inputBias) {
                            weightKernelSum_dz = post->weightKernelSum + bk * GEMMINT8_AVX512_H + dz * blockNum * GEMMINT8_AVX512_H;
                            auto wsum0 = _mm512_loadu_ps(weightKernelSum_dz);
                            auto wsum1 = _mm512_loadu_ps(weightKernelSum_dz + 1 * PACK_UNIT);
                            auto wsum2 = _mm512_loadu_ps(weightKernelSum_dz + 2 * PACK_UNIT);
                            auto wsum3 = _mm512_loadu_ps(weightKernelSum_dz + 3 * PACK_UNIT);
                            bias00 = _mm512_mul_ps(inputbias0, wsum0);
                            bias01 = _mm512_mul_ps(inputbias1, wsum0);
                            bias02 = _mm512_mul_ps(inputbias2, wsum0);
                            bias03 = _mm512_mul_ps(inputbias3, wsum0);
                            bias10 = _mm512_mul_ps(inputbias0, wsum1);
                            bias11 = _mm512_mul_ps(inputbias1, wsum1);
                            bias12 = _mm512_mul_ps(inputbias2, wsum1);
                            bias13 = _mm512_mul_ps(inputbias3, wsum1);
                            bias20 = _mm512_mul_ps(inputbias0, wsum2);
                            bias21 = _mm512_mul_ps(inputbias1, wsum2);
                            bias22 = _mm512_mul_ps(inputbias2, wsum2);
                            bias23 = _mm512_mul_ps(inputbias3, wsum2);
                            bias30 = _mm512_mul_ps(inputbias0, wsum3);
                            bias31 = _mm512_mul_ps(inputbias1, wsum3);
                            bias32 = _mm512_mul_ps(inputbias2, wsum3);
                            bias33 = _mm512_mul_ps(inputbias3, wsum3);
                        } else if (bk == 0) { // if input not block quant, only accum once!
                            weightKernelSum_dz = post->weightKernelSum + dz * GEMMINT8_AVX512_H;
                            auto wsum0 = _mm512_loadu_ps(weightKernelSum_dz);
                            auto wsum1 = _mm512_loadu_ps(weightKernelSum_dz + 1 * PACK_UNIT);
                            auto wsum2 = _mm512_loadu_ps(weightKernelSum_dz + 2 * PACK_UNIT);
                            auto wsum3 = _mm512_loadu_ps(weightKernelSum_dz + 3 * PACK_UNIT);
                            bias00 = _mm512_mul_ps(_mm512_mul_ps(inputscale0, neg128f), wsum0);
                            bias01 = _mm512_mul_ps(_mm512_mul_ps(inputscale1, neg128f), wsum0);
                            bias02 = _mm512_mul_ps(_mm512_mul_ps(inputscale2, neg128f), wsum0);
                            bias03 = _mm512_mul_ps(_mm512_mul_ps(inputscale3, neg128f), wsum0);
                            bias10 = _mm512_mul_ps(_mm512_mul_ps(inputscale0, neg128f), wsum1);
                            bias11 = _mm512_mul_ps(_mm512_mul_ps(inputscale1, neg128f), wsum1);
                            bias12 = _mm512_mul_ps(_mm512_mul_ps(inputscale2, neg128f), wsum1);
                            bias13 = _mm512_mul_ps(_mm512_mul_ps(inputscale3, neg128f), wsum1);
                            bias20 = _mm512_mul_ps(_mm512_mul_ps(inputscale0, neg128f), wsum2);
                            bias21 = _mm512_mul_ps(_mm512_mul_ps(inputscale1, neg128f), wsum2);
                            bias22 = _mm512_mul_ps(_mm512_mul_ps(inputscale2, neg128f), wsum2);
                            bias23 = _mm512_mul_ps(_mm512_mul_ps(inputscale3, neg128f), wsum2);
                            bias30 = _mm512_mul_ps(_mm512_mul_ps(inputscale0, neg128f), wsum3);
                            bias31 = _mm512_mul_ps(_mm512_mul_ps(inputscale1, neg128f), wsum3);
                            bias32 = _mm512_mul_ps(_mm512_mul_ps(inputscale2, neg128f), wsum3);
                            bias33 = _mm512_mul_ps(_mm512_mul_ps(inputscale3, neg128f), wsum3);
                        }
                        f0 = _mm512_add_ps(f0, bias00);
                        f1 = _mm512_add_ps(f1, bias01);
                        f2 = _mm512_add_ps(f2, bias02);
                        f3 = _mm512_add_ps(f3, bias03);
                        f4 = _mm512_add_ps(f4, bias10);
                        f5 = _mm512_add_ps(f5, bias11);
                        f6 = _mm512_add_ps(f6, bias12);
                        f7 = _mm512_add_ps(f7, bias13);
                        f8 = _mm512_add_ps(f8, bias20);
                        f9 = _mm512_add_ps(f9, bias21);
                        f10 = _mm512_add_ps(f10, bias22);
                        f11 = _mm512_add_ps(f11, bias23);
                        f12 = _mm512_add_ps(f12, bias30);
                        f13 = _mm512_add_ps(f13, bias31);
                        f14 = _mm512_add_ps(f14, bias32);
                        f15 = _mm512_add_ps(f15, bias33);
                    }
                }
                f0 = _mm512_add_ps(_mm512_mul_ps(kernelSum0, weightBiasValue0), f0);
                f1 = _mm512_add_ps(_mm512_mul_ps(kernelSum1, weightBiasValue0), f1);
                f2 = _mm512_add_ps(_mm512_mul_ps(kernelSum2, weightBiasValue0), f2);
                f3 = _mm512_add_ps(_mm512_mul_ps(kernelSum3, weightBiasValue0), f3);
                f4 = _mm512_add_ps(_mm512_mul_ps(kernelSum0, weightBiasValue1), f4);
                f5 = _mm512_add_ps(_mm512_mul_ps(kernelSum1, weightBiasValue1), f5);
                f6 = _mm512_add_ps(_mm512_mul_ps(kernelSum2, weightBiasValue1), f6);
                f7 = _mm512_add_ps(_mm512_mul_ps(kernelSum3, weightBiasValue1), f7);
                f8 = _mm512_add_ps(_mm512_mul_ps(kernelSum0, weightBiasValue2), f8);
                f9 = _mm512_add_ps(_mm512_mul_ps(kernelSum1, weightBiasValue2), f9);
                f10 = _mm512_add_ps(_mm512_mul_ps(kernelSum2, weightBiasValue2),f10);
                f11 = _mm512_add_ps(_mm512_mul_ps(kernelSum3, weightBiasValue2),f11);
                f12 = _mm512_add_ps(_mm512_mul_ps(kernelSum0, weightBiasValue3),f12);
                f13 = _mm512_add_ps(_mm512_mul_ps(kernelSum1, weightBiasValue3),f13);
                f14 = _mm512_add_ps(_mm512_mul_ps(kernelSum2, weightBiasValue3),f14);
                f15 = _mm512_add_ps(_mm512_mul_ps(kernelSum3, weightBiasValue3),f15);

                if (bk > 0) {
                    f0 = _mm512_add_ps(_mm512_loadu_ps(accum_x), f0);
                    f1 = _mm512_add_ps(_mm512_loadu_ps(accum_x + 16), f1);
                    f2 = _mm512_add_ps(_mm512_loadu_ps(accum_x + 16 * 2), f2);
                    f3 = _mm512_add_ps(_mm512_loadu_ps(accum_x + 16 * 3), f3);

                    f4 = _mm512_add_ps(_mm512_loadu_ps(accum_x + source_step), f4);
                    f5 = _mm512_add_ps(_mm512_loadu_ps(accum_x + source_step + 16 * 1), f5);
                    f6 = _mm512_add_ps(_mm512_loadu_ps(accum_x + source_step + 16 * 2), f6);
                    f7 = _mm512_add_ps(_mm512_loadu_ps(accum_x + source_step + 16 * 3), f7);

                    f8 = _mm512_add_ps(_mm512_loadu_ps(accum_x + 2 * source_step), f8);
                    f9 = _mm512_add_ps(_mm512_loadu_ps(accum_x + 2 * source_step + 16 * 1), f9);
                    f10 = _mm512_add_ps(_mm512_loadu_ps(accum_x + 2 * source_step + 16 * 2), f10);
                    f11 = _mm512_add_ps(_mm512_loadu_ps(accum_x + 2 * source_step + 16 * 3), f11);

                    f12 = _mm512_add_ps(_mm512_loadu_ps(accum_x + 3 * source_step), f12);
                    f13 = _mm512_add_ps(_mm512_loadu_ps(accum_x + 3 * source_step + 16 * 1), f13);
                    f14 = _mm512_add_ps(_mm512_loadu_ps(accum_x + 3 * source_step + 16 * 2), f14);
                    f15 = _mm512_add_ps(_mm512_loadu_ps(accum_x + 3 * source_step + 16 * 3), f15);
                }
                if (bk == blockNum - 1) {
                    if (biasPtr) {
                        auto biasValue0 = _mm512_loadu_ps(bias_dz);
                        auto biasValue4 = _mm512_loadu_ps(bias_dz + 1 * PACK_UNIT);
                        auto biasValue8 = _mm512_loadu_ps(bias_dz + 2 * PACK_UNIT);
                        auto biasValue12 = _mm512_loadu_ps(bias_dz + 3 * PACK_UNIT);
                        f0 = _mm512_add_ps(f0, biasValue0);
                        f1 = _mm512_add_ps(f1, biasValue0);
                        f2 = _mm512_add_ps(f2, biasValue0);
                        f3 = _mm512_add_ps(f3, biasValue0);
                        f4 = _mm512_add_ps(f4, biasValue4);
                        f5 = _mm512_add_ps(f5, biasValue4);
                        f6 = _mm512_add_ps(f6, biasValue4);
                        f7 = _mm512_add_ps(f7, biasValue4);
                        f8 = _mm512_add_ps(f8, biasValue8);
                        f9 = _mm512_add_ps(f9, biasValue8);
                        f10 = _mm512_add_ps(f10, biasValue8);
                        f11 = _mm512_add_ps(f11, biasValue8);
                        f12 = _mm512_add_ps(f12, biasValue12);
                        f13 = _mm512_add_ps(f13, biasValue12);
                        f14 = _mm512_add_ps(f14, biasValue12);
                        f15 = _mm512_add_ps(f15, biasValue12);
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
                    _mm512_storeu_ps(accum_x, f0);
                    _mm512_storeu_ps(accum_x + 16, f1);
                    _mm512_storeu_ps(accum_x + 16 * 2, f2);
                    _mm512_storeu_ps(accum_x + 16 * 3, f3);
                    _mm512_storeu_ps(accum_x + source_step, f4);
                    _mm512_storeu_ps(accum_x + source_step + 16 * 1, f5);
                    _mm512_storeu_ps(accum_x + source_step + 16 * 2, f6);
                    _mm512_storeu_ps(accum_x + source_step + 16 * 3, f7);
                    _mm512_storeu_ps(accum_x + 2 * source_step, f8);
                    _mm512_storeu_ps(accum_x + 2 * source_step + 16 * 1, f9);
                    _mm512_storeu_ps(accum_x + 2 * source_step + 16 * 2, f10);
                    _mm512_storeu_ps(accum_x + 2 * source_step + 16 * 3, f11);
                    _mm512_storeu_ps(accum_x + 3 * source_step, f12);
                    _mm512_storeu_ps(accum_x + 3 * source_step + 16 * 1, f13);
                    _mm512_storeu_ps(accum_x + 3 * source_step + 16 * 2, f14);
                    _mm512_storeu_ps(accum_x + 3 * source_step + 16 * 3, f15);
                }
            }
        } // dzU
        // the remaining ocDivPack
        auto weight_dz = weight + dzU * blockNum * weight_step_Z;                                            // weight address for remaining
        if (biasPtr) {
            bias_dz = post->biasFloat + dzU * PACK_UNIT * dzUnit;
        }

        auto dst_x = dst + dzU * dst_step_tmp * dzUnit;
        for (int i=0; i<dzR; ++i) {
            auto accum_x = accumbuff;
            for (int bk = 0; bk < blockNum; ++bk) {
                __m512i D0 = _mm512_set1_epi32(0);
                __m512i D1 = _mm512_set1_epi32(0);
                __m512i D2 = _mm512_set1_epi32(0);
                __m512i D3 = _mm512_set1_epi32(0);
                auto weightDzSub = weight_dz + bk * weight_step_Z + weightPackStride * suborder[i];
                auto scaleDz = (float*)(weight_dz + bk * weight_step_Z + src_depth_quad * weight_step_Y);
                 auto biasDz = scaleDz + GEMMINT8_AVX512_H;
                const auto src_x = src + bk * src_depth_quad * GEMMINT8_AVX512_L * realDst;

                for (int sz = 0; sz < src_depth_quad; ++sz) {
                    const auto weight_sz = weightDzSub + weight_step_Y * sz;
                    const auto src_z     = (const float*)(src_x + sz * realDst * GEMMINT8_AVX512_L);
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

                auto scaleValue0 = _mm512_loadu_ps(scaleDz + i * PACK_UNIT);
                auto weightBiasValue0 = _mm512_loadu_ps(biasDz + i * PACK_UNIT);
                // input info
                kernelSum0 = _mm512_set1_ps((post->srcKernelSum + bk * realDst)[0]);
                kernelSum1 = _mm512_set1_ps((post->srcKernelSum + bk * realDst)[1]);
                kernelSum2 = _mm512_set1_ps((post->srcKernelSum + bk * realDst)[2]);
                kernelSum3 = _mm512_set1_ps((post->srcKernelSum + bk * realDst)[3]);
                if (post->inputBias) {
                    inputscale0 = _mm512_set1_ps((post->inputScale + bk * realDst)[0]);
                    inputscale1 = _mm512_set1_ps((post->inputScale + bk * realDst)[1]);
                    inputscale2 = _mm512_set1_ps((post->inputScale + bk * realDst)[2]);
                    inputscale3 = _mm512_set1_ps((post->inputScale + bk * realDst)[3]);
                    inputbias0 = _mm512_set1_ps((post->inputBias + bk * realDst)[0]);
                    inputbias1 = _mm512_set1_ps((post->inputBias + bk * realDst)[1]);
                    inputbias2 = _mm512_set1_ps((post->inputBias + bk * realDst)[2]);
                    inputbias3 = _mm512_set1_ps((post->inputBias + bk * realDst)[3]);
                }
                MUL_WEIGHT_SCALE(0, 0);
                MUL_WEIGHT_SCALE(1, 0);
                MUL_WEIGHT_SCALE(2, 0);
                MUL_WEIGHT_SCALE(3, 0);

                if (post->inputScale) { // Batch quant
                    f0 = _mm512_mul_ps(f0, inputscale0);
                    f1 = _mm512_mul_ps(f1, inputscale1);
                    f2 = _mm512_mul_ps(f2, inputscale2);
                    f3 = _mm512_mul_ps(f3, inputscale3);
                    if ((post->useInt8 == 0) && post->weightKernelSum && (post->inputBias || (bk == 0))) {
                        if (post->inputBias) {
                            weightKernelSum_dz = post->weightKernelSum + dzU * blockNum * GEMMINT8_AVX512_H + bk * GEMMINT8_AVX512_H;
                            auto wsum0 = _mm512_loadu_ps(weightKernelSum_dz + i * PACK_UNIT);
                            bias00 = _mm512_mul_ps(inputbias0, wsum0);
                            bias01 = _mm512_mul_ps(inputbias1, wsum0);
                            bias02 = _mm512_mul_ps(inputbias2, wsum0);
                            bias03 = _mm512_mul_ps(inputbias3, wsum0);
                        } else if (bk == 0) { // if input not block quant, only accum once!
                            weightKernelSum_dz = post->weightKernelSum + dzU * PACK_UNIT * dzUnit;
                            auto wsum0 = _mm512_loadu_ps(weightKernelSum_dz + i * PACK_UNIT);
                            bias00 = _mm512_mul_ps(_mm512_mul_ps(inputscale0, neg128f), wsum0);
                            bias01 = _mm512_mul_ps(_mm512_mul_ps(inputscale1, neg128f), wsum0);
                            bias02 = _mm512_mul_ps(_mm512_mul_ps(inputscale2, neg128f), wsum0);
                            bias03 = _mm512_mul_ps(_mm512_mul_ps(inputscale3, neg128f), wsum0);
                        }
                        f0 = _mm512_add_ps(f0, bias00);
                        f1 = _mm512_add_ps(f1, bias01);
                        f2 = _mm512_add_ps(f2, bias02);
                        f3 = _mm512_add_ps(f3, bias03);
                    }
                }
                f0 = _mm512_add_ps(_mm512_mul_ps(kernelSum0, weightBiasValue0), f0);
                f1 = _mm512_add_ps(_mm512_mul_ps(kernelSum1, weightBiasValue0), f1);
                f2 = _mm512_add_ps(_mm512_mul_ps(kernelSum2, weightBiasValue0), f2);
                f3 = _mm512_add_ps(_mm512_mul_ps(kernelSum3, weightBiasValue0), f3);
                

                if (bk > 0) {
                    f0 = _mm512_add_ps(_mm512_loadu_ps((float*)accum_x), f0);
                    f1 = _mm512_add_ps(_mm512_loadu_ps(((float*)accum_x) + 16), f1);
                    f2 = _mm512_add_ps(_mm512_loadu_ps(((float*)accum_x) + 16 * 2), f2);
                    f3 = _mm512_add_ps(_mm512_loadu_ps(((float*)accum_x) + 16 * 3), f3);
                }
                if (bk == blockNum - 1) {
                    if (nullptr != biasPtr) {
                        auto biasValue0 = _mm512_loadu_ps(bias_dz + i * PACK_UNIT);
                        f0 = _mm512_add_ps(f0, biasValue0);
                        f1 = _mm512_add_ps(f1, biasValue0);
                        f2 = _mm512_add_ps(f2, biasValue0);
                        f3 = _mm512_add_ps(f3, biasValue0);
                    }
                    if (post->fp32minmax) {
                        POST_TREAT_FLOAT(0,1,2,3);
                    }
                    _mm512_storeu_ps((float*)(dst_x + i * dst_step_tmp), f0);
                    _mm512_storeu_ps((float*)(dst_x + i * dst_step_tmp) + 16, f1);
                    _mm512_storeu_ps((float*)(dst_x + i * dst_step_tmp) + 16 * 2, f2);
                    _mm512_storeu_ps((float*)(dst_x + i * dst_step_tmp) + 16 * 3, f3);
                } else {
                    _mm512_storeu_ps(((float*)accum_x), f0);
                    _mm512_storeu_ps(((float*)accum_x) + 16, f1);
                    _mm512_storeu_ps(((float*)accum_x) + 16 * 2, f2);
                    _mm512_storeu_ps(((float*)accum_x) + 16 * 3, f3);
                }
            }
        }
        return;
    }
    
    if (realDst == 3) {
        for (int dz = 0; dz < dzU; ++dz) {
            if (biasPtr) {
                bias_dz = post->biasFloat + dz * PACK_UNIT * dzUnit;
            }
            auto dst_x = dst + dz * dst_step_tmp * dzUnit;
            auto accum_x = accumbuff;

            for (int bk = 0; bk < blockNum; ++bk) {
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

                // block's weight&scale&bias
                const auto weight_dz = weight + dz * blockNum * weight_step_Z + bk *  weight_step_Z;
                const float* scale_dz = reinterpret_cast<const float*>(weight_dz + src_depth_quad * weight_step_Y);
                const auto weightBias_dz = scale_dz + GEMMINT8_AVX512_H;
                // block's input
                const auto src_x = src + bk * src_depth_quad * GEMMINT8_AVX512_L * realDst;
                for (int sz = 0; sz < src_depth_quad; ++sz) {
                    const auto weight_sz = weight_dz + weight_step_Y * sz;
                    const auto src_z     = (const float*)(src_x + sz * realDst * GEMMINT8_AVX512_L);

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
                // int32_t -> float
                auto scaleValue0 = _mm512_loadu_ps(scale_dz);
                auto scaleValue1 = _mm512_loadu_ps(scale_dz + 1 * PACK_UNIT);
                auto scaleValue2 = _mm512_loadu_ps(scale_dz + 2 * PACK_UNIT);
                auto scaleValue3 = _mm512_loadu_ps(scale_dz + 3 * PACK_UNIT);
                auto weightBiasValue0 = _mm512_loadu_ps(weightBias_dz);
                auto weightBiasValue1 = _mm512_loadu_ps(weightBias_dz + 1 * PACK_UNIT);
                auto weightBiasValue2 = _mm512_loadu_ps(weightBias_dz + 2 * PACK_UNIT);
                auto weightBiasValue3 = _mm512_loadu_ps(weightBias_dz + 3 * PACK_UNIT);
                // input info
                kernelSum0 = _mm512_set1_ps((post->srcKernelSum + bk * realDst)[0]);
                kernelSum1 = _mm512_set1_ps((post->srcKernelSum + bk * realDst)[1]);
                kernelSum2 = _mm512_set1_ps((post->srcKernelSum + bk * realDst)[2]);
                if (post->inputBias) {
                    inputscale0 = _mm512_set1_ps((post->inputScale + bk * realDst)[0]);
                    inputscale1= _mm512_set1_ps((post->inputScale + bk * realDst)[1]);
                    inputscale2 = _mm512_set1_ps((post->inputScale + bk * realDst)[2]);
                    inputbias0 = _mm512_set1_ps((post->inputBias + bk * realDst)[0]);
                    inputbias1 = _mm512_set1_ps((post->inputBias + bk * realDst)[1]);
                    inputbias2 = _mm512_set1_ps((post->inputBias + bk * realDst)[2]);
                }
                MUL_WEIGHT_SCALE(0, 0);
                MUL_WEIGHT_SCALE(1, 0);
                MUL_WEIGHT_SCALE(2, 0);
                MUL_WEIGHT_SCALE(4, 1);
                MUL_WEIGHT_SCALE(5, 1);
                MUL_WEIGHT_SCALE(6, 1);
                MUL_WEIGHT_SCALE(8, 2);
                MUL_WEIGHT_SCALE(9, 2);
                MUL_WEIGHT_SCALE(10, 2);
                MUL_WEIGHT_SCALE(12, 3);
                MUL_WEIGHT_SCALE(13, 3);
                MUL_WEIGHT_SCALE(14, 3);

                if (post->inputScale) { // Batch quant
                    f0 = _mm512_mul_ps(f0, inputscale0);
                    f1 = _mm512_mul_ps(f1, inputscale1);
                    f2 = _mm512_mul_ps(f2, inputscale2);
                    f4 = _mm512_mul_ps(f4, inputscale0);
                    f5 = _mm512_mul_ps(f5, inputscale1);
                    f6 = _mm512_mul_ps(f6, inputscale2);
                    f8 = _mm512_mul_ps(f8, inputscale0);
                    f9 = _mm512_mul_ps(f9, inputscale1);
                    f10 = _mm512_mul_ps(f10, inputscale2);
                    f12 = _mm512_mul_ps(f12, inputscale0);
                    f13 = _mm512_mul_ps(f13, inputscale1);
                    f14 = _mm512_mul_ps(f14, inputscale2);
                    if ((post->useInt8 == 0) && post->weightKernelSum && (post->inputBias || (bk == 0))) {
                        if (post->inputBias) {
                            weightKernelSum_dz = post->weightKernelSum + bk * GEMMINT8_AVX512_H + dz * blockNum * GEMMINT8_AVX512_H;
                            auto wsum0 = _mm512_loadu_ps(weightKernelSum_dz);
                            auto wsum1 = _mm512_loadu_ps(weightKernelSum_dz + 1 * PACK_UNIT);
                            auto wsum2 = _mm512_loadu_ps(weightKernelSum_dz + 2 * PACK_UNIT);
                            auto wsum3 = _mm512_loadu_ps(weightKernelSum_dz + 3 * PACK_UNIT);
                            bias00 = _mm512_mul_ps(inputbias0, wsum0);
                            bias01 = _mm512_mul_ps(inputbias1, wsum0);
                            bias02 = _mm512_mul_ps(inputbias2, wsum0);
                            bias10 = _mm512_mul_ps(inputbias0, wsum1);
                            bias11 = _mm512_mul_ps(inputbias1, wsum1);
                            bias12 = _mm512_mul_ps(inputbias2, wsum1);
                            bias20 = _mm512_mul_ps(inputbias0, wsum2);
                            bias21 = _mm512_mul_ps(inputbias1, wsum2);
                            bias22 = _mm512_mul_ps(inputbias2, wsum2);
                            bias30 = _mm512_mul_ps(inputbias0, wsum3);
                            bias31 = _mm512_mul_ps(inputbias1, wsum3);
                            bias32 = _mm512_mul_ps(inputbias2, wsum3);
                        } else if (bk == 0) { // if input not block quant, only accum once!
                            weightKernelSum_dz = post->weightKernelSum + dz * GEMMINT8_AVX512_H;
                            auto wsum0 = _mm512_loadu_ps(weightKernelSum_dz);
                            auto wsum1 = _mm512_loadu_ps(weightKernelSum_dz + 1 * PACK_UNIT);
                            auto wsum2 = _mm512_loadu_ps(weightKernelSum_dz + 2 * PACK_UNIT);
                            auto wsum3 = _mm512_loadu_ps(weightKernelSum_dz + 3 * PACK_UNIT);
                            bias00 = _mm512_mul_ps(_mm512_mul_ps(inputscale0, neg128f), wsum0);
                            bias01 = _mm512_mul_ps(_mm512_mul_ps(inputscale1, neg128f), wsum0);
                            bias02 = _mm512_mul_ps(_mm512_mul_ps(inputscale2, neg128f), wsum0);
                            bias10 = _mm512_mul_ps(_mm512_mul_ps(inputscale0, neg128f), wsum1);
                            bias11 = _mm512_mul_ps(_mm512_mul_ps(inputscale1, neg128f), wsum1);
                            bias12 = _mm512_mul_ps(_mm512_mul_ps(inputscale2, neg128f), wsum1);
                            bias20 = _mm512_mul_ps(_mm512_mul_ps(inputscale0, neg128f), wsum2);
                            bias21 = _mm512_mul_ps(_mm512_mul_ps(inputscale1, neg128f), wsum2);
                            bias22 = _mm512_mul_ps(_mm512_mul_ps(inputscale2, neg128f), wsum2);
                            bias30 = _mm512_mul_ps(_mm512_mul_ps(inputscale0, neg128f), wsum3);
                            bias31 = _mm512_mul_ps(_mm512_mul_ps(inputscale1, neg128f), wsum3);
                            bias32 = _mm512_mul_ps(_mm512_mul_ps(inputscale2, neg128f), wsum3);
                        }
                        f0 = _mm512_add_ps(f0, bias00);
                        f1 = _mm512_add_ps(f1, bias01);
                        f2 = _mm512_add_ps(f2, bias02);
                        f4 = _mm512_add_ps(f4, bias10);
                        f5 = _mm512_add_ps(f5, bias11);
                        f6 = _mm512_add_ps(f6, bias12);
                        f8 = _mm512_add_ps(f8, bias20);
                        f9 = _mm512_add_ps(f9, bias21);
                        f10 = _mm512_add_ps(f10, bias22);
                        f12 = _mm512_add_ps(f12, bias30);
                        f13 = _mm512_add_ps(f13, bias31);
                        f14 = _mm512_add_ps(f14, bias32);
                    }
                }
                f0 = _mm512_add_ps(_mm512_mul_ps(kernelSum0, weightBiasValue0), f0);
                f1 = _mm512_add_ps(_mm512_mul_ps(kernelSum1, weightBiasValue0), f1);
                f2 = _mm512_add_ps(_mm512_mul_ps(kernelSum2, weightBiasValue0), f2);
                f4 = _mm512_add_ps(_mm512_mul_ps(kernelSum0, weightBiasValue1), f4);
                f5 = _mm512_add_ps(_mm512_mul_ps(kernelSum1, weightBiasValue1), f5);
                f6 = _mm512_add_ps(_mm512_mul_ps(kernelSum2, weightBiasValue1), f6);
                f8 = _mm512_add_ps(_mm512_mul_ps(kernelSum0, weightBiasValue2), f8);
                f9 = _mm512_add_ps(_mm512_mul_ps(kernelSum1, weightBiasValue2), f9);
                f10 = _mm512_add_ps(_mm512_mul_ps(kernelSum2, weightBiasValue2),f10);
                f12 = _mm512_add_ps(_mm512_mul_ps(kernelSum0, weightBiasValue3),f12);
                f13 = _mm512_add_ps(_mm512_mul_ps(kernelSum1, weightBiasValue3),f13);
                f14 = _mm512_add_ps(_mm512_mul_ps(kernelSum2, weightBiasValue3),f14);

                if (bk > 0) {
                    f0 = _mm512_add_ps(_mm512_loadu_ps(accum_x), f0);
                    f1 = _mm512_add_ps(_mm512_loadu_ps(accum_x + 16), f1);
                    f2 = _mm512_add_ps(_mm512_loadu_ps(accum_x + 16 * 2), f2);

                    f4 = _mm512_add_ps(_mm512_loadu_ps(accum_x + source_step), f4);
                    f5 = _mm512_add_ps(_mm512_loadu_ps(accum_x + source_step + 16 * 1), f5);
                    f6 = _mm512_add_ps(_mm512_loadu_ps(accum_x + source_step + 16 * 2), f6);

                    f8 = _mm512_add_ps(_mm512_loadu_ps(accum_x + 2 * source_step), f8);
                    f9 = _mm512_add_ps(_mm512_loadu_ps(accum_x + 2 * source_step + 16 * 1), f9);
                    f10 = _mm512_add_ps(_mm512_loadu_ps(accum_x + 2 * source_step + 16 * 2), f10);

                    f12 = _mm512_add_ps(_mm512_loadu_ps(accum_x + 3 * source_step), f12);
                    f13 = _mm512_add_ps(_mm512_loadu_ps(accum_x + 3 * source_step + 16 * 1), f13);
                    f14 = _mm512_add_ps(_mm512_loadu_ps(accum_x + 3 * source_step + 16 * 2), f14);
                }
                if (bk == blockNum - 1) {
                    if (biasPtr) {
                        auto biasValue0 = _mm512_loadu_ps(bias_dz);
                        auto biasValue4 = _mm512_loadu_ps(bias_dz + 1 * PACK_UNIT);
                        auto biasValue8 = _mm512_loadu_ps(bias_dz + 2 * PACK_UNIT);
                        auto biasValue12 = _mm512_loadu_ps(bias_dz + 3 * PACK_UNIT);
                        f0 = _mm512_add_ps(f0, biasValue0);
                        f1 = _mm512_add_ps(f1, biasValue0);
                        f2 = _mm512_add_ps(f2, biasValue0);
                        f4 = _mm512_add_ps(f4, biasValue4);
                        f5 = _mm512_add_ps(f5, biasValue4);
                        f6 = _mm512_add_ps(f6, biasValue4);
                        f8 = _mm512_add_ps(f8, biasValue8);
                        f9 = _mm512_add_ps(f9, biasValue8);
                        f10 = _mm512_add_ps(f10, biasValue8);
                        f12 = _mm512_add_ps(f12, biasValue12);
                        f13 = _mm512_add_ps(f13, biasValue12);
                        f14 = _mm512_add_ps(f14, biasValue12);
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
                    _mm512_storeu_ps(accum_x, f0);
                    _mm512_storeu_ps(accum_x + 16, f1);
                    _mm512_storeu_ps(accum_x + 16 * 2, f2);
                    _mm512_storeu_ps(accum_x + source_step, f4);
                    _mm512_storeu_ps(accum_x + source_step + 16 * 1, f5);
                    _mm512_storeu_ps(accum_x + source_step + 16 * 2, f6);
                    _mm512_storeu_ps(accum_x + 2 * source_step, f8);
                    _mm512_storeu_ps(accum_x + 2 * source_step + 16 * 1, f9);
                    _mm512_storeu_ps(accum_x + 2 * source_step + 16 * 2, f10);
                    _mm512_storeu_ps(accum_x + 3 * source_step, f12);
                    _mm512_storeu_ps(accum_x + 3 * source_step + 16 * 1, f13);
                    _mm512_storeu_ps(accum_x + 3 * source_step + 16 * 2, f14);
                }
            }
        } // dzU
        // the remaining ocDivPack
        auto weight_dz = weight + dzU * blockNum * weight_step_Z;                                            // weight address for remaining
        if (biasPtr) {
            bias_dz = post->biasFloat + dzU * PACK_UNIT * dzUnit;
        }

        auto dst_x = dst + dzU * dst_step_tmp * dzUnit;
        for (int i=0; i<dzR; ++i) {
            auto accum_x = accumbuff;
            for (int bk = 0; bk < blockNum; ++bk) {
                __m512i D0 = _mm512_set1_epi32(0);
                __m512i D1 = _mm512_set1_epi32(0);
                __m512i D2 = _mm512_set1_epi32(0);
                auto weightDzSub = weight_dz + bk * weight_step_Z + weightPackStride * suborder[i];
                auto scaleDz = (float*)(weight_dz + bk * weight_step_Z + src_depth_quad * weight_step_Y);
                 auto biasDz = scaleDz + GEMMINT8_AVX512_H;
                const auto src_x = src + bk * src_depth_quad * GEMMINT8_AVX512_L * realDst;

                for (int sz = 0; sz < src_depth_quad; ++sz) {
                    const auto weight_sz = weightDzSub + weight_step_Y * sz;
                    const auto src_z     = (const float*)(src_x + sz * realDst * GEMMINT8_AVX512_L);
                    auto w0_int4_64 = _mm512_loadu_si512(weight_sz); // 128xint4_t=64 byte
                    // 256xint4_t->256xint8_t
                    auto w0 = _mm512_and_si512(mask, _mm512_srli_epi16(w0_int4_64, 4)); // 64xint8_t

                    auto s0 = AVX512_BROADCAST_INT32(src_z + 0);
                    auto s1 = AVX512_BROADCAST_INT32(src_z + 1);
                    auto s2 = AVX512_BROADCAST_INT32(src_z + 2);

                    D0 = _mm512_dpbusds_epi32(D0, s0, w0);
                    D1 = _mm512_dpbusds_epi32(D1, s1, w0);
                    D2 = _mm512_dpbusds_epi32(D2, s2, w0);
                }

                auto scaleValue0 = _mm512_loadu_ps(scaleDz + i * PACK_UNIT);
                auto weightBiasValue0 = _mm512_loadu_ps(biasDz + i * PACK_UNIT);
                // input info
                kernelSum0 = _mm512_set1_ps((post->srcKernelSum + bk * realDst)[0]);
                kernelSum1 = _mm512_set1_ps((post->srcKernelSum + bk * realDst)[1]);
                kernelSum2 = _mm512_set1_ps((post->srcKernelSum + bk * realDst)[2]);
                if (post->inputBias) {
                    inputscale0 = _mm512_set1_ps((post->inputScale + bk * realDst)[0]);
                    inputscale1 = _mm512_set1_ps((post->inputScale + bk * realDst)[1]);
                    inputscale2 = _mm512_set1_ps((post->inputScale + bk * realDst)[2]);
                    inputbias0 = _mm512_set1_ps((post->inputBias + bk * realDst)[0]);
                    inputbias1 = _mm512_set1_ps((post->inputBias + bk * realDst)[1]);
                    inputbias2 = _mm512_set1_ps((post->inputBias + bk * realDst)[2]);
                }
                MUL_WEIGHT_SCALE(0, 0);
                MUL_WEIGHT_SCALE(1, 0);
                MUL_WEIGHT_SCALE(2, 0);

                if (post->inputScale) { // Batch quant
                    f0 = _mm512_mul_ps(f0, inputscale0);
                    f1 = _mm512_mul_ps(f1, inputscale1);
                    f2 = _mm512_mul_ps(f2, inputscale2);
                    if ((post->useInt8 == 0) && post->weightKernelSum && (post->inputBias || (bk == 0))) {
                        if (post->inputBias) {
                            weightKernelSum_dz = post->weightKernelSum + dzU * blockNum * GEMMINT8_AVX512_H + bk * GEMMINT8_AVX512_H;
                            auto wsum0 = _mm512_loadu_ps(weightKernelSum_dz + i * PACK_UNIT);
                            bias00 = _mm512_mul_ps(inputbias0, wsum0);
                            bias01 = _mm512_mul_ps(inputbias1, wsum0);
                            bias02 = _mm512_mul_ps(inputbias2, wsum0);
                        } else if (bk == 0) { // if input not block quant, only accum once!
                            weightKernelSum_dz = post->weightKernelSum + dzU * PACK_UNIT * dzUnit;
                            auto wsum0 = _mm512_loadu_ps(weightKernelSum_dz + i * PACK_UNIT);
                            bias00 = _mm512_mul_ps(_mm512_mul_ps(inputscale0, neg128f), wsum0);
                            bias01 = _mm512_mul_ps(_mm512_mul_ps(inputscale1, neg128f), wsum0);
                            bias02 = _mm512_mul_ps(_mm512_mul_ps(inputscale2, neg128f), wsum0);
                        }
                        f0 = _mm512_add_ps(f0, bias00);
                        f1 = _mm512_add_ps(f1, bias01);
                        f2 = _mm512_add_ps(f2, bias02);
                    }
                }
                f0 = _mm512_add_ps(_mm512_mul_ps(kernelSum0, weightBiasValue0), f0);
                f1 = _mm512_add_ps(_mm512_mul_ps(kernelSum1, weightBiasValue0), f1);
                f2 = _mm512_add_ps(_mm512_mul_ps(kernelSum2, weightBiasValue0), f2);

                if (bk > 0) {
                    f0 = _mm512_add_ps(_mm512_loadu_ps((float*)accum_x), f0);
                    f1 = _mm512_add_ps(_mm512_loadu_ps(((float*)accum_x) + 16), f1);
                    f2 = _mm512_add_ps(_mm512_loadu_ps(((float*)accum_x) + 16 * 2), f2);
                }
                if (bk == blockNum - 1) {
                    if (nullptr != biasPtr) {
                        auto biasValue0 = _mm512_loadu_ps(bias_dz + i * PACK_UNIT);
                        f0 = _mm512_add_ps(f0, biasValue0);
                        f1 = _mm512_add_ps(f1, biasValue0);
                        f2 = _mm512_add_ps(f2, biasValue0);
                    }
                    if (post->fp32minmax) {
                        POST_TREAT_FLOAT_3(0,1,2);
                    }
                    _mm512_storeu_ps((float*)(dst_x + i * dst_step_tmp), f0);
                    _mm512_storeu_ps((float*)(dst_x + i * dst_step_tmp) + 16, f1);
                    _mm512_storeu_ps((float*)(dst_x + i * dst_step_tmp) + 16 * 2, f2);
                } else {
                    _mm512_storeu_ps(((float*)accum_x), f0);
                    _mm512_storeu_ps(((float*)accum_x) + 16, f1);
                    _mm512_storeu_ps(((float*)accum_x) + 16 * 2, f2);
                }
            }
        }
        return;
    }

    if (realDst == 2) {
        for (int dz = 0; dz < dzU; ++dz) {
            if (biasPtr) {
                bias_dz = post->biasFloat + dz * PACK_UNIT * dzUnit;
            }
            auto dst_x = dst + dz * dst_step_tmp * dzUnit;
            auto accum_x = accumbuff;

            for (int bk = 0; bk < blockNum; ++bk) {
                __m512i D0 = _mm512_set1_epi32(0);
                __m512i D1 = _mm512_set1_epi32(0);

                __m512i D4 = _mm512_set1_epi32(0);
                __m512i D5 = _mm512_set1_epi32(0);

                __m512i D8 = _mm512_set1_epi32(0);
                __m512i D9 = _mm512_set1_epi32(0);

                __m512i D12 = _mm512_set1_epi32(0);
                __m512i D13 = _mm512_set1_epi32(0);

                // block's weight&scale&bias
                const auto weight_dz = weight + dz * blockNum * weight_step_Z + bk *  weight_step_Z;
                const float* scale_dz = reinterpret_cast<const float*>(weight_dz + src_depth_quad * weight_step_Y);
                const auto weightBias_dz = scale_dz + GEMMINT8_AVX512_H;
                // block's input
                const auto src_x = src + bk * src_depth_quad * GEMMINT8_AVX512_L * realDst;
                for (int sz = 0; sz < src_depth_quad; ++sz) {
                    const auto weight_sz = weight_dz + weight_step_Y * sz;
                    const auto src_z     = (const float*)(src_x + sz * realDst * GEMMINT8_AVX512_L);

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
                // int32_t -> float
                auto scaleValue0 = _mm512_loadu_ps(scale_dz);
                auto scaleValue1 = _mm512_loadu_ps(scale_dz + 1 * PACK_UNIT);
                auto scaleValue2 = _mm512_loadu_ps(scale_dz + 2 * PACK_UNIT);
                auto scaleValue3 = _mm512_loadu_ps(scale_dz + 3 * PACK_UNIT);
                auto weightBiasValue0 = _mm512_loadu_ps(weightBias_dz);
                auto weightBiasValue1 = _mm512_loadu_ps(weightBias_dz + 1 * PACK_UNIT);
                auto weightBiasValue2 = _mm512_loadu_ps(weightBias_dz + 2 * PACK_UNIT);
                auto weightBiasValue3 = _mm512_loadu_ps(weightBias_dz + 3 * PACK_UNIT);
                // input info
                kernelSum0 = _mm512_set1_ps((post->srcKernelSum + bk * realDst)[0]);
                kernelSum1 = _mm512_set1_ps((post->srcKernelSum + bk * realDst)[1]);
                if (post->inputBias) {
                    inputscale0 = _mm512_set1_ps((post->inputScale + bk * realDst)[0]);
                    inputscale1= _mm512_set1_ps((post->inputScale + bk * realDst)[1]);
                    inputbias0 = _mm512_set1_ps((post->inputBias + bk * realDst)[0]);
                    inputbias1 = _mm512_set1_ps((post->inputBias + bk * realDst)[1]);
                }

                MUL_WEIGHT_SCALE(0, 0);
                MUL_WEIGHT_SCALE(1, 0);
                MUL_WEIGHT_SCALE(4, 1);
                MUL_WEIGHT_SCALE(5, 1);
                MUL_WEIGHT_SCALE(8, 2);
                MUL_WEIGHT_SCALE(9, 2);
                MUL_WEIGHT_SCALE(12, 3);
                MUL_WEIGHT_SCALE(13, 3);

                if (post->inputScale) { // Batch quant
                    f0 = _mm512_mul_ps(f0, inputscale0);
                    f1 = _mm512_mul_ps(f1, inputscale1);
                    f4 = _mm512_mul_ps(f4, inputscale0);
                    f5 = _mm512_mul_ps(f5, inputscale1);
                    f8 = _mm512_mul_ps(f8, inputscale0);
                    f9 = _mm512_mul_ps(f9, inputscale1);
                    f12 = _mm512_mul_ps(f12, inputscale0);
                    f13 = _mm512_mul_ps(f13, inputscale1);
                    if ((post->useInt8 == 0) && post->weightKernelSum && (post->inputBias || (bk == 0))) {
                        if (post->inputBias) {
                            weightKernelSum_dz = post->weightKernelSum + bk * GEMMINT8_AVX512_H + dz * blockNum * GEMMINT8_AVX512_H;
                            auto wsum0 = _mm512_loadu_ps(weightKernelSum_dz);
                            auto wsum1 = _mm512_loadu_ps(weightKernelSum_dz + 1 * PACK_UNIT);
                            auto wsum2 = _mm512_loadu_ps(weightKernelSum_dz + 2 * PACK_UNIT);
                            auto wsum3 = _mm512_loadu_ps(weightKernelSum_dz + 3 * PACK_UNIT);
                            bias00 = _mm512_mul_ps(inputbias0, wsum0);
                            bias01 = _mm512_mul_ps(inputbias1, wsum0);
                            bias10 = _mm512_mul_ps(inputbias0, wsum1);
                            bias11 = _mm512_mul_ps(inputbias1, wsum1);
                            bias20 = _mm512_mul_ps(inputbias0, wsum2);
                            bias21 = _mm512_mul_ps(inputbias1, wsum2);
                            bias30 = _mm512_mul_ps(inputbias0, wsum3);
                            bias31 = _mm512_mul_ps(inputbias1, wsum3);
                        } else if (bk == 0) { // if input not block quant, only accum once!
                            weightKernelSum_dz = post->weightKernelSum + dz * GEMMINT8_AVX512_H;
                            auto wsum0 = _mm512_loadu_ps(weightKernelSum_dz);
                            auto wsum1 = _mm512_loadu_ps(weightKernelSum_dz + 1 * PACK_UNIT);
                            auto wsum2 = _mm512_loadu_ps(weightKernelSum_dz + 2 * PACK_UNIT);
                            auto wsum3 = _mm512_loadu_ps(weightKernelSum_dz + 3 * PACK_UNIT);
                            bias00 = _mm512_mul_ps(_mm512_mul_ps(inputscale0, neg128f), wsum0);
                            bias01 = _mm512_mul_ps(_mm512_mul_ps(inputscale1, neg128f), wsum0);
                            bias10 = _mm512_mul_ps(_mm512_mul_ps(inputscale0, neg128f), wsum1);
                            bias11 = _mm512_mul_ps(_mm512_mul_ps(inputscale1, neg128f), wsum1);
                            bias20 = _mm512_mul_ps(_mm512_mul_ps(inputscale0, neg128f), wsum2);
                            bias21 = _mm512_mul_ps(_mm512_mul_ps(inputscale1, neg128f), wsum2);
                            bias30 = _mm512_mul_ps(_mm512_mul_ps(inputscale0, neg128f), wsum3);
                            bias31 = _mm512_mul_ps(_mm512_mul_ps(inputscale1, neg128f), wsum3);
                        }
                        f0 = _mm512_add_ps(f0, bias00);
                        f1 = _mm512_add_ps(f1, bias01);
                        f4 = _mm512_add_ps(f4, bias10);
                        f5 = _mm512_add_ps(f5, bias11);
                        f8 = _mm512_add_ps(f8, bias20);
                        f9 = _mm512_add_ps(f9, bias21);
                        f12 = _mm512_add_ps(f12, bias30);
                        f13 = _mm512_add_ps(f13, bias31);
                    }
                }
                f0 = _mm512_add_ps(_mm512_mul_ps(kernelSum0, weightBiasValue0), f0);
                f1 = _mm512_add_ps(_mm512_mul_ps(kernelSum1, weightBiasValue0), f1);
                f4 = _mm512_add_ps(_mm512_mul_ps(kernelSum0, weightBiasValue1), f4);
                f5 = _mm512_add_ps(_mm512_mul_ps(kernelSum1, weightBiasValue1), f5);
                f8 = _mm512_add_ps(_mm512_mul_ps(kernelSum0, weightBiasValue2), f8);
                f9 = _mm512_add_ps(_mm512_mul_ps(kernelSum1, weightBiasValue2), f9);
                f12 = _mm512_add_ps(_mm512_mul_ps(kernelSum0, weightBiasValue3),f12);
                f13 = _mm512_add_ps(_mm512_mul_ps(kernelSum1, weightBiasValue3),f13);

                if (bk > 0) {
                    f0 = _mm512_add_ps(_mm512_loadu_ps(accum_x), f0);
                    f1 = _mm512_add_ps(_mm512_loadu_ps(accum_x + 16), f1);

                    f4 = _mm512_add_ps(_mm512_loadu_ps(accum_x + source_step), f4);
                    f5 = _mm512_add_ps(_mm512_loadu_ps(accum_x + source_step + 16 * 1), f5);

                    f8 = _mm512_add_ps(_mm512_loadu_ps(accum_x + 2 * source_step), f8);
                    f9 = _mm512_add_ps(_mm512_loadu_ps(accum_x + 2 * source_step + 16 * 1), f9);

                    f12 = _mm512_add_ps(_mm512_loadu_ps(accum_x + 3 * source_step), f12);
                    f13 = _mm512_add_ps(_mm512_loadu_ps(accum_x + 3 * source_step + 16 * 1), f13);
                }
                if (bk == blockNum - 1) {
                    if (biasPtr) {
                        auto biasValue0 = _mm512_loadu_ps(bias_dz);
                        auto biasValue4 = _mm512_loadu_ps(bias_dz + 1 * PACK_UNIT);
                        auto biasValue8 = _mm512_loadu_ps(bias_dz + 2 * PACK_UNIT);
                        auto biasValue12 = _mm512_loadu_ps(bias_dz + 3 * PACK_UNIT);
                        f0 = _mm512_add_ps(f0, biasValue0);
                        f1 = _mm512_add_ps(f1, biasValue0);
                        f4 = _mm512_add_ps(f4, biasValue4);
                        f5 = _mm512_add_ps(f5, biasValue4);
                        f8 = _mm512_add_ps(f8, biasValue8);
                        f9 = _mm512_add_ps(f9, biasValue8);
                        f12 = _mm512_add_ps(f12, biasValue12);
                        f13 = _mm512_add_ps(f13, biasValue12);
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
                    _mm512_storeu_ps(accum_x, f0);
                    _mm512_storeu_ps(accum_x + 16, f1);
                    _mm512_storeu_ps(accum_x + source_step, f4);
                    _mm512_storeu_ps(accum_x + source_step + 16 * 1, f5);
                    _mm512_storeu_ps(accum_x + 2 * source_step, f8);
                    _mm512_storeu_ps(accum_x + 2 * source_step + 16 * 1, f9);
                    _mm512_storeu_ps(accum_x + 3 * source_step, f12);
                    _mm512_storeu_ps(accum_x + 3 * source_step + 16 * 1, f13);
                }
            }
        } // dzU
        // the remaining ocDivPack
        auto weight_dz = weight + dzU * blockNum * weight_step_Z;                                            // weight address for remaining
        if (biasPtr) {
            bias_dz = post->biasFloat + dzU * PACK_UNIT * dzUnit;
        }

        auto dst_x = dst + dzU * dst_step_tmp * dzUnit;
        for (int i=0; i<dzR; ++i) {
            auto accum_x = accumbuff;
            for (int bk = 0; bk < blockNum; ++bk) {
                __m512i D0 = _mm512_set1_epi32(0);
                __m512i D1 = _mm512_set1_epi32(0);
                auto weightDzSub = weight_dz + bk * weight_step_Z + weightPackStride * suborder[i];
                auto scaleDz = (float*)(weight_dz + bk * weight_step_Z + src_depth_quad * weight_step_Y);
                 auto biasDz = scaleDz + GEMMINT8_AVX512_H;
                const auto src_x = src + bk * src_depth_quad * GEMMINT8_AVX512_L * realDst;

                for (int sz = 0; sz < src_depth_quad; ++sz) {
                    const auto weight_sz = weightDzSub + weight_step_Y * sz;
                    const auto src_z     = (const float*)(src_x + sz * realDst * GEMMINT8_AVX512_L);
                    auto w0_int4_64 = _mm512_loadu_si512(weight_sz); // 128xint4_t=64 byte
                    // 256xint4_t->256xint8_t
                    auto w0 = _mm512_and_si512(mask, _mm512_srli_epi16(w0_int4_64, 4)); // 64xint8_t

                    auto s0 = AVX512_BROADCAST_INT32(src_z + 0);
                    auto s1 = AVX512_BROADCAST_INT32(src_z + 1);

                    D0 = _mm512_dpbusds_epi32(D0, s0, w0);
                    D1 = _mm512_dpbusds_epi32(D1, s1, w0);
                }

                auto scaleValue0 = _mm512_loadu_ps(scaleDz + i * PACK_UNIT);
                auto weightBiasValue0 = _mm512_loadu_ps(biasDz + i * PACK_UNIT);
                // input info
                kernelSum0 = _mm512_set1_ps((post->srcKernelSum + bk * realDst)[0]);
                kernelSum1 = _mm512_set1_ps((post->srcKernelSum + bk * realDst)[1]);
                if (post->inputBias) {
                    inputscale0 = _mm512_set1_ps((post->inputScale + bk * realDst)[0]);
                    inputscale1 = _mm512_set1_ps((post->inputScale + bk * realDst)[1]);
                    inputbias0 = _mm512_set1_ps((post->inputBias + bk * realDst)[0]);
                    inputbias1 = _mm512_set1_ps((post->inputBias + bk * realDst)[1]);
                }
                MUL_WEIGHT_SCALE(0, 0);
                MUL_WEIGHT_SCALE(1, 0);

                if (post->inputScale) { // Batch quant
                    f0 = _mm512_mul_ps(f0, inputscale0);
                    f1 = _mm512_mul_ps(f1, inputscale1);
                    if ((post->useInt8 == 0) && post->weightKernelSum && (post->inputBias || (bk == 0))) {
                        if (post->inputBias) {
                            weightKernelSum_dz = post->weightKernelSum + dzU * blockNum * GEMMINT8_AVX512_H + bk * GEMMINT8_AVX512_H;
                            auto wsum0 = _mm512_loadu_ps(weightKernelSum_dz + i * PACK_UNIT);
                            bias00 = _mm512_mul_ps(inputbias0, wsum0);
                            bias01 = _mm512_mul_ps(inputbias1, wsum0);
                        } else if (bk == 0) { // if input not block quant, only accum once!
                            weightKernelSum_dz = post->weightKernelSum + dzU * PACK_UNIT * dzUnit;
                            auto wsum0 = _mm512_loadu_ps(weightKernelSum_dz + i * PACK_UNIT);
                            bias00 = _mm512_mul_ps(_mm512_mul_ps(inputscale0, neg128f), wsum0);
                            bias01 = _mm512_mul_ps(_mm512_mul_ps(inputscale1, neg128f), wsum0);
                        }
                        f0 = _mm512_add_ps(f0, bias00);
                        f1 = _mm512_add_ps(f1, bias01);
                    }
                }
                f0 = _mm512_add_ps(_mm512_mul_ps(kernelSum0, weightBiasValue0), f0);
                f1 = _mm512_add_ps(_mm512_mul_ps(kernelSum1, weightBiasValue0), f1);

                if (bk > 0) {
                    f0 = _mm512_add_ps(_mm512_loadu_ps((float*)accum_x), f0);
                    f1 = _mm512_add_ps(_mm512_loadu_ps(((float*)accum_x) + 16), f1);
                }
                if (bk == blockNum - 1) {
                    if (nullptr != biasPtr) {
                        auto biasValue0 = _mm512_loadu_ps(bias_dz + i * PACK_UNIT);
                        f0 = _mm512_add_ps(f0, biasValue0);
                        f1 = _mm512_add_ps(f1, biasValue0);
                    }
                    if (post->fp32minmax) {
                        POST_TREAT_FLOAT_2(0,1);
                    }
                    _mm512_storeu_ps((float*)(dst_x + i * dst_step_tmp), f0);
                    _mm512_storeu_ps((float*)(dst_x + i * dst_step_tmp) + 16, f1);
                } else {
                    _mm512_storeu_ps(((float*)accum_x), f0);
                    _mm512_storeu_ps(((float*)accum_x) + 16, f1);
                }
            }
        }
        return;
    }
    if (realDst == 1) {
        for (int dz = 0; dz < dzU; ++dz) {
            if (biasPtr) {
                bias_dz = post->biasFloat + dz * PACK_UNIT * dzUnit;
            }
            auto dst_x = dst + dz * dst_step_tmp * dzUnit;
            auto accum_x = accumbuff;

            for (int bk = 0; bk < blockNum; ++bk) {
                __m512i D0 = _mm512_set1_epi32(0);
                __m512i D4 = _mm512_set1_epi32(0);
                __m512i D8 = _mm512_set1_epi32(0);
                __m512i D12 = _mm512_set1_epi32(0);

                // block's weight&scale&bias
                const auto weight_dz = weight + dz * blockNum * weight_step_Z + bk *  weight_step_Z;
                const float* scale_dz = reinterpret_cast<const float*>(weight_dz + src_depth_quad * weight_step_Y);
                const auto weightBias_dz = scale_dz + GEMMINT8_AVX512_H;
                // block's input
                const auto src_x = src + bk * src_depth_quad * GEMMINT8_AVX512_L * realDst;

                for (int sz = 0; sz < src_depth_quad; ++sz) {
                    const auto weight_sz = weight_dz + weight_step_Y * sz;
                    const auto src_z     = (const float*)(src_x + sz * realDst * GEMMINT8_AVX512_L);
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
                // int32_t -> float
                auto scaleValue0 = _mm512_loadu_ps(scale_dz);
                auto scaleValue1 = _mm512_loadu_ps(scale_dz + 1 * PACK_UNIT);
                auto scaleValue2 = _mm512_loadu_ps(scale_dz + 2 * PACK_UNIT);
                auto scaleValue3 = _mm512_loadu_ps(scale_dz + 3 * PACK_UNIT);
                auto weightBiasValue0 = _mm512_loadu_ps(weightBias_dz);
                auto weightBiasValue1 = _mm512_loadu_ps(weightBias_dz + 1 * PACK_UNIT);
                auto weightBiasValue2 = _mm512_loadu_ps(weightBias_dz + 2 * PACK_UNIT);
                auto weightBiasValue3 = _mm512_loadu_ps(weightBias_dz + 3 * PACK_UNIT);
                // input info
                kernelSum0 = _mm512_set1_ps((post->srcKernelSum + bk * realDst)[0]);
                if (post->inputBias) {
                    inputscale0 = _mm512_set1_ps((post->inputScale + bk * realDst)[0]);
                    inputbias0 = _mm512_set1_ps((post->inputBias + bk * realDst)[0]);
                }

                MUL_WEIGHT_SCALE(0, 0);
                MUL_WEIGHT_SCALE(4, 1);
                MUL_WEIGHT_SCALE(8, 2);
                MUL_WEIGHT_SCALE(12, 3);

                if (post->inputScale) { // Batch quant
                    f0 = _mm512_mul_ps(f0, inputscale0);
                    f4 = _mm512_mul_ps(f4, inputscale0);
                    f8 = _mm512_mul_ps(f8, inputscale0);
                    f12 = _mm512_mul_ps(f12, inputscale0);
                    if ((post->useInt8 == 0) && post->weightKernelSum && (post->inputBias || (bk == 0))) {
                        if (post->inputBias) {
                            weightKernelSum_dz = post->weightKernelSum + bk * GEMMINT8_AVX512_H + dz * blockNum * GEMMINT8_AVX512_H;
                            auto wsum0 = _mm512_loadu_ps(weightKernelSum_dz);
                            auto wsum1 = _mm512_loadu_ps(weightKernelSum_dz + 1 * PACK_UNIT);
                            auto wsum2 = _mm512_loadu_ps(weightKernelSum_dz + 2 * PACK_UNIT);
                            auto wsum3 = _mm512_loadu_ps(weightKernelSum_dz + 3 * PACK_UNIT);
                            bias00 = _mm512_mul_ps(inputbias0, wsum0);
                            bias01 = _mm512_mul_ps(inputbias0, wsum1);
                            bias02 = _mm512_mul_ps(inputbias0, wsum2);
                            bias03 = _mm512_mul_ps(inputbias0, wsum3);
                        } else if (bk == 0) { // if input not block quant, only accum once!
                            weightKernelSum_dz = post->weightKernelSum + dz * GEMMINT8_AVX512_H;
                            auto wsum0 = _mm512_loadu_ps(weightKernelSum_dz);
                            auto wsum1 = _mm512_loadu_ps(weightKernelSum_dz + 1 * PACK_UNIT);
                            auto wsum2 = _mm512_loadu_ps(weightKernelSum_dz + 2 * PACK_UNIT);
                            auto wsum3 = _mm512_loadu_ps(weightKernelSum_dz + 3 * PACK_UNIT);
                            bias00 = _mm512_mul_ps(_mm512_mul_ps(inputscale0, neg128f), wsum0);
                            bias01 = _mm512_mul_ps(_mm512_mul_ps(inputscale0, neg128f), wsum1);
                            bias02 = _mm512_mul_ps(_mm512_mul_ps(inputscale0, neg128f), wsum2);
                            bias03 = _mm512_mul_ps(_mm512_mul_ps(inputscale0, neg128f), wsum3);
                        }
                        f0 = _mm512_add_ps(f0, bias00);
                        f4 = _mm512_add_ps(f4, bias01);
                        f8 = _mm512_add_ps(f8, bias02);
                        f12 = _mm512_add_ps(f12, bias03);
                    }
                }
                f0 = _mm512_add_ps(_mm512_mul_ps(kernelSum0, weightBiasValue0), f0);
                f4 = _mm512_add_ps(_mm512_mul_ps(kernelSum0, weightBiasValue1), f4);
                f8 = _mm512_add_ps(_mm512_mul_ps(kernelSum0, weightBiasValue2), f8);
                f12 = _mm512_add_ps(_mm512_mul_ps(kernelSum0, weightBiasValue3), f12);

                if (bk > 0) { // Add accumbuffer if blockNum>1
                    f0 = _mm512_add_ps(_mm512_loadu_ps(accum_x), f0);
                    f4 = _mm512_add_ps(_mm512_loadu_ps(accum_x + source_step), f4);
                    f8 = _mm512_add_ps(_mm512_loadu_ps(accum_x + 2 * source_step), f8);
                    f12 = _mm512_add_ps(_mm512_loadu_ps(accum_x + 3 * source_step), f12);
                }
                if (bk == blockNum - 1) { // If last block, post process before saving to dest address.
                    if (biasPtr) {
                        auto biasValue0 = _mm512_loadu_ps(bias_dz);
                        auto biasValue4 = _mm512_loadu_ps(bias_dz + 1 * PACK_UNIT);
                        auto biasValue8 = _mm512_loadu_ps(bias_dz + 2 * PACK_UNIT);
                        auto biasValue12 = _mm512_loadu_ps(bias_dz + 3 * PACK_UNIT);
                        f0 = _mm512_add_ps(f0, biasValue0);
                        f4 = _mm512_add_ps(f4, biasValue4);
                        f8 = _mm512_add_ps(f8, biasValue8);
                        f12 = _mm512_add_ps(f12, biasValue12);
                    }
                    if (post->fp32minmax) {
                        POST_TREAT_FLOAT_1(0);
                        POST_TREAT_FLOAT_1(4);
                        POST_TREAT_FLOAT_1(8);
                        POST_TREAT_FLOAT_1(12);
                    }
                    _mm512_storeu_ps((float*)dst_x, f0);
                    _mm512_storeu_ps((float*)(dst_x + dst_step_tmp), f4);
                    _mm512_storeu_ps((float*)(dst_x + 2 * dst_step_tmp), f8);
                    _mm512_storeu_ps((float*)(dst_x + 3 * dst_step_tmp), f12);
                } else { // save to accumbuffer to added to next block
                    _mm512_storeu_ps(accum_x, f0);
                    _mm512_storeu_ps(accum_x + source_step, f4);
                    _mm512_storeu_ps(accum_x + 2 * source_step, f8);
                    _mm512_storeu_ps(accum_x + 3 * source_step, f12);
                }
            }
        }
        // the remaining ocDivPack
        auto weight_dz = weight + dzU * blockNum * weight_step_Z;                                            // weight address for remaining
        if (biasPtr) {
            bias_dz = post->biasFloat + dzU * PACK_UNIT * dzUnit;
        }

        auto dst_x = dst + dzU * dst_step_tmp * dzUnit;
        for (int i=0; i<dzR; ++i) {
            auto accum_x = accumbuff;
            for (int bk = 0; bk < blockNum; ++bk) {
                auto weightDzSub = weight_dz + bk * weight_step_Z + weightPackStride * suborder[i];
                auto scaleDz = (float*)(weight_dz + bk * weight_step_Z + src_depth_quad * weight_step_Y);
                 auto biasDz = scaleDz + GEMMINT8_AVX512_H;
                const auto src_x = src + bk * src_depth_quad * GEMMINT8_AVX512_L * realDst;

                __m512i D0 = _mm512_set1_epi32(0);

                for (int sz = 0; sz < src_depth_quad; ++sz) {
                    const auto weight_sz = weightDzSub + weight_step_Y * sz;
                    const auto src_z     = (const float*)(src_x + sz * realDst * GEMMINT8_AVX512_L);
                    auto w0_int4_64 = _mm512_loadu_si512(weight_sz); // 128xint4_t=64 byte
                    auto w0 = _mm512_and_si512(mask, _mm512_srli_epi16(w0_int4_64, 4)); // 64xint8_t
                    auto s0 = AVX512_BROADCAST_INT32(src_z + 0);
                    D0 = _mm512_dpbusds_epi32(D0, s0, w0);
                }

                auto scaleValue0 = _mm512_loadu_ps(scaleDz + i * PACK_UNIT);
                auto weightBiasValue0 = _mm512_loadu_ps(biasDz + i * PACK_UNIT);
                // input info
                kernelSum0 = _mm512_set1_ps((post->srcKernelSum + bk * realDst)[0]);
                if (post->inputBias) {
                    inputscale0 = _mm512_set1_ps((post->inputScale + bk * realDst)[0]);
                    inputbias0 = _mm512_set1_ps((post->inputBias + bk * realDst)[0]);
                }
                MUL_WEIGHT_SCALE(0, 0);

                if (post->inputScale) { // Batch quant
                    f0 = _mm512_mul_ps(f0, inputscale0);
                    if ((post->useInt8 == 0) && post->weightKernelSum && (post->inputBias || (bk == 0))) {
                        if (post->inputBias) {
                            weightKernelSum_dz = post->weightKernelSum + dzU * blockNum * GEMMINT8_AVX512_H + bk * GEMMINT8_AVX512_H;
                            auto wsum0 = _mm512_loadu_ps(weightKernelSum_dz + i * PACK_UNIT);
                            bias00 = _mm512_mul_ps(inputbias0, wsum0);
                        } else if (bk == 0) { // if input not block quant, only accum once!
                            weightKernelSum_dz = post->weightKernelSum + dzU * PACK_UNIT * dzUnit;
                            auto wsum0 = _mm512_loadu_ps(weightKernelSum_dz + i * PACK_UNIT);
                            bias00 = _mm512_mul_ps(_mm512_mul_ps(inputscale0, neg128f), wsum0);
                        }
                        f0 = _mm512_add_ps(f0, bias00);
                    }
                }
                f0 = _mm512_add_ps(_mm512_mul_ps(kernelSum0, weightBiasValue0), f0);

                if (bk > 0) {
                    f0 = _mm512_add_ps(_mm512_loadu_ps(accum_x), f0);
                }
                if (bk == blockNum - 1) {
                    if (biasPtr) {
                        auto biasValue = _mm512_loadu_ps(bias_dz + i * PACK_UNIT);
                        SCALE_BIAS_VEC(0);
                    }
                    if (post->fp32minmax) {
                        POST_TREAT_FLOAT_1(0);
                    }
                    _mm512_storeu_ps((float*)(dst_x + i * dst_step_tmp), f0);
                } else {
                    _mm512_storeu_ps(((float*)accum_x), f0);
                }
            }
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
