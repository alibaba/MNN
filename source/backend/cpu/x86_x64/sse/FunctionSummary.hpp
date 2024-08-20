//
//  FunctionSummary.hpp
//  MNN
//
//  Created by MNN on 2019/08/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include <MNN/MNNDefine.h>
#include <stdint.h>
#include "backend/cpu/compute/Int8FunctionsOpt.h"
#include "backend/cpu/compute/CommonOptFunction.h"

#ifndef _MM_TRANSPOSE4_PS
#define _MM_TRANSPOSE4_PS(row0, row1, row2, row3) \
    do {                                          \
        __m128 tmp3, tmp2, tmp1, tmp0;            \
        tmp0   = _mm_unpacklo_ps((row0), (row1)); \
        tmp2   = _mm_unpacklo_ps((row2), (row3)); \
        tmp1   = _mm_unpackhi_ps((row0), (row1)); \
        tmp3   = _mm_unpackhi_ps((row2), (row3)); \
        (row0) = _mm_movelh_ps(tmp0, tmp2);       \
        (row1) = _mm_movehl_ps(tmp2, tmp0);       \
        (row2) = _mm_movelh_ps(tmp1, tmp3);       \
        (row3) = _mm_movehl_ps(tmp3, tmp1);       \
    } while (0)
#endif
void _SSE_MNNAxByClampBroadcastUnit(float* C, const float* A, const float* B, size_t width, size_t cStride, size_t aStride, size_t height, const float* parameters);

void _SSE_MNNCopyC4WithStride(const float* source, float* dest, size_t srcStride, size_t dstStride, size_t count);

void _SSE_MNNAddC4WithStride(const float* source, float* dest, size_t srcStride, size_t dstStride, size_t count);

void _SSE_MNNReluWithSlopeChannel(float* dst, const float* src, const float* slope, size_t sizeQuad, size_t depthQuad);

void _SSE_MNNGelu(float* dst, const float* src, size_t size, float* parameters);
void _SSE_MNNReluWithSlopeChannelInt8(int8_t* dst, const int8_t* src, const float* slope, size_t planeNumber, size_t depthQuad, QuanPrePostParameters *params);

void _SSE_MNNHardSwish(float* dst, const float* src, size_t size);

void _SSE_MNNStrassenMergeCFunction(float* c11, float* c12, float* c21, float* c22, float* xAddr, size_t cStride,
                                    size_t length, size_t hSub);

void _SSE_MNNPackedMatMul(float* C, const float* A, const float* B, const size_t* parameter,
                          const float* postParameters, const float* bias, const float* k, const float* b);
void _SSE_MNNPackedMatMulRemain(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter,
                                 const float* postParameters, const float* bias, const float* k, const float* b);
#ifdef MNN_LOW_MEMORY
void _SSE_MNNPackedMatMul_int4(float* C, const float* A, const float* B, const size_t* parameter,
                               const float* postParameters, const float* bias, const float* k, const float* b);
void _SSE_MNNPackedMatMulRemain_int4(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter,
                                     const float* postParameters, const float* bias, const float* k, const float* b);
void _SSE_MNNPackedMatMul_int8(float* C, const float* A, const float* B, const size_t* parameter,
                               const float* postParameters, const float* bias, const float* k, const float* b);
void _SSE_MNNPackedMatMulRemain_int8(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter,
                                     const float* postParameters, const float* bias, const float* k, const float* b);
void _SSE_MNNAbsMaxFP32(const float* source, float* absmax, size_t src_depth_quad, size_t realSize, int pack);
void _SSE_MNNGemmInt8AddBiasScale_16x4_w4(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step,
                                            size_t dst_depth_quad, const QuanPostTreatParameters* post, size_t realDst);
#endif
void _SSE_MNNPackC4ForMatMul_A(float* destOrigin, float const** sourceGroup, const int32_t* info, const int32_t* el);
void _SSE_MNNConvRunForLineDepthwise(float* dst, const float* src, const float* weight, size_t width, size_t src_w_setup,
                                size_t fw, size_t fh, size_t dilateX_step, size_t dilateY_step, size_t height,
                                size_t srcHStep, size_t dstHStep);
void _SSE_MNNGemmInt8AddBiasScale_16x4_Unit(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step,
                                            size_t dst_depth_quad, const QuanPostTreatParameters* post, size_t realDst);
void _SSE_MNNExpC8(float* dest, const float* source, float* offset, const float* parameters, size_t countC8);
void _SSE_MNNPackForMatMul_B(float* dest, const float* source, size_t h, size_t l, bool transpose);
void _SSE_MNNFloat2Int8(const float* src, int8_t* dst, size_t sizeQuad, const float* scalep, ssize_t minValue, ssize_t maxValue, ssize_t zeroPoint);

void _SSE_MNNInt8ScaleToFloat(float* dst, const int8_t* src, const float* scale, size_t size, ssize_t zeroPoint);
void _SSE_MNNLineDepthWiseInt8AddBiasScaleUnit(int8_t* dst, const int8_t* src, const int8_t* weight, const QuanPostTreatParameters* parameters, size_t width, size_t src_w_step, size_t fw, size_t fh, size_t dilateX_step, size_t dilateY_step, int8_t* idxOrder=nullptr);
void _SSE_MNNInt8ToInt16(int16_t* dest, const int8_t* source, size_t count);

void _SSE_MNNPackForMatMul_B_BF16(float* dest, const float* source, size_t h, size_t l, bool transpose);
void _SSE_MNNReluInt8(int8_t* dst, const int8_t* src, size_t size, ssize_t zeroPoint);
void _SSE_MNNSoftmax(float* dest, const float* source, size_t size);
void _SSE_ExtraInit(void* functions);
void _SSE_MNNNorm(float *dst, const float *src, const float *gamma, const float *beta, float epsilon, size_t size, bool RMSNorm);
void _SSE_ImageProcessInit(void* functions, int cpuFlags);
void _SSE_MNNNormInt8(int8_t* dst, const int8_t* src, const float* gamma, const float* beta, float epsilon, size_t size, QuanPrePostParameters* params, bool RMSNorm);

/* Image process functions */
void _SSE_MNNRGBAToBGRA(const unsigned char* source, unsigned char* dest, size_t count);
void _SSE_MNNNV21ToRGB(const unsigned char* source, unsigned char* dest, size_t count);
void _SSE_MNNNV21ToRGBA(const unsigned char* source, unsigned char* dest, size_t count);
void _SSE_MNNNV21ToBGRA(const unsigned char* source, unsigned char* dest, size_t count);
void _SSE_MNNNV21ToBGR(const unsigned char* source, unsigned char* dest, size_t count);
void _SSE_MNNC1ToFloatC1(const unsigned char* source, float* dest, const float* mean, const float* normal, size_t count);
void _SSE_MNNC3ToFloatC3(const unsigned char* source, float* dest, const float* mean, const float* normal, size_t count);
void _SSE_MNNC3ToFloatRGBA(const unsigned char* source, float* dest, const float* mean, const float* normal, size_t count);
void _SSE_MNNSamplerC4Nearest(const unsigned char* source, unsigned char* dest, MNN::CV::Point* points, size_t sta,
                              size_t count, size_t capacity, size_t iw, size_t ih, size_t yStride);
void _SSE_MNNSamplerNearest(const unsigned char* source, unsigned char* dest, MNN::CV::Point* points, size_t sta, size_t count,
                            size_t iw, size_t ih, size_t yStride, int bpp);
void _SSE_MNNSampleC4Bilinear(const unsigned char* source, unsigned char* dest, MNN::CV::Point* points, size_t sta,
                          size_t count, size_t capacity, size_t iw, size_t ih, size_t yStride);
void _SSE_MNNSampleBilinear(const unsigned char* source, unsigned char* dest, MNN::CV::Point* points, size_t count,
                                  size_t iw, size_t ih, size_t yStride, size_t bpp);

// Dynamic Quant
void _SSE_MNNComputeScaleZeroScalar(float* source, float* minVal, float* maxVal, size_t size);