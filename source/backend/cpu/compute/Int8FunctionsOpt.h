//
//  Int8FunctionsOpt.h
//  MNN
//
//  Created by MNN on 2018/08/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef Int8FunctionsOpt_h
#define Int8FunctionsOpt_h

#include <stdint.h>
#include <stdio.h>
#include <sys/types.h>
#include "core/Macro.h"
#if defined(_MSC_VER)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

#ifdef __aarch64__
#define DST_XUNIT 6 // used by MNNGemmInt8toFloat32_8x4_Unit
#else
#define DST_XUNIT 2
#endif

void MNNInt8C4ToC8(int8_t* dst, const int8_t* src, size_t area, size_t depth);
void MNNInt8ClipInplace(int8_t* data, size_t size, int8_t minVal, int8_t maxVal);

#ifdef __cplusplus
extern "C" {
#endif

void MNNConvRunForUnitDepthWiseInt8(float* dst, const int8_t* src, const int8_t* weight, size_t fw, size_t fh,
                                    size_t weight_y_step, size_t dilateX_step, size_t dilateY_step, const float* scale);
void MNNFloat2Int8(const float* src, int8_t* dst, size_t sizeQuad, const float* scalep, ssize_t minValue,
                   ssize_t maxValue, ssize_t zeroPoint);

void MNNInt8ScaleToFloat(float* dst, const int8_t* src, const float* scale, size_t size, ssize_t zeroPoint);

void MNNInt8ToInt16C4(const int8_t* source, int16_t* dest, size_t sizeQuad);

void MNNGemmInt8toFloat32_8x4_Unit(float* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad,
                                   size_t dst_step, size_t dst_depth_quad);
void MNNGemmInt8toFloat32_8x4_Common(float* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad,
                                     size_t width, size_t dst_step, size_t dst_depth_quad);

void MNNMatrixAddInt32(int32_t* C, const int32_t* A, const int32_t* B, size_t widthC4, size_t cStride,
                       size_t aStride, size_t bStride, size_t height);
// int8x16 * int8x16
#define GEMM_INT8_UNIT 4
#define GEMM_INT8_SRC_UNIT 16

#ifndef MNN_USE_SSE
#ifdef __aarch64__
#define GEMM_INT8_DST_XUNIT 4
#else
#define GEMM_INT8_DST_XUNIT 2
#endif
#else
#define GEMM_INT8_DST_XUNIT 4
#endif

struct QuanPostTreatParameters {
    const float* scale;
    const int32_t* bias;
    int32_t maxValue;
    int32_t minValue;
    float roundValuePos = 0.5f;
    float roundValueNeg = -0.5f;
};
void MNNGemmInt8AddBiasScale_16x4_Unit(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad, const QuanPostTreatParameters* post, size_t realCount);
void MNNGemmInt8AddBiasScale_16x4_Unit_FAST(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad, const QuanPostTreatParameters* post, size_t realCount);

#if defined(__aarch64__) && defined(ENABLE_ARMV82)
void MNNGemmInt8AddBiasScale_ARMV82_Unit(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad, size_t realDstCount, const QuanPostTreatParameters* parameters);
// default TILE size
#define DST_XUNIT_ARMV82 16

#endif
int8_t MNNInt32ToInt8(int data, int bias, float scale, float maxValue ,float minValue);
void MNNLineDepthWiseInt8AddBiasScaleUnit(int8_t* dst, const int8_t* src, const int8_t* weight, const QuanPostTreatParameters* parameters,
                                          size_t width, size_t src_w_step, size_t fw, size_t fh, size_t dilateX_step,
                                          size_t dilateY_step);
#ifdef __cplusplus
}
#endif
#endif /* Int8FunctionsOpt_h */
