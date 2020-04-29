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

#if defined(_MSC_VER)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*INT8GEMM_KERNEL)(int8_t* dst, const int8_t* src, const int8_t* weight, const int32_t* bias, const float* scale, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad);

void MNNConvRunForUnitDepthWiseInt8(float* dst, const int8_t* src, const int8_t* weight, size_t fw, size_t fh,
                                    size_t weight_y_step, size_t dilateX_step, size_t dilateY_step, const float* scale);
void MNNFloat2Int8(const float* src, int8_t* dst, size_t sizeQuad, const float* scalep, ssize_t minValue,
                   ssize_t maxValue);
void MNNGemmInt8AddBiasScale_16x4_Unit(int8_t* dst, const int8_t* src, const int8_t* weight, const int32_t* bias,
                                       const float* scale, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad);

// int8x16 * int8x16
void MNNGemmInt8AddBiasScale_16x4_Unit_FAST(int8_t* dst, const int8_t* src, const int8_t* weight, const int32_t* bias,
const float* scale, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad);

#define GEMM_INT8_UNIT 4
#define GEMM_INT8_SRC_UNIT 16

#ifdef __aarch64__
#define GEMM_INT8_DST_XUNIT 4
#else
#define GEMM_INT8_DST_XUNIT 2
#endif

#ifdef ENABLE_ARMV82
void MNNGemmInt8AddBiasScale_ARMV82_Unit(int8_t* dst, const int8_t* src, const int8_t* weight, const int32_t* bias, const float* scale, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad, size_t relu, size_t realDstCount);
// default TILE size
#define DST_XUNIT_ARMV82 16

#endif

#ifdef __cplusplus
}
#endif
#endif /* Int8FunctionsOpt_h */
