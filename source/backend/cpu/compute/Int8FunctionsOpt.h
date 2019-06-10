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

void MNNConvolutionInt8Run8x8(int16_t* dst_x, const int8_t* src_unit, const int8_t* weight_start, size_t icD8,
                              size_t xCount, size_t yCount, size_t dilateY_step, size_t dilateX_step,
                              size_t weight_sy_step);
void MNNScaleBias2FloatC4(float* dst, const int16_t* src, const float* alpha, const float* bias, size_t sizeQuad);
void MNNScaleBias2FloatC4Relu(float* dst, const int16_t* src, const float* alpha, const float* bias, size_t sizeQuad);
void MNNScaleBias2FloatC4Relu6(float* dst, const int16_t* src, const float* alpha, const float* bias, size_t sizeQuad);

void MNNConvRunForUnitDepthWiseInt8(float* dst, const int8_t* src, const int8_t* weight, size_t fw, size_t fh,
                                    size_t weight_y_step, size_t dilateX_step, size_t dilateY_step, const float* scale);
void MNNFloat2Int8(const float* src, int8_t* dst, size_t sizeQuad, const float* scalep, ssize_t minValue,
                   ssize_t maxValue);

#ifdef __cplusplus
}
#endif
#endif /* Int8FunctionsOpt_h */
