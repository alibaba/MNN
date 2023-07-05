//
//  ResizeFunction.h
//  MNN
//
//  Created by MNN on 2018/07/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ResizeFunction_h
#define ResizeFunction_h

#include <stdint.h>
#include <stdio.h>
#include "core/Macro.h"
#ifdef __cplusplus
extern "C" {
#endif

void MNNCubicSampleC4(const float* src, float* dst, int32_t* position, const float* factor, int8_t* zeroPoint, size_t number);
void MNNCubicLineC4(float* dst, const float* A, const float* B, const float* C, const float* D, float* t, int8_t* zeroPoint, 
                    size_t number, ssize_t minValue, ssize_t maxValue);
void CPUBilinearSampleC4(const float* src, float* dst, const int32_t* position, const float* factor, int8_t* zeroPoint, size_t number);
void CPUBilinearLineC4(float* dst, const float* A, const float* B, const float* t, int8_t* zeroPoint, size_t number);
void MNNCubicSampleC16(const int8_t* src, float* dst, int32_t* position, const float* factor, int8_t* zeroPoint, size_t number);
void MNNCubicLineC16(int8_t* dst, const float* A, const float* B, const float* C, const float* D, float* t, int8_t* zeroPoint,
                     size_t number, ssize_t minValue, ssize_t maxValue);
void MNNBilinearSampleC8(const int8_t* src, int16_t* dst, const int32_t* position, const float* factor, int8_t* zeroPoint, size_t number);
void MNNBilinearLineC8(int8_t* dst, const int16_t* A, const int16_t* B, const float* t, int8_t* zeroPoint, size_t number);
#ifdef __cplusplus
}
#endif

#endif /* ResizeFunction_hpp */
