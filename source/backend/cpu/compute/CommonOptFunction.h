//
//  CommonOptFunction.h
//  MNN
//
//  Created by MNN on 2018/07/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CommonOptFunction_h
#define CommonOptFunction_h

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "core/Macro.h"

#ifdef __cplusplus
extern "C" {
#endif

void MNNAddBias(float* dst, const float* bias, size_t planeNumber, size_t biasNumber);
void MNNAddBiasRelu(float* dst, const float* bias, size_t planeNumber, size_t biasNumber);
void MNNAddBiasRelu6(float* dst, const float* bias, size_t planeNumber, size_t biasNumber);

void MNNReluWithSlope(float* dst, const float* src, size_t sizeQuad, float slope);

void MNNRelu6(float* dst, const float* src, size_t size);

void MNNReluInt8(int8_t* dst, const int8_t* src, size_t size);

void MNNReluWithSlopeChannel(float* dst, const float* src, const float* slope, size_t sizeQuad, size_t depthQuad);

void MNNPackC4(float* dst, const float* src, size_t area, size_t depth);

void MNNPackC4Uint8(uint8_t* dst, const uint8_t* src, size_t area, size_t depth);

void MNNUnpackC4(float* dst, const float* src, size_t area, size_t depth);

void MNNUnpackC4Uint8(uint8_t* dst, const uint8_t* src, size_t area, size_t depth);

void MNNScaleAndAddBias(float* dst, const float* src, const float* bias, const float* alpha, size_t planeNumber,
                        size_t biasNumber);
void MNNScaleAndAddBiasScalar(float* dst, const float* src, float bias, float alpha, size_t number);

void MNNScaleAndAddBiasOutside(float* dst, const float* src, const float* bias, const float* alpha, size_t planeNumber,
                               size_t biasNumber);

void MNNUnpackTranspose(float* dst, const float* src, size_t area, size_t depth);
void MNNUnpackTransposeUint8(uint8_t* dst, const uint8_t* src, size_t area, size_t depth);

void MNNPackTranspose(float* dst, const float* src, size_t area, size_t depth);
void MNNPackTransposeUint8(uint8_t* dst, const uint8_t* src, size_t area, size_t depth);

void MNNUnpackC4(float* dst, const float* src, size_t area, size_t depth);

void MNNCopyC4WithStride(const float* source, float* dest, size_t srcStride, size_t dstStride, size_t count);
void MNNAddC4WithStride(const float* source, float* dest, size_t srcStride, size_t dstStride, size_t count);

void MNNUInt8ToInt16WithOffsetC4Common(int16_t* dst, const uint8_t* src, size_t zeroPoint, size_t sizeQuad,
                                       size_t dstStride, size_t srcStride);
void MNNUInt8ToInt16WithOffsetC4Fast(int16_t* dst, const uint8_t* src, size_t zeroPoint, size_t sizeQuad,
                                     size_t depthQuad, size_t dstZStep, size_t srcZStep);
void MNNMaxFloat(float* input, float* maxBuffer, int32_t inputCountUnit);
void MNNMinFloat(float* input, float* maxBuffer, int32_t inputCountUnit);
void MNNExpC8(float* dest, const float* source, const float* parameters, size_t countC8);
void MNNPowC8(float* dest, const float* source, const float* powfParam, size_t betaInt, size_t countC8);

void MNNExp(float* dst, const float* src, size_t dataSize);
void MNNTanh(float* dst, const float* src, size_t dataSize);
void MNNReluWithSlopeCommon(float* dst, const float* src, size_t size, float slope);
bool MNNReorder4x4ByPlatform(float* dst, size_t size);

// Get Pack for MatMul's e , l , h , the pack number must be 1 or 4 * n
void MNNGetMatMulPackMode(int* eP, int *lP, int* hP);
void MNNPackC4ForMatMul_A(float* dest, const float* source, size_t e, size_t l, size_t eReal);
void MNNPackForMatMul_B(float* dest, const float* source, size_t h, size_t l, bool transpose);

// parameters: e, l, h, CStride, AStride, BStride
void MNNPackedMatMul(float* C, const float* A, const float* B, const size_t* parameter, float* cache, const float* postParameters, const float* bias);
void MNNFunctionInit();
void MNNPackedMatMulRemain(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter, float* cache, const float* postParameters, const float* bias);
int MNNGetC4DivNumber(int hP);

// C = clamp(alpha * A + beta * B, min, max)
// paramters: alpha, beta, min, max
void MNNAxByClamp(float* C, const float* A, const float* B, size_t width, size_t cStride, size_t aStride, size_t bStride, size_t height, const float* parameters);

void MNNAxByClampBroadcastC4(float* C, const float* A, const float* B, size_t width, size_t cStride, size_t aStride, size_t height, const float* parameters);

// dim: 4-element, sizeDW, sizeDH, strideSW, strideDH
void MNNTranspose32Bit(int32_t* dstO, const int32_t* srcO, int32_t* dim); // not C4

void MNNVectorTop1Float(float* input, float* maxValue, int32_t* maxIndex, size_t inputCountUnit);
void MNNVectorTop1Int32(int32_t* input, int32_t* maxValue, int32_t* maxIndex, size_t inputCountUnit);
struct MatMulParam {
    int32_t e;
    int32_t l;
    int32_t h;
    int32_t numberThread;
    bool ATranspose;
    bool BTranspose;
};
void MNNComputeMatMulForE_1(const float* A, const float* B, float* C, const float* biasPtr, const MatMulParam* param, size_t tId);

#ifdef MNN_USE_SSE
void MNNInt8ToInt16(int16_t* dest, const int8_t* source, size_t count);
#endif
#ifdef __cplusplus
}
#endif

#endif /* CommonOptFunction_h */
