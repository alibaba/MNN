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

void MNNTensorConvertNHWCToNC4HW4(float* dst, const float* src, size_t area, size_t depth);
void MNNTensorConvertNHWCToNC4HW4Uint8(uint8_t* dst, const uint8_t* src, size_t area, size_t depth);

void MNNTensorConvertNC4HW4ToNHWC(float* dst, const float* src, size_t area, size_t depth);
void MNNTensorConvertNC4HW4ToNHWCUint8(uint8_t* dst, const uint8_t* src, size_t area, size_t depth);

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

#ifdef __cplusplus
}
#endif

#endif /* CommonOptFunction_h */
