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

extern "C" {

void MNNReluWithSlope(float* dst, const float* src, size_t sizeQuad, float slope);

void MNNRelu6(float* dst, const float* src, size_t size);

void MNNReluInt8(int8_t* dst, const int8_t* src, size_t size);

void MNNReluWithSlopeChannel(float* dst, const float* src, const float* slope, size_t sizeQuad, size_t depthQuad);

void MNNHardSwish(float* dst, const float* src, size_t size);

void MNNPackC4(float* dst, const float* src, size_t area, size_t depth);

void MNNPackC4Int16(int16_t* dst, const int16_t* src, size_t area, size_t depth);

void MNNPackC4Uint8(uint8_t* dst, const uint8_t* src, size_t area, size_t depth);

void MNNUnpackC4(float* dst, const float* src, size_t area, size_t depth);

void MNNUnpackC4Int16(int16_t* dst, const int16_t* src, size_t area, size_t depth);

void MNNUnpackC4Uint8(uint8_t* dst, const uint8_t* src, size_t area, size_t depth);

void MNNScaleAndAddBias(float* dst, const float* src, const float* bias, const float* alpha, size_t planeNumber,
                        size_t biasNumber);
void MNNScaleAndAddBiasScalar(float* dst, const float* src, float bias, float alpha, size_t number);

void MNNUnpackTranspose(float* dst, const float* src, size_t area, size_t depth);
void MNNUnpackTransposeInt16(int16_t* dst, const int16_t* src, size_t area, size_t depth);
void MNNUnpackTransposeUint8(uint8_t* dst, const uint8_t* src, size_t area, size_t depth);

void MNNPackTranspose(float* dst, const float* src, size_t area, size_t depth);
void MNNPackTransposeInt16(int16_t* dst, const int16_t* src, size_t area, size_t depth);
void MNNPackTransposeUint8(uint8_t* dst, const uint8_t* src, size_t area, size_t depth);

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
void MNNSin(float* dst, const float* src, size_t dataSize);
void MNNTanh(float* dst, const float* src, size_t dataSize);
void MNNSigmoid(float* dst, const float* src, size_t dataSize);
void MNNSigmoidLowp(float* dst, const float* src, size_t dataSize);
void MNNReluWithSlopeCommon(float* dst, const float* src, size_t size, float slope);
void MNNHardSwishCommon(float* dst, const float* src, size_t size);

// Get Pack for MatMul's e , l , h , the pack number must be 1 or 4 * n
void MNNGetMatMulPackMode(int* eP, int *lP, int* hP);


/**
 int number = info[0];
 int eSrcStride = info[1];
 int eDstStride = info[2];
 int xStride = info[3];

el: number * 4
 0: e
 1: l
 2: e-offset
 3: l-offset
 */
void MNNPackC4ForMatMul_A(float* destOrigin, float const** sourceGroup, const int32_t* info, const int32_t* el);

void MNNPackForMatMul_B(float* dest, const float* source, size_t h, size_t l, bool transpose);

// parameters: e, l, h, CStride, AStride, BStride
void MNNPackedMatMul(float* C, const float* A, const float* B, const size_t* parameter, const float* postParameters, const float* bias);
void MNNFunctionInit();
void MNNPackedMatMulRemain(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter, const float* postParameters, const float* bias);
int MNNGetC4DivNumber(int hP);

// C = clamp(alpha * A + beta * B, min, max)
// paramters: alpha, beta, min, max
void MNNAxByClamp(float* C, const float* A, const float* B, size_t width, size_t cStride, size_t aStride, size_t bStride, size_t height, const float* parameters);

void MNNAxByClampBroadcastUnit(float* C, const float* A, const float* B, size_t width, size_t cStride, size_t aStride, size_t height, const float* parameters);

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

void MNNCopyC4Int16WithStride(const float* sourceF, float* destF, size_t srcStride, size_t dstStride, size_t count);
void MNNSourceTransformCommonF23(const float *source, float *dest, int unit, int iw, int pad, int su, int eu);
void MNNConvDwF23MulTransUnit(float **cacheLine, const float *weigth, float *dest, size_t ow);
void MNNMultiAndDestTransformCommon23(float **cacheLine, const float *weigth, float *dest, int cacheLineSize, int ow);
#ifdef MNN_USE_SSE
void MNNInt8ToInt16(int16_t* dest, const int8_t* source, size_t count);
#endif
}

// c++ template function should not in extern C
template<typename DataType>
void MNNPackForMatMul_B_Template(DataType* dest, const DataType* source, size_t h, size_t l, bool transpose);

namespace MNN {
struct CoreFunctions {
    /**MatMul Pack and Functions*/
    void(*MNNGetMatMulPackMode)(int* eP, int *lP, int* hP);
    void(*MNNPackC4ForMatMul_A)(float* destOrigin, float const** sourceGroup, const int32_t* info, const int32_t* el);
    void(*MNNPackForMatMul_B)(float* dest, const float* source, size_t h, size_t l, bool transpose);
    // parameters: e, l, h, CStride, AStride, BStride
    void(*MNNPackedMatMul)(float* C, const float* A, const float* B, const size_t* parameter, const float* postParameters, const float* bias);
    void(*MNNPackedMatMulRemain)(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter, const float* postParameters, const float* bias);

    /**Lowp Backend Setting*/
    void(*MNNFp32ToLowp)(const float* src, int16_t* dst, size_t size);
    void(*MNNLowpToFp32)(const int16_t* src, float* dst, size_t size);
    int bytes;
    
    /**NC4HW4's Functions*/
    int pack;
    void(*MNNPackCUnit)(float* dst, const float* src, size_t area, size_t depth);
    void(*MNNUnpackCUnit)(float* dst, const float* src, size_t area, size_t depth);
    void(*MNNPackCUnitTranspose)(float* dst, const float* src, size_t area, size_t depth);
    void(*MNNUnpackCUnitTranspose)(float* dst, const float* src, size_t area, size_t depth);

    void(*MNNConvRunForUnitDepthWise)(float* dst, const float* src, const float* weight, size_t fw, size_t fh,
                                        size_t weight_y_step, size_t dilateX_step, size_t dilateY_step);
    void(*MNNConvRunForLineDepthwise)(float* dst, const float* src, const float* weight, size_t width, size_t src_w_setup,
                                    size_t fw, size_t fh, size_t dilateX_step, size_t dilateY_step, size_t height,
                                    size_t srcHStep, size_t dstHStep);
    void(*MNNAxByClampBroadcastUnit)(float* C, const float* A, const float* B, size_t width, size_t cStride, size_t aStride, size_t height, const float* parameters);
    void(*MNNMultiAndDestTransformCommon23)(float **cacheLine, const float *weigth, float *dest, int cacheLineSize, int ow);
    void(*MNNSourceTransformCommonF23)(const float *source, float *dest, int unit, int iw, int pad, int su, int eu);
    void(*MNNConvDwF23MulTransUnit)(float **cacheLine, const float *weigth, float *dest, size_t ow);
    void(*MNNMatrixAdd)(float* C, const float* A, const float* B, size_t widthC4, size_t cStride, size_t aStride,
                      size_t bStride, size_t height);
    void(*MNNMatrixSub)(float* C, const float* A, const float* B, size_t widthC4, size_t cStride, size_t aStride,
                      size_t bStride, size_t height);
    void(*MNNStrassenMergeCFunction)(float* c11, float* c12, float* c21, float* c22, float* xAddr, size_t cStride, size_t eSub, size_t hSub);
    void(*MNNScaleAndAddBias)(float* dst, const float* src, const float* bias, const float* alpha, size_t planeNumber, size_t biasNumber);
    float penalty;

    void(*MNNCopyC4WithStride)(const float* source, float* dest, size_t srcStride, size_t dstStride, size_t count);
    void(*MNNAddC4WithStride)(const float* source, float* dest, size_t srcStride, size_t dstStride, size_t count);
    
    typedef void (*WinoTransFunc)(const float* srcBlock, float* dstStart, size_t srcStep, size_t dstStep);
    WinoTransFunc(*chooseWinoSourceTransform)(int k, int w);
    WinoTransFunc(*chooseWinoDestTransform)(int k, int h);

    void(*MNNDeconvRunForUnitDepthWise)(const float* dst, float* src, const float* weight, size_t fw, size_t fh,
                                      size_t weight_y_step, size_t dilateX_step, size_t dilateY_step);
    void(*MNNDeconvRunForLineDepthwise)(const float* dst, float* src, const float* weight, size_t width, size_t src_w_setup,
                                      size_t fw, size_t fh, size_t dilateX_step, size_t dilateY_step);
};
void MNNCoreFunctionInit();
CoreFunctions* MNNGetCoreFunctions();
};

#endif /* CommonOptFunction_h */
