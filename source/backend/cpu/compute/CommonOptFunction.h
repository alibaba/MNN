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
#include <vector>

#include "core/Macro.h"

extern "C" {

void MNNReluWithSlope(float* dst, const float* src, size_t sizeQuad, float slope);

void MNNReluInt8(int8_t* dst, const int8_t* src, size_t size);

void MNNReluWithSlopeChannel(float* dst, const float* src, const float* slope, size_t sizeQuad, size_t depthQuad);

void MNNHardSwish(float* dst, const float* src, size_t size);

void MNNGelu(float* dst, const float* src, size_t size);

void MNNPackC4(float* dst, const float* src, size_t area, size_t depth, int* areaOffset);
void MNNPackC4Origin(float* dst, const float* src, size_t area, size_t depth, int areaOffset);

void MNNPackC4Int16(int16_t* dst, const int16_t* src, size_t area,size_t depth, int* areaOffset);

void MNNPackC4Uint8(uint8_t* dst, const uint8_t* src, size_t area,size_t depth, int* areaOffset);

void MNNUnpackC4(float* dst, const float* src, size_t area, size_t depth, int* areaOffset);
void MNNUnpackC4Origin(float* dst, const float* src, size_t area, size_t depth, int areaOffset);

void MNNUnpackC4Int16(int16_t* dst, const int16_t* src, size_t area,size_t depth, int* areaOffset);

void MNNUnpackC4Uint8(uint8_t* dst, const uint8_t* src, size_t area,size_t depth, int* areaOffset);

void MNNScaleAndAddBias(float* dst, const float* src, const float* bias, const float* alpha, size_t planeNumber,
                        size_t biasNumber);
void MNNScaleAndAddBiasScalar(float* dst, const float* src, float bias, float alpha, size_t number);

// TODO: Swap the name for MNNUnpackTranspose and MNNPackTranspose
void MNNUnpackTranspose(float* dst, const float* src, size_t area, size_t depth, int* areaOffset);
void MNNUnpackTransposeInt16(int16_t* dst, const int16_t* src, size_t area,size_t depth, int* areaOffset);
void MNNUnpackTransposeUint8(uint8_t* dst, const uint8_t* src, size_t area,size_t depth, int* areaOffset);

void MNNPackTranspose(float* dst, const float* src, size_t area, size_t depth, int* areaOffset);
void MNNPackTransposeInt16(int16_t* dst, const int16_t* src, size_t area,size_t depth, int* areaOffset);
void MNNPackTransposeUint8(uint8_t* dst, const uint8_t* src, size_t area,size_t depth, int* areaOffset);

void MNNCopyC4WithStride(const float* source, float* dest, size_t srcStride, size_t dstStride, size_t count);
void MNNAddC4WithStride(const float* source, float* dest, size_t srcStride, size_t dstStride, size_t count);

void MNNUInt8ToInt16WithOffsetC4Common(int16_t* dst, const uint8_t* src, size_t zeroPoint, size_t sizeQuad,
                                       size_t dstStride, size_t srcStride);
void MNNUInt8ToInt16WithOffsetC4Fast(int16_t* dst, const uint8_t* src, size_t zeroPoint, size_t sizeQuad,
                                     size_t depthQuad, size_t dstZStep, size_t srcZStep);
void MNNMaxFloat(float* input, float* maxBuffer, int32_t inputCountUnit);
void MNNMinFloat(float* input, float* maxBuffer, int32_t inputCountUnit);
void MNNPowC8(float* dest, const float* source, const float* powfParam, size_t betaInt, size_t countC8);

void MNNExpC8(float* dest, const float* source, const float* parameters, const float* offset, size_t countC8);
void MNNExp(float* dst, const float* src, const float* offset, size_t dataSize);
void MNNSin(float* dst, const float* src, size_t dataSize);
void MNNTanh(float* dst, const float* src, size_t dataSize);
void MNNSigmoid(float* dst, const float* src, size_t dataSize);
void MNNSigmoidLowp(float* dst, const float* src, size_t dataSize);
void MNNReluWithSlopeCommon(float* dst, const float* src, size_t size, float slope);
void MNNHardSwishCommon(float* dst, const float* src, size_t size);
void MNNGeluCommon(float* dst, const float* src, size_t size);
void MNNGeluStandardCommon(float* dst, const float* src, size_t size);
void MNNSoftmax(float* dest, const float* source, size_t size);
void MNNNorm(float* dest, const float* source, const float *gamma, const float *beta, float epsilon, size_t size);

// Get Pack for MatMul's e , l , h , the pack number must be 1 or 4 * n
void MNNGetMatMulPackMode(int* eP, int *lP, int* hP);

void MNNGetSparseMatMulPackMode(int* eP, int *lP, int* hP);

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

void MNNPackForSparseMatMul_B(float* dest, unsigned int* NNZMap, int* dataOffsetMap, int sparseBlockOC, const float* source, size_t h, size_t l, const int eP, bool transpose);
struct SparseMatMulParas
{
    float* C;
    const float* A;
    const float* B;
    unsigned int* NNZMap;
    int* dataOffsetMap;
};
void MNNPackedSparseMatMulEpx1(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter, const float* postParameters, const float* bias, unsigned int* NNZMap, int* dataOffsetMap);

void MNNPackedSparseMatMulEpx4(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter, const float* postParameters, const float* bias, unsigned int* NNZMap, int* dataOffsetMap);


int MNNGetC4DivNumber(int hP);

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
void MNNConvDwF23MulTransUnit(float **cacheLine, const float *weigth, float *dest, size_t ow, const float* bias, const float* postParameter);
void MNNMultiAndDestTransformCommon23(float **cacheLine, const float *weigth, float *dest, int cacheLineSize, int ow);
void MNNInt8ToInt16(int16_t* dest, const int8_t* source, size_t count);
}


void MNNGridSampleComputeCord(float* dst, const float* src, size_t inH, size_t inW, size_t outH, size_t outW, size_t stride, bool alignCorners);
void MNNGridSampleInterp(float* outputPtr, const float* inputPtr, const float* cordPtr, size_t inH, size_t inW, size_t outW,
                            size_t channelCUnit, size_t inOffset, size_t outOffset, bool sampleMode, bool padMode);
void MNNGridSampleComputeCord3D(float* dst, const float* src, size_t inD, size_t inH, size_t inW, size_t outD, size_t outH, size_t outW, size_t stride, bool alignCorners);
void MNNGridSampleInterp3D(float* outputPtr, const float* inputPtr, const float* cordPtr, size_t inD, size_t inH, size_t inW, size_t outW, size_t channelCUnit, size_t inOffset, size_t outOffset, bool sampleMode, bool padMode);
void MNNRoiPoolingMax(float* dst, const float* src, int hLen, int wLen, int iw);
void MNNRoiAlignMax(float* dst, const float* src, const std::vector<std::vector<int>> &vecPos, const std::vector<std::vector<float>> &vecArea, int samplingRatioArea, int pooledHeight, int pooledWidth);
void MNNRoiAlignAvg(float* dst, const float* src, const std::vector<std::vector<int>> &vecPos, const std::vector<std::vector<float>> &vecArea, int samplingRatioArea, int pooledHeight, int pooledWidth);


typedef void(*MNNBinaryExecute)(void* outputRaw, const void* inputRaw0, const void* inputRaw1, int elementSize, int broadcastIndex);
typedef void(*MNNUnaryExecute)(void* outputRaw, const void* inputRaw, int elementSize);
typedef void(*MNNCopyWithStride)(uint8_t* dstO, const uint8_t* srcO, int size, int stride, int ds);


constexpr int InputTileMax = 14; // same value from DynamicGemm.h, cannot include from different backend code.

namespace MNN {
struct CoreFunctions {
    // cpu feature
    bool supportFp16arith = false;
    bool supportSDot = false;
    bool supportI8mm = false;
    /**MatMul Pack and Functions*/
    void(*MNNGetMatMulPackMode)(int* eP, int *lP, int* hP);
    void(*MNNGetSparseMatMulPackMode)(int* eP, int *lP, int* hP);
    void(*MNNPackC4ForMatMul_A)(float* destOrigin, float const** sourceGroup, const int32_t* info, const int32_t* el);
    void(*MNNPackForMatMul_B)(float* dest, const float* source, size_t h, size_t l, bool transpose);
    // parameters: e, l, h, CStride, AStride, BStride
    void(*MNNPackedMatMul)(float* C, const float* A, const float* B, const size_t* parameter, const float* postParameters, const float* bias);
    void(*MNNPackedMatMulRemain)(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter, const float* postParameters, const float* bias);
    void(*MNNComputeMatMulForH_1)(const float* A, const float* B, float* C, const float* biasPtr, const MatMulParam* param, size_t tId);
    void(*MNNComputeMatMulForE_1)(const float* A, const float* B, float* C, const float* biasPtr, const MatMulParam* param, size_t tId);


    typedef void(*MNNPackedMatMulKernel)(float* C, const float* A, const float* B, const size_t* parameter, const float* postParameters, const float* bias);

    MNNPackedMatMulKernel MNNPackedMatMulOC16Functions[InputTileMax] = {0};
    MNNPackedMatMulKernel MNNPackedMatMulOC32Functions[InputTileMax] = {0};
    MNNPackedMatMulKernel MNNPackedMatMulOC48Functions[InputTileMax] = {0};

    // For Atomic Op
    MNNBinaryExecute(*MNNSelectBinaryFunctionForFloat)(int opType);
    MNNUnaryExecute(*MNNSelectUnaryFunctionForFloat)(int opType, int precisionMode);

    // sparse matrix multiply
    void(*MNNPackForSparseMatMul_B)(float* dest, unsigned int* NNZMap, int* dataOffsetMap, int sparseBlockOC, const float* source, size_t h, size_t l, const int eP, bool transpose);
    void(*MNNGetOptimalBlockShape)(size_t& weightNNZElement, size_t& weightBlockNumber, const float* source, int sparseBlockOC, size_t h, size_t l);

    // B matrix is sparsed
    typedef void(*MNNPackedSparseMatMul)(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter, const float* postParameters, const float* bias, unsigned int* NNZMap, int* dataOffsetMap);
    void(*MNNAdjustOptimalSparseKernel)(int& sparseBlockOC, MNNPackedSparseMatMul& packedSparseMatMul);
    /**Lowp Backend Setting*/
    void(*MNNFp32ToLowp)(const float* src, int16_t* dst, size_t size);
    void(*MNNLowpToFp32)(const int16_t* src, float* dst, size_t size);
    int bytes; // Byte for float

    /**NC4HW4's Functions*/
    int pack;
    // For pack * bytes > 16
    MNNCopyWithStride(*MNNSelectBlitFunction)(int blitBytes) = nullptr;

    void(*MNNPackCUnitInt16)(int16_t* dst, const int16_t* src, size_t area, size_t depth, int* areaOffset);
    void(*MNNUnpackCUnitInt16)(int16_t* dst, const int16_t* src, size_t area, size_t depth, int* areaOffset);
    void(*MNNPackCUnitTransposeInt16)(int16_t* dst, const int16_t* src, size_t area, size_t depth, int* areaOffset);
    void(*MNNUnpackCUnitTransposeInt16)(int16_t* dst, const int16_t* src, size_t area, size_t depth, int* areaOffset);

    void(*MNNPackCUnitInt8)(int8_t* dst, const int8_t* src, size_t area, size_t depth, int* areaOffset);
    void(*MNNUnpackCUnitInt8)(int8_t* dst, const int8_t* src, size_t area, size_t depth, int* areaOffset);
    void(*MNNPackCUnitTransposeInt8)(int8_t* dst, const int8_t* src, size_t area, size_t depth, int* areaOffset);
    void(*MNNUnpackCUnitTransposeInt8)(int8_t* dst, const int8_t* src, size_t area, size_t depth, int* areaOffset);

    void(*MNNPackCUnit)(float* dst, const float* src, size_t area, size_t depth, int* areaOffset);
    void(*MNNUnpackCUnit)(float* dst, const float* src, size_t area, size_t depth, int* areaOffset);
    void(*MNNPackCUnitTranspose)(float* dst, const float* src, size_t area, size_t depth, int* areaOffset);
    void(*MNNUnpackCUnitTranspose)(float* dst, const float* src, size_t area, size_t depth, int* areaOffset);

    // NC4HW4's compute function
    void(*MNNConvRunForUnitDepthWise)(float* dst, const float* src, const float* weight, size_t fw, size_t fh,
                                        size_t weight_y_step, size_t dilateX_step, size_t dilateY_step);
    void(*MNNConvRunForLineDepthwise)(float* dst, const float* src, const float* weight, size_t width, size_t src_w_setup,
                                    size_t fw, size_t fh, size_t dilateX_step, size_t dilateY_step, size_t height,
                                    size_t srcHStep, size_t dstHStep);
    void(*MNNAxByClampBroadcastUnit)(float* C, const float* A, const float* B, size_t width, size_t cStride, size_t aStride, size_t height, const float* parameters);
    void(*MNNMultiAndDestTransformCommon23)(float **cacheLine, const float *weigth, float *dest, int cacheLineSize, int ow, const float* bias, const float* post);
    void(*MNNSourceTransformCommonF23)(const float *source, float *dest, int unit, int iw, int pad, int su, int eu);
    void(*MNNConvDwF23MulTransUnit)(float **cacheLine, const float *weigth, float *dest, size_t ow, const float* bias, const float* post);
    void(*MNNMatrixAdd)(float* C, const float* A, const float* B, size_t widthC4, size_t cStride, size_t aStride,
                      size_t bStride, size_t height);
    void(*MNNMatrixSub)(float* C, const float* A, const float* B, size_t widthC4, size_t cStride, size_t aStride,
                      size_t bStride, size_t height);
    void(*MNNStrassenMergeCFunction)(float* c11, float* c12, float* c21, float* c22, float* xAddr, size_t cStride, size_t eSub, size_t hSub);
    void(*MNNScaleAndAddBias)(float* dst, const float* src, const float* bias, const float* alpha, size_t planeNumber, size_t biasNumber);
    void(*MNNGridSampleComputeCord)(float* dst, const float* src, size_t inH, size_t inW, size_t outH, size_t outW, size_t stride, bool alignCorners);
    void(*MNNGridSampleInterp)(float* outputPtr, const float* inputPtr, const float* cordPtr, size_t inH, size_t inW, size_t outW, size_t channelCUnit, size_t inOffset, size_t outOffset, bool sampleMode, bool padMode);
    void(*MNNGridSampleComputeCord3D)(float* dst, const float* src, size_t inD, size_t inH, size_t inW, size_t outD, size_t outH, size_t outW, size_t stride1, size_t stride2, bool alignCorners);
    void(*MNNGridSampleInterp3D)(float* outputPtr, const float* inputPtr, const float* cordPtr, size_t inD, size_t inH, size_t inW, size_t outW, size_t channelCUnit, size_t inOffset, size_t outOffset, bool sampleMode, bool padMode);
    void(*MNNRoiPoolingMax)(float* dst, const float* src, int hLen, int wLen, int iw);
    void(*MNNRoiAlignMax)(float* dst, const float* src, const std::vector<std::vector<int>> &vecPos, const std::vector<std::vector<float>> &vecArea, int samplingRatioArea, int pooledHeight, int pooledWidth);
    void(*MNNRoiAlignAvg)(float* dst, const float* src, const std::vector<std::vector<int>> &vecPos, const std::vector<std::vector<float>> &vecArea, int samplingRatioArea, int pooledHeight, int pooledWidth);

    float penalty;

    void(*MNNCopyC4WithStride)(const float* source, float* dest, size_t srcStride, size_t dstStride, size_t count);
    void(*MNNAddC4WithStride)(const float* source, float* dest, size_t srcStride, size_t dstStride, size_t count);

    typedef void (*WinoTransPackFunc)(float* srcBlock, float* dstStart, size_t dstStep);
    WinoTransPackFunc(*chooseWinoSourceTransformPack)(int k, int w, int ePack, int lPack, int packCUnit);

    typedef void (*WinoUnrollTransFunc)(const float* srcBlock, float* dstStart, size_t srcRowStep, size_t dstRowStep, size_t srcStep, size_t dstStep);
    typedef void (*WinoUnrollDestTransFunc)(const float* srcBlock, float* dstStart,  const float* bias, const float* postParameters, size_t srcRowStep, size_t dstRowStep, size_t srcStep, size_t dstStep);
    WinoUnrollTransFunc(*chooseWinoSourceUnrollTransform)(int k, int w);
    void(*chooseWinoDestUnrollTransform)(WinoUnrollDestTransFunc *destFunctions, size_t maxUnit, int k, int h);

    void(*MNNDeconvRunForUnitDepthWise)(const float* dst, float* src, const float* weight, size_t fw, size_t fh,
                                      size_t weight_y_step, size_t dilateX_step, size_t dilateY_step);
    void(*MNNDeconvRunForLineDepthwise)(const float* dst, float* src, const float* weight, size_t width, size_t src_w_setup,
                                      size_t fw, size_t fh, size_t dilateX_step, size_t dilateY_step);
    void(*MNNReluWithSlopeChannel)(float* dst, const float* src, const float* slope, size_t sizeQuad, size_t depthQuad);
    void(*MNNPoolingAvg)(const void* channelInput, int inputWidth, int inputHeight, void *channelOutput,
                           int outputWidth, int outputHeight, int kernelWidth, int kernelHeight, int strideWidth,
                           int strideHeight, int padWidth, int padHeight, int padType, int countType);
    void(*MNNPoolingMax)(const void* channelInput, int inputWidth, int inputHeight, void *channelOutput,
                           int outputWidth, int outputHeight, int kernelWidth, int kernelHeight, int strideWidth,
                           int strideHeight, int padWidth, int padHeight, int padType, int countType);
    // ImageProcess Funtions
    void(*MNNRGBAToBGRA)(const unsigned char* source, unsigned char* dest, size_t count);
    void(*MNNNV21ToRGBA)(const unsigned char* source, unsigned char* dest, size_t count);
    void(*MNNNV21ToRGB)(const unsigned char* source, unsigned char* dest, size_t count);
    void(*MNNNV21ToBGRA)(const unsigned char* source, unsigned char* dest, size_t count);
    void(*MNNNV21ToBGR)(const unsigned char* source, unsigned char* dest, size_t count);
    void(*MNNC1ToFloatC1)(const unsigned char* source, float* dest, const float* mean, const float* normal, size_t count);
    void(*MNNC3ToFloatC3)(const unsigned char* source, float* dest, const float* mean, const float* normal, size_t count);
    void(*MNNC3ToFloatRGBA)(const unsigned char* source, float* dest, const float* mean, const float* normal, size_t count);
};
void MNNCoreFunctionInit();
CoreFunctions* MNNGetCoreFunctions();
};

#endif /* CommonOptFunction_h */
