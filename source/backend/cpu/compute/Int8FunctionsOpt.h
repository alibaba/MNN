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
#include "core/ConvolutionCommon.hpp"
#if defined(_MSC_VER)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

/* CPU without sdot */
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

#ifdef __cplusplus
extern "C" {
#endif

struct QuanPostTreatParameters {
    const float* scale;
    const float* biasFloat;
    int32_t maxValue;
    int32_t minValue;
    int32_t useInt8 = 1; // Save result as int8_t dataType; otherwise float32.
    float roundValuePos = 0.5f;
    float roundValueNeg = -0.5f;
    float* srcKernelSum;
    float* weightQuanBias;
    float* fp32minmax;
    ssize_t blockNum = 1;
    const int32_t* bias;
    const float* extraScale = nullptr;
    const float* extraBias = nullptr;
};
struct QuanPrePostParameters{
    float* inputScale;
    float* outputScale;
    ssize_t* inputZeroPoint;
    ssize_t* outputZeroPoint;
    ssize_t minValue;
    ssize_t maxValue;
};
void MNNFloat2Int8(const float* src, int8_t* dst, size_t sizeQuad, const float* scalep, ssize_t minValue,
                   ssize_t maxValue, ssize_t zeroPoint);
void MNNInt8ScaleToFloat(float* dst, const int8_t* src, const float* scale, size_t size, ssize_t zeroPoint);
void MNNInt8FunctionInit();
void MNNPackedSparseQuantMatMulEpx1(int8_t* C, const int8_t* A, const int8_t* B, const size_t* sparseQuantParam, const QuanPostTreatParameters* post, unsigned int* NNZMap, int* dataOffsetMap);
void MNNPackedSparseQuantMatMulEpx4(int8_t* C, const int8_t* A, const int8_t* B, const size_t* sparseQuantParam, const QuanPostTreatParameters* post, unsigned int* NNZMap, int* dataOffsetMap);
void MNNBinaryAddInt8(int8_t* outputRaw, const int8_t* inputRaw0, const int8_t* inputRaw1, ssize_t* inputScalesInt32, float* inputScalesFp32, const QuanPrePostParameters* params, size_t elementSize, size_t needBroadcast);
void MNNBinarySubInt8(int8_t* outputRaw, const int8_t* inputRaw0, const int8_t* inputRaw1, ssize_t* inputScalesInt32, float* inputScalesFp32, const QuanPrePostParameters* params, size_t elementSize, size_t needBroadcast);
void MNNBinaryMulInt8(int8_t* outputRaw, const int8_t* inputRaw0, const int8_t* inputRaw1, ssize_t* inputScalesInt32, float* inputScalesFp32, const QuanPrePostParameters* params, size_t elementSize, size_t needBroadcast);
void MNNBinarySqdInt8(int8_t* outputRaw, const int8_t* inputRaw0, const int8_t* inputRaw1, ssize_t* inputScalesInt32, float* inputScalesFp32, const QuanPrePostParameters* params, size_t elementSize, size_t needBroadcast);
void MNNBinaryMaxInt8(int8_t* outputRaw, const int8_t* inputRaw0, const int8_t* inputRaw1, ssize_t* inputScalesInt32, float* inputScalesFp32, const QuanPrePostParameters* params, size_t elementSize, size_t needBroadcast);
void MNNBinaryMinInt8(int8_t* outputRaw, const int8_t* inputRaw0, const int8_t* inputRaw1, ssize_t* inputScalesInt32, float* inputScalesFp32, const QuanPrePostParameters* params, size_t elementSize, size_t needBroadcast);
void MNNScaleAndAddBiasInt8(int8_t* dst, const int8_t* src, const int32_t* bias, const int32_t* alpha, int32_t mShiftBits, ssize_t minValue, ssize_t maxValue, int8_t* inputZeroPoint, int8_t* outputZeroPoint, ssize_t planeNumber, ssize_t biasNumber, ssize_t pack = 4);
#ifdef __cplusplus
}
#endif

namespace MNN {
struct CoreInt8Functions {
    // MatMul
    void(*Int8GemmKernel)(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad, const QuanPostTreatParameters* post, size_t realCount);
    void(*Int8GemmKernelFast)(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad, const QuanPostTreatParameters* post, size_t realCount);
    void(*MNNGetGemmUnit)(int* UNIT, int* SRC_UNIT, int* DST_XUNIT);
    void(*MNNPackC4Int8ForMatMul_A)(int8_t* destOrigin, int8_t const** sourceGroup, const int32_t* info, const int32_t* el);
    void(*MNNPackC4Int8ForMatMul_A_ARM86FP16)(int8_t* destOrigin, int8_t const** sourceGroup, const int32_t* info, const int32_t* el) = nullptr;
    void(*MNNGemmInt8AddBiasScale_Unit_FP16)(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad,
                                        const QuanPostTreatParameters* post, size_t realDstCount);
    void(*MNNGemmInt8AddBiasScale_w4_Unit_FP16)(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad,
                                        const QuanPostTreatParameters* post, size_t realDstCount);
    void(*Int8GemmKernel_W4)(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad,
                                           const QuanPostTreatParameters* post, size_t realDstCount);
    // sparse
    void(*MNNGetSparseQuantMatMulPackMode)(int* eP, int *lP, int* hP);
    void(*MNNPackForSparseQuantMatMul_B)(int8_t* dest, unsigned int* NNZMap, int* dataOffsetMap, int sparseBlockOC, const int8_t* source, size_t h, size_t kernelCount, size_t icCount, const int eP);
    void(*MNNPackedSparseQuantMatMulEpx1)(int8_t* C, const int8_t* A, const int8_t* B, const size_t* sparseQuantParam, const QuanPostTreatParameters* post, unsigned int* NNZMap, int* dataOffsetMap);
    void(*MNNPackedSparseQuantMatMulEpx4)(int8_t* C, const int8_t* A, const int8_t* B, const size_t* sparseQuantParam, const QuanPostTreatParameters* post, unsigned int* NNZMap, int* dataOffsetMap);
    void(*MNNPackC4Int8ForMatMul_ASparse)(int8_t* destOrigin, int8_t const** sourceGroup, const int32_t* info, const int32_t* el);

    void(*ConvDepthwiseLineInt8)(int8_t* dst, const int8_t* src, const int8_t* weight, const QuanPostTreatParameters* parameters, size_t width,
                                 size_t src_w_step, size_t fw, size_t fh, size_t dilateX_step, size_t dilateY_step, int8_t* idxOrder);
    void(*ConvDepthwise3x3LineInt8_ARM82)(int8_t* dst, const int8_t* src, const int8_t* weight, const QuanPostTreatParameters* parameters, size_t width,
                                 size_t src_w_step, size_t fw, size_t fh, size_t dilateX_step, size_t dilateY_step, int8_t* idxOrder) = nullptr;
    void(*DynamicQuanInput_ARM82)(const float* src, int8_t* dst, size_t sizeQuad, const float* scalep, ssize_t minValue, ssize_t maxValue, ssize_t zeroPoint) = nullptr;
    void (*DynamicQuanInputAndReorder_ARM82)(const float* src, int8_t* dst, size_t planeSize, const float* scale, ssize_t aMin, ssize_t aMax, ssize_t zeroPoint, size_t ocQuad, size_t offset) = nullptr;
    void(*MNNFloat2Int8)(const float* src, int8_t* dst, size_t sizeQuad, const float* scalep, ssize_t minValue, ssize_t maxValue, ssize_t zeroPoint);
    void(*MNNInt8ScaleToFloat)(float* dst, const int8_t* src, const float* scale, size_t size, ssize_t zeroPoint);

    void(*MNNScaleAndAddBias)(float* dst, const float* src, const float* bias, const float* alpha, size_t planeNumber, size_t biasNumber);

    // Pooling
    void (*MNNMaxPoolInt8)(int8_t* dst, int8_t* src, size_t outputWidth, size_t inputWidth, size_t kernelx, size_t kernely, size_t stridesx);
    void (*MNNAvgPoolInt8)(int8_t* dst, int8_t* src, size_t outputWidth, size_t inputWidth, size_t kernelx, size_t kernely, size_t stridesx, ssize_t paddingx, ssize_t factor);
    
    // Relu
    void (*MNNReluWithSlopeChannelInt8)(int8_t* dst, const int8_t* src, const float* slope, size_t planeNumber, size_t depthQuad, QuanPrePostParameters *params);
};
void MNNCoreInt8FunctionInit();
CoreInt8Functions* MNNGetInt8CoreFunctions();
}

#endif /* Int8FunctionsOpt_h */
