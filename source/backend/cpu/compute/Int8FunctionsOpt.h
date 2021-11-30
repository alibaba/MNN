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
    const int32_t* bias;
    int32_t maxValue;
    int32_t minValue;
    float roundValuePos = 0.5f;
    float roundValueNeg = -0.5f;
};
void MNNFloat2Int8(const float* src, int8_t* dst, size_t sizeQuad, const float* scalep, ssize_t minValue,
                   ssize_t maxValue, ssize_t zeroPoint);
void MNNInt8ScaleToFloat(float* dst, const int8_t* src, const float* scale, size_t size, ssize_t zeroPoint);
void MNNInt8FunctionInit();
void MNNPackedSparseQuantMatMulEpx1(int8_t* C, const int8_t* A, const int8_t* B, const size_t* sparseQuantParam, const QuanPostTreatParameters* post, unsigned int* NNZMap, int* dataOffsetMap);
void MNNPackedSparseQuantMatMulEpx4(int8_t* C, const int8_t* A, const int8_t* B, const size_t* sparseQuantParam, const QuanPostTreatParameters* post, unsigned int* NNZMap, int* dataOffsetMap);


#ifdef __cplusplus
}
#endif

namespace MNN {
struct CoreInt8Functions {
    // MatMul
    void(*Int8GemmKernel)(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad, const QuanPostTreatParameters* post, size_t realCount);
    void(*Int8GemmKernelFast)(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad, const QuanPostTreatParameters* post, size_t realCount);
    void(*MNNGetGemmUnit)(int* UNIT, int* SRC_UNIT, int* DST_XUNIT);
    // Im2Col
    typedef void(*Im2ColFunc)(int8_t* colAddr, const int8_t* inputOrigin, int32_t inputZeroPoint,
                              const ConvolutionCommon::Im2ColParameter* im2colParameter, size_t xIndexStart,
                              size_t realDstCount);
    Im2ColFunc(*chooseIm2Col)(const ConvolutionCommon::Im2ColParameter* im2colParam, size_t inputChannel);

    // sparse
    void(*MNNGetSparseQuantMatMulPackMode)(int* eP, int *lP, int* hP);
    void(*MNNPackForSparseQuantMatMul_B)(int8_t* dest, unsigned int* NNZMap, int* dataOffsetMap, int sparseBlockOC, const int8_t* source, size_t h, size_t kernelCount, size_t icCount, const int eP);
    void(*MNNPackedSparseQuantMatMulEpx1)(int8_t* C, const int8_t* A, const int8_t* B, const size_t* sparseQuantParam, const QuanPostTreatParameters* post, unsigned int* NNZMap, int* dataOffsetMap);
    void(*MNNPackedSparseQuantMatMulEpx4)(int8_t* C, const int8_t* A, const int8_t* B, const size_t* sparseQuantParam, const QuanPostTreatParameters* post, unsigned int* NNZMap, int* dataOffsetMap);
    void(*MNNSparseQuantIm2col)(int8_t* colAddr, const int8_t* inputOrigin, int8_t inputZeroPoint,
                              const ConvolutionCommon::Im2ColParameter* im2colParameter, const size_t* sparseQuantParam, size_t xIndexStart);

    void(*ConvDepthwiseLineInt8)(int8_t* dst, const int8_t* src, const int8_t* weight, const QuanPostTreatParameters* parameters, size_t width,
                                 size_t src_w_step, size_t fw, size_t fh, size_t dilateX_step, size_t dilateY_step);
    void(*MNNFloat2Int8)(const float* src, int8_t* dst, size_t sizeQuad, const float* scalep, ssize_t minValue,
                       ssize_t maxValue, ssize_t zeroPoint);
    void(*MNNInt8ScaleToFloat)(float* dst, const int8_t* src, const float* scale, size_t size, ssize_t zeroPoint);

};
void MNNCoreInt8FunctionInit();
CoreInt8Functions* MNNGetInt8CoreFunctions();
}

#endif /* Int8FunctionsOpt_h */
