//
//  Int8FunctionsOpt.cpp
//  MNN
//
//  Created by MNN on 2018/08/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <math.h>
#include <cstring> // for memset
#include "Int8FunctionsOpt.h"
#include "core/Macro.h"
#include "core/CommonCompute.hpp"
#include "CommonOptFunction.h"
#include "math/Vec.hpp"

#ifdef MNN_USE_NEON
#include <arm_neon.h>

extern "C" {
void MNNGemmInt8AddBiasScale_16x4_Unit(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad,
                                       const QuanPostTreatParameters* post, size_t realCount);
void MNNGemmInt8AddBiasScale_16x4_Unit_FAST(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad,
                                            const QuanPostTreatParameters* post, size_t realCount);
void MNNGemmInt8AddBiasScale_16x4_w4_Unit(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad,
                                    const QuanPostTreatParameters* post, size_t realCount);
void MNNLineDepthWiseInt8AddBiasScaleUnit(int8_t* dst, const int8_t* src, const int8_t* weight, const QuanPostTreatParameters* parameters, size_t width,
                                          size_t src_w_step, size_t fw, size_t fh, size_t dilateX_step, size_t dilateY_step, int8_t* idxOrder=nullptr);
void MNNMaxPoolInt8(int8_t* dst, int8_t* src, size_t outputWidth, size_t inputWidth, size_t kernelx, size_t kernely, size_t stridesx);

void MNNAvgPoolInt8(int8_t* dst, int8_t* src, size_t outputWidth, size_t inputWidth, size_t kernelx, size_t kernely, size_t stridesx, ssize_t paddingx, ssize_t factor);
void MNNReluWithSlopeChannelInt8(int8_t* dst, const int8_t* src, const float* slope, size_t planeNumber, size_t depthQuad, const QuanPrePostParameters *params, size_t pack);
#if defined(__aarch64__) // aarch32 sdot workaround
void MNNGemmInt8AddBiasScale_ARMV82_Unit(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad,
                                        const QuanPostTreatParameters* post, size_t realDstCount);
void MNNGemmInt8AddBiasScale_ARMV86_Unit(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad,
                                        const QuanPostTreatParameters* post, size_t realDstCount);
void MNNLineDepthWiseInt8AddBiasScale_ARMV82_Unit3X3(int8_t* dst, const int8_t* src, const int8_t* weight, const QuanPostTreatParameters* parameters, size_t width,
                                        size_t src_w_step, size_t fw, size_t fh, size_t dilateX_step, size_t dilateY_step, int8_t* idxOrder=nullptr);
void MNNSumByAxisLForMatmul_A_ARM86(float* dest, int8_t* source, const float* dequantScale, ssize_t realDstCount, SumByAxisParams sumParams);
void MNNSumByAxisLForMatmul_A_ARM82(float* dest, int8_t* source, const float* dequantScale, ssize_t realDstCount, SumByAxisParams sumParams);
#if defined(MNN_LOW_MEMORY)
// int4 weight gemmInt8 kernel
void MNNGemmInt8AddBiasScale_ARMV82_w4_Unit(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad,
                                        const QuanPostTreatParameters* post, size_t realDstCount);
void MNNGemmInt8AddBiasScale_ARMV86_w4_Unit(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad,
                                        const QuanPostTreatParameters* post, size_t realDstCount);
void MNNGemmInt8AddBiasScale_16x4_w4_Unit(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad,
                                        const QuanPostTreatParameters* post, size_t realDstCount);
// Tools to dynamic-quant fp16-input data.
#ifdef MNN_USE_ARMV82
void DynamicQuanInput_ARM82(const float* src, int8_t* dst, size_t sizeQuad, const float* scalep, ssize_t minValue,
                        ssize_t maxValue, const float* zeroPoint, ssize_t quanParamVec);
// int8 weight gemmInt8 kernel to return fp16-output data.
void MNNGemmInt8AddBiasScale_ARMV82_Unit_FP16(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad,
                                              const QuanPostTreatParameters* post, size_t realDstCount);
void MNNGemmInt8AddBiasScale_ARMV82_w4_Unit_FP16(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad,
                                              const QuanPostTreatParameters* post, size_t realDstCount);
void MNNGemmInt8AddBiasScale_ARMV86_Unit_FP16(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad,
                                              const QuanPostTreatParameters* post, size_t realDstCount);
void MNNGemmInt8AddBiasScale_ARMV86_w4_Unit_FP16(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad,
                                              const QuanPostTreatParameters* post, size_t realDstCount);
void DynamicQuanInputAndReorder_ARM82(const float* src, int8_t* dst, size_t planeSize, const float* scale, ssize_t aMin,
                                     ssize_t aMax, const float* zeroPoint, size_t ocQuad, size_t offset);
#endif
#endif
#endif // __aarch64__
}
#endif // MNN_USE_NEON

/*
    layout should be optimized for int8
    source: source matrix is h x l
    transpose: if false, export compressed matrix as h x l, other export as l x h.
 */
void MNNPackForSparseQuantMatMul_B(int8_t* dest, unsigned int* NNZMap, int* dataOffsetMap, int sparseBlockOC, const int8_t* source, size_t h, size_t kernelCount, size_t icCount, const int eP) {
    // 1. in quant convolution, source B layout is OC x (IC * KH * KW),
    //    the dest layout of weight is BCSC(block compressed sparse colum) format, which is OC(!=0) x (KH*KW*IC!=0), as a canceled result, just do BCSR
    // 2. IC would be switched into the last dim.

     // BCSC
    int columOffset = 0;
    int i = 0;
    auto subSource = source;
    size_t l = kernelCount * icCount;
    for (; i + sparseBlockOC <= h; i += sparseBlockOC) {
        *NNZMap = 0;
        for(int ik = 0; ik < kernelCount; ik += 1) {
            auto  kernelSource = subSource + ik;
            for(int ic = 0; ic < icCount; ic += 1) {
                if (!MNN::CommonCompute::checkAllZeros(kernelSource, l, sparseBlockOC, 1)) {
                    for (int ioc = 0; ioc < sparseBlockOC; ioc++) {
                        *dest = *(kernelSource + ioc * l);
                        dest++;
                    }
                    *NNZMap = *NNZMap + 1;
                    *dataOffsetMap = columOffset;
                    dataOffsetMap++;
                    columOffset = 0;
                }
                columOffset += eP;
                kernelSource += kernelCount;
            }
        }
        NNZMap++;
        columOffset -= l * eP;
        subSource += sparseBlockOC * l;
    }

    for (; i < h; i++) {
        *NNZMap = 0;
        for(int ik = 0; ik < kernelCount; ik += 1) {
            auto  kernelSource = subSource + ik;
            for(int ic = 0; ic < icCount; ic += 1) {
                if (*kernelSource != 0) {
                    *dest = *kernelSource;
                    dest++;
                    *NNZMap = *NNZMap + 1;
                    *dataOffsetMap = columOffset;
                    dataOffsetMap++;
                    columOffset = 0;
                }
                columOffset += eP;
                kernelSource += kernelCount;
            }
        }
        NNZMap++;
        columOffset -= l * eP;
        subSource += l;
    }

    *dataOffsetMap = columOffset; //

    return;
}


void MNNGetSparseQuantMatMulPackMode(int* eP, int *lP, int* hP) {
#if defined(__arm__) && !defined(__aarch64__)
    *eP = 8;
#else
    *eP = 16;
#endif
    *lP = 1;
    *hP = 4;
    // hp is corresponding to sparse block along right matrix colum dimension. in ramdom sparse, it is 1.
    return;
}

static void _MNNPackC4Int8ForMatMul_ASparse(int8_t* destOrigin, int8_t const** sourceGroup, const int32_t* info, const int32_t* el) {
    int number = info[0];
    int eReal = info[1];
    int eDest = info[2];
    int offset = info[3];
    for (int n=0; n<number; ++n) {
        int e = el[4 * n + 0];
        int l = el[4 * n + 1];
        int eOffset = el[4 * n + 2];
        int lOffset = el[4 * n + 3];
        auto dest = destOrigin + lOffset * eDest + eOffset;
        auto source = sourceGroup[n];

        for (int y=0; y<e; ++y) {
            auto yR = y % eDest;
            for (int x=0; x<l; ++x) {
                auto xR = x % 4;
                auto xC = x / 4;
                dest[(x) * eDest + yR] = source[xC * eReal * 4 + y * 4 * offset + xR];
            }
        }
    }
}

#ifndef MNN_USE_NEON

void MNNPackedSparseQuantMatMulEpx1(int8_t* C, const int8_t* A, const int8_t* B, const size_t* sparseQuantParam, const QuanPostTreatParameters* post, unsigned int* NNZMap, int* dataOffsetMap) {

    size_t eSize = sparseQuantParam[0];
    size_t eP = sparseQuantParam[1];
    size_t aStride = sparseQuantParam[2];
    size_t l = sparseQuantParam[3];
    size_t h = sparseQuantParam[4];
    size_t cStride = sparseQuantParam[5];

    const int32_t* bias = post->bias;
    const float* scales = post->scale;
    const int32_t maxValue = post->maxValue;
    const int32_t minValue = post->minValue;

    const int sparseBlockOC = 4;
    const int8_t * a = A;
    size_t ie = 0;
    for (ie = 0; ie < eSize && eP <= eSize; ie += eP) {
        const int* dataOffset = dataOffsetMap;
        const int diff = *dataOffset++;
        a += diff;
        const int8_t * w = B;
        int8_t * blockC = C + (ie << 2);
        const unsigned int* nnz = NNZMap;

        for (size_t ih = 0; ih < h; ih++) {
            auto ihPack = ih >> 2;
            auto ihSubIndex = ih & 0x03;
            auto c = blockC + ihPack * cStride + ihSubIndex;
            const int32_t initValue = nullptr != bias ? bias[ih] : 0;
            int32_t acc0 = initValue;
            int32_t acc1 = initValue;
            int32_t acc2 = initValue;
            int32_t acc3 = initValue;
            int32_t acc4 = initValue;
            int32_t acc5 = initValue;
            int32_t acc6 = initValue;
            int32_t acc7 = initValue;
            int32_t acc8 = initValue;
            int32_t acc9 = initValue;
            int32_t acc10 = initValue;
            int32_t acc11 = initValue;
            int32_t acc12 = initValue;
            int32_t acc13 = initValue;
            int32_t acc14 = initValue;
            int32_t acc15 = initValue;
            const int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {

                const int diff = *dataOffset++;
                const int8_t a0 = a[0];
                const int8_t a1 = a[1];
                const int8_t a2 = a[2];
                const int8_t a3 = a[3];
                const int8_t a4 = a[4];
                const int8_t a5 = a[5];
                const int8_t a6 = a[6];
                const int8_t a7 = a[7];
                const int8_t a8 = a[8];
                const int8_t a9 = a[9];
                const int8_t a10 = a[10];
                const int8_t a11 = a[11];
                const int8_t a12 = a[12];
                const int8_t a13 = a[13];
                const int8_t a14 = a[14];
                const int8_t a15 = a[15];

                const int8_t oneW = *w++;

                // MNN_PRINT("16-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0-15]:", ie, a - A, w - B - 1, c - C, oneW);
                // formatMatrix(a, {16});
                // MNN_PRINT("\n");
                a = a + diff;
                acc0 += (int32_t)a0 * (int32_t)oneW;
                acc1 += (int32_t)a1 * (int32_t)oneW;
                acc2 += (int32_t)a2 * (int32_t)oneW;
                acc3 += (int32_t)a3 * (int32_t)oneW;
                acc4 += (int32_t)a4 * (int32_t)oneW;
                acc5 += (int32_t)a5 * (int32_t)oneW;
                acc6 += (int32_t)a6 * (int32_t)oneW;
                acc7 += (int32_t)a7 * (int32_t)oneW;
                acc8 += (int32_t)a8 * (int32_t)oneW;
                acc9 += (int32_t)a9 * (int32_t)oneW;
                acc10 += (int32_t)a10 * (int32_t)oneW;
                acc11 += (int32_t)a11 * (int32_t)oneW;
                acc12 += (int32_t)a12 * (int32_t)oneW;
                acc13 += (int32_t)a13 * (int32_t)oneW;
                acc14 += (int32_t)a14 * (int32_t)oneW;
                acc15 += (int32_t)a15 * (int32_t)oneW;
            }

            int8_t result0; // in assemmbly code, consider reuse acc0[0-8] bit
            int8_t result1;
            int8_t result2;
            int8_t result3;
            int8_t result4;
            int8_t result5;
            int8_t result6;
            int8_t result7;
            int8_t result8;
            int8_t result9;
            int8_t result10;
            int8_t result11;
            int8_t result12;
            int8_t result13;
            int8_t result14;
            int8_t result15;

            if (scales) {
                result0  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih] * float(acc0)), float(minValue))));
                result1  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih] * float(acc1)), float(minValue))));
                result2  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih] * float(acc2)), float(minValue))));
                result3  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih] * float(acc3)), float(minValue))));
                result4  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih] * float(acc4)), float(minValue))));
                result5  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih] * float(acc5)), float(minValue))));
                result6  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih] * float(acc6)), float(minValue))));
                result7  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih] * float(acc7)), float(minValue))));
                result8  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih] * float(acc8)), float(minValue))));
                result9  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih] * float(acc9)), float(minValue))));
                result10 = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih] * float(acc10)), float(minValue))));
                result11 = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih] * float(acc11)), float(minValue))));
                result12 = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih] * float(acc12)), float(minValue))));
                result13 = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih] * float(acc13)), float(minValue))));
                result14 = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih] * float(acc14)), float(minValue))));
                result15 = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih] * float(acc15)), float(minValue))));
            } else {
                result0  = static_cast<int8_t>(std::max(std::min(maxValue, acc0), minValue));
                result1  = static_cast<int8_t>(std::max(std::min(maxValue, acc1), minValue));
                result2  = static_cast<int8_t>(std::max(std::min(maxValue, acc2), minValue));
                result3  = static_cast<int8_t>(std::max(std::min(maxValue, acc3), minValue));
                result4  = static_cast<int8_t>(std::max(std::min(maxValue, acc4), minValue));
                result5  = static_cast<int8_t>(std::max(std::min(maxValue, acc5), minValue));
                result6  = static_cast<int8_t>(std::max(std::min(maxValue, acc6), minValue));
                result7  = static_cast<int8_t>(std::max(std::min(maxValue, acc7), minValue));
                result8  = static_cast<int8_t>(std::max(std::min(maxValue, acc8), minValue));
                result9  = static_cast<int8_t>(std::max(std::min(maxValue, acc9), minValue));
                result10 = static_cast<int8_t>(std::max(std::min(maxValue, acc10), minValue));
                result11 = static_cast<int8_t>(std::max(std::min(maxValue, acc11), minValue));
                result12 = static_cast<int8_t>(std::max(std::min(maxValue, acc12), minValue));
                result13 = static_cast<int8_t>(std::max(std::min(maxValue, acc13), minValue));
                result14 = static_cast<int8_t>(std::max(std::min(maxValue, acc14), minValue));
                result15 = static_cast<int8_t>(std::max(std::min(maxValue, acc15), minValue));
            }

            // how to store faster: st4 / transpose /
            c[0] = result0;
            c[4] = result1;
            c[4 * 2] = result2;
            c[4 * 3] = result3;
            c[4 * 4] = result4;
            c[4 * 5] = result5;
            c[4 * 6] = result6;
            c[4 * 7] = result7;
            c[4 * 8] = result8;
            c[4 * 9] = result9;
            c[4 * 10] = result10;
            c[4 * 11] = result11;
            c[4 * 12] = result12;
            c[4 * 13] = result13;
            c[4 * 14] = result14;
            c[4 * 15] = result15;
        }
        a += aStride;
    }
    if (eSize & 0x08) {
        const int* dataOffset = dataOffsetMap;
        const int diff = *dataOffset++;
        // a = blockA + diff;
        a += diff;
        const int8_t* w = B;
        int8_t* blockC = C + (ie << 2);
        const unsigned int* nnz = NNZMap;
        for (size_t ih = 0; ih < h; ih++) {
            auto ihPack = ih >> 2;
            auto ihSubIndex = ih & 0x03;
            auto c = blockC + ihPack * cStride + ihSubIndex;
            const int32_t initValue = nullptr != bias ? bias[ih] : 0;
            int32_t acc0 = initValue;
            int32_t acc1 = initValue;
            int32_t acc2 = initValue;
            int32_t acc3 = initValue;
            int32_t acc4 = initValue;
            int32_t acc5 = initValue;
            int32_t acc6 = initValue;
            int32_t acc7 = initValue;

            const int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {
                const int diff = *dataOffset++;
                const int8_t a0 = a[0];
                const int8_t a1 = a[1];
                const int8_t a2 = a[2];
                const int8_t a3 = a[3];
                const int8_t a4 = a[4];
                const int8_t a5 = a[5];
                const int8_t a6 = a[6];
                const int8_t a7 = a[7];
                const int8_t oneW = *w++;
                // MNN_PRINT("8-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%d, a value[0-7]:\n", ie, a - A, w - B - 1, c - C, oneW);
                // formatMatrix(a, {8});
                // MNN_PRINT("\n");
                a = a + diff;
                acc0 += int32_t(a0) * int32_t(oneW);
                acc1 += int32_t(a1) * int32_t(oneW);
                acc2 += int32_t(a2) * int32_t(oneW);
                acc3 += int32_t(a3) * int32_t(oneW);
                acc4 += int32_t(a4) * int32_t(oneW);
                acc5 += int32_t(a5) * int32_t(oneW);
                acc6 += int32_t(a6) * int32_t(oneW);
                acc7 += int32_t(a7) * int32_t(oneW);
            }

            int8_t result0;
            int8_t result1;
            int8_t result2;
            int8_t result3;
            int8_t result4;
            int8_t result5;
            int8_t result6;
            int8_t result7;
            if (scales) {
                result0  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih] * float(acc0)), float(minValue))));
                result1  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih] * float(acc1)), float(minValue))));
                result2  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih] * float(acc2)), float(minValue))));
                result3  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih] * float(acc3)), float(minValue))));
                result4  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih] * float(acc4)), float(minValue))));
                result5  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih] * float(acc5)), float(minValue))));
                result6  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih] * float(acc6)), float(minValue))));
                result7  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih] * float(acc7)), float(minValue))));

            } else {
                result0  = static_cast<int8_t>(std::max(std::min(maxValue, acc0), minValue));
                result1  = static_cast<int8_t>(std::max(std::min(maxValue, acc1), minValue));
                result2  = static_cast<int8_t>(std::max(std::min(maxValue, acc2), minValue));
                result3  = static_cast<int8_t>(std::max(std::min(maxValue, acc3), minValue));
                result4  = static_cast<int8_t>(std::max(std::min(maxValue, acc4), minValue));
                result5  = static_cast<int8_t>(std::max(std::min(maxValue, acc5), minValue));
                result6  = static_cast<int8_t>(std::max(std::min(maxValue, acc6), minValue));
                result7  = static_cast<int8_t>(std::max(std::min(maxValue, acc7), minValue));
            }

            // how to store faster: st4 / transpose /
            c[0] = result0;
            c[4] = result1;
            c[4 * 2] = result2;
            c[4 * 3] = result3;
            c[4 * 4] = result4;
            c[4 * 5] = result5;
            c[4 * 6] = result6;
            c[4 * 7] = result7;
        }
        ie += 8;
        a += 8;
    }
    if (eSize & 0x04) {
        const int* dataOffset = dataOffsetMap;
        const int diff = *dataOffset++;
        // a = blockA + diff;
        a += diff;
        const int8_t* w = B;
        int8_t* blockC = C + (ie << 2);
        const unsigned int* nnz = NNZMap;

        for (size_t ih = 0; ih < h; ih++) {
            auto ihPack = ih >> 2;
            auto ihSubIndex = ih & 0x03;
            auto c = blockC + ihPack * cStride + ihSubIndex;
            const int32_t initValue = nullptr != bias ? bias[ih] : 0;
            int32_t acc0 = initValue;
            int32_t acc1 = initValue;
            int32_t acc2 = initValue;
            int32_t acc3 = initValue;

            const int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {
                const int diff = *dataOffset++;
                const int8_t a0 = a[0];
                const int8_t a1 = a[1];
                const int8_t a2 = a[2];
                const int8_t a3 = a[3];
                const int8_t oneW = *w++;
                // MNN_PRINT("4-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%d, a value[0-3]:\n", ie, a - A, w - B - 1, c - C, oneW);
                // formatMatrix(a, {4});
                // MNN_PRINT("\n");
                a = a + diff;
                acc0 += int32_t(a0) * int32_t(oneW);
                acc1 += int32_t(a1) * int32_t(oneW);
                acc2 += int32_t(a2) * int32_t(oneW);
                acc3 += int32_t(a3) * int32_t(oneW);
            }

            int8_t result0;
            int8_t result1;
            int8_t result2;
            int8_t result3;
            if (scales) {
                result0  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih] * float(acc0)), float(minValue))));
                result1  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih] * float(acc1)), float(minValue))));
                result2  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih] * float(acc2)), float(minValue))));
                result3  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih] * float(acc3)), float(minValue))));
            } else {
                result0  = static_cast<int8_t>(std::max(std::min(maxValue, acc0), minValue));
                result1  = static_cast<int8_t>(std::max(std::min(maxValue, acc1), minValue));
                result2  = static_cast<int8_t>(std::max(std::min(maxValue, acc2), minValue));
                result3  = static_cast<int8_t>(std::max(std::min(maxValue, acc3), minValue));
            }

            // how to store faster: st4 / transpose /
            c[0] = result0;
            c[4] = result1;
            c[4 * 2] = result2;
            c[4 * 3] = result3;
        }
        ie += 4;
        a += 4;
    }
    if (eSize & 0x02) {
        const int* dataOffset = dataOffsetMap;
        const int diff = *dataOffset++;
        // a = blockA + diff;
        a += diff;
        const int8_t* w = B;
        int8_t* blockC = C + (ie << 2);
        const unsigned int* nnz = NNZMap;
        for (size_t ih = 0; ih < h; ih++) {
            auto ihPack = ih >> 2;
            auto ihSubIndex = ih & 0x03;
            auto c = blockC + ihPack * cStride + ihSubIndex;
            const int32_t initValue = nullptr != bias ? bias[ih] : 0;
            int32_t acc0 = initValue;
            int32_t acc1 = initValue;

            const int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {
                const int diff = *dataOffset++;
                const int8_t a0 = a[0];
                const int8_t a1 = a[1];
                const int8_t oneW = *w++;
                // MNN_PRINT("2-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%d, a value[0-1]:\n", ie, a - A, w - B - 1, c - C, oneW);
                // formatMatrix(a, {2});
                // MNN_PRINT("\n");
                a = a + diff;
                acc0 += int32_t(a0) * int32_t(oneW);
                acc1 += int32_t(a1) * int32_t(oneW);
            }

            int8_t result0;
            int8_t result1;
            if (scales) {
                result0  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih] * float(acc0)), float(minValue))));
                result1  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih] * float(acc1)), float(minValue))));
            } else {
                result0  = static_cast<int8_t>(std::max(std::min(maxValue, acc0), minValue));
                result1  = static_cast<int8_t>(std::max(std::min(maxValue, acc1), minValue));
            }

            // how to store faster: st4 / transpose /
            c[0] = result0;
            c[4] = result1;
        }
        ie += 2;
        a += 2;
    }
    if (eSize & 0x01) {
        const int* dataOffset = dataOffsetMap;
        const int diff = *dataOffset++;
        // const float* a = blockA + diff;
        a += diff;
        const int8_t * w = B;
        int8_t * blockC = C + (ie << 2);
        const unsigned int* nnz = NNZMap;
        for (size_t ih = 0; ih < h; ih++) {
            auto ihPack = ih >> 2;
            auto ihSubIndex = ih & 0x03;
            auto c = blockC + ihPack * cStride + ihSubIndex;
            const int32_t initValue = nullptr != bias ? bias[ih] : 0;
            int32_t acc0 = initValue;

            const int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {
                const int diff = *dataOffset++;
                const int8_t a0 = a[0];
                const int8_t oneW = *w++;

                // MNN_PRINT("1-loop: ie:%zu, a offset:%ld, c offset:%ld, w offset:%ld, w value:%d, a value[0]:\n", ie, a - A, w - B - 1, c - C, oneW);
                // formatMatrix(a, {1});
                // MNN_PRINT("\n");
                a = a + diff;
                acc0 += int32_t(a0) * int32_t(oneW);
            }
            int8_t result0;
            if (scales) {
                result0  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih] * float(acc0)), float(minValue))));
            } else {
                result0  = static_cast<int8_t>(std::max(std::min(maxValue, acc0), minValue));
            }
            // how to store faster: st4 / transpose /
            c[0] = result0;
        }
        ie += 1;
        // a += 1;
    }

}

void MNNPackedSparseQuantMatMulEpx4(int8_t* C, const int8_t* A, const int8_t* B, const size_t* sparseQuantParam, const QuanPostTreatParameters* post, unsigned int* NNZMap, int* dataOffsetMap) {

    size_t eSize = sparseQuantParam[0];
    size_t eP = sparseQuantParam[1];
    size_t aStride = sparseQuantParam[2];
    size_t l = sparseQuantParam[3];
    size_t h = sparseQuantParam[4];
    size_t cStride = sparseQuantParam[5];

    const int32_t* bias = post->bias;
    const float* scales = post->scale;
    const int32_t maxValue = post->maxValue;
    const int32_t minValue = post->minValue;

    const int sparseBlockOC = 4;
    const int8_t * a = A;
    size_t ie = 0;
    for (ie = 0; ie < eSize && eP <= eSize; ie += eP) {
        const int* dataOffset = dataOffsetMap;
        const int diff = *dataOffset++;
        a += diff;
        const int8_t * w = B;
        int8_t * blockC = C + (ie << 2);
        const unsigned int* nnz = NNZMap;

        size_t ih = 0;
        for (; ih < (h & (~0x03)); ih += sparseBlockOC) {
            auto ihPack = ih >> 2;
            auto c = blockC + ihPack * cStride;

            int32_t initValue[4] = {0, 0, 0, 0};
            if (nullptr != bias) {
                memcpy(initValue, bias + ih, 4 * sizeof(int32_t));
            }
            int32_t acc0[4];
            int32_t acc1[4];
            int32_t acc2[4];
            int32_t acc3[4];
            int32_t acc4[4];
            int32_t acc5[4];
            int32_t acc6[4];
            int32_t acc7[4];
            int32_t acc8[4];
            int32_t acc9[4];
            int32_t acc10[4];
            int32_t acc11[4];
            int32_t acc12[4];
            int32_t acc13[4];
            int32_t acc14[4];
            int32_t acc15[4];

            memcpy(acc0, initValue, 4 * sizeof(int32_t));
            memcpy(acc1, initValue, 4 * sizeof(int32_t));
            memcpy(acc2, initValue, 4 * sizeof(int32_t));
            memcpy(acc3, initValue, 4 * sizeof(int32_t));
            memcpy(acc4, initValue, 4 * sizeof(int32_t));
            memcpy(acc5, initValue, 4 * sizeof(int32_t));
            memcpy(acc6, initValue, 4 * sizeof(int32_t));
            memcpy(acc7, initValue, 4 * sizeof(int32_t));
            memcpy(acc8, initValue, 4 * sizeof(int32_t));
            memcpy(acc9, initValue, 4 * sizeof(int32_t));
            memcpy(acc10, initValue, 4 * sizeof(int32_t));
            memcpy(acc11, initValue, 4 * sizeof(int32_t));
            memcpy(acc12, initValue, 4 * sizeof(int32_t));
            memcpy(acc13, initValue, 4 * sizeof(int32_t));
            memcpy(acc14, initValue, 4 * sizeof(int32_t));
            memcpy(acc15, initValue, 4 * sizeof(int32_t));

            const int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {

                const int diff = *dataOffset++;
                const int8_t a0 = a[0];
                const int8_t a1 = a[1];
                const int8_t a2 = a[2];
                const int8_t a3 = a[3];
                const int8_t a4 = a[4];
                const int8_t a5 = a[5];
                const int8_t a6 = a[6];
                const int8_t a7 = a[7];
                const int8_t a8 = a[8];
                const int8_t a9 = a[9];
                const int8_t a10 = a[10];
                const int8_t a11 = a[11];
                const int8_t a12 = a[12];
                const int8_t a13 = a[13];
                const int8_t a14 = a[14];
                const int8_t a15 = a[15];

                const int8_t wv[4] = {*w++, *w++, *w++, *w++};

                // MNN_PRINT("16-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0-15]:", ie, a - A, w - B - 1, c - C, oneW);
                // formatMatrix(a, {16});
                // MNN_PRINT("\n");
                a = a + diff;
                for (int lane = 0; lane < 4; lane++) {
                    acc0[lane] += (int32_t)a0 * (int32_t)wv[lane];
                    acc1[lane] += (int32_t)a1 * (int32_t)wv[lane];
                    acc2[lane] += (int32_t)a2 * (int32_t)wv[lane];
                    acc3[lane] += (int32_t)a3 * (int32_t)wv[lane];
                    acc4[lane] += (int32_t)a4 * (int32_t)wv[lane];
                    acc5[lane] += (int32_t)a5 * (int32_t)wv[lane];
                    acc6[lane] += (int32_t)a6 * (int32_t)wv[lane];
                    acc7[lane] += (int32_t)a7 * (int32_t)wv[lane];
                    acc8[lane] += (int32_t)a8 * (int32_t)wv[lane];
                    acc9[lane] += (int32_t)a9 * (int32_t)wv[lane];
                    acc10[lane] += (int32_t)a10 * (int32_t)wv[lane];
                    acc11[lane] += (int32_t)a11 * (int32_t)wv[lane];
                    acc12[lane] += (int32_t)a12 * (int32_t)wv[lane];
                    acc13[lane] += (int32_t)a13 * (int32_t)wv[lane];
                    acc14[lane] += (int32_t)a14 * (int32_t)wv[lane];
                    acc15[lane] += (int32_t)a15 * (int32_t)wv[lane];
                }
            }

            int8_t result0[4];
            int8_t result1[4];
            int8_t result2[4];
            int8_t result3[4];
            int8_t result4[4];
            int8_t result5[4];
            int8_t result6[4];
            int8_t result7[4];
            int8_t result8[4];
            int8_t result9[4];
            int8_t result10[4];
            int8_t result11[4];
            int8_t result12[4];
            int8_t result13[4];
            int8_t result14[4];
            int8_t result15[4];

            if (scales) {
                for (int lane = 0; lane < 4; lane++) {
                    result0[lane]  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih + lane] * float(acc0[lane])), float(minValue))));
                    result1[lane]  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih + lane] * float(acc1[lane])), float(minValue))));
                    result2[lane]  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih + lane] * float(acc2[lane])), float(minValue))));
                    result3[lane]  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih + lane] * float(acc3[lane])), float(minValue))));
                    result4[lane]  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih + lane] * float(acc4[lane])), float(minValue))));
                    result5[lane]  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih + lane] * float(acc5[lane])), float(minValue))));
                    result6[lane]  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih + lane] * float(acc6[lane])), float(minValue))));
                    result7[lane]  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih + lane] * float(acc7[lane])), float(minValue))));
                    result8[lane]  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih + lane] * float(acc8[lane])), float(minValue))));
                    result9[lane]  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih + lane] * float(acc9[lane])), float(minValue))));
                    result10[lane] = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih + lane] * float(acc10[lane])), float(minValue))));
                    result11[lane] = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih + lane] * float(acc11[lane])), float(minValue))));
                    result12[lane] = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih + lane] * float(acc12[lane])), float(minValue))));
                    result13[lane] = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih + lane] * float(acc13[lane])), float(minValue))));
                    result14[lane] = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih + lane] * float(acc14[lane])), float(minValue))));
                    result15[lane] = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih + lane] * float(acc15[lane])), float(minValue))));
                }
            } else {
                for (int lane = 0; lane < 4; lane++) {
                    result0[lane]  = static_cast<int8_t>(roundf(std::max(std::min(maxValue, acc0[lane]), minValue)));
                    result1[lane]  = static_cast<int8_t>(roundf(std::max(std::min(maxValue, acc1[lane]), minValue)));
                    result2[lane]  = static_cast<int8_t>(roundf(std::max(std::min(maxValue, acc2[lane]), minValue)));
                    result3[lane]  = static_cast<int8_t>(roundf(std::max(std::min(maxValue, acc3[lane]), minValue)));
                    result4[lane]  = static_cast<int8_t>(roundf(std::max(std::min(maxValue, acc4[lane]), minValue)));
                    result5[lane]  = static_cast<int8_t>(roundf(std::max(std::min(maxValue, acc5[lane]), minValue)));
                    result6[lane]  = static_cast<int8_t>(roundf(std::max(std::min(maxValue, acc6[lane]), minValue)));
                    result7[lane]  = static_cast<int8_t>(roundf(std::max(std::min(maxValue, acc7[lane]), minValue)));
                    result8[lane]  = static_cast<int8_t>(roundf(std::max(std::min(maxValue, acc8[lane]), minValue)));
                    result9[lane]  = static_cast<int8_t>(roundf(std::max(std::min(maxValue, acc9[lane]), minValue)));
                    result10[lane] = static_cast<int8_t>(roundf(std::max(std::min(maxValue, acc10[lane]), minValue)));
                    result11[lane] = static_cast<int8_t>(roundf(std::max(std::min(maxValue, acc11[lane]), minValue)));
                    result12[lane] = static_cast<int8_t>(roundf(std::max(std::min(maxValue, acc12[lane]), minValue)));
                    result13[lane] = static_cast<int8_t>(roundf(std::max(std::min(maxValue, acc13[lane]), minValue)));
                    result14[lane] = static_cast<int8_t>(roundf(std::max(std::min(maxValue, acc14[lane]), minValue)));
                    result15[lane] = static_cast<int8_t>(roundf(std::max(std::min(maxValue, acc15[lane]), minValue)));
                }
            }

            memcpy(c         , result0, 4 * sizeof(int8_t));  // store continuous c
            memcpy(c + 4     , result1, 4 * sizeof(int8_t));
            memcpy(c + 4 * 2 , result2, 4 * sizeof(int8_t));
            memcpy(c + 4 * 3 , result3, 4 * sizeof(int8_t));
            memcpy(c + 4 * 4 , result4, 4 * sizeof(int8_t));
            memcpy(c + 4 * 5 , result5, 4 * sizeof(int8_t));
            memcpy(c + 4 * 6 , result6, 4 * sizeof(int8_t));
            memcpy(c + 4 * 7 , result7, 4 * sizeof(int8_t));
            memcpy(c + 4 * 8 , result8, 4 * sizeof(int8_t));
            memcpy(c + 4 * 9 , result9, 4 * sizeof(int8_t));
            memcpy(c + 4 * 10, result10, 4 * sizeof(int8_t));
            memcpy(c + 4 * 11, result11, 4 * sizeof(int8_t));
            memcpy(c + 4 * 12, result12, 4 * sizeof(int8_t));
            memcpy(c + 4 * 13, result13, 4 * sizeof(int8_t));
            memcpy(c + 4 * 14, result14, 4 * sizeof(int8_t));
            memcpy(c + 4 * 15, result15, 4 * sizeof(int8_t));
        }

        blockC += (h >> 2) * cStride;
        for (; ih < h; ih++) {
            auto ihSubIndex = ih & 0x03;
            auto c = blockC + ihSubIndex;
            const int32_t initValue = nullptr != bias ? bias[ih] : 0;
            int32_t acc0 = initValue;
            int32_t acc1 = initValue;
            int32_t acc2 = initValue;
            int32_t acc3 = initValue;
            int32_t acc4 = initValue;
            int32_t acc5 = initValue;
            int32_t acc6 = initValue;
            int32_t acc7 = initValue;
            int32_t acc8 = initValue;
            int32_t acc9 = initValue;
            int32_t acc10 = initValue;
            int32_t acc11 = initValue;
            int32_t acc12 = initValue;
            int32_t acc13 = initValue;
            int32_t acc14 = initValue;
            int32_t acc15 = initValue;
            const int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {

                const int diff = *dataOffset++;
                const int8_t a0 = a[0];
                const int8_t a1 = a[1];
                const int8_t a2 = a[2];
                const int8_t a3 = a[3];
                const int8_t a4 = a[4];
                const int8_t a5 = a[5];
                const int8_t a6 = a[6];
                const int8_t a7 = a[7];
                const int8_t a8 = a[8];
                const int8_t a9 = a[9];
                const int8_t a10 = a[10];
                const int8_t a11 = a[11];
                const int8_t a12 = a[12];
                const int8_t a13 = a[13];
                const int8_t a14 = a[14];
                const int8_t a15 = a[15];

                const int8_t oneW = *w++;

                // MNN_PRINT("16-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0-15]:", ie, a - A, w - B - 1, c - C, oneW);
                // formatMatrix(a, {16});
                // MNN_PRINT("\n");
                a = a + diff;
                acc0 += (int32_t)a0 * (int32_t)oneW;
                acc1 += (int32_t)a1 * (int32_t)oneW;
                acc2 += (int32_t)a2 * (int32_t)oneW;
                acc3 += (int32_t)a3 * (int32_t)oneW;
                acc4 += (int32_t)a4 * (int32_t)oneW;
                acc5 += (int32_t)a5 * (int32_t)oneW;
                acc6 += (int32_t)a6 * (int32_t)oneW;
                acc7 += (int32_t)a7 * (int32_t)oneW;
                acc8 += (int32_t)a8 * (int32_t)oneW;
                acc9 += (int32_t)a9 * (int32_t)oneW;
                acc10 += (int32_t)a10 * (int32_t)oneW;
                acc11 += (int32_t)a11 * (int32_t)oneW;
                acc12 += (int32_t)a12 * (int32_t)oneW;
                acc13 += (int32_t)a13 * (int32_t)oneW;
                acc14 += (int32_t)a14 * (int32_t)oneW;
                acc15 += (int32_t)a15 * (int32_t)oneW;
            }

            int8_t result0; // in assemmbly code, consider reuse acc0[0-8] bit
            int8_t result1;
            int8_t result2;
            int8_t result3;
            int8_t result4;
            int8_t result5;
            int8_t result6;
            int8_t result7;
            int8_t result8;
            int8_t result9;
            int8_t result10;
            int8_t result11;
            int8_t result12;
            int8_t result13;
            int8_t result14;
            int8_t result15;

            if (scales) {
                result0  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih] * float(acc0)), float(minValue))));
                result1  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih] * float(acc1)), float(minValue))));
                result2  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih] * float(acc2)), float(minValue))));
                result3  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih] * float(acc3)), float(minValue))));
                result4  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih] * float(acc4)), float(minValue))));
                result5  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih] * float(acc5)), float(minValue))));
                result6  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih] * float(acc6)), float(minValue))));
                result7  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih] * float(acc7)), float(minValue))));
                result8  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih] * float(acc8)), float(minValue))));
                result9  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih] * float(acc9)), float(minValue))));
                result10 = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih] * float(acc10)), float(minValue))));
                result11 = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih] * float(acc11)), float(minValue))));
                result12 = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih] * float(acc12)), float(minValue))));
                result13 = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih] * float(acc13)), float(minValue))));
                result14 = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih] * float(acc14)), float(minValue))));
                result15 = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih] * float(acc15)), float(minValue))));
            } else {
                result0  = static_cast<int8_t>(std::max(std::min(maxValue, acc0), minValue));
                result1  = static_cast<int8_t>(std::max(std::min(maxValue, acc1), minValue));
                result2  = static_cast<int8_t>(std::max(std::min(maxValue, acc2), minValue));
                result3  = static_cast<int8_t>(std::max(std::min(maxValue, acc3), minValue));
                result4  = static_cast<int8_t>(std::max(std::min(maxValue, acc4), minValue));
                result5  = static_cast<int8_t>(std::max(std::min(maxValue, acc5), minValue));
                result6  = static_cast<int8_t>(std::max(std::min(maxValue, acc6), minValue));
                result7  = static_cast<int8_t>(std::max(std::min(maxValue, acc7), minValue));
                result8  = static_cast<int8_t>(std::max(std::min(maxValue, acc8), minValue));
                result9  = static_cast<int8_t>(std::max(std::min(maxValue, acc9), minValue));
                result10 = static_cast<int8_t>(std::max(std::min(maxValue, acc10), minValue));
                result11 = static_cast<int8_t>(std::max(std::min(maxValue, acc11), minValue));
                result12 = static_cast<int8_t>(std::max(std::min(maxValue, acc12), minValue));
                result13 = static_cast<int8_t>(std::max(std::min(maxValue, acc13), minValue));
                result14 = static_cast<int8_t>(std::max(std::min(maxValue, acc14), minValue));
                result15 = static_cast<int8_t>(std::max(std::min(maxValue, acc15), minValue));
            }

            // how to store faster: st4 / transpose /
            c[0] = result0;
            c[4] = result1;
            c[4 * 2] = result2;
            c[4 * 3] = result3;
            c[4 * 4] = result4;
            c[4 * 5] = result5;
            c[4 * 6] = result6;
            c[4 * 7] = result7;
            c[4 * 8] = result8;
            c[4 * 9] = result9;
            c[4 * 10] = result10;
            c[4 * 11] = result11;
            c[4 * 12] = result12;
            c[4 * 13] = result13;
            c[4 * 14] = result14;
            c[4 * 15] = result15;
        }
        a += aStride;
    }
    if (eSize & 0x08) {
        const int* dataOffset = dataOffsetMap;
        const int diff = *dataOffset++;
        // a = blockA + diff;
        a += diff;
        const int8_t* w = B;
        int8_t* blockC = C + (ie << 2);
        const unsigned int* nnz = NNZMap;

        size_t ih = 0;
        for (; ih < (h & (~0x03)); ih += sparseBlockOC) {
            auto ihPack = ih >> 2;
            auto c = blockC + ihPack * cStride;
            int32_t initValue[4] = {0, 0, 0, 0};
            if (nullptr != bias) {
                memcpy(initValue, bias + ih, 4 * sizeof(int32_t));
            }
            int32_t acc0[4];
            int32_t acc1[4];
            int32_t acc2[4];
            int32_t acc3[4];
            int32_t acc4[4];
            int32_t acc5[4];
            int32_t acc6[4];
            int32_t acc7[4];

            memcpy(acc0, initValue, 4 * sizeof(int32_t));
            memcpy(acc1, initValue, 4 * sizeof(int32_t));
            memcpy(acc2, initValue, 4 * sizeof(int32_t));
            memcpy(acc3, initValue, 4 * sizeof(int32_t));
            memcpy(acc4, initValue, 4 * sizeof(int32_t));
            memcpy(acc5, initValue, 4 * sizeof(int32_t));
            memcpy(acc6, initValue, 4 * sizeof(int32_t));
            memcpy(acc7, initValue, 4 * sizeof(int32_t));

            const int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {

                const int diff = *dataOffset++;
                const int8_t a0 = a[0];
                const int8_t a1 = a[1];
                const int8_t a2 = a[2];
                const int8_t a3 = a[3];
                const int8_t a4 = a[4];
                const int8_t a5 = a[5];
                const int8_t a6 = a[6];
                const int8_t a7 = a[7];
                const int8_t wv[4] = {*w++, *w++, *w++, *w++};
                // MNN_PRINT("8-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value[0-3]:, a value[0-7]:\n", ie, a - A, w - B - 1, c - C);
                // formatMatrix(wv, {4});
                // formatMatrix(a, {8});
                // MNN_PRINT("\n");
                a = a + diff;
                for (int lane = 0; lane < 4; lane++) {
                    acc0[lane] += int32_t(a0) * int32_t(wv[lane]);
                    acc1[lane] += int32_t(a1) * int32_t(wv[lane]);
                    acc2[lane] += int32_t(a2) * int32_t(wv[lane]);
                    acc3[lane] += int32_t(a3) * int32_t(wv[lane]);
                    acc4[lane] += int32_t(a4) * int32_t(wv[lane]);
                    acc5[lane] += int32_t(a5) * int32_t(wv[lane]);
                    acc6[lane] += int32_t(a6) * int32_t(wv[lane]);
                    acc7[lane] += int32_t(a7) * int32_t(wv[lane]);
                }
            }

            int8_t result0[4];
            int8_t result1[4];
            int8_t result2[4];
            int8_t result3[4];
            int8_t result4[4];
            int8_t result5[4];
            int8_t result6[4];
            int8_t result7[4];

            if (scales) {
                for (int lane = 0; lane < 4; lane++) {
                    result0[lane]  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih + lane] * float(acc0[lane])), float(minValue))));
                    result1[lane]  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih + lane] * float(acc1[lane])), float(minValue))));
                    result2[lane]  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih + lane] * float(acc2[lane])), float(minValue))));
                    result3[lane]  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih + lane] * float(acc3[lane])), float(minValue))));
                    result4[lane]  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih + lane] * float(acc4[lane])), float(minValue))));
                    result5[lane]  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih + lane] * float(acc5[lane])), float(minValue))));
                    result6[lane]  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih + lane] * float(acc6[lane])), float(minValue))));
                    result7[lane]  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih + lane] * float(acc7[lane])), float(minValue))));
                }
            } else {
                for (int lane = 0; lane < 4; lane++) {
                    result0[lane]  = static_cast<int8_t>(roundf(std::max(std::min(maxValue, acc0[lane]), minValue)));
                    result1[lane]  = static_cast<int8_t>(roundf(std::max(std::min(maxValue, acc1[lane]), minValue)));
                    result2[lane]  = static_cast<int8_t>(roundf(std::max(std::min(maxValue, acc2[lane]), minValue)));
                    result3[lane]  = static_cast<int8_t>(roundf(std::max(std::min(maxValue, acc3[lane]), minValue)));
                    result4[lane]  = static_cast<int8_t>(roundf(std::max(std::min(maxValue, acc4[lane]), minValue)));
                    result5[lane]  = static_cast<int8_t>(roundf(std::max(std::min(maxValue, acc5[lane]), minValue)));
                    result6[lane]  = static_cast<int8_t>(roundf(std::max(std::min(maxValue, acc6[lane]), minValue)));
                    result7[lane]  = static_cast<int8_t>(roundf(std::max(std::min(maxValue, acc7[lane]), minValue)));
                }
            }

            memcpy(c         , result0, 4 * sizeof(int8_t));  // store continuous c
            memcpy(c + 4     , result1, 4 * sizeof(int8_t));
            memcpy(c + 4 * 2 , result2, 4 * sizeof(int8_t));
            memcpy(c + 4 * 3 , result3, 4 * sizeof(int8_t));
            memcpy(c + 4 * 4 , result4, 4 * sizeof(int8_t));
            memcpy(c + 4 * 5 , result5, 4 * sizeof(int8_t));
            memcpy(c + 4 * 6 , result6, 4 * sizeof(int8_t));
            memcpy(c + 4 * 7 , result7, 4 * sizeof(int8_t));

        }
        blockC += (ih >> 2) * cStride;
        for (; ih < h; ih++) {
            auto ihSubIndex = ih & 0x03;
            auto c = blockC + ihSubIndex;
            const int32_t initValue = nullptr != bias ? bias[ih] : 0;
            int32_t acc0 = initValue;
            int32_t acc1 = initValue;
            int32_t acc2 = initValue;
            int32_t acc3 = initValue;
            int32_t acc4 = initValue;
            int32_t acc5 = initValue;
            int32_t acc6 = initValue;
            int32_t acc7 = initValue;

            const int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {
                const int diff = *dataOffset++;
                const int8_t a0 = a[0];
                const int8_t a1 = a[1];
                const int8_t a2 = a[2];
                const int8_t a3 = a[3];
                const int8_t a4 = a[4];
                const int8_t a5 = a[5];
                const int8_t a6 = a[6];
                const int8_t a7 = a[7];
                const int8_t oneW = *w++;
                // MNN_PRINT("8-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%d, a value[0-7]:\n", ie, a - A, w - B - 1, c - C, oneW);
                // formatMatrix(a, {8});
                // MNN_PRINT("\n");
                a = a + diff;
                acc0 += int32_t(a0) * int32_t(oneW);
                acc1 += int32_t(a1) * int32_t(oneW);
                acc2 += int32_t(a2) * int32_t(oneW);
                acc3 += int32_t(a3) * int32_t(oneW);
                acc4 += int32_t(a4) * int32_t(oneW);
                acc5 += int32_t(a5) * int32_t(oneW);
                acc6 += int32_t(a6) * int32_t(oneW);
                acc7 += int32_t(a7) * int32_t(oneW);
            }

            int8_t result0;
            int8_t result1;
            int8_t result2;
            int8_t result3;
            int8_t result4;
            int8_t result5;
            int8_t result6;
            int8_t result7;
            if (scales) {
                result0  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih] * float(acc0)), float(minValue))));
                result1  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih] * float(acc1)), float(minValue))));
                result2  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih] * float(acc2)), float(minValue))));
                result3  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih] * float(acc3)), float(minValue))));
                result4  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih] * float(acc4)), float(minValue))));
                result5  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih] * float(acc5)), float(minValue))));
                result6  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih] * float(acc6)), float(minValue))));
                result7  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih] * float(acc7)), float(minValue))));

            } else {
                result0  = static_cast<int8_t>(std::max(std::min(maxValue, acc0), minValue));
                result1  = static_cast<int8_t>(std::max(std::min(maxValue, acc1), minValue));
                result2  = static_cast<int8_t>(std::max(std::min(maxValue, acc2), minValue));
                result3  = static_cast<int8_t>(std::max(std::min(maxValue, acc3), minValue));
                result4  = static_cast<int8_t>(std::max(std::min(maxValue, acc4), minValue));
                result5  = static_cast<int8_t>(std::max(std::min(maxValue, acc5), minValue));
                result6  = static_cast<int8_t>(std::max(std::min(maxValue, acc6), minValue));
                result7  = static_cast<int8_t>(std::max(std::min(maxValue, acc7), minValue));
            }

            // how to store faster: st4 / transpose /
            c[0] = result0;
            c[4] = result1;
            c[4 * 2] = result2;
            c[4 * 3] = result3;
            c[4 * 4] = result4;
            c[4 * 5] = result5;
            c[4 * 6] = result6;
            c[4 * 7] = result7;
        }
        ie += 8;
        a += 8;
    }
    if (eSize & 0x04) {
        const int* dataOffset = dataOffsetMap;
        const int diff = *dataOffset++;
        // a = blockA + diff;
        a += diff;
        const int8_t* w = B;
        int8_t* blockC = C + (ie << 2);
        const unsigned int* nnz = NNZMap;

        size_t ih = 0;
        for (; ih < (h & (~0x03)); ih += sparseBlockOC) {
            auto ihPack = ih >> 2;
            auto c = blockC + ihPack * cStride;
            int32_t initValue[4] = {0, 0, 0, 0};
            if (nullptr != bias) {
                memcpy(initValue, bias + ih, 4 * sizeof(int32_t));
            }
            int32_t acc0[4];
            int32_t acc1[4];
            int32_t acc2[4];
            int32_t acc3[4];

            memcpy(acc0, initValue, 4 * sizeof(int32_t));
            memcpy(acc1, initValue, 4 * sizeof(int32_t));
            memcpy(acc2, initValue, 4 * sizeof(int32_t));
            memcpy(acc3, initValue, 4 * sizeof(int32_t));

            const int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {

                const int diff = *dataOffset++;
                const int8_t a0 = a[0];
                const int8_t a1 = a[1];
                const int8_t a2 = a[2];
                const int8_t a3 = a[3];
                const int8_t wv[4] = {*w++, *w++, *w++, *w++};
                // MNN_PRINT("4-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:, a value[0-3]:\n", ie, a - A, w - B - 1, c - C);
                // formatMatrix(wv, {4});
                // formatMatrix(a, {4});
                // MNN_PRINT("\n");
                a = a + diff;
                for (int lane = 0; lane < 4; lane++) {
                    acc0[lane] += int32_t(a0) * int32_t(wv[lane]);
                    acc1[lane] += int32_t(a1) * int32_t(wv[lane]);
                    acc2[lane] += int32_t(a2) * int32_t(wv[lane]);
                    acc3[lane] += int32_t(a3) * int32_t(wv[lane]);
                }
            }

            int8_t result0[4];
            int8_t result1[4];
            int8_t result2[4];
            int8_t result3[4];

            if (scales) {
                for (int lane = 0; lane < 4; lane++) {
                    result0[lane]  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih + lane] * float(acc0[lane])), float(minValue))));
                    result1[lane]  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih + lane] * float(acc1[lane])), float(minValue))));
                    result2[lane]  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih + lane] * float(acc2[lane])), float(minValue))));
                    result3[lane]  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih + lane] * float(acc3[lane])), float(minValue))));
                }
            } else {
                for (int lane = 0; lane < 4; lane++) {
                    result0[lane]  = static_cast<int8_t>(roundf(std::max(std::min(maxValue, acc0[lane]), minValue)));
                    result1[lane]  = static_cast<int8_t>(roundf(std::max(std::min(maxValue, acc1[lane]), minValue)));
                    result2[lane]  = static_cast<int8_t>(roundf(std::max(std::min(maxValue, acc2[lane]), minValue)));
                    result3[lane]  = static_cast<int8_t>(roundf(std::max(std::min(maxValue, acc3[lane]), minValue)));
                }
            }

            memcpy(c         , result0, 4 * sizeof(int8_t));  // store continuous c
            memcpy(c + 4     , result1, 4 * sizeof(int8_t));
            memcpy(c + 4 * 2 , result2, 4 * sizeof(int8_t));
            memcpy(c + 4 * 3 , result3, 4 * sizeof(int8_t));

        }
        blockC += (ih >> 2) * cStride;
        for (; ih < h; ih++) {
            auto ihSubIndex = ih & 0x03;
            auto c = blockC + ihSubIndex;
            const int32_t initValue = nullptr != bias ? bias[ih] : 0;
            int32_t acc0 = initValue;
            int32_t acc1 = initValue;
            int32_t acc2 = initValue;
            int32_t acc3 = initValue;

            const int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {
                const int diff = *dataOffset++;
                const int8_t a0 = a[0];
                const int8_t a1 = a[1];
                const int8_t a2 = a[2];
                const int8_t a3 = a[3];
                const int8_t oneW = *w++;
                // MNN_PRINT("4-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%d, a value[0-3]:\n", ie, a - A, w - B - 1, c - C, oneW);
                // formatMatrix(a, {4});
                // MNN_PRINT("\n");
                a = a + diff;
                acc0 += int32_t(a0) * int32_t(oneW);
                acc1 += int32_t(a1) * int32_t(oneW);
                acc2 += int32_t(a2) * int32_t(oneW);
                acc3 += int32_t(a3) * int32_t(oneW);
            }

            int8_t result0;
            int8_t result1;
            int8_t result2;
            int8_t result3;
            if (scales) {
                result0  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih] * float(acc0)), float(minValue))));
                result1  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih] * float(acc1)), float(minValue))));
                result2  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih] * float(acc2)), float(minValue))));
                result3  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih] * float(acc3)), float(minValue))));
            } else {
                result0  = static_cast<int8_t>(std::max(std::min(maxValue, acc0), minValue));
                result1  = static_cast<int8_t>(std::max(std::min(maxValue, acc1), minValue));
                result2  = static_cast<int8_t>(std::max(std::min(maxValue, acc2), minValue));
                result3  = static_cast<int8_t>(std::max(std::min(maxValue, acc3), minValue));
            }

            // how to store faster: st4 / transpose /
            c[0] = result0;
            c[4] = result1;
            c[4 * 2] = result2;
            c[4 * 3] = result3;
        }
        ie += 4;
        a += 4;
    }
    if (eSize & 0x02) {
        const int* dataOffset = dataOffsetMap;
        const int diff = *dataOffset++;
        // a = blockA + diff;
        a += diff;
        const int8_t* w = B;
        int8_t* blockC = C + (ie << 2);
        const unsigned int* nnz = NNZMap;

        size_t ih = 0;
        for (; ih < (h & (~0x03)); ih += sparseBlockOC) {
            auto ihPack = ih >> 2;
            auto c = blockC + ihPack * cStride;
            int32_t initValue[4] = {0, 0, 0, 0};
            if (nullptr != bias) {
                memcpy(initValue, bias + ih, 4 * sizeof(int32_t));
            }
            int32_t acc0[4];
            int32_t acc1[4];
            memcpy(acc0, initValue, 4 * sizeof(int32_t));
            memcpy(acc1, initValue, 4 * sizeof(int32_t));

            const int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {

                const int diff = *dataOffset++;
                const int8_t a0 = a[0];
                const int8_t a1 = a[1];
                const int8_t wv[4] = {*w++, *w++, *w++, *w++};
                // MNN_PRINT("2-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:, a value[0-1]:\n", ie, a - A, w - B - 1, c - C);
                // formatMatrix(wv, {4});
                // formatMatrix(a, {2});
                // MNN_PRINT("\n");
                a = a + diff;
                for (int lane = 0; lane < 4; lane++) {
                    acc0[lane] += int32_t(a0) * int32_t(wv[lane]);
                    acc1[lane] += int32_t(a1) * int32_t(wv[lane]);
                }
            }

            int8_t result0[4];
            int8_t result1[4];
            if (scales) {
                for (int lane = 0; lane < 4; lane++) {
                    result0[lane]  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih + lane] * float(acc0[lane])), float(minValue))));
                    result1[lane]  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih + lane] * float(acc1[lane])), float(minValue))));
                }
            } else {
                for (int lane = 0; lane < 4; lane++) {
                    result0[lane]  = static_cast<int8_t>(roundf(std::max(std::min(maxValue, acc0[lane]), minValue)));
                    result1[lane]  = static_cast<int8_t>(roundf(std::max(std::min(maxValue, acc1[lane]), minValue)));
                }
            }

            memcpy(c         , result0, 4 * sizeof(int8_t));  // store continuous c
            memcpy(c + 4     , result1, 4 * sizeof(int8_t));
        }
        blockC += (ih >> 2) * cStride;
        for (; ih < h; ih++) {
            auto ihSubIndex = ih & 0x03;
            auto c = blockC + ihSubIndex;
            const int32_t initValue = nullptr != bias ? bias[ih] : 0;
            int32_t acc0 = initValue;
            int32_t acc1 = initValue;

            const int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {
                const int diff = *dataOffset++;
                const int8_t a0 = a[0];
                const int8_t a1 = a[1];
                const int8_t oneW = *w++;
                // MNN_PRINT("2-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%d, a value[0-1]:\n", ie, a - A, w - B - 1, c - C, oneW);
                // formatMatrix(a, {2});
                // MNN_PRINT("\n");
                a = a + diff;
                acc0 += int32_t(a0) * int32_t(oneW);
                acc1 += int32_t(a1) * int32_t(oneW);
            }

            int8_t result0;
            int8_t result1;
            if (scales) {
                result0  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih] * float(acc0)), float(minValue))));
                result1  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih] * float(acc1)), float(minValue))));
            } else {
                result0  = static_cast<int8_t>(std::max(std::min(maxValue, acc0), minValue));
                result1  = static_cast<int8_t>(std::max(std::min(maxValue, acc1), minValue));
            }

            // how to store faster: st4 / transpose /
            c[0] = result0;
            c[4] = result1;
        }
        ie += 2;
        a += 2;
    }
    if (eSize & 0x01) {
        const int* dataOffset = dataOffsetMap;
        const int diff = *dataOffset++;
        // const float* a = blockA + diff;
        a += diff;
        const int8_t * w = B;
        int8_t * blockC = C + (ie << 2);
        const unsigned int* nnz = NNZMap;

        size_t ih = 0;
        for (; ih < (h & (~0x03)); ih += sparseBlockOC) {
            auto ihPack = ih >> 2;
            auto c = blockC + ihPack * cStride;
            int32_t initValue[4] = {0, 0, 0, 0};
            if (nullptr != bias) {
                memcpy(initValue, bias + ih, 4 * sizeof(int32_t));
            }
            int32_t acc0[4];
            memcpy(acc0, initValue, 4 * sizeof(int32_t));
            const int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {

                const int diff = *dataOffset++;
                const int8_t a0 = a[0];
                const int8_t wv[4] = {*w++, *w++, *w++, *w++};
                // MNN_PRINT("16-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:, a value[0-1]:\n", ie, a - A, w - B - 1, c - C);
                // formatMatrix(wv, {4});
                // formatMatrix(a, {16});
                // MNN_PRINT("\n");
                a = a + diff;
                for (int lane = 0; lane < 4; lane++) {
                    acc0[lane] += int32_t(a0) * int32_t(wv[lane]);
                }
            }

            int8_t result0[4];
            if (scales) {
                for (int lane = 0; lane < 4; lane++) {
                    result0[lane]  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih + lane] * float(acc0[lane])), float(minValue))));
                }
            } else {
                for (int lane = 0; lane < 4; lane++) {
                    result0[lane]  = static_cast<int8_t>(roundf(std::max(std::min(maxValue, acc0[lane]), minValue)));
                }
            }
            memcpy(c, result0, 4 * sizeof(int8_t));  // store continuous c
        }
        blockC += (ih >> 2) * cStride;
        for (; ih < h; ih++) {
            auto ihSubIndex = ih & 0x03;
            auto c = blockC + ihSubIndex;
            const int32_t initValue = nullptr != bias ? bias[ih] : 0;
            int32_t acc0 = initValue;

            const int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {
                const int diff = *dataOffset++;
                const int8_t a0 = a[0];
                const int8_t oneW = *w++;

                // MNN_PRINT("1-loop: ie:%zu, a offset:%ld, c offset:%ld, w offset:%ld, w value:%d, a value[0]:\n", ie, a - A, w - B - 1, c - C, oneW);
                // formatMatrix(a, {1});
                // MNN_PRINT("\n");
                a = a + diff;
                acc0 += int32_t(a0) * int32_t(oneW);
            }
            int8_t result0;
            if (scales) {
                result0  = static_cast<int8_t>(roundf(std::max(std::min(float(maxValue), scales[ih] * float(acc0)), float(minValue))));
            } else {
                result0  = static_cast<int8_t>(std::max(std::min(maxValue, acc0), minValue));
            }
            // how to store faster: st4 / transpose /
            c[0] = result0;
        }
        ie += 1;
        // a += 1;
    }

}


static int8_t MNNInt32ToInt8(int data, int bias, float scale, float maxValue, float minValue)
{
    float value = (float)(data + bias) * scale;
    value       = ALIMAX(value, minValue);
    value       = ALIMIN(value, maxValue);
    return static_cast<int8_t>(roundf(value));
}

static void MNNGemmInt8AddBiasScale_16x4_Unit(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step,
                                              size_t dst_depth_quad, const QuanPostTreatParameters* post, size_t realCount) {
    const int bytes = ((post->useInt8 == 1) ? 1 : 4);
    float fp32min = 0, fp32max = 0;
    int weight_step_Z = src_depth_quad * (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT);
    int weight_step_Y = (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT);
    const auto srcSumPtr = post->srcKernelSum;
    if (0 == post->useInt8 && post->fp32minmax) {
        fp32min = (post->fp32minmax)[0];
        fp32max = (post->fp32minmax)[1];
    }

    float* biasPtr = (float*)post->biasFloat;
    
    for (int dz = 0; dz < dst_depth_quad; ++dz) {
        const auto weight_dz = weight + weight_step_Z * dz;
        const auto bias_dz   = biasPtr + dz * GEMM_INT8_UNIT;
        const auto weight_zero = post->weightQuanBias + (dz * GEMM_INT8_UNIT);
        const float* scale_dz = nullptr;
        scale_dz  = post->scale + (dz * GEMM_INT8_UNIT);
        auto dst_z           = dst + dz * dst_step;
        for (int w = 0; w < realCount; ++w) {
            const auto src_x   = src + w * GEMM_INT8_SRC_UNIT;
            auto dst_x         = dst_z + w * GEMM_INT8_UNIT * bytes;
            int32_t dstTemp[4] = {0, 0, 0, 0};

            for (int sz = 0; sz < src_depth_quad; ++sz) {
                const auto weight_sz = weight_dz + weight_step_Y * sz;
                const auto src_z     = src_x + sz * realCount * GEMM_INT8_SRC_UNIT;

                for (int j = 0; j < GEMM_INT8_UNIT; ++j) {
                    const auto weight_j = weight_sz + j * GEMM_INT8_SRC_UNIT;
                    for (int i = 0; i < GEMM_INT8_SRC_UNIT; ++i) {
                        dstTemp[j] += (int32_t)src_z[i] * (int32_t)weight_j[i];
                    }
                }
            }

            for (int j = 0; j < GEMM_INT8_UNIT; ++j) {
                float value = dstTemp[j] * scale_dz[j] + srcSumPtr[w] * weight_zero[j];
                if (post->extraScale) {
                    value = dstTemp[j] * scale_dz[j] * post->extraScale[w] + srcSumPtr[w] * weight_zero[j];
                }
                if (post->useInt8 == 0) {
                    if (biasPtr) {
                        value += bias_dz[j];
                    } else {
                        float dstv = ((float*)dst_x)[j];
                        value += dstv;
                    }
                    if (post->fp32minmax) {
                        value = std::min(std::max(fp32min, value), fp32max);
                    }
                    ((float*)dst_x)[j] = value;
                } else {
                    value += bias_dz[j];
                    value       = ALIMAX(value, post->minValue);
                    value       = ALIMIN(value, post->maxValue);
                    dst_x[j] = static_cast<int8_t>(roundf(value));
                }
            }
        }
    }
}

static void MNNGemmInt8AddBiasScale_16x4_w4_Unit(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad, const QuanPostTreatParameters* post, size_t realCount) {
    uint32_t c = 0xf;
    const int bytes = 4;
    float fp32min = 0, fp32max = 0;
    int weight_step_Z = 0.5 * (src_depth_quad) * (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT);
    int weight_step_Y = 0.5 * (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT);
    MNN_ASSERT(post->useInt8==0);
    if (post->fp32minmax) {
        fp32min = (post->fp32minmax)[0];
        fp32max = (post->fp32minmax)[1];
    }

    float* biasPtr = (float*)post->biasFloat;

    const auto srcSumPtr = post->srcKernelSum;
    for (int dz = 0; dz < dst_depth_quad; ++dz) {
        const auto weight_dz = weight + weight_step_Z * dz;
        const auto bias_dz   = biasPtr + dz * GEMM_INT8_UNIT;
        const auto weight_zero = post->weightQuanBias + (dz * GEMM_INT8_UNIT);
        const float* scale_dz = nullptr;
        scale_dz  = post->scale + (dz * GEMM_INT8_UNIT);
        auto dst_z           = dst + dz * dst_step;
        for (int w = 0; w < realCount; ++w) {
            const auto src_x   = src + w * GEMM_INT8_SRC_UNIT;
            auto dst_x         = dst_z + w * GEMM_INT8_UNIT * bytes;
            int32_t dstTemp[4] = {0, 0, 0, 0};

            for (int sz = 0; sz < src_depth_quad; ++sz) {
                const auto weight_sz = (uint8_t*)weight_dz + weight_step_Y * sz;
                const auto src_z     = src_x + sz * realCount * GEMM_INT8_SRC_UNIT;

                int w8[64]; // 64=GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT
                for (int k = 0; k < 32; ++k) {
                    w8[k] = (weight_sz[k]>>4);
                    w8[k + 32] = (weight_sz[k] & c);
                }

                for (int j = 0; j < GEMM_INT8_UNIT; ++j) {
                    const auto weight_j = w8 + j * GEMM_INT8_SRC_UNIT;
                    for (int i = 0; i < GEMM_INT8_SRC_UNIT; ++i) {
                        dstTemp[j] += (int32_t)src_z[i] * (int32_t)weight_j[i];
                    }
                }
            }

            for (int j = 0; j < GEMM_INT8_UNIT; ++j) {
                float value = dstTemp[j] * scale_dz[j] + srcSumPtr[w] * weight_zero[j];
                if (post->extraScale) {
                    value = dstTemp[j] * scale_dz[j] * post->extraScale[w] + srcSumPtr[w] * weight_zero[j];
                }

                if (biasPtr) {
                    value += bias_dz[j];
                } else {
                    float dstv = ((float*)dst_x)[j];
                    value += dstv;
                }
                if (post->fp32minmax) {
                    value = std::min(std::max(fp32min, value), fp32max);
                }
                ((float*)dst_x)[j] = value;
            }
        }
    }
}

static void MNNReluWithSlopeChannelInt8(int8_t* dst, const int8_t* src, const float* slope, size_t planeNumber, size_t depthQuad, const QuanPrePostParameters *params, size_t pack) {
#ifdef MNN_USE_SSE
float offset = 128.f;
uint8_t* srcPtr = (uint8_t*)src;
uint8_t* dstPtr = (uint8_t*)dst;
#else
float offset = 0.f;
const int8_t* srcPtr = src;
int8_t* dstPtr = dst;
#endif
    float mulVal = 0.f;
    float inputZero = static_cast<float>(params->inputZeroPoint[0]) + offset;
    float outputZero = static_cast<float>(params->outputZeroPoint[0]) + offset;
    int32_t minval = params->minValue + offset;
    int32_t maxval = params->maxValue + offset;
    for (int j = 0;j < depthQuad; ++j) {
        const float* slopeZ = slope + pack * j;
        const auto srcZ = srcPtr + pack * j * planeNumber;
        auto dstZ = dstPtr + pack * j * planeNumber;
        for (int i = 0; i < planeNumber; ++i) {
            for (int c = 0; c < pack; ++c) {
                float valInput = (static_cast<float>(srcZ[pack * i + c]) - inputZero) * params->inputScale[0];
                if (valInput < 0) {
                    valInput *= slopeZ[c];
                }
                auto mulVal = valInput * params->outputScale[0] + outputZero;
                dstZ[pack * i + c] = ALIMIN(ALIMAX(static_cast<int32_t>(roundf(mulVal)), minval), maxval);
            }
        }
    }
}

static void MNNGemmInt8AddBiasScale_16x4_Unit_FAST(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad, const QuanPostTreatParameters* post, size_t realCount) {
    return MNNGemmInt8AddBiasScale_16x4_Unit(dst, src, weight, src_depth_quad, dst_step, dst_depth_quad, post, realCount);
}

static void MNNLineDepthWiseInt8AddBiasScaleUnit(int8_t* dst, const int8_t* src, const int8_t* weight, const QuanPostTreatParameters* parameters,
                                          size_t width, size_t src_w_step, size_t fw, size_t fh, size_t dilateX_step,
                                          size_t dilateY_step, int8_t* idxOrder) {
#ifdef MNN_USE_SSE
    int offset = 128;
    uint8_t* dstPtr = (uint8_t*)dst;
    const int16_t* srcPtr = (int16_t*)src;
    const int16_t* weightPtr = (int16_t*)weight;
#else
    int offset = 0;
    int8_t* dstPtr = dst;
    const int8_t* srcPtr = src;
    const int8_t* weightPtr = weight;
#endif
    int pack = 16;
    auto bias_z = parameters->bias;
    auto scale_z = parameters->scale;
    int dx, fx, fy;
    for (dx = 0; dx < width; ++dx) {
        auto dst_x          = dstPtr + dx * pack;
        int32_t dstInt32[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        const auto src_z    = srcPtr + src_w_step * dx;
        for (fy = 0; fy < fh; ++fy) {
            const auto src_y    = src_z + fy * dilateY_step;
            const auto weight_y = weightPtr + fy * fw * pack;
            for (fx = 0; fx < fw; ++fx) {
                const auto src_x    = src_y + fx * dilateX_step;
                const auto weight_x = weight_y + pack * fx;
                for (int j = 0; j < pack; ++j) {
                    dstInt32[j] += static_cast<int32_t>(src_x[j]) * static_cast<int32_t>(weight_x[j]);
                }
            }
        }

        for (int i = 0; i < pack; ++i) {

            float val = (dstInt32[i] + bias_z[i]) * scale_z[i];
            int valOut = roundf(val) + offset;
            if (valOut > parameters->maxValue + offset) {
                valOut = parameters->maxValue + offset;
            }
            if (valOut < parameters->minValue + offset) {
                valOut = parameters->minValue + offset;
            }
            dst_x[i] = static_cast<int>(valOut);
        }
    }
}

static void MNNLineDepthWiseInt8AddBiasScaleUnit3x3(int8_t* dst, const int8_t* src, const int8_t* weight, const QuanPostTreatParameters* parameters,
                                          size_t width, size_t src_w_step, size_t fw, size_t fh, size_t dilateX_step, size_t dilateY_step, int8_t* idxOrder) {
    MNNLineDepthWiseInt8AddBiasScaleUnit(dst, src, weight, parameters, width, src_w_step, fw, fh, dilateX_step, dilateY_step, idxOrder);
}
#endif

#ifndef MNN_USE_NEON
void MNNFloat2Int8(const float* src, int8_t* dst, size_t sizeQuad, const float* scalep, ssize_t minValue,
                   ssize_t maxValue, const float* zeroPoint, ssize_t quanParamVec) {
    // quanParamVec:
    // 00: scale is vector
    // 10: zero is vector
    // 11: both are vector
    float scale4[4] = {scalep[0], scalep[0], scalep[0], scalep[0] };
    float zero4[4] = {zeroPoint[0], zeroPoint[0], zeroPoint[0], zeroPoint[0]};
    if (quanParamVec % 2 == 1) {
        scale4[0] = scalep[0];
        scale4[1] = scalep[1];
        scale4[2] = scalep[2];
        scale4[3] = scalep[3];
    }
    if (quanParamVec >> 1 == 1) {
        zero4[0] = zeroPoint[0];
        zero4[1] = zeroPoint[1];
        zero4[2] = zeroPoint[2];
        zero4[3] = zeroPoint[3];
    }
    for (int i = 0; i < sizeQuad; ++i) {
        for (int j=0; j<4; ++j) {
            int v = (int)roundf(src[4*i+j] * scale4[j]) + zero4[j];
            if (v > maxValue) {
                v = maxValue;
            }
            if (v < minValue) {
                v = minValue;
            }
            dst[4*i+j] = v;
        }
    }
}

void MNNInt8ScaleToFloat(float* dst, const int8_t* src, const float* scale, size_t size, const float* zeroPoint, ssize_t quantParamVec) {
    float scale_[4] = {scale[0], scale[0], scale[0], scale[0]};
    float zero_[4] = {zeroPoint[0], zeroPoint[0], zeroPoint[0], zeroPoint[0]};
    if (quantParamVec & 1) {
        ::memcpy(scale_, scale, 4 * sizeof(float));
    }
    if (quantParamVec >> 1) {
        ::memcpy(zero_, zeroPoint, 4 * sizeof(float));
    }
    for (int i = 0; i < size; ++i) {
        const auto srcStart = src + i * 4;
        auto dstStart       = dst + i * 4;
        for (int j = 0; j < 4; ++j) {
            dstStart[j] = static_cast<float>(srcStart[j] - zero_[j]) * scale_[j];
        }
    }
}

void MNNAvgPoolInt8(int8_t* dst, int8_t* src, size_t outputWidth, size_t inputWidth, size_t kernelx, size_t kernely, size_t stridesx, ssize_t paddingx, ssize_t factor) {
    int pack = 16;
    int8_t* dstPtr = dst;
    const int8_t* srcPtr = src;
    for (int ox = 0; ox < outputWidth; ++ox) {
        std::vector<int> sum_(pack, 0);
        for (int y = 0; y < kernely; ++y) {
            for (int x = 0; x < kernelx; ++x) {
                const int8_t *inputPtr = srcPtr + pack* (x + inputWidth* y);
                for (int idx = 0; idx < pack; ++idx) {
                    sum_[idx] += *(inputPtr + idx);
                }
            }
        }
        for (int idx = 0; idx < pack; ++idx) {
            *(dstPtr + idx) = static_cast<int8_t>((sum_[idx] * factor)>>24);
        }
        dstPtr = dstPtr + pack;
        srcPtr = srcPtr + pack* stridesx;
    }
}

void MNNMaxPoolInt8(int8_t* dst, int8_t* src, size_t outputWidth, size_t inputWidth, size_t kernelx, size_t kernely, size_t stridesx) {
    int pack = 16;
    int8_t* dstPtr = dst;
    const int8_t* srcPtr = src;
    for (int ox = 0; ox < outputWidth; ++ox){
        std::vector<int8_t> results(pack, INT8_MIN);
        for (int y = 0; y < kernely; ++y) {
            for (int x = 0; x < kernelx; ++x) {
                const int8_t* inputPtr = srcPtr + pack* (x + inputWidth* y);
                for (int idx = 0; idx < pack; ++idx) {
                    results[idx] = std::max(results[idx], *(inputPtr + idx));
                }
            }
        }

        for (int idx = 0; idx < pack;++idx) {
            *(dstPtr + idx) = results[idx];
        }
        dstPtr = dstPtr + pack;
        srcPtr = srcPtr + pack* stridesx;
    }
}

void MNNBinaryAddInt8 (int8_t* outputRaw, const int8_t* inputRaw0, const int8_t* inputRaw1, ssize_t* inputScalesInt32, float* inputScalesFp32, const QuanPrePostParameters* params, size_t elementSize, size_t needBroadcast) {
    float sum = 0;
#ifdef MNN_USE_SSE
    const int offset = 128;
    const uint8_t* inputData0 = (uint8_t*)inputRaw0;
    const uint8_t* inputData1 = (uint8_t*)inputRaw1;
    uint8_t* outputData = (uint8_t*)outputRaw;
#else
    const int offset = 0;
    const int8_t* inputData0 = inputRaw0;
    const int8_t* inputData1 = inputRaw1;
    int8_t* outputData = outputRaw;
#endif
    const int maxValue = static_cast<int32_t>(params->maxValue) + offset;
    const int minValue = static_cast<int32_t>(params->minValue) + offset;
    for (int i = 0; i < elementSize; ++i) {
        if (needBroadcast == 0) {
            float inp0 = static_cast<int32_t>(inputData0[0] - offset - (int32_t)params->inputZeroPoint[0]) * static_cast<float>(inputScalesFp32[0]);
           float inp1 = static_cast<int32_t>(inputData1[i] - offset - (int32_t)params->inputZeroPoint[1]) * static_cast<float>(inputScalesFp32[1]);
            sum =  inp0 + inp1;
        } else if (needBroadcast == 1) {
            float inp0 = static_cast<int32_t>(inputData0[i] - offset - (int32_t)params->inputZeroPoint[0]) * static_cast<float>(inputScalesFp32[0]);
           float inp1 = static_cast<int32_t>(inputData1[0] - offset - (int32_t)params->inputZeroPoint[1]) * static_cast<float>(inputScalesFp32[1]);
            sum = inp0 + inp1;
        } else {
           float inp0 = static_cast<int32_t>(inputData0[i] - offset - (int32_t)params->inputZeroPoint[0]) * static_cast<float>(inputScalesFp32[0]);
           float inp1 = static_cast<int32_t>(inputData1[i] - offset - (int32_t)params->inputZeroPoint[1]) * static_cast<float>(inputScalesFp32[1]);
           sum = inp0 + inp1;
        }
        int value = (int)roundf(sum * inputScalesFp32[2]) + offset + static_cast<int32_t>(params->outputZeroPoint[0]);
        if (value > maxValue) {
            value = maxValue;
        }
        if (value < minValue) {
            value = minValue;
        }
        outputData[i] = value;
    }
}

void MNNBinarySubInt8 (int8_t* outputRaw, const int8_t* inputRaw0, const int8_t* inputRaw1, ssize_t* inputScalesInt32, float* inputScalesFp32, const QuanPrePostParameters* params, size_t elementSize, size_t needBroadcast) {
    float res = 0;
#ifdef MNN_USE_SSE
    const int offset = 128;
    const uint8_t* inputData0 = (uint8_t*)inputRaw0;
    const uint8_t* inputData1 = (uint8_t*)inputRaw1;
    uint8_t* outputData = (uint8_t*)outputRaw;
#else
    const int offset = 0;
    const int8_t* inputData0 = inputRaw0;
    const int8_t* inputData1 = inputRaw1;
    int8_t* outputData = outputRaw;
#endif
    const int maxValue = static_cast<int32_t>(params->maxValue) + offset;
    const int minValue = static_cast<int32_t>(params->minValue) + offset;
    for (int i = 0; i < elementSize; ++i) {
        if (needBroadcast == 0) {
           float inp0 = static_cast<int32_t>(inputData0[0] - offset - (int32_t)params->inputZeroPoint[0]) * static_cast<float>(inputScalesFp32[0]);
           float inp1 = static_cast<int32_t>(inputData1[i] - offset - (int32_t)params->inputZeroPoint[1]) * static_cast<float>(inputScalesFp32[1]);
            res = inp0 - inp1;
        } else if (needBroadcast == 1) {
            float inp0 = static_cast<int32_t>(inputData0[i] - offset - (int32_t)params->inputZeroPoint[0]) * static_cast<float>(inputScalesFp32[0]);
            float inp1 = static_cast<int32_t>(inputData1[0] - offset - (int32_t)params->inputZeroPoint[1]) * static_cast<float>(inputScalesFp32[1]);
            res = inp0 - inp1;
        } else {
            float inp0 = static_cast<int32_t>(inputData0[i] - offset - (int32_t)params->inputZeroPoint[0]) * static_cast<float>(inputScalesFp32[0]);
            float inp1 = static_cast<int32_t>(inputData1[i] - offset - (int32_t)params->inputZeroPoint[1]) * static_cast<float>(inputScalesFp32[1]);
            res = inp0 - inp1;
        }
        int value = (int)roundf(res * inputScalesFp32[2]) + offset + static_cast<int32_t>(params->outputZeroPoint[0]);
        if (value > maxValue) {
            value = maxValue;
        }
        if (value < minValue) {
            value = minValue;
        }
        outputData[i] = value;
    }
}

void MNNBinaryMulInt8 (int8_t* outputRaw, const int8_t* inputRaw0, const int8_t* inputRaw1, ssize_t* inputScalesInt32, float* inputScalesFp32, const QuanPrePostParameters* params, size_t elementSize, size_t needBroadcast) {
    float res = 0;
#ifdef MNN_USE_SSE
    const int offset = 128;
    const uint8_t* inputData0 = (uint8_t*)inputRaw0;
    const uint8_t* inputData1 = (uint8_t*)inputRaw1;
    uint8_t* outputData = (uint8_t*)outputRaw;
#else
    const int offset = 0;
    const int8_t* inputData0 = inputRaw0;
    const int8_t* inputData1 = inputRaw1;
    int8_t* outputData = outputRaw;
#endif
    const int maxValue = static_cast<int32_t>(params->maxValue) + offset;
    const int minValue = static_cast<int32_t>(params->minValue) + offset;
    for (int i = 0; i < elementSize; ++i) {
        if (needBroadcast == 0) {
            float inp0 = (inputData0[0] - offset - params->inputZeroPoint[0]) * inputScalesFp32[0];
            float inp1 = (inputData1[i] - offset - params->inputZeroPoint[1]) * inputScalesFp32[1];
            res = inp0 * inp1;
        } else if (needBroadcast == 1) {
            float inp0 = (inputData0[i] - offset - params->inputZeroPoint[0]) * inputScalesFp32[0];
            float inp1 = (inputData1[0] - offset - params->inputZeroPoint[1]) * inputScalesFp32[1];
            res = inp0 * inp1;
        } else {
            float inp0 = (inputData0[i] - offset - params->inputZeroPoint[0]) * inputScalesFp32[0];
            float inp1 = (inputData1[i] - offset - params->inputZeroPoint[1]) * inputScalesFp32[1];
            res = inp0 * inp1;
        }
        int value = (int)roundf(res * inputScalesFp32[2]) + offset + static_cast<int32_t>(params->outputZeroPoint[0]);
        if (value > maxValue) {
            value = maxValue;
        }
        if (value < minValue) {
            value = minValue;
        }
        outputData[i] = value;
    }
}

void MNNBinaryMinInt8 (int8_t* outputRaw, const int8_t* inputRaw0, const int8_t* inputRaw1, ssize_t* inputScalesInt32, float* inputScalesFp32, const QuanPrePostParameters* params, size_t elementSize, size_t needBroadcast) {
    int res = 0;
#ifdef MNN_USE_SSE
    const int offset = 128;
    const uint8_t* inputData0 = (uint8_t*)inputRaw0;
    const uint8_t* inputData1 = (uint8_t*)inputRaw1;
    uint8_t* outputData = (uint8_t*)outputRaw;
#else
    const int offset = 0;
    const int8_t* inputData0 = inputRaw0;
    const int8_t* inputData1 = inputRaw1;
    int8_t* outputData = outputRaw;
#endif
    const int maxValue = static_cast<int32_t>(params->maxValue) + offset;
    const int minValue = static_cast<int32_t>(params->minValue) + offset;
    for (int i = 0; i < elementSize; ++i) {
        if (needBroadcast == 0) {
            int32_t inp0 = static_cast<int32_t>(inputData0[0] - offset - params->inputZeroPoint[0]) * static_cast<int32_t>(inputScalesInt32[0]);
            int32_t inp1 = static_cast<int32_t>(inputData1[i] - offset - params->inputZeroPoint[1]) * static_cast<int32_t>(inputScalesInt32[1]);
            res = std::min(inp0, inp1);
        } else if (needBroadcast == 1) {
            int32_t inp0 = static_cast<int32_t>(inputData0[i] - offset - params->inputZeroPoint[0]) * static_cast<int32_t>(inputScalesInt32[0]);
            int32_t inp1 = static_cast<int32_t>(inputData1[0] - offset - params->inputZeroPoint[1]) * static_cast<int32_t>(inputScalesInt32[1]);
            res = std::min(inp0, inp1);
        } else {
            int32_t inp0 = static_cast<int32_t>(inputData0[i] - offset - params->inputZeroPoint[0]) * static_cast<int32_t>(inputScalesInt32[0]);
            int32_t inp1 = static_cast<int32_t>(inputData1[i] - offset - params->inputZeroPoint[1]) * static_cast<int32_t>(inputScalesInt32[1]);
            res = std::min(inp0, inp1);
        }
        int value  = roundf((res + (1<<15)) / (1 << 16)) + offset + static_cast<int32_t>(params->outputZeroPoint[0]);
        if (res < 0) {
            value  = roundf((res - (1<<15)) / (1 << 16)) + offset + static_cast<int32_t>(params->outputZeroPoint[0]);
        }
        if (value > maxValue) {
            value = maxValue;
        }
        if (value < minValue) {
            value = minValue;
        }
        outputData[i] = value;
    }
}

void MNNBinaryMaxInt8 (int8_t* outputRaw, const int8_t* inputRaw0, const int8_t* inputRaw1, ssize_t* inputScalesInt32, float* inputScalesFp32, const QuanPrePostParameters* params, size_t elementSize, size_t needBroadcast) {
    int res = 0;
#ifdef MNN_USE_SSE
    const int offset = 128;
    const uint8_t* inputData0 = (uint8_t*)inputRaw0;
    const uint8_t* inputData1 = (uint8_t*)inputRaw1;
    uint8_t* outputData = (uint8_t*)outputRaw;
#else
    const int offset = 0;
    const int8_t* inputData0 = inputRaw0;
    const int8_t* inputData1 = inputRaw1;
    int8_t* outputData = outputRaw;
#endif
    const int maxValue = static_cast<int32_t>(params->maxValue) + offset;
    const int minValue = static_cast<int32_t>(params->minValue) + offset;
    for (int i = 0; i < elementSize; ++i) {
        if (needBroadcast == 0) {
            int32_t inp0 = static_cast<int32_t>(inputData0[0] - offset - params->inputZeroPoint[0]) * static_cast<int32_t>(inputScalesInt32[0]);
            int32_t inp1 = static_cast<int32_t>(inputData1[i] - offset - params->inputZeroPoint[1]) * static_cast<int32_t>(inputScalesInt32[1]);
            res = std::max(inp0, inp1);
        } else if (needBroadcast == 1) {
            int32_t inp0 = static_cast<int32_t>(inputData0[i] - offset - params->inputZeroPoint[0]) * static_cast<int32_t>(inputScalesInt32[0]);
            int32_t inp1 = static_cast<int32_t>(inputData1[0] - offset - params->inputZeroPoint[1]) * static_cast<int32_t>(inputScalesInt32[1]);
            res = std::max(inp0, inp1);
        } else {
            int32_t inp0 = static_cast<int32_t>(inputData0[i] - offset - params->inputZeroPoint[0]) * static_cast<int32_t>(inputScalesInt32[0]);
            int32_t inp1 = static_cast<int32_t>(inputData1[i] - offset - params->inputZeroPoint[1]) * static_cast<int32_t>(inputScalesInt32[1]);
            res = std::max(inp0, inp1);
        }
        int value  = (res + (1<<15)) / (1 << 16) + offset + static_cast<int32_t>(params->outputZeroPoint[0]);
        if (res < 0) {
            value  = (res - (1<<15)) / (1 << 16) + offset + static_cast<int32_t>(params->outputZeroPoint[0]);
        }
        if (value > maxValue) {
            value = maxValue;
        }
        if (value < minValue) {
            value = minValue;
        }
        outputData[i] = value;
    }
}
void MNNBinarySqdInt8 (int8_t* outputRaw, const int8_t* inputRaw0, const int8_t* inputRaw1, ssize_t* inputScalesInt32, float* inputScalesFp32, const QuanPrePostParameters* params, size_t elementSize, size_t needBroadcast) {
    float res = 0;
#ifdef MNN_USE_SSE
    const int offset = 128;
    const uint8_t* inputData0 = (uint8_t*)inputRaw0;
    const uint8_t* inputData1 = (uint8_t*)inputRaw1;
    uint8_t* outputData = (uint8_t*)outputRaw;
#else
    const int offset = 0;
    const int8_t* inputData0 = inputRaw0;
    const int8_t* inputData1 = inputRaw1;
    int8_t* outputData = outputRaw;
#endif
    const int maxValue = static_cast<int32_t>(params->maxValue) + offset;
    const int minValue = static_cast<int32_t>(params->minValue) + offset;
    for (int i = 0; i < elementSize; ++i) {
        if (needBroadcast == 0) {
            float inp0 = (inputData0[0] - offset - params->inputZeroPoint[0]) * inputScalesFp32[0];
            float inp1 = (inputData1[i] - offset - params->inputZeroPoint[1]) * inputScalesFp32[1];
            res = (inp0 - inp1) * (inp0 - inp1);
        } else if (needBroadcast == 1) {
            float inp0 = (inputData0[i] - offset - params->inputZeroPoint[0]) * inputScalesFp32[0];
            float inp1 = (inputData1[0] - offset - params->inputZeroPoint[1]) * inputScalesFp32[1];
            res = (inp0 - inp1) * (inp0 - inp1);
        } else {
            float inp0 = (inputData0[i] - offset - params->inputZeroPoint[0]) * inputScalesFp32[0];
            float inp1 = (inputData1[i] - offset - params->inputZeroPoint[1]) * inputScalesFp32[1];
            res = (inp0 - inp1) * (inp0 - inp1);
        }
        int value = (int)roundf(res * inputScalesFp32[2]) + offset + static_cast<int32_t>(params->outputZeroPoint[0]);
        if (value > maxValue) {
            value = maxValue;
        }
        if (value < minValue) {
            value = minValue;
        }
        outputData[i] = value;
    }
}

void MNNScaleAndAddBiasInt8(int8_t* dst, const int8_t* src, const int32_t* bias, const int32_t* alpha, int32_t mShiftBits, ssize_t minValue, ssize_t maxValue, int8_t* inputZeroPoint, int8_t* outputZeroPoint, ssize_t planeNumber, ssize_t biasNumber, ssize_t pack) {
#ifdef MNN_USE_SSE
    const uint8_t* srcPtr = (uint8_t*)src;
    uint8_t*       dstPtr = (uint8_t*)dst;
    int offset   = 128;
#else
    const int8_t*  srcPtr = src;
    int8_t*        dstPtr = dst;
    int offset   = 0;
#endif
    int intputZeroPointValue = *inputZeroPoint + offset;
    int outputZeroPointValue = *outputZeroPoint + offset;
    int d = mShiftBits - 1;

    for (int z = 0; z < biasNumber; ++z) {
        auto dstZ         = dstPtr + planeNumber * pack * z;
        const auto srcZ   = srcPtr + planeNumber * pack * z;
        std::vector<int32_t> biasZ(pack), alphaZ(pack);
        for (int i = 0; i < pack; ++i) {
            biasZ[i] = *(bias + pack * z + i);
            alphaZ[i] = *(alpha + pack * z + i);
        }
        for (int p = 0; p < planeNumber; ++p) {
            auto dstX       = dstZ + pack * p;
            const auto srcX = srcZ + pack * p;

            for (int i = 0; i < pack; ++i) {
                int32_t val = static_cast<int32_t>(srcX[i] - intputZeroPointValue) * alphaZ[i] + biasZ[i];

                int valOut  = roundf((val + (1<<d)) / (1 << mShiftBits)) + outputZeroPointValue;
                if (val < 0) {
                    valOut  = roundf((val - (1<<d)) / (1 << mShiftBits)) + outputZeroPointValue;
                }

                if (valOut > maxValue + offset) {
                    valOut = maxValue + offset;
                }
                if (valOut < minValue + offset) {
                    valOut = minValue + offset;
                }
                dstX[i] = valOut;
            }
        }
    }
}

#endif // #ifndef MNN_USE_NEON
#ifndef MNN_USE_SSE

void MNNInt8FunctionInit() {
    // do nothing
}
#endif // #ifndef MNN_USE_SSE

template<int EP, int LP, int HP>
static void _ArmBasicMNNPackC4ForMatMul_A(int8_t* destOrigin, int8_t const** sourceGroup, const int32_t* info, const int32_t* el) {
    int number = info[0];
    int eReal = info[1];
    int eOutsideStride = info[2] / sizeof(float);
    int eDest = EP;
    int offset = info[3];
    const int LUNIT = LP / sizeof(float);
    int realDstCount = info[4];
    for (int n=0; n<number; ++n) {
        int e = el[4 * n + 0];       // to fill
        int l = el[4 * n + 1];
        int eOffset = el[4 * n + 2]; // have filled
        int lOffset = el[4 * n + 3];
        int lC = lOffset / LP;
        int lR = lOffset % LP;
        int eC = eOffset / eDest;
        int eR = eOffset % eDest;
        int eS = eDest - eR;
//        printf("e=%d, eC=%d, lC=%d, eR=%d, lR=%d\n", e, eC, lC, eR, lR);
        bool lastBag = false;
        int eOutsideStride4LastBag = eOutsideStride;
        if (realDstCount % EP > 0) {
            int jobsE = realDstCount - eOffset - e;
            if (jobsE == 0 || (jobsE < (realDstCount % EP))) {
                lastBag = true;
            }
        }
        auto dest = (int32_t*)(destOrigin + lC * eDest * LP + lR + eC * info[2] + eR * LP);
        auto source = (int32_t*)sourceGroup[n];
        int lRemain = l / 4;
        int lR4 = lR / 4;
        int lS = LUNIT - lR4;
        
        if (lastBag && e + eR < EP) {
            int elast = ALIMAX(eR + e, realDstCount % EP);
            dest = (int32_t*)(destOrigin + lC * elast * LP + lR + eC * info[2] + eR * LP);
        }
        // Step for start
        int offsetLC = lC * LUNIT + lR / 4;

        if (lR4 > 0) {
            int step = ALIMIN(lS, lRemain);
            for (int x=0; x<step; ++x) {
                int eRemain = e;
                auto d = dest + x;
                auto s = source + x * eReal;
                if (eR > 0) {
                    int eStep = ALIMIN(eRemain, eS);
                    for (int yi=0; yi<eStep; ++yi) {
                        d[yi * LUNIT] = s[yi * offset];
                    }
                    eRemain-=eStep;
                    if (!lastBag ||eRemain >= EP) {
                        d += (eOutsideStride - eR * LUNIT);
                    } else {
                        int eFill = ALIMAX(eRemain, realDstCount % EP); // maybe padding>0
                        eOutsideStride4LastBag = eOutsideStride - ((offsetLC / LUNIT) * EP * LUNIT);
                        d += (eOutsideStride4LastBag - eR * LUNIT + (offsetLC / LUNIT) * eFill * LUNIT);
                    }
                    s += eS * offset;
                }
                while (eRemain > 0) {
                    int eStep = ALIMIN(eDest, eRemain);
                    for (int yi=0; yi<eStep; ++yi) {
                        d[yi * LUNIT] = s[yi * offset];
                    }
                    eRemain-=eStep;
                    if (!lastBag || eRemain >= EP) {
                        d+= eOutsideStride;
                    } else {
                        int eFill = ALIMAX(eRemain, realDstCount % EP); // maybe padding>0
                        eOutsideStride4LastBag = eOutsideStride - ((offsetLC / LUNIT) * EP * LUNIT);
                        d+= (eOutsideStride4LastBag + (offsetLC / LUNIT) * eFill * LUNIT);
                    }
                    s+= eStep * offset;
                }
                offsetLC++;
            }
            lRemain -= step;
            if (lastBag && e + eR < EP) {
                int eFill = ALIMAX(realDstCount % EP, e + eR);
                int nextLP = (eFill * LP - lR) / sizeof(int32_t);
                dest += nextLP;
            } else {
                int nextLP = (eDest * LP - lR) / sizeof(int32_t);
                dest += nextLP;
            }
            source += eReal * step;
        }
        
        while (lRemain > 0) {
            int step = ALIMIN(lRemain, LUNIT);
            for (int x=0; x<step; ++x) {
                int eRemain = e;
                auto d = dest + x;
                auto s = source + x * eReal;
                if (eR > 0) {
                    int eStep = ALIMIN(eRemain, eS);
                    for (int yi=0; yi<eStep; ++yi) {
                        d[yi * LUNIT] = s[yi * offset];
                    }
                    eRemain-=eStep;
                    if (!lastBag ||eRemain >= EP) {
                        d += (eOutsideStride - eR * LUNIT);
                    } else {
                        int eFill = ALIMAX(eRemain, realDstCount % EP); // maybe padding>0
                        eOutsideStride4LastBag = eOutsideStride - ((offsetLC / LUNIT) * EP * LUNIT);
                        d += (eOutsideStride4LastBag - eR * LUNIT + (offsetLC / LUNIT) * eFill * LUNIT);
                    }
                    s += eS * offset;
                }
                while (eRemain > 0) {
                    int eStep = ALIMIN(eDest, eRemain);
                    for (int yi=0; yi<eStep; ++yi) {
                        d[yi * LUNIT] = s[yi * offset];
                    }
                    eRemain-=eStep;
                    if (!lastBag || eRemain >= EP) {
                        d+= eOutsideStride;
                    } else {
                        int eFill = ALIMAX(eRemain, realDstCount % EP); // maybe padding>0
                        eOutsideStride4LastBag = eOutsideStride - ((offsetLC / LUNIT) * EP * LUNIT);
                        d+= (eOutsideStride4LastBag + (offsetLC / LUNIT) * eFill * LUNIT);
                    }
                    s+= eStep * offset;
                }
                offsetLC++;
            }
            
            lRemain -= step;
            if (lastBag && e + eR < EP) {
                int efill = ALIMAX(e + eR, realDstCount % EP);
                dest += efill * LUNIT;
            } else {
                dest += eDest * LUNIT;
            }
            source += eReal * step;
        }
    }
}

static void MNNGetGemmUnit(int* UNIT, int* SRC_UNIT, int* DST_XUNIT) {
    *UNIT = GEMM_INT8_UNIT;
    *SRC_UNIT = GEMM_INT8_SRC_UNIT;
    *DST_XUNIT = GEMM_INT8_DST_XUNIT;
}

static void MNNGetGemmUnitSdot(int* UNIT, int* SRC_UNIT, int* DST_XUNIT) {
    *UNIT = 8;
    *SRC_UNIT = 4;
    *DST_XUNIT = 12;
}

static void MNNGetGemmUnitI8mm(int* UNIT, int* SRC_UNIT, int* DST_XUNIT) {
    *UNIT = 8;
    *SRC_UNIT = 8;
    *DST_XUNIT = 10;
}

template<int EP, int HP>
static void _ArmBasicMNNPackC4ForMatMul_A_L4(int8_t* destOrigin, int8_t const** sourceGroup, const int32_t* info, const int32_t* el) {
    int number = info[0];
    int eReal = info[1];
    int eDest = EP;
    int offset = info[3];
    const int LP = 4;
    int eOutsideStride = info[2] / sizeof(float);
    int kernelCountUnit = eOutsideStride;
    int realDstCount = info[4];
    for (int n=0; n<number; ++n) {
        int e = el[4 * n + 0];
        int l = el[4 * n + 1];
        int eOffset = el[4 * n + 2];
        int lOffset = el[4 * n + 3];
        int eC = eOffset / EP;
        int eR = eOffset % EP;
        int eS = eDest - eR;
        bool lastBag = false;
        int eOutsideStride4LastBag = eOutsideStride;
        int eres = realDstCount - eOffset;
        if (realDstCount % EP > 0) {
            int jobsE = realDstCount - eOffset - e;
            if (jobsE == 0 || (jobsE < (realDstCount % EP))) {
                lastBag = true;
            }
        }
        auto dest = (int32_t*)(destOrigin + lOffset * eDest + eC * info[2] + eR * LP);
        auto source = (int32_t*)sourceGroup[n];
        int lRemain = l / sizeof(float);
        if (lastBag && e + eR < EP) {
            int elast = ALIMIN(ALIMAX(eR + e, realDstCount % EP), EP);
            dest = (int32_t*)(destOrigin + lOffset * elast + eC * info[2] + eR * LP);
        }
        int offsetLC = lOffset / 4;
        for (int x=0; x<lRemain; ++x) {
            int eRemain = e;
            auto d = dest;
            auto s = source;
            if (1 == offset) {
                if (eR > 0) {
                    int eStep = ALIMIN(eRemain, eS);
                    ::memcpy(d, s, eStep * sizeof(int32_t));
                    eRemain-=eStep;
                    if (!lastBag ||eRemain >= EP) {
                        d += (eOutsideStride - eR);
                    } else {
                        int eFill = ALIMAX(eRemain, realDstCount % EP); // maybe padding>0
                        eOutsideStride4LastBag = eOutsideStride - (EP * 4 * offsetLC / sizeof(float));
                        d += (eOutsideStride4LastBag - eR + offsetLC * eFill);
                    }
                    s += eS * offset;
                }
                while (eRemain > 0) {
                    int eStep = ALIMIN(eDest, eRemain);
                    ::memcpy(d, s, eStep * sizeof(int32_t));
                    eRemain-=eStep;
                    if (!lastBag || eRemain >= EP) {
                        d+= eOutsideStride;
                    } else {
                        int eFill = ALIMAX(eRemain, realDstCount % EP); // maybe padding>0
                        eOutsideStride4LastBag = eOutsideStride - (EP * 4 * offsetLC / sizeof(float));
                        d+= (eOutsideStride4LastBag + offsetLC * eFill);
                    }
                    s+= eStep * offset;
                }
            } else {
                if (eR > 0) {
                    int eStep = ALIMIN(eRemain, eS);
                    for (int yi=0; yi<eStep; ++yi) {
                        d[yi] = s[yi * offset];
                    }
                    eRemain-=eStep;
                    if (!lastBag ||eRemain >= EP) {
                        d += (eOutsideStride - eR);
                    } else {
                        int eFill = ALIMAX(eRemain, realDstCount % EP); // maybe padding>0
                        eOutsideStride4LastBag = eOutsideStride - (EP * 4 * offsetLC / sizeof(float));
                        d += (eOutsideStride4LastBag - eR + offsetLC * eFill);
                    }
                    s += eS * offset;
                }
                while (eRemain > 0) {
                    int eStep = ALIMIN(eDest, eRemain);
                    for (int yi=0; yi<eStep; ++yi) {
                        d[yi] = s[yi * offset];
                    }
                    eRemain-=eStep;
                    if (!lastBag || eRemain >= EP) {
                        d+= eOutsideStride;
                    } else {
                        int eFill = ALIMAX(eRemain, realDstCount % EP); // maybe padding>0
                        eOutsideStride4LastBag = eOutsideStride - (EP * 4 * offsetLC / sizeof(float));
                        d+= (eOutsideStride4LastBag + offsetLC * eFill);
                    }
                    s+= eStep * offset;
                }
            }
            source += eReal;
            if (lastBag && e + eR < EP ) { // eR=0;eR>0
                int efill = ALIMAX(e + eR, realDstCount % EP);
                dest += efill;
            } else {
                dest += eDest;
            }
            offsetLC++;
        }
    }
}

namespace MNN {

static CoreInt8Functions* gCoreFunc = nullptr;

void MNNCoreInt8FunctionInit() {
    /* CoreInt8Functions without sdot */
    gCoreFunc = new CoreInt8Functions;

    // MatMul
    gCoreFunc->Int8GemmKernel = MNNGemmInt8AddBiasScale_16x4_Unit;
    gCoreFunc->Int8GemmKernelFast = MNNGemmInt8AddBiasScale_16x4_Unit_FAST;
    gCoreFunc->MNNGetGemmUnit = MNNGetGemmUnit;
#ifdef MNN_LOW_MEMORY
    gCoreFunc->Int8GemmKernel_W4 = MNNGemmInt8AddBiasScale_16x4_w4_Unit;
#endif

    // Im2Col
    gCoreFunc->MNNPackC4Int8ForMatMul_A = _ArmBasicMNNPackC4ForMatMul_A<GEMM_INT8_DST_XUNIT, GEMM_INT8_SRC_UNIT, GEMM_INT8_UNIT>;
    // conv depthwise
    gCoreFunc->ConvDepthwiseLineInt8 = MNNLineDepthWiseInt8AddBiasScaleUnit;
    gCoreFunc->MNNFloat2Int8 = MNNFloat2Int8;
    gCoreFunc->MNNInt8ScaleToFloat = MNNInt8ScaleToFloat;

    // sparse
    gCoreFunc->MNNGetSparseQuantMatMulPackMode = MNNGetSparseQuantMatMulPackMode;
    gCoreFunc->MNNPackForSparseQuantMatMul_B = MNNPackForSparseQuantMatMul_B;
    gCoreFunc->MNNPackedSparseQuantMatMulEpx1 = MNNPackedSparseQuantMatMulEpx1;
    gCoreFunc->MNNPackedSparseQuantMatMulEpx4 = MNNPackedSparseQuantMatMulEpx4;
    gCoreFunc->MNNPackC4Int8ForMatMul_ASparse = _MNNPackC4Int8ForMatMul_ASparse;

    // pooling
    gCoreFunc->MNNAvgPoolInt8 = MNNAvgPoolInt8;
    gCoreFunc->MNNMaxPoolInt8 = MNNMaxPoolInt8;

    // ReluWithSlopeChannel
    gCoreFunc->MNNReluWithSlopeChannelInt8 = MNNReluWithSlopeChannelInt8;

#if defined(__aarch64__)
    auto core = MNNGetCoreFunctions();
    if (core->supportSDot) {
        // MatMul
        gCoreFunc->Int8GemmKernel = MNNGemmInt8AddBiasScale_ARMV82_Unit;
        gCoreFunc->Int8GemmKernelFast = MNNGemmInt8AddBiasScale_ARMV82_Unit;
        gCoreFunc->MNNGetGemmUnit = MNNGetGemmUnitSdot;
        // Im2Col
        gCoreFunc->MNNPackC4Int8ForMatMul_A = _ArmBasicMNNPackC4ForMatMul_A_L4<12, 8>;
        // ConvDepthwise
        gCoreFunc->ConvDepthwise3x3LineInt8_ARM82 = MNNLineDepthWiseInt8AddBiasScale_ARMV82_Unit3X3;
        core->MNNSumByAxisLForMatmul_A = MNNSumByAxisLForMatmul_A_ARM82;
#if defined(MNN_LOW_MEMORY)
    #ifdef MNN_USE_ARMV82
        gCoreFunc->DynamicQuanInput_ARM82 = DynamicQuanInput_ARM82;
        gCoreFunc->MNNGemmInt8AddBiasScale_Unit_FP16 = MNNGemmInt8AddBiasScale_ARMV82_Unit_FP16;
        gCoreFunc->MNNGemmInt8AddBiasScale_w4_Unit_FP16 = MNNGemmInt8AddBiasScale_ARMV82_w4_Unit_FP16;
        gCoreFunc->DynamicQuanInputAndReorder_ARM82 = DynamicQuanInputAndReorder_ARM82;
    #endif
        gCoreFunc->Int8GemmKernel_W4 = MNNGemmInt8AddBiasScale_ARMV82_w4_Unit;
#endif
    }
    if (core->supportI8mm) {
        // MatMul
        gCoreFunc->Int8GemmKernel = MNNGemmInt8AddBiasScale_ARMV86_Unit;
        gCoreFunc->Int8GemmKernelFast = MNNGemmInt8AddBiasScale_ARMV86_Unit;
        gCoreFunc->MNNGetGemmUnit = MNNGetGemmUnitI8mm;
        core->MNNSumByAxisLForMatmul_A = MNNSumByAxisLForMatmul_A_ARM86;
#if defined(MNN_LOW_MEMORY)
        gCoreFunc->Int8GemmKernel_W4 = MNNGemmInt8AddBiasScale_ARMV86_w4_Unit;
    #ifdef MNN_USE_ARMV82
        gCoreFunc->MNNGemmInt8AddBiasScale_Unit_FP16 = MNNGemmInt8AddBiasScale_ARMV86_Unit_FP16;
        gCoreFunc->MNNGemmInt8AddBiasScale_w4_Unit_FP16 = MNNGemmInt8AddBiasScale_ARMV86_w4_Unit_FP16;
    #endif
#endif
        // Im2Col
        gCoreFunc->MNNPackC4Int8ForMatMul_A = _ArmBasicMNNPackC4ForMatMul_A<10, 8, 8>;
    }
#endif
    MNNInt8FunctionInit();
}
CoreInt8Functions* MNNGetInt8CoreFunctions() {
    return gCoreFunc;
}
};
