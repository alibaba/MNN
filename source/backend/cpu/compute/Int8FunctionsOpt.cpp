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
#include "common/CommonCompute.hpp"
#include "CommonOptFunction.h"

#ifdef MNN_USE_NEON
#include <arm_neon.h>

extern "C" {
void MNNGemmInt8AddBiasScale_16x4_Unit(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad,
                                       const QuanPostTreatParameters* post, size_t realCount);
void MNNGemmInt8AddBiasScale_16x4_Unit_FAST(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad,
                                            const QuanPostTreatParameters* post, size_t realCount);
void MNNLineDepthWiseInt8AddBiasScaleUnit(int8_t* dst, const int8_t* src, const int8_t* weight, const QuanPostTreatParameters* parameters, size_t width,
                                          size_t src_w_step, size_t fw, size_t fh, size_t dilateX_step, size_t dilateY_step);
#if defined(__aarch64__) // aarch32 sdot workaround
void MNNGemmInt8AddBiasScale_ARMV82_Unit(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad,
                                         const QuanPostTreatParameters* post, size_t realDstCount);
void MNNGemmInt8AddBiasScale_ARMV86_Unit(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad,
                                         const QuanPostTreatParameters* post, size_t realDstCount);
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


static void MNNSparseQuantIm2col(int8_t* colAddr, const int8_t* inputOrigin, int8_t inputZeroPoint,
                          const MNN::ConvolutionCommon::Im2ColParameter* im2colParameter, const size_t* sparseQuantParam, size_t xIndexStart) {
    auto ih                     = im2colParameter->ih;
    auto iw                     = im2colParameter->iw;
    auto kh                     = im2colParameter->kernelY;
    auto kw                     = im2colParameter->kernelX;
    auto dilateX                = im2colParameter->dilateX;
    auto dilateY                = im2colParameter->dilateY;
    auto icDiv4                 = im2colParameter->icDiv4;
    auto srcZStep               = im2colParameter->srcZStep;
    auto srcYStep               = im2colParameter->srcYStep;
    auto destICStride           = im2colParameter->destICStride;
    auto packCUnit              = im2colParameter->packCUnit;

    size_t eSize= sparseQuantParam[0];
    size_t eP= sparseQuantParam[1];
    size_t l= sparseQuantParam[3];
    size_t ePx4 = eP << 2;
    const int col_buffer_size = l * eP * sizeof(int8_t);
    ::memset(colAddr, inputZeroPoint, col_buffer_size); // the padding process, since per-channel is removed, this is all right

    for (int i = 0; i < eSize; ++i) {
        int xIndex = (int)xIndexStart + i;
        int ox     = xIndex % im2colParameter->ow;
        int oy     = xIndex / im2colParameter->ow;

        int sx = ox * im2colParameter->strideX - im2colParameter->padX;
        int sy = oy * im2colParameter->strideY - im2colParameter->padY;

        int sfy = ALIMAX(0, (UP_DIV(-sy, im2colParameter->dilateY)));
        int efy = ALIMIN(kh, UP_DIV(ih - sy, im2colParameter->dilateY));
        int sfx = ALIMAX(0, (UP_DIV(-sx, im2colParameter->dilateX)));
        int efx = ALIMIN(kw, UP_DIV(iw - sx, im2colParameter->dilateX));
        int fyC = efy - sfy;
        int fxC = efx - sfx;

        auto inputOffset = inputOrigin + (sy + sfy * dilateY) * srcYStep + (sx + sfx * dilateX) * packCUnit; // offset in (c/4, ih, iw, 4),
        auto destBase = colAddr + (sfy * kw + sfx) * destICStride + i;
        for (int fy = 0; fy < fyC; ++fy) {
            for (int fx = 0; fx < fxC; ++fx) {
                auto inputK     = inputOffset + fy * dilateY * srcYStep + fx * dilateX * packCUnit;// origin data matrix offset inside kernel
                auto destWrite = destBase + (fy * kw + fx) * destICStride;
                int8_t* destWrite4[4] = {
                    destWrite,
                    destWrite + eP,
                    destWrite + 2 * eP,
                    destWrite + 3 * eP
                };
                for (int sz = 0; sz < icDiv4; ++sz) {
                    // for (int ic4 = 0; ic4 < packCUnit; ic4++) {
                    //     *destWrite = inputK[ic4];
                    //     destWrite += eP;
                    // }
                    int8_t c4[4];
                    memcpy(c4, inputK, sizeof(int32_t));
                    *(destWrite4[0]) = c4[0];
                    *(destWrite4[1]) = c4[1];
                    *(destWrite4[2]) = c4[2];
                    *(destWrite4[3]) = c4[3];

                    destWrite4[0]+= ePx4;
                    destWrite4[1]+= ePx4;
                    destWrite4[2]+= ePx4;
                    destWrite4[3]+= ePx4;
                    inputK += srcZStep;
                }
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
    const int bytes = (post->scale != nullptr ? 1 : 4);
    for (int dz = 0; dz < dst_depth_quad; ++dz) {
        const auto weight_dz = weight + dz * src_depth_quad * (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT);
        const auto bias_dz   = post->bias + dz * GEMM_INT8_UNIT;
        const float* scale_dz = nullptr;
        if (post->scale != nullptr) {
            scale_dz  = post->scale + dz * GEMM_INT8_UNIT;
        }
        auto dst_z           = dst + dz * dst_step;
        for (int w = 0; w < realCount; ++w) {
            const auto src_x   = src + w * GEMM_INT8_SRC_UNIT;
            auto dst_x         = dst_z + w * GEMM_INT8_UNIT * bytes;
            int32_t dstTemp[4] = {0, 0, 0, 0};

            for (int sz = 0; sz < src_depth_quad; ++sz) {
                const auto weight_sz = weight_dz + (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT) * sz;
                const auto src_z     = src_x + sz * GEMM_INT8_DST_XUNIT * GEMM_INT8_SRC_UNIT;

                for (int j = 0; j < GEMM_INT8_UNIT; ++j) {
                    const auto weight_j = weight_sz + j * GEMM_INT8_SRC_UNIT;
                    for (int i = 0; i < GEMM_INT8_SRC_UNIT; ++i) {
                        dstTemp[j] += (int32_t)src_z[i] * (int32_t)weight_j[i];
                    }
                }
            }

            for (int j = 0; j < GEMM_INT8_UNIT; ++j) {
                if (post->scale != nullptr) {
                    dst_x[j] = MNNInt32ToInt8(dstTemp[j], bias_dz[j], scale_dz[j], post->maxValue, post->minValue);
                } else {
                    ((float*)dst_x)[j] = (float)(dstTemp[j] + bias_dz[j]);
                }
            }
        }
    }
}

static void MNNGemmInt8AddBiasScale_16x4_Unit_FAST(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad, const QuanPostTreatParameters* post, size_t realCount) {
    return MNNGemmInt8AddBiasScale_16x4_Unit(dst, src, weight, src_depth_quad, dst_step, dst_depth_quad, post, realCount);
}

static void MNNLineDepthWiseInt8AddBiasScaleUnit(int8_t* dst, const int8_t* src, const int8_t* weight, const QuanPostTreatParameters* parameters,
                                          size_t width, size_t src_w_step, size_t fw, size_t fh, size_t dilateX_step,
                                          size_t dilateY_step) {
    auto bias_z = parameters->bias;
    auto scale_z = parameters->scale;
    int dx, fx, fy;
    for (dx = 0; dx < width; ++dx) {
        auto dst_x          = dst + dx * 4;
        int32_t dstInt32[4] = {0, 0, 0, 0};
        const auto src_z    = src + src_w_step * dx;
        for (fy = 0; fy < fh; ++fy) {
            const auto src_y    = src_z + fy * dilateY_step;
            const auto weight_y = weight + fy * fw * 4;
            for (fx = 0; fx < fw; ++fx) {
                const auto src_x    = src_y + fx * dilateX_step;
                const auto weight_x = weight_y + 4 * fx;
                for (int j = 0; j < GEMM_INT8_UNIT; ++j) {
                    dstInt32[j] += (int32_t)src_x[j] * (int32_t)weight_x[j];
                }
            }
        }

        for (int i = 0; i < GEMM_INT8_UNIT; ++i) {
            dst_x[i] = MNNInt32ToInt8(dstInt32[i], bias_z[i], scale_z[i], parameters->maxValue, parameters->minValue);
        }
    }
}
#endif

#ifndef MNN_USE_NEON
void MNNFloat2Int8(const float* src, int8_t* dst, size_t sizeQuad, const float* scalep, ssize_t minValue,
                   ssize_t maxValue, ssize_t zeroPoint) {
    for (int i = 0; i < sizeQuad; ++i) {
        for (int j=0; j<4; ++j) {
            int v = (int)roundf(src[4*i+j] * scalep[j]) + zeroPoint;
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

void MNNInt8ScaleToFloat(float* dst, const int8_t* src, const float* scale, size_t size, ssize_t zeroPoint) {
    for (int i = 0; i < size; ++i) {
        const auto srcStart = src + i * 4;
        auto dstStart       = dst + i * 4;
        for (int j = 0; j < 4; ++j) {
            dstStart[j] = static_cast<float>(srcStart[j] - zeroPoint) * scale[j];
        }
    }
}
#endif // #ifndef MNN_USE_NEON
#ifndef MNN_USE_SSE
void MNNInt8FunctionInit() {
    // do nothing
}
#endif // #ifndef MNN_USE_SSE

/* CPU without sdot */
// Assume GEMM_INT8_UNIT == 4 && GEMM_INT8_SRC_UNIT == 16
static void _fastIm2Col(int8_t* colAddr, const int8_t* inputOrigin, int32_t inputZeroPoint,
                        const MNN::ConvolutionCommon::Im2ColParameter* im2colParameter, size_t xIndexStart,
                        size_t realDstCount) {
    const int col_buffer_size = im2colParameter->kernelCountUnit * GEMM_INT8_SRC_UNIT * GEMM_INT8_DST_XUNIT * sizeof(int8_t);
    ::memset(colAddr, inputZeroPoint, col_buffer_size); // the padding process, since per-channel is removed, this is all right

    const int icDiv8   = im2colParameter->icDiv4 / 2;
    const int srcZStep = im2colParameter->iw * im2colParameter->ih * GEMM_INT8_UNIT;
    inputOrigin += xIndexStart * GEMM_INT8_UNIT;
    for (int i = 0; i < realDstCount; ++i) {
        auto colAddrI = colAddr + GEMM_INT8_SRC_UNIT * i;
        auto inputK   = inputOrigin + GEMM_INT8_UNIT * i;
        for (int sz = 0; sz < icDiv8; ++sz) {
            auto inputZ0           = inputK + srcZStep * (2 * sz + 0);
            auto inputZ1           = inputK + srcZStep * (2 * sz + 1);
            const int indexOutside = sz / 2;
            const int indexInsize  = sz % 2;

            auto dstK0         = colAddrI + (indexOutside * GEMM_INT8_DST_XUNIT * 2 + indexInsize) * (2 * GEMM_INT8_UNIT);
            auto dstK1         = dstK0 + GEMM_INT8_UNIT;
            *((int32_t*)dstK0) = *((int32_t*)inputZ0);
            *((int32_t*)dstK1) = *((int32_t*)inputZ1);
        }
    }
}

static void _im2colCommonZ1(int8_t* colAddr, const int8_t* inputOrigin, int32_t inputZeroPoint,
                            const MNN::ConvolutionCommon::Im2ColParameter* im2colParameter, size_t xIndexStart,
                            size_t realDstCount) {
    int col_buffer_size = im2colParameter->kernelCountUnit * GEMM_INT8_DST_XUNIT * GEMM_INT8_SRC_UNIT * sizeof(int8_t);
    ::memset(colAddr, inputZeroPoint, col_buffer_size); // the padding process, since per-channel is removed, this is all right

    auto ih                     = im2colParameter->ih;
    auto iw                     = im2colParameter->iw;
    auto kh                     = im2colParameter->kernelY;
    auto kw                     = im2colParameter->kernelX;
    auto dilateX                = im2colParameter->dilateX;
    auto dilateY                = im2colParameter->dilateY;
    auto srcYStep               = im2colParameter->srcYStep;
    constexpr int dstXStepInt32 = GEMM_INT8_SRC_UNIT * GEMM_INT8_DST_XUNIT / sizeof(int32_t);
    for (int i = 0; i < realDstCount; ++i) {
        int xIndex = (int)xIndexStart + i;
        int ox     = xIndex % im2colParameter->ow;
        int oy     = xIndex / im2colParameter->ow;

        int sx = ox * im2colParameter->strideX - im2colParameter->padX;
        int sy = oy * im2colParameter->strideY - im2colParameter->padY;

        int sfy = ALIMAX(0, (UP_DIV(-sy, im2colParameter->dilateY)));
        int efy = ALIMIN(kh, UP_DIV(ih - sy, im2colParameter->dilateY));
        int sfx = ALIMAX(0, (UP_DIV(-sx, im2colParameter->dilateX)));
        int efx = ALIMIN(kw, UP_DIV(iw - sx, im2colParameter->dilateX));
        int fyC = efy - sfy;
        int fxC = efx - sfx;

        auto colAddrI    = colAddr + GEMM_INT8_SRC_UNIT * i;

        auto inputOffset = inputOrigin + (sy + sfy * dilateY) * srcYStep + (sx + sfx * dilateX) * GEMM_INT8_UNIT;
        auto indexOffset = sfy * kw + sfx;
        for (int fy = 0; fy < fyC; ++fy) {
            for (int fx = 0; fx < fxC; ++fx) {
                auto inputK       = inputOffset + fy * dilateY * srcYStep + fx * dilateX * GEMM_INT8_UNIT;
                auto indexStart   = indexOffset + fy * kw + fx;
                auto indexInside  = indexStart % 4;
                auto indexOutside = indexStart / 4;
                auto dstK0        = (int32_t*)colAddrI + indexOutside * dstXStepInt32 + indexInside;
                dstK0[0]          = *((int32_t*)inputK);
            }
        }
    }
}

static void _im2colCommon(int8_t* colAddr, const int8_t* inputOrigin, int32_t inputZeroPoint,
                          const MNN::ConvolutionCommon::Im2ColParameter* im2colParameter, size_t xIndexStart,
                          size_t realDstCount) {
    const int col_buffer_size = im2colParameter->kernelCountUnit * GEMM_INT8_DST_XUNIT * GEMM_INT8_SRC_UNIT * sizeof(int8_t);
    ::memset(colAddr, inputZeroPoint, col_buffer_size); // the padding process, since per-channel is removed, this is all right

    auto ih                     = im2colParameter->ih;
    auto iw                     = im2colParameter->iw;
    auto kh                     = im2colParameter->kernelY;
    auto kw                     = im2colParameter->kernelX;
    auto dilateX                = im2colParameter->dilateX;
    auto dilateY                = im2colParameter->dilateY;
    auto icDiv4                 = im2colParameter->icDiv4;
    auto srcZStep               = im2colParameter->srcZStep;
    auto srcYStep               = im2colParameter->srcYStep;
    constexpr int dstXStepInt32 = GEMM_INT8_SRC_UNIT * GEMM_INT8_DST_XUNIT / sizeof(int32_t);
    for (int i = 0; i < realDstCount; ++i) {
        int xIndex = (int)xIndexStart + i;
        int ox     = xIndex % im2colParameter->ow;
        int oy     = xIndex / im2colParameter->ow;

        int sx = ox * im2colParameter->strideX - im2colParameter->padX;
        int sy = oy * im2colParameter->strideY - im2colParameter->padY;

        int sfy = ALIMAX(0, (UP_DIV(-sy, im2colParameter->dilateY)));
        int efy = ALIMIN(kh, UP_DIV(ih - sy, im2colParameter->dilateY));
        int sfx = ALIMAX(0, (UP_DIV(-sx, im2colParameter->dilateX)));
        int efx = ALIMIN(kw, UP_DIV(iw - sx, im2colParameter->dilateX));
        int fyC = efy - sfy;
        int fxC = efx - sfx;

        auto colAddrI    = colAddr + GEMM_INT8_SRC_UNIT * i;

        auto inputOffset = inputOrigin + (sy + sfy * dilateY) * srcYStep + (sx + sfx * dilateX) * GEMM_INT8_UNIT;
        auto indexOffset = (sfy * kw + sfx) * icDiv4;
        for (int fy = 0; fy < fyC; ++fy) {
            for (int fx = 0; fx < fxC; ++fx) {
                auto inputK     = inputOffset + fy * dilateY * srcYStep + fx * dilateX * GEMM_INT8_UNIT;
                auto indexStart = indexOffset + (fy * kw + fx) * icDiv4;
                for (int sz = 0; sz < icDiv4; ++sz) {
                    const int yIndex      = indexStart + sz;
                    const int ySubOutside = yIndex / GEMM_INT8_UNIT;
                    const int ySubInside  = yIndex % GEMM_INT8_UNIT;
                    auto dstK0            = (int32_t*)colAddrI + ySubOutside * dstXStepInt32 + ySubInside;
                    dstK0[0]              = *((int32_t*)inputK);
                    inputK += srcZStep;
                }
            }
        }
    }
}

static MNN::CoreInt8Functions::Im2ColFunc chooseIm2Col(const MNN::ConvolutionCommon::Im2ColParameter* im2colParam, size_t inputChannel) {
    bool fastIm2Col = im2colParam->kernelX == 1 && im2colParam->kernelY == 1 && im2colParam->icDiv4 % 2 == 0 &&
                      im2colParam->strideX == 1 && im2colParam->strideY == 1 && im2colParam->padX == 0 &&
                      im2colParam->padY == 0;
    int ih = im2colParam->ih, iw = im2colParam->iw;
    fastIm2Col &= (im2colParam->srcYStep == iw * GEMM_INT8_UNIT && im2colParam->srcZStep == ih * iw * GEMM_INT8_UNIT);
    if (fastIm2Col) {
        return _fastIm2Col;
    } else if (inputChannel <= 4) {
        return _im2colCommonZ1;
    } else {
        return _im2colCommon;
    }
}

static void MNNGetGemmUnit(int* UNIT, int* SRC_UNIT, int* DST_XUNIT) {
    *UNIT = GEMM_INT8_UNIT;
    *SRC_UNIT = GEMM_INT8_SRC_UNIT;
    *DST_XUNIT = GEMM_INT8_DST_XUNIT;
}
#undef GEMM_INT8_UNIT
#undef GEMM_INT8_SRC_UNIT
#undef GEMM_INT8_DST_XUNIT
/* End */

/* CPU with sdot */
#define GEMM_INT8_UNIT 4
#define GEMM_INT8_SRC_UNIT 4

#ifdef __aarch64__
#define GEMM_INT8_DST_XUNIT 12
#else
#define GEMM_INT8_DST_XUNIT 8
#endif

static void _im2colCommonSdot(int8_t* colAddr, const int8_t* src, int32_t inputZeroPoint,
                                const MNN::ConvolutionCommon::Im2ColParameter* im2colParameter, size_t xIndexStart,
                                size_t realDstCount) {
    const int colBufferSize = im2colParameter->kernelCountUnit * GEMM_INT8_DST_XUNIT * GEMM_INT8_SRC_UNIT * sizeof(int8_t);
    memset(colAddr, inputZeroPoint, colBufferSize);
    auto ih = im2colParameter->ih;
    auto iw = im2colParameter->iw;
    // auto oh = im2colParameter->oh;
    auto ow                     = im2colParameter->ow;
    auto kh                     = im2colParameter->kernelY;
    auto kw                     = im2colParameter->kernelX;
    auto dilateX                = im2colParameter->dilateX;
    auto dilateY                = im2colParameter->dilateY;
    auto icDiv4                 = im2colParameter->icDiv4;
    auto srcChannleStride       = im2colParameter->srcZStep;
    auto srcYStep               = im2colParameter->srcYStep;
    constexpr int dstXStepInt32 = GEMM_INT8_UNIT * GEMM_INT8_DST_XUNIT / sizeof(int32_t);

    for (int i = 0; i < realDstCount; ++i) {
        int xIndex = (int)xIndexStart + i;
        int ox     = xIndex % ow;
        int oy     = xIndex / ow;
        int sx     = ox * im2colParameter->strideX - im2colParameter->padX;
        int sy     = oy * im2colParameter->strideY - im2colParameter->padY;
        int sfy    = ALIMAX(0, (UP_DIV(-sy, im2colParameter->dilateY)));
        int efy    = ALIMIN(kh, UP_DIV(ih - sy, im2colParameter->dilateY));
        int sfx    = ALIMAX(0, (UP_DIV(-sx, im2colParameter->dilateX)));
        int efx    = ALIMIN(kw, UP_DIV(iw - sx, im2colParameter->dilateX));
        int fyC    = efy - sfy;
        int fxC    = efx - sfx;

        auto colAddrI    = colAddr + GEMM_INT8_UNIT * i;
        auto inputOffset = src + (sy + sfy * dilateY) * srcYStep + (sx + sfx * dilateX) * GEMM_INT8_UNIT;
        auto indexOffset = (sfy * kw + sfx) * icDiv4;

        for (int fy = 0; fy < fyC; ++fy) {
            for (int fx = 0; fx < fxC; ++fx) {
                auto inputK     = inputOffset + fy * dilateY * srcYStep + fx * dilateX * GEMM_INT8_UNIT;
                auto indexStart = (indexOffset + (fy * kw + fx) * icDiv4) * dstXStepInt32;
                for (int sz = 0; sz < icDiv4; ++sz) {
                    auto dstK0 = (int32_t*)colAddrI + indexStart + sz * dstXStepInt32;
                    dstK0[0]   = *((int32_t*)inputK);
                    inputK += srcChannleStride;
                }
            }
        }
    }
}

static void _fastIm2ColSdot(int8_t* colAddr, const int8_t* inputOrigin, int32_t inputZeroPoint,
                              const MNN::ConvolutionCommon::Im2ColParameter* im2colParameter, size_t xIndexStart,
                              size_t realDstCount) {
    const int col_buffer_size = im2colParameter->kernelCountUnit * GEMM_INT8_DST_XUNIT * GEMM_INT8_SRC_UNIT * sizeof(int8_t);
    ::memset(colAddr, inputZeroPoint, col_buffer_size);
    const int icDiv4    = im2colParameter->icDiv4;
    const int srcZStep = im2colParameter->iw * im2colParameter->ih * GEMM_INT8_UNIT;
    inputOrigin += xIndexStart * GEMM_INT8_UNIT;
    for (int i = 0; i < realDstCount; ++i) {
        auto colAddrI = colAddr + GEMM_INT8_UNIT * i;
        auto inputK   = inputOrigin + GEMM_INT8_UNIT * i;
        for (int sz = 0; sz < icDiv4; ++sz) {
            auto inputZ0       = inputK + srcZStep * sz;
            auto dstK0         = colAddrI + sz * GEMM_INT8_UNIT * GEMM_INT8_DST_XUNIT;
            *((int32_t*)dstK0) = *((int32_t*)inputZ0);
        }
    }
}

static MNN::CoreInt8Functions::Im2ColFunc chooseIm2ColSdot(const MNN::ConvolutionCommon::Im2ColParameter* im2colParam, size_t inputChannel) {
    bool fastIm2Col = im2colParam->kernelX == 1 && im2colParam->kernelY == 1 && im2colParam->icDiv4 % 2 == 0 &&
                      im2colParam->strideX == 1 && im2colParam->strideY == 1 && im2colParam->padX == 0 &&
                      im2colParam->padY == 0;
    int ih = im2colParam->ih, iw = im2colParam->iw;
    fastIm2Col &= (im2colParam->srcYStep == iw * GEMM_INT8_UNIT && im2colParam->srcZStep == ih * iw * GEMM_INT8_UNIT);
    if (fastIm2Col) {
        return _fastIm2ColSdot;
    } else {
        return _im2colCommonSdot;
    }
}

static void MNNGetGemmUnitSdot(int* UNIT, int* SRC_UNIT, int* DST_XUNIT) {
    *UNIT = GEMM_INT8_UNIT;
    *SRC_UNIT = GEMM_INT8_SRC_UNIT;
    *DST_XUNIT = GEMM_INT8_DST_XUNIT;
}

#undef GEMM_INT8_UNIT
#undef GEMM_INT8_SRC_UNIT
#undef GEMM_INT8_DST_XUNIT
/* End */


/* CPU with i8mm */
#define GEMM_INT8_UNIT 4
#define GEMM_INT8_SRC_UNIT 8
#define GEMM_INT8_DST_XUNIT 20

// icDiv4 % 2 == 0 will call this function
static void _im2colCommonI8mm(int8_t* colAddr, const int8_t* src, int32_t inputZeroPoint,
                              const MNN::ConvolutionCommon::Im2ColParameter* im2colParameter, size_t xIndexStart,
                              size_t realDstCount) {
    const int col_buffer_size = im2colParameter->kernelCountUnit * GEMM_INT8_DST_XUNIT * GEMM_INT8_SRC_UNIT * sizeof(int8_t);
    ::memset(colAddr, inputZeroPoint, col_buffer_size); // the padding process, since per-channel is removed, this is all right
    auto ih                     = im2colParameter->ih;
    auto iw                     = im2colParameter->iw;
    auto kh                     = im2colParameter->kernelY;
    auto kw                     = im2colParameter->kernelX;
    auto dilateX                = im2colParameter->dilateX;
    auto dilateY                = im2colParameter->dilateY;
    auto icDiv4                 = im2colParameter->icDiv4;
    auto srcZStep               = im2colParameter->srcZStep;
    auto srcYStep               = im2colParameter->srcYStep;
    constexpr int dstXStepInt32 = GEMM_INT8_SRC_UNIT * GEMM_INT8_DST_XUNIT / sizeof(int32_t);
    constexpr int SRC_DIV_UNIT = GEMM_INT8_SRC_UNIT / GEMM_INT8_UNIT; // 2
    auto icDiv8 = icDiv4 / 2;
    for (int i = 0; i < realDstCount; ++i) {
        int xIndex = (int)xIndexStart + i;
        int ox     = xIndex % im2colParameter->ow;
        int oy     = xIndex / im2colParameter->ow;
        int sx = ox * im2colParameter->strideX - im2colParameter->padX;
        int sy = oy * im2colParameter->strideY - im2colParameter->padY;
        int sfy = ALIMAX(0, (UP_DIV(-sy, im2colParameter->dilateY)));
        int efy = ALIMIN(kh, UP_DIV(ih - sy, im2colParameter->dilateY));
        int sfx = ALIMAX(0, (UP_DIV(-sx, im2colParameter->dilateX)));
        int efx = ALIMIN(kw, UP_DIV(iw - sx, im2colParameter->dilateX));
        int fyC = efy - sfy;
        int fxC = efx - sfx;
        auto colAddrI    = colAddr + GEMM_INT8_SRC_UNIT * i;
        auto inputOffset = src + (sy + sfy * dilateY) * srcYStep + (sx + sfx * dilateX) * GEMM_INT8_UNIT;
        auto indexOffset = (sfy * kw + sfx) * icDiv8;
        for (int fy = 0; fy < fyC; ++fy) {
            for (int fx = 0; fx < fxC; ++fx) {
                auto inputK     = inputOffset + fy * dilateY * srcYStep + fx * dilateX * GEMM_INT8_UNIT;
                auto indexStart = indexOffset + (fy * kw + fx) * icDiv8;
                for (int sz = 0; sz < icDiv8; ++sz) {
                    const int yIndex      = indexStart + sz;
                    auto dstK0            = (int32_t*)colAddrI + yIndex * dstXStepInt32;
                    dstK0[0]              = *((int32_t*)inputK);
                    dstK0[1]              = *((int32_t*)(inputK + srcZStep));
                    inputK += 2 * srcZStep;
                }
            }
        }
    }
}

static void _slowIm2ColI8mm(int8_t* colAddr, const int8_t* src, int32_t inputZeroPoint,
                            const MNN::ConvolutionCommon::Im2ColParameter* im2colParameter, size_t xIndexStart,
                            size_t realDstCount) {
    const int col_buffer_size = im2colParameter->kernelCountUnit * GEMM_INT8_DST_XUNIT * GEMM_INT8_SRC_UNIT * sizeof(int8_t);
    ::memset(colAddr, inputZeroPoint, col_buffer_size); // the padding process, since per-channel is removed, this is all right
    auto ih                     = im2colParameter->ih;
    auto iw                     = im2colParameter->iw;
    auto kh                     = im2colParameter->kernelY;
    auto kw                     = im2colParameter->kernelX;
    auto dilateX                = im2colParameter->dilateX;
    auto dilateY                = im2colParameter->dilateY;
    auto icDiv4                 = im2colParameter->icDiv4;
    auto srcZStep               = im2colParameter->srcZStep;
    auto srcYStep               = im2colParameter->srcYStep;
    constexpr int dstXStepInt32 = GEMM_INT8_SRC_UNIT * GEMM_INT8_DST_XUNIT / sizeof(int32_t);
    constexpr int SRC_DIV_UNIT = GEMM_INT8_SRC_UNIT / GEMM_INT8_UNIT;

    for (int i = 0; i < realDstCount; ++i) {
        int xIndex = (int)xIndexStart + i;
        int ox     = xIndex % im2colParameter->ow;
        int oy     = xIndex / im2colParameter->ow;
        int sx = ox * im2colParameter->strideX - im2colParameter->padX;
        int sy = oy * im2colParameter->strideY - im2colParameter->padY;
        int sfy = ALIMAX(0, (UP_DIV(-sy, im2colParameter->dilateY)));
        int efy = ALIMIN(kh, UP_DIV(ih - sy, im2colParameter->dilateY));
        int sfx = ALIMAX(0, (UP_DIV(-sx, im2colParameter->dilateX)));
        int efx = ALIMIN(kw, UP_DIV(iw - sx, im2colParameter->dilateX));
        int fyC = efy - sfy;
        int fxC = efx - sfx;
        auto colAddrI    = colAddr + GEMM_INT8_SRC_UNIT * i;
        auto inputOffset = src + (sy + sfy * dilateY) * srcYStep + (sx + sfx * dilateX) * GEMM_INT8_UNIT;
        auto indexOffset = (sfy * kw + sfx) * icDiv4;
        for (int fy = 0; fy < fyC; ++fy) {
            for (int fx = 0; fx < fxC; ++fx) {
                auto inputK     = inputOffset + fy * dilateY * srcYStep + fx * dilateX * GEMM_INT8_UNIT;
                auto indexStart = indexOffset + (fy * kw + fx) * icDiv4;
                for (int sz = 0; sz < icDiv4; ++sz) {
                    const int yIndex      = indexStart + sz;
                    const int ySubOutside = yIndex / SRC_DIV_UNIT;
                    const int ySubInside  = yIndex % SRC_DIV_UNIT;
                    auto dstK0            = (int32_t*)colAddrI + ySubOutside * dstXStepInt32 + ySubInside;
                    dstK0[0]              = *((int32_t*)inputK);
                    inputK += srcZStep;
                }
            }
        }
    }
}

static void _fastIm2ColI8mm(int8_t* colAddr, const int8_t* inputOrigin, int32_t inputZeroPoint,
                              const MNN::ConvolutionCommon::Im2ColParameter* im2colParameter, size_t xIndexStart,
                              size_t realDstCount) {
    const int col_buffer_size = im2colParameter->kernelCountUnit * GEMM_INT8_DST_XUNIT * GEMM_INT8_SRC_UNIT * sizeof(int8_t);
    ::memset(colAddr, inputZeroPoint, col_buffer_size);
    const int icDiv4    = im2colParameter->icDiv4;
    const int srcZStep = im2colParameter->iw * im2colParameter->ih * GEMM_INT8_UNIT;
    inputOrigin += xIndexStart * GEMM_INT8_UNIT;
    for (int i = 0; i < realDstCount; ++i) {
        auto colAddrI = colAddr + GEMM_INT8_UNIT * i;
        auto inputK   = inputOrigin + GEMM_INT8_UNIT * i;
        for (int sz = 0; sz < icDiv4; ++sz) {
            auto inputZ0       = inputK + srcZStep * sz;
            auto dstK0         = colAddrI + sz * GEMM_INT8_UNIT * GEMM_INT8_DST_XUNIT;
            *((int32_t*)dstK0) = *((int32_t*)inputZ0);
        }
    }
}

static MNN::CoreInt8Functions::Im2ColFunc chooseIm2ColI8mm(const MNN::ConvolutionCommon::Im2ColParameter* im2colParam, size_t inputChannel) {
    bool fastIm2Col = im2colParam->kernelX == 1 && im2colParam->kernelY == 1 && im2colParam->icDiv4 % 2 == 0 &&
                      im2colParam->strideX == 1 && im2colParam->strideY == 1 && im2colParam->padX == 0 &&
                      im2colParam->padY == 0;
    int ih = im2colParam->ih, iw = im2colParam->iw;
    fastIm2Col &= (im2colParam->srcYStep == iw * GEMM_INT8_UNIT && im2colParam->srcZStep == ih * iw * GEMM_INT8_UNIT);
    if (fastIm2Col) {
        return _fastIm2ColI8mm;
    } else {
        if (im2colParam->icDiv4 % 2) {
            return _slowIm2ColI8mm;
        } else {
            return _im2colCommonI8mm;
        }
    }
}

static void MNNGetGemmUnitI8mm(int* UNIT, int* SRC_UNIT, int* DST_XUNIT) {
    *UNIT = GEMM_INT8_UNIT;
    *SRC_UNIT = GEMM_INT8_SRC_UNIT;
    *DST_XUNIT = GEMM_INT8_DST_XUNIT;
}
#undef GEMM_INT8_UNIT
#undef GEMM_INT8_SRC_UNIT
#undef GEMM_INT8_DST_XUNIT
/* End */

namespace MNN {

static CoreInt8Functions* gCoreFunc = nullptr;

void MNNCoreInt8FunctionInit() {
    /* CoreInt8Functions without sdot */
    gCoreFunc = new CoreInt8Functions;

    // MatMul
    gCoreFunc->Int8GemmKernel = MNNGemmInt8AddBiasScale_16x4_Unit;
    gCoreFunc->Int8GemmKernelFast = MNNGemmInt8AddBiasScale_16x4_Unit_FAST;
    gCoreFunc->MNNGetGemmUnit = MNNGetGemmUnit;
    // Im2Col
    gCoreFunc->chooseIm2Col = chooseIm2Col;
    // conv depthwise
    gCoreFunc->ConvDepthwiseLineInt8 = MNNLineDepthWiseInt8AddBiasScaleUnit;
    gCoreFunc->MNNFloat2Int8 = MNNFloat2Int8;
    gCoreFunc->MNNInt8ScaleToFloat = MNNInt8ScaleToFloat;

    // sparse
    gCoreFunc->MNNGetSparseQuantMatMulPackMode = MNNGetSparseQuantMatMulPackMode;
    gCoreFunc->MNNPackForSparseQuantMatMul_B = MNNPackForSparseQuantMatMul_B;
    gCoreFunc->MNNPackedSparseQuantMatMulEpx1 = MNNPackedSparseQuantMatMulEpx1;
    gCoreFunc->MNNPackedSparseQuantMatMulEpx4 = MNNPackedSparseQuantMatMulEpx4;
    gCoreFunc->MNNSparseQuantIm2col = MNNSparseQuantIm2col;

#if defined(__aarch64__)
    auto core = MNNGetCoreFunctions();
    if (core->supportSDot) {
        // MatMul
        gCoreFunc->Int8GemmKernel = MNNGemmInt8AddBiasScale_ARMV82_Unit;
        gCoreFunc->Int8GemmKernelFast = MNNGemmInt8AddBiasScale_ARMV82_Unit;
        gCoreFunc->MNNGetGemmUnit = MNNGetGemmUnitSdot;
        // Im2Col
        gCoreFunc->chooseIm2Col = chooseIm2ColSdot;
    }
    if (core->supportI8mm) {
        // MatMul
        gCoreFunc->Int8GemmKernel = MNNGemmInt8AddBiasScale_ARMV86_Unit;
        gCoreFunc->Int8GemmKernelFast = MNNGemmInt8AddBiasScale_ARMV86_Unit;
        gCoreFunc->MNNGetGemmUnit = MNNGetGemmUnitI8mm;
        // Im2Col
        gCoreFunc->chooseIm2Col = chooseIm2ColI8mm;
    }
#endif
    MNNInt8FunctionInit();
}
CoreInt8Functions* MNNGetInt8CoreFunctions() {
    return gCoreFunc;
}
};
