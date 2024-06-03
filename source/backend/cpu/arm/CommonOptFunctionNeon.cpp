#include "core/Macro.h"
#include "../compute/CommonOptFunction.h"
#ifdef MNN_USE_NEON
#include <arm_neon.h>
#include "./FunctionSummary.hpp"
#include "core/MemoryFormater.h"
extern "C" {
void MNNTranspose32Bit4x4(int32_t* dstO, const int32_t* srcO, int32_t* dim);
void MNNTranspose16Bit8x8(int16_t* dstO, const int16_t* srcO, int32_t* dim);
}
void MNNTranspose32Bit(int32_t* dstO, const int32_t* srcO, int32_t* dim) {
    int w = dim[0];
    int h = dim[1];
    auto wC4 = w / 4;
    auto hC4 = h / 4;
    int srcStride = dim[2];
    int dstStride = dim[3];
    if (wC4 > 0 && hC4 > 0) {
        MNNTranspose32Bit4x4(dstO, srcO, dim);
    }
    // Down
    for (int i=hC4 * 4; i<h; ++i) {
        auto si = srcO + i;
        auto di = dstO + i * dstStride;
        for (int j=0; j<w; ++j) {
            auto sj = si + j * srcStride;
            auto dj = di + j;
            *dj = *sj;
        }
    }
    // Right
    for (int i=0; i<hC4 * 4; ++i) {
        auto si = srcO + i;
        auto di = dstO + i * dstStride;
        for (int j=wC4 * 4; j<w; ++j) {
            auto sj = si + j * srcStride;
            auto dj = di + j;
            *dj = *sj;
        }
    }
}

void MNNTranspose16Bit(int16_t* dstO, const int16_t* srcO, int32_t* dim) {
    int w = dim[0];
    int h = dim[1];
    auto wC8 = w / 8;
    auto hC8 = h / 8;
    int srcStride = dim[2];
    int dstStride = dim[3];
    if (wC8 > 0 && hC8 > 0) {
        MNNTranspose16Bit8x8(dstO, srcO, dim);
    }

    // Down
    for (int i = hC8 * 8; i < h; ++i) {
        auto si = srcO + i;
        auto di = dstO + i * dstStride;
        for (int j = 0; j < w; ++j) {
            auto sj = si + j * srcStride;
            auto dj = di + j;
            *dj = *sj;
        }
    }
    // Right
    for (int i = 0; i < hC8 * 8; ++i) {
        auto si = srcO + i;
        auto di = dstO + i * dstStride;
        for (int j = wC8 * 8; j < w; ++j) {
            auto sj = si + j * srcStride;
            auto dj = di + j;
            *dj = *sj;
        }
    }
}

#ifndef MNN_USE_NEON

void MNNPackedSparseMatMulEpx1(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter, const float* postParameters, const float* bias, unsigned int* NNZMap, int* dataOffsetMap) {

    auto eP = parameter[0] / sizeof(float);
    MNN_ASSERT((eP & 0x03) == 0); // In sparse calculate, eP should be evenly divided by 4
    auto h = parameter[2];
    auto l = parameter[1];
    auto cStride = parameter[3] / sizeof(float);
    auto hRemain = parameter[4];
    auto bExtraStride = parameter[5] / sizeof(float);
    auto bStride = bExtraStride + l * 4;
    auto aStride = eP * l; // sizeof(float);
    auto hC4 = UP_DIV(h, 4);
    float minValue = -std::numeric_limits<float>().max();
    float maxValue = std::numeric_limits<float>().max();
    if (nullptr != postParameters) {
        minValue = postParameters[2];
        maxValue = postParameters[3];
    }
    const float32x4_t vmin = vld1q_dup_f32(&minValue);
    const float32x4_t vmax = vld1q_dup_f32(&maxValue);

    // MNN_PRINT("NEON MNNPackedSparseMatMul eP:%lu, eSize:%lu, l:%lu, h:%lu, cStride:%lu, aStride:%lu\n", eP, eSize, l, h, cStride, aStride);

    const float* a = A;
    size_t ie = 0;
    for (ie = 0; ie + eP <= eSize; ie += eP) {
        const int* dataOffset = dataOffsetMap;
        const int diff = *dataOffset++;
        a += diff;
        const float* w = B;
        float* blockC = C + (ie << 2);
        const unsigned int* nnz = NNZMap;
        for (auto ih = 0; ih < h; ih++) {
            auto ihPack = ih >> 2;
            auto ihSubIndex = ih & 0x03;
            auto c = blockC + ihPack * cStride + ihSubIndex;
            const float initValue = nullptr != bias ? bias[ih] : 0;
            float32x4_t vacc0123 = vld1q_dup_f32(&initValue);
            float32x4_t vacc4567 = vacc0123;
            float32x4_t vacc89AB = vacc0123;
            float32x4_t vaccCDEF = vacc0123;
            const unsigned int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {

                const int diff = *dataOffset++;
                const float32x4_t va0123 = vld1q_f32(a);
                const float32x4_t va4567 = vld1q_f32(a + 4);
                const float32x4_t va89AB = vld1q_f32(a + 8);
                const float32x4_t vaCDEF = vld1q_f32(a + 12);
                // __builtin_prefetch(a + aStride);

                float32x4_t w4 = vld1q_dup_f32(w);
                // MNN_PRINT("16-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0-15]:", ie, a - A, w - B, c - C, *w);
                // formatMatrix(a, {16});
                // MNN_PRINT("\n");
                w++;
                a = a + diff;

                vacc0123 = vfmaq_f32(vacc0123, va0123, w4);
                vacc4567 = vfmaq_f32(vacc4567, va4567, w4);
                vacc89AB = vfmaq_f32(vacc89AB, va89AB, w4);
                vaccCDEF = vfmaq_f32(vaccCDEF, vaCDEF, w4);

            }
            vacc0123 = vminq_f32(vacc0123, vmax);
            vacc4567 = vminq_f32(vacc4567, vmax);
            vacc89AB = vminq_f32(vacc89AB, vmax);
            vaccCDEF = vminq_f32(vaccCDEF, vmax);
            vacc0123 = vmaxq_f32(vacc0123, vmin);
            vacc4567 = vmaxq_f32(vacc4567, vmin);
            vacc89AB = vmaxq_f32(vacc89AB, vmin);
            vaccCDEF = vmaxq_f32(vaccCDEF, vmin);

            // how to store faster: st4 / transpose /
            vst1q_lane_f32(c, vacc0123, 0);
            vst1q_lane_f32(c + 4, vacc0123, 1);
            vst1q_lane_f32(c + 4 * 2, vacc0123, 2);
            vst1q_lane_f32(c + 4 * 3, vacc0123, 3);
            vst1q_lane_f32(c+  4 * 4, vacc4567, 0);
            vst1q_lane_f32(c + 4 * 5, vacc4567, 1);
            vst1q_lane_f32(c + 4 * 6, vacc4567, 2);
            vst1q_lane_f32(c + 4 * 7, vacc4567, 3);
            vst1q_lane_f32(c + 4 * 8, vacc89AB, 0);
            vst1q_lane_f32(c + 4 * 9, vacc89AB, 1);
            vst1q_lane_f32(c + 4 * 10, vacc89AB, 2);
            vst1q_lane_f32(c + 4 * 11, vacc89AB, 3);
            vst1q_lane_f32(c + 4 * 12, vaccCDEF, 0);
            vst1q_lane_f32(c + 4 * 13, vaccCDEF, 1);
            vst1q_lane_f32(c + 4 * 14, vaccCDEF, 2);
            vst1q_lane_f32(c + 4 * 15, vaccCDEF, 3);
        }
        a += aStride;
    }
    // const float* blockA = A + ie * l;
    if (eSize & 0x08) {
        const int* dataOffset = dataOffsetMap;
        const int diff = *dataOffset++;
        // a = blockA + diff;
        a += diff;
        const float* w = B;
        float* blockC = C + (ie << 2);
        const unsigned int* nnz = NNZMap;
        for (auto ih = 0; ih < h; ih++) {
            auto ihPack = ih >> 2;
            auto ihSubIndex = ih & 0x03;
            auto c = blockC + ihPack * cStride + ihSubIndex;
            const float initValue = nullptr != bias ? bias[ih] : 0;
            float32x4_t vacc0123 = vld1q_dup_f32(&initValue);
            float32x4_t vacc4567 = vacc0123;

            const unsigned int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {
                const int diff = *dataOffset++;
                const float32x4_t va0123 = vld1q_f32(a);
                const float32x4_t va4567 = vld1q_f32(a + 4);
                // __builtin_prefetch(a + aStride);

                float32x4_t w4 = vld1q_dup_f32(w);
                // MNN_PRINT("8-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0-7]:", ie, a - A, w - B, c - C, *w);
                // formatMatrix(a, {8});
                // MNN_PRINT("\n");
                w++;
                a = a + diff;
                vacc0123 = vfmaq_f32(vacc0123, va0123, w4);
                vacc4567 = vfmaq_f32(vacc4567, va4567, w4);
            }
            vacc0123 = vminq_f32(vacc0123, vmax);
            vacc4567 = vminq_f32(vacc4567, vmax);
            vacc0123 = vmaxq_f32(vacc0123, vmin);
            vacc4567 = vmaxq_f32(vacc4567, vmin);

            // how to store faster: st4 / transpose /
            vst1q_lane_f32(c, vacc0123, 0);
            vst1q_lane_f32(c + 4, vacc0123, 1);
            vst1q_lane_f32(c + 4 * 2, vacc0123, 2);
            vst1q_lane_f32(c + 4 * 3, vacc0123, 3);
            vst1q_lane_f32(c + 4 * 4, vacc4567, 0);
            vst1q_lane_f32(c + 4 * 5, vacc4567, 1);
            vst1q_lane_f32(c + 4 * 6, vacc4567, 2);
            vst1q_lane_f32(c + 4 * 7, vacc4567, 3);
        }
        ie += 8;
        a += 8;
    }

    if (eSize & 0x04) {
        const int* dataOffset = dataOffsetMap;
        const int diff = *dataOffset++;
        // const float* a = blockA + diff;
        a += diff;
        const float* w = B;
        float* blockC = C + (ie << 2);
        const unsigned int* nnz = NNZMap;
        for (auto ih = 0; ih < h; ih++) {
            auto ihPack = ih >> 2;
            auto ihSubIndex = ih & 0x03;
            auto c = blockC + ihPack * cStride + ihSubIndex;
            const float initValue = nullptr != bias ? bias[ih] : 0;
            float32x4_t vacc0123 = vld1q_dup_f32(&initValue);

            const unsigned int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {
                const int diff = *dataOffset++;
                const float32x4_t va0123 = vld1q_f32(a);
                // __builtin_prefetch(a + aStride);

                float32x4_t w4 = vld1q_dup_f32(w);
                // MNN_PRINT("4-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0-3]:", ie, a - A, w - B, c - C, *w);
                // formatMatrix(a, {4});
                // MNN_PRINT("\n");
                w++;
                a = a + diff;
                vacc0123 = vfmaq_f32(vacc0123, va0123, w4);
            }
            vacc0123 = vminq_f32(vacc0123, vmax);
            vacc0123 = vmaxq_f32(vacc0123, vmin);

            // how to store faster: st4 / transpose /
            vst1q_lane_f32(c, vacc0123, 0);
            vst1q_lane_f32(c + 4, vacc0123, 1);
            vst1q_lane_f32(c + 4 * 2, vacc0123, 2);
            vst1q_lane_f32(c + 4 * 3, vacc0123, 3);
        }
        ie += 4;
        a += 4;
    }
    if (eSize & 0x02) {
        const int* dataOffset = dataOffsetMap;
        const int diff = *dataOffset++;
        // const float* a = blockA + diff;
        a += diff;
        const float* w = B;
        float* blockC = C + (ie << 2);
        const unsigned int* nnz = NNZMap;
        for (auto ih = 0; ih < h; ih++) {
            auto ihPack = ih >> 2;
            auto ihSubIndex = ih & 0x03;
            auto c = blockC + ihPack * cStride + ihSubIndex;
            const float initValue = nullptr != bias ? bias[ih] : 0;
            float32x2_t vacc01 = vld1_dup_f32(&initValue);

            const unsigned int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {
                const int diff = *dataOffset++;
                const float32x2_t va01 = vld1_f32(a);
                // __builtin_prefetch(a + aStride);

                float32x2_t w2 = vld1_dup_f32(w);
                // MNN_PRINT("2-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0-1]:", ie, a - A, w - B, c - C, *w);
                // formatMatrix(a, {2});
                // MNN_PRINT("\n");
                w++;
                a = a + diff;
                vacc01 = vfma_f32(vacc01, va01, w2);
            }
            vacc01 = vmin_f32(vacc01, vget_low_f32(vmax));
            vacc01 = vmax_f32(vacc01, vget_low_f32(vmin));
            // how to store faster: st4 / transpose /
            vst1_lane_f32(c, vacc01, 0);
            vst1_lane_f32(c + 4, vacc01, 1);
        }
        ie += 2;
        a += 2;
    }
    if (eSize & 0x01) {
        const int* dataOffset = dataOffsetMap;
        const int diff = *dataOffset++;
        // const float* a = blockA + diff;
        a += diff;
        const float* w = B;
        float* blockC = C + (ie << 2);
        const unsigned int* nnz = NNZMap;
        for (auto ih = 0; ih < h; ih++) {
            auto ihPack = ih >> 2;
            auto ihSubIndex = ih & 0x03;
            auto c = blockC + ihPack * cStride + ihSubIndex;
            const float initValue = nullptr != bias ? bias[ih] : 0;
            float acc0 = initValue;

            const unsigned int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {
                const int diff = *dataOffset++;
                const float a0 = a[0];
                const float oneW = *w++;
                // MNN_PRINT("1-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0]:", ie, a - A, w - B - 1, c - C, oneW);
                // formatMatrix(a, {1});
                // MNN_PRINT("\n");
                a = a + diff;
                acc0 += a0 * oneW;
            }
            acc0  = std::max(std::min(maxValue, acc0), minValue);
            // how to store faster: st4 / transpose /
            c[0] = acc0;
        }
        ie += 1;
        // a += 1;
    }

    return;
}

void MNNPackedSparseMatMulEpx4(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter, const float* postParameters, const float* bias,  unsigned int* NNZMap, int* dataOffsetMap) {

    auto eP = parameter[0] / sizeof(float);
    MNN_ASSERT((eP & 0x03) == 0); // In sparse calculate, eP should be evenly divided by 4
    auto h = parameter[2];
    auto l = parameter[1];
    auto cStride = parameter[3] / sizeof(float);
    auto hRemain = parameter[4];
    auto bExtraStride = parameter[5] / sizeof(float);
    // auto bStride = bExtraStride + l * 4;
    auto aStride = eP * l; // sizeof(float);
    auto hC4 = UP_DIV(h, 4);
    float minValue = -std::numeric_limits<float>().max();
    float maxValue = std::numeric_limits<float>().max();
    if (nullptr != postParameters) {
        minValue = postParameters[2];
        maxValue = postParameters[3];
    }
    const float32x4_t vmin = vld1q_dup_f32(&minValue);
    const float32x4_t vmax = vld1q_dup_f32(&maxValue);
    const int sparseBlockOC = 4;
    // MNN_PRINT("NEON MNNPackedSparseMatMul eP:%lu, eSize:%lu, l:%lu, h:%lu, cStride:%lu, aStride:%lu\n", eP, eSize, l, h, cStride, aStride);

    const float* a = A;
    size_t ie = 0;
    for (ie = 0; ie + eP <= eSize; ie += eP) {
        const int* dataOffset = dataOffsetMap;
        const int diff = *dataOffset++;
        a += diff;
        const float* w = B;
        float* blockC = C + (ie << 2);
        const unsigned int* nnz = NNZMap;

        size_t ih = 0;
        for (; ih < (h & (~0x03)); ih += sparseBlockOC) {
            auto ihPack = ih >> 2;
            auto c = blockC + ihPack * cStride;

            // tobe merged in to weight data
            float32x4_t vacc0c4 = nullptr != bias ? vld1q_f32(bias + ih) : vdupq_n_f32(0);
            float32x4_t vacc1c4 = vacc0c4;
            float32x4_t vacc2c4 = vacc0c4;
            float32x4_t vacc3c4 = vacc0c4;
            float32x4_t vacc4c4 = vacc0c4;
            float32x4_t vacc5c4 = vacc0c4;
            float32x4_t vacc6c4 = vacc0c4;
            float32x4_t vacc7c4 = vacc0c4;
            float32x4_t vacc8c4 = vacc0c4;
            float32x4_t vacc9c4 = vacc0c4;
            float32x4_t vacc10c4 = vacc0c4;
            float32x4_t vacc11c4 = vacc0c4;
            float32x4_t vacc12c4 = vacc0c4;
            float32x4_t vacc13c4 = vacc0c4;
            float32x4_t vacc14c4 = vacc0c4;
            float32x4_t vacc15c4 = vacc0c4;
            const unsigned int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {

                const int diff = *dataOffset++;
                const float32x4_t va0123 = vld1q_f32(a);
                const float32x4_t va4567 = vld1q_f32(a + 4);
                const float32x4_t va89AB = vld1q_f32(a + 8);
                const float32x4_t vaCDEF = vld1q_f32(a + 12);
                // __builtin_prefetch(a + aStride);

                float32x4_t w4 = vld1q_f32(w);

                // MNN_PRINT("16-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0-15]:", ie, a - A, w - B, c - C, *w);
                // formatMatrix(a, {16});
                // MNN_PRINT("\n");
                w += 4;
                a = a + diff;
                vacc0c4 = vfmaq_laneq_f32(vacc0c4, w4, va0123, 0);
                vacc4c4 = vfmaq_laneq_f32(vacc4c4, w4, va4567, 0);
                vacc8c4 = vfmaq_laneq_f32(vacc8c4, w4, va89AB, 0);
                vacc12c4 = vfmaq_laneq_f32(vacc12c4, w4, vaCDEF, 0);

                vacc1c4 = vfmaq_laneq_f32(vacc1c4, w4, va0123, 1);
                vacc5c4 = vfmaq_laneq_f32(vacc5c4, w4, va4567, 1);
                vacc9c4 = vfmaq_laneq_f32(vacc9c4, w4, va89AB, 1);
                vacc13c4 = vfmaq_laneq_f32(vacc13c4, w4, vaCDEF, 1);

                vacc2c4 = vfmaq_laneq_f32(vacc2c4, w4, va0123, 2);
                vacc6c4 = vfmaq_laneq_f32(vacc6c4, w4, va4567, 2);
                vacc10c4 = vfmaq_laneq_f32(vacc10c4, w4, va89AB, 2);
                vacc14c4 = vfmaq_laneq_f32(vacc14c4, w4, vaCDEF, 2);

                vacc3c4 = vfmaq_laneq_f32(vacc3c4, w4, va0123, 3);
                vacc7c4 = vfmaq_laneq_f32(vacc7c4, w4, va4567, 3);
                vacc11c4 = vfmaq_laneq_f32(vacc11c4, w4, va89AB, 3);
                vacc15c4 = vfmaq_laneq_f32(vacc15c4, w4, vaCDEF, 3);
            }
            vacc0c4 = vminq_f32(vacc0c4, vmax);
            vacc1c4 = vminq_f32(vacc1c4, vmax);
            vacc2c4 = vminq_f32(vacc2c4, vmax);
            vacc3c4 = vminq_f32(vacc3c4, vmax);
            vacc4c4 = vminq_f32(vacc4c4, vmax);
            vacc5c4 = vminq_f32(vacc5c4, vmax);
            vacc6c4 = vminq_f32(vacc6c4, vmax);
            vacc7c4 = vminq_f32(vacc7c4, vmax);
            vacc8c4 = vminq_f32(vacc8c4, vmax);
            vacc9c4 = vminq_f32(vacc9c4, vmax);
            vacc10c4 = vminq_f32(vacc10c4, vmax);
            vacc11c4 = vminq_f32(vacc11c4, vmax);
            vacc12c4 = vminq_f32(vacc12c4, vmax);
            vacc13c4 = vminq_f32(vacc13c4, vmax);
            vacc14c4 = vminq_f32(vacc14c4, vmax);
            vacc15c4 = vminq_f32(vacc15c4, vmax);

            vacc0c4 = vmaxq_f32(vacc0c4, vmin);
            vacc1c4 = vmaxq_f32(vacc1c4, vmin);
            vacc2c4 = vmaxq_f32(vacc2c4, vmin);
            vacc3c4 = vmaxq_f32(vacc3c4, vmin);
            vacc4c4 = vmaxq_f32(vacc4c4, vmin);
            vacc5c4 = vmaxq_f32(vacc5c4, vmin);
            vacc6c4 = vmaxq_f32(vacc6c4, vmin);
            vacc7c4 = vmaxq_f32(vacc7c4, vmin);
            vacc8c4 = vmaxq_f32(vacc8c4, vmin);
            vacc9c4 = vmaxq_f32(vacc9c4, vmin);
            vacc10c4 = vmaxq_f32(vacc10c4, vmin);
            vacc11c4 = vmaxq_f32(vacc11c4, vmin);
            vacc12c4 = vmaxq_f32(vacc12c4, vmin);
            vacc13c4 = vmaxq_f32(vacc13c4, vmin);
            vacc14c4 = vmaxq_f32(vacc14c4, vmin);
            vacc15c4 = vmaxq_f32(vacc15c4, vmin);

            // vacc is continuous along c
            vst1q_f32(c, vacc0c4);
            vst1q_f32(c + 4 , vacc1c4);
            vst1q_f32(c + 4 * 2 , vacc2c4);
            vst1q_f32(c + 4 * 3 , vacc3c4);
            vst1q_f32(c + 4 * 4 , vacc4c4);
            vst1q_f32(c + 4 * 5 , vacc5c4);
            vst1q_f32(c + 4 * 6 , vacc6c4);
            vst1q_f32(c + 4 * 7 , vacc7c4);
            vst1q_f32(c + 4 * 8 , vacc8c4);
            vst1q_f32(c + 4 * 9 , vacc9c4);
            vst1q_f32(c + 4 * 10 , vacc10c4);
            vst1q_f32(c + 4 * 11 , vacc11c4);
            vst1q_f32(c + 4 * 12 , vacc12c4);
            vst1q_f32(c + 4 * 13 , vacc13c4);
            vst1q_f32(c + 4 * 14 , vacc14c4);
            vst1q_f32(c + 4 * 15 , vacc15c4);
        }
        blockC += (h >> 2) * cStride;
        for (; ih < h; ih++) {
            auto ihSubIndex = ih & 0x03;
            auto c = blockC + ihSubIndex;
            const float initValue = nullptr != bias ? bias[ih] : 0;
            float32x4_t vacc0123 = vld1q_dup_f32(&initValue);
            float32x4_t vacc4567 = vacc0123;
            float32x4_t vacc89AB = vacc0123;
            float32x4_t vaccCDEF = vacc0123;
            const unsigned int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {

                const int diff = *dataOffset++;
                const float32x4_t va0123 = vld1q_f32(a);
                const float32x4_t va4567 = vld1q_f32(a + 4);
                const float32x4_t va89AB = vld1q_f32(a + 8);
                const float32x4_t vaCDEF = vld1q_f32(a + 12);
                // __builtin_prefetch(a + aStride);

                float32x4_t w4 = vld1q_dup_f32(w);
                // MNN_PRINT("16-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0-15]:", ie, a - A, w - B, c - C, *w);
                // formatMatrix(a, {16});
                // MNN_PRINT("\n");
                w++;
                a = a + diff;

                vacc0123 = vfmaq_f32(vacc0123, va0123, w4);
                vacc4567 = vfmaq_f32(vacc4567, va4567, w4);
                vacc89AB = vfmaq_f32(vacc89AB, va89AB, w4);
                vaccCDEF = vfmaq_f32(vaccCDEF, vaCDEF, w4);

            }
            vacc0123 = vminq_f32(vacc0123, vmax);
            vacc4567 = vminq_f32(vacc4567, vmax);
            vacc89AB = vminq_f32(vacc89AB, vmax);
            vaccCDEF = vminq_f32(vaccCDEF, vmax);
            vacc0123 = vmaxq_f32(vacc0123, vmin);
            vacc4567 = vmaxq_f32(vacc4567, vmin);
            vacc89AB = vmaxq_f32(vacc89AB, vmin);
            vaccCDEF = vmaxq_f32(vaccCDEF, vmin);

            // how to store faster: st4 / transpose /
            vst1q_lane_f32(c, vacc0123, 0);
            vst1q_lane_f32(c + 4, vacc0123, 1);
            vst1q_lane_f32(c + 4 * 2, vacc0123, 2);
            vst1q_lane_f32(c + 4 * 3, vacc0123, 3);
            vst1q_lane_f32(c+  4 * 4, vacc4567, 0);
            vst1q_lane_f32(c + 4 * 5, vacc4567, 1);
            vst1q_lane_f32(c + 4 * 6, vacc4567, 2);
            vst1q_lane_f32(c + 4 * 7, vacc4567, 3);
            vst1q_lane_f32(c + 4 * 8, vacc89AB, 0);
            vst1q_lane_f32(c + 4 * 9, vacc89AB, 1);
            vst1q_lane_f32(c + 4 * 10, vacc89AB, 2);
            vst1q_lane_f32(c + 4 * 11, vacc89AB, 3);
            vst1q_lane_f32(c + 4 * 12, vaccCDEF, 0);
            vst1q_lane_f32(c + 4 * 13, vaccCDEF, 1);
            vst1q_lane_f32(c + 4 * 14, vaccCDEF, 2);
            vst1q_lane_f32(c + 4 * 15, vaccCDEF, 3);
        }
        a += aStride;
    }
    // const float* blockA = A + ie * l;
    if (eSize & 0x08) {
        const int* dataOffset = dataOffsetMap;
        const int diff = *dataOffset++;
        // a = blockA + diff;
        a += diff;
        const float* w = B;
        float* blockC = C + (ie << 2);
        const unsigned int* nnz = NNZMap;

        size_t ih = 0;
        for (; ih < (h & (~0x03)); ih += sparseBlockOC) {
            auto ihPack = ih >> 2;
            auto c = blockC + ihPack * cStride;
            float32x4_t vacc0c4 = nullptr != bias ? vld1q_f32(bias + ih) : vdupq_n_f32(0.f);
            float32x4_t vacc1c4 = vacc0c4;
            float32x4_t vacc2c4 = vacc0c4;
            float32x4_t vacc3c4 = vacc0c4;
            float32x4_t vacc4c4 = vacc0c4;
            float32x4_t vacc5c4 = vacc0c4;
            float32x4_t vacc6c4 = vacc0c4;
            float32x4_t vacc7c4 = vacc0c4;

            const unsigned int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {

                const int diff = *dataOffset++;
                const float32x4_t va0123 = vld1q_f32(a);
                const float32x4_t va4567 = vld1q_f32(a + 4);
                const float32x4_t va89AB = vld1q_f32(a + 8);
                const float32x4_t vaCDEF = vld1q_f32(a + 12);
                // __builtin_prefetch(a + aStride);

                float32x4_t w4 = vld1q_f32(w);

                // MNN_PRINT("16-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0-15]:", ie, a - A, w - B, c - C, *w);
                // formatMatrix(a, {16});
                // MNN_PRINT("\n");
                w += 4;
                a = a + diff;
                vacc0c4 = vfmaq_laneq_f32(vacc0c4, w4, va0123, 0);
                vacc4c4 = vfmaq_laneq_f32(vacc4c4, w4, va4567, 0);

                vacc1c4 = vfmaq_laneq_f32(vacc1c4, w4, va0123, 1);
                vacc5c4 = vfmaq_laneq_f32(vacc5c4, w4, va4567, 1);

                vacc2c4 = vfmaq_laneq_f32(vacc2c4, w4, va0123, 2);
                vacc6c4 = vfmaq_laneq_f32(vacc6c4, w4, va4567, 2);

                vacc3c4 = vfmaq_laneq_f32(vacc3c4, w4, va0123, 3);
                vacc7c4 = vfmaq_laneq_f32(vacc7c4, w4, va4567, 3);
            }
            vacc0c4 = vminq_f32(vacc0c4, vmax);
            vacc1c4 = vminq_f32(vacc1c4, vmax);
            vacc2c4 = vminq_f32(vacc2c4, vmax);
            vacc3c4 = vminq_f32(vacc3c4, vmax);
            vacc4c4 = vminq_f32(vacc4c4, vmax);
            vacc5c4 = vminq_f32(vacc5c4, vmax);
            vacc6c4 = vminq_f32(vacc6c4, vmax);
            vacc7c4 = vminq_f32(vacc7c4, vmax);

            vacc0c4 = vmaxq_f32(vacc0c4, vmin);
            vacc1c4 = vmaxq_f32(vacc1c4, vmin);
            vacc2c4 = vmaxq_f32(vacc2c4, vmin);
            vacc3c4 = vmaxq_f32(vacc3c4, vmin);
            vacc4c4 = vmaxq_f32(vacc4c4, vmin);
            vacc5c4 = vmaxq_f32(vacc5c4, vmin);
            vacc6c4 = vmaxq_f32(vacc6c4, vmin);
            vacc7c4 = vmaxq_f32(vacc7c4, vmin);

            // vacc is continuous along c
            vst1q_f32(c, vacc0c4);
            vst1q_f32(c + 4 , vacc1c4);
            vst1q_f32(c + 4 * 2 , vacc2c4);
            vst1q_f32(c + 4 * 3 , vacc3c4);
            vst1q_f32(c + 4 * 4 , vacc4c4);
            vst1q_f32(c + 4 * 5 , vacc5c4);
            vst1q_f32(c + 4 * 6 , vacc6c4);
            vst1q_f32(c + 4 * 7 , vacc7c4);
        }
        blockC += (h >> 2) * cStride;
        for (; ih < h; ih++) {
            auto ihSubIndex = ih & 0x03;
            auto c = blockC + ihSubIndex;
            const float initValue = nullptr != bias ? bias[ih] : 0;
            float32x4_t vacc0123 = vld1q_dup_f32(&initValue);
            float32x4_t vacc4567 = vacc0123;

            const unsigned int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {
                const int diff = *dataOffset++;
                const float32x4_t va0123 = vld1q_f32(a);
                const float32x4_t va4567 = vld1q_f32(a + 4);
                // __builtin_prefetch(a + aStride);

                float32x4_t w4 = vld1q_dup_f32(w);
                // MNN_PRINT("8-4-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0-7]:", ie, a - A, w - B, c - C, *w);
                // formatMatrix(a, {8});
                // MNN_PRINT("\n");
                w++;
                a = a + diff;
                vacc0123 = vfmaq_f32(vacc0123, va0123, w4);
                vacc4567 = vfmaq_f32(vacc4567, va4567, w4);
            }
            vacc0123 = vminq_f32(vacc0123, vmax);
            vacc4567 = vminq_f32(vacc4567, vmax);
            vacc0123 = vmaxq_f32(vacc0123, vmin);
            vacc4567 = vmaxq_f32(vacc4567, vmin);

            // how to store faster: st4 / transpose /
            vst1q_lane_f32(c, vacc0123, 0);
            vst1q_lane_f32(c + 4, vacc0123, 1);
            vst1q_lane_f32(c + 4 * 2, vacc0123, 2);
            vst1q_lane_f32(c + 4 * 3, vacc0123, 3);
            vst1q_lane_f32(c + 4 * 4, vacc4567, 0);
            vst1q_lane_f32(c + 4 * 5, vacc4567, 1);
            vst1q_lane_f32(c + 4 * 6, vacc4567, 2);
            vst1q_lane_f32(c + 4 * 7, vacc4567, 3);
        }
        ie += 8;
        a += 8;
    }

    if (eSize & 0x04) {
        const int* dataOffset = dataOffsetMap;
        const int diff = *dataOffset++;
        // const float* a = blockA + diff;
        a += diff;
        const float* w = B;
        float* blockC = C + (ie << 2);
        const unsigned int* nnz = NNZMap;
        size_t ih = 0;
        for (; ih < (h & (~0x03)); ih += sparseBlockOC) {
            auto ihPack = ih >> 2;
            auto c = blockC + ihPack * cStride;
            float32x4_t vacc0c4 = nullptr != bias ? vld1q_f32(bias + ih) : vdupq_n_f32(0);
            float32x4_t vacc1c4 = vacc0c4;
            float32x4_t vacc2c4 = vacc0c4;
            float32x4_t vacc3c4 = vacc0c4;
            const unsigned int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {

                const int diff = *dataOffset++;
                const float32x4_t va0123 = vld1q_f32(a);
                // __builtin_prefetch(a + aStride);

                float32x4_t w4 = vld1q_f32(w);

                // MNN_PRINT("4-4-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0-15]:", ie, a - A, w - B, c - C, *w);
                // formatMatrix(a, {16});
                // MNN_PRINT("\n");
                w += 4;
                a = a + diff;
                vacc0c4 = vfmaq_laneq_f32(vacc0c4, w4, va0123, 0);
                vacc1c4 = vfmaq_laneq_f32(vacc1c4, w4, va0123, 1);
                vacc2c4 = vfmaq_laneq_f32(vacc2c4, w4, va0123, 2);
                vacc3c4 = vfmaq_laneq_f32(vacc3c4, w4, va0123, 3);
            }
            vacc0c4 = vminq_f32(vacc0c4, vmax);
            vacc1c4 = vminq_f32(vacc1c4, vmax);
            vacc2c4 = vminq_f32(vacc2c4, vmax);
            vacc3c4 = vminq_f32(vacc3c4, vmax);
            vacc0c4 = vmaxq_f32(vacc0c4, vmin);
            vacc1c4 = vmaxq_f32(vacc1c4, vmin);
            vacc2c4 = vmaxq_f32(vacc2c4, vmin);
            vacc3c4 = vmaxq_f32(vacc3c4, vmin);
            // vacc is continuous along c
            vst1q_f32(c, vacc0c4);
            vst1q_f32(c + 4 , vacc1c4);
            vst1q_f32(c + 4 * 2 , vacc2c4);
            vst1q_f32(c + 4 * 3 , vacc3c4);
        }
        blockC += (h >> 2) * cStride;
        for (; ih < h; ih++) {
            auto ihSubIndex = ih & 0x03;
            auto c = blockC + ihSubIndex;
            const float initValue = nullptr != bias ? bias[ih] : 0;
            float32x4_t vacc0123 = vld1q_dup_f32(&initValue);

            const unsigned int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {
                const int diff = *dataOffset++;
                const float32x4_t va0123 = vld1q_f32(a);
                // __builtin_prefetch(a + aStride);

                float32x4_t w4 = vld1q_dup_f32(w);
                // MNN_PRINT("4-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0-3]:", ie, a - A, w - B, c - C, *w);
                // formatMatrix(a, {4});
                // MNN_PRINT("\n");
                w++;
                a = a + diff;
                vacc0123 = vfmaq_f32(vacc0123, va0123, w4);
            }
            vacc0123 = vminq_f32(vacc0123, vmax);
            vacc0123 = vmaxq_f32(vacc0123, vmin);

            // how to store faster: st4 / transpose /
            vst1q_lane_f32(c, vacc0123, 0);
            vst1q_lane_f32(c + 4, vacc0123, 1);
            vst1q_lane_f32(c + 4 * 2, vacc0123, 2);
            vst1q_lane_f32(c + 4 * 3, vacc0123, 3);
        }
        ie += 4;
        a += 4;
    }
    if (eSize & 0x02) {
        const int* dataOffset = dataOffsetMap;
        const int diff = *dataOffset++;
        // const float* a = blockA + diff;
        a += diff;
        const float* w = B;
        float* blockC = C + (ie << 2);
        const unsigned int* nnz = NNZMap;
        size_t ih = 0;
        for (; ih < (h & (~0x03)); ih += sparseBlockOC) {
            auto ihPack = ih >> 2;
            auto c = blockC + ihPack * cStride;
            float32x4_t vacc0c4 = nullptr != bias ? vld1q_f32(bias + ih) : vdupq_n_f32(0.f);
            float32x4_t vacc1c4 = vacc0c4;
            const unsigned int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {

                const int diff = *dataOffset++;
                const float32x2_t va01 = vld1_f32(a);
                // __builtin_prefetch(a + aStride);

                float32x4_t w4 = vld1q_f32(w);

                // MNN_PRINT("2-4-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0-15]:", ie, a - A, w - B, c - C, *w);
                // formatMatrix(a, {16});
                // MNN_PRINT("\n");
                w += 4;
                a = a + diff;
                vacc0c4 = vfmaq_lane_f32(vacc0c4, w4, va01, 0);
                vacc1c4 = vfmaq_lane_f32(vacc1c4, w4, va01, 1);
            }
            vacc0c4 = vminq_f32(vacc0c4, vmax);
            vacc1c4 = vminq_f32(vacc1c4, vmax);
            vacc0c4 = vmaxq_f32(vacc0c4, vmin);
            vacc1c4 = vmaxq_f32(vacc1c4, vmin);
            // vacc is continuous along c
            vst1q_f32(c, vacc0c4);
            vst1q_f32(c + 4, vacc1c4);
        }
        blockC += (h >> 2) * cStride;
        for (; ih < h; ih++) {
            auto ihSubIndex = ih & 0x03;
            auto c = blockC + ihSubIndex;
            const float initValue = nullptr != bias ? bias[ih] : 0;
            float32x2_t vacc01 = vld1_dup_f32(&initValue);

            const unsigned int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {
                const int diff = *dataOffset++;
                const float32x2_t va01 = vld1_f32(a);
                // __builtin_prefetch(a + aStride);

                float32x2_t w2 = vld1_dup_f32(w);
                // MNN_PRINT("2-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0-1]:", ie, a - A, w - B, c - C, *w);
                // formatMatrix(a, {2});
                // MNN_PRINT("\n");
                w++;
                a = a + diff;
                vacc01 = vfma_f32(vacc01, va01, w2);
            }
            vacc01 = vmin_f32(vacc01, vget_low_f32(vmax));
            vacc01 = vmax_f32(vacc01, vget_low_f32(vmin));
            // how to store faster: st4 / transpose /
            vst1_lane_f32(c, vacc01, 0);
            vst1_lane_f32(c + 4, vacc01, 1);
        }
        ie += 2;
        a += 2;
    }
    if (eSize & 0x01) {
        const int* dataOffset = dataOffsetMap;
        const int diff = *dataOffset++;
        // const float* a = blockA + diff;
        a += diff;
        const float* w = B;
        float* blockC = C + (ie << 2);
        const unsigned int* nnz = NNZMap;
        size_t ih = 0;
        for (; ih < (h & (~0x03)); ih += sparseBlockOC) {
            auto ihPack = ih >> 2;
            auto c = blockC + ihPack * cStride;
            float32x4_t vacc0c4 = nullptr != bias ? vld1q_f32(bias + ih) : vdupq_n_f32(0);
            const unsigned int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {

                const int diff = *dataOffset++;
                const float32x2_t va01 = vld1_f32(a);
                // __builtin_prefetch(a + aStride);

                float32x4_t w4 = vld1q_f32(w);

                // MNN_PRINT("1-4-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0-15]:", ie, a - A, w - B, c - C, *w);
                // formatMatrix(a, {16});
                // MNN_PRINT("\n");
                w += 4;
                a = a + diff;
                vacc0c4 = vfmaq_lane_f32(vacc0c4, w4, va01, 0);
            }
            vacc0c4 = vminq_f32(vacc0c4, vmax);
            vacc0c4 = vmaxq_f32(vacc0c4, vmin);
            // vacc is continuous along c
            vst1q_f32(c, vacc0c4);
        }
        blockC += (h >> 2) * cStride;
        for (; ih < h; ih++) {
            auto ihSubIndex = ih & 0x03;
            auto c = blockC + ihSubIndex;
            const float initValue = nullptr != bias ? bias[ih] : 0;
            float acc0 = initValue;

            const unsigned int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {
                const int diff = *dataOffset++;
                const float a0 = a[0];
                const float oneW = *w++;
                // MNN_PRINT("1-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0]:", ie, a - A, w - B - 1, c - C, oneW);
                // formatMatrix(a, {1});
                // MNN_PRINT("\n");
                a = a + diff;
                acc0 += a0 * oneW;
            }
            acc0  = std::max(std::min(maxValue, acc0), minValue);
            // how to store faster: st4 / transpose /
            c[0] = acc0;
        }
        ie += 1;
        // a += 1;
    }

    return;
}

#endif

void MNNGetSparseMatMulPackMode(int* eP, int *lP, int* hP) {
#ifdef __aarch64__
    *eP = 16;
#else
    *eP = 8; // total vector number is 16, we choose to use 8 for output.
#endif
    *lP = 1;
    *hP = 4;
    // hp is corresponding to sparse block along right matrix colum dimension. in ramdom sparse, it is 1.
    return;
}


void MNNGetMatMulPackMode(int* eP, int *lP, int* hP) {
    *eP = 12;
    *lP = 1;
#ifdef __aarch64__
    *hP = 8;
#else
    *hP = 4;
#endif
}

#ifdef __aarch64__

// input shape is (l, h) when transpose=false, else input shape is (h, l)
// output shape is (UP_DIV(h, 8), l, 8)
void MNNPackForMatMul_B(float* dest, const float* source, size_t h, size_t l, bool transpose) {
    auto hP = (int)h / 8;
    auto hR = (int)hP * 8;
    if (hR != h) {
        ::memset(dest, 0, UP_DIV(h, 8)*8*l*sizeof(float));
    }
    if (!transpose) {
        for (int y=0; y<hP; ++y) {
            auto destY = dest + y * 8 * l;
            auto sourceY = source + y * 8;
            for (int x=0; x<l; ++x) {
                ::memcpy(destY + 8 * x, sourceY + x * h, 8 * sizeof(float));
            }
        }
        auto hRemain = h - hR;
        if (hRemain > 0) {
            auto destY = dest + hP * 8 * l;
            auto sourceY = source + hP * 8;
            for (int x=0; x<l; ++x) {
                ::memcpy(destY + 8 * x, sourceY + x * h, hRemain * sizeof(float));
            }
        }
        return;
    }
    int lC8 = (int)l / 8;
    auto lR = lC8 * 8;
    if (hP > 0 && lC8 > 0) {
        MNNPackC8(dest, source, l, h);
    }
    for (int y=hR; y<h; ++y) {
        auto yR = y % 8;
        auto yC = hP;
        for (int x=0; x<l; ++x) {
            dest[x * 8 + yR + yC * 8 * l] = source[x + y * l];
        }
    }
    for (int y=0; y<hR; ++y) {
        auto yR = y % 8;
        auto yC = y / 8;
        for (int x=lR; x<l; ++x) {
            dest[x * 8 + yR + yC * 8 * l] = source[x + y * l];
        }
    }
}
#else
void MNNPackForMatMul_B(float* dest, const float* source, size_t h, size_t l, bool transpose) {
    if (!transpose) {
        auto hP = h / 4;
        auto hR = hP * 4;
        if (hR != h) {
            ::memset(dest, 0, UP_DIV(h, 4)*4*l*sizeof(float));
        }
        for (int y=0; y<hP; ++y) {
            auto destY = dest + y * 4 * l;
            auto sourceY = source + y * 4;
            for (int x=0; x<l; ++x) {
                ::memcpy(destY + 4 * x, sourceY + x * h, 4 * sizeof(float));
            }
        }
        auto hRemain = h - hR;
        if (hRemain > 0) {
            auto destY = dest + hP * 4 * l;
            auto sourceY = source + hP * 4;
            for (int x=0; x<l; ++x) {
                ::memcpy(destY + 4 * x, sourceY + x * h, hRemain * sizeof(float));
            }
        }
        return;
    }
    int offset[] = {
        (int)l, (int)l
    };
    MNNPackC4(dest, source, l, h, offset);
}
#endif


#endif
