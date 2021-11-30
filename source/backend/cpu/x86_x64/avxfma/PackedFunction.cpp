//
//  PackedFunction.cpp
//  MNN
//
//  Created by MNN on b'2021/07/05'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "FunctionSummary.hpp"
#include "core/Macro.h"

#define PACK_UNIT 8

void _AVX_MNNConvRunForUnitDepthWiseFMA(float* dst, const float* src, const float* weight, size_t fw, size_t fh,
                                  size_t weight_y_step, size_t dilateX_step, size_t dilateY_step) {
    int fx, fy;
    __m256 dstValue = _mm256_setzero_ps();
    const float* src_z    = src;
    const float* weight_z = weight;
    for (fy = 0; fy < fh; ++fy) {
        const float* src_y    = src_z + fy * dilateY_step;
        const float* weight_y = weight_z + fy * weight_y_step;
        for (fx = 0; fx < fw; ++fx) {
            const float* weight_x = weight_y + PACK_UNIT * fx;
            const float* src_x    = src_y + fx * dilateX_step;
            dstValue = _mm256_fmadd_ps(_mm256_loadu_ps(src_x), _mm256_loadu_ps(weight_x), dstValue);
        }
    }
    _mm256_storeu_ps(dst, dstValue);
}

void _AVX_MNNConvRunForLineDepthwiseFMA(float* dst, const float* src, const float* weight, size_t width, size_t src_w_setup,
                                size_t fw, size_t fh, size_t dilateX_step, size_t dilateY_step, size_t height,
                                     size_t srcHStep, size_t dstHStep) {
    int dx, fx, fy;
    const int unit = 4;
    int widthUnit = width / unit;
    int widthRemain = width - widthUnit * unit;
    const float* weight_z = weight;
    for (int y = 0; y < height; ++y) {
        auto srcY = src + y * srcHStep;
        auto dstY = dst + y * dstHStep;
        for (dx = 0; dx < widthUnit; ++dx) {
            auto dstValue0 = _mm256_setzero_ps();
            auto dstValue1 = _mm256_setzero_ps();
            auto dstValue2 = _mm256_setzero_ps();
            auto dstValue3 = _mm256_setzero_ps();
            for (fy = 0; fy < fh; ++fy) {
                const float* src_y    = srcY + fy * dilateY_step;
                const float* weight_y = weight_z + fy * fw * PACK_UNIT;
                for (fx = 0; fx < fw; ++fx) {
                    const float* src_x    = src_y + fx * dilateX_step;
                    const float* weight_x = weight_y + PACK_UNIT * fx;
                    auto weightValue = _mm256_loadu_ps(weight_x);
                    dstValue0 = _mm256_fmadd_ps(_mm256_loadu_ps(src_x + 0 * src_w_setup), weightValue, dstValue0);
                    dstValue1 = _mm256_fmadd_ps(_mm256_loadu_ps(src_x + 1 * src_w_setup), weightValue, dstValue1);
                    dstValue2 = _mm256_fmadd_ps(_mm256_loadu_ps(src_x + 2 * src_w_setup), weightValue, dstValue2);
                    dstValue3 = _mm256_fmadd_ps(_mm256_loadu_ps(src_x + 3 * src_w_setup), weightValue, dstValue3);
                }
            }
            _mm256_storeu_ps(dstY + PACK_UNIT * 0, dstValue0);
            _mm256_storeu_ps(dstY + PACK_UNIT * 1, dstValue1);
            _mm256_storeu_ps(dstY + PACK_UNIT * 2, dstValue2);
            _mm256_storeu_ps(dstY + PACK_UNIT * 3, dstValue3);
            dstY += PACK_UNIT * unit;
            srcY += unit * src_w_setup;
        }
        for (dx = 0; dx < widthRemain; ++dx) {
            float* dst_x          = dstY + dx * PACK_UNIT;
            auto dstValue = _mm256_setzero_ps();
            const float* src_z    = srcY + src_w_setup * dx;
            const float* weight_z = weight;
            for (fy = 0; fy < fh; ++fy) {
                const float* src_y    = src_z + fy * dilateY_step;
                const float* weight_y = weight_z + fy * fw * PACK_UNIT;
                for (fx = 0; fx < fw; ++fx) {
                    const float* weight_x = weight_y + PACK_UNIT * fx;
                    const float* src_x    = src_y + fx * dilateX_step;
                    dstValue = _mm256_fmadd_ps(_mm256_loadu_ps(src_x), _mm256_loadu_ps(weight_x), dstValue);
                }
            }
            _mm256_storeu_ps(dst_x, dstValue);
        }
    }
}

void _AVX_MNNConvDwF23MulTransUnitFMA(float **cacheLine, const float *weigth, float *dest, size_t ow, const float* bias, const float* parameter) {
    int unit = ow / 2;
    auto SRC_TILE_UNIT = 4 * PACK_UNIT;
    auto DST_TILE_UNIT = 2 * PACK_UNIT;

    auto w00 = _mm256_loadu_ps(weigth + 0 * SRC_TILE_UNIT + PACK_UNIT * 0);
    auto w01 = _mm256_loadu_ps(weigth + 0 * SRC_TILE_UNIT + PACK_UNIT * 1);
    auto w02 = _mm256_loadu_ps(weigth + 0 * SRC_TILE_UNIT + PACK_UNIT * 2);
    auto w03 = _mm256_loadu_ps(weigth + 0 * SRC_TILE_UNIT + PACK_UNIT * 3);
    auto w10 = _mm256_loadu_ps(weigth + 1 * SRC_TILE_UNIT + PACK_UNIT * 0);
    auto w11 = _mm256_loadu_ps(weigth + 1 * SRC_TILE_UNIT + PACK_UNIT * 1);
    auto w12 = _mm256_loadu_ps(weigth + 1 * SRC_TILE_UNIT + PACK_UNIT * 2);
    auto w13 = _mm256_loadu_ps(weigth + 1 * SRC_TILE_UNIT + PACK_UNIT * 3);
    auto w20 = _mm256_loadu_ps(weigth + 2 * SRC_TILE_UNIT + PACK_UNIT * 0);
    auto w21 = _mm256_loadu_ps(weigth + 2 * SRC_TILE_UNIT + PACK_UNIT * 1);
    auto w22 = _mm256_loadu_ps(weigth + 2 * SRC_TILE_UNIT + PACK_UNIT * 2);
    auto w23 = _mm256_loadu_ps(weigth + 2 * SRC_TILE_UNIT + PACK_UNIT * 3);
    auto biasF = _mm256_loadu_ps(bias);
    auto minF = _mm256_broadcast_ss(parameter + 2);
    auto maxF = _mm256_broadcast_ss(parameter + 3);

    for (int x = 0; x < unit; ++x) {
        auto offset = PACK_UNIT * 4 * x;
        int i = 0;
        auto m0     = _mm256_mul_ps(w00, _mm256_loadu_ps(cacheLine[0] + offset + PACK_UNIT * 0));
        auto m1     = _mm256_mul_ps(w01, _mm256_loadu_ps(cacheLine[0] + offset + PACK_UNIT * 1));
        auto m2     = _mm256_mul_ps(w02, _mm256_loadu_ps(cacheLine[0] + offset + PACK_UNIT * 2));
        auto m3     = _mm256_mul_ps(w03, _mm256_loadu_ps(cacheLine[0] + offset + PACK_UNIT * 3));

        m0 = _mm256_fmadd_ps(w10, _mm256_loadu_ps(cacheLine[1] + offset + PACK_UNIT * 0), m0);
        m1 = _mm256_fmadd_ps(w11, _mm256_loadu_ps(cacheLine[1] + offset + PACK_UNIT * 1), m1);
        m2 = _mm256_fmadd_ps(w12, _mm256_loadu_ps(cacheLine[1] + offset + PACK_UNIT * 2), m2);
        m3 = _mm256_fmadd_ps(w13, _mm256_loadu_ps(cacheLine[1] + offset + PACK_UNIT * 3), m3);

        m0 = _mm256_fmadd_ps(w20, _mm256_loadu_ps(cacheLine[2] + offset + PACK_UNIT * 0), m0);
        m1 = _mm256_fmadd_ps(w21, _mm256_loadu_ps(cacheLine[2] + offset + PACK_UNIT * 1), m1);
        m2 = _mm256_fmadd_ps(w22, _mm256_loadu_ps(cacheLine[2] + offset + PACK_UNIT * 2), m2);
        m3 = _mm256_fmadd_ps(w23, _mm256_loadu_ps(cacheLine[2] + offset + PACK_UNIT * 3), m3);

        auto o0 = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(m0, m1), m2), biasF);
        auto o1 = _mm256_add_ps(_mm256_add_ps(_mm256_sub_ps(m1, m2), m3), biasF);
        o0 = _mm256_min_ps(maxF, o0);
        o1 = _mm256_min_ps(maxF, o1);
        o0 = _mm256_max_ps(minF, o0);
        o1 = _mm256_max_ps(minF, o1);
        _mm256_storeu_ps(dest + DST_TILE_UNIT * x + 0 * PACK_UNIT, o0);
        _mm256_storeu_ps(dest + DST_TILE_UNIT * x + 1 * PACK_UNIT, o1);
    }
    if (unit * 2 < ow) {
        auto offset = PACK_UNIT * 4 * unit;
        auto m0     = _mm256_mul_ps(w00, _mm256_loadu_ps(cacheLine[0] + offset + PACK_UNIT * 0));
        auto m1     = _mm256_mul_ps(w01, _mm256_loadu_ps(cacheLine[0] + offset + PACK_UNIT * 1));
        auto m2     = _mm256_mul_ps(w02, _mm256_loadu_ps(cacheLine[0] + offset + PACK_UNIT * 2));

        m0 = _mm256_fmadd_ps(w10, _mm256_loadu_ps(cacheLine[1] + offset + PACK_UNIT * 0), m0);
        m1 = _mm256_fmadd_ps(w11, _mm256_loadu_ps(cacheLine[1] + offset + PACK_UNIT * 1), m1);
        m2 = _mm256_fmadd_ps(w12, _mm256_loadu_ps(cacheLine[1] + offset + PACK_UNIT * 2), m2);

        m0 = _mm256_fmadd_ps(w20, _mm256_loadu_ps(cacheLine[2] + offset + PACK_UNIT * 0), m0);
        m1 = _mm256_fmadd_ps(w21, _mm256_loadu_ps(cacheLine[2] + offset + PACK_UNIT * 1), m1);
        m2 = _mm256_fmadd_ps(w22, _mm256_loadu_ps(cacheLine[2] + offset + PACK_UNIT * 2), m2);

        auto o0 = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(m0, m1), m2), biasF);
        o0 = _mm256_min_ps(maxF, o0);
        o0 = _mm256_max_ps(minF, o0);
        _mm256_storeu_ps(dest + DST_TILE_UNIT * unit, o0);
    }
}

static void _AVXFMA_MNNAdjustOptimalSparseKernel(int& sparseBlockOC, MNN::CoreFunctions::MNNPackedSparseMatMul& packedSparseMatMul) {
    if(sparseBlockOC == 4) {
        packedSparseMatMul = _AVX_MNNPackedSparseMatMulEpx4NFMA;
        return;
    } else if(sparseBlockOC % 4 == 0) {
        // MNN_PRINT("avxfma downgrade sparse from:%d, ",sparseBlockOC);
        sparseBlockOC = 4;
        packedSparseMatMul = _AVX_MNNPackedSparseMatMulEpx4NFMA;
        // MNN_PRINT(" to:%d\n",sparseBlockOC);
        return;
    } else {
        sparseBlockOC = 1;
        packedSparseMatMul = _AVX_MNNPackedSparseMatMulEpx1NFMA;
        return;
    }
}

void _AVX_ExtraInitFMA(void* functions) {
    auto coreFunction = static_cast<MNN::CoreFunctions*>(functions);
    coreFunction->MNNConvRunForLineDepthwise = _AVX_MNNConvRunForLineDepthwiseFMA;
    coreFunction->MNNConvRunForUnitDepthWise = _AVX_MNNConvRunForUnitDepthWiseFMA;
    coreFunction->MNNConvDwF23MulTransUnit = _AVX_MNNConvDwF23MulTransUnitFMA;
    // sparse conv init
    coreFunction->MNNAdjustOptimalSparseKernel = _AVXFMA_MNNAdjustOptimalSparseKernel;

}
