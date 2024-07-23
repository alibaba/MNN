//
//  PackedFunction.cpp
//  MNN
//
//  Created by MNN on 2021/07/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <float.h>
#include <string.h>
#include <algorithm>
#include <limits>
#include <vector>
#include "FunctionSummary.hpp"
#include "core/Macro.h"
#include "backend/cpu/CPUPool.hpp"
#include "backend/cpu/BinaryUtils.hpp"
#include "Vec16.hpp"
#define PACK_UNIT 16
#define PACK PACK_UNIT
#define FLOAT float
using Vec = Vec16;
#include "backend/cpu/GridSampler.hpp"

void _AVX512_MNNCopyC4WithStride(const float* source, float* dest, size_t srcStride, size_t dstStride, size_t count) {
    for (int i = 0; i < count; ++i) {
        auto s = source + i * srcStride;
        auto d = dest + i * dstStride;
        _mm512_storeu_ps(d, _mm512_loadu_ps(s));
    }
}
void _AVX512_MNNAddC4WithStride(const float* source, float* dest, size_t srcStride, size_t dstStride, size_t count) {
    for (int i = 0; i < count; ++i) {
        auto s = source + i * srcStride;
        auto d = dest + i * dstStride;
        _mm512_storeu_ps(d, _mm512_add_ps(_mm512_loadu_ps(s), _mm512_loadu_ps(d)));
    }
}

void _AVX512_MNNComputeScaleZeroScalar(float* source, float* min, float* max, size_t size) {
    int pack = 16;
    int sizeDiv16 = size / pack;
    __m512 minVal = _mm512_set1_ps(source[0]);
    __m512 maxVal = minVal;
    float maxArr[16], minArr[16];
    for (int i = 0; i < sizeDiv16; ++i) {
        auto src0 = source + pack * i;
        __m512 vecA = _mm512_loadu_ps(src0);
        auto maskMax = _mm512_cmp_ps_mask(vecA, maxVal, 14);
        auto maskMin = _mm512_cmp_ps_mask(vecA, minVal, 1);
        maxVal = _mm512_mask_blend_ps(maskMax, maxVal, vecA);
        minVal = _mm512_mask_blend_ps(maskMin, minVal, vecA);
    }
    _mm512_storeu_ps(maxArr, maxVal);
    _mm512_storeu_ps(minArr, minVal);
    float max_ = maxArr[0], min_ = minArr[0];
    for (int k = 1; k < pack; ++k) {
        if (max_ < maxArr[k]) {
            max_ = maxArr[k];
        }
        if (min_ > minArr[k]) {
            min_ = minArr[k];
        }
    }
    for (int i = pack * sizeDiv16; i < size; ++i) {
        min_ = ALIMIN(min_, source[i]);
        max_ = ALIMAX(max_, source[i]);
    }
    min[0] = min_;
    max[0] = max_;
}

void _AVX512_MNNAbsMaxFP32(const float* source, float* absmax, size_t src_depth_quad, size_t realSize, int pack) {
    // source: (ic/4, N, 4)
    auto srcStep = pack * realSize;
    for (int i = 0; i < realSize; ++i) {
        float absmaxVal = 0.f; // absmaxVal>=0
        for (int c = 0; c < src_depth_quad; ++c) {
            auto src = source + c * srcStep + i * pack;
            for (int k = 0; k < pack; ++k) {
                absmaxVal = std::max(absmaxVal, std::abs(src[k]));
            }
        }
        absmax[i] = absmaxVal;
    }
}

void _AVX512_MNNReluWithSlopeChannel(float* dst, const float* src, const float* slope, size_t sizeQuad, size_t depthQuad) {
    auto zero = _mm_set1_ps(0.0f);
    auto zero2 = _mm512_set1_ps(0.0f);
    int sizeC8 = sizeQuad;
    for (int j = 0; j < depthQuad; j++) {
        auto slopeZ       = _mm512_loadu_ps(slope + PACK_UNIT * j);
        const float* srcZ = src + PACK_UNIT * j * sizeQuad;
        float* dstZ       = dst + PACK_UNIT * j * sizeQuad;
        for (int i = 0; i < sizeC8; i++) {
            auto src   = _mm512_loadu_ps(srcZ);
            auto mask0 = _mm512_cmp_ps_mask(src, zero2, 0x01);
            auto other = _mm512_mul_ps(src, slopeZ);
            _mm512_storeu_ps(dstZ, _mm512_mask_blend_ps(mask0, src, other));
            srcZ += PACK_UNIT;
            dstZ += PACK_UNIT;
        }
    }
}

void _AVX512_MNNAxByClampBroadcastUnit(float* C, const float* A, const float* B, size_t width, size_t cStride, size_t aStride, size_t height, const float* parameters) {
    auto minF = _mm512_broadcastss_ps(_mm_load_ss(parameters + 2));
    auto maxF = _mm512_broadcastss_ps(_mm_load_ss(parameters + 3));
    for (int y = 0; y < height; ++y) {
        auto a = A + aStride * y;
        auto b = B + PACK_UNIT * y;
        auto bv = _mm512_loadu_ps(b);
        auto c = C + cStride * y;
        for (int x = 0; x < width; ++x) {
            auto av = _mm512_loadu_ps(a);
            auto cv = _mm512_add_ps(av, bv);
            cv = _mm512_min_ps(cv, maxF);
            cv = _mm512_max_ps(cv, minF);
            _mm512_storeu_ps(c, cv);
            a += PACK_UNIT;
            c += PACK_UNIT;
        }
    }
}

void _AVX512_MNNConvRunForUnitDepthWise(float* dst, const float* src, const float* weight, size_t fw, size_t fh,
                                  size_t weight_y_step, size_t dilateX_step, size_t dilateY_step) {
    int fx, fy;
    __m512 dstValue = _mm512_setzero_ps();
    const float* src_z    = src;
    const float* weight_z = weight;
    for (fy = 0; fy < fh; ++fy) {
        const float* src_y    = src_z + fy * dilateY_step;
        const float* weight_y = weight_z + fy * weight_y_step;
        for (fx = 0; fx < fw; ++fx) {
            const float* weight_x = weight_y + PACK_UNIT * fx;
            const float* src_x    = src_y + fx * dilateX_step;
            dstValue = _mm512_fmadd_ps(_mm512_loadu_ps(src_x), _mm512_loadu_ps(weight_x), dstValue);
        }
    }
    _mm512_storeu_ps(dst, dstValue);
}

void _AVX512_MNNConvRunForLineDepthwise(float* dst, const float* src, const float* weight, size_t width, size_t src_w_setup,
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
            auto dstValue0 = _mm512_setzero_ps();
            auto dstValue1 = _mm512_setzero_ps();
            auto dstValue2 = _mm512_setzero_ps();
            auto dstValue3 = _mm512_setzero_ps();
            for (fy = 0; fy < fh; ++fy) {
                const float* src_y    = srcY + fy * dilateY_step;
                const float* weight_y = weight_z + fy * fw * PACK_UNIT;
                for (fx = 0; fx < fw; ++fx) {
                    const float* src_x    = src_y + fx * dilateX_step;
                    const float* weight_x = weight_y + PACK_UNIT * fx;
                    auto weightValue = _mm512_loadu_ps(weight_x);
                    dstValue0 = _mm512_fmadd_ps(_mm512_loadu_ps(src_x + 0 * src_w_setup), weightValue, dstValue0);
                    dstValue1 = _mm512_fmadd_ps(_mm512_loadu_ps(src_x + 1 * src_w_setup), weightValue, dstValue1);
                    dstValue2 = _mm512_fmadd_ps(_mm512_loadu_ps(src_x + 2 * src_w_setup), weightValue, dstValue2);
                    dstValue3 = _mm512_fmadd_ps(_mm512_loadu_ps(src_x + 3 * src_w_setup), weightValue, dstValue3);
                }
            }
            _mm512_storeu_ps(dstY + PACK_UNIT * 0, dstValue0);
            _mm512_storeu_ps(dstY + PACK_UNIT * 1, dstValue1);
            _mm512_storeu_ps(dstY + PACK_UNIT * 2, dstValue2);
            _mm512_storeu_ps(dstY + PACK_UNIT * 3, dstValue3);
            dstY += PACK_UNIT * unit;
            srcY += unit * src_w_setup;
        }
        for (dx = 0; dx < widthRemain; ++dx) {
            float* dst_x          = dstY + dx * PACK_UNIT;
            auto dstValue = _mm512_setzero_ps();
            const float* src_z    = srcY + src_w_setup * dx;
            const float* weight_z = weight;
            for (fy = 0; fy < fh; ++fy) {
                const float* src_y    = src_z + fy * dilateY_step;
                const float* weight_y = weight_z + fy * fw * PACK_UNIT;
                for (fx = 0; fx < fw; ++fx) {
                    const float* weight_x = weight_y + PACK_UNIT * fx;
                    const float* src_x    = src_y + fx * dilateX_step;
                    dstValue = _mm512_fmadd_ps(_mm512_loadu_ps(src_x), _mm512_loadu_ps(weight_x), dstValue);
                }
            }
            _mm512_storeu_ps(dst_x, dstValue);
        }
    }
}

static MNNBinaryExecute _AVX512_MNNSelectBinaryFunctionForFloat(int opType) {
    auto vecF = MNN::selectVector<Vec16, 16, float>(opType);
    if (nullptr != vecF) {
        return vecF;
    }
    return MNN::MNNGetCoreFunctions()->MNNSelectBinaryFunctionForFloat(opType);
}

void _AVX512_MNNScaleAndAddBias(float* dst, const float* src, const float* bias, const float* alpha, size_t planeNumber,
                        size_t biasNumber) {
    for (int z = 0; z < biasNumber; ++z) {
        float* dstZ         = dst + planeNumber * PACK_UNIT * z;
        const float* srcZ   = src + planeNumber * PACK_UNIT * z;
        auto biasZ = _mm512_loadu_ps(bias + PACK_UNIT * z);
        auto alphaZ = _mm512_loadu_ps(alpha + PACK_UNIT * z);
        for (int p = 0; p < planeNumber; ++p) {
            float* dstX       = dstZ + PACK_UNIT * p;
            const float* srcX = srcZ + PACK_UNIT * p;
            _mm512_storeu_ps(dstX, _mm512_fmadd_ps(_mm512_loadu_ps(srcX), alphaZ, biasZ));
        }
    }
}

void _AVX512_MNNDeconvRunForUnitDepthWise(const float* dst, float* src, const float* weight, size_t fw, size_t fh,
                                  size_t weight_y_step, size_t dilateX_step, size_t dilateY_step) {
    int fx, fy;
    float* src_z          = src;
    const float* weight_z = weight;
    Vec16 dstV             = Vec16::load(dst);
    for (fy = 0; fy < fh; ++fy) {
        float* src_y          = src_z + fy * dilateY_step;
        const float* weight_y = weight_z + fy * weight_y_step;
        for (fx = 0; fx < fw; ++fx) {
            Vec16 weight_x = Vec16::load(weight_y + PACK_UNIT * fx);
            Vec16 src_x    = Vec16::load(src_y + fx * dilateX_step);
            Vec16::save(src_y + fx * dilateX_step, src_x + weight_x * dstV);
        }
    }
}
void _AVX512_MNNDeconvRunForLineDepthwise(const float* dst, float* src, const float* weight, size_t width, size_t src_w_setup,
                                  size_t fw, size_t fh, size_t dilateX_step, size_t dilateY_step) {
    int dx;
    for (dx = 0; dx < width; ++dx) {
        const float* dst_x = dst + dx * PACK_UNIT;
        float* src_dx      = src + src_w_setup * dx;
        _AVX512_MNNDeconvRunForUnitDepthWise(dst_x, src_dx, weight, fw, fh, fw * PACK_UNIT, dilateX_step, dilateY_step);
    }
}

void _AVX512_MNNGridSampleComputeCord(float* dst, const float* src, size_t inH, size_t inW, size_t outH, size_t outW, size_t stride, bool alignCorners) {
    __m512 zero = _mm512_setzero_ps();
    __m512 one = _mm512_set1_ps(1);
    __m512 half = _mm512_set1_ps(0.5f);
    __m512 a = alignCorners ? one : zero;
    __m512 b = alignCorners ? zero : one;
    __m512 inW_sub_a = _mm512_sub_ps(_mm512_set1_ps(inW), a);
    __m512 inH_sub_a = _mm512_sub_ps(_mm512_set1_ps(inH), a);

    int area = outH * outW;
    int areaC4 = area / PACK_UNIT;
    int areaRemain = area - areaC4 * PACK_UNIT;
    for (int i = 0; i < areaC4; ++i) {
        __m512 grid0 = _mm512_loadu_ps(src);
        __m512 grid1 = _mm512_loadu_ps(src + PACK_UNIT);
        __m512 x = _mm512_shuffle_ps(grid0, grid1, 0x88);
        __m512 y = _mm512_shuffle_ps(grid0, grid1, 0xdd);
        __m512 cord_x = _mm512_mul_ps(half, _mm512_sub_ps(_mm512_mul_ps(_mm512_add_ps(one, x), inW_sub_a), b));
        __m512 cord_y = _mm512_mul_ps(half, _mm512_sub_ps(_mm512_mul_ps(_mm512_add_ps(one, y), inH_sub_a), b));
        __m512 cord0 = _mm512_unpacklo_ps(cord_x, cord_y);
        __m512 cord1 = _mm512_unpackhi_ps(cord_x, cord_y);

        _mm512_storeu_ps(dst, cord0);
        _mm512_storeu_ps(dst + PACK_UNIT, cord1);

        src += PACK_UNIT * 2;
        dst += PACK_UNIT * 2;
    }

    if (areaRemain > 0) {
        __mmask16 mask = 0xffff;
        if (areaRemain > PACK_UNIT / 2) {
            int shift = areaRemain * 2 - PACK_UNIT;
            mask = (1 << shift) - 1;
            __m512 grid0 = _mm512_loadu_ps(src);
            __m512 grid1 = _mm512_maskz_loadu_ps(mask, src + PACK_UNIT);
            __m512 x = _mm512_shuffle_ps(grid0, grid1, 0x88);
            __m512 y = _mm512_shuffle_ps(grid0, grid1, 0xdd);
            __m512 cord_x = _mm512_mul_ps(half, _mm512_sub_ps(_mm512_mul_ps(_mm512_add_ps(one, x), inW_sub_a), b));
            __m512 cord_y = _mm512_mul_ps(half, _mm512_sub_ps(_mm512_mul_ps(_mm512_add_ps(one, y), inH_sub_a), b));
            __m512 cord0 = _mm512_unpacklo_ps(cord_x, cord_y);
            __m512 cord1 = _mm512_unpackhi_ps(cord_x, cord_y);

            _mm512_storeu_ps(dst, cord0);
            _mm512_mask_storeu_ps(dst + PACK_UNIT, mask, cord1);
        } else {
            int shift = areaRemain * 2;
            mask = (1 << shift) - 1;
            __m512 grid0 = _mm512_maskz_loadu_ps(mask, src);
            __m512 grid1 = zero;
            __m512 x = _mm512_shuffle_ps(grid0, grid1, 0x88);
            __m512 y = _mm512_shuffle_ps(grid0, grid1, 0xdd);
            __m512 cord_x = _mm512_mul_ps(half, _mm512_sub_ps(_mm512_mul_ps(_mm512_add_ps(one, x), inW_sub_a), b));
            __m512 cord_y = _mm512_mul_ps(half, _mm512_sub_ps(_mm512_mul_ps(_mm512_add_ps(one, y), inH_sub_a), b));
            __m512 cord0 = _mm512_unpacklo_ps(cord_x, cord_y);

            _mm512_mask_storeu_ps(dst, mask, cord0);
        }
    }
}

static size_t _AVX512_MNNGridSampleComputeOffset(int h, int w, int height, int width, bool padMode) {
    if (padMode == true) { //padMode == BorderMode_ZEROS
        if (h < 0 || h >= height || w < 0 || w >= width) {
            return -1;
        }
    } else {
        // Clearly, CLAMP is the right way to go for GridSamplePaddingMode_BORDER
        // For GridSamplePaddingMode_REFLECTION, since we have reflected the values into (-1, 1),
        // the leftover reflections degrade to GridSamplePaddingMode_BORDER
        h = h < 0 ? 0 : ( h > (height - 1) ? (height - 1) : h);
        w = w < 0 ? 0 : ( w > (width - 1) ? (width - 1) : w);
    }
    return h * width * PACK_UNIT + w * PACK_UNIT;
}

void _AVX512_MNNGridSampleInterp(float* outputPtr, const float* inputPtr, const float* cordPtr, size_t inH, size_t inW, size_t outW, size_t channelCUnit, size_t inOffset, size_t outOffset, bool sampleMode, bool padMode) {
    for (auto ow = 0; ow < outW; ++ow) {
        auto w = cordPtr[2 * ow + 0];
        auto h = cordPtr[2 * ow + 1];
        __m512 interp;

        if (sampleMode == true) { //sampleMode == SampleMode_NEAREST
            int nh = ::floor(h + 0.5f);
            int nw = ::floor(w + 0.5f);
            size_t ns = _AVX512_MNNGridSampleComputeOffset(nh, nw, inH, inW, padMode);
            for (int k = 0; k < channelCUnit; ++k) {
                interp = ns == -1 ? _mm512_set1_ps(0.f) : _mm512_loadu_ps(inputPtr + k * inOffset + ns);
                _mm512_storeu_ps(outputPtr + k * outOffset + PACK_UNIT * ow, interp);
            }
        } else { //sampleMode == GridSampleMode_BILINEAR
            int w0_h = ::floor(h);
            int w0_w = ::floor(w);
            int w1_h = ::ceil(h);
            int w1_w = ::ceil(w);
            auto oneV = _mm512_set1_ps(1.0f);

            auto f0 = _mm512_set1_ps((float)w1_w - w);
            auto f1 = _mm512_sub_ps(oneV, f0);
            auto h0 = _mm512_set1_ps((float)w1_h - h);
            auto h1 = _mm512_sub_ps(oneV, h0);

            size_t s00 = _AVX512_MNNGridSampleComputeOffset(w0_h, w0_w, inH, inW, padMode);
            size_t s01 = _AVX512_MNNGridSampleComputeOffset(w0_h, w1_w, inH, inW, padMode);
            size_t s10 = _AVX512_MNNGridSampleComputeOffset(w1_h, w0_w, inH, inW, padMode);
            size_t s11 = _AVX512_MNNGridSampleComputeOffset(w1_h, w1_w, inH, inW, padMode);

            for (int k = 0; k < channelCUnit; ++k) {
                __m512 i00 = s00 == -1 ? _mm512_setzero_ps() : _mm512_loadu_ps(inputPtr + k * inOffset + s00);
                __m512 i01 = s01 == -1 ? _mm512_setzero_ps() : _mm512_loadu_ps(inputPtr + k * inOffset + s01);
                __m512 i10 = s10 == -1 ? _mm512_setzero_ps() : _mm512_loadu_ps(inputPtr + k * inOffset + s10);
                __m512 i11 = s11 == -1 ? _mm512_setzero_ps() : _mm512_loadu_ps(inputPtr + k * inOffset + s11);

                __m512 i0 = _mm512_add_ps(_mm512_mul_ps(i00, f0), _mm512_mul_ps(i01, f1));
                __m512 i1 = _mm512_add_ps(_mm512_mul_ps(i10, f0), _mm512_mul_ps(i11, f1));

                interp = _mm512_add_ps(_mm512_mul_ps(i0, h0), _mm512_mul_ps(i1, h1));
                _mm512_storeu_ps(outputPtr + k * outOffset + PACK_UNIT * ow, interp);
            }
        }
    }
}

void _AVX512_MNNRoiPoolingMax(float* dst, const float* src, int hLen, int wLen, int iw) {
    Vec16 max = Vec16(-FLT_MAX);
    for (int h = 0; h < hLen; h++, src += iw * PACK_UNIT) {
        for (int w = 0; w < wLen; w++) {
            Vec16 in = Vec16::load(src + w * PACK_UNIT);
            max = Vec16::max(max, in);
        }
    }
    Vec16::save(dst, max);
}

void _AVX512_MNNRoiAlignMax(float* dst, const float* src, const std::vector<std::vector<int>> &vecPos, const std::vector<std::vector<float>> &vecArea, int samplingRatioArea, int pooledHeight, int pooledWidth) {
    for (int h = 0; h < pooledHeight; ++h, dst += pooledWidth * PACK_UNIT) {
        int preCalcIdx = h * pooledWidth * samplingRatioArea;
        for (int w = 0; w < pooledWidth; ++w) {
            Vec16 res = Vec16(-FLT_MAX);
            for (int i = 0; i < samplingRatioArea; ++i) {
                const std::vector<int>& pos    = vecPos[preCalcIdx];
                const std::vector<float>& area = vecArea[preCalcIdx];

                Vec16 val0 = Vec16::load(src + pos[0] * PACK_UNIT);
                Vec16 val1 = Vec16::load(src + pos[1] * PACK_UNIT);
                Vec16 val2 = Vec16::load(src + pos[2] * PACK_UNIT);
                Vec16 val3 = Vec16::load(src + pos[3] * PACK_UNIT);
                Vec16 mla  = val0 * area[0];
                mla       = Vec16::fma(mla, val1, area[1]);
                mla       = Vec16::fma(mla, val2, area[2]);
                mla       = Vec16::fma(mla, val3, area[3]);
                res       = Vec16::max(res, mla);
                preCalcIdx++;
            }
            Vec16::save(dst + w * PACK_UNIT, res);
        }
    }
}

void _AVX512_MNNRoiAlignAvg(float* dst, const float* src, const std::vector<std::vector<int>> &vecPos, const std::vector<std::vector<float>> &vecArea, int samplingRatioArea, int pooledHeight, int pooledWidth) {
    float invSamplingCnt = 1.f / samplingRatioArea;
    for (int h = 0; h < pooledHeight; ++h, dst += pooledWidth * PACK_UNIT) {
        int preCalcIdx = h * pooledWidth * samplingRatioArea;
        for (int w = 0; w < pooledWidth; ++w) {
            Vec16 res = Vec16(0.f);
            for (int i = 0; i < samplingRatioArea; ++i) {
                const std::vector<int>& pos    = vecPos[preCalcIdx];
                const std::vector<float>& area = vecArea[preCalcIdx];

                Vec16 val0 = Vec16::load(src + pos[0] * PACK_UNIT);
                Vec16 val1 = Vec16::load(src + pos[1] * PACK_UNIT);
                Vec16 val2 = Vec16::load(src + pos[2] * PACK_UNIT);
                Vec16 val3 = Vec16::load(src + pos[3] * PACK_UNIT);
                Vec16 mla  = val0 * area[0];
                mla       = Vec16::fma(mla, val1, area[1]);
                mla       = Vec16::fma(mla, val2, area[2]);
                mla       = Vec16::fma(mla, val3, area[3]);
                res       += mla;
                preCalcIdx++;
            }
            res = res * invSamplingCnt;
            Vec16::save(dst + w * PACK_UNIT, res);
        }
    }
}

void _AVX512_MNNMatrixAdd(float* C, const float* A, const float* B, size_t widthC4, size_t cStride, size_t aStride,
                       size_t bStride, size_t height) {
    for (int y = 0; y < height; ++y) {
        auto a = A + aStride * y;
        auto b = B + bStride * y;
        auto c = C + cStride * y;
        for (int x = 0; x < widthC4; ++x) {
            _mm512_storeu_ps(c + PACK_UNIT * x, _mm512_add_ps(_mm512_loadu_ps(b + PACK_UNIT * x), _mm512_loadu_ps(a + PACK_UNIT * x)));
        }
    }
}

void _AVX512_MNNStrassenMergeCFunction(float* c11, float* c12, float* c21, float* c22, float* xAddr, size_t cStride, size_t eSub, size_t hSub) {
    const int unit = PACK_UNIT;
    for (int y=0; y<hSub; ++y) {
        auto c11Y = c11 + y * cStride;
        auto c12Y = c12 + y * cStride;
        auto c22Y = c22 + y * cStride;
        auto c21Y = c21 + y * cStride;
        auto xY = xAddr + y * eSub * unit;
        for (int x=0; x<eSub; ++x) {
            auto xv = _mm512_loadu_ps(xY + unit*x);
            auto c21v = _mm512_loadu_ps(c21Y + unit*x);
            auto c11v = _mm512_loadu_ps(c11Y + unit*x);
            auto c22v = _mm512_loadu_ps(c22Y + unit*x);
            auto c12v = _mm512_loadu_ps(c12Y + unit*x);
            c12v = _mm512_add_ps(c12v, xv);
            c21v = _mm512_add_ps(c12v, c21v);
            c12v = _mm512_add_ps(c22v, c12v);
            c22v = _mm512_add_ps(c22v, c21v);
            c12v = _mm512_add_ps(c11v, c12v);
            _mm512_storeu_ps(c12Y + unit*x, c12v);
            _mm512_storeu_ps(c22Y + unit*x, c22v);
            _mm512_storeu_ps(c21Y + unit*x, c21v);
        }
    }
}

void _AVX512_MNNMatrixSub(float* C, const float* A, const float* B, size_t widthC4, size_t cStride, size_t aStride,
                       size_t bStride, size_t height) {
    for (int y = 0; y < height; ++y) {
        auto a = A + aStride * y;
        auto b = B + bStride * y;
        auto c = C + cStride * y;
        for (int x = 0; x < widthC4; ++x) {
            _mm512_storeu_ps(c + PACK_UNIT * x, _mm512_sub_ps(_mm512_loadu_ps(a + PACK_UNIT * x), _mm512_loadu_ps(b + PACK_UNIT * x)));
        }
    }
}

void _AVX512_MNNMultiAndDestTransformCommon23(float **cacheLine, const float *weigth, float *dest, int cacheLineSize, int ow, const float* bias, const float* parameter) {
    int unit = ow / 2;
    MNN_ASSERT(cacheLineSize >= 1);
    auto biasF = Vec16::load(bias);
    auto minF = Vec16(parameter[2]);
    auto maxF = Vec16(parameter[3]);
    auto SRC_TILE_UNIT = 4 * PACK_UNIT;
    auto DST_TILE_UNIT = 2 * PACK_UNIT;
    for (int x = 0; x < unit; ++x) {
        auto offset = SRC_TILE_UNIT * x;
        int i = 0;
        Vec16 m0     = Vec16::load(weigth + i * SRC_TILE_UNIT + PACK_UNIT * 0) * Vec16::load(cacheLine[i] + offset + PACK_UNIT * 0);
        Vec16 m1     = Vec16::load(weigth + i * SRC_TILE_UNIT + PACK_UNIT * 1) * Vec16::load(cacheLine[i] + offset + PACK_UNIT * 1);
        Vec16 m2     = Vec16::load(weigth + i * SRC_TILE_UNIT + PACK_UNIT * 2) * Vec16::load(cacheLine[i] + offset + PACK_UNIT * 2);
        Vec16 m3     = Vec16::load(weigth + i * SRC_TILE_UNIT + PACK_UNIT * 3) * Vec16::load(cacheLine[i] + offset + PACK_UNIT * 3);

        for (i = 1; i < cacheLineSize; ++i) {
            m0 = m0 + Vec16::load(weigth + i * SRC_TILE_UNIT + PACK_UNIT * 0) * Vec16::load(cacheLine[i] + offset + PACK_UNIT * 0);
            m1 = m1 + Vec16::load(weigth + i * SRC_TILE_UNIT + PACK_UNIT * 1) * Vec16::load(cacheLine[i] + offset + PACK_UNIT * 1);
            m2 = m2 + Vec16::load(weigth + i * SRC_TILE_UNIT + PACK_UNIT * 2) * Vec16::load(cacheLine[i] + offset + PACK_UNIT * 2);
            m3 = m3 + Vec16::load(weigth + i * SRC_TILE_UNIT + PACK_UNIT * 3) * Vec16::load(cacheLine[i] + offset + PACK_UNIT * 3);
        }
        auto o0 = m0 + m1 + m2 + biasF;
        auto o1 = m1 - m2 + m3 + biasF;
        o0 = Vec16::min(maxF, o0);
        o1 = Vec16::min(maxF, o1);
        o0 = Vec16::max(minF, o0);
        o1 = Vec16::max(minF, o1);

        Vec16::save(dest + DST_TILE_UNIT * x + 0 * PACK_UNIT, o0);
        Vec16::save(dest + DST_TILE_UNIT * x + 1 * PACK_UNIT, o1);
    }
    if (unit * 2 < ow) {
        auto offset = SRC_TILE_UNIT * unit;
        int i = 0;
        Vec16 m0     = Vec16::load(weigth + i * SRC_TILE_UNIT + PACK_UNIT * 0) * Vec16::load(cacheLine[i] + offset + PACK_UNIT * 0);
        Vec16 m1     = Vec16::load(weigth + i * SRC_TILE_UNIT + PACK_UNIT * 1) * Vec16::load(cacheLine[i] + offset + PACK_UNIT * 1);
        Vec16 m2     = Vec16::load(weigth + i * SRC_TILE_UNIT + PACK_UNIT * 2) * Vec16::load(cacheLine[i] + offset + PACK_UNIT * 2);

        for (i = 1; i < cacheLineSize; ++i) {
            m0 = m0 + Vec16::load(weigth + i * SRC_TILE_UNIT + PACK_UNIT * 0) * Vec16::load(cacheLine[i] + offset + PACK_UNIT * 0);
            m1 = m1 + Vec16::load(weigth + i * SRC_TILE_UNIT + PACK_UNIT * 1) * Vec16::load(cacheLine[i] + offset + PACK_UNIT * 1);
            m2 = m2 + Vec16::load(weigth + i * SRC_TILE_UNIT + PACK_UNIT * 2) * Vec16::load(cacheLine[i] + offset + PACK_UNIT * 2);
        }
        auto o0 = m0 + m1 + m2 + biasF;
        o0 = Vec16::min(maxF, o0);
        o0 = Vec16::max(minF, o0);
        Vec16::save(dest + DST_TILE_UNIT * unit, o0);
    }
}
static void _AVX512_MNNConvDwF23SourceTransUnit(const float *source, float *dest, size_t unit) {
    if (unit <= 0) {
        return;
    }
    Vec16 v0 = Vec16::load(source + PACK_UNIT * 0);
    Vec16 v1 = Vec16::load(source + PACK_UNIT * 1);
    Vec16 v2;
    Vec16 v3;
    source += 2 * PACK_UNIT;

    for (int x = 0; x < unit; ++x) {
        v2 = Vec16::load(source + 0 * PACK_UNIT);
        v3 = Vec16::load(source + 1 * PACK_UNIT);
        auto m0 = v0 - v2;
        auto m1 = v1 + v2;
        auto m2 = v2 - v1;
        auto m3 = v3 - v1;

        Vec16::save(dest + PACK_UNIT * 0, m0);
        Vec16::save(dest + PACK_UNIT * 1, m1);
        Vec16::save(dest + PACK_UNIT * 2, m2);
        Vec16::save(dest + PACK_UNIT * 3, m3);

        source += (2 * PACK_UNIT);
        dest += (4 * PACK_UNIT);

        v0 = v2;
        v1 = v3;
    }
}

void _AVX512_MNNSourceTransformCommonF23(const float *source, float *dest, int unit, int iw, int pad, int su, int eu) {
    for (int x = 0; x < su; ++x) {
        auto dstX = dest + 4 * PACK_UNIT * x;
        auto sx   = x * 2 - (int)pad;
        auto ex   = sx + 4;

        auto clampSx = std::max(sx, 0);
        auto clampEx = std::min(ex, (int)iw);

        Vec16 v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int i = clampSx; i < clampEx; ++i) {
            v[i - sx] = Vec16::load(source + PACK_UNIT * i);
        }
        auto m0 = v[0] - v[2];
        auto m1 = v[1] + v[2];
        auto m2 = v[2] - v[1];
        auto m3 = v[3] - v[1];

        Vec16::save(dstX + PACK_UNIT * 0, m0);
        Vec16::save(dstX + PACK_UNIT * 1, m1);
        Vec16::save(dstX + PACK_UNIT * 2, m2);
        Vec16::save(dstX + PACK_UNIT * 3, m3);
    }
    _AVX512_MNNConvDwF23SourceTransUnit(source + PACK_UNIT * (su * 2 - pad), dest + PACK_UNIT * 4 * su, eu - su);

    for (int x = eu; x < unit; ++x) {
        auto dstX = dest + PACK_UNIT * 4 * x;
        auto sx   = x * 2 - (int)pad;
        auto ex   = sx + 4;

        auto clampSx = std::max(sx, 0);
        auto clampEx = std::min(ex, (int)iw);

        Vec16 v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int i = clampSx; i < clampEx; ++i) {
            v[i - sx] = Vec16::load(source + PACK_UNIT * i);
        }
        auto m0 = v[0] - v[2];
        auto m1 = v[1] + v[2];
        auto m2 = v[2] - v[1];
        auto m3 = v[3] - v[1];

        Vec16::save(dstX + PACK_UNIT * 0, m0);
        Vec16::save(dstX + PACK_UNIT * 1, m1);
        Vec16::save(dstX + PACK_UNIT * 2, m2);
        Vec16::save(dstX + PACK_UNIT * 3, m3);
    }
}

void _AVX512_MNNConvDwF23MulTransUnit(float **cacheLine, const float *weigth, float *dest, size_t ow, const float* bias, const float* parameter) {
    int unit = ow / 2;
    auto SRC_TILE_UNIT = 4 * PACK_UNIT;
    auto DST_TILE_UNIT = 2 * PACK_UNIT;

    auto w00 = _mm512_loadu_ps(weigth + 0 * SRC_TILE_UNIT + PACK_UNIT * 0);
    auto w01 = _mm512_loadu_ps(weigth + 0 * SRC_TILE_UNIT + PACK_UNIT * 1);
    auto w02 = _mm512_loadu_ps(weigth + 0 * SRC_TILE_UNIT + PACK_UNIT * 2);
    auto w03 = _mm512_loadu_ps(weigth + 0 * SRC_TILE_UNIT + PACK_UNIT * 3);
    auto w10 = _mm512_loadu_ps(weigth + 1 * SRC_TILE_UNIT + PACK_UNIT * 0);
    auto w11 = _mm512_loadu_ps(weigth + 1 * SRC_TILE_UNIT + PACK_UNIT * 1);
    auto w12 = _mm512_loadu_ps(weigth + 1 * SRC_TILE_UNIT + PACK_UNIT * 2);
    auto w13 = _mm512_loadu_ps(weigth + 1 * SRC_TILE_UNIT + PACK_UNIT * 3);
    auto w20 = _mm512_loadu_ps(weigth + 2 * SRC_TILE_UNIT + PACK_UNIT * 0);
    auto w21 = _mm512_loadu_ps(weigth + 2 * SRC_TILE_UNIT + PACK_UNIT * 1);
    auto w22 = _mm512_loadu_ps(weigth + 2 * SRC_TILE_UNIT + PACK_UNIT * 2);
    auto w23 = _mm512_loadu_ps(weigth + 2 * SRC_TILE_UNIT + PACK_UNIT * 3);
    auto biasF = _mm512_loadu_ps(bias);
    auto minF = _mm512_broadcastss_ps(_mm_load_ss(parameter + 2));
    auto maxF = _mm512_broadcastss_ps(_mm_load_ss(parameter + 3));

    for (int x = 0; x < unit; ++x) {
        auto offset = PACK_UNIT * 4 * x;
        int i = 0;
        auto m0     = _mm512_mul_ps(w00, _mm512_loadu_ps(cacheLine[0] + offset + PACK_UNIT * 0));
        auto m1     = _mm512_mul_ps(w01, _mm512_loadu_ps(cacheLine[0] + offset + PACK_UNIT * 1));
        auto m2     = _mm512_mul_ps(w02, _mm512_loadu_ps(cacheLine[0] + offset + PACK_UNIT * 2));
        auto m3     = _mm512_mul_ps(w03, _mm512_loadu_ps(cacheLine[0] + offset + PACK_UNIT * 3));

        m0 = _mm512_fmadd_ps(w10, _mm512_loadu_ps(cacheLine[1] + offset + PACK_UNIT * 0), m0);
        m1 = _mm512_fmadd_ps(w11, _mm512_loadu_ps(cacheLine[1] + offset + PACK_UNIT * 1), m1);
        m2 = _mm512_fmadd_ps(w12, _mm512_loadu_ps(cacheLine[1] + offset + PACK_UNIT * 2), m2);
        m3 = _mm512_fmadd_ps(w13, _mm512_loadu_ps(cacheLine[1] + offset + PACK_UNIT * 3), m3);

        m0 = _mm512_fmadd_ps(w20, _mm512_loadu_ps(cacheLine[2] + offset + PACK_UNIT * 0), m0);
        m1 = _mm512_fmadd_ps(w21, _mm512_loadu_ps(cacheLine[2] + offset + PACK_UNIT * 1), m1);
        m2 = _mm512_fmadd_ps(w22, _mm512_loadu_ps(cacheLine[2] + offset + PACK_UNIT * 2), m2);
        m3 = _mm512_fmadd_ps(w23, _mm512_loadu_ps(cacheLine[2] + offset + PACK_UNIT * 3), m3);

        auto o0 = _mm512_add_ps(_mm512_add_ps(m0, _mm512_add_ps(m1, m2)), biasF);
        auto o1 = _mm512_add_ps(_mm512_add_ps(m3, _mm512_sub_ps(m1, m2)), biasF);
        o0 = _mm512_min_ps(maxF, o0);
        o1 = _mm512_min_ps(maxF, o1);
        o0 = _mm512_max_ps(minF, o0);
        o1 = _mm512_max_ps(minF, o1);
        _mm512_storeu_ps(dest + DST_TILE_UNIT * x + 0 * PACK_UNIT, o0);
        _mm512_storeu_ps(dest + DST_TILE_UNIT * x + 1 * PACK_UNIT, o1);
    }
    if (unit * 2 < ow) {
        auto offset = PACK_UNIT * 4 * unit;
        auto m0     = _mm512_mul_ps(w00, _mm512_loadu_ps(cacheLine[0] + offset + PACK_UNIT * 0));
        auto m1     = _mm512_mul_ps(w01, _mm512_loadu_ps(cacheLine[0] + offset + PACK_UNIT * 1));
        auto m2     = _mm512_mul_ps(w02, _mm512_loadu_ps(cacheLine[0] + offset + PACK_UNIT * 2));

        m0 = _mm512_fmadd_ps(w10, _mm512_loadu_ps(cacheLine[1] + offset + PACK_UNIT * 0), m0);
        m1 = _mm512_fmadd_ps(w11, _mm512_loadu_ps(cacheLine[1] + offset + PACK_UNIT * 1), m1);
        m2 = _mm512_fmadd_ps(w12, _mm512_loadu_ps(cacheLine[1] + offset + PACK_UNIT * 2), m2);

        m0 = _mm512_fmadd_ps(w20, _mm512_loadu_ps(cacheLine[2] + offset + PACK_UNIT * 0), m0);
        m1 = _mm512_fmadd_ps(w21, _mm512_loadu_ps(cacheLine[2] + offset + PACK_UNIT * 1), m1);
        m2 = _mm512_fmadd_ps(w22, _mm512_loadu_ps(cacheLine[2] + offset + PACK_UNIT * 2), m2);

        auto o0 = _mm512_add_ps(_mm512_add_ps(m0, _mm512_add_ps(m1, m2)), biasF);
        o0 = _mm512_min_ps(maxF, o0);
        o0 = _mm512_max_ps(minF, o0);
        _mm512_storeu_ps(dest + DST_TILE_UNIT * unit, o0);
    }
}
static void _8BitcopyWithStrideC4(uint8_t* dstO, const uint8_t* srcO, int size, int stride, int ds) {
    auto src = (float*)srcO;
    auto dst = (float*)dstO;
    for (int i=0; i<size; ++i) {
        _mm256_storeu_ps(dst, _mm256_loadu_ps(src));
        src+= (8 * stride);
        dst+= (8 * ds);
    }
}
static void _16BitcopyWithStrideC4(uint8_t* dstO, const uint8_t* srcO, int size, int stride, int ds) {
    auto src = (float*)srcO;
    auto dst = (float*)dstO;
    for (int i=0; i<size; ++i) {
        _mm512_storeu_ps(dst, _mm512_loadu_ps(src));
        src+= (16 * stride);
        dst+= (16 * ds);
    }
}
static MNNCopyWithStride _selectBlit(int bytesC4) {
    if (64 == bytesC4) {
        return _16BitcopyWithStrideC4;
    }
    if (32 == bytesC4) {
        return _8BitcopyWithStrideC4;
    }
    return nullptr;
}

static void _AVX512_MNNAdjustOptimalSparseKernel(int& sparseBlockOC, MNN::CoreFunctions::MNNPackedSparseMatMul& packedSparseMatMul) {
    if(sparseBlockOC == 8) {
        packedSparseMatMul = _AVX512_MNNPackedSparseMatMulEpx8;
        return;
    } else if(sparseBlockOC % 8 == 0) {
        // MNN_PRINT("avx512 downgrade sparse from:%d, ",sparseBlockOC);
        sparseBlockOC = 8;
        packedSparseMatMul = _AVX512_MNNPackedSparseMatMulEpx8;
        // MNN_PRINT(" to:%d\n",sparseBlockOC);
        return;
    }
    else if(sparseBlockOC == 4) {
        packedSparseMatMul = _AVX512_MNNPackedSparseMatMulEpx4;
        return;
    } else if(sparseBlockOC % 4 == 0) {
        // MNN_PRINT("avx512 downgrade sparse from:%d, ",sparseBlockOC);
        sparseBlockOC = 4;
        packedSparseMatMul = _AVX512_MNNPackedSparseMatMulEpx4;
        // MNN_PRINT(" to:%d\n",sparseBlockOC);
        return;
    } else {
        sparseBlockOC = 1;
        packedSparseMatMul = _AVX512_MNNPackedSparseMatMulEpx1;
        return;
    }
}


void _AVX512_ExtraInit(void* functions) {
    auto coreFunction = static_cast<MNN::CoreFunctions*>(functions);
    coreFunction->MNNSelectBlitFunction = _selectBlit;
    coreFunction->MNNPoolingAvg = (decltype(coreFunction->MNNPoolingAvg))(MNN::poolingAvg<float, Vec16, 16>);
    // Set min value as 1 << 24
    coreFunction->MNNPoolingMax = (decltype(coreFunction->MNNPoolingMax))(MNN::poolingMax<float, Vec16, 16, -16777216>);
    coreFunction->MNNPoolingMaxWithRedice = (decltype(coreFunction->MNNPoolingMaxWithRedice))(MNN::poolingMaxWithRedice<float, -16777216>);
    coreFunction->MNNSelectBinaryFunctionForFloat = _AVX512_MNNSelectBinaryFunctionForFloat;
    coreFunction->MNNCopyC4WithStride = _AVX512_MNNCopyC4WithStride;
    coreFunction->MNNAddC4WithStride = _AVX512_MNNAddC4WithStride;
    coreFunction->MNNScaleAndAddBias = _AVX512_MNNScaleAndAddBias;
    coreFunction->MNNMatrixAdd          = _AVX512_MNNMatrixAdd;
    coreFunction->MNNMatrixSub          = _AVX512_MNNMatrixSub;
    coreFunction->MNNCountMaxMinValue = _AVX512_MNNComputeScaleZeroScalar;
    coreFunction->MNNAbsMax = _AVX512_MNNAbsMaxFP32;

    coreFunction->MNNConvRunForUnitDepthWise = _AVX512_MNNConvRunForUnitDepthWise;
    coreFunction->MNNConvRunForLineDepthwise = _AVX512_MNNConvRunForLineDepthwise;
    coreFunction->MNNAxByClampBroadcastUnit = _AVX512_MNNAxByClampBroadcastUnit;
    coreFunction->MNNStrassenMergeCFunction = _AVX512_MNNStrassenMergeCFunction;
    coreFunction->MNNMultiAndDestTransformCommon23 = _AVX512_MNNMultiAndDestTransformCommon23;
    coreFunction->MNNSourceTransformCommonF23 = _AVX512_MNNSourceTransformCommonF23;
    coreFunction->MNNConvDwF23MulTransUnit = _AVX512_MNNConvDwF23MulTransUnit;
    coreFunction->MNNReluWithSlopeChannel = _AVX512_MNNReluWithSlopeChannel;
    coreFunction->MNNDeconvRunForLineDepthwise = _AVX512_MNNDeconvRunForLineDepthwise;
    coreFunction->MNNDeconvRunForUnitDepthWise = _AVX512_MNNDeconvRunForUnitDepthWise;
    coreFunction->MNNGridSampleComputeCord = _AVX512_MNNGridSampleComputeCord;
    coreFunction->MNNRoiPoolingMax = _AVX512_MNNRoiPoolingMax;
    coreFunction->MNNRoiAlignMax = _AVX512_MNNRoiAlignMax;
    coreFunction->MNNRoiAlignAvg = _AVX512_MNNRoiAlignAvg;
    coreFunction->MNNGridSampleInterp = MNNGridSampleInterp;
    coreFunction->MNNGridSampleInterpGrad = MNNGridSampleInterpGrad;

    coreFunction->MNNGetSparseMatMulPackMode = _AVX512_MNNGetSparseMatMulPackMode;
    coreFunction->MNNAdjustOptimalSparseKernel = _AVX512_MNNAdjustOptimalSparseKernel;

}
