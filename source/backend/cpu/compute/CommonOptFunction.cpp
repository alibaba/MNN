//
//  CommonOptFunction.cpp
//  MNN
//
//  Created by MNN on 2018/09/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CommonOptFunction.h"
#include "ConvOpt.h"
#include "WinogradOptFunction.hpp"
#include "Int8FunctionsOpt.h"
#include "ImageProcessFunction.hpp"
#include <string.h>
#include <algorithm>
#include <cmath>
#include <math.h>
#include "math/Vec.hpp"
#include <vector>
#include "../CPURuntime.hpp"
#include "core/MemoryFormater.h"
// TODO: Find better way to optimize it
#include "../CPUBinary.hpp"
#include "../CPUUnary.hpp"
#include "../CPUPool.hpp"
#define PACK 4
#define FLOAT float
using Vec = MNN::Math::Vec<float, 4>;
#include "../GridSampler.hpp"
#ifdef MNN_LOW_MEMORY
#ifdef __aarch64__
#include "backend/cpu/arm/arm64/low_memory/MNNDynamicQuantFunctions.hpp"
#endif
#endif

#ifndef MNN_USE_SSE
void MNNInt8ToInt16(int16_t* dest, const int8_t* source, size_t count) {
    // Should not be called
    MNN_ASSERT(false);
}
#endif

#ifndef __aarch64__
#ifdef MNN_CPU_WEIGHT_DEQUANT_GEMM
static void _MNNPackedMatMulRemain_int4(float* C, const float* A, const float* fB, size_t eSize, const size_t* parameter, const float* postParameters, const float* bias, int aStride, const float* k, const float* b) {
    auto B = reinterpret_cast<const uint8_t*>(fB);
    auto h = parameter[2];
    auto l = parameter[1];
    auto cStride = parameter[3] / sizeof(float);
    auto hRemain = parameter[4];
    float weightBytes = 0.5; // sizeof(int4_t)
    auto bExtraStride = static_cast<int32_t>(parameter[5] / weightBytes);
    auto bStride = bExtraStride + 4 * l;
    auto hC4 = UP_DIV(h, 4);
    float minValue = -std::numeric_limits<float>().max();
    float maxValue = std::numeric_limits<float>().max();
    if (nullptr != postParameters) {
        minValue = postParameters[2];
        maxValue = postParameters[3];
    }
    int blockId = parameter[6];

    for (int x=0; x<eSize; ++x) {
        auto dst = C + 4 * x;
        auto src = A + x;
        for (int y=0; y<hC4; ++y) {
            auto dstY = dst + y * cStride;
            auto weight = B + y * bStride / 2;
            auto alpha = k + y * 4;
            auto qbias  = b + y * 4;
            float summer[4] = {
                0.0f,
                0.0f,
                0.0f,
                0.0f,
            };
            if (blockId > 0) {
                summer[0] = dstY[0];
                summer[1] = dstY[1];
                summer[2] = dstY[2];
                summer[3] = dstY[3];
            }
            if (nullptr != bias && nullptr != postParameters) {
                for (int v=0; v<4; ++v) {
                    summer[v] += bias[4 * y + v];
                }
            }
            for (int z=0; z<l; ++z) {
                auto aZ = src + z * aStride;
                auto i4wZ = weight + z * 2;
                float wZ[4];
                {
                    auto w01    = i4wZ[0];
                    auto w23    = i4wZ[1];
                    int iw01    = w01;
                    int iw23    = w23;
                    int iw0     = iw01 / 16;
                    int iw1     = iw01 % 16;
                    int iw2     = iw23 / 16;
                    int iw3     = iw23 % 16;
                    wZ[0]       = iw0 * alpha[0] + qbias[0];
                    wZ[1]       = iw1 * alpha[1] + qbias[1];
                    wZ[2]       = iw2 * alpha[2] + qbias[2];
                    wZ[3]       = iw3 * alpha[3] + qbias[3];
                }
                summer[0] += wZ[0] * aZ[0];
                summer[1] += wZ[1] * aZ[0];
                summer[2] += wZ[2] * aZ[0];
                summer[3] += wZ[3] * aZ[0];
            }
            for (int v=0; v<4; ++v) {
                auto dstValue = std::min(summer[v], maxValue);
                dstValue = std::max(dstValue, minValue);
                dstY[v] = dstValue;
            }
        }
    }
}
static void _MNNPackedMatMulRemain_int8(float* C, const float* A, const float* fB, size_t eSize, const size_t* parameter, const float* postParameters, const float* bias, int aStride, const float* k, const float* b) {
    auto B = reinterpret_cast<const int8_t*>(fB);
    auto h = parameter[2];
    auto l = parameter[1];
    auto cStride = parameter[3] / sizeof(float);
    auto hRemain = parameter[4];
    float weightBytes = 1; // sizeof(int8_t)
    auto bExtraStride = static_cast<int32_t>(parameter[5] / weightBytes);
    auto bStride = bExtraStride + 4 * l;
    auto hC4 = UP_DIV(h, 4);
    float minValue = -std::numeric_limits<float>().max();
    float maxValue = std::numeric_limits<float>().max();
    if (nullptr != postParameters) {
        minValue = postParameters[2];
        maxValue = postParameters[3];
    }
    int blockId = parameter[6];

    for (int x=0; x<eSize; ++x) {
        auto dst = C + 4 * x;
        auto src = A + x;
        for (int y=0; y<hC4; ++y) {
            auto dstY = dst + y * cStride;
            auto weight = B + y * bStride;
            auto alpha = k + y * 4;
            auto qbias  = b + y * 4;
            float summer[4] = {
                0.0f,
                0.0f,
                0.0f,
                0.0f,
            };
            if (blockId > 0) {
                summer[0] = dstY[0];
                summer[1] = dstY[1];
                summer[2] = dstY[2];
                summer[3] = dstY[3];
            }
            if (nullptr != bias && nullptr != postParameters) {
                for (int v=0; v<4; ++v) {
                    summer[v] += bias[4 * y + v];
                }
            }
            for (int z=0; z<l; ++z) {
                auto aZ = src + z * aStride;
                auto i8wZ = weight + z * 4;
                float wZ[4];
                {
                    wZ[0]       = i8wZ[0] * alpha[0] + qbias[0];
                    wZ[1]       = i8wZ[1] * alpha[1] + qbias[1];
                    wZ[2]       = i8wZ[2] * alpha[2] + qbias[2];
                    wZ[3]       = i8wZ[3] * alpha[3] + qbias[3];
                }
                summer[0] += wZ[0] * aZ[0];
                summer[1] += wZ[1] * aZ[0];
                summer[2] += wZ[2] * aZ[0];
                summer[3] += wZ[3] * aZ[0];
            }
            for (int v=0; v<4; ++v) {
                auto dstValue = std::min(summer[v], maxValue);
                dstValue = std::max(dstValue, minValue);
                dstY[v] = dstValue;
            }
        }
    }
}
void MNNPackedMatMul_int4(float* C, const float* A, const float* B, const size_t* parameter, const float* postParameters, const float* bias, const float* k, const float* b) {
    _MNNPackedMatMulRemain_int4(C, A, B, 16, parameter, postParameters, bias, 16, k, b);
}
void MNNPackedMatMulRemain_int4(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter, const float* postParameters, const float* bias, const float* k, const float* b) {
    auto aStride = parameter[0] / sizeof(float);
    _MNNPackedMatMulRemain_int4(C, A, B, eSize, parameter, postParameters, bias, aStride, k, b);
}
void MNNPackedMatMul_int8(float* C, const float* A, const float* B, const size_t* parameter, const float* postParameters, const float* bias, const float* k, const float* b) {
    _MNNPackedMatMulRemain_int8(C, A, B, 16, parameter, postParameters, bias, 16, k, b);
}
void MNNPackedMatMulRemain_int8(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter, const float* postParameters, const float* bias, const float* k, const float* b) {
    auto aStride = parameter[0] / sizeof(float);
    _MNNPackedMatMulRemain_int8(C, A, B, eSize, parameter, postParameters, bias, aStride, k, b);
}
#endif // MNN_CPU_WEIGHT_DEQUANT_GEMM

#ifdef MNN_LOW_MEMORY
void MNNQuantScaleFP32(float* absmax, float* quant_scale, float* dequant_scale, size_t thread, size_t batch) {
    for (int i = 0; i < batch; ++i) {
        auto absmaxPtr = absmax + i;
        float absVal = 0.f;
        for (int t = 0; t < thread; ++t) {
            absVal = std::max(absVal, absmaxPtr[t * batch]);
        }
        if (absVal < 1e-7) {
            quant_scale[i] = 1.f;
            dequant_scale[i] = 1.f;
        } else {
            quant_scale[i] = 127.0f / absVal;
            dequant_scale[i] = absVal / 127.0f;
        }
    }
}

void MNNDynamicUpdateConvBiasScale(float* newbias, float* oldbias, float* weightKernelSum, float* inputBias, size_t ocQuad) {
    int ocUp4 = 4 * ocQuad;
    int pack = 4;
    for (int i = 0; i < ocUp4; ++i) {
        newbias[i] = oldbias[i] + weightKernelSum[i] * inputBias[0];
    }
}

#endif // LOW_MEMORY
#endif // not __aarch64__

static void MNNCountMaxMinValue(const float* source, float* minVal, float* maxVal, size_t size) {
#ifndef MNN_USE_NEON
    int pack = 4;
    float max_ = source[0], min_ = source[0];
    for (int i = 1; i < size; ++i) {
        if (max_ < source[i]) {
            max_ = source[i];
        }
        if (min_ > source[i]) {
            min_ = source[i];
        }
    }
    *minVal = min_;
    *maxVal = max_;
#else
    auto sizeDiv4 = size / 4;
    auto remain = size - 4 * sizeDiv4;
    auto srcPtr = source;
    auto max0 = vdupq_n_f32(srcPtr[0]);
    auto min0 = vdupq_n_f32(srcPtr[0]);
    while (sizeDiv4 > 15) {
        sizeDiv4 -= 16;
        auto data0 = vld1q_f32(srcPtr);
        auto data1 = vld1q_f32(srcPtr + 4);
        auto data2 = vld1q_f32(srcPtr + 8);
        auto data3 = vld1q_f32(srcPtr + 12);
        auto data4 = vld1q_f32(srcPtr + 16);
        auto data5 = vld1q_f32(srcPtr + 20);
        auto data6 = vld1q_f32(srcPtr + 24);
        auto data7 = vld1q_f32(srcPtr + 28);
        auto data8 = vld1q_f32(srcPtr + 32);
        auto data9 = vld1q_f32(srcPtr + 36);
        auto data10 = vld1q_f32(srcPtr + 40);
        auto data11 = vld1q_f32(srcPtr + 44);
        auto data12 = vld1q_f32(srcPtr + 48);
        auto data13 = vld1q_f32(srcPtr + 52);
        auto data14 = vld1q_f32(srcPtr + 56);
        auto data15 = vld1q_f32(srcPtr + 60);

        auto lmin0  = vminq_f32(data0, data1);
        auto lmin2  = vminq_f32(data2, data3);
        auto lmin4  = vminq_f32(data4, data5);
        auto lmin6  = vminq_f32(data6, data7);
        auto lmin8  = vminq_f32(data8, data9);
        auto lmin10 = vminq_f32(data10, data11);
        auto lmin12 = vminq_f32(data12, data13);
        auto lmin14 = vminq_f32(data14, data15);

        auto lmax0  = vmaxq_f32(data0, data1);
        auto lmax2  = vmaxq_f32(data2, data3);
        auto lmax4  = vmaxq_f32(data4, data5);
        auto lmax6  = vmaxq_f32(data6, data7);
        auto lmax8  = vmaxq_f32(data8, data9);
        auto lmax10 = vmaxq_f32(data10, data11);
        auto lmax12 = vmaxq_f32(data12, data13);
        auto lmax14 = vmaxq_f32(data14, data15);

        lmin0 = vminq_f32(lmin0, lmin2);
        lmin4 = vminq_f32(lmin4, lmin6);
        lmin8 = vminq_f32(lmin8, lmin10);
        lmin12 = vminq_f32(lmin12, lmin14);

        lmax0 = vmaxq_f32(lmax0, lmax2);
        lmax4 = vmaxq_f32(lmax4, lmax6);
        lmax8 = vmaxq_f32(lmax8, lmax10);
        lmax12 = vmaxq_f32(lmax12, lmax14);

        lmin0 = vminq_f32(lmin0, lmin8);
        lmin4 = vminq_f32(lmin4, lmin12);
        lmax0 = vmaxq_f32(lmax0, lmax8);
        lmax4 = vmaxq_f32(lmax4, lmax12);
        lmin0 = vminq_f32(lmin0, lmin4);
        lmax0 = vmaxq_f32(lmax0, lmax4);

        max0 = vmaxq_f32(max0, lmax0);
        min0 = vminq_f32(min0, lmin0);
        srcPtr += 64;
    }
    if (sizeDiv4 > 7) {
        sizeDiv4 -= 8;
        auto data0 = vld1q_f32(srcPtr);
        auto data1 = vld1q_f32(srcPtr + 4);
        auto data2 = vld1q_f32(srcPtr + 8);
        auto data3 = vld1q_f32(srcPtr + 12);
        auto data4 = vld1q_f32(srcPtr + 16);
        auto data5 = vld1q_f32(srcPtr + 20);
        auto data6 = vld1q_f32(srcPtr + 24);
        auto data7 = vld1q_f32(srcPtr + 28);

        auto lmin0  = vminq_f32(data0, data1);
        auto lmin2  = vminq_f32(data2, data3);
        auto lmin4  = vminq_f32(data4, data5);
        auto lmin6  = vminq_f32(data6, data7);

        auto lmax0  = vmaxq_f32(data0, data1);
        auto lmax2  = vmaxq_f32(data2, data3);
        auto lmax4  = vmaxq_f32(data4, data5);
        auto lmax6  = vmaxq_f32(data6, data7);

        lmin0 = vminq_f32(lmin0, lmin2);
        lmin4 = vminq_f32(lmin4, lmin6);

        lmax0 = vmaxq_f32(lmax0, lmax2);
        lmax4 = vmaxq_f32(lmax4, lmax6);

        lmin0 = vminq_f32(lmin0, lmin4);
        lmax0 = vmaxq_f32(lmax0, lmax4);

        max0 = vmaxq_f32(max0, lmax0);
        min0 = vminq_f32(min0, lmin0);
        srcPtr += 32;
    }
    if (sizeDiv4 > 3) {
        sizeDiv4 -= 4;
        auto data0 = vld1q_f32(srcPtr);
        auto data1 = vld1q_f32(srcPtr + 4);
        auto data2 = vld1q_f32(srcPtr + 8);
        auto data3 = vld1q_f32(srcPtr + 12);

        auto lmin0  = vminq_f32(data0, data1);
        auto lmin2  = vminq_f32(data2, data3);

        auto lmax0  = vmaxq_f32(data0, data1);
        auto lmax2  = vmaxq_f32(data2, data3);

        lmin0 = vminq_f32(lmin0, lmin2);
        lmax0 = vmaxq_f32(lmax0, lmax2);

        max0 = vmaxq_f32(max0, lmax0);
        min0 = vminq_f32(min0, lmin0);
        srcPtr += 16;
    }
    if (sizeDiv4 > 1) {
        sizeDiv4 -= 2;
        auto data0 = vld1q_f32(srcPtr);
        auto data1 = vld1q_f32(srcPtr + 4);

        auto lmin0  = vminq_f32(data0, data1);
        auto lmax0  = vmaxq_f32(data0, data1);

        max0 = vmaxq_f32(max0, lmax0);
        min0 = vminq_f32(min0, lmin0);
        srcPtr += 8;
    }
    if (sizeDiv4 > 0) {
        sizeDiv4--;
        auto data0 = vld1q_f32(srcPtr);
        max0 = vmaxq_f32(max0, data0);
        min0 = vminq_f32(min0, data0);
        srcPtr += 4;
    }
    float temp0[4];
    float temp1[4];
    vst1q_f32(temp0, max0);
    vst1q_f32(temp1, min0);
    auto maxval = temp0[0];
    auto minval = temp1[0];
    for (int i = 1; i < 4; ++i) {
        maxval = ALIMAX(maxval, temp0[i]);
        minval = ALIMIN(minval, temp1[i]);
    }
    while (remain > 0) {
        maxval = ALIMAX(maxval, srcPtr[0]);
        minval = ALIMIN(minval, srcPtr[0]);
        remain--;
        srcPtr += 1;
    }
    minVal[0] = minval;
    maxVal[0] = maxval;
#endif
}

#ifdef MNN_LOW_MEMORY
static void MNNAbsMaxFP32(const float* source, float* absmax, size_t src_depth_quad, size_t realSize, int pack) {
#ifdef __aarch64__
    if (pack == 4) {
        MNNAbsMaxFP32_Pack4(source, absmax, src_depth_quad, realSize, pack);
        return;
    }
    if (pack == 8) {
        MNNAbsMaxFP32_Pack8(source, absmax, src_depth_quad, realSize, pack);
        return;
    }
#endif
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

void MNNDynamicQuantFP32(const float* src, int8_t* dst, const float* scale, size_t src_depth_quad, size_t realSize, int pack, const float* bias = nullptr) {
#ifdef __aarch64__
    if (pack == 4) {
        MNNDynamicQuantFP32_Pack4(src, dst, scale, src_depth_quad, realSize, nullptr, pack);
        return;
    }
    if (pack == 8) {
        MNNDynamicQuantFP32_Pack8(src, dst, scale, src_depth_quad, realSize, nullptr, pack);
        return;
    }
#endif
#ifdef MNN_USE_SSE
    uint8_t* dstPtr = reinterpret_cast<uint8_t*>(dst);
    int offset = 128;
#else
    int8_t* dstPtr = dst;
    int offset = 0;
#endif
    for (int i = 0; i < realSize; ++i) {
        auto scaleVal = scale[i];
        for (int c = 0; c < src_depth_quad; ++c) {
            auto srcZ = src + c * pack * realSize + i * pack;
            auto dstZ = dstPtr + c * pack * realSize + i * pack;
            for (int k = 0; k < pack; ++k) {
                int val = (int)roundf(srcZ[k] * scaleVal);
                dstZ[k] = val + offset;
            }
        }
    }
}

static void MNNAsyQuantFunc(int8_t* dst, const float* src, float* qscale, float* qbias, const size_t* info) {
    // input shape: [kernelsize, blockNum, blockLU, EP, LP]
    auto blockNum = info[0];
    auto EP = info[1];        // real area for data
    auto LP = info[2];        // Innermost data layout, may come from backend's pack or gemmint8 units' SRC_UNIT
    auto DST_XUNIT = info[3]; // backend gemmint8 units
    auto SRC_UNIT = info[4];
    auto kernelsize = info[5];
    auto blockLU = info[6];
    auto stride0 = blockNum * blockLU * EP * LP;
    auto stride1 = blockLU * EP * LP;
    int int8Max = 127;
    int int8Min = -128;
    // qscale&qbias [blockNum, EP]
#ifdef __aarch64__
    if (LP == 4 || LP == 8) {
        for (int k = 0; k < kernelsize; ++k) {
            for (int i = 0; i < blockNum; ++i) {
                if (LP == 4) {
                    MNNDynamicQuantFP32_Pack4(src + k * stride0 + i * stride1, dst + k * stride0 + i * stride1, qscale + i * EP, blockLU, EP, qbias + i * EP, LP);
                }
                if (LP == 8) {
                    MNNDynamicQuantFP32_Pack8(src + k * stride0 + i * stride1, dst + k * stride0 + i * stride1, qscale + i * EP, blockLU, EP, qbias + i * EP, LP);
                }
            }
        }
        return;
    }
#endif
    for (int i = 0; i < EP; ++i) {
        for (int bk = 0; bk < blockNum; ++bk) {
            float quant_scale = qscale[i + bk * EP];
            float quant_bias  = qbias[i + bk * EP];
            for (int n = 0; n < kernelsize; ++n) {
                for (int k = 0; k < blockLU; ++k) {
                    for (int j = 0; j < LP; ++j) {
                        int dataIndx = n * stride0 + bk * stride1 + k * EP * LP + i * LP + j;
                        float data_ = src[dataIndx];
                        int qval = static_cast<int32_t>(roundf(data_ * quant_scale + quant_bias));
#ifdef MNN_USE_SSE
                        ((uint8_t*)dst)[dataIndx] = qval + 128;
#else
                        dst[dataIndx] = ALIMIN(int8Max, ALIMAX(int8Min, qval));
#endif
                    }
                }
            }
        }
    }
}

static void MNNAsyQuantInfo_FP32(float* scale, float* bias, float* qscale, float* qbias, float* dstMin, float* dstMax, const float* src, const size_t* info) {
    auto blockNum = info[0];
    auto plane = info[1];        // real area for data
    auto innerSide = info[2];    // Innermost data layout, may come from backend's pack or gemmint8 units' SRC_UNIT
    auto DST_XUNIT = info[3];
    auto kernelsize = info[5];
    auto blockLU = info[6];
    auto stride0 = blockNum * blockLU * plane * innerSide;
    auto stride1 = blockLU * plane * innerSide;

    if (info[7] == 1) { // scale&bias:[1]
        float maxval, minval;
        MNNCountMaxMinValue(src, &minval, &maxval, kernelsize * stride0);
        if (info[8] == 1 && (maxval -minval) > 1e-7) {
            if (minval > 0.f) {
                minval = 0;
            } else if (maxval < 0.f){
                maxval = 0;
            }
        }
        auto range = maxval - minval;
        if (range <= 1e-7) {
            scale[0] = 0.f;
            qscale[0] = 0.f;
            qbias[0] = 0.f;
            bias[0] = maxval;
        } else {
            qscale[0] = 255.f / range;
            scale[0] = range / 255.f;
            qbias[0] = -minval * 255.f / range - 128.f;
            bias[0] = minval + 128.f * range / 255.f;
        }
        return;
    }

    // input              : [kernelsize, blockNum, blockLU, plane, pack]
    // dequant scale/bias : [EU, blockNum, step], step=ALIMIN(step, EP), EU=UP_DIV(plane, EP)
    // quant scale/bias   : [blockNum, plane]
#ifdef __aarch64__
    if ((DST_XUNIT == 12 || DST_XUNIT == 16) && innerSide == 4) { // Arm82,fp32: SRC_UNIT=4, core->pack=4
        // max,min shape: [blockNum, EP]
        for (int i = 0; i < kernelsize; ++i) {
            MNNLocalMinMaxFP32_Pack4(dstMin, dstMax, src + i * stride0, blockNum, blockLU, plane, innerSide, i);
        }
        // scale, bias
        if (DST_XUNIT == 12) {
            bool success = MNNAsyLocalQuantInfo_EP12_FP32(scale, bias, qscale, qbias, dstMin, dstMax, info);
            if (!success) {
                MNN_ERROR("Call error for:MNNAsyLocalQuantInfo_EP12\n");
                return;
            }
            return;
        }
        if (DST_XUNIT == 16) {
            bool success = MNNAsyLocalQuantInfo_EP16_FP32(scale, bias, qscale, qbias, dstMin, dstMax, info);
            if (!success) {
                MNN_ERROR("Call error for:MNNAsyLocalQuantInfo_EP16_FP32\n");
                return;
            }
            return;
        }
    }
    if (DST_XUNIT == 10) { // Arm86,fp32: SRC_UNIT=8,core->pack=4
        // max,min shape: [blockNum, EP]
        if (innerSide == 4) {
            for (int i = 0; i < kernelsize; ++i) {
                MNNLocalMinMaxFP32_Pack4(dstMin, dstMax, src + i * stride0, blockNum, blockLU, plane, innerSide, i);
            }
        }
        if (innerSide == 8) {
            for (int i = 0; i < kernelsize; ++i) {
                MNNLocalMinMaxFP32_Pack8(dstMin, dstMax, src + i * stride0, blockNum, blockLU, plane, innerSide, i);
            }
        }
        // scale, bias
        bool success = MNNAsyLocalQuantInfo_EP10_FP32(scale, bias, qscale, qbias, dstMin, dstMax, info);
        if (!success) {
            MNN_ERROR("Call error for:MNNAsyLocalQuantInfo_EP10\n");
            return;
        }
        return;
    }
#endif
    // max,min shape: [blockNum, plane]
    for (int i = 0; i < plane; ++i) {
        for (int bk = 0; bk < blockNum; ++bk) {
            auto idx0 = i *innerSide + bk * stride1;
            float max_ = src[idx0];
            float min_ = max_;
            for (int n = 0; n < kernelsize; ++n) {
                for (int k = 0; k < blockLU; ++k) {
                    for (int j = 0; j < innerSide; ++j) {
                        auto dataIndx = idx0 + n * stride0 + k * (plane * innerSide) + j;
                        float data_ = src[dataIndx];
                        max_ = ALIMAX(max_, data_);
                        min_ = ALIMIN(min_, data_);
                    }
                }
            }
            auto sindx = i + bk * plane;
            dstMin[sindx] = min_;
            dstMax[sindx] = max_;
        }
    }
    // scale, bias
    for (int i = 0; i < plane; ++i) {
        auto step = ALIMIN(DST_XUNIT, plane - (i / DST_XUNIT) * DST_XUNIT);
        auto sind0 = (i / DST_XUNIT) * DST_XUNIT * blockNum + (i % DST_XUNIT);
        for (int k = 0; k < blockNum; ++k) {
            auto sind = sind0 + k * step;
            auto qind = i + k * plane;
            auto max_ = dstMax[qind];
            auto min_ = dstMin[qind];
            if (fabs(max_ - min_) < 1e-7) {
                qscale[qind] = 0.f;
                qbias[qind] = 0.f;
                scale[sind] = 0.f;
                bias[sind] = max_;
            } else {
                qscale[qind] = 255.f / (max_ - min_);
                qbias[qind] = roundf(-min_ * 255.f / (max_ - min_)) - 128.0f;
                scale[sind] = (max_ - min_) / 255.f;
#ifndef MNN_USE_SSE
                bias[sind] = min_ + (128.f / 255.f) * (max_ - min_);
#else
                bias[sind] = min_;
#endif
            }
        }
    }
}
#endif // MNN_LOW_MEMORY

static void MNNReorderWeightInt4(uint8_t* dest, const uint8_t* source, int32_t* shape, size_t size, float* kernelsum) {
    MNN_ASSERT(size > 4);
    auto blocknum = shape[0];
    auto hu       = shape[1];
    auto lu       = shape[2];
    auto hp       = shape[3];
    auto lp       = shape[4];
    auto ic       = blocknum * lu * lp;
    auto stride0  = blocknum * hp * lu * lp;
    auto stride1  = lu * hp * lp;
    auto stride2  = hp * lp;
    // [oc,ic]->[hu,blocknum,lu,hp,lp]
    for (int i = 0; i < hu; ++i) {
        for (int k = 0; k < hp; ++k) {
            for (int bl = 0; bl < blocknum; ++bl) {
                for (int j = 0; j < lu; ++j) {
                    int srcindex = (i * hp + k) * ic + bl * (lu * lp) + j * lp;
                    int dstindex = i * stride0 + bl * stride1 + j * stride2 + k * lp;
                    memcpy(dest + dstindex, source + srcindex, lp);
                }
            }
        }
    }
    // [hu,blocknum,lu,hp,lp] address [hp,lp] for int4
    auto inside = lp * hp;
    auto outside = blocknum * hu;
    std::vector<uint8_t> buffer(inside);
    for (int i = 0; i < outside; ++i) {
        std::vector<float> accum(hp, 0);
        for (int k = 0; k < lu; ++k) {
            for (int j = 0; j < inside / 2; ++j) {
                auto w0 = dest[j + (i * lu + k) * inside] >> 4;
                auto w1 = dest[j + (i * lu + k) * inside] & 0x0f;
                auto w2 = dest[(i * lu + k) * inside + j + inside / 2] >> 4;
                auto w3 = dest[(i * lu + k) * inside + j + inside / 2] & 0x0f;
                buffer[2 * j + 0] = w0 * 16 + w2;
                buffer[2 * j + 1] = w1 * 16 + w3;
                // sum
                accum[j / lp] += ((float)w0 + (float)w1);
                accum[(j + inside / 2) / lp] += ((float)w2 + (float)w3);
            }
            memcpy(dest + (i * lu + k) * inside, buffer.data(), inside);
        }
        memcpy(kernelsum + i * hp, accum.data(), hp * sizeof(float));
    }
}
#ifdef __aarch64__
static void MNNReorderWeightInt4Arm86(uint8_t* dest, const uint8_t* source, int32_t* shape, size_t size, float* kernelsum) {
    MNN_ASSERT(size > 4);
    auto blocknum = shape[0];
    auto hu       = shape[1];
    auto lu       = shape[2];
    auto hp       = shape[3];
    auto lp       = shape[4];
    auto ic       = blocknum *lu * lp;
    auto stride0  = blocknum * hp * lu * lp;
    auto stride1  = lu * hp * lp;
    auto stride2  = hp * lp;
    auto dstPtr   = (int32_t*)dest;
    auto srcPtr   = (int32_t*)source;
    int unitpacksize = sizeof(int32_t) / sizeof(uint8_t);

    for (int i = 0; i < hu; ++i) {
        for (int k = 0; k < hp; ++k) {
            for (int bl = 0; bl < blocknum; ++bl) {
                int j = 0;
                while (j + 7 < lu) {
                    auto srcindex0 = ((i * hp + k) * ic + bl * (lu * lp) + j * lp) / unitpacksize;
                    auto srcindex1 = ((i * hp + k) * ic + bl * (lu * lp) + (j + 4) * lp) / unitpacksize;
                    auto dstindex0 = (bl * stride1 + i * stride0 + j * stride2 + k * lp) / unitpacksize;
                    auto dstindex1 = (bl * stride1 + i * stride0 + (j + 1) * stride2 + k * lp) / unitpacksize;
                    auto dstindex2 = (bl * stride1 + i * stride0 + (j + 2) * stride2 + k * lp) / unitpacksize;
                    auto dstindex3 = (bl * stride1 + i * stride0 + (j + 3) * stride2 + k * lp) / unitpacksize;
                    auto dstindex4 = (bl * stride1 + i * stride0 + (j + 4) * stride2 + k * lp) / unitpacksize;
                    auto dstindex5 = (bl * stride1 + i * stride0 + (j + 5) * stride2 + k * lp) / unitpacksize;
                    auto dstindex6 = (bl * stride1 + i * stride0 + (j + 6) * stride2 + k * lp) / unitpacksize;
                    auto dstindex7 = (bl * stride1 + i * stride0 + (j + 7) * stride2 + k * lp) / unitpacksize;
                    j += 8;
                    auto srcdata0 = vld1q_s32(srcPtr + srcindex0);
                    auto srcdata1 = vld1q_s32(srcPtr + srcindex1);
                    vst1q_lane_s32(dstPtr + dstindex0, srcdata0, 0);
                    vst1q_lane_s32(dstPtr + dstindex1, srcdata0, 1);
                    vst1q_lane_s32(dstPtr + dstindex2, srcdata0, 2);
                    vst1q_lane_s32(dstPtr + dstindex3, srcdata0, 3);
                    vst1q_lane_s32(dstPtr + dstindex4, srcdata1, 0);
                    vst1q_lane_s32(dstPtr + dstindex5, srcdata1, 1);
                    vst1q_lane_s32(dstPtr + dstindex6, srcdata1, 2);
                    vst1q_lane_s32(dstPtr + dstindex7, srcdata1, 3);
                }
                while (j + 3 < lu) {
                    auto srcindex = ((i * hp + k) * ic + bl * (lu * lp) + j * lp) / unitpacksize;
                    auto dstindex0 = (bl * stride1 + i * stride0 + j * stride2 + k * lp) / unitpacksize;
                    auto dstindex1 = (bl * stride1 + i * stride0 + (j + 1) * stride2 + k * lp) / unitpacksize;
                    auto dstindex2 = (bl * stride1 + i * stride0 + (j + 2) * stride2 + k * lp) / unitpacksize;
                    auto dstindex3 = (bl * stride1 + i * stride0 + (j + 3) * stride2 + k * lp) / unitpacksize;
                    j += 4;
                    auto srcdata = vld1q_s32(srcPtr + srcindex);
                    vst1q_lane_s32(dstPtr + dstindex0, srcdata, 0);
                    vst1q_lane_s32(dstPtr + dstindex1, srcdata, 1);
                    vst1q_lane_s32(dstPtr + dstindex2, srcdata, 2);
                    vst1q_lane_s32(dstPtr + dstindex3, srcdata, 3);
                }
                while (j < lu) {
                    auto srcindex = ((i * hp + k) * ic + bl * (lu * lp) + j * lp) / unitpacksize;
                    auto dstindex = (bl * stride1+ i * stride0 + j * stride2 + k * lp) / unitpacksize;
                    dstPtr[dstindex] = srcPtr[srcindex];
                    j++;
                }
            }
        }
    }
    MNNPermuteSumWeightInt4Arm86(dest, dest, blocknum * hu, lu, kernelsum);
}

static void MNNReorderWeightInt4Arm82(uint8_t* dest, const uint8_t* source, int32_t* shape, size_t size, float* kernelsum) {
    MNN_ASSERT(size > 4);
    // dst shape: [hu, blocknum, kernelCount, lu, hp, lp], kernelCount=1 in this case
    auto blocknum = shape[0];
    auto hu       = shape[1];
    auto lu       = shape[2];
    auto hp       = shape[3];
    auto lp       = shape[4];
    auto ic       = blocknum *lu * lp;
    auto stride0  = blocknum * hp * lu * lp;
    auto stride1  = lu * hp * lp;
    auto stride2  = hp * lp;
    auto dstPtr   = (int16_t*)dest;
    auto srcPtr   = (int16_t*)source;
    int unitpacksize = sizeof(int16_t) / sizeof(uint8_t);
    for (int i = 0; i < hu; ++i) {
        for (int k = 0; k < hp; ++k) {
            for (int bl = 0; bl < blocknum; ++bl) {
                int j = 0;
                while (j + 7 < lu) {
                    auto srcindex = ((i * hp + k) * ic + bl * (lu * lp) + j * lp) / unitpacksize;
                    auto dstindex0 = (bl * stride1 + i * stride0 + j * stride2 + k * lp) / unitpacksize;
                    auto dstindex1 = (bl * stride1 + i * stride0 + (j + 1) * stride2 + k * lp) / unitpacksize;
                    auto dstindex2 = (bl * stride1 + i * stride0 + (j + 2) * stride2 + k * lp) / unitpacksize;
                    auto dstindex3 = (bl * stride1 + i * stride0 + (j + 3) * stride2 + k * lp) / unitpacksize;
                    auto dstindex4 = (bl * stride1 + i * stride0 + (j + 4) * stride2 + k * lp) / unitpacksize;
                    auto dstindex5 = (bl * stride1 + i * stride0 + (j + 5) * stride2 + k * lp) / unitpacksize;
                    auto dstindex6 = (bl * stride1 + i * stride0 + (j + 6) * stride2 + k * lp) / unitpacksize;
                    auto dstindex7 = (bl * stride1 + i * stride0 + (j + 7) * stride2 + k * lp) / unitpacksize;
                    j += 8;
                    auto srcdata = vld1q_s16(srcPtr + srcindex);
                    vst1q_lane_s16(dstPtr + dstindex0, srcdata, 0);
                    vst1q_lane_s16(dstPtr + dstindex1, srcdata, 1);
                    vst1q_lane_s16(dstPtr + dstindex2, srcdata, 2);
                    vst1q_lane_s16(dstPtr + dstindex3, srcdata, 3);
                    vst1q_lane_s16(dstPtr + dstindex4, srcdata, 4);
                    vst1q_lane_s16(dstPtr + dstindex5, srcdata, 5);
                    vst1q_lane_s16(dstPtr + dstindex6, srcdata, 6);
                    vst1q_lane_s16(dstPtr + dstindex7, srcdata, 7);
                }
                while (j + 3 < lu) {
                    auto srcindex = ((i * hp + k) * ic + bl * (lu * lp) + j * lp) / unitpacksize;
                    auto dstindex0 = (bl * stride1 + i * stride0 + j * stride2 + k * lp) / unitpacksize;
                    auto dstindex1 = (bl * stride1 + i * stride0 + (j + 1) * stride2 + k * lp) / unitpacksize;
                    auto dstindex2 = (bl * stride1 + i * stride0 + (j + 2) * stride2 + k * lp) / unitpacksize;
                    auto dstindex3 = (bl * stride1 + i * stride0 + (j + 3) * stride2 + k * lp) / unitpacksize;
                    j += 4;
                    auto srcdata = vld1_s16(srcPtr + srcindex);
                    vst1_lane_s16(dstPtr + dstindex0, srcdata, 0);
                    vst1_lane_s16(dstPtr + dstindex1, srcdata, 1);
                    vst1_lane_s16(dstPtr + dstindex2, srcdata, 2);
                    vst1_lane_s16(dstPtr + dstindex3, srcdata, 3);

                }
                while (j < lu)
                {
                    auto srcindex = ((i * hp + k) * ic + bl * (lu * lp) + j * lp) / 2;
                    auto dstindex = (bl * stride1 + i * stride0 + j * stride2 + k * lp) / 2;
                    dstPtr[dstindex] = srcPtr[srcindex];
                    j++;
                }
            }
        }
    }
    MNNPermuteSumWeightInt4Arm82(dest, dest, blocknum * hu, lu, kernelsum);
}
#ifdef MNN_SME2
static void MNNReorderWeightInt4Sme2(uint8_t* dest, const uint8_t* source, int32_t* shape, size_t size, float* kernelsum) {
    MNN_ASSERT(size > 4);
    // dst shape: [hu, blocknum, kernelCount, lu, hp, lp], kernelCount=1 in this case
    auto blocknum = shape[0];
    auto hu       = shape[1];
    auto lu       = shape[2];
    auto hp       = shape[3];
    auto lp       = shape[4];
    auto ic       = blocknum *lu * lp;
    auto stride0  = blocknum * hp * lu * lp;
    auto stride1  = lu * hp * lp;
    auto stride2  = hp * lp;
    auto dstPtr   = (int16_t*)dest;
    auto srcPtr   = (int16_t*)source;
    int unitpacksize = sizeof(int16_t) / sizeof(uint8_t);
    for (int i = 0; i < hu; ++i) {
        for (int k = 0; k < hp; ++k) {
            for (int bl = 0; bl < blocknum; ++bl) {
                int j = 0;
                while (j + 7 < lu) {
                    auto srcindex = ((i * hp + k) * ic + bl * (lu * lp) + j * lp) / unitpacksize;
                    auto dstindex0 = (bl * stride1 + i * stride0 + j * stride2 + k * lp) / unitpacksize;
                    auto dstindex1 = (bl * stride1 + i * stride0 + (j + 1) * stride2 + k * lp) / unitpacksize;
                    auto dstindex2 = (bl * stride1 + i * stride0 + (j + 2) * stride2 + k * lp) / unitpacksize;
                    auto dstindex3 = (bl * stride1 + i * stride0 + (j + 3) * stride2 + k * lp) / unitpacksize;
                    auto dstindex4 = (bl * stride1 + i * stride0 + (j + 4) * stride2 + k * lp) / unitpacksize;
                    auto dstindex5 = (bl * stride1 + i * stride0 + (j + 5) * stride2 + k * lp) / unitpacksize;
                    auto dstindex6 = (bl * stride1 + i * stride0 + (j + 6) * stride2 + k * lp) / unitpacksize;
                    auto dstindex7 = (bl * stride1 + i * stride0 + (j + 7) * stride2 + k * lp) / unitpacksize;
                    j += 8;
                    auto srcdata = vld1q_s16(srcPtr + srcindex);
                    vst1q_lane_s16(dstPtr + dstindex0, srcdata, 0);
                    vst1q_lane_s16(dstPtr + dstindex1, srcdata, 1);
                    vst1q_lane_s16(dstPtr + dstindex2, srcdata, 2);
                    vst1q_lane_s16(dstPtr + dstindex3, srcdata, 3);
                    vst1q_lane_s16(dstPtr + dstindex4, srcdata, 4);
                    vst1q_lane_s16(dstPtr + dstindex5, srcdata, 5);
                    vst1q_lane_s16(dstPtr + dstindex6, srcdata, 6);
                    vst1q_lane_s16(dstPtr + dstindex7, srcdata, 7);
                }
                while (j + 3 < lu) {
                    auto srcindex = ((i * hp + k) * ic + bl * (lu * lp) + j * lp) / unitpacksize;
                    auto dstindex0 = (bl * stride1 + i * stride0 + j * stride2 + k * lp) / unitpacksize;
                    auto dstindex1 = (bl * stride1 + i * stride0 + (j + 1) * stride2 + k * lp) / unitpacksize;
                    auto dstindex2 = (bl * stride1 + i * stride0 + (j + 2) * stride2 + k * lp) / unitpacksize;
                    auto dstindex3 = (bl * stride1 + i * stride0 + (j + 3) * stride2 + k * lp) / unitpacksize;
                    j += 4;
                    auto srcdata = vld1_s16(srcPtr + srcindex);
                    vst1_lane_s16(dstPtr + dstindex0, srcdata, 0);
                    vst1_lane_s16(dstPtr + dstindex1, srcdata, 1);
                    vst1_lane_s16(dstPtr + dstindex2, srcdata, 2);
                    vst1_lane_s16(dstPtr + dstindex3, srcdata, 3);

                }
                while (j < lu)
                {
                    auto srcindex = ((i * hp + k) * ic + bl * (lu * lp) + j * lp) / 2;
                    auto dstindex = (bl * stride1 + i * stride0 + j * stride2 + k * lp) / 2;
                    dstPtr[dstindex] = srcPtr[srcindex];
                    j++;
                }
            }
        }
    }
    int32_t table[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    if (hp == 32) {
        MNNPermuteSumWeightInt4Sme2_Hp32(dest, dest, blocknum * hu, lu, kernelsum, table);
    } else if (hp == 128) { // [hu,blocknum,lu,hp,lp]
        MNNPermuteSumWeightInt4Sme2_Hp128(dest, dest, blocknum * hu, lu, kernelsum, table);
    } else {
        for (int i = 0; i < blocknum * hu; ++i) {
            std::vector<float> sum(hp, 0);
            for (int j = 0; j < lu; ++j) {
                auto destPtr = dest + i * lu * lp * hp + j * lp * hp;
                for (int k = 0; k < hp; ++k) {
                    for (int x = 0; x < lp; ++x) {
                        uint8_t data = destPtr[k * lp + x];
                        auto d0 = data / 16;
                        auto d1 = data % 16;
                        sum[k] = sum[k] + float(d0 + d1);
                        destPtr[k * lp + x] = d0 + d1 * 16;
                    }
                }
            }
            memcpy(kernelsum + i * hp, sum.data(), hp * sizeof(float));
        }
    }

}
#endif // sme2
#endif // __aarch64__

static void MNNSumWeightInt8(float* kernelsum, int8_t* source, size_t outside, size_t reduceAxis, size_t hP, size_t lP) {
    // weight shape: [outside, axis, hP, lP]
    // outside    = blocknum * hU
    // reduceAxis = kernelCount * lU
    auto inside = hP * lP;
    auto stride0 = inside * reduceAxis;
    std::vector<float> accum(hP);
    for (int i = 0; i < outside; ++i) {
        memset(accum.data(), 0, hP * 4);
        for (int j = 0; j < reduceAxis; ++j) {
            for (int k = 0; k < hP; ++k) {
                for (int x = 0; x < lP; ++x) {
                    accum[k] += (float)source[x + k * lP + j * inside + i * stride0];
                }
            }
        }
        memcpy(kernelsum + i * hP, accum.data(), hP * sizeof(float));
    }
}

static void MNNSumByAxisLForMatmul_A(float* dest, int8_t* source, const float* scale, ssize_t realDstCount, SumByAxisParams sumParams) {
#ifdef MNN_USE_SSE
    uint8_t* srcInt8 = reinterpret_cast<uint8_t*>(source);
#else
    int8_t* srcInt8 = source;
#endif
    auto scalePtr = scale;
    auto blockNum = sumParams.blockNum;
    auto EP = sumParams.DST_XUNIT;
    auto LP = sumParams.SRC_UNIT;
    auto col_buffer_unit_size = sumParams.unitColBufferSize;
    auto oneScale = sumParams.oneScale;
    auto LU = sumParams.LU;
    auto valid = sumParams.valid;
    auto kernelxy = sumParams.kernelxy;
    auto blockSizeQuad = LU / blockNum;
    auto inputBlockQuant = sumParams.inputBlock;
    auto lastL = LP;
    if (valid) {
        lastL = valid;
    }
    float singlescale = scale[0];
    do {
        int step = ALIMIN(EP, realDstCount);
        int scaleOffset = inputBlockQuant ? (step * blockNum) : step;

        for (int k = 0; k < blockNum; ++k) {
            const auto src_x = srcInt8 + k * (step * LP * blockSizeQuad * kernelxy);
            for (int w = 0; w < step; ++w) {
                float dequantScale = singlescale;
                if (oneScale == 0 && inputBlockQuant) {
                    dequantScale = scalePtr[w + k * step];
                } else if (oneScale == 0) {
                    dequantScale = scalePtr[w];
                }
                int sumint32 = 0;
                const auto src_y = src_x + w * LP;
                for (int j = 0; j < kernelxy; ++j) {
                    for (int i = 0; i < blockSizeQuad; ++i) {
                        auto sumsize = i == (blockSizeQuad - 1) ? lastL : LP;
                        const auto src_z = src_y + j * (blockSizeQuad * step * LP) + i * step * LP;
                        for (int x = 0; x < sumsize; ++x) {
                            sumint32 += src_z[x];
                        }
                    }
                }
                dest[w + k * step] = dequantScale * static_cast<float>(sumint32);
            }
        }
        scalePtr += scaleOffset;

        dest += (step * blockNum);
        realDstCount -= step;
        srcInt8 += col_buffer_unit_size;
    } while(realDstCount > 0);
}

template<typename T>
void MNNPackC4Common(T* dst, const T* src, size_t area, size_t depth, int* areaOffset) {
    int depthC4     = depth / 4;
    int depthRemain = depthC4 * 4;
    int remain      = depth - depthRemain;
    int z, x, y;
    const T* srcChannel[4];
    const T* srcOffset = src;
    for(z = 0; z < depthC4; ++z) {
        auto dstZ = dst + z * areaOffset[1] * 4;
        for(y = 0; y < 4; ++y) {
            srcChannel[y] = srcOffset + areaOffset[0] * y;
        }
        for(x = 0; x < area; ++x) {
            for(y = 0; y < 4; ++y) {
                dstZ[0] = srcChannel[y][x];
                dstZ++;
            }
        }
        srcOffset += areaOffset[0] * 4;
    }
    if(remain > 0){
        auto dstZ = dst + depthC4 * areaOffset[1] * 4;
        for(y = 0; y < remain; ++y) {
            srcChannel[y] = srcOffset + areaOffset[0] * y;
        }
        for(x = 0; x < area; ++x) {
            for(y = 0; y < remain; ++y) {
                dstZ[0] = srcChannel[y][x];
                dstZ++;
            }
            for(y = remain; y < 4; ++y) {
                dstZ[0] = 0;
                dstZ++;
            }
        }
    }
}

template<typename T>
void MNNUnpackC4Common(T* dst, const T* src, size_t area, size_t depth, int* areaOffset) {
    int depthC4     = depth / 4;
    int depthRemain = depthC4 * 4;
    int remain      = depth - depthRemain;
    int z, x, y;
    const T* srcChannel[4];
    const T* srcOffset = src;
    for(z = 0; z < depthC4; ++z) {
        for(y = 0; y < 4; ++y) {
            auto dstZ = dst + (z * 4 + y) * areaOffset[1];
            srcChannel[y] = srcOffset + y;
            for(x = 0; x < area; ++x) {
                dstZ[x] = srcChannel[y][0];
                srcChannel[y] += 4;
            }
        }
        srcOffset += areaOffset[0] * 4;
    }
    if(remain > 0){
        auto dstZ = dst + depthC4 * areaOffset[1] * 4;
        for(y = 0; y < remain; ++y) {
            srcChannel[y] = srcOffset + y;
            for(x = 0; x < area; ++x) {
                dstZ[x] = srcChannel[y][0];
                srcChannel[y] += 4;
            }
            dstZ += areaOffset[1];
        }
    }
}

template<typename T>
void MNNPackC2Common(T* dst, const T* src, size_t area, size_t depth, int* areaOffset) {
    int depthC2     = depth / 2;
    int depthRemain = depthC2 * 2;
    int remain      = depth - depthRemain;
    int z, x, y;
    const T* srcChannel[2];
    const T* srcOffset = src;
    for(z = 0; z < depthC2; ++z) {
        auto dstZ = dst + z * areaOffset[1] * 2;
        for(y = 0; y < 2; ++y) {
            srcChannel[y] = srcOffset + areaOffset[0] * y;
        }
        for(x = 0; x < area; ++x) {
            for(y = 0; y < 2; ++y) {
                dstZ[0] = srcChannel[y][x];
                dstZ++;
            }
        }
        srcOffset += areaOffset[0] * 2;
    }
    if(remain > 0){
        auto dstZ = dst + depthC2 * areaOffset[1] * 2;
        for(y = 0; y < remain; ++y) {
            srcChannel[y] = srcOffset + areaOffset[0] * y;
        }
        for(x = 0; x < area; ++x) {
            for(y = 0; y < remain; ++y) {
                dstZ[0] = srcChannel[y][x];
                dstZ++;
            }
            for(y = remain; y < 2; ++y) {
                dstZ[0] = 0;
                dstZ++;
            }
        }
    }
}

template<typename T>
void MNNUnpackC2Common(T* dst, const T* src, size_t area, size_t depth, int* areaOffset, int pack = 1) {
    int depthC2     = depth / 2;
    int depthRemain = depthC2 * 2;
    int remain      = depth - depthRemain;
    int z, x, y;
    const T* srcChannel[2];
    const T* srcOffset = src;
    for(z = 0; z < depthC2; ++z) {
        for(y = 0; y < 2; ++y) {
            auto dstZ = dst + (z * 2 + y) * areaOffset[1] * pack;
            srcChannel[y] = srcOffset + y * pack;
            for(x = 0; x < area; ++x) {
                for (int p = 0; p < pack; ++p) {
                    dstZ[x * pack + p] = srcChannel[y][p];
                }
                srcChannel[y] += (2 * pack);
            }
        }
        srcOffset += areaOffset[0] * 2 * pack;
    }
    if(remain > 0){
        auto dstZ = dst + depthC2 * areaOffset[1] * 2 * pack;
        for(y = 0; y < remain; ++y) {
            srcChannel[y] = srcOffset + y * pack;
            for(x = 0; x < area; ++x) {
                for (int p = 0; p < pack; ++p) {
                    dstZ[x * pack + p] = srcChannel[y][p];
                }
                srcChannel[y] += 2 * pack;
            }
            dstZ += areaOffset[1] * pack;
        }
    }
}

void MNN4BitcopyWithStride (uint8_t* dstO, const uint8_t* srcO, int size, int stride, int ds) {
    auto src = (uint32_t*)srcO;
    auto dst = (uint32_t*)dstO;
    for (int i = 0; i < size; ++i) {
        dst[0] = *src;
        dst += ds;
        src += stride;
    }
}

void MNN4BitcopyFast (uint8_t* dstO, const uint8_t* srcO, int size, int stride, int ds) {
    // ds=1, stride=0||1
    auto src = (float*)srcO;
    auto dst = (float*)dstO;
    int cnt  = size;
    if (stride == 1) { // stride=1
#ifdef MNN_USE_NEON
        for (; cnt >= 8; cnt -= 8) {
            auto v4 = vld1q_f32(src);
            auto u4 = vld1q_f32(src + 4);
            vst1q_f32(dst, v4);
            vst1q_f32(dst + 4, u4);
            dst += 8;
            src += 8;
        }
        for (; cnt >= 4; cnt -= 4) {
            auto v4 = vld1q_f32(src);
            vst1q_f32(dst, v4);
            dst += 4;
            src += 4;
        }
#elif defined(MNN_USE_SSE)
        for (; cnt >= 8; cnt -= 8) {
            __m128 v4 = _mm_loadu_ps(src);
            __m128 u4 = _mm_loadu_ps(src + 4);
            _mm_storeu_ps(dst, v4);
            _mm_storeu_ps(dst + 4, u4);
            dst += 8;
            src += 8;
        }
        for (; cnt >= 4; cnt -= 4) {
            __m128 v4 = _mm_loadu_ps(src);
            _mm_storeu_ps(dst, v4);
            dst += 4;
            src += 4;
        }
#endif
    } else { // stride=0
        int i = 0;
        float val = *src;
#ifdef MNN_USE_NEON
        auto val4 = vdupq_n_f32(val);
        for (; cnt >= 8; cnt -= 8) {
            vst1q_f32(dst, val4);
            vst1q_f32(dst + 4, val4);
            dst += 8;
        }
        for (; cnt >= 4; cnt -= 4) {
            vst1q_f32(dst, val4);
            dst += 4;
        }
#elif defined(MNN_USE_SSE)
        __m128 val4 = _mm_set_ps(val, val, val, val);
        for (; cnt >= 8; cnt -= 8) {
            _mm_storeu_ps(dst, val4);
            _mm_storeu_ps((dst + 4), val4);
            dst += 8;
        }
        for (; cnt >= 4; cnt -= 4) {
            _mm_storeu_ps(dst, val4);
            dst += 4;
        }
#endif
    }
    for (; cnt > 0; --cnt) {
        dst[0] = *src;
        dst += ds;
        src += stride;
    }
}

void MNN2BitcopyWithStride(uint8_t* dstO, const uint8_t* srcO, int size, int stride, int ds) {
    auto src = (uint16_t*)srcO;
    auto dst = (uint16_t*)dstO;
    for (int i=0; i<size; ++i) {
        *dst = *src;
        src+=stride;
        dst+=ds;
    }
}

void MNN2BitcopyFast(uint8_t* dstO, const uint8_t* srcO, int size, int stride, int ds) {
    auto src = (uint16_t*)srcO;
    auto dst = (uint16_t*)dstO;
    int cnt  = size;
    uint16_t val = *src;
    if (stride == 1) {
#ifdef MNN_USE_NEON
        for (; cnt >= 8; cnt-=8) {
            auto val8 = vld1q_u16(src);
            vst1q_u16(dst, val8);
            src += 8;
            dst += 8;
        }
        for (; cnt >= 4; cnt-=4) {
            auto val4 = vld1_u16(src);
            vst1_u16(dst, val4);
            src += 4;
            dst += 4;
        }
#elif defined(MNN_USE_SSE)
        for (; cnt >= 8; cnt-=8) {
            auto tmp = _mm_loadu_ps((float*)src);
            _mm_storeu_ps((float*)dst, tmp);
            src += 8;
            dst += 8;
        }
#endif
    } else { // stride=0
#ifdef MNN_USE_NEON
        auto val4 = vdup_n_u16(val);
        auto val8 = vdupq_n_u16(val);
        for (; cnt >= 8; cnt-=8) {
            vst1q_u16(dst, val8);
            dst += 8;
        }
        for (; cnt >= 4; cnt-=4) {
            vst1_u16(dst, val4);
            dst += 4;
        }
#elif defined(MNN_USE_SSE)
        uint16_t arr[8] = {val, val, val, val, val, val, val, val};
        auto val8 = _mm_loadu_ps((float*)arr);
        for (; cnt >= 8; cnt-=8) {
            _mm_storeu_ps((float*)dst, val8);
            dst += 8;
        }
#endif
    }
    for (; cnt > 0; --cnt) {
        *dst = *src;
        src += stride;
        dst += ds;
    }
}

void MNN1BitcopyWithStride (uint8_t* dstO, const uint8_t* srcO, int size, int stride, int ds) {
    for (int i = 0; i < size; ++i) {
        dstO[0] = *srcO;
        dstO += ds;
        srcO += stride;
    }

}

void MNN1BitCopyFast (uint8_t* dstO, const uint8_t* srcO, int size, int stride, int ds) {
    int cnt = size;
    uint8_t val = *srcO;
    if (stride == 1) {
#ifdef MNN_USE_SSE
        for (; cnt >= 16; cnt-=16) {
            auto tmp = _mm_loadu_ps((float*)srcO);
            _mm_storeu_ps((float*)dstO, tmp);
            srcO += 16;
            dstO += 16;
        }
#elif defined(MNN_USE_NEON)
        for (; cnt >= 16; cnt-=16) {
            auto val16 = vld1q_u8(srcO);
            vst1q_u8(dstO, val16);
            srcO += 16;
            dstO += 16;
        }
        for (; cnt >= 8; cnt-=8) {
            auto val8 = vld1_u8(srcO);
            vst1_u8(dstO, val8);
            srcO += 8;
            dstO += 8;
        }
#endif
    } else { // stride=0
#ifdef MNN_USE_SSE
        std::vector<uint8_t> arr(16, val);
        auto val16 = _mm_loadu_ps((float*)arr.data());

        for (; cnt >= 16; cnt-=16) {
            _mm_storeu_ps((float*)dstO, val16);
            dstO += 16;
        }
#elif defined(MNN_USE_NEON)
        auto val16 = vdupq_n_u8(val);
        auto val8 = vdup_n_u8(val);
        for (; cnt >= 16; cnt-=16) {
            vst1q_u8(dstO, val16);
            dstO += 16;
        }
        for (; cnt >= 8; cnt-=8) {
            vst1_u8(dstO, val8);
            dstO += 8;
        }
#endif
    }
    for (; cnt > 0; --cnt) {
        dstO[0] = *srcO;
        dstO += ds;
        srcO += stride;
    }
}

void MNNAccumulateSequenceNumber (float* dst, const float* src, int size) {
    // mode: 0:Add, 1:Sub, 2:Min, 3:Max
    int size8 = (size / 8) * 8;
    int i = 0;
    float sum = 0.f;
    float tmp[4];
#ifdef MNN_USE_NEON
    if (size >= 8) {
        auto sum4_1 = vdupq_n_f32(0.f);
        auto sum4_2 = vdupq_n_f32(0.f);
        for (; i < size8; i += 8) {
            auto v4 = vld1q_f32(src);
            auto u4 = vld1q_f32(src + 4);
            sum4_1 = vaddq_f32(sum4_1, v4);
            sum4_2 = vaddq_f32(sum4_2, u4);
            src += 8;
        }
        sum4_1 = vaddq_f32(sum4_1, sum4_2);
        sum = (sum4_1[0] + sum4_1[1]) + (sum4_1[2] + sum4_1[3]);
    }
#elif defined(MNN_USE_SSE)
    if (size >= 8) {
        auto sum4_1 = _mm_set_ps1(0.f);
        auto sum4_2 = _mm_set_ps1(0.f);

        for (; i < size8; i += 8) {
            auto v4 = _mm_loadu_ps(src);
            auto u4 = _mm_loadu_ps(src + 4);
            sum4_1 = _mm_add_ps(sum4_1, v4);
            sum4_2 = _mm_add_ps(sum4_2, u4);
            src += 8;
        }

        sum4_1 = _mm_add_ps(sum4_1, sum4_2);
        _mm_storeu_ps(tmp, sum4_1);
        sum += (tmp[0] + tmp[1] + tmp[2] + tmp[3]);
    }
#endif
    for (; i < size; ++i) {
        sum += (*src);
        src += 1;
    }
    *dst = sum;
}

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

static void MNNFlashAttentionUpdateBlockOutput(float* dst, float* src, float* scale, float* normalizeScale, int depthQuad, int plane, int pack, int idx, int kvBlocks, int size, int bytes, int seqStart) {
    // source shape:                 [headDim/pack, seqLen, pack]
    // scale & normalizeScale shape: [seqLen]
    // dest shape:                   [headDim/pack, seqLen, pack]
    auto stride0 = plane * pack;

    if (idx > 0) {
        for (int j = 0; j < depthQuad; ++j) {
            int i = seqStart;
            for (; i < plane; ++i) {
                auto dataNew = Vec::load(src + j * stride0 + i * pack);
                auto dataOld = Vec::load(dst + j * stride0 + i * pack);
                auto s = Vec(scale[i]);
                dataNew = Vec::fma(dataNew, dataOld, s);
                Vec::save(dst + j * stride0 + i * pack, dataNew);
            }
        }
    } else {
        memcpy(dst, src, size * bytes);
    }
    if (idx == kvBlocks - 1) { // if last subBlock, exp(xi)/sum(exp(xi))
        for (int j = 0; j < depthQuad; ++j) {
            for (int i = 0; i < plane; ++i) {
                auto dataNew = Vec::load(dst + j * stride0 + i * pack);
                auto ns = Vec(1.0f / normalizeScale[i]);
                dataNew = dataNew * ns;
                Vec::save(dst + j * stride0 + i * pack, dataNew);
            }
        }
    }
}

static void MNNAttenPackAndScaleSingleHead(float* dst, const float* srcHeadBase, size_t srcRowStride, const float* scale, const int32_t* units, size_t seqLen, size_t headDim) {
    const int32_t eP = units[0];
    const int32_t lP = units[1];

    if (lP != 1) {
        MNN_ERROR("This function only supports lP=1 or 2\n");
        return;
    }

    const float scaleVal = scale[0];
#ifdef MNN_USE_NEON
    const float32x4_t vScale = vdupq_n_f32(scaleVal);
#endif
    const size_t packedHeadDim = UP_DIV(headDim, lP);
    const size_t dstStrideDOuter = (size_t)eP * lP;
    const size_t dstStrideSOuter = packedHeadDim * dstStrideDOuter;

    for (int s = 0; s < seqLen; ++s) {
        const int sOuter = s / eP;
        const int sInner = s % eP;
        const float* srcRowPtr = srcHeadBase + s * srcRowStride;
        float* dstBasePtr = dst + sOuter * dstStrideSOuter + sInner * lP;

        size_t d = 0;
#ifdef MNN_USE_NEON
        for (; d + 7 < headDim; d += 8) {
            float32x4_t sVec0 = vld1q_f32(srcRowPtr + d);
            float32x4_t sVec1 = vld1q_f32(srcRowPtr + d + 4);
            sVec0 = vmulq_f32(sVec0, vScale);
            sVec1 = vmulq_f32(sVec1, vScale);

            dstBasePtr[(d + 0) * dstStrideDOuter] = sVec0[0];
            dstBasePtr[(d + 1) * dstStrideDOuter] = sVec0[1];
            dstBasePtr[(d + 2) * dstStrideDOuter] = sVec0[2];
            dstBasePtr[(d + 3) * dstStrideDOuter] = sVec0[3];
            dstBasePtr[(d + 4) * dstStrideDOuter] = sVec1[0];
            dstBasePtr[(d + 5) * dstStrideDOuter] = sVec1[1];
            dstBasePtr[(d + 6) * dstStrideDOuter] = sVec1[2];
            dstBasePtr[(d + 7) * dstStrideDOuter] = sVec1[3];
        }
        for (; d < headDim; ++d) {
            dstBasePtr[d * dstStrideDOuter] = srcRowPtr[d] * scaleVal;
        }
#else
        for (; d < headDim; ++d) {
            dstBasePtr[d * dstStrideDOuter] = srcRowPtr[d] * scaleVal;
        }
#endif
    }
}

#ifndef __aarch64__
void MNNQuantAttentionKey(int8_t* dst, const float* source, float* sumKeyPtr, float* maxKeyPtr, int32_t* params) {
    int32_t kvNumHead = params[0];
    int32_t seqLen = params[1];
    int32_t headDim = params[2];
    int32_t blockNum = params[3];
    int32_t eP = params[4];
    int32_t lP = params[5];
    int32_t hP = params[6];
    int32_t pastLength = params[7];
    int32_t kvHeadIdx = params[8];

    auto blockL = UP_DIV(headDim, blockNum);
    auto weightStride1 = ROUND_UP(blockL, lP) * hP;
    auto weightStride2 = lP * hP;
    auto packedWeightStride1 = weightStride1 + 2 * 4 * hP;

    if (seqLen > 1) {
        // get max
        for (int s = 0; s < seqLen; ++s) {
            const float* keySrc = source + s * kvNumHead * headDim + kvHeadIdx * headDim;
            for (int d = 0; d < headDim; d++) {
                maxKeyPtr[d] = ALIMAX(maxKeyPtr[d], keySrc[d]);
            }
        }
    }

    for (int s = 0; s < seqLen; s++) {
        const float* keySrc = source + s * kvNumHead * headDim + kvHeadIdx * headDim;
        float minKey, maxKey;
        minKey = keySrc[0] - maxKeyPtr[0];
        maxKey = keySrc[0] - maxKeyPtr[0];
        for (int d = 1; d < headDim; d++) {
            auto keydata = keySrc[d] - maxKeyPtr[d];
            minKey = ALIMIN(minKey, keydata);
            maxKey = ALIMAX(maxKey, keydata);
        }

        int outIndex = (pastLength + s) / hP;
        int inIndex  = (pastLength + s) % hP;

        float sumKey = 0;
        for (int k = 0; k < blockNum; ++k) {
            int8_t* weightDst = dst + outIndex * blockNum * packedWeightStride1 + k * packedWeightStride1;
            float* scaleDst = (float*)(weightDst + weightStride1);
            float* biasDst = scaleDst + hP;

            scaleDst[inIndex] = (maxKey - minKey) / 255.0f;
            biasDst[inIndex] = minKey + 128.f * (maxKey - minKey) / 255.f;

            for (int d = 0; d < blockL; d++) {
                int i = d / lP;
                int j = d % lP;

                int int8v = (int)(roundf((keySrc[d + k * blockL] - maxKeyPtr[d + k * blockL] - minKey) / (maxKey - minKey) * 255.0f - 128.0f));
                weightDst[i * weightStride2 + inIndex * lP + j] = int8v;
                sumKey += (int8v * scaleDst[inIndex] + biasDst[inIndex]);
            }
        }
        sumKeyPtr[outIndex * hP + inIndex] = sumKey;
    }
}

void MNNQuantAttentionValue(int8_t* dst, const float* source, float* valueSum, int32_t* params) {
    // float   value src : [kvSeq,kvNumHead,headDim]
    // int8_t  value dest: [updiv(maxLength,flashAttentionBlockKv), updiv(headDim,hp),updiv(flashAttentionBlockKv,lp),hp,lp]
    // float   value sum: [updiv(maxLength,flashAttentionBlockKv), roundup(headDim,hp)]
    int32_t kvNumHead = params[0];
    int32_t seqLen = params[1];
    int32_t headDim = params[2];
    int32_t blockNum = params[3];
    int32_t maxLength = params[4];

    int32_t lP = params[5];
    int32_t hP = params[6];
    int32_t pastLength = params[7];
    int32_t kvHeadIdx = params[8];

    int32_t flashAttentionBlockKv = params[9];

    auto blockKvseq = UP_DIV(seqLen + pastLength, blockNum);
    auto weightStride2 = lP * hP;
    auto weightStride1 = UP_DIV(flashAttentionBlockKv, lP) * weightStride2;

    auto packedStride1 = (int)(weightStride1 + 2 * hP * sizeof(float));
    auto packedStride0 = UP_DIV(headDim, hP) * packedStride1;

    auto srcStride0 = kvNumHead * headDim;

    auto sourceFp32 = (float*)source;

    // quant scale & bias
    if (pastLength == 0) {
        for (int d = 0; d < headDim; ++d) {
            float* scalePtr = (float*)(dst + (d / hP) * packedStride1 + weightStride1) + (d % hP);
            float* biasPtr = scalePtr + hP;

            // find min,max
            float dMax = sourceFp32[d + kvHeadIdx * headDim];
            float dMin = dMax;
            for (int s = 0; s < seqLen; ++s) {
                float data = sourceFp32[s * srcStride0 + d + kvHeadIdx * headDim];
                dMax = ALIMAX(dMax, data);
                dMin = ALIMIN(dMin, data);
            }

            // scale & bias
            float range = dMax - dMin;
            if (range < 1e-6) {
                scalePtr[0] = 0.f;
                biasPtr[0] = dMax;
            } else {
                float scale = range / 255.f;
                float bias  = range / 255.f * 128.f + dMin;
                scalePtr[0] = scale;
                biasPtr[0] = bias;
            }
        }
    }

    // copy the scale&bias to each blockKv
    //                                    pastLength == 0: First time prefill
    // (seqLen + pastLength) % flashAttentionBlockKv == 0: Open a new blockKv
    if (pastLength == 0 || (pastLength % flashAttentionBlockKv) == 0) {
        int32_t d0 = UP_DIV(maxLength, flashAttentionBlockKv);
        int32_t d1 = UP_DIV(headDim, hP);
        for (int k = 0; k < d0; ++k) {
            for (int r = 0; r < d1; ++r) {
                float* scalePtr = (float*)(dst + k * packedStride0 + r * packedStride1 + weightStride1);
                float* biasPtr  = scalePtr + hP;
                memcpy(scalePtr, dst + r * packedStride1 + weightStride1, hP * sizeof(float));
                memcpy(biasPtr, dst + r * packedStride1 + weightStride1 + hP * sizeof(float), hP * sizeof(float));
            }
        }
    }

    for (int d = 0; d < headDim; ++d) {
        // dst address
        int idxBase = (d / hP) * packedStride1 + (d % hP) * lP;
        int8_t*   dstBase = dst + idxBase;
        float*  scaleBase = (float*)(dst + (d / hP) * packedStride1 + weightStride1) + (d % hP);
        float*   biasBase = scaleBase + hP;
        float*   sumBase = valueSum + (d / hP) * hP + (d % hP);

        float qscale = scaleBase[0] < 1e-6 ? 0 : 1.0f / scaleBase[0];
        float qbias = scaleBase[0] < 1e-6 ? 0 : (-biasBase[0] / scaleBase[0]);
        // quant
        for (int s = 0; s < seqLen; ++s) {
            int kvSeqIndx = s + pastLength;
            int idxInner = (kvSeqIndx / flashAttentionBlockKv) * packedStride0 + (kvSeqIndx % flashAttentionBlockKv) / lP * weightStride2 + (kvSeqIndx % flashAttentionBlockKv) % lP;
            float xf = sourceFp32[s * srcStride0 + d + kvHeadIdx * headDim];
            int8_t xq = ALIMAX(ALIMIN(127, static_cast<int32_t>(roundf(xf * qscale + qbias))), -128);
            dstBase[idxInner] = xq;

            // sum
            int idxSum = (kvSeqIndx / flashAttentionBlockKv) * ROUND_UP(headDim, hP);
            sumBase[idxSum] += ((float)xq * scaleBase[0] + biasBase[0]);
        }
    }
}

#endif

#endif // MNN_SUPPORT_TRANSFORMER_FUSE


#ifndef MNN_USE_NEON

void MNNGetMatMulPackMode(int* eP, int *lP, int* hP) {
    *eP = 16;
    *lP = 1;
    *hP = 4;
}

void MNNGetSparseMatMulPackMode(int* eP, int *lP, int* hP) {
    *eP = 16;
    *lP = 1;
    *hP = 4;
    // hp is corresponding to sparse block along right matrix colum dimension. in ramdom sparse, it is 1.
    return;
}

void MNNPackForMatMul_B(float* dest, const float* source, size_t h, size_t kernelsize, size_t ic, bool transpose) {
    // src: [h, kernelsize, ic]
    auto hP = h / 4;
    auto hR = hP * 4;
    auto l = kernelsize * ic;
    if (hR != h) {
        ::memset(dest, 0, UP_DIV(h, 4)*4*l*sizeof(float));
    }
    if (!transpose) {
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
        (int)l,
        (int)l
    };
    MNNPackC4(dest, source, l, h, offset);
}

static void _MNNPackedMatMulRemain(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter, const float* postParameters, const float* bias, int aStride) {
    auto h = parameter[2];
    auto l = parameter[1];
    auto cStride = parameter[3] / sizeof(float);
    auto hRemain = parameter[4];
    auto bExtraStride = parameter[5] / sizeof(float);
    auto bStride = bExtraStride + l * 4;
    auto hC4 = UP_DIV(h, 4);
    for (int y=0; y<hC4; ++y) {
        ::memset(C + y * cStride, 0, eSize * 4 * sizeof(float));
    }
    float alpha = 1.0f;
    float beta = 0.0f;
    float minValue = -std::numeric_limits<float>().max();
    float maxValue = std::numeric_limits<float>().max();
    if (nullptr != postParameters) {
        minValue = postParameters[2];
        maxValue = postParameters[3];
        alpha = postParameters[0];
        beta = postParameters[1];
    }

    for (int x=0; x<eSize; ++x) {
        auto dst = C + 4 * x;
        auto src = A + x;
        for (int y=0; y<hC4; ++y) {
            auto dstY = dst + y * cStride;
            auto weight = B + y * bStride;
            float summer[4] = {
                0.0f,
                0.0f,
                0.0f,
                0.0f,
            };
            if (nullptr != bias) {
                for (int v=0; v<4; ++v) {
                    summer[v] = bias[4 * y + v];
                }
            }
            for (int z=0; z<l; ++z) {
                auto aZ = src + z * aStride;
                auto wZ = weight + z * 4;
                summer[0] += wZ[0] * aZ[0];
                summer[1] += wZ[1] * aZ[0];
                summer[2] += wZ[2] * aZ[0];
                summer[3] += wZ[3] * aZ[0];
            }
            for (int v=0; v<4; ++v) {
                auto dstValue = std::min(summer[v], maxValue);
                dstValue = std::max(dstValue, minValue);
                dstY[v] = dstValue;
            }
        }
    }
}

void MNNPackedMatMul(float* C, const float* A, const float* B, const size_t* parameter, const float* postParameters, const float* bias, const float* k, const float* b) {
    return _MNNPackedMatMulRemain(C, A, B, 16, parameter, postParameters, bias, 16);
}

void MNNPackedMatMulRemain(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter, const float* postParameters, const float* bias, const float* k, const float* b) {
    auto aStride = parameter[0] / sizeof(float);
    _MNNPackedMatMulRemain(C, A, B, eSize, parameter, postParameters, bias, aStride);
}

void MNNPackC4ForMatMul_A(float* destOrigin, float const** sourceGroup, const int32_t* info, const int32_t* el) {
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

void MNNPackedSparseMatMulEpx1(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter, const float* postParameters, const float* bias, unsigned int* NNZMap, int* dataOffsetMap) {

    auto eP = parameter[0] / sizeof(float);
    MNN_ASSERT((eP & 0x03) == 0); // In sparse calculate, eP should be evenly divided by 4
    auto h = parameter[2];
    auto l = parameter[1];
    auto cStride = parameter[3] / sizeof(float);
    auto aStride = eP * l;
    auto hRemain = parameter[4];
    auto bExtraStride = parameter[5] / sizeof(float);
    auto bStride = bExtraStride + l * 4;
    auto hC4 = UP_DIV(h, 4);
    float minValue = -std::numeric_limits<float>().max();
    float maxValue = std::numeric_limits<float>().max();
    if (nullptr != postParameters) {
        minValue = postParameters[2];
        maxValue = postParameters[3];
    }
    // MNN_PRINT("MNNPackedSparseMatMul eP:%lu, eSize:%lu, l:%lu, h:%lu, cStride:%lu, aStride:%lu\n", eP, eSize, l, h, cStride, aStride);

    const float* a = A;
    size_t ie = 0;
    for (ie = 0; ie < eSize && eP <= eSize; ie += eP) {
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
            float acc0 = initValue;
            float acc1 = initValue;
            float acc2 = initValue;
            float acc3 = initValue;
            float acc4 = initValue;
            float acc5 = initValue;
            float acc6 = initValue;
            float acc7 = initValue;
            float acc8 = initValue;
            float acc9 = initValue;
            float acc10 = initValue;
            float acc11 = initValue;
            float acc12 = initValue;
            float acc13 = initValue;
            float acc14 = initValue;
            float acc15 = initValue;
            const int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {

                const int diff = *dataOffset++;
                const float a0 = a[0];
                const float a1 = a[1];
                const float a2 = a[2];
                const float a3 = a[3];
                const float a4 = a[4];
                const float a5 = a[5];
                const float a6 = a[6];
                const float a7 = a[7];
                const float a8 = a[8];
                const float a9 = a[9];
                const float a10 = a[10];
                const float a11 = a[11];
                const float a12 = a[12];
                const float a13 = a[13];
                const float a14 = a[14];
                const float a15 = a[15];

                const float oneW = *w++;

                // MNN_PRINT("16-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0-15]:", ie, a - A, w - B - 1, c - C, oneW);
                // formatMatrix(a, {16});
                // MNN_PRINT("\n");
                a = a + diff;
                acc0 += a0 * oneW;
                acc1 += a1 * oneW;
                acc2 += a2 * oneW;
                acc3 += a3 * oneW;
                acc4 += a4 * oneW;
                acc5 += a5 * oneW;
                acc6 += a6 * oneW;
                acc7 += a7 * oneW;
                acc8 += a8 * oneW;
                acc9 += a9 * oneW;
                acc10 += a10 * oneW;
                acc11 += a11 * oneW;
                acc12 += a12 * oneW;
                acc13 += a13 * oneW;
                acc14 += a14 * oneW;
                acc15 += a15 * oneW;
            }
            acc0  = std::max(std::min(maxValue, acc0), minValue);
            acc1  = std::max(std::min(maxValue, acc1), minValue);
            acc2  = std::max(std::min(maxValue, acc2), minValue);
            acc3  = std::max(std::min(maxValue, acc3), minValue);
            acc4  = std::max(std::min(maxValue, acc4), minValue);
            acc5  = std::max(std::min(maxValue, acc5), minValue);
            acc6  = std::max(std::min(maxValue, acc6), minValue);
            acc7  = std::max(std::min(maxValue, acc7), minValue);
            acc8  = std::max(std::min(maxValue, acc8), minValue);
            acc9  = std::max(std::min(maxValue, acc9), minValue);
            acc10 = std::max(std::min(maxValue, acc10), minValue);
            acc11 = std::max(std::min(maxValue, acc11), minValue);
            acc12 = std::max(std::min(maxValue, acc12), minValue);
            acc13 = std::max(std::min(maxValue, acc13), minValue);
            acc14 = std::max(std::min(maxValue, acc14), minValue);
            acc15 = std::max(std::min(maxValue, acc15), minValue);

            // how to store faster: st4 / transpose /
            c[0] = acc0;
            c[4] = acc1;
            c[4 * 2] = acc2;
            c[4 * 3] = acc3;
            c[4 * 4] = acc4;
            c[4 * 5] = acc5;
            c[4 * 6] = acc6;
            c[4 * 7] = acc7;
            c[4 * 8] = acc8;
            c[4 * 9] = acc9;
            c[4 * 10] = acc10;
            c[4 * 11] = acc11;
            c[4 * 12] = acc12;
            c[4 * 13] = acc13;
            c[4 * 14] = acc14;
            c[4 * 15] = acc15;
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
            float acc0 = initValue;
            float acc1 = initValue;
            float acc2 = initValue;
            float acc3 = initValue;
            float acc4 = initValue;
            float acc5 = initValue;
            float acc6 = initValue;
            float acc7 = initValue;

            const int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {
                const int diff = *dataOffset++;
                const float a0 = a[0];
                const float a1 = a[1];
                const float a2 = a[2];
                const float a3 = a[3];
                const float a4 = a[4];
                const float a5 = a[5];
                const float a6 = a[6];
                const float a7 = a[7];
                const float oneW = *w++;
                // MNN_PRINT("8-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0-7]:", ie, a - A, w - B - 1, c - C, oneW);
                // formatMatrix(a, {8});
                // MNN_PRINT("\n");
                a = a + diff;
                acc0 += a0 * oneW;
                acc1 += a1 * oneW;
                acc2 += a2 * oneW;
                acc3 += a3 * oneW;
                acc4 += a4 * oneW;
                acc5 += a5 * oneW;
                acc6 += a6 * oneW;
                acc7 += a7 * oneW;
            }
            acc0  = std::max(std::min(maxValue, acc0), minValue);
            acc1  = std::max(std::min(maxValue, acc1), minValue);
            acc2  = std::max(std::min(maxValue, acc2), minValue);
            acc3  = std::max(std::min(maxValue, acc3), minValue);
            acc4  = std::max(std::min(maxValue, acc4), minValue);
            acc5  = std::max(std::min(maxValue, acc5), minValue);
            acc6  = std::max(std::min(maxValue, acc6), minValue);
            acc7  = std::max(std::min(maxValue, acc7), minValue);
            // how to store faster: st4 / transpose /
            c[0] = acc0;
            c[4] = acc1;
            c[4 * 2] = acc2;
            c[4 * 3] = acc3;
            c[4 * 4] = acc4;
            c[4 * 5] = acc5;
            c[4 * 6] = acc6;
            c[4 * 7] = acc7;
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
            float acc0 = initValue;
            float acc1 = initValue;
            float acc2 = initValue;
            float acc3 = initValue;

            const int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {
                const int diff = *dataOffset++;
                const float a0 = a[0];
                const float a1 = a[1];
                const float a2 = a[2];
                const float a3 = a[3];
                const float oneW = *w++;
                // MNN_PRINT("4-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0-3]:", ie, a - A, w - B - 1, c - C, oneW);
                // formatMatrix(a, {4});
                // MNN_PRINT("\n");
                a = a + diff;
                acc0 += a0 * oneW;
                acc1 += a1 * oneW;
                acc2 += a2 * oneW;
                acc3 += a3 * oneW;
            }
            acc0  = std::max(std::min(maxValue, acc0), minValue);
            acc1  = std::max(std::min(maxValue, acc1), minValue);
            acc2  = std::max(std::min(maxValue, acc2), minValue);
            acc3  = std::max(std::min(maxValue, acc3), minValue);
            // how to store faster: st4 / transpose /
            c[0] = acc0;
            c[4] = acc1;
            c[4 * 2] = acc2;
            c[4 * 3] = acc3;
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
            float acc0 = initValue;
            float acc1 = initValue;

            const int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {
                const int diff = *dataOffset++;
                const float a0 = a[0];
                const float a1 = a[1];
                const float oneW = *w++;
                // MNN_PRINT("2-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0-1]:", ie, a - A, w - B - 1, c - C, oneW);
                // formatMatrix(a, {2});
                // MNN_PRINT("\n");
                a = a + diff;
                acc0 += a0 * oneW;
                acc1 += a1 * oneW;
            }
            acc0  = std::max(std::min(maxValue, acc0), minValue);
            acc1  = std::max(std::min(maxValue, acc1), minValue);
            // how to store faster: st4 / transpose /
            c[0] = acc0;
            c[4] = acc1;
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

            const int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {
                const int diff = *dataOffset++;
                const float a0 = a[0];
                const float oneW = *w++;

                // MNN_PRINT("1-loop: ie:%zu, a offset:%ld, c offset:%ld, w offset:%ld, w value:%f, a value[0]:", ie, a - A, w - B - 1, c - C, oneW);
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

void MNNPackedSparseMatMulEpx4(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter, const float* postParameters, const float* bias, unsigned int* NNZMap, int* dataOffsetMap) {

    auto eP = parameter[0] / sizeof(float);
    MNN_ASSERT((eP & 0x03) == 0); // In sparse calculate, eP should be evenly divided by 4
    auto h = parameter[2];
    auto l = parameter[1];
    auto cStride = parameter[3] / sizeof(float);
    auto aStride = eP * l;
    auto hRemain = parameter[4];
    auto bExtraStride = parameter[5] / sizeof(float);
    auto bStride = bExtraStride + l * 4;
    auto hC4 = UP_DIV(h, 4);
    float minValue = -std::numeric_limits<float>().max();
    float maxValue = std::numeric_limits<float>().max();
    if (nullptr != postParameters) {
        minValue = postParameters[2];
        maxValue = postParameters[3];
    }
    // MNN_PRINT("MNNPackedSparseMatMul 16x4 eP:%lu, eSize:%lu, l:%lu, h:%lu, cStride:%lu, aStride:%lu\n", eP, eSize, l, h, cStride, aStride);
    const int sparseBlockOC = 4;
    const float* a = A;
    size_t ie = 0;
    for (ie = 0; ie < eSize && eP <= eSize; ie += eP) {
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

            float initValue[4] = {0, 0, 0, 0};
            if (nullptr != bias) {
                memcpy(initValue, bias + ih, 4 * sizeof(float));
            }
            float acc0[4];
            float acc1[4];
            float acc2[4];
            float acc3[4];
            float acc4[4];
            float acc5[4];
            float acc6[4];
            float acc7[4];
            float acc8[4];
            float acc9[4];
            float acc10[4];
            float acc11[4];
            float acc12[4];
            float acc13[4];
            float acc14[4];
            float acc15[4];
            memcpy(acc0, initValue, 4 * sizeof(float));
            memcpy(acc1, initValue, 4 * sizeof(float));
            memcpy(acc2, initValue, 4 * sizeof(float));
            memcpy(acc3, initValue, 4 * sizeof(float));
            memcpy(acc4, initValue, 4 * sizeof(float));
            memcpy(acc5, initValue, 4 * sizeof(float));
            memcpy(acc6, initValue, 4 * sizeof(float));
            memcpy(acc7, initValue, 4 * sizeof(float));
            memcpy(acc8, initValue, 4 * sizeof(float));
            memcpy(acc9, initValue, 4 * sizeof(float));
            memcpy(acc10, initValue, 4 * sizeof(float));
            memcpy(acc11, initValue, 4 * sizeof(float));
            memcpy(acc12, initValue, 4 * sizeof(float));
            memcpy(acc13, initValue, 4 * sizeof(float));
            memcpy(acc14, initValue, 4 * sizeof(float));
            memcpy(acc15, initValue, 4 * sizeof(float));

            const int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {

                const int diff = *dataOffset++;
                const float a0 = a[0];
                const float a1 = a[1];
                const float a2 = a[2];
                const float a3 = a[3];
                const float a4 = a[4];
                const float a5 = a[5];
                const float a6 = a[6];
                const float a7 = a[7];
                const float a8 = a[8];
                const float a9 = a[9];
                const float a10 = a[10];
                const float a11 = a[11];
                const float a12 = a[12];
                const float a13 = a[13];
                const float a14 = a[14];
                const float a15 = a[15];

                const float wv[4] = {*w++, *w++, *w++, *w++};

                // MNN_PRINT("16-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0-15]:", ie, a - A, w - B - 1, c - C, oneW);
                // formatMatrix(a, {16});
                // MNN_PRINT("\n");
                a = a + diff;
                for (int lane = 0; lane < 4; lane++) {
                    acc0[lane] += a0 * wv[lane];
                    acc1[lane] += a1 * wv[lane];
                    acc2[lane] += a2 * wv[lane];
                    acc3[lane] += a3 * wv[lane];
                    acc4[lane] += a4 * wv[lane];
                    acc5[lane] += a5 * wv[lane];
                    acc6[lane] += a6 * wv[lane];
                    acc7[lane] += a7 * wv[lane];
                    acc8[lane] += a8 * wv[lane];
                    acc9[lane] += a9 * wv[lane];
                    acc10[lane] += a10 * wv[lane];
                    acc11[lane] += a11 * wv[lane];
                    acc12[lane] += a12 * wv[lane];
                    acc13[lane] += a13 * wv[lane];
                    acc14[lane] += a14 * wv[lane];
                    acc15[lane] += a15 * wv[lane];
                }
            }

            for (int lane = 0; lane < 4; lane++) {
                acc0[lane]  = std::max(std::min(maxValue, acc0[lane]), minValue);
                acc1[lane]  = std::max(std::min(maxValue, acc1[lane]), minValue);
                acc2[lane]  = std::max(std::min(maxValue, acc2[lane]), minValue);
                acc3[lane]  = std::max(std::min(maxValue, acc3[lane]), minValue);
                acc4[lane]  = std::max(std::min(maxValue, acc4[lane]), minValue);
                acc5[lane]  = std::max(std::min(maxValue, acc5[lane]), minValue);
                acc6[lane]  = std::max(std::min(maxValue, acc6[lane]), minValue);
                acc7[lane]  = std::max(std::min(maxValue, acc7[lane]), minValue);
                acc8[lane]  = std::max(std::min(maxValue, acc8[lane]), minValue);
                acc9[lane]  = std::max(std::min(maxValue, acc9[lane]), minValue);
                acc10[lane] = std::max(std::min(maxValue, acc10[lane]), minValue);
                acc11[lane] = std::max(std::min(maxValue, acc11[lane]), minValue);
                acc12[lane] = std::max(std::min(maxValue, acc12[lane]), minValue);
                acc13[lane] = std::max(std::min(maxValue, acc13[lane]), minValue);
                acc14[lane] = std::max(std::min(maxValue, acc14[lane]), minValue);
                acc15[lane] = std::max(std::min(maxValue, acc15[lane]), minValue);
            }

            memcpy(c, acc0, 4 * sizeof(float));  // store continuous c
            memcpy(c + 4, acc1, 4 * sizeof(float));
            memcpy(c + 4 * 2, acc2, 4 * sizeof(float));
            memcpy(c + 4 * 3, acc3, 4 * sizeof(float));
            memcpy(c + 4 * 4, acc4, 4 * sizeof(float));
            memcpy(c + 4 * 5, acc5, 4 * sizeof(float));
            memcpy(c + 4 * 6, acc6, 4 * sizeof(float));
            memcpy(c + 4 * 7, acc7, 4 * sizeof(float));
            memcpy(c + 4 * 8, acc8, 4 * sizeof(float));
            memcpy(c + 4 * 9, acc9, 4 * sizeof(float));
            memcpy(c + 4 * 10, acc10, 4 * sizeof(float));
            memcpy(c + 4 * 11, acc11, 4 * sizeof(float));
            memcpy(c + 4 * 12, acc12, 4 * sizeof(float));
            memcpy(c + 4 * 13, acc13, 4 * sizeof(float));
            memcpy(c + 4 * 14, acc14, 4 * sizeof(float));
            memcpy(c + 4 * 15, acc15, 4 * sizeof(float));
        }

        blockC += (h >> 2) * cStride;
        for (; ih < h; ih++) {
            auto ihSubIndex = ih & 0x03;
            auto c = blockC + ihSubIndex;
            const float initValue = nullptr != bias ? bias[ih] : 0;
            float acc0 = initValue;
            float acc1 = initValue;
            float acc2 = initValue;
            float acc3 = initValue;
            float acc4 = initValue;
            float acc5 = initValue;
            float acc6 = initValue;
            float acc7 = initValue;
            float acc8 = initValue;
            float acc9 = initValue;
            float acc10 = initValue;
            float acc11 = initValue;
            float acc12 = initValue;
            float acc13 = initValue;
            float acc14 = initValue;
            float acc15 = initValue;
            const int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {

                const int diff = *dataOffset++;
                const float a0 = a[0];
                const float a1 = a[1];
                const float a2 = a[2];
                const float a3 = a[3];
                const float a4 = a[4];
                const float a5 = a[5];
                const float a6 = a[6];
                const float a7 = a[7];
                const float a8 = a[8];
                const float a9 = a[9];
                const float a10 = a[10];
                const float a11 = a[11];
                const float a12 = a[12];
                const float a13 = a[13];
                const float a14 = a[14];
                const float a15 = a[15];

                const float oneW = *w++;

                // MNN_PRINT("16-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0-15]:", ie, a - A, w - B - 1, c - C, oneW);
                // formatMatrix(a, {16});
                // MNN_PRINT("\n");
                a = a + diff;
                acc0 += a0 * oneW;
                acc1 += a1 * oneW;
                acc2 += a2 * oneW;
                acc3 += a3 * oneW;
                acc4 += a4 * oneW;
                acc5 += a5 * oneW;
                acc6 += a6 * oneW;
                acc7 += a7 * oneW;
                acc8 += a8 * oneW;
                acc9 += a9 * oneW;
                acc10 += a10 * oneW;
                acc11 += a11 * oneW;
                acc12 += a12 * oneW;
                acc13 += a13 * oneW;
                acc14 += a14 * oneW;
                acc15 += a15 * oneW;
            }
            acc0  = std::max(std::min(maxValue, acc0), minValue);
            acc1  = std::max(std::min(maxValue, acc1), minValue);
            acc2  = std::max(std::min(maxValue, acc2), minValue);
            acc3  = std::max(std::min(maxValue, acc3), minValue);
            acc4  = std::max(std::min(maxValue, acc4), minValue);
            acc5  = std::max(std::min(maxValue, acc5), minValue);
            acc6  = std::max(std::min(maxValue, acc6), minValue);
            acc7  = std::max(std::min(maxValue, acc7), minValue);
            acc8  = std::max(std::min(maxValue, acc8), minValue);
            acc9  = std::max(std::min(maxValue, acc9), minValue);
            acc10 = std::max(std::min(maxValue, acc10), minValue);
            acc11 = std::max(std::min(maxValue, acc11), minValue);
            acc12 = std::max(std::min(maxValue, acc12), minValue);
            acc13 = std::max(std::min(maxValue, acc13), minValue);
            acc14 = std::max(std::min(maxValue, acc14), minValue);
            acc15 = std::max(std::min(maxValue, acc15), minValue);

            // how to store faster: st4 / transpose /
            c[0] = acc0;
            c[4] = acc1;
            c[4 * 2] = acc2;
            c[4 * 3] = acc3;
            c[4 * 4] = acc4;
            c[4 * 5] = acc5;
            c[4 * 6] = acc6;
            c[4 * 7] = acc7;
            c[4 * 8] = acc8;
            c[4 * 9] = acc9;
            c[4 * 10] = acc10;
            c[4 * 11] = acc11;
            c[4 * 12] = acc12;
            c[4 * 13] = acc13;
            c[4 * 14] = acc14;
            c[4 * 15] = acc15;
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
            float initValue[4] = {0, 0, 0, 0};
            if (nullptr != bias) {
                memcpy(initValue, bias + ih, 4 * sizeof(float));
            }
            float acc0[4];
            float acc1[4];
            float acc2[4];
            float acc3[4];
            float acc4[4];
            float acc5[4];
            float acc6[4];
            float acc7[4];
            memcpy(acc0, initValue, 4 * sizeof(float));
            memcpy(acc1, initValue, 4 * sizeof(float));
            memcpy(acc2, initValue, 4 * sizeof(float));
            memcpy(acc3, initValue, 4 * sizeof(float));
            memcpy(acc4, initValue, 4 * sizeof(float));
            memcpy(acc5, initValue, 4 * sizeof(float));
            memcpy(acc6, initValue, 4 * sizeof(float));
            memcpy(acc7, initValue, 4 * sizeof(float));
            const int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {

                const int diff = *dataOffset++;
                const float a0 = a[0];
                const float a1 = a[1];
                const float a2 = a[2];
                const float a3 = a[3];
                const float a4 = a[4];
                const float a5 = a[5];
                const float a6 = a[6];
                const float a7 = a[7];
                const float wv[4] = {*w++, *w++, *w++, *w++};
                // MNN_PRINT("16-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0-15]:", ie, a - A, w - B - 1, c - C, oneW);
                // formatMatrix(a, {16});
                // MNN_PRINT("\n");
                a = a + diff;
                for (int lane = 0; lane < 4; lane++) {
                    acc0[lane] += a0 * wv[lane];
                    acc1[lane] += a1 * wv[lane];
                    acc2[lane] += a2 * wv[lane];
                    acc3[lane] += a3 * wv[lane];
                    acc4[lane] += a4 * wv[lane];
                    acc5[lane] += a5 * wv[lane];
                    acc6[lane] += a6 * wv[lane];
                    acc7[lane] += a7 * wv[lane];
                }
            }

            for (int lane = 0; lane < 4; lane++) {
                acc0[lane]  = std::max(std::min(maxValue, acc0[lane]), minValue);
                acc1[lane]  = std::max(std::min(maxValue, acc1[lane]), minValue);
                acc2[lane]  = std::max(std::min(maxValue, acc2[lane]), minValue);
                acc3[lane]  = std::max(std::min(maxValue, acc3[lane]), minValue);
                acc4[lane]  = std::max(std::min(maxValue, acc4[lane]), minValue);
                acc5[lane]  = std::max(std::min(maxValue, acc5[lane]), minValue);
                acc6[lane]  = std::max(std::min(maxValue, acc6[lane]), minValue);
                acc7[lane]  = std::max(std::min(maxValue, acc7[lane]), minValue);
            }

            memcpy(c, acc0, 4 * sizeof(float));  // store continuous c
            memcpy(c + 4, acc1, 4 * sizeof(float));
            memcpy(c + 4 * 2, acc2, 4 * sizeof(float));
            memcpy(c + 4 * 3, acc3, 4 * sizeof(float));
            memcpy(c + 4 * 4, acc4, 4 * sizeof(float));
            memcpy(c + 4 * 5, acc5, 4 * sizeof(float));
            memcpy(c + 4 * 6, acc6, 4 * sizeof(float));
            memcpy(c + 4 * 7, acc7, 4 * sizeof(float));
        }
        blockC += (ih >> 2) * cStride;
        for (; ih < h; ih++) {
            auto ihSubIndex = ih & 0x03;
            auto c = blockC + ihSubIndex;
            const float initValue = nullptr != bias ? bias[ih] : 0;
            float acc0 = initValue;
            float acc1 = initValue;
            float acc2 = initValue;
            float acc3 = initValue;
            float acc4 = initValue;
            float acc5 = initValue;
            float acc6 = initValue;
            float acc7 = initValue;

            const int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {
                const int diff = *dataOffset++;
                const float a0 = a[0];
                const float a1 = a[1];
                const float a2 = a[2];
                const float a3 = a[3];
                const float a4 = a[4];
                const float a5 = a[5];
                const float a6 = a[6];
                const float a7 = a[7];
                const float oneW = *w++;
                // MNN_PRINT("8-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0-7]:", ie, a - A, w - B - 1, c - C, oneW);
                // formatMatrix(a, {8});
                // MNN_PRINT("\n");
                a = a + diff;
                acc0 += a0 * oneW;
                acc1 += a1 * oneW;
                acc2 += a2 * oneW;
                acc3 += a3 * oneW;
                acc4 += a4 * oneW;
                acc5 += a5 * oneW;
                acc6 += a6 * oneW;
                acc7 += a7 * oneW;
            }
            acc0  = std::max(std::min(maxValue, acc0), minValue);
            acc1  = std::max(std::min(maxValue, acc1), minValue);
            acc2  = std::max(std::min(maxValue, acc2), minValue);
            acc3  = std::max(std::min(maxValue, acc3), minValue);
            acc4  = std::max(std::min(maxValue, acc4), minValue);
            acc5  = std::max(std::min(maxValue, acc5), minValue);
            acc6  = std::max(std::min(maxValue, acc6), minValue);
            acc7  = std::max(std::min(maxValue, acc7), minValue);
            // how to store faster: st4 / transpose /
            c[0] = acc0;
            c[4] = acc1;
            c[4 * 2] = acc2;
            c[4 * 3] = acc3;
            c[4 * 4] = acc4;
            c[4 * 5] = acc5;
            c[4 * 6] = acc6;
            c[4 * 7] = acc7;
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
            float initValue[4] = {0, 0, 0, 0};
            if (nullptr != bias) {
                memcpy(initValue, bias + ih, 4 * sizeof(float));
            }
            float acc0[4];
            float acc1[4];
            float acc2[4];
            float acc3[4];
            memcpy(acc0, initValue, 4 * sizeof(float));
            memcpy(acc1, initValue, 4 * sizeof(float));
            memcpy(acc2, initValue, 4 * sizeof(float));
            memcpy(acc3, initValue, 4 * sizeof(float));

            const int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {
                const int diff = *dataOffset++;
                const float a0 = a[0];
                const float a1 = a[1];
                const float a2 = a[2];
                const float a3 = a[3];
                const float wv[4] = {*w++, *w++, *w++, *w++};
                // MNN_PRINT("16-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0-15]:", ie, a - A, w - B - 1, c - C, oneW);
                // formatMatrix(a, {16});
                // MNN_PRINT("\n");
                a = a + diff;
                for (int lane = 0; lane < 4; lane++) {
                    acc0[lane] += a0 * wv[lane];
                    acc1[lane] += a1 * wv[lane];
                    acc2[lane] += a2 * wv[lane];
                    acc3[lane] += a3 * wv[lane];
                }
            }

            for (int lane = 0; lane < 4; lane++) {
                acc0[lane]  = std::max(std::min(maxValue, acc0[lane]), minValue);
                acc1[lane]  = std::max(std::min(maxValue, acc1[lane]), minValue);
                acc2[lane]  = std::max(std::min(maxValue, acc2[lane]), minValue);
                acc3[lane]  = std::max(std::min(maxValue, acc3[lane]), minValue);
            }

            memcpy(c, acc0, 4 * sizeof(float));  // store continuous c
            memcpy(c + 4, acc1, 4 * sizeof(float));
            memcpy(c + 4 * 2, acc2, 4 * sizeof(float));
            memcpy(c + 4 * 3, acc3, 4 * sizeof(float));
        }
        blockC += (ih >> 2) * cStride;
        for (; ih < h; ih++) {
            auto ihSubIndex = ih & 0x03;
            auto c = blockC + ihSubIndex;
            const float initValue = nullptr != bias ? bias[ih] : 0;
            float acc0 = initValue;
            float acc1 = initValue;
            float acc2 = initValue;
            float acc3 = initValue;

            const int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {
                const int diff = *dataOffset++;
                const float a0 = a[0];
                const float a1 = a[1];
                const float a2 = a[2];
                const float a3 = a[3];
                const float oneW = *w++;
                // MNN_PRINT("4-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0-3]:", ie, a - A, w - B - 1, c - C, oneW);
                // formatMatrix(a, {4});
                // MNN_PRINT("\n");
                a = a + diff;
                acc0 += a0 * oneW;
                acc1 += a1 * oneW;
                acc2 += a2 * oneW;
                acc3 += a3 * oneW;
            }
            acc0  = std::max(std::min(maxValue, acc0), minValue);
            acc1  = std::max(std::min(maxValue, acc1), minValue);
            acc2  = std::max(std::min(maxValue, acc2), minValue);
            acc3  = std::max(std::min(maxValue, acc3), minValue);
            // how to store faster: st4 / transpose /
            c[0] = acc0;
            c[4] = acc1;
            c[4 * 2] = acc2;
            c[4 * 3] = acc3;
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
            float initValue[4] = {0, 0, 0, 0};
            if (nullptr != bias) {
                memcpy(initValue, bias + ih, 4 * sizeof(float));
            }
            float acc0[4];
            float acc1[4];
            memcpy(acc0, initValue, 4 * sizeof(float));
            memcpy(acc1, initValue, 4 * sizeof(float));
            const int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {

                const int diff = *dataOffset++;
                const float a0 = a[0];
                const float a1 = a[1];
                const float wv[4] = {*w++, *w++, *w++, *w++};
                // MNN_PRINT("16-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0-15]:", ie, a - A, w - B - 1, c - C, oneW);
                // formatMatrix(a, {16});
                // MNN_PRINT("\n");
                a = a + diff;
                for (int lane = 0; lane < 4; lane++) {
                    acc0[lane] += a0 * wv[lane];
                    acc1[lane] += a1 * wv[lane];
                }
            }

            for (int lane = 0; lane < 4; lane++) {
                acc0[lane]  = std::max(std::min(maxValue, acc0[lane]), minValue);
                acc1[lane]  = std::max(std::min(maxValue, acc1[lane]), minValue);
            }

            memcpy(c, acc0, 4 * sizeof(float));  // store continuous c
            memcpy(c + 4, acc1, 4 * sizeof(float));
        }
        blockC += (ih >> 2) * cStride;
        for (; ih < h; ih++) {
            auto ihPack = ih >> 2;
            auto ihSubIndex = ih & 0x03;
            auto c = blockC + ihSubIndex;
            const float initValue = nullptr != bias ? bias[ih] : 0;
            float acc0 = initValue;
            float acc1 = initValue;

            const int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {
                const int diff = *dataOffset++;
                const float a0 = a[0];
                const float a1 = a[1];
                const float oneW = *w++;
                // MNN_PRINT("2-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0-1]:", ie, a - A, w - B - 1, c - C, oneW);
                // formatMatrix(a, {2});
                // MNN_PRINT("\n");
                a = a + diff;
                acc0 += a0 * oneW;
                acc1 += a1 * oneW;
            }
            acc0  = std::max(std::min(maxValue, acc0), minValue);
            acc1  = std::max(std::min(maxValue, acc1), minValue);
            // how to store faster: st4 / transpose /
            c[0] = acc0;
            c[4] = acc1;
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
            float initValue[4] = {0, 0, 0, 0};
            if (nullptr != bias) {
                memcpy(initValue, bias + ih, 4 * sizeof(float));
            }
            float acc0[4];
            memcpy(acc0, initValue, 4 * sizeof(float));
            const int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {

                const int diff = *dataOffset++;
                const float a0 = a[0];
                const float wv[4] = {*w++, *w++, *w++, *w++};
                // MNN_PRINT("16-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0-15]:", ie, a - A, w - B - 1, c - C, oneW);
                // formatMatrix(a, {16});
                // MNN_PRINT("\n");
                a = a + diff;
                for (int lane = 0; lane < 4; lane++) {
                    acc0[lane] += a0 * wv[lane];
                }
            }

            for (int lane = 0; lane < 4; lane++) {
                acc0[lane]  = std::max(std::min(maxValue, acc0[lane]), minValue);
            }
            memcpy(c, acc0, 4 * sizeof(float));  // store continuous c
        }
        blockC += (ih >> 2) * cStride;
        for (; ih < h; ih++) {
            auto ihSubIndex = ih & 0x03;
            auto c = blockC + ihSubIndex;
            const float initValue = nullptr != bias ? bias[ih] : 0;
            float acc0 = initValue;

            const int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {
                const int diff = *dataOffset++;
                const float a0 = a[0];
                const float oneW = *w++;

                // MNN_PRINT("1-loop: ie:%zu, a offset:%ld, c offset:%ld, w offset:%ld, w value:%f, a value[0]:", ie, a - A, w - B - 1, c - C, oneW);
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

#ifndef MNN_USE_SSE
#ifndef MNN_USE_NEON
void MNNTranspose32Bit(int32_t* dstO, const int32_t* srcO, int32_t* dim) {
    int w = dim[0];
    int h = dim[1];
    int srcStride = dim[2];
    int dstStride = dim[3];
    for (int i=0; i<h; ++i) {
        auto si = srcO + i;
        auto di = dstO + i * dstStride;
        for (int j=0; j<w; ++j) {
            auto sj = si + j * srcStride;
            auto dj = di + j;
            *dj = *sj;
        }
    }
}
void MNNTranspose16Bit(int16_t* dstO, const int16_t* srcO, int32_t* dim) {
    int w = dim[0];
    int h = dim[1];
    int srcStride = dim[2];
    int dstStride = dim[3];
    for (int i=0; i<h; ++i) {
        auto si = srcO + i;
        auto di = dstO + i * dstStride;
        for (int j=0; j<w; ++j) {
            auto sj = si + j * srcStride;
            auto dj = di + j;
            *dj = *sj;
        }
    }
}
#endif
void MNNFunctionInit() {
    // Do nothing
}
#endif

#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif

#define UNIT 4
using Vec4 = MNN::Math::Vec<float, 4>;

#ifndef MNN_USE_NEON

#ifndef MNN_USE_SSE

void MNNCopyC4WithStride(const float* source, float* dest, size_t srcStride, size_t dstStride, size_t count) {
    for (int i = 0; i < count; ++i) {
        auto s = source + i * srcStride;
        auto d = dest + i * dstStride;
        for (int j = 0; j < 4; ++j) {
            d[j] = s[j];
        }
    }
}

void MNNAddC4WithStride(const float* source, float* dest, size_t srcStride, size_t dstStride, size_t count) {
    for (int i = 0; i < count; ++i) {
        auto s = source + i * srcStride;
        auto d = dest + i * dstStride;
        for (int j = 0; j < 4; ++j) {
            d[j] += s[j];
        }
    }
}

void MNNReluWithSlopeChannel(float* dst, const float* src, const float* slope, size_t sizeQuad, size_t depthQuad) {
    for (int j = 0; j < depthQuad; j++) {
        const float* slopeZ = slope + 4 * j;
        const float* srcZ   = src + 4 * j * sizeQuad;
        float* dstZ         = dst + 4 * j * sizeQuad;
        for (int i = 0; i < sizeQuad; i++) {
            for (int c = 0; c < 4; c++) {
                if (srcZ[4 * i + c] < 0) {
                    dstZ[4 * i + c] = srcZ[4 * i + c] * slopeZ[c];
                } else {
                    dstZ[4 * i + c] = srcZ[4 * i + c];
                }
            }
        }
    }
}

void MNNPackC4(float* dst, const float* src, size_t area, size_t depth, int* areaOffset) {
    MNNPackC4Common<float>(dst, src, area, depth, areaOffset);
}

void MNNUnpackC4(float* dst, const float* src, size_t area, size_t depth, int* areaOffset) {
    MNNUnpackC4Common<float>(dst, src, area, depth, areaOffset);
}

void MNNExpC8(float* dest, const float* source, float* offset, const float* parameters, size_t countC8) {
    auto count = countC8 * 8;
    auto param = parameters[0];
    float xLimit = 87;
    float summer = offset[3];
    for (int i = 0; i < count; ++i) {
        auto x         = source[i] * offset[0] + offset[2];
        x = ALIMAX(x, -xLimit);
        x = ALIMIN(x, xLimit);
        int div        = (x * parameters[1]);
        int div2       = (div + 127) << 23;
        auto xReamin   = x - div * param;
        float expBasic = *(float*)(&div2);
        auto t = xReamin * 0.25f;
        auto expRemain =
        ((((parameters[7] * t + parameters[6]) * t + parameters[5]) * t + parameters[4]) * t + 1.0f) * t +
            1.0f;
        expRemain = expRemain * expRemain;
        expRemain = expRemain * expRemain;
        dest[i] = expBasic * expRemain + offset[1];
        summer+= dest[i];
    }
    offset[3] = summer;
}

void MNNSoftmax(float* softmaxDst, const float* softmaxSrc, float* runningMax, float* runningSum, float* updateScale, int outside, int reduceSize, int kvSeqOffset, int validOffset, int pack, bool mask) {

    // source shape: [reduceSizeOuter, outside, reduceSizeInner]
    // for C4, [up_div(reduceSize,4), outside,4] => reduceSizeOuter=up_div(reduceSize,4), reduceSizeInner=4
    // for C,  [outside, reduceSize]             => reduceSizeOuter=1, reduceSizeInner=reduceSize

    const int packUnit = 4;
    int reduceSizeOuter = 1;
    int reduceSizeInner = reduceSize;
    int stride0         = packUnit;
    if (pack > 1) {
        reduceSizeOuter = UP_DIV(reduceSize, pack);
        reduceSizeInner = pack;
        stride0         = outside * reduceSizeInner;
    }

    float exprOffset[4] = {1.0f, 0.0f, 0.0f, 0.0f };
    for (int k = 0; k < outside; ++k) {
        exprOffset[3] = 0.0f; // init sum to zero for each outer loop
        if (mask && kvSeqOffset > k + validOffset) {
            if (updateScale){
                updateScale[k] = 1;
            }
            for (int j = 0; j < reduceSizeOuter; ++j) {
                int i = 0;
                for (; i < reduceSizeInner; i += packUnit) {
                    auto destPtr = softmaxDst + j * stride0 + k * reduceSizeInner + i;
                    memset(destPtr, 0, packUnit * sizeof(float));
                }
                if (i < reduceSizeInner) {
                    memset(softmaxDst + j * stride0 + k * reduceSizeInner + i, 0, (reduceSizeInner - i) * sizeof(float));
                }
            }
            continue;
        }

        const int validReduceSize = mask ? ALIMIN(reduceSize, k + (validOffset + 1) - kvSeqOffset) : reduceSize;
        const int remain = validReduceSize % packUnit;
        const int sizeDiv = validReduceSize / packUnit;

        // 1. newMax
        float oldMax = std::numeric_limits<float>::lowest();
        if (runningMax) {
            oldMax = runningMax[k];
        }

        float newMax = std::numeric_limits<float>::lowest();

        for (int j = 0; j < sizeDiv; ++j) {
            auto srcPtr = softmaxSrc + j * stride0 + k * reduceSizeInner;
            for (int i = 0; i < packUnit; ++i) {
                newMax = ALIMAX(newMax, srcPtr[i]);
            }
        }

        if (remain > 0) {
            auto srcPtr = softmaxSrc + sizeDiv * stride0  + k * reduceSizeInner;
            for (int i = 0; i < remain; ++i) {
                newMax = ALIMAX(newMax, srcPtr[i]);
            }
        }

        const float finalMax = ALIMAX(oldMax, newMax);

        // 2. exp(x - finalMax)
        exprOffset[2] = -finalMax;

        for (int j = 0; j < sizeDiv; ++j) {
            auto idx = j * stride0 + k * reduceSizeInner;
            auto srcPtr = softmaxSrc + idx;
            auto dstPtr = softmaxDst + idx;
            MNNExp(dstPtr, srcPtr, exprOffset, packUnit);
        }

        float sum = exprOffset[3];

        if (remain > 0) {
            auto idx = sizeDiv * stride0  + k * reduceSizeInner;
            auto srcPtr = softmaxSrc + idx;
            auto dstPtr = softmaxDst + idx;

            for(int i = 0; i < remain; ++i) {
                float val = expf(srcPtr[i] - finalMax);
                sum += val;
                dstPtr[i] = val;
            }
        }

        // 3.
        if (runningMax != nullptr && runningSum != nullptr && updateScale != nullptr) {
            // update runningSum, runningMax, scale
            float scaleForSum = expf(oldMax - finalMax);
            runningSum[k] = runningSum[k] * scaleForSum + sum;
            runningMax[k] = finalMax;
            updateScale[k] = scaleForSum;
        } else {
            // Normalization
            if (runningMax != nullptr && runningSum != nullptr) {
                sum += runningSum[k] * expf(oldMax - finalMax);
            }
            float scale = 1.0f / (sum + 1e-20f);

            for (int j = 0; j < sizeDiv; ++j) {
                auto pDest = softmaxDst + j * stride0 + k * reduceSizeInner;
                for (int i = 0; i < packUnit; ++i) {
                    pDest[i] = pDest[i] * scale;
                }
            }
            if (remain > 0) {
                auto pDest = softmaxDst + sizeDiv * stride0 + k * reduceSizeInner;
                for (int i = 0; i < remain; ++i) {
                    pDest[i] = pDest[i] * scale;
                }
            }
        }

        // 4. memset 0
        if (pack > 1) {
            if (validReduceSize % packUnit > 0) {
                memset(softmaxDst + sizeDiv * stride0 + k * reduceSizeInner + (validReduceSize % packUnit), 0, (packUnit - (validReduceSize % packUnit)) * sizeof(float));
            }
            auto validDiv4 = UP_DIV(validReduceSize, packUnit);
            auto allDiv4 = UP_DIV(reduceSize, packUnit);
            for (int j = validDiv4; j < allDiv4; ++j) {
                auto destPtr = softmaxDst + j * stride0 + k * reduceSizeInner;
                memset(destPtr, 0, packUnit * sizeof(float));
            }
        } else {
            memset(softmaxDst + k * reduceSizeInner + validReduceSize, 0, (reduceSize - validReduceSize) * sizeof(float));
        }
    }
}

void MNNReluInt8(int8_t* dst, const int8_t* src, size_t size, ssize_t zeroPoint) {
    for (int i = 0; i < size; ++i) {
        if (src[i] < zeroPoint) {
            dst[i] = zeroPoint;
        } else {
            dst[i] = src[i];
        }
    }
}
#endif // no MNN_USE_SSE



void MNNExp(float* dst, const float* src, float* offset, size_t dataSize) {
    int countC8        = static_cast<int32_t>(dataSize) / 8;
    int remain = static_cast<int32_t>(dataSize) % 8;
    static const float parameters[] = {
        (float)logf(2.0f), 1.0f / (float)logf(2.0f), 0.25f, 1.0f, 0.5f, 1.0f / 6.0f, 1.0f / 24.0f, 1.0f / 120.0f};
    if (countC8 > 0) {
        // Align to eight so asm is easier to write
        MNNExpC8(dst, src, offset, parameters, countC8);
    }
    if (remain > 0) {
        auto param = parameters[0];
        float xLimit = 87;
        float summer = offset[3];
        auto source = src + countC8 * 8;
        auto dest = dst + countC8 * 8;
        for (int i = 0; i < remain; ++i) {
            auto x         = source[i] * offset[0] + offset[2];
            x = ALIMAX(x, -xLimit);
            x = ALIMIN(x, xLimit);
            int div        = (x * parameters[1]);
            int div2       = (div + 127) << 23;
            auto xReamin   = x - div * param;
            float expBasic = *(float*)(&div2);
            auto t = xReamin * 0.25f;
            auto expRemain =
            ((((parameters[7] * t + parameters[6]) * t + parameters[5]) * t + parameters[4]) * t + 1.0f) * t +
                1.0f;
            expRemain = expRemain * expRemain;
            expRemain = expRemain * expRemain;
            dest[i] = expBasic * expRemain + offset[1];
            summer+= dest[i];
        }
        offset[3] = summer;
    }
}


inline void smartCopy(void* dest, const void* src, size_t size) {
    switch (size) {
        case 1:
            *(uint8_t*)dest = *(const uint8_t*)src;
            break;
        case 2:
            *(uint16_t*)dest = *(const uint16_t*)src;
            break;
        case 4:
            *(uint32_t*)dest = *(const uint32_t*)src;
            break;
        case 8:
            *(uint64_t*)dest = *(const uint64_t*)src;
            break;
        default:
            ::memcpy(dest, src, size);
            break;
    }
}

void MNNPackForMatMul_A(float* dst, const float* src, size_t E, size_t L, size_t eP, size_t lP, size_t bytes) {
    if (E == 0 || L == 0) {
        return;
    }
    // [e,l] -> [e/eP,l/lP,eP,lP]
    auto eU = UP_DIV(E, eP);
    auto lU = UP_DIV(L, lP);
    if (lP > 1) {
        const int lC = L / lP;
        const int lR = L % lP;
        const size_t copySizeBytes = (size_t)lP * bytes;

        const size_t srcStride0 = (size_t)L * bytes;
        const size_t dstStride0 = (size_t)lU * eP * lP * bytes;
        const size_t dstStride1 = eP * lP * bytes;
        const size_t dstStride2 = lP * bytes;

        for (int i = 0; i < eU; ++i) {
            const int xC = ALIMIN(eP, E - i * eP);
            const uint8_t* APtr = (uint8_t*)src + (i * eP) * srcStride0;
            uint8_t* ADst = (uint8_t*)dst + i * dstStride0;

            if (lC > 0) {
                for (int x = 0; x < xC; ++x) {
                    auto srcBase = APtr + x * srcStride0;
                    auto destBase = ADst + x * dstStride2;

                    for (int yy = 0; yy < lC; ++yy) {
                        auto srcPtr = srcBase + (size_t)yy * copySizeBytes;
                        auto destPtr = destBase + (size_t)yy * dstStride1;

                        smartCopy(destPtr, srcPtr, copySizeBytes);
                    }
                }
            }

            if (lR > 0) {
                const int yy = lC;
                const size_t remainderCopyBytes = (size_t)lR * bytes;

                for (int x = 0; x < xC; ++x) {
                    auto srcPtr = APtr + x * srcStride0 + lC * lP * bytes;
                    auto destPtr = ADst + lC * dstStride1 + x * dstStride2;// (lC * eP * lP + x * lP) * bytes;

                    ::memcpy(destPtr, srcPtr, remainderCopyBytes);
                    ::memset(destPtr + remainderCopyBytes, 0, copySizeBytes - remainderCopyBytes);
                }
            }
        }
    } else { // lP=1
        // e, l -> eU, l, eP, 1
        for (int i = 0; i < eU; ++i) {
            const int xC = ALIMIN(eP, E - i * eP);
            auto APtr = (uint8_t*)src + (i * eP * L) * bytes;
            auto ADst = (uint8_t*)dst + (i * lU * eP * lP) * bytes;
            int dims[4] = {xC, (int)L, (int)L, (int)eP};
            if (bytes == 2) {
                auto S = (const int16_t*)APtr;
                auto D = (int16_t*)ADst;
                MNNTranspose16Bit(D, S, dims);
            } else if (bytes == 4) {
                auto S = (const int32_t*)APtr;
                auto D = (int32_t*)ADst;
                MNNTranspose32Bit(D, S, dims);
            }
        }
    }
}

void MNNMaxFloat(float* input, float* maxBuffer, int32_t inputCountUnit) {
    for (int i = 0; i < inputCountUnit; i++) {
        for (int j = 0; j < UNIT; j++) {
            for (int m = 0; m < 2; m++) {
                maxBuffer[j] = std::max(input[i * UNIT * 2 + j * 2 + m], maxBuffer[j]);
            }
        }
    }
}
void MNNMinFloat(float* input, float* minBuffer, int32_t inputCountUnit) {
    for (int i = 0; i < inputCountUnit; i++) {
        for (int j = 0; j < UNIT; j++) {
            for (int m = 0; m < 2; m++) {
                minBuffer[j] = std::min(input[i * UNIT * 2 + j * 2 + m], minBuffer[j]);
            }
        }
    }
}
void MNNScaleAndAddBias(float* dst, const float* src, const float* bias, const float* alpha, size_t planeNumber,
                        size_t biasNumber) {
    for (int z = 0; z < biasNumber; ++z) {
        float* dstZ         = dst + planeNumber * 4 * z;
        const float* srcZ   = src + planeNumber * 4 * z;
        auto biasZ = Vec4::load(bias + 4 * z);
        auto alphaZ = Vec4::load(alpha + 4 * z);
        for (int p = 0; p < planeNumber; ++p) {
            float* dstX       = dstZ + 4 * p;
            const float* srcX = srcZ + 4 * p;
            Vec4::save(dstX, (Vec4::load(srcX) * alphaZ) + biasZ);
        }
    }
}



void MNNUInt8ToInt16WithOffsetC4Common(int16_t* dst, const uint8_t* src, size_t zeroPoint, size_t sizeQuad,
                                       size_t dstStride, size_t srcStride) {
    dstStride /= sizeof(int16_t);
    srcStride /= sizeof(uint8_t);
    for (int z = 0; z < sizeQuad; ++z) {
        auto dstZ = dst + dstStride * z;
        auto srcZ = src + srcStride * z;
        for (int j = 0; j < 4; ++j) {
            dstZ[j] = (int16_t)((int32_t)srcZ[j] - (int32_t)zeroPoint);
        }
    }
}

void MNNUInt8ToInt16WithOffsetC4Fast(int16_t* colAddr, const uint8_t* srcStart, size_t zeroPoint, size_t sizeQuad,
                                     size_t depthQuad, size_t dstZStep, size_t srcZStep) {
    dstZStep /= sizeof(int16_t);
    srcZStep /= sizeof(uint8_t);
    for (int sz = 0; sz < depthQuad; ++sz) {
        auto dstZ = colAddr + sz * dstZStep;
        auto srcZ = srcStart + sz * srcZStep;
        MNNUInt8ToInt16WithOffsetC4Common(dstZ, srcZ, zeroPoint, sizeQuad, 4 * sizeof(int16_t), 4 * sizeof(uint8_t));
    }
}

void MNNPowC8(float* dest, const float* source, const float* powfParam, size_t betaInt, size_t countC8) {
    const int count          = countC8 * 8;
    const float powfConstant = powfParam[6];
    for (int i = 0; i < count; ++i) {
        float result = 1, x, xInv = 1 / source[i];
        for (int j = 0; j < betaInt; result *= xInv, ++j)
            ;
        for (x = source[i]; x >= 1.25; x /= 1.5, result *= powfConstant)
            ;
        float t = x - 1;
        float powRemain =
            powfParam[0] +
            t * (powfParam[1] + t * (powfParam[2] + t * (powfParam[3] + t * (powfParam[4] + t * powfParam[5]))));
        result *= powRemain;
        dest[i] = result;
    }
}
#endif // no MNN_USE_NEON

void MNNGridSampleComputeCord(float* dst, const float* src, size_t inH, size_t inW, size_t outH, size_t outW, bool alignCorners) {
    float a = alignCorners ? 1.0f : 0.0f;
    float b = alignCorners ? 0.0f : 1.0f;
    int area = outH * outW;
    float kx = 0.5f * ((float)inW - a);
    float bx = 0.5f * ((float)inW - a - b);
    float ky = 0.5f * ((float)inH - a);
    float by = 0.5f * ((float)inH - a - b);
    for (int w = 0; w < area; ++w) {
        auto x = src[2 * w + 0];
        auto y = src[2 * w + 1];
        dst[2 * w + 0] = kx * x + bx;
        dst[2 * w + 1] = ky * y + by;
    }
}
void MNNGridSampleComputeCord3D(float* dst, const float* src, size_t inD, size_t inH, size_t inW, size_t outD, size_t outH, size_t outW, bool alignCorners) {
    int strideD = outH * outW * 3;
    int strideH = outW * 3;
    float a = alignCorners ? 1.0f : 0.0f;
    float b = alignCorners ? 0.0f : 1.0f;
    int area = outD * outH * outW;
    float kx = 0.5f * ((float)inW - a);
    float bx = 0.5f * ((float)inW - a - b);
    float ky = 0.5f * ((float)inH - a);
    float by = 0.5f * ((float)inH - a - b);
    float kz = 0.5f * ((float)inD - a);
    float bz = 0.5f * ((float)inD - a - b);

    for (int w=0; w<area; ++w) {
        auto x = src[3 * w + 0];
        auto y = src[3 * w + 1];
        auto z = src[3 * w + 2];
        dst[3 * w + 0] = kx * x + bx;
        dst[3 * w + 1] = ky * y + by;
        dst[3 * w + 2] = kz * z + bz;
    }
}

#ifndef MNN_USE_SSE
void MNNNorm(float *dst, const float *src, const float *gamma, const float *beta, float epsilon, size_t size, bool RMSNorm) {
    float mean = 0;
    if(false == RMSNorm){
        float sum = 0.f;
        for (int j = 0; j < size; ++j) {
            sum += src[j];
        }
        mean = sum / size;
    }
    float square_sum = 0.f;
    for (int j = 0; j < size; ++j) {
        square_sum += (src[j] - mean) * (src[j] - mean);
    }
#ifdef __aarch64__
    auto vs = vadd_f32(vdiv_f32(vdup_n_f32(square_sum), vdup_n_f32(size)), vdup_n_f32(epsilon));
    auto vecs = vdiv_f32(vdup_n_f32(1.0f), vsqrt_f32(vs));
    float vars[2];
    vst1_f32(vars, vecs);
    float variable = vars[0];
#else
    float variable = square_sum / size;
    variable = 1.f / std::sqrt(variable + epsilon);
#endif

    if (gamma && beta) {
        for (int j = 0; j < size; ++j) {
            dst[j] = (src[j] - mean) * variable * gamma[j] + beta[j];
        }
    } else {
        for (int j = 0; j < size; ++j) {
            dst[j] = (src[j] - mean) * variable;
        }
    }
}
#endif

void MNNRoiPoolingMax(float* dst, const float* src, int hLen, int wLen, int iw) {
    Vec4 max = Vec4(-FLT_MAX);
    for (int h = 0; h < hLen; h++, src += iw * UNIT) {
        for (int w = 0; w < wLen; w++) {
            Vec4 in = Vec4::load(src + w * UNIT);
            max = Vec4::max(max, in);
        }
    }
    Vec4::save(dst, max);
}

void MNNRoiAlignMax(float* dst, const float* src, const std::vector<std::vector<int>> &vecPos, const std::vector<std::vector<float>> &vecArea, int samplingRatioArea, int pooledHeight, int pooledWidth) {
    for (int h = 0; h < pooledHeight; ++h, dst += pooledWidth * UNIT) {
        int preCalcIdx = h * pooledWidth * samplingRatioArea;
        for (int w = 0; w < pooledWidth; ++w) {
            Vec4 res = Vec4(-FLT_MAX);
            for (int i = 0; i < samplingRatioArea; ++i) {
                const std::vector<int>& pos    = vecPos[preCalcIdx];
                const std::vector<float>& area = vecArea[preCalcIdx];

                Vec4 val0 = Vec4::load(src + pos[0] * UNIT);
                Vec4 val1 = Vec4::load(src + pos[1] * UNIT);
                Vec4 val2 = Vec4::load(src + pos[2] * UNIT);
                Vec4 val3 = Vec4::load(src + pos[3] * UNIT);
                Vec4 mla  = val0 * area[0];
                mla       = Vec4::fma(mla, val1, area[1]);
                mla       = Vec4::fma(mla, val2, area[2]);
                mla       = Vec4::fma(mla, val3, area[3]);
                res       = Vec4::max(res, mla);
                preCalcIdx++;
            }
            Vec4::save(dst + w * UNIT, res);
        }
    }
}

void MNNRoiAlignAvg(float* dst, const float* src, const std::vector<std::vector<int>> &vecPos, const std::vector<std::vector<float>> &vecArea, int samplingRatioArea, int pooledHeight, int pooledWidth) {
    float invSamplingCnt = 1.f / samplingRatioArea;
    for (int h = 0; h < pooledHeight; ++h, dst += pooledWidth * UNIT) {
        int preCalcIdx = h * pooledWidth * samplingRatioArea;
        for (int w = 0; w < pooledWidth; ++w) {
            Vec4 res = Vec4(0.f);
            for (int i = 0; i < samplingRatioArea; ++i) {
                const std::vector<int>& pos    = vecPos[preCalcIdx];
                const std::vector<float>& area = vecArea[preCalcIdx];

                Vec4 val0 = Vec4::load(src + pos[0] * UNIT);
                Vec4 val1 = Vec4::load(src + pos[1] * UNIT);
                Vec4 val2 = Vec4::load(src + pos[2] * UNIT);
                Vec4 val3 = Vec4::load(src + pos[3] * UNIT);
                Vec4 mla  = val0 * area[0];
                mla       = Vec4::fma(mla, val1, area[1]);
                mla       = Vec4::fma(mla, val2, area[2]);
                mla       = Vec4::fma(mla, val3, area[3]);
                res       += mla;
                preCalcIdx++;
            }
            res = res * invSamplingCnt;
            Vec4::save(dst + w * UNIT, res);
        }
    }
}

void MNNPackC4Uint8(uint8_t* dst, const uint8_t* src, size_t area,size_t depth, int* areaOffset) {
    MNNPackC4Common(dst, src, area, depth, areaOffset);
}

void MNNUnpackC4Uint8(uint8_t* dst, const uint8_t* src, size_t area,size_t depth, int* areaOffset) {
    MNNUnpackC4Common(dst, src, area, depth, areaOffset);
}

void MNNUnpackTransposeUint8(uint8_t* dst, const uint8_t* src, size_t area,size_t depth, int* areaOffset) {
    if (depth == 4) {
        ::memcpy(dst, src, area * depth * sizeof(uint8_t));
        return;
    }
#ifdef MNN_USE_NEON
    if (depth == 3) {
        uint8x16x4_t rgba;
        rgba.val[3] = vdupq_n_u8(0);
        int sta     = 0;
        int staC16  = (int)area / 16;
        for (int i = 0; i < staC16; sta += 16, ++i) {
            auto rgb    = vld3q_u8(src + sta * 3);
            rgba.val[0] = rgb.val[0];
            rgba.val[1] = rgb.val[1];
            rgba.val[2] = rgb.val[2];
            vst4q_u8(dst + 4 * sta, rgba);
        }
        sta = staC16 * 16;

        for (; sta < area; ++sta) {
            auto s = src + sta * 3;
            auto d = dst + sta * 4;
            d[0]   = s[0];
            d[1]   = s[1];
            d[2]   = s[2];
            d[3]   = 0;
        }

        return;
    }
    if (depth == 1) {
        uint8x16x4_t rgba;
        rgba.val[1] = vdupq_n_u8(0);
        rgba.val[2] = vdupq_n_u8(0);
        rgba.val[3] = vdupq_n_u8(0);
        int sta     = 0;
        for (; sta < area; sta += 16) {
            rgba.val[0] = vld1q_u8(src + sta);
            vst4q_u8(dst + 4 * sta, rgba);
        }

        for (; sta < area; ++sta) {
            auto s = src + sta;
            auto d = dst + sta * 4;
            d[0]   = s[0];
            d[1]   = 0;
            d[2]   = 0;
            d[3]   = 0;
        }

        return;
    }
#endif
    int c      = (int)depth;
    int cDiv4  = c / 4;
    int cAlign = cDiv4 * 4;

    if (cAlign == c) {
        for (int hi = 0; hi < area; ++hi) {
            auto srcHeight = reinterpret_cast<const int32_t*>(src + hi * c);
            auto dstHeight = reinterpret_cast<int32_t*>(dst + hi * 4);
            for (int ci = 0; ci < cDiv4; ++ci) {
                dstHeight[ci * areaOffset[1]] = srcHeight[ci];
            }
        }
        return;
    } else {
        for (int hi = 0; hi < area; ++hi) {
            auto srcHeight = src + hi * c;
            auto dstHeight = dst + hi * 4;
            for (int ci = 0; ci < cDiv4; ++ci) {
                dstHeight[ci * areaOffset[1] * 4 + 0] = srcHeight[ci * 4 + 0];
                dstHeight[ci * areaOffset[1] * 4 + 1] = srcHeight[ci * 4 + 1];
                dstHeight[ci * areaOffset[1] * 4 + 2] = srcHeight[ci * 4 + 2];
                dstHeight[ci * areaOffset[1] * 4 + 3] = srcHeight[ci * 4 + 3];
            }
        }
    }

    int cReamin   = c - cAlign;
    auto srcAlign = src + cAlign;
    auto dstAlign = dst + areaOffset[1] * cAlign;

    for (int hi = 0; hi < area; ++hi) {
        auto srcHeight = srcAlign + hi * c;
        auto dstHeight = dstAlign + hi * 4;
        for (int i = 0; i < 4; ++i) {
            dstHeight[i] = 0;
        }
        for (int ci = 0; ci < cReamin; ++ci) {
            dstHeight[ci] = srcHeight[ci];
        }
    }
}

void MNNUnpackTranspose(float* dst, const float* src, size_t area, size_t depth, int* areaOffset) {
    int srcAreaOffset = areaOffset[0];
    int dstAreaOffset = areaOffset[1];
#ifdef MNN_USE_NEON
    if (1 == depth) {
        auto zeroValue = vmovq_n_f32(0.0f);
        int areaC4     = (int)area / 4;
        int remain     = areaC4 * 4;
        for (int i = 0; i < areaC4; ++i) {
            auto srcCur   = src + 4 * i;
            auto dstCur   = dst + 16 * i;
            auto srcValue = vld1q_f32(srcCur);
            float32x4x4_t dstValue;
            dstValue.val[0] = srcValue;
            dstValue.val[1] = zeroValue;
            dstValue.val[2] = zeroValue;
            dstValue.val[3] = zeroValue;
            vst4q_f32(dstCur, dstValue);
        }
        for (int i = remain; i < area; ++i) {
            dst[4 * i + 0] = src[i];
            dst[4 * i + 1] = 0.0f;
            dst[4 * i + 2] = 0.0f;
            dst[4 * i + 3] = 0.0f;
        }
        return;
    }
    if (3 == depth) {
        auto zeroValue = vmovq_n_f32(0.0f);
        int areaC4     = (int)area / 4;
        int remain     = areaC4 * 4;
        for (int i = 0; i < areaC4; ++i) {
            auto srcCur   = src + 12 * i;
            auto dstCur   = dst + 16 * i;
            auto srcValue = vld3q_f32(srcCur);
            float32x4x4_t dstValue;
            dstValue.val[0] = srcValue.val[0];
            dstValue.val[1] = srcValue.val[1];
            dstValue.val[2] = srcValue.val[2];
            dstValue.val[3] = zeroValue;
            vst4q_f32(dstCur, dstValue);
        }
        for (int i = remain; i < area; ++i) {
            dst[4 * i + 0] = src[3 * i + 0];
            dst[4 * i + 1] = src[3 * i + 1];
            dst[4 * i + 2] = src[3 * i + 2];
            dst[4 * i + 3] = 0.0f;
        }
        return;
    }
#endif
    int c      = (int)depth;
    int cDiv4  = c / 4;
    int cAlign = cDiv4 * 4;
    for (int hi = 0; hi < area; ++hi) {
        const float* srcHeight = src + hi * c;
        float* dstHeight       = dst + hi * 4;
        for (int ci = 0; ci < cDiv4; ++ci) {
            Vec4::save(dstHeight + 4 * ci * dstAreaOffset, Vec4::load(srcHeight + 4 * ci));
        }
    }

    if (cAlign == c) {
        return;
    }

    int cReamin   = c - cAlign;
    auto srcAlign = src + cAlign;
    auto dstAlign = dst + dstAreaOffset * cAlign;

#ifdef MNN_USE_NEON
    auto zeroVector = vdupq_n_f32(0.0f);
#endif

    for (int hi = 0; hi < area; ++hi) {
        const float* srcHeight = srcAlign + hi * c;
        float* dstHeight       = dstAlign + hi * 4;
#ifdef MNN_USE_NEON
        vst1q_f32(dstHeight, zeroVector);
#else
        for (int i = 0; i < 4; ++i) {
            dstHeight[i] = 0;
        }
#endif
        for (int ci = 0; ci < cReamin; ++ci) {
            dstHeight[ci] = srcHeight[ci];
        }
    }
}

void MNNPackTransposeUint8(uint8_t* dst, const uint8_t* src, size_t area,size_t depth, int* areaOffset) {
    int c      = (int)depth;
    int cDiv4  = c / 4;
    int cAlign = cDiv4 * 4;
    if (cAlign == c) {
        int32_t* dst32       = (int32_t*)dst;
        const int32_t* src32 = (int32_t*)src;
        for (int hi = 0; hi < area; ++hi) {
            auto srcHeight = src32 + hi;
            auto dstHeight = dst32 + hi * cDiv4;
            for (int ci = 0; ci < cDiv4; ++ci) {
                dstHeight[ci] = srcHeight[ci * areaOffset[0]];
            }
        }
        return;
    }

    for (int hi = 0; hi < area; ++hi) {
        auto srcHeight = src + hi * 4;
        auto dstHeight = dst + hi * c;
        for (int ci = 0; ci < cDiv4; ++ci) {
            for (int i = 0; i < 4; ++i) {
                dstHeight[ci * 4 + i] = srcHeight[4 * ci * areaOffset[0] + i];
            }
        }
    }

    int cReamin   = c - cAlign;
    auto srcAlign = src + areaOffset[0] * cAlign;
    auto dstAlign = dst + cAlign;

    for (int hi = 0; hi < area; ++hi) {
        auto srcHeight = srcAlign + hi * 4;
        auto dstHeight = dstAlign + hi * c;

        for (int ci = 0; ci < cReamin; ++ci) {
            dstHeight[ci] = srcHeight[ci];
        }
    }
}

void MNNPackTranspose(float* dst, const float* src, size_t area, size_t depth, int* areaOffset) {
#if defined(MNN_USE_NEON)
    if (3 == depth) {
        int areaC4     = (int)area / 4;
        int remain     = areaC4 * 4;
        for (int i = 0; i < areaC4; ++i) {
            auto srcCur   = src + 16 * i;
            auto dstCur   = dst + 12 * i;
            auto srcValue = vld4q_f32(srcCur);
            float32x4x3_t dstValue;
            dstValue.val[0] = srcValue.val[0];
            dstValue.val[1] = srcValue.val[1];
            dstValue.val[2] = srcValue.val[2];
            vst3q_f32(dstCur, dstValue);
        }
        for (int i = remain; i < area; ++i) {
            dst[3 * i + 0] = src[4 * i + 0];
            dst[3 * i + 1] = src[4 * i + 1];
            dst[3 * i + 2] = src[4 * i + 2];
        }
        return;
    }
#elif defined(MNN_USE_SSE)
    if (3 == depth) {
        if (area < 1) return;
        for (int i = 0; i < area - 1; ++i) {
            auto srcValue = Vec4::load(src + 4 * i);
            Vec4::save(dst + 3 * i, srcValue);
        }
        for (int i = 0; i < 3; ++i) {
            dst[3 * (area - 1) + i] = src[4 * (area - 1) + i];
        }
        return;
    }
#endif
    int c      = (int)depth;
    int cDiv4  = c / 4;
    int cAlign = cDiv4 * 4;
    auto srcArea = areaOffset[0];
    auto dstDepthOffset = areaOffset[1];
    for (int hi = 0; hi < area; ++hi) {
        const float* srcHeight = src + hi * 4;
        float* dstHeight       = dst + hi * dstDepthOffset;
        for (int ci = 0; ci < cDiv4; ++ci) {
            Vec4::save(dstHeight + 4 * ci, Vec4::load(srcHeight + 4 * ci * srcArea));
        }
    }

    if (cAlign == c) {
        return;
    }

    int cReamin   = c - cAlign;
    auto srcAlign = src + srcArea * cAlign;
    auto dstAlign = dst + cAlign;

    for (int hi = 0; hi < area; ++hi) {
        const float* srcHeight = srcAlign + hi * 4;
        float* dstHeight       = dstAlign + hi * dstDepthOffset;

        for (int ci = 0; ci < cReamin; ++ci) {
            dstHeight[ci] = srcHeight[ci];
        }
    }
}

// Lambert's series with 7 divisions
// reference from
// https://varietyofsound.wordpress.com/2011/02/14/efficient-tanh-computation-using-lamberts-continued-fraction/
inline float tanhf_poly(float value) {
    if (value > 5.0) {
        return 1.0;
    } else if (value <= -5.0) {
        return -1.0;
    } else {
        float x2 = value * value;
        float a  = value * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)));
        float b  = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f));
        return a / b;
    }
}
void MNNTanh(float* dst, const float* src, size_t dataSize) {
    /* Origin Code
    for (int i = 0; i < dataSize; i++) {
        // outputData[i] = 1 - 2 / (expf(2 * inputData[i]) + 1);
        dst[i] = tanhf_poly(src[i]);
    }
     */
    float offset[4] = {
        -2.0f,
        0.0f,
        0.0f,
        0.0f
    };
    MNNExp(dst, src, offset, dataSize);
    for (int i = 0; i < dataSize; i++) {
        // outputData[i] = 1 - 2 / (expf(2 * inputData[i]) + 1);
        auto expX2 = dst[i];
        dst[i] = (1.0f - expX2) / (1.0f + expX2);
    }
}

void MNNReluWithSlope(float* dst, const float* src, size_t sizeQuad, float slope) {
    float slopeValue[4];
    for (int i=0; i<4; ++i) {
        slopeValue[i] = slope;
    }
    MNNReluWithSlopeChannel(dst, src, slopeValue, sizeQuad, 1);
}

void MNNReluWithSlopeCommon(float* dst, const float* src, size_t size, float slope) {
    int sizeQuad = static_cast<int32_t>(size) / 4;
    int remain = static_cast<int32_t>(size) % 4;
    if (sizeQuad > 0) {
        MNNReluWithSlope(dst, src, sizeQuad, slope);
    }
    if (remain > 0) {
        float intmp[4] = {0}, outmp[4] = {0};
        ::memcpy(intmp, src + sizeQuad * 4, remain * sizeof(float));
        MNNReluWithSlope(outmp, intmp, 1, slope);
        ::memcpy(dst + sizeQuad * 4, outmp, remain * sizeof(float));
    }
}

void MNNHardSwishCommon(float* dst, const float* src, size_t size) {
    int sizeQuad = static_cast<int32_t>(size / 4);
    int remain = static_cast<int32_t>(size) % 4;
#ifdef MNN_USE_SSE
    if (sizeQuad > 0) {
        MNNHardSwish(dst, src, sizeQuad);
    }
    if (remain > 0) {
        float intmp[4] = {0}, outmp[4] = {0};
        ::memcpy(intmp, src + sizeQuad * 4, remain * sizeof(float));
        MNNHardSwish(outmp, intmp, 1);
        ::memcpy(dst + sizeQuad * 4, outmp, remain * sizeof(float));
    }
#else
#ifdef MNN_USE_NEON
    float32x4_t zero = vdupq_n_f32(0.f);
    float32x4_t three = vdupq_n_f32(3.f);
    float32x4_t six = vdupq_n_f32(6.f);
    float32x4_t divsix = vdupq_n_f32(1.0f/6.f);
    for (int i = 0; i < sizeQuad; i++) {
        auto x = vld1q_f32(src + 4 * i);
        auto y = vmulq_f32(vmulq_f32(x, vminq_f32(vmaxq_f32(vaddq_f32(x, three), zero), six)), divsix);
        vst1q_f32(dst + 4 * i, y);
    }
    if (remain > 0) {
        float intmp[4] = {0}, outmp[4] = {0};
        ::memcpy(intmp, src + sizeQuad * 4, remain * sizeof(float));
        auto x = vld1q_f32(intmp);
        auto y = vmulq_f32(vmulq_f32(x, vminq_f32(vmaxq_f32(vaddq_f32(x, three), zero), six)), divsix);
        vst1q_f32(outmp, y);
        ::memcpy(dst + sizeQuad * 4, outmp, remain * sizeof(float));
    }
#else
    for (int j = 0; j < size; j++) {
        if (src[j] <= -3) {
            dst[j] = 0;
        } else if (src[j] >= 3){
            dst[j] = src[j];
        } else {
            dst[j] = src[j] * (src[j] + 3) / 6.f;
        }
    }
#endif
#endif
}

void MNNGeluStandardCommon(float* dst, const float* src, size_t size) {
    for (int i = 0; i < size; i++) {
        dst[i] = (erf(src[i] * 0.7071067932881648) + 1) * src[i] * 0.5;
    }
}

void MNNGeluCommon(float* dst, const float* src, size_t size) {
    int sizeQuad = static_cast<int32_t>(size / 8);
    int remain = static_cast<int32_t>(size) % 8;
#if defined(MNN_USE_SSE) || defined(MNN_USE_NEON)
    float parameters[8] = {0.044715f, 0.79788458f, 378.f, 17325.f, 135135.f, 28.f, 3150.f, 62370.f};
    if (sizeQuad > 0) {
        MNNGelu(dst, src, sizeQuad, parameters);
    }
    if (remain > 0) {
        float intmp[8] = {0};
        float outmp[8] = {0};
        ::memcpy(intmp, src + 8 * sizeQuad, remain * sizeof(float));
        MNNGelu(outmp, intmp, 1, parameters);
        ::memcpy(dst + 8 * sizeQuad, outmp, remain * sizeof(float));
    }
#else
    auto tanhf_poly = [](float value) -> float {
        if (value > 5.0f) {
            return 1.0f;
        } else if (value <= -5.0f) {
            return -1.0f;
        } else {
            float x2 = value * value;
            float a  = value * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)));
            float b  = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f));
            return a / b;
        }
    };
    for (int i = 0; i < size; i++) {
        float temp = 0.044715f * src[i] * src[i] * src[i];
        temp = 0.79788458f * (temp + src[i]);
        dst[i] = (1.0f + tanhf_poly(temp)) * src[i] * 0.5f;
    }
#endif
}

void MNNScaleAndAddBiasScalar(float* dst, const float* src, float bias, float alpha, size_t number) {
    int numberC4 = (int)number / 4;
    int start = 0;
    if (numberC4 > 0) {
        float biasC4[4] = {
            bias,
            bias,
            bias,
            bias
        };
        float alphaC4[4] = {
            alpha,
            alpha,
            alpha,
            alpha
        };
        MNNScaleAndAddBias(dst, src, biasC4, alphaC4, numberC4, 1);
        start = numberC4 * 4;
    }
    for (int i=start; i<number; ++i) {
        dst[i] = src[i] * alpha + bias;
    }
}
#ifndef MNN_USE_NEON
void MNNAxByClampBroadcastUnit(float* C, const float* A, const float* B, size_t width, size_t cStride, size_t aStride, size_t height, const float* parameters) {
    auto minF = Vec4(parameters[2]);
    auto maxF = Vec4(parameters[3]);
    auto beta = Vec4(parameters[1]);
    for (int y = 0; y < height; ++y) {
        auto a = A + aStride * y;
        auto b = B + 4 * y;
        auto bv = Vec4::load(b);
        auto c = C + cStride * y;
        for (int x = 0; x < width; ++x) {
            auto av = Vec4::load(a + 4 * x);
            auto cv = av + bv * beta;
            cv = Vec4::min(cv, maxF);
            cv = Vec4::max(cv, minF);
            Vec4::save(c + 4 * x, cv);
        }
    }
}
void MNNVectorTop1Float(float* input, float* maxValue, int32_t* maxIndex, size_t inputCountUnit) {
    float maxV = input[0];
    int maxIdx = 0;
    for (int i = 0; i < inputCountUnit; i++) {
        int offset = i * UNIT;
        for (int j = 0; j < UNIT; j++) {
            if (input[offset + j] > maxV) {
                maxV = input[offset + j];
                maxIdx = offset + j;
            }
        }
    }
    maxValue[0] = maxV;
    maxIndex[0] = maxIdx;
}

void MNNVectorTop1Int32(int32_t* input, int32_t* maxValue, int32_t* maxIndex, size_t inputCountUnit) {
    int32_t maxV = input[0];
    int maxIdx = 0;
    for (int i = 0; i < inputCountUnit; i++) {
        int offset = i * UNIT;
        for (int j = 0; j < UNIT; j++) {
            if (input[offset + j] > maxV) {
                maxV = input[offset + j];
                maxIdx = offset + j;
            }
        }
    }
    maxValue[0] = maxV;
    maxIndex[0] = maxIdx;
}

#endif

void MNNComputeMatMulForE_1(const float* A, const float* B, float* C, const float* biasPtr, const MatMulParam* param, size_t tIdL) {
    auto l = param->l;
    auto h = param->h;
    auto numberThread = param->numberThread;
    auto lC4 = l / 4;
    auto lR = lC4 * 4;
    auto tId = (int)tIdL;
    if (param->BTranspose) {
        for (int y=tId; y<h; y+=numberThread) {
            Vec4 sumValue = Vec4(0.0f);
            auto by = B + y * l;
            for (int x=0; x<lC4; ++x) {
                sumValue = Vec4::fma(sumValue, Vec4::load(A + x * 4), Vec4::load(by + x * 4));
            }
            float sumRemain = 0.0f;
            for (int x=lR; x<l; ++x) {
                sumRemain = sumRemain + A[x] * by[x];
            }
            if (nullptr != biasPtr) {
                sumRemain += biasPtr[y];
            }
            C[y] = sumRemain + sumValue[0] + sumValue[1] + sumValue[2] + sumValue[3];
        }
    } else {
        auto hC4 = h / 16;
        auto hR = hC4 * 16;
        for (int y=tId; y<hC4; y+=numberThread) {
            auto bs = B + 16 * y;
            Vec4 sumValue0;
            Vec4 sumValue1;
            Vec4 sumValue2;
            Vec4 sumValue3;
            if (biasPtr != nullptr) {
                sumValue0 = Vec4::load(biasPtr + 16 * y + 0);
                sumValue1 = Vec4::load(biasPtr + 16 * y + 4);
                sumValue2 = Vec4::load(biasPtr + 16 * y + 8);
                sumValue3 = Vec4::load(biasPtr + 16 * y + 12);
            } else {
                sumValue0 = Vec4(0.0f);
                sumValue1 = Vec4(0.0f);
                sumValue2 = Vec4(0.0f);
                sumValue3 = Vec4(0.0f);
            }
            auto srcY = A + y * l;
            for (int x=0; x<l; ++x) {
                auto a = Vec4(A[x]);
                sumValue0 = Vec4::fma(sumValue0, a, Vec4::load(bs + h * x));
                sumValue1 = Vec4::fma(sumValue1, a, Vec4::load(bs + h * x + 4));
                sumValue2 = Vec4::fma(sumValue2, a, Vec4::load(bs + h * x + 8));
                sumValue3 = Vec4::fma(sumValue3, a, Vec4::load(bs + h * x + 12));
            }
            Vec4::save(C + 16 * y, sumValue0);
            Vec4::save(C + 16 * y + 4, sumValue1);
            Vec4::save(C + 16 * y + 8, sumValue2);
            Vec4::save(C + 16 * y + 12, sumValue3);
        }
        int hEnd = hR;
        if ((h-hR) >= 8) {
            if (0 == tId) {
                auto bs = B + hEnd;
                Vec4 sumValue0;
                Vec4 sumValue1;
                if (biasPtr != nullptr) {
                    sumValue0 = Vec4::load(biasPtr + hEnd + 0);
                    sumValue1 = Vec4::load(biasPtr + hEnd + 4);
                } else {
                    sumValue0 = Vec4(0.0f);
                    sumValue1 = Vec4(0.0f);
                }
                auto srcY = A + hEnd * l;
                for (int x=0; x<l; ++x) {
                    auto a = Vec4(A[x]);
                    sumValue0 = Vec4::fma(sumValue0, a, Vec4::load(bs + h * x));
                    sumValue1 = Vec4::fma(sumValue1, a, Vec4::load(bs + h * x + 4));
                }
                Vec4::save(C + hEnd, sumValue0);
                Vec4::save(C + hEnd + 4, sumValue1);
            }
            hEnd = hEnd + 8;
        }
        if ((h-hEnd) >= 4) {
            if (0 == tId) {
                auto bs = B + hEnd;
                Vec4 sumValue0;
                if (biasPtr != nullptr) {
                    sumValue0 = Vec4::load(biasPtr + hEnd + 0);
                } else {
                    sumValue0 = Vec4(0.0f);
                }
                auto srcY = A + hEnd * l;
                for (int x=0; x<l; ++x) {
                    auto a = Vec4(A[x]);
                    sumValue0 = Vec4::fma(sumValue0, a, Vec4::load(bs + h * x));
                }
                Vec4::save(C + hEnd, sumValue0);
            }
            hEnd = hEnd + 4;
        }
        hEnd = hEnd + tId;
        for (int y=hEnd; y<h; y+=numberThread) {
            auto bs = B + y;
            float sumValue = 0.0f;
            if (biasPtr != nullptr) {
                sumValue = biasPtr[y];
            }
            auto srcY = A + y * l;
            for (int x=0; x<l; ++x) {
                sumValue = sumValue + A[x] * bs[h * x];
            }
            C[y] = sumValue;
        }
    }
}

void MNNComputeMatMulForH_1(const float* A, const float* B, float* C, const float* biasPtr, const MatMulParam* param, size_t tId) {
    int e = param->e;
    int l = param->l;
    int numberThread = param->numberThread;
    if (param->ATranspose) {
        float biasValue = 0.0f;
        if (nullptr != biasPtr) {
            biasValue = *biasPtr;
        }
        auto eC4 = e / 4;
        auto eR = eC4 * 4;
        for (int y=tId; y<eC4; y+=numberThread) {
            Vec4 sumValue = Vec4(biasValue);
            auto srcY = A + y * 4;
            for (int x=0; x<l; ++x) {
                sumValue = sumValue + Vec4::load(srcY + x * e) * Vec4(B[x]);
            }
            Vec4::save(C + 4 * y, sumValue);
        }
        if (0 == tId) {
            for (int y=eR; y<e; ++y) {
                float sumValue = biasValue;
                auto srcY = A + y;
                for (int x=0; x<l; ++x) {
                    sumValue = sumValue + srcY[x * e] * B[x];
                }
                C[y] = sumValue;
            }
        }
        return;
    }
    float biasValue = 0.0f;
    if (nullptr != biasPtr) {
        biasValue = *biasPtr;
    }
    auto lC4 = l / 16;
    auto lRO = lC4 * 16;
    for (int y=tId; y<e; y+=numberThread) {
        auto lR = lRO;
        Vec4 sumValue = Vec4(biasValue);
        Vec4 sum1(0.0f);
        Vec4 sum2(0.0f);
        Vec4 sum3(0.0f);
        auto srcY = A + y * l;
        for (int x=0; x<lC4; ++x) {
            sumValue = Vec::fma(sumValue, Vec4::load(srcY + 16 * x + 0), Vec4::load(B + 16 * x + 0));
            sum1 = Vec::fma(sum1, Vec4::load(srcY + 16 * x + 4), Vec4::load(B + 16 * x + 4));
            sum2 = Vec::fma(sum2, Vec4::load(srcY + 16 * x + 8), Vec4::load(B + 16 * x + 8));
            sum3 = Vec::fma(sum3, Vec4::load(srcY + 16 * x + 12), Vec4::load(B + 16 * x + 12));
        }
        if (l - lR >= 8) {
            sumValue = Vec::fma(sumValue, Vec4::load(srcY + lR), Vec4::load(B + lR));
            sum1 = Vec::fma(sum1, Vec4::load(srcY + lR + 4), Vec4::load(B + lR + 4));
            lR += 8;
        }
        if (l - lR >= 4) {
            sumValue = Vec::fma(sumValue, Vec4::load(srcY + lR), Vec4::load(B + lR));
            lR += 4;
        }
        sum2 = sum2 + sum3;
        sumValue = sumValue + sum1;
        sumValue = sumValue + sum2;
        float sumSingle = sumValue[0] + sumValue[1] + sumValue[2] + sumValue[3];
        for (int x=lR; x<l; ++x) {
            sumSingle += srcY[x] * B[x];
        }
        C[y] = sumSingle;
    }
}

void MNNPackC4Int16(int16_t* dst, const int16_t* src, size_t area,size_t depth, int* areaOffset) {
    MNNPackC4Common(dst, src, area, depth, areaOffset);
}

void MNNUnpackC4Int16(int16_t* dst, const int16_t* src, size_t area,size_t depth, int* areaOffset) {
    MNNUnpackC4Common(dst, src, area, depth, areaOffset);
}

void MNNUnpackTransposeInt16(int16_t* dst, const int16_t* src, size_t area,size_t depth, int* areaOffset) {
    if (depth == 4) {
        ::memcpy(dst, src, area * depth * sizeof(int16_t));
        return;
    }
    int c      = (int)depth;
    int cDiv4  = c / 4;
    int cAlign = cDiv4 * 4;
    for (int hi = 0; hi < area; ++hi) {
        auto srcHeight = (src + hi * c);
        auto dstHeight = (dst + hi * 4);
        for (int ci = 0; ci < cDiv4; ++ci) {
            for (int i = 0; i < 4; ++i) {
                dstHeight[ci * areaOffset[1] * 4 + i] = srcHeight[4 * ci + i];
            }
        }
    }

    if (cAlign == c) {
        return;
    }

    int cReamin   = c - cAlign;
    auto srcAlign = src + cAlign;
    auto dstAlign = dst + areaOffset[1] * cAlign;

    for (int hi = 0; hi < area; ++hi) {
        auto srcHeight = srcAlign + hi * c;
        auto dstHeight = dstAlign + hi * 4;
        for (int i = 0; i < 4; ++i) {
            dstHeight[i] = 0;
        }
        for (int ci = 0; ci < cReamin; ++ci) {
            dstHeight[ci] = srcHeight[ci];
        }
    }
}
void MNNPackTransposeInt16(int16_t* dst, const int16_t* src, size_t area,size_t depth, int* offset) {
    int c      = (int)depth;
    int cDiv4  = c / 4;
    int cAlign = cDiv4 * 4;
    int srcAreaOffset = offset[0];
    int dstDepthOffset = offset[1];
    if (cAlign == c) {
        for (int hi = 0; hi < area; ++hi) {
            auto srcHeight = (int64_t*)src + hi;
            auto dstHeight = (int64_t*)(dst + hi * dstDepthOffset);
            for (int ci = 0; ci < cDiv4; ++ci) {
                dstHeight[ci] = srcHeight[ci * srcAreaOffset];
            }
        }
        return;
    }

    for (int hi = 0; hi < area; ++hi) {
        auto srcHeight = src + hi * 4;
        auto dstHeight = dst + hi * dstDepthOffset;
        for (int ci = 0; ci < cDiv4; ++ci) {
            for (int i = 0; i < 4; ++i) {
                dstHeight[ci * 4 + i] = srcHeight[4 * ci * srcAreaOffset + i];
            }
        }
    }

    int cReamin   = c - cAlign;
    auto srcAlign = src + srcAreaOffset * cAlign;
    auto dstAlign = dst + cAlign;

    for (int hi = 0; hi < area; ++hi) {
        auto srcHeight = srcAlign + hi * 4;
        auto dstHeight = dstAlign + hi * dstDepthOffset;

        for (int ci = 0; ci < cReamin; ++ci) {
            dstHeight[ci] = srcHeight[ci];
        }
    }
}

void MNNCopyC4Int16WithStride(const float* sourceF, float* destF, size_t srcStride, size_t dstStride, size_t count) {
    auto source = (int16_t*)sourceF;
    auto dest = (int16_t*)destF;
    for (int i = 0; i < count; ++i) {
        auto s = source + i * srcStride;
        auto d = dest + i * dstStride;
        *(int64_t*)(d) = *((int64_t*)s);
    }
}


void MNNSin(float* dst, const float* src, size_t dataSize) {
    for (int i = 0; i < dataSize; i++) {
        dst[i] = sinf(src[i]);
    }
}

void MNNSigmoid(float* dst, const float* src, size_t dataSize) {
    float offset[4] = {
       -1.0f,
        0.0f,
        0.0f,
        0.0f
    };
    MNNExp(dst, src, offset, dataSize);
    for (int i = 0; i < dataSize; ++i) {
        dst[i] = 1.0f / (1.0f + dst[i]);
    }
}

void MNNSiLu(float* dst, const float* src, size_t dataSize) {
    float offset[4] = {
       -1.0f,
        0.0f,
        0.0f,
        0.0f
    };
    MNNExp(dst, src, offset, dataSize);
    for (int i = 0; i < dataSize; ++i) {
        dst[i] = src[i] / (1.0f + dst[i]);
    }
}

/**
 Modified from https://github.com/alibaba/MNN/pull/1359
 Thanks for https://github.com/hroken
 */
void MNNSigmoidLowp(float* dst, const float* src, size_t dataSize) {
    float offset[4] = {
       -1.0f,
        0.0f,
        0.0f,
        0.0f
    };
    MNNExp(dst, src, offset, dataSize);
#ifdef MNN_USE_NEON
    int dataC4 = static_cast<int32_t>(dataSize) / 4;
    int remain = static_cast<int32_t>(dataSize) % 4;
    float32x4_t value = vdupq_n_f32(1.0f);

    if(dataC4 > 0) {
        float32x4_t out = vld1q_f32(dst);
        // neon optimization for sigmid cpu
        for (int i = 1; i < dataC4; ++i) {
            out = vrecpeq_f32(vaddq_f32(value,out));
            vst1q_f32(dst ,out);
            dst += 4;
            out = vld1q_f32(dst);
        }
        out = vrecpeq_f32(vaddq_f32(value,out));
        vst1q_f32(dst, out);
        dst += 4;
    }
    if (remain > 0) {
        float intmp[4] = {0};
        ::memcpy(intmp, dst, remain * sizeof(float));
        float32x4_t out = vld1q_f32(intmp);
        out = vrecpeq_f32(vaddq_f32(value,out));
        vst1q_f32(intmp, out);
        ::memcpy(dst, intmp, remain * sizeof(float));
    }
#else
    for (int i = 0; i < dataSize; ++i) {
        dst[i] = 1.0f / (1.0f + dst[i]);
    }
#endif
}

void MNNSiLuLowp(float* dst, const float* src, size_t dataSize) {
    float offset[4] = {
       -1.0f,
        0.0f,
        0.0f,
        0.0f
    };
    MNNExp(dst, src, offset, dataSize);
#ifdef __aarch64__
    int dataC4 = static_cast<int32_t>(dataSize) / 4;
    int remain = static_cast<int32_t>(dataSize) % 4;
    float32x4_t one = vdupq_n_f32(1.0f);

    if(dataC4 > 0) {
        float32x4_t out = vld1q_f32(dst);
        float32x4_t in = vld1q_f32(src);
        // neon optimization for sigmid cpu
        for (int i = 1; i < dataC4; ++i) {
            out = vdivq_f32(in, vaddq_f32(one,out));
            vst1q_f32(dst ,out);
            dst += 4;
            src += 4;
            out = vld1q_f32(dst);
            in = vld1q_f32(src);
        }
        out = vdivq_f32(in, vaddq_f32(one,out));
        vst1q_f32(dst, out);
        dst += 4;
        src += 4;
    }
    if (remain > 0) {
        float intmp[4] = {0};
        float atmp[4] = {0};
        ::memcpy(intmp, dst, remain * sizeof(float));
        ::memcpy(atmp, src, remain * sizeof(float));
        float32x4_t out = vld1q_f32(intmp);
        float32x4_t in = vld1q_f32(atmp);
        out = vdivq_f32(in, vaddq_f32(one, out));
        vst1q_f32(intmp, out);
        ::memcpy(dst, intmp, remain * sizeof(float));
    }
#else
    for (int i = 0; i < dataSize; ++i) {
        dst[i] = src[i] / (1.0f + dst[i]);
    }
#endif
}

static void _MNNAdjustOptimalSparseKernel(int& sparseBlockOC, MNN::CoreFunctions::MNNPackedSparseMatMul& packedSparseMatMul) {
    if(sparseBlockOC == 4) {
        packedSparseMatMul = MNNPackedSparseMatMulEpx4;
        return;
    } else if(sparseBlockOC % 4 == 0) {
        sparseBlockOC = 4;
        packedSparseMatMul = MNNPackedSparseMatMulEpx4;
        // MNN_PRINT("common downgrade sparse to:%d\n",sparseBlockOC);
        return;
    } else {
        sparseBlockOC = 1;
        packedSparseMatMul = MNNPackedSparseMatMulEpx1;
        return;
    }
}

// fp32 <--> fp8
static const int FP32_EXP_BIAS = 127;
static const int FP8_EXP_BIAS = 24;   // [0, 31] --> [-24, 7] --> [1 / 2^24, 2^7]
void MNNFp32ToFp8(uint8_t* dst, const float* src, size_t size) {
    for (int i = 0; i < size; i++) {
        uint32_t rawData = *((uint32_t *)(&src[i]));
        uint32_t sign = (rawData >> 31) & 1U;
        uint32_t exp = (int)((rawData >> 23) & 0x0ffU);
        uint32_t mant = (rawData >> 21) & 3U;
        int realExp = (int)exp - FP32_EXP_BIAS;
        realExp = ALIMAX(realExp,  0 - FP8_EXP_BIAS);
        realExp = ALIMIN(realExp, 31 - FP8_EXP_BIAS);
        exp = (uint32_t)(realExp + FP8_EXP_BIAS);
        dst[i] = (int8_t)((sign << 7) | (exp << 2) | mant);
    }
}
void MNNFp8ToFp32(float* dst, const uint8_t* src, size_t size) {
    for (int i = 0; i < size; i++) {
        uint32_t sign = (src[i] >> 7) & 1U;
        uint32_t exp = (int)((src[i] >> 2) & 0x1fU);
        uint32_t mant = (src[i] & 3U) << 21;
        int realExp = (int)exp - FP8_EXP_BIAS;
        exp = (uint32_t)(realExp + FP32_EXP_BIAS);
        uint32_t rawData = (sign << 31) | (exp << 23) | mant;
        dst[i] = *((float *)(&rawData));
    }
}
// fp16 <--> fp8
void MNNFp16ToFp8(uint8_t* dst, const uint16_t* src, size_t size) {
#ifdef MNN_USE_NEON
#ifdef __aarch64__
    int loopN = size / 16;
    for (int i = 0; i < loopN; i++) {
        uint8x16_t v1 = vld1q_u8((uint8_t*)(src + i * 16));
        uint8x16_t v2 = vld1q_u8((uint8_t*)(src + i * 16 + 8));
        uint8x16_t res = vuzp2q_u8(v1, v2);
        vst1q_u8(dst + i * 16, res);
    }
    for (int i = loopN * 16; i < size; i++) {
        dst[i] = static_cast<int8_t>(src[i] >> 8);
    }
#else
    int loopN = size / 8;
    for (int i = 0; i < loopN; i++) {
        uint16x8_t vec = vld1q_u16(src + i * 8);
        uint8x8_t  res = vshrn_n_u16(vec, 8);
        vst1_u8(dst + i * 8, res);
    }
    for (int i = loopN * 8; i < size; i++) {
        dst[i] = static_cast<int8_t>(src[i] >> 8);
    }
#endif // ARM64
#else
    for (int i = 0; i < size; i++) {
        dst[i] = static_cast<int8_t>(src[i] >> 8);
    }
#endif // USE_NEON
}
void MNNFp8ToFp16(uint16_t* dst, const uint8_t* src, size_t size) {
#ifdef MNN_USE_NEON
    int loopN = size / 8;
    for (int i = 0; i < loopN; i++) {
        uint8x8_t vec8x8 = vld1_u8(src + i * 8);
        uint16x8_t vec16x8 = vshll_n_u8(vec8x8, 8);
        vst1q_u16(dst + i * 8, vec16x8);
    }
    for (int i = loopN * 8; i < size; i++) {
        dst[i] = static_cast<int16_t>(src[i]) << 8;
    }
#else
    for (int i = 0; i < size; i++) {
        dst[i] = static_cast<int16_t>(src[i]) << 8;
    }
#endif // USE_NEON
}

#ifdef MNN_LOW_MEMORY
static void generalIm2col(float* destOrigin, float const** sourceGroup, const int32_t* info, const int32_t* el, int LP, int pack) {
    // LP >= pack
    int number = info[0];
    int eReal = info[1];
    int eDest = info[2];
    int offset = info[3];
    for (int n=0; n<number; ++n) {
        int e = el[4 * n + 0];
        int l = el[4 * n + 1];
        int eOffset = el[4 * n + 2];
        int lOffset = el[4 * n + 3];
        int lC = lOffset / LP;
        int lR = lOffset % LP;
        auto dest = destOrigin + eOffset * LP + lC * eDest * LP + lR;
        auto source = sourceGroup[n];

        for (int y=0; y<e; ++y) {
            auto yR = y % eDest;
            for (int x=0; x<l; ++x) {
                auto xR = x % pack;
                auto xC = x / pack;
                auto xOut = x / LP;
                auto xIn = x % LP;
                dest[xOut * eDest * LP + yR * LP + xIn] = source[xC * eReal * pack + y * pack * offset + xR];
            }
        }
    }
}
#endif // MNN_LOW_MEMORY

#ifdef MNN_SME2
#define SME2_MATMUL_EP 16
#define SME2_MATMUL_LP 1
#define SME2_MATMUL_HP 64

static void SME2MNNGetMatMulPackMode(int* eP, int *lP, int* hP) {
    *eP = SME2_MATMUL_EP;
    *lP = SME2_MATMUL_LP;
    *hP = SME2_MATMUL_HP;
}
static void MNNPackedMatMulFP32_SME2(float* C, const float* A, const float* B, const size_t* parameter, const float* postParameters, const float* bias, const float* k, const float* b) {
    MNNPackedMatMulRemainFP32_SME2(C, A, B, 16, parameter, postParameters, bias, k, b);
    return;
}
static void Sme2MNNPackForMatMul_B(float* destC, const float* sourceC, size_t h, size_t kernelsize, size_t ic, bool transpose) {
    // src: [h, kernelsize, ic]
    // dst: [h/hp, kernelsize, ic/lp, hp, lp]
    auto dest = (int32_t*)destC;
    auto source = (int32_t*)sourceC;
    int LP = SME2_MATMUL_LP;
    int HP = SME2_MATMUL_HP;
    auto l = kernelsize * ic;
    memset(dest, 0, ROUND_UP(h, HP) * ROUND_UP(ic, LP) * kernelsize * 4);
    auto stride0 = kernelsize * ROUND_UP(ic, LP) * HP;
    auto stride1 = ROUND_UP(ic, LP) * HP;
    auto stride2 = HP * LP;

    auto srcStride0 = l; // [h,l]->[hu,lu,hp,lp]
    auto srcStride1 = 1;
    if (!transpose) { // [l,h]->[hu,lu,hp,lp]
        srcStride0 = 1;
        srcStride1 = h;
    }
    for (int y = 0; y < h; ++y) {
        auto yHu = y / HP;
        auto yHp = y % HP;
        for (int k = 0; k < kernelsize; ++k) {
            for (int x = 0; x < ic; ++x) {
                auto xLu = x / LP;
                auto xLp = x % LP;
                dest[yHu * stride0 + k * stride1 + xLu * stride2 + yHp * LP + xLp] = source[y * srcStride0 + (x + k * ic) * srcStride1];
            }
        }
    }
}
static void Sme2MNNPackC4ForMatMul_A(float* destOrigin, float const** sourceGroup, const int32_t* info, const int32_t* el) {
    MNNPackC4ForMatMul_A(destOrigin, sourceGroup, info, el);
    return;
}
#endif

namespace MNN {

static CoreFunctions* gCoreFunction = nullptr;

void MNNCoreFunctionInit() {
    gCoreFunction = new CoreFunctions;
    // fp8
    gCoreFunction->MNNFp32ToFp8 = MNNFp32ToFp8;
    gCoreFunction->MNNFp16ToFp8 = MNNFp16ToFp8;
    gCoreFunction->MNNFp8ToFp32 = MNNFp8ToFp32;
    gCoreFunction->MNNFp8ToFp16 = MNNFp8ToFp16;

    // MatMul
    gCoreFunction->MNNGetMatMulPackMode = MNNGetMatMulPackMode;
    gCoreFunction->MNNPackC4ForMatMul_A = MNNPackC4ForMatMul_A;
    gCoreFunction->MNNPackForMatMul_B = MNNPackForMatMul_B;
    gCoreFunction->MNNPackedMatMul = MNNPackedMatMul;
    gCoreFunction->MNNPackedMatMulRemain = MNNPackedMatMulRemain;
    gCoreFunction->MNNCountMaxMinValue = MNNCountMaxMinValue;
    gCoreFunction->MNNGetSparseMatMulPackMode = MNNGetSparseMatMulPackMode;
    gCoreFunction->MNNAdjustOptimalSparseKernel = _MNNAdjustOptimalSparseKernel;

    gCoreFunction->MNNComputeMatMulForE_1 = MNNComputeMatMulForE_1;
    gCoreFunction->MNNComputeMatMulForH_1 = MNNComputeMatMulForH_1;

    // Lowp
    gCoreFunction->MNNFp32ToLowp = nullptr;
    gCoreFunction->MNNLowpToFp32 = nullptr;
    gCoreFunction->bytes = 4;// sizeof(float)

    // Packed Function
    gCoreFunction->pack = 4;
    // FIXME: MNNPackTranspose and MNNUnpackTranspose is reverted
    gCoreFunction->MNNPackCUnit = MNNPackC4;
    gCoreFunction->MNNUnpackCUnit = MNNUnpackC4;
    gCoreFunction->MNNUnpackCUnitTranspose = MNNPackTranspose;
    gCoreFunction->MNNPackCUnitTranspose = MNNUnpackTranspose;
    gCoreFunction->MNNPackCUnitInt8 = decltype(gCoreFunction->MNNPackCUnitInt8)(MNNPackC4Uint8);
    gCoreFunction->MNNUnpackCUnitInt8 = decltype(gCoreFunction->MNNUnpackCUnitInt8)(MNNUnpackC4Uint8);
    gCoreFunction->MNNPackCUnitTransposeInt8 = decltype(gCoreFunction->MNNPackCUnitTransposeInt8)(MNNUnpackTransposeUint8);
    gCoreFunction->MNNUnpackCUnitTransposeInt8 = decltype(gCoreFunction->MNNUnpackCUnitTransposeInt8)(MNNPackTransposeUint8);
    gCoreFunction->MNNPackCUnitInt16 = MNNPackC4Int16;
    gCoreFunction->MNNUnpackCUnitInt16 = MNNUnpackC4Int16;
    gCoreFunction->MNNPackCUnitTransposeInt16 = MNNUnpackTransposeInt16;
    gCoreFunction->MNNUnpackCUnitTransposeInt16 = MNNPackTransposeInt16;

    gCoreFunction->MNNAxByClampBroadcastUnit = MNNAxByClampBroadcastUnit;
    gCoreFunction->MNNConvRunForLineDepthwise = MNNConvRunForLineDepthwise;
    gCoreFunction->MNNMatrixAdd = MNNMatrixAdd;
    gCoreFunction->MNNMatrixSub = MNNMatrixSub;
    gCoreFunction->MNNStrassenMergeCFunction = MNNStrassenMergeCFunction;
    gCoreFunction->penalty = 1.5f;
    gCoreFunction->MNNScaleAndAddBias = MNNScaleAndAddBias;
    gCoreFunction->MNNGridSampleComputeCord = MNNGridSampleComputeCord;
    gCoreFunction->MNNGridSampleInterp = MNNGridSampleInterp;
#ifndef MNN_REDUCE_SIZE
    gCoreFunction->MNNGridSampleInterpGrad = MNNGridSampleInterpGrad;
#endif
    gCoreFunction->MNNGridSampleComputeCord3D = MNNGridSampleComputeCord3D;
    gCoreFunction->MNNGridSampleInterp3D = MNNGridSampleInterp3D;
    gCoreFunction->MNNRoiPoolingMax = MNNRoiPoolingMax;
    gCoreFunction->MNNRoiAlignMax = MNNRoiAlignMax;
    gCoreFunction->MNNRoiAlignAvg = MNNRoiAlignAvg;
    gCoreFunction->MNNAddC4WithStride = MNNAddC4WithStride;
    gCoreFunction->MNNCopyC4WithStride = MNNCopyC4WithStride;

    gCoreFunction->chooseWinoSourceTransformPack = WinogradFunction::chooseWinoSourceTransformPack;
    gCoreFunction->chooseWinoSourceUnrollTransform = WinogradFunction::chooseSourceUnrollTransform;
    gCoreFunction->chooseWinoDestUnrollTransform = WinogradFunction::chooseWinoDestUnrollTransform;
    gCoreFunction->MNNDeconvRunForLineDepthwise = MNNDeconvRunForLineDepthwise;
    gCoreFunction->MNNDeconvRunForUnitDepthWise = MNNDeconvRunForUnitDepthWise;
    gCoreFunction->MNNSoftmax = MNNSoftmax;
#ifdef MNN_USE_NEON
    gCoreFunction->MNNDepthwiseConvFastKernel = MNNDepthwiseConvFastKernel;
#endif
    gCoreFunction->MNNSelectBinaryFunctionForFloat = CPUBinary::selectForFloat;
    gCoreFunction->MNNSelectUnaryFunctionForFloat = CPUUnary::selectForFloat;
#ifdef MNN_SUPPORT_QUANT_EXTEND
    gCoreFunction->MNNSelectUnaryFunctionForInt8 = CPUUnary::selectForInt8;
#endif

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
    gCoreFunction->MNNAttenPackAndScaleSingleHead = MNNAttenPackAndScaleSingleHead;
    gCoreFunction->MNNFlashAttentionUpdateBlockOutput = MNNFlashAttentionUpdateBlockOutput;
    gCoreFunction->MNNQuantAttentionKey = MNNQuantAttentionKey;
    gCoreFunction->MNNQuantAttentionValue = MNNQuantAttentionValue;
#endif // MNN_SUPPORT_TRANSFORMER_FUSE

    gCoreFunction->MNNReluWithSlopeChannel = MNNReluWithSlopeChannel;
    gCoreFunction->MNNPoolingAvg = (decltype(gCoreFunction->MNNPoolingAvg))(poolingAvg<float, Vec4, 4>);
    // Set min value as 1 << 24
    gCoreFunction->MNNPoolingMax = (decltype(gCoreFunction->MNNPoolingMax))(poolingMax<float, Vec4, 4, -16777216>);

    gCoreFunction->MNNPoolingMaxWithRedice = (decltype(gCoreFunction->MNNPoolingMaxWithRedice))(poolingMaxWithRedice<float, -16777216>);
    // ImageProcess Functions
    gCoreFunction->MNNRGBAToBGRA = MNNRGBAToBGRA;
    gCoreFunction->MNNNV21ToRGBA = MNNNV21ToRGBA;
    gCoreFunction->MNNNV21ToRGB = MNNNV21ToRGB;
    gCoreFunction->MNNNV21ToBGRA = MNNNV21ToBGRA;
    gCoreFunction->MNNNV21ToBGR = MNNNV21ToBGR;
    gCoreFunction->MNNC1ToFloatC1 = MNNC1ToFloatC1;
    gCoreFunction->MNNC3ToFloatC3 = MNNC3ToFloatC3;
    gCoreFunction->MNNC3ToFloatRGBA = MNNC3ToFloatRGBA;
    gCoreFunction->MNNSamplerC4Nearest = MNNSamplerC4Nearest;
    gCoreFunction->MNNSamplerC4Bilinear = MNNSamplerC4Bilinear;

    gCoreFunction->MNN4BitcopyWithStride = MNN4BitcopyWithStride;
    gCoreFunction->MNN1BitcopyWithStride = MNN1BitcopyWithStride;
    gCoreFunction->MNN2BitcopyWithStride = MNN2BitcopyWithStride;
    gCoreFunction->MNN4BitcopyFast = MNN4BitcopyFast;
    gCoreFunction->MNN2BitcopyFast = MNN2BitcopyFast;
    gCoreFunction->MNN1BitcopyFast = MNN1BitCopyFast;

    gCoreFunction->MNNAccumulateSequenceNumber = MNNAccumulateSequenceNumber;

    const MNNCPUInfo& gCPUInfo = *MNNGetCPUInfo();
    gCoreFunction->supportFp16arith = gCPUInfo.fp16arith;
    gCoreFunction->supportSDot = gCPUInfo.dot;
    gCoreFunction->supportI8mm = gCPUInfo.i8mm;
    gCoreFunction->supportSME2 = gCPUInfo.sme2;
    gCoreFunction->smeCoreNumber = gCPUInfo.smeCoreNumber;
    gCoreFunction->MNNSumByAxisLForMatmul_A = MNNSumByAxisLForMatmul_A;
    gCoreFunction->MNNReorderWeightInt4 = MNNReorderWeightInt4;
    gCoreFunction->MNNSumWeightInt8  = MNNSumWeightInt8;
#ifdef __aarch64__
    if (gCoreFunction->supportSDot) {
        gCoreFunction->MNNReorderWeightInt4 = MNNReorderWeightInt4Arm82;
        gCoreFunction->MNNSumWeightInt8 = MNNSumWeightInt8Arm82;
        gCoreFunction->arm82MatmulRelatedFunctions.MNNReorderWeightInt4 = MNNReorderWeightInt4Arm82;
        gCoreFunction->arm82MatmulRelatedFunctions.MNNSumWeightInt8 = MNNSumWeightInt8Arm82;
    }
    if (gCoreFunction->supportI8mm) {
        gCoreFunction->MNNReorderWeightInt4 = MNNReorderWeightInt4Arm86;
        gCoreFunction->MNNSumWeightInt8 = MNNSumWeightInt8Arm86;
    }
#endif
#ifdef MNN_CPU_WEIGHT_DEQUANT_GEMM
    // Weight Dequant Gemm Kernels
    gCoreFunction->MNNPackedMatMul_int8 = MNNPackedMatMul_int8;
    gCoreFunction->MNNPackedMatMulRemain_int8 = MNNPackedMatMulRemain_int8;
#endif
#ifdef MNN_LOW_MEMORY
    gCoreFunction->MNNAbsMax = MNNAbsMaxFP32;                      // abs max value for [icDiv4,plane,4] -> abs max:[plane]
    gCoreFunction->MNNDynamicQuant = MNNDynamicQuantFP32;          // symmetric 'batch' quant for [icDiv4,plane,4]
    gCoreFunction->MNNAsyQuantFunc = MNNAsyQuantFunc;              // asymmetric 'batch' quant for [icDiv4,plane,4]
    gCoreFunction->MNNAsyQuantInfo = MNNAsyQuantInfo_FP32;              // asymmetric quant/dequant scale&bias for [icDiv4,plane,4] -> scale&bias:[blockNum,plane]
    gCoreFunction->MNNQuantScale = MNNQuantScaleFP32;              // symmetric quant/dequant scale&bias for [icDiv4,plane,4] -> scale&bias:[plane]
    gCoreFunction->MNNGeneralIm2Col = generalIm2col;               // Im2Col based on float data -> output:[eU,kernelsize,lU,ep,lp]
    gCoreFunction->MNNDynamicUpdateConvBiasScale = MNNDynamicUpdateConvBiasScale;
#ifdef __aarch64__
    if (gCoreFunction->supportSDot) {
        gCoreFunction->MNNGeneralIm2Col = MNNGeneralIm2col_Fp32Arm82;
        gCoreFunction->arm82MatmulRelatedFunctions.MNNGeneralIm2Col = MNNGeneralIm2col_Fp32Arm82;
    }
    if (gCoreFunction->supportI8mm) {
        gCoreFunction->MNNGeneralIm2Col = MNNGeneralIm2col_Fp32Arm86;
    }
#endif
#endif


#ifdef __aarch64__
#ifdef MNN_SME2
    if (gCoreFunction->supportSME2) {
        // Int8 Gemm related
        gCoreFunction->MNNSumWeightInt8 = MNNSumWeightInt8Sme2_Hp32;
        gCoreFunction->MNNSumWeightInt8SmeHp128 = MNNSumWeightInt8Sme2_Hp128;
        gCoreFunction->MNNReorderWeightInt4 = MNNReorderWeightInt4Sme2;

#ifdef MNN_LOW_MEMORY
        gCoreFunction->MNNGeneralIm2Col = MNNGeneralIm2col_Fp32Sme2;
#endif

        gCoreFunction->int8MatmulRelatedFunctions.MNNSumWeightInt8SmeHp128 = MNNSumWeightInt8Sme2_Hp128;

        // Float Gemm related
        gCoreFunction->MNNPackedMatMul = MNNPackedMatMulFP32_SME2;
        gCoreFunction->MNNPackedMatMulRemain = MNNPackedMatMulRemainFP32_SME2;
        gCoreFunction->MNNGetMatMulPackMode = SME2MNNGetMatMulPackMode;
        gCoreFunction->MNNPackC4ForMatMul_A = Sme2MNNPackC4ForMatMul_A;
        gCoreFunction->MNNPackForMatMul_B = Sme2MNNPackForMatMul_B;
    }
#endif // MNN_SME2
#endif // __aarch64__


    {   // Update the function pointers in the int8MatmulRelatedFunctions struct.
        gCoreFunction->int8MatmulRelatedFunctions.MNNReorderWeightInt4 = gCoreFunction->MNNReorderWeightInt4;
        gCoreFunction->int8MatmulRelatedFunctions.MNNSumWeightInt8 = gCoreFunction->MNNSumWeightInt8;
        gCoreFunction->int8MatmulRelatedFunctions.MNNGeneralIm2Col = gCoreFunction->MNNGeneralIm2Col;
    }
    MNNCoreInt8FunctionInit();
    MNNFunctionInit();
}
CoreFunctions* MNNGetCoreFunctions() {
    return gCoreFunction;
}
};

void MNNUnpackC4Origin(float* dst, const float* src, size_t area, size_t depth, int areaOffset) {
    int offset[] = {
        areaOffset,
        areaOffset,
    };
    MNNUnpackC4(dst, src, area, depth, offset);
}
void MNNPackC4Origin(float* dst, const float* src, size_t area, size_t depth, int areaOffset) {
    int offset[] = {
        areaOffset,
        areaOffset,
    };
    MNNPackC4(dst, src, area, depth, offset);
}

void MNNPackC2(double* dst, const double* src, size_t area, size_t depth, int* areaOffset) {
    MNNPackC2Common<double>(dst, src, area, depth, areaOffset);
}

void MNNUnpackC2(double* dst, const double* src, size_t area, size_t depth, int* areaOffset) {
    MNNUnpackC2Common<double>(dst, src, area, depth, areaOffset);
}

void MNNUnpackC2Float(float* dst, const float* src, size_t area, size_t depth, int* areaOffset, int pack) {
    MNNUnpackC2Common<float>(dst, src, area, depth, areaOffset, pack);
}
#ifndef __aarch64__
void MNNPackInt8C2(float* dst, const float* src, size_t area, size_t depth, int* areaOffset) {
    MNNPackC2Common<float>(dst, src, area, depth, areaOffset);
}
#endif

void MNNUnpackInt8C2(float* dst, const float* src, size_t area, size_t depth, int* areaOffset) {
    MNNUnpackC2Common<float>(dst, src, area, depth, areaOffset);
}


void MNNUnpackC2Origin(double* dst, const double* src, size_t area, size_t depth, int areaOffset) {
    int offset[] = {
        areaOffset,
        areaOffset,
    };
    MNNUnpackC2(dst, src, area, depth, offset);
}
void MNNPackC2Origin(double* dst, const double* src, size_t area, size_t depth, int areaOffset) {
    int offset[] = {
        areaOffset,
        areaOffset,
    };
    MNNPackC2(dst, src, area, depth, offset);
}

void MNNUnpackInt8C2Origin(float* dst, const float* src, size_t area, size_t depth, int areaOffset) {
    int offset[] = {
        areaOffset,
        areaOffset,
    };
    MNNUnpackInt8C2(dst, src, area, depth, offset);
}
void MNNPackInt8C2Origin(float* dst, const float* src, size_t area, size_t depth, int areaOffset) {
    int offset[] = {
        areaOffset,
        areaOffset,
    };
    MNNPackInt8C2(dst, src, area, depth, offset);
}
