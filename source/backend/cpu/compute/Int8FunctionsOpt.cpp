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
#if defined(__aarch64__) && defined(MNN_USE_ARMV82) // aarch32 sdot workaround
void MNNGemmInt8AddBiasScale_ARMV82_Unit(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad,
                                         const QuanPostTreatParameters* post, size_t realDstCount);
#endif // __aarch64__ && MNN_USE_ARMV82
}
#endif // MNN_USE_NEON

#ifndef MNN_USE_NEON
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

#ifndef MNN_USE_SSE
void MNNInt8FunctionInit() {
    // do nothing
}
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
#endif // #ifndef MNN_USE_SSE

/* CPU without sdot */
// Assume GEMM_INT8_UNIT == 4 && GEMM_INT8_SRC_UNIT == 16
static void _fastIm2Col(int8_t* colAddr, const int8_t* inputOrigin, int8_t inputZeroPoint,
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

static void _im2colCommonZ1(int8_t* colAddr, const int8_t* inputOrigin, int8_t inputZeroPoint,
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

static void _im2colCommon(int8_t* colAddr, const int8_t* inputOrigin, int8_t inputZeroPoint,
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

static void _im2colCommonSdot(int8_t* colAddr, const int8_t* src, int8_t inputZeroPoint,
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

static void _fastIm2ColSdot(int8_t* colAddr, const int8_t* inputOrigin, int8_t inputZeroPoint,
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

/* End */
#undef GEMM_INT8_UNIT
#undef GEMM_INT8_SRC_UNIT
#undef GEMM_INT8_DST_XUNIT

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

#if defined(__aarch64__) && defined(MNN_USE_ARMV82)
    auto core = MNNGetCoreFunctions();
    if (core->supportSDot) {
        // MatMul
        gCoreFunc->Int8GemmKernel = MNNGemmInt8AddBiasScale_ARMV82_Unit;
        gCoreFunc->Int8GemmKernelFast = MNNGemmInt8AddBiasScale_ARMV82_Unit;
        gCoreFunc->MNNGetGemmUnit = MNNGetGemmUnitSdot;
        // Im2Col
        gCoreFunc->chooseIm2Col = chooseIm2ColSdot;
    }
#endif
    MNNInt8FunctionInit();
}
CoreInt8Functions* MNNGetInt8CoreFunctions() {
    return gCoreFunc;
}
};
