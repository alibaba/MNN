//
//  Int8FunctionsOpt.cpp
//  MNN
//
//  Created by MNN on 2018/08/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Int8FunctionsOpt.h"
#include <algorithm>
#include "core/Macro.h"
#include <math.h>

#ifndef MNN_USE_NEON
#ifndef MNN_USE_SSE

inline int8_t int32ToInt8(int data, int bias, float scale) {
    float value = (float)(data + bias) * scale;
    value       = std::max(value, -127.0f);
    value       = std::min(value, 127.0f);
    return static_cast<int8_t>(roundf(value));
}

void MNNGemmInt8AddBiasScale_16x4_Unit(int8_t* dst, const int8_t* src, const int8_t* weight, const int32_t* bias,
                                              const float* scale, size_t src_depth_quad, size_t dst_step,
                                              size_t dst_depth_quad) {
    const auto dst_step_tmp = dst_step / sizeof(int8_t);
    for (int dz = 0; dz < dst_depth_quad; ++dz) {
        const auto weight_dz = weight + dz * src_depth_quad * (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT);
        const auto bias_dz   = bias + dz * GEMM_INT8_UNIT;
        const auto scale_dz  = scale + dz * GEMM_INT8_UNIT;
        auto dst_z           = dst + dz * dst_step_tmp;
        for (int w = 0; w < GEMM_INT8_DST_XUNIT; ++w) {
            const auto src_x   = src + w * GEMM_INT8_SRC_UNIT;
            auto dst_x         = dst_z + w * GEMM_INT8_UNIT;
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

            for (int j = 0; j < 4; ++j) {
                dst_x[j] = int32ToInt8(dstTemp[j], bias_dz[j], scale_dz[j]);
            }
        }
    }
}

void MNNGemmInt8AddBiasScale_16x4_Unit_FAST(int8_t* dst, const int8_t* src, const int8_t* weight, const int32_t* bias, const float* scale, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad){
    return MNNGemmInt8AddBiasScale_16x4_Unit(dst, src, weight, bias, scale, src_depth_quad, dst_step, dst_depth_quad);
}

#endif // no MNN_USE_SSE
void MNNFloat2Int8(const float* src, int8_t* dst, size_t sizeQuad, const float* scalep, ssize_t minValue,
                   ssize_t maxValue) {
    for (int i = 0; i < sizeQuad; ++i) {
        for (int j=0; j<4; ++j) {
            int v = (int)roundf((src[4*i+j] * scalep[j]));
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

static int gDepthwiseUnit = 4;
void MNNConvRunForUnitDepthWiseInt8(float* dst, const int8_t* src, const int8_t* weight, size_t fw, size_t fh,
                                    size_t weight_y_step, size_t dilateX_step, size_t dilateY_step,
                                    const float* scale) {
    int fx, fy;
    for (int i = 0; i < gDepthwiseUnit; ++i) {
        dst[i] = 0;
    }
    auto src_z    = src;
    auto weight_z = weight;
    for (fy = 0; fy < fh; ++fy) {
        auto src_y    = src_z + fy * dilateY_step;
        auto weight_y = weight_z + fy * weight_y_step;
        for (fx = 0; fx < fw; ++fx) {
            auto weight_x = weight_y + gDepthwiseUnit * fx;
            auto src_x    = src_y + fx * dilateX_step;
            for (int j = 0; j < gDepthwiseUnit; ++j) {
                dst[j] += (float)src_x[j] * (float)weight_x[j];
            }
        }
    }
    for (int i = 0; i < gDepthwiseUnit; ++i) {
        dst[i] = dst[i] * scale[i];
    }
}

#ifdef ENABLE_ARMV82

inline int8_t int32ToInt8T(int data, int bias, float scale) {
    float value = (float)(data + bias) * scale;
    value       = std::max(value, -127.0f);
    value       = std::min(value, 127.0f);
    return static_cast<int8_t>(roundf(value));
}
void MNNGemmInt8AddBiasScale_ARMV82_Unit(int8_t* dst, const int8_t* src, const int8_t* weight,
                                                const int32_t* bias, const float* scale, size_t src_depth_quad,
                                                size_t dst_step, size_t dst_depth_quad, size_t relu,
                                                size_t realDstCount) {
    const auto dst_step_tmp = dst_step / sizeof(int8_t);
    for (int dz = 0; dz < dst_depth_quad; ++dz) {
        const auto weight_dz = weight + dz * src_depth_quad * (GEMM_INT8_UNIT * GEMM_INT8_UNIT);
        const auto bias_dz   = bias + dz * GEMM_INT8_UNIT;
        const auto scale_dz  = scale + dz * GEMM_INT8_UNIT;
        auto dst_z           = dst + dz * dst_step_tmp;

        for (int w = 0; w < DST_XUNIT_ARMV82; ++w) {
            const auto src_x      = src + w * GEMM_INT8_UNIT;
            auto dst_x            = dst_z + w * GEMM_INT8_UNIT;
            int32_t dstTemp[GEMM_INT8_UNIT] = {0, 0, 0, 0};

            for (int sz = 0; sz < src_depth_quad; ++sz) {
                const auto weight_sz = weight_dz + (GEMM_INT8_UNIT * GEMM_INT8_UNIT) * sz;
                const auto src_z     = src_x + sz * DST_XUNIT_ARMV82 * GEMM_INT8_UNIT;

                for (int j = 0; j < GEMM_INT8_UNIT; ++j) {
                    const auto weight_j = weight_sz + j * GEMM_INT8_UNIT;
                    for (int i = 0; i < GEMM_INT8_UNIT; ++i) {
                        dstTemp[j] += (int32_t)src_z[i] * (int32_t)weight_j[i];
                    }
                }
            }

            for (int j = 0; j < GEMM_INT8_UNIT; ++j) {
                dst_x[j] = int32ToInt8T(dstTemp[j], bias_dz[j], scale_dz[j]);
            }

            if (relu) {
                for (int j = 0; j < GEMM_INT8_UNIT; ++j) {
                    if (dst_x[j] < 0) {
                        dst_x[j] = 0;
                    }
                }
            }
        }
    }
}
#endif // ENABLE_ARMV82

#endif // no MNN_USE_NEON
