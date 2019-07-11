//
//  Int8FunctionsOpt.cpp
//  MNN
//
//  Created by MNN on 2018/08/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Int8FunctionsOpt.h"
#include <algorithm>
#include "Macro.h"

static const int gUnit = 8;

#ifndef MNN_USE_NEON
#include <math.h>
static const int gUnit2 = 64;

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

void MNNConvolutionInt8Run8x8(int16_t* dst_x, const int8_t* src_unit, const int8_t* weight_start, size_t icD8,
                              size_t xCount, size_t yCount, size_t dilateY_step, size_t dilateX_step,
                              size_t weight_sy_step) {
    dilateX_step   = dilateX_step + gUnit * icD8;
    dilateY_step   = dilateY_step + dilateX_step * xCount;
    weight_sy_step = weight_sy_step + gUnit2 * icD8 * xCount;
    for (int i = 0; i < gUnit; ++i) {
        dst_x[i] = 0;
    }

    for (int fy = 0; fy < yCount; ++fy) {
        auto src_y    = src_unit + fy * dilateY_step;
        auto weight_y = weight_start + fy * weight_sy_step;
        for (int fx = 0; fx < xCount; ++fx) {
            auto src_x    = src_y + fx * dilateX_step;
            auto weight_x = weight_y + fx * icD8 * gUnit2;
            for (int sz = 0; sz < icD8; ++sz) {
                auto src_z    = src_x + sz * gUnit;
                auto weight_z = weight_x + sz * gUnit2;
                for (int i = 0; i < gUnit; ++i) {
                    for (int j = 0; j < gUnit; ++j) {
                        dst_x[i] += src_z[j] * weight_z[gUnit * i + j];
                    }
                }
            }
        }
    }
}

void MNNScaleBias2FloatC4(float* dst, const int16_t* src, const float* alpha, const float* bias, size_t sizeQuad) {
    for (int i = 0; i < sizeQuad; ++i) {
        for (int j = 0; j < 4; ++j) {
            dst[4 * i + j] = (float)src[gUnit * i + j] * alpha[j] + bias[j];
        }
    }
}

void MNNScaleBias2FloatC4Relu(float* dst, const int16_t* src, const float* alpha, const float* bias, size_t sizeQuad) {
    for (int i = 0; i < sizeQuad; ++i) {
        for (int j = 0; j < 4; ++j) {
            dst[4 * i + j] = (float)src[gUnit * i + j] * alpha[j] + bias[j];
            dst[4 * i + j] = dst[4 * i + j] > 0 ? dst[4 * i + j] : 0;
        }
    }
}

#endif
void MNNScaleBias2FloatC4Relu6(float* dst, const int16_t* src, const float* alpha, const float* bias, size_t sizeQuad) {
    for (int i = 0; i < sizeQuad; ++i) {
        for (int j = 0; j < 4; ++j) {
            dst[4 * i + j] = (float)src[gUnit * i + j] * alpha[j] + bias[j];
            dst[4 * i + j] = std::max(dst[4 * i + j], 0.0f);
            dst[4 * i + j] = std::min(dst[4 * i + j], 6.0f);
        }
    }
}
#ifndef MNN_USE_NEON

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
#endif
