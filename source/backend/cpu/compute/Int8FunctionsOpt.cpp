//
//  Int8FunctionsOpt.cpp
//  MNN
//
//  Created by MNN on 2018/08/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <cstring> // for memset
#include "Int8FunctionsOpt.h"
#include "core/Macro.h"

#ifndef MNN_USE_NEON
#include <math.h>

int8_t MNNInt32ToInt8(int data, int bias, float scale, float maxValue, float minValue)
{
    float value = (float)(data + bias) * scale;
    value       = ALIMAX(value, minValue);
    value       = ALIMIN(value, maxValue);
    return static_cast<int8_t>(roundf(value));
}

void MNNGemmInt8toFloat32_8x4_Unit(float* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad,
                                   size_t dst_step, size_t dst_depth_quad) {
    MNNGemmInt8toFloat32_8x4_Common(dst, src, weight, src_depth_quad, DST_XUNIT, dst_step, dst_depth_quad);
}

void MNNGemmInt8toFloat32_8x4_Common(float* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad,
                                     size_t width, size_t dst_step, size_t dst_depth_quad) {
    for (int dz = 0; dz < dst_depth_quad; ++dz) {
        auto weight_dz = weight + src_depth_quad * dz * 32;
        auto dst_z     = dst + dz * dst_step;
        for (int w = 0; w < width; ++w) {
            auto dst_x     = dst_z + 4 * w;
            float dst_4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            auto src_x     = src + 8 * w;
            for (int sz = 0; sz < src_depth_quad; ++sz) {
                auto weight_sz = weight_dz + 32 * sz;
                auto src_z     = src_x + sz * width * 8;
                for (int j = 0; j < 4; ++j) {
                    auto weight_j = weight_sz + j * 8;
                    for (int i = 0; i < 8; ++i) {
                        dst_4[j] += (int32_t)src_z[i] * (int32_t)weight_j[i];
                    }
                }
            }
            for (int j = 0; j < 4; ++j) {
                dst_x[j] = dst_4[j];
            }
        }
    }
}
#ifndef MNN_USE_SSE
void MNNGemmInt8AddBiasScale_16x4_Unit(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step,
                                              size_t dst_depth_quad, const QuanPostTreatParameters* post, size_t realCount) {
    const auto dst_step_tmp = dst_step / sizeof(int8_t);
    for (int dz = 0; dz < dst_depth_quad; ++dz) {
        const auto weight_dz = weight + dz * src_depth_quad * (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT);
        const auto bias_dz   = post->bias + dz * GEMM_INT8_UNIT;
        const auto scale_dz  = post->scale + dz * GEMM_INT8_UNIT;
        auto dst_z           = dst + dz * dst_step_tmp;
        for (int w = 0; w < realCount; ++w) {
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
                dst_x[j] = MNNInt32ToInt8(dstTemp[j], bias_dz[j], scale_dz[j], post->maxValue, post->minValue);
            }
        }
    }
}

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

void MNNGemmInt8AddBiasScale_16x4_Unit_Acc16(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step,
                                              size_t dst_depth_quad, const QuanPostTreatParameters* post, size_t realCount) {
    const auto dst_step_tmp = dst_step / sizeof(int8_t);
    for (int dz = 0; dz < dst_depth_quad; ++dz) {
        const auto weight_dz = weight + dz * src_depth_quad * (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT);
        const auto bias_dz   = post->bias + dz * GEMM_INT8_UNIT;
        const auto scale_dz  = post->scale + dz * GEMM_INT8_UNIT;
        auto dst_z           = dst + dz * dst_step_tmp;
        for (int w = 0; w < realCount; ++w) {
            const auto src_x   = src + w * GEMM_INT8_SRC_UNIT;
            auto dst_x         = dst_z + w * GEMM_INT8_UNIT;
            int16_t dstTemp[4] = {0, 0, 0, 0};

            for (int sz = 0; sz < src_depth_quad; ++sz) {
                const auto weight_sz = weight_dz + (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT) * sz;
                const auto src_z     = src_x + sz * GEMM_INT8_DST_XUNIT * GEMM_INT8_SRC_UNIT;

                for (int j = 0; j < GEMM_INT8_UNIT; ++j) {
                    const auto weight_j = weight_sz + j * GEMM_INT8_SRC_UNIT;
                    for (int i = 0; i < GEMM_INT8_SRC_UNIT; ++i) {
                        dstTemp[j] += (int16_t)src_z[i] * (int16_t)weight_j[i];
                    }
                }
            }

            for (int j = 0; j < 4; ++j) {
                dst_x[j] = MNNInt32ToInt8(dstTemp[j], bias_dz[j], scale_dz[j], post->maxValue, post->minValue);
            }
        }
    }
}

void MNNGemmInt8AddBiasScale_16x4_Unit_FAST(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad, const QuanPostTreatParameters* post, size_t realCount) {
    return MNNGemmInt8AddBiasScale_16x4_Unit(dst, src, weight, src_depth_quad, dst_step, dst_depth_quad, post, realCount);
}
#endif

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

#if defined(ENABLE_ARMV82) && (defined(__ANDROID__) || defined(__aarch64__))

inline int8_t MNNInt32ToInt8T(int data, int bias, float scale) {
    float value = (float)(data + bias) * scale;
    value       = ALIMAX(value, -127.0f);
    value       = ALIMIN(value, 127.0f);
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
                dst_x[j] = MNNInt32ToInt8T(dstTemp[j], bias_dz[j], scale_dz[j]);
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

#endif

#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif

void MNNInt8ToInt16C4(const int8_t* source, int16_t* dest, size_t sizeQuad) {
    auto sizeC8 = sizeQuad / 2;
    for (int i=0; i<sizeC8; ++i) {
        auto d = dest + 8 * i;
        auto s = source + 8 * i;
#ifdef MNN_USE_NEON
        auto s0 = vld1_s8(s);
        auto d0 = vmovl_s8(s0);
        vst1q_s16(d, d0);
#else
        for (int j=0; j<8; ++j) {
            d[j] = s[j];
        }
#endif
    }
    for (int i=sizeC8*2; i<sizeQuad; ++i) {
        auto d = dest + 4 * i;
        auto s = source + 4 * i;
        for (int j=0; j<4; ++j) {
            d[j] = s[j];
        }
    }
}

void MNNMatrixAddInt32(int32_t* C, const int32_t* A, const int32_t* B, size_t widthC4, size_t cStride,
                       size_t aStride, size_t bStride, size_t height) {
    for (int h = 0; h < height; ++h) {
        auto c = C + h * cStride;
        auto a = A + h * aStride;
        auto b = B + h * bStride;
        for (int w = 0; w < widthC4; ++w) {
#ifdef MNN_USE_NEON
            auto tmp = vld1q_s32(a + w * 4) + vld1q_s32(b + w * 4);
            vst1q_s32(c + w * 4, tmp);
#else
            for (int j = 0; j < 4; ++j) {
                c[4 * w + j] = a[4 * w + j] + b[4 * w + j];
            }
#endif
        }
    }
}

void MNNInt8C4ToC8(int8_t* dst, const int8_t* src, size_t area, size_t depth) {
    for (int d = 0; d * 2 + 1 < depth; ++d) {
        auto src_0 = src + 2 * d * area * 4, src_1 = src_0 + area * 4;
        auto dst_ = dst + d * area * 8;
        int i = 0;
#ifdef MNN_USE_NEON
        for (; i * 2 < area - 1; ++i) {
            auto dst_0 = dst_ + i * 8 * 2;
            auto m = vtrn_s32(vreinterpret_s32_s8(vld1_s8(src_0 + i * 8)), vreinterpret_s32_s8(vld1_s8(src_1 + i * 8)));
            vst1_s8(dst_0, vreinterpret_s8_s32(m.val[0]));
            vst1_s8(dst_0 + 8, vreinterpret_s8_s32(m.val[1]));
        }
#endif
        for (i = i * 2; i < area; ++i) {
            for (int j = 0; j < 4; ++j) {
                dst_[i * 8 + j] = src_0[i * 4 + j];
                dst_[i * 8 + j + 4] = src_1[i * 4 + j];
            }
        }
    }
    if (depth % 2 != 0) {
        auto src_ = src + (depth - 1) * area * 4;
        auto dst_ = dst + (depth / 2) * area * 8;
        for (int i = 0; i < area; ++i) {
            for (int k = 0; k < 4; ++k) {
                dst_[i * 8 + k] = src_[i * 4 + k];
                dst_[i * 8 + k + 4] = 0;
            }
        }
    }
}

void MNNInt8ClipInplace(int8_t* data, size_t size, int8_t minVal, int8_t maxVal) {
    auto sizeC8 = size / 8;
    for (int i = 0; i < sizeC8; ++i) {
        int8_t* ptr = data + i * 8;
#ifdef MNN_USE_NEON
        auto s = vld1_s8(ptr);
        auto d = vmin_s8(vmax_s8(s, vdup_n_s8(minVal)), vdup_n_s8(maxVal));
        vst1_s8(ptr, d);
#else
        for (int j=0; j<8; ++j) {
            ptr[j] = ALIMIN(ALIMAX(ptr[j], minVal), maxVal);
        }
#endif
    }
    for (auto i = sizeC8 * 8; i < size; ++i) {
        data[i] = ALIMIN(ALIMAX(data[i], minVal), maxVal);
    }
}

#ifndef MNN_USE_NEON
#ifndef MNN_USE_SSE
#define UNIT 4
void MNNLineDepthWiseInt8AddBiasScaleUnit(int8_t* dst, const int8_t* src, const int8_t* weight, const QuanPostTreatParameters* parameters,
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
                for (int j = 0; j < UNIT; ++j) {
                    dstInt32[j] += (int32_t)src_x[j] * (int32_t)weight_x[j];
                }
            }
        }

        for (int i = 0; i < UNIT; ++i) {
            dst_x[i] = MNNInt32ToInt8(dstInt32[i], bias_z[i], scale_z[i], parameters->maxValue, parameters->minValue);
        }
    }
}
#undef UNIT
#endif
#endif
