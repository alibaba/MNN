//
//  ConvOpt.cpp
//  MNN
//
//  Created by MNN on 2018/07/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/compute/ConvOpt.h"
#include <algorithm>
#include "core/Macro.h"
#include "math/Vec4.hpp"
using namespace MNN::Math;
#ifndef MNN_USE_NEON
#ifndef MNN_USE_SSE

void MNNMatrixSub(float* C, const float* A, const float* B, size_t widthC4, size_t cStride, size_t aStride,
                  size_t bStride, size_t height) {
    for (int y = 0; y < height; ++y) {
        auto a = A + aStride * y;
        auto b = B + bStride * y;
        auto c = C + cStride * y;
        for (int x = 0; x < widthC4; ++x) {
            for (int j = 0; j < 4; ++j) {
                c[4 * x + j] = a[4 * x + j] - b[4 * x + j];
            }
        }
    }
}
void MNNMatrixAdd(float* C, const float* A, const float* B, size_t widthC4, size_t cStride, size_t aStride,
                  size_t bStride, size_t height) {
    for (int y = 0; y < height; ++y) {
        auto a = A + aStride * y;
        auto b = B + bStride * y;
        auto c = C + cStride * y;
        for (int x = 0; x < widthC4; ++x) {
            for (int j = 0; j < 4; ++j) {
                c[4 * x + j] = a[4 * x + j] + b[4 * x + j];
            }
        }
    }
}

void MNNConvSlideWindowBorder(float* dst, const float* src, const float* weight, size_t src_depth_quad,
                              size_t src_depth_step, size_t fw, size_t fh, size_t weight_y_step, size_t weight_z_step,
                              size_t dilateX_step, size_t dilateY_step, float* alpha) {
    int sz, fx, fy;
    for (int i = 0; i < 4; ++i) {
        dst[i] = 0.0f;
    }
    for (sz = 0; sz < src_depth_quad; ++sz) {
        const float* src_z    = src + sz * src_depth_step;
        const float* weight_z = weight + sz * weight_z_step;
        for (fy = 0; fy < fh; ++fy) {
            const float* src_y    = src_z + fy * dilateY_step;
            const float* weight_y = weight_z + fy * weight_y_step;
            for (fx = 0; fx < fw; ++fx) {
                const float* weight_x = weight_y + 16 * fx;
                const float* src_x    = src_y + fx * dilateX_step;
                for (int i = 0; i < 4; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        dst[j] += src_x[i] * weight_x[4 * i + j];
                    }
                }
            }
        }
    }
}

void MNNConvSlideWindowMiddle(float* dst, const float* src, const float* weight, size_t width, size_t src_w_setup,
                              size_t src_depth_quad, size_t src_depth_step, size_t fw, size_t fh, size_t dilateX_step,
                              size_t dilateY_step, float* alpha) {
    int dx, sz, fx, fy;
    for (dx = 0; dx < width; ++dx) {
        float* dst_x        = dst + dx * 4;
        dst_x[0]            = 0.0f;
        dst_x[1]            = 0.0f;
        dst_x[2]            = 0.0f;
        dst_x[3]            = 0.0f;
        const float* src_dx = src + src_w_setup * dx;
        for (sz = 0; sz < src_depth_quad; ++sz) {
            const float* src_z    = src_dx + sz * src_depth_step;
            const float* weight_z = weight + sz * fh * fw * 16;
            for (fy = 0; fy < fh; ++fy) {
                const float* src_y    = src_z + fy * dilateY_step;
                const float* weight_y = weight_z + fy * fw * 16;
                for (fx = 0; fx < fw; ++fx) {
                    const float* weight_x = weight_y + 16 * fx;
                    const float* src_x    = src_y + fx * dilateX_step;
                    for (int i = 0; i < 4; ++i) {
                        for (int j = 0; j < 4; ++j) {
                            dst_x[j] += src_x[i] * weight_x[4 * i + j];
                        }
                    }
                }
            }
        }
    }
}
void MNNGemmFloatCommon_4(float* dst, const float* src, const float* weight, size_t src_depth_quad, size_t dst_step,
                          size_t dst_depth_quad, size_t width, size_t weight_depth_offset) {
    int dx, sz, dz;
    auto src_depth_step = 4 * width;
    for (dz = 0; dz < dst_depth_quad; ++dz) {
        float* dst_z   = dst + dz * dst_step;
        auto weight_dz = weight + dz * (src_depth_quad * 16 + weight_depth_offset);
        for (dx = 0; dx < width; ++dx) {
            float* dst_x        = dst_z + dx * 4;
            dst_x[0]            = 0.0f;
            dst_x[1]            = 0.0f;
            dst_x[2]            = 0.0f;
            dst_x[3]            = 0.0f;
            const float* src_dx = src + 4 * dx;
            for (sz = 0; sz < src_depth_quad; ++sz) {
                const float* src_z    = src_dx + sz * src_depth_step;
                const float* weight_z = weight_dz + sz * 16;
                for (int i = 0; i < 4; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        dst_x[j] += src_z[i] * weight_z[4 * i + j];
                    }
                }
            }
        }
    }
}

#endif

void MNNConvRunForUnitDepthWise(float* dst, const float* src, const float* weight, size_t fw, size_t fh,
                                size_t weight_y_step, size_t dilateX_step, size_t dilateY_step) {
    int fx, fy;
    Vec4 dstValue(0.0f);
    const float* src_z    = src;
    const float* weight_z = weight;
    for (fy = 0; fy < fh; ++fy) {
        const float* src_y    = src_z + fy * dilateY_step;
        const float* weight_y = weight_z + fy * weight_y_step;
        for (fx = 0; fx < fw; ++fx) {
            const float* weight_x = weight_y + 4 * fx;
            const float* src_x    = src_y + fx * dilateX_step;
            dstValue = dstValue + Vec4::load(src_x) * Vec4::load(weight_x);
        }
    }
    Vec4::save(dst, dstValue);
}

void MNNConvRunForLineDepthwise(float* dst, const float* src, const float* weight, size_t width, size_t src_w_setup,
                                size_t fw, size_t fh, size_t dilateX_step, size_t dilateY_step, size_t height,
                                size_t srcHStep, size_t dstHStep) {
    int dx, fx, fy;
    for (int y = 0; y < height; ++y) {
        auto srcY = src + y * srcHStep;
        auto dstY = dst + y * dstHStep;
        for (dx = 0; dx < width; ++dx) {
            float* dst_x          = dstY + dx * 4;
            Vec4 dstValue(0.0f);
            const float* src_z    = srcY + src_w_setup * dx;
            const float* weight_z = weight;
            for (fy = 0; fy < fh; ++fy) {
                const float* src_y    = src_z + fy * dilateY_step;
                const float* weight_y = weight_z + fy * fw * 4;
                for (fx = 0; fx < fw; ++fx) {
                    const float* weight_x = weight_y + 4 * fx;
                    const float* src_x    = src_y + fx * dilateX_step;
                    dstValue = dstValue + Vec4::load(src_x) * Vec4::load(weight_x);
                }
            }
            Vec4::save(dst_x, dstValue);
        }
    }
}

void MNNConvRunForUnitint8_t(float* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad,
                             size_t src_depth_step, size_t fw, size_t fh, size_t weight_y_step, size_t weight_z_step,
                             size_t dilateX_step, size_t dilateY_step, float* alpha) {
    int sz, fx, fy;
    for (int i = 0; i < 4; ++i) {
        dst[i] = 0;
    }
    for (sz = 0; sz < src_depth_quad; ++sz) {
        const int8_t* src_z    = src + sz * src_depth_step;
        const int8_t* weight_z = weight + sz * weight_z_step;
        for (fy = 0; fy < fh; ++fy) {
            const int8_t* src_y    = src_z + fy * dilateY_step;
            const int8_t* weight_y = weight_z + fy * weight_y_step;
            for (fx = 0; fx < fw; ++fx) {
                const int8_t* weight_x = weight_y + 16 * fx;
                const int8_t* src_x    = src_y + fx * dilateX_step;
                for (int i = 0; i < 4; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        dst[j] += src_x[i] * weight_x[4 * i + j];
                    }
                }
            }
        }
    }
    for (int i = 0; i < 4; ++i) {
        dst[i] = ((float)dst[i]) * alpha[i];
    }
}

void MNNConvRunForLineint8_t(float* dst, const int8_t* src, const int8_t* weight, size_t width, size_t src_w_setup,
                             size_t src_depth_quad, size_t src_depth_step, size_t fw, size_t fh, size_t dilateX_step,
                             size_t dilateY_step, float* alpha) {
    int dx, sz, fx, fy;
    for (dx = 0; dx < width; ++dx) {
        float* dst_x         = dst + dx * 4;
        dst_x[0]             = 0.0f;
        dst_x[1]             = 0.0f;
        dst_x[2]             = 0.0f;
        dst_x[3]             = 0.0f;
        const int8_t* src_dx = src + src_w_setup * dx;
        for (sz = 0; sz < src_depth_quad; ++sz) {
            const int8_t* src_z    = src_dx + sz * src_depth_step;
            const int8_t* weight_z = weight + sz * fh * fw * 16;
            for (fy = 0; fy < fh; ++fy) {
                const int8_t* src_y    = src_z + fy * dilateY_step;
                const int8_t* weight_y = weight_z + fy * fw * 16;
                for (fx = 0; fx < fw; ++fx) {
                    const int8_t* weight_x = weight_y + 16 * fx;
                    const int8_t* src_x    = src_y + fx * dilateX_step;
                    for (int i = 0; i < 4; ++i) {
                        for (int j = 0; j < 4; ++j) {
                            dst_x[j] += src_x[i] * weight_x[4 * i + j];
                        }
                    }
                }
            }
        }
        for (int i = 0; i < 4; ++i) {
            dst_x[i] *= alpha[i];
        }
    }
}

void MNNGemmFloatUnit_4(float* dstOrigin, const float* src, const float* weight, size_t src_depth_quad, size_t dst_step,
                        size_t dst_depth_quad, size_t weight_depth_offset) {
    MNNGemmFloatCommon_4(dstOrigin, src, weight, src_depth_quad, dst_step, dst_depth_quad, CONVOLUTION_TILED_NUMBER,
                         weight_depth_offset);
}
void MNNGemmFloatOne_4(float* dstOrigin, const float* src, const float* weight, size_t src_depth_quad, size_t dst_step,
                       size_t dst_depth_quad, size_t weight_depth_offset) {
    MNNGemmFloatCommon_4(dstOrigin, src, weight, src_depth_quad, dst_step, dst_depth_quad, 1, weight_depth_offset);
}

void MNNDeconvRunForUnitDepthWise(const float* dst, float* src, const float* weight, size_t fw, size_t fh,
                                  size_t weight_y_step, size_t dilateX_step, size_t dilateY_step) {
    int fx, fy;
    float* src_z          = src;
    const float* weight_z = weight;
    Vec4 dstV             = Vec4::load(dst);
    for (fy = 0; fy < fh; ++fy) {
        float* src_y          = src_z + fy * dilateY_step;
        const float* weight_y = weight_z + fy * weight_y_step;
        for (fx = 0; fx < fw; ++fx) {
            Vec4 weight_x = Vec4::load(weight_y + 4 * fx);
            Vec4 src_x    = Vec4::load(src_y + fx * dilateX_step);
            Vec4::save(src_y + fx * dilateX_step, src_x + weight_x * dstV);
        }
    }
}
void MNNMatrixProd(float* C, const float* A, const float* B, size_t widthC4, size_t cStride, size_t aStride,
                   size_t bStride, size_t height) {
    for (int y = 0; y < height; ++y) {
        auto a = A + aStride * y;
        auto b = B + bStride * y;
        auto c = C + cStride * y;
        for (int x = 0; x < widthC4; ++x) {
            auto aV = Vec4::load(a + 4 * x);
            auto bV = Vec4::load(b + 4 * x);
            Vec4::save(c + 4 * x, aV * bV);
        }
    }
}
void MNNMatrixMax(float* C, const float* A, const float* B, size_t widthC4, size_t cStride, size_t aStride,
                  size_t bStride, size_t height) {
    for (int y = 0; y < height; ++y) {
        auto a = A + aStride * y;
        auto b = B + bStride * y;
        auto c = C + cStride * y;
        for (int x = 0; x < widthC4; ++x) {
            for (int j = 0; j < 4; ++j) {
                c[4 * x + j] = std::max(a[4 * x + j], b[4 * x + j]);
            }
        }
    }
}
#endif

void MNNDeconvRunForLineDepthwise(const float* dst, float* src, const float* weight, size_t width, size_t src_w_setup,
                                  size_t fw, size_t fh, size_t dilateX_step, size_t dilateY_step) {
    int dx;
    for (dx = 0; dx < width; ++dx) {
        const float* dst_x = dst + dx * 4;
        float* src_dx      = src + src_w_setup * dx;
        MNNDeconvRunForUnitDepthWise(dst_x, src_dx, weight, fw, fh, fw * 4, dilateX_step, dilateY_step);
    }
}

void MNNMatrixProdCommon(float* C, const float* A, const float* B, size_t width, size_t cStride, size_t aStride, size_t bStride, size_t height) {
    int widthC4 = (int)width / 4;
    if (widthC4 > 0) {
        MNNMatrixProd(C, A, B, widthC4, cStride, aStride, bStride, height);
        width = width - 4*widthC4;
        C = C + widthC4 * 4;
        A = A + widthC4 * 4;
        B = B + widthC4 * 4;
    }
    if (width > 0) {
        for (int y = 0; y < height; ++y) {
            auto a = A + aStride * y;
            auto b = B + bStride * y;
            auto c = C + cStride * y;
            for (int x = 0; x < width; ++x) {
                c[x] = b[x] * a[x];
            }
        }
    }
}

void MNNMatrixAddCommon(float* C, const float* A, const float* B, size_t width, size_t cStride, size_t aStride, size_t bStride, size_t height) {
    int widthC4 = (int)width / 4;
    if (widthC4 > 0) {
        MNNMatrixAdd(C, A, B, widthC4, cStride, aStride, bStride, height);
        width = width - 4*widthC4;
        C = C + widthC4 * 4;
        A = A + widthC4 * 4;
        B = B + widthC4 * 4;
    }
    if (width > 0) {
        for (int y = 0; y < height; ++y) {
            auto a = A + aStride * y;
            auto b = B + bStride * y;
            auto c = C + cStride * y;
            for (int x = 0; x < width; ++x) {
                c[x] = a[x] + b[x];
            }
        }
    }
}

void MNNMatrixSubCommon(float* C, const float* A, const float* B, size_t width, size_t cStride, size_t aStride, size_t bStride, size_t height) {
    int widthC4 = (int)width / 4;
    if (widthC4 > 0) {
        MNNMatrixSub(C, A, B, widthC4, cStride, aStride, bStride, height);
        width = width - 4*widthC4;
        C = C + widthC4 * 4;
        A = A + widthC4 * 4;
        B = B + widthC4 * 4;
    }
    if (width > 0) {
        for (int y = 0; y < height; ++y) {
            auto a = A + aStride * y;
            auto b = B + bStride * y;
            auto c = C + cStride * y;
            for (int x = 0; x < width; ++x) {
                c[x] = a[x] - b[x];
            }
        }
    }
}

void MNNMatrixMaxCommon(float* C, const float* A, const float* B, size_t width, size_t cStride, size_t aStride, size_t bStride, size_t height) {
    int widthC4 = (int)width / 4;
    if (widthC4 > 0) {
        MNNMatrixMax(C, A, B, widthC4, cStride, aStride, bStride, height);
        width = width - 4*widthC4;
        C = C + widthC4 * 4;
        A = A + widthC4 * 4;
        B = B + widthC4 * 4;
    }
    if (width > 0) {
        for (int y = 0; y < height; ++y) {
            auto a = A + aStride * y;
            auto b = B + bStride * y;
            auto c = C + cStride * y;
            for (int x = 0; x < width; ++x) {
                c[x] = std::max(b[x], a[x]);
            }
        }
    }
}
