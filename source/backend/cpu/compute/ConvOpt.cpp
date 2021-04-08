//
//  ConvOpt.cpp
//  MNN
//
//  Created by MNN on 2018/07/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/compute/ConvOpt.h"
#include <algorithm>
#include <string.h>
#include "core/Macro.h"
#include "math/Vec.hpp"
using Vec4 = MNN::Math::Vec<float, 4>;
#ifndef MNN_USE_NEON

void MNNMatrixSub(float* C, const float* A, const float* B, size_t widthC4, size_t cStride, size_t aStride,
                  size_t bStride, size_t height) {
    for (int y = 0; y < height; ++y) {
        auto a = A + aStride * y;
        auto b = B + bStride * y;
        auto c = C + cStride * y;
        for (int x = 0; x < widthC4; ++x) {
            Vec4::save(c + 4 * x, Vec4::load(a + 4 * x) - Vec4::load(b + 4 * x));
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
            Vec4::save(c + 4 * x, Vec4::load(a + 4 * x) + Vec4::load(b + 4 * x));
        }
    }
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
#ifndef MNN_USE_NEON
void MNNStrassenMergeCFunction(float* c11, float* c12, float* c21, float* c22, float* xAddr, size_t cStride,
                               size_t eSub, size_t hSub) {
    for (int y=0; y<hSub; ++y) {
        auto c11Y = c11 + y * cStride;
        auto c12Y = c12 + y * cStride;
        auto c22Y = c22 + y * cStride;
        auto c21Y = c21 + y * cStride;
        auto xY = xAddr + y * eSub * 4;
        for (int x=0; x<eSub; ++x) {
            auto xv = Vec4::load(xY + 4*x);
            auto c21v = Vec4::load(c21Y + 4*x);
            auto c11v = Vec4::load(c11Y + 4*x);
            auto c22v = Vec4::load(c22Y + 4*x);
            auto c12v = Vec4::load(c12Y + 4*x);
            c12v = c12v + xv;
            c21v = c12v + c21v;
            c12v = c22v + c12v;
            c22v = c22v + c21v;
            c12v = c11v + c12v;
            Vec4::save(c12Y + 4*x, c12v);
            Vec4::save(c22Y + 4*x, c22v);
            Vec4::save(c21Y + 4*x, c21v);
        }
    }
}
#endif
