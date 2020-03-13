//
//  ImageSampler.cpp
//  MNN
//
//  Created by MNN on 2018/12/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "cv/ImageSampler.hpp"
#include <algorithm>
#include "core/Macro.h"
#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif
extern "C" {
void MNNSamplerC4BilinearOpt(const unsigned char* source, unsigned char* dest, float* points, size_t count, size_t xMax,
                             size_t yMax, size_t yStride);
void MNNSamplerC1BilinearOpt(const unsigned char* source, unsigned char* dest, float* points, size_t count, size_t xMax,
                             size_t yMax, size_t yStride);

void MNNSamplerC4NearestOpt(const unsigned char* source, unsigned char* dest, float* points, size_t count, size_t iw,
                            size_t ih, size_t yStride);
void MNNSamplerC1NearestOpt(const unsigned char* source, unsigned char* dest, float* points, size_t count, size_t iw,
                            size_t ih, size_t yStride);
}

namespace MNN {
namespace CV {

static inline float __clamp(float v, float minV, float maxV) {
    return std::max(std::min(v, maxV), minV);
}

static void _sampleBilinearCommon(const unsigned char* source, unsigned char* dest, Point* points, size_t count,
                                  size_t iw, size_t ih, size_t yStride, size_t bpp) {
    float dy   = points[1].fY;
    float dx   = points[1].fX;
    float xMax = iw - 1;
    float yMax = ih - 1;

    Point curPoints;
    curPoints.fX = points[0].fX;
    curPoints.fY = points[0].fY;
    for (int i = 0; i < count; ++i) {
        float y  = __clamp(curPoints.fY, 0, yMax);
        float x  = __clamp(curPoints.fX, 0, xMax);
        int y0   = (int)y;
        int x0   = (int)x;
        int y1   = (int)ceilf(y);
        int x1   = (int)ceilf(x);
        float xF = x - (float)x0;
        float yF = y - (float)y0;

        for (int b = 0; b < bpp; ++b) {
            unsigned char c00 = source[y0 * yStride + bpp * x0 + b];
            unsigned char c01 = source[y0 * yStride + bpp * x1 + b];
            unsigned char c10 = source[y1 * yStride + bpp * x0 + b];
            unsigned char c11 = source[y1 * yStride + bpp * x1 + b];

            float v =
                (1.0f - xF) * (1.0f - yF) * c00 + xF * (1.0f - yF) * c01 + yF * (1.0 - xF) * c10 + xF * yF * (c11);
            v                 = std::min(std::max(v, 0.0f), 255.0f);
            dest[bpp * i + b] = (unsigned char)v;
        }
        curPoints.fY += dy;
        curPoints.fX += dx;
    }
}

static void MNNSamplerC4Bilinear(const unsigned char* source, unsigned char* dest, Point* points, size_t sta,
                                 size_t count, size_t capacity, size_t iw, size_t ih, size_t yStride) {
#ifdef MNN_USE_NEON
    MNNSamplerC4BilinearOpt(source, dest + 4 * sta, reinterpret_cast<float*>(points), count, iw - 1, ih - 1, yStride);
#else
    _sampleBilinearCommon(source, dest + 4 * sta, points, count, iw, ih, yStride, 4);
#endif
}
static void MNNSamplerC3Bilinear(const unsigned char* source, unsigned char* dest, Point* points, size_t sta,
                                 size_t count, size_t capacity, size_t iw, size_t ih, size_t yStride) {
    _sampleBilinearCommon(source, dest + 3 * sta, points, count, iw, ih, yStride, 3);
}
static void MNNSamplerC1Bilinear(const unsigned char* source, unsigned char* dest, Point* points, size_t sta,
                                 size_t count, size_t capacity, size_t iw, size_t ih, size_t yStride) {
#ifdef MNN_USE_NEON
    MNNSamplerC1BilinearOpt(source, dest + sta, reinterpret_cast<float*>(points), count, iw - 1, ih - 1, yStride);
#else
    _sampleBilinearCommon(source, dest + sta, points, count, iw, ih, yStride, 1);
#endif
}
static void MNNSamplerNearest(const unsigned char* source, unsigned char* dest, Point* points, size_t sta, size_t count,
                              size_t iw, size_t ih, size_t yStride, int bpp) {
    dest = dest + bpp * sta;
    Point curPoints;
    curPoints.fX = points[0].fX;
    curPoints.fY = points[0].fY;
    float dy     = points[1].fY;
    float dx     = points[1].fX;
    float xMax   = iw - 1;
    float yMax   = ih - 1;

    for (int i = 0; i < count; ++i) {
        int y = (int)roundf(__clamp(curPoints.fY, 0, yMax));
        int x = (int)roundf(__clamp(curPoints.fX, 0, xMax));
        curPoints.fY += dy;
        curPoints.fX += dx;
        auto sourcePos = y * yStride + bpp * x;
        for (int j = 0; j < bpp; ++j) {
            dest[bpp * i + j] = source[sourcePos + j];
        }
    }
}

static void MNNSamplerC4Nearest(const unsigned char* source, unsigned char* dest, Point* points, size_t sta,
                                size_t count, size_t capacity, size_t iw, size_t ih, size_t yStride) {
#ifdef MNN_USE_NEON
    MNNSamplerC4NearestOpt(source, dest + 4 * sta, (float*)points, count, iw - 1, ih - 1, yStride);
#else
    MNNSamplerNearest(source, dest, points, sta, count, iw, ih, yStride, 4);
#endif
}

static void MNNSamplerC1Nearest(const unsigned char* source, unsigned char* dest, Point* points, size_t sta,
                                size_t count, size_t capacity, size_t iw, size_t ih, size_t yStride) {
#ifdef MNN_USE_NEON
    MNNSamplerC1NearestOpt(source, dest + sta, (float*)points, count, iw - 1, ih - 1, yStride);
#else
    MNNSamplerNearest(source, dest, points, sta, count, iw, ih, yStride, 1);
#endif
}

static void MNNSamplerC3Nearest(const unsigned char* source, unsigned char* dest, Point* points, size_t sta,
                                size_t count, size_t capacity, size_t iw, size_t ih, size_t yStride) {
    MNNSamplerNearest(source, dest, points, sta, count, iw, ih, yStride, 3);
}
static void MNNSamplerCopyCommon(const unsigned char* source, unsigned char* dest, Point* points, size_t sta,
                                 size_t count, size_t iw, size_t ih, size_t yStride, int bpp) {
    dest = dest + bpp * sta;
    Point curPoints;
    curPoints.fX   = points[0].fX;
    curPoints.fY   = points[0].fY;
    float xMax     = iw - 1;
    float yMax     = ih - 1;
    int y          = (int)roundf(__clamp(curPoints.fY, 0, yMax));
    int x          = (int)roundf(__clamp(curPoints.fX, 0, xMax));
    auto sourcePos = y * yStride + bpp * x;
    ::memcpy(dest, source + sourcePos, bpp * count);
}

static void MNNSamplerC1Copy(const unsigned char* source, unsigned char* dest, Point* points, size_t sta, size_t count,
                             size_t capacity, size_t iw, size_t ih, size_t yStride) {
    MNNSamplerCopyCommon(source, dest, points, sta, count, iw, ih, yStride, 1);
}
static void MNNSamplerC3Copy(const unsigned char* source, unsigned char* dest, Point* points, size_t sta, size_t count,
                             size_t capacity, size_t iw, size_t ih, size_t yStride) {
    MNNSamplerCopyCommon(source, dest, points, sta, count, iw, ih, yStride, 3);
}

static void MNNSamplerC4Copy(const unsigned char* source, unsigned char* dest, Point* points, size_t sta, size_t count,
                             size_t capacity, size_t iw, size_t ih, size_t yStride) {
    MNNSamplerCopyCommon(source, dest, points, sta, count, iw, ih, yStride, 4);
}

static void MNNSamplerNV21Copy(const unsigned char* source, unsigned char* dest, Point* points, size_t sta,
                               size_t count, size_t capacity, size_t iw, size_t ih, size_t yStride) {
    Point curPoints;
    curPoints.fX    = points[0].fX;
    curPoints.fY    = points[0].fY;
    float xMax      = iw - 1;
    float yMax      = ih - 1;
    int y           = (int)roundf(__clamp(curPoints.fY, 0, yMax));
    int x           = (int)roundf(__clamp(curPoints.fX, 0, xMax));
    int sourcePosY  = y * (int)iw + x;
    int sourcePosUV = (int)iw * (int)ih + (y / 2) * (((int)iw + 1) / 2) * 2 + (x / 2) * 2;

    ::memcpy(dest + sta, source + sourcePosY, count);
    ::memcpy(dest + (capacity) + (sta / 2) * 2, source + sourcePosUV, ((count + 1) / 2) * 2);
}

static void MNNSamplerNV21Nearest(const unsigned char* source, unsigned char* dest, Point* points, size_t sta,
                                  size_t count, size_t capacity, size_t iw, size_t ih, size_t yStride) {
    auto srcY  = source;

    auto dstY  = dest + sta;
    auto dstUV = dest + (capacity) + (sta / 2) * 2;
    auto stride = yStride;
    if (yStride == 0) {
        stride = iw;
    }
    auto srcUV = source + stride * ih;
    MNNSamplerC1Nearest(srcY, dstY, points, 0, count, capacity, iw, ih, stride);

    Point uvPoints[2];
    uvPoints[0].fX = (points[0].fX - 0.01f) / 2.0f;
    uvPoints[0].fY = (points[0].fY - 0.01f) / 2.0f;
    uvPoints[1].fX = points[1].fX;
    uvPoints[1].fY = points[1].fY;
    if (yStride == 0) {
        stride =  ((iw + 1) / 2) * 2;
    }
    MNNSamplerNearest(srcUV, dstUV, uvPoints, 0, (count + 1) / 2, (iw + 1) / 2, (ih + 1) / 2, stride, 2);
}

static void _swapUV(const unsigned char* source, unsigned char* dest, size_t countC2) {
    int sta = 0;
#ifdef MNN_USE_NEON
    int countC2C16 = (int)countC2 / 16;
    sta = countC2C16 * 16;
    for (int i=0; i<countC2C16; ++i) {
        auto src = vld2q_u8(source + i * 32);
        auto temp = src.val[0];
        src.val[0] = src.val[1];
        src.val[1] = temp;
        vst2q_u8(dest + i * 32, src);
    }
#endif
    for (int i=sta; i < countC2; ++i) {
        auto temp = source[2*i];
        dest[2*i] = source[2*i+1];
        dest[2*i+1] = temp;
    }
}

static void MNNSamplerNV12Copy(const unsigned char* source, unsigned char* dest, Point* points, size_t sta,
                               size_t count, size_t capacity, size_t iw, size_t ih, size_t yStride) {
    MNNSamplerNV21Copy(source, dest, points, sta, count, capacity, iw, ih, yStride);
    auto destUV = dest + (capacity) + (sta / 2) * 2;
    auto countC2 = ((count + 1) / 2);
    _swapUV(destUV, destUV, countC2);
}

static void MNNSamplerNV12Nearest(const unsigned char* source, unsigned char* dest, Point* points, size_t sta,
                                  size_t count, size_t capacity, size_t iw, size_t ih, size_t yStride) {
    MNNSamplerNV21Nearest(source, dest, points, sta, count, capacity, iw, ih, yStride);
    auto destUV = dest + (capacity) + (sta / 2) * 2;
    auto countC2 = ((count + 1) / 2);
    _swapUV(destUV, destUV, countC2);
}


ImageSampler::PROC ImageSampler::choose(ImageFormat format, Filter type, bool identity) {
    if (identity) {
        switch (format) {
            case RGBA:
            case BGRA:
                return MNNSamplerC4Copy;
            case GRAY:
                return MNNSamplerC1Copy;

            case RGB:
            case BGR:
                return MNNSamplerC3Copy;
            case YUV_NV21:
                return MNNSamplerNV21Copy;
            case YUV_NV12:
                return MNNSamplerNV12Copy;
            default:
                break;
        }
    }
    if (BILINEAR == type) {
        switch (format) {
            case RGBA:
            case BGRA:
                return MNNSamplerC4Bilinear;
            case GRAY:
                return MNNSamplerC1Bilinear;

            case RGB:
            case BGR:
                return MNNSamplerC3Bilinear;
            default:
                break;
        }
    }

    // Nearest
    switch (format) {
        case RGBA:
        case BGRA:
            return MNNSamplerC4Nearest;
        case GRAY:
            return MNNSamplerC1Nearest;

        case RGB:
        case BGR:
            return MNNSamplerC3Nearest;
        case YUV_NV12:
            return MNNSamplerNV12Nearest;
        case YUV_NV21:
            return MNNSamplerNV21Nearest;
        default:
            break;
    }
    MNN_PRINT("Don't support sampler for format:%d, type:%d", format, type);
    return nullptr;
}

} // namespace CV
} // namespace MNN
