//
//  ImageProcessFunction.cpp
//  MNN
//
//  Created by MNN on 2021/10/29.
//  Copyright © 2018 Alibaba. All rights reserved.
//

#include "backend/cpu/compute/ImageProcessFunction.hpp"
#include "core/Macro.h"
#include <algorithm>
#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif

extern "C" {
void MNNNV21ToRGBUnit(const unsigned char* source, unsigned char* dest, size_t countDiv8, const unsigned char* uv);
void MNNNV21ToBGRUnit(const unsigned char* source, unsigned char* dest, size_t countDiv8, const unsigned char* uv);
void MNNNV21ToRGBAUnit(const unsigned char* source, unsigned char* dest, size_t countDiv8, const unsigned char* uv);
void MNNNV21ToBGRAUnit(const unsigned char* source, unsigned char* dest, size_t countDiv8, const unsigned char* uv);
void MNNSamplerC4BilinearOpt(const unsigned char* source, unsigned char* dest, float* points, size_t count, size_t xMax, size_t yMax, size_t yStride);
void MNNSamplerC1BilinearOpt(const unsigned char* source, unsigned char* dest, float* points, size_t count, size_t xMax, size_t yMax, size_t yStride);
void MNNSamplerC4NearestOpt(const unsigned char* source, unsigned char* dest, float* points, size_t count, size_t iw, size_t ih, size_t yStride);
void MNNSamplerC1NearestOpt(const unsigned char* source, unsigned char* dest, float* points, size_t count, size_t iw, size_t ih, size_t yStride);
void MNNBlitC1ToFloatRGBA(const unsigned char* source, float* dest, const float* mean, const float* normal, size_t count);
void MNNBlitC3ToFloatRGBA(const unsigned char* source, float* dest, const float* mean, const float* normal, size_t count);
}

void MNNGRAYToC4(const unsigned char* source, unsigned char* dest, size_t count) {
    int sta = 0;
#ifdef MNN_USE_NEON
    int countD8 = (int)count / 8;
    if (countD8 > 0) {
        for (int i = 0; i < countD8; ++i) {
            auto gray = vld1_u8(source + 8 * i);

            uint8x8x4_t rgba;
            rgba.val[0] = gray;
            rgba.val[1] = gray;
            rgba.val[2] = gray;
            rgba.val[3] = vdup_n_u8(255);
            vst4_u8(dest + 32 * i, rgba);
        }
        sta = countD8 * 8;
    }
#endif
    for (int i = sta; i < count; ++i) {
        dest[4 * i + 0] = source[i];
        dest[4 * i + 1] = source[i];
        dest[4 * i + 2] = source[i];
        dest[4 * i + 3] = 255;
    }
}

void MNNGRAYToC3(const unsigned char* source, unsigned char* dest, size_t count) {
    int sta = 0;
#ifdef MNN_USE_NEON
    int countD8 = (int)count / 8;
    if (countD8 > 0) {
        for (int i = 0; i < countD8; ++i) {
            auto gray = vld1_u8(source + 8 * i);

            uint8x8x3_t rgba;
            rgba.val[0] = gray;
            rgba.val[1] = gray;
            rgba.val[2] = gray;
            vst3_u8(dest + 24 * i, rgba);
        }
        sta = countD8 * 8;
    }
#endif
    for (int i = sta; i < count; ++i) {
        dest[3 * i + 0] = source[i];
        dest[3 * i + 1] = source[i];
        dest[3 * i + 2] = source[i];
    }
}

void MNNC3ToC4(const unsigned char* source, unsigned char* dest, size_t count) {
    int sta = 0;
#ifdef MNN_USE_NEON
    int countD8 = (int)count / 8;
    if (countD8 > 0) {
        for (int i = 0; i < countD8; ++i) {
            uint8x8x3_t c3 = vld3_u8(source + 24 * i);

            uint8x8x4_t c4;
            c4.val[0] = c3.val[0];
            c4.val[1] = c3.val[1];
            c4.val[2] = c3.val[2];
            c4.val[3] = vdup_n_u8(255);
            vst4_u8(dest + 32 * i, c4);
        }
        sta = countD8 * 8;
    }
#endif
    for (int i = sta; i < count; i++) {
        dest[i * 4 + 0] = source[i * 3 + 0];
        dest[i * 4 + 1] = source[i * 3 + 1];
        dest[i * 4 + 2] = source[i * 3 + 2];
        dest[i * 4 + 3] = 255;
    }
}

void MNNRGBAToBGRA(const unsigned char* source, unsigned char* dest, size_t count) {
    int sta = 0;
#ifdef MNN_USE_NEON
    int countD8 = (int)count / 8;
    if (countD8 > 0) {
        for (int i = 0; i < countD8; ++i) {
            uint8x8x4_t rgba = vld4_u8(source + 32 * i);
            auto t           = rgba.val[0];
            rgba.val[0]      = rgba.val[2];
            rgba.val[2]      = t;
            vst4_u8(dest + 32 * i, rgba);
        }
        sta = countD8 * 8;
    }
#endif
    for (int i = sta; i < count; ++i) {
        dest[4 * i + 0] = source[4 * i + 2];
        dest[4 * i + 1] = source[4 * i + 1];
        dest[4 * i + 2] = source[4 * i + 0];
        dest[4 * i + 3] = source[4 * i + 3];
    }
}

void MNNRGBAToBGR(const unsigned char* source, unsigned char* dest, size_t count) {
    int sta = 0;
#ifdef MNN_USE_NEON
    int countD8 = (int)count / 8;
    if (countD8 > 0) {
        for (int i = 0; i < countD8; ++i) {
            uint8x8x4_t rgba = vld4_u8(source + 32 * i);

            uint8x8x3_t bgr;
            bgr.val[0] = rgba.val[2];
            bgr.val[1] = rgba.val[1];
            bgr.val[2] = rgba.val[0];
            vst3_u8(dest + 24 * i, bgr);
        }
        sta = countD8 * 8;
    }
#endif
    for (int i = sta; i < count; ++i) {
        dest[3 * i + 0] = source[4 * i + 2];
        dest[3 * i + 1] = source[4 * i + 1];
        dest[3 * i + 2] = source[4 * i + 0];
    }
}

void MNNRGBToBGR(const unsigned char* source, unsigned char* dest, size_t count) {
    int sta = 0;
#ifdef MNN_USE_NEON
    int countD8 = (int)count / 8;
    if (countD8 > 0) {
        for (int i = 0; i < countD8; ++i) {
            uint8x8x3_t rgba = vld3_u8(source + 24 * i);
            uint8x8x3_t bgr;
            bgr.val[0] = rgba.val[2];
            bgr.val[1] = rgba.val[1];
            bgr.val[2] = rgba.val[0];
            vst3_u8(dest + 24 * i, bgr);
        }
        sta = countD8 * 8;
    }
#endif
    for (int i = sta; i < count; ++i) {
        dest[3 * i + 0] = source[3 * i + 2];
        dest[3 * i + 1] = source[3 * i + 1];
        dest[3 * i + 2] = source[3 * i + 0];
    }
}

void MNNBGRAToBGR(const unsigned char* source, unsigned char* dest, size_t count) {
    int sta = 0;
#ifdef MNN_USE_NEON
    int countD8 = (int)count / 8;
    if (countD8 > 0) {
        for (int i = 0; i < countD8; ++i) {
            uint8x8x4_t bgra = vld4_u8(source + 32 * i);

            uint8x8x3_t bgr;
            bgr.val[0] = bgra.val[0];
            bgr.val[1] = bgra.val[1];
            bgr.val[2] = bgra.val[2];
            vst3_u8(dest + 24 * i, bgr);
        }
        sta = countD8 * 8;
    }
#endif
    for (int i = sta; i < count; ++i) {
        dest[3 * i + 0] = source[4 * i + 0];
        dest[3 * i + 1] = source[4 * i + 1];
        dest[3 * i + 2] = source[4 * i + 2];
    }
}

void MNNBGRAToGRAY(const unsigned char* source, unsigned char* dest, size_t count) {
    int sta = 0;
    /*
#ifdef MNN_USE_NEON
    int countD8 = (int)count / 8;
    if (countD8 > 0) {
        auto rC = vdup_n_u8(19);
        auto gC = vdup_n_u8(38);
        auto bC = vdup_n_u8(7);
        for (int i = 0; i < countD8; ++i) {
            auto rgb   = vld4_u8(source + 32 * i);
            auto res   = vmull_u8(rC, rgb.val[2]) + vmull_u8(gC, rgb.val[1]) + vmull_u8(bC, rgb.val[0]);
            auto resU8 = vshrn_n_u16(res, 6);
            vst1_u8(dest + 8 * i, resU8);
        }
        sta = countD8 * 8;
    }
#endif
    */
    for (int i = sta; i < count; ++i) {
        int r = source[4 * i + 2];
        int g = source[4 * i + 1];
        int b = source[4 * i + 0];

        int y = (19 * r + 38 * g + 7 * b) >> 6;

        dest[i] = y;
    }
}

void MNNRGBAToGRAY(const unsigned char* source, unsigned char* dest, size_t count) {
    int sta = 0;
    /*
#ifdef MNN_USE_NEON
    int countD8 = (int)count / 8;
    if (countD8 > 0) {
        auto rC = vdup_n_u8(19);
        auto gC = vdup_n_u8(38);
        auto bC = vdup_n_u8(7);
        for (int i = 0; i < countD8; ++i) {
            auto rgb   = vld4_u8(source + 32 * i);
            auto res   = vmull_u8(rC, rgb.val[0]) + vmull_u8(gC, rgb.val[1]) + vmull_u8(bC, rgb.val[2]);
            auto resU8 = vshrn_n_u16(res, 6);
            vst1_u8(dest + 8 * i, resU8);
        }
        sta = countD8 * 8;
    }
#endif
    */

    for (int i = sta; i < count; ++i) {
        int r = source[4 * i + 0];
        int g = source[4 * i + 1];
        int b = source[4 * i + 2];

        int y = (19 * r + 38 * g + 7 * b) >> 6;

        dest[i] = y;
    }
}

uint8_t saturate_cast(int v) { return (uint8_t)((unsigned)v <= 255 ? v : v > 0 ? 255 : 0); }
#define CV_DESCALE(x,n)     (((x) + (1 << ((n)-1))) >> (n))
#define CV_MUL_SHIFT(rC, gC, bC, n) vshrn_n_u16((vmull_u8(rC, rgb.val[0]) + vmull_u8(gC, rgb.val[1]) + vmull_u8(bC, rgb.val[2])), n)

void MNNC3ToYUV(const unsigned char* source, unsigned char* dest, size_t count, bool bgr, bool yuv) {
    static const int coeffs[] = {
        // Y
         4899,    9617,    1868,
        // Cr
         8192,   -6860,   -1332,
        // Cb
        -2765,   -5427,    8192,
        // U
        -2412,   -4734,    7146,
        // V
        10076,  -8438,   -1638
    };
    int r0 = 0, r1 = 3, r2 = 6,
        g0 = 1, g1 = 4, g2 = 7,
        b0 = 2, b1 = 5, b2 = 8;
    if (yuv) {
        r1 = 9,  r2 = 12;
        g1 = 10, g2 = 13;
        b1 = 11, b2 = 14;
    }
    if (bgr) {
        std::swap(r0, b0);
        std::swap(r1, b1);
        std::swap(r2, b2);
    }
    int C0 = coeffs[r0], C1 = coeffs[g0], C2 = coeffs[b0],
        C3 = coeffs[r1], C4 = coeffs[g1], C5 = coeffs[b1],
        C6 = coeffs[r2], C7 = coeffs[g2], C8 = coeffs[b2];
    int sta = 0;
    /*
#ifdef MNN_USE_NEON
    int countD8 = (int)count / 8;
    if (countD8 > 0) {
        auto rC0 = vdup_n_u8(C0), rC1 = vdup_n_u8(C1), rC2 = vdup_n_u8(C2),
             rC3 = vdup_n_u8(C3), rC4 = vdup_n_u8(C4), rC5 = vdup_n_u8(C5),
             rC6 = vdup_n_u8(C6), rC7 = vdup_n_u8(C7), rC8 = vdup_n_u8(C8);
        auto delta = vdup_n_u8(128);
        for (int i = 0; i < countD8; ++i) {
            auto rgb   = vld4_u8(source + 24 * i);
            uint8x8x3_t yuv;
            yuv.val[0] = CV_MUL_SHIFT(rC0, rC1, rC2, 14);
            yuv.val[1] = CV_MUL_SHIFT(rC3, rC4, rC5, 14);
            yuv.val[2] = CV_MUL_SHIFT(rC6, rC7, rC8, 14);
            yuv.val[1] = vadd_u8(yuv.val[1], delta);
            yuv.val[2] = vadd_u8(yuv.val[2], delta);
            vst3_u8(dest + 24 * i, yuv);
        }
        sta = countD8 * 8;
    }
#endif
     */
    for (int i = sta; i < count; ++i) {
        int r = source[3 * i + 0];
        int g = source[3 * i + 1];
        int b = source[3 * i + 2];
        int y = CV_DESCALE(r*C0 + g*C1 + b*C2, 14);
        int u = CV_DESCALE(r*C3 + g*C4 + b*C5, 14) + 128;
        int v = CV_DESCALE(r*C6 + g*C7 + b*C8, 14) + 128;
        dest[3 * i + 0] = y;
        dest[3 * i + 1] = u;
        dest[3 * i + 2] = v;
    }
}

void MNNC3ToXYZ(const unsigned char* source, unsigned char* dest, size_t count, bool bgr) {
    static const int coeffs[] = {
        1689,    1465,    739,
        871,     2929,    296,
        79,      488,     3892
    };
    int r0 = 0, r1 = 3, r2 = 6, b0 = 2, b1 = 5, b2 = 8;
    if (bgr) {
        std::swap(r0, b0);
        std::swap(r1, b1);
        std::swap(r2, b2);
    }
    int C0 = coeffs[r0], C1 = coeffs[1], C2 = coeffs[b0],
        C3 = coeffs[r1], C4 = coeffs[4], C5 = coeffs[b1],
        C6 = coeffs[r2], C7 = coeffs[7], C8 = coeffs[b2];
    int sta = 0;
    /*
#ifdef MNN_USE_NEON
    int countD8 = (int)count / 8;
    if (countD8 > 0) {
        auto rC0 = vdup_n_u8(C0), rC1 = vdup_n_u8(C1), rC2 = vdup_n_u8(C2),
             rC3 = vdup_n_u8(C3), rC4 = vdup_n_u8(C4), rC5 = vdup_n_u8(C5),
             rC6 = vdup_n_u8(C6), rC7 = vdup_n_u8(C7), rC8 = vdup_n_u8(C8);
        for (int i = 0; i < countD8; ++i) {
            auto rgb   = vld4_u8(source + 24 * i);
            uint8x8x3_t xyz;
            xyz.val[0] = CV_MUL_SHIFT(rC0, rC1, rC2, 12);
            xyz.val[1] = CV_MUL_SHIFT(rC3, rC4, rC5, 12);
            xyz.val[2] = CV_MUL_SHIFT(rC6, rC7, rC8, 12);
            vst3_u8(dest + 24 * i, xyz);
        }
        sta = countD8 * 8;
    }
#endif
    */
    for (int i = sta; i < count; ++i) {
        int r = source[3 * i + 0];
        int g = source[3 * i + 1];
        int b = source[3 * i + 2];

        int x = CV_DESCALE(r*C0 + g*C1 + b*C2, 12);
        int y = CV_DESCALE(r*C3 + g*C4 + b*C5, 12);
        int z = CV_DESCALE(r*C6 + g*C7 + b*C8, 12);

        dest[3 * i + 0] = saturate_cast(x);
        dest[3 * i + 1] = saturate_cast(y);
        dest[3 * i + 2] = saturate_cast(z);
    }
}

void MNNC3ToHSV(const unsigned char* source, unsigned char* dest, size_t count, bool bgr, bool full) {
    int hrange = full ? 256 : 180;
    int i = 0;
    for (; i < count; ++i) {
        int r = source[3 * i + 0];
        int g = source[3 * i + 1];
        int b = source[3 * i + 2];
        if (bgr) std::swap(r, b);
        int h, s, v = b, vmin = b, vr, vg;
        vmin = std::min({r, g, b});
        v = std::max({r, g, b});
        uint8_t diff = saturate_cast(v - vmin);
        vr = v == r ? -1 : 0;
        vg = v == g ? -1 : 0;
        s = (int(diff * (255 << 12) * (1.0f/(float)v)) + (1 << (11))) >> 12;
        h = (vr & (g - b)) + (~vr & ((vg & (b - r + 2 * diff)) + ((~vg) & (r - g + 4 * diff))));
        h = ((h * int((hrange << 12)/(6.f*diff) + 0.5)) + (1 << (11))) >> 12;
        h += h < 0 ? hrange : 0;

        dest[3 * i + 0] = saturate_cast(h);
        dest[3 * i + 1] = s;
        dest[3 * i + 2] = v;
    }
}

void MNNC3ToBGR555(const unsigned char* source, unsigned char* dest, size_t count, bool bgr) {
    int i = 0;
    for (; i < count; ++i) {
        int r = source[3 * i + 0];
        int g = source[3 * i + 1];
        int b = source[3 * i + 2];
        if (bgr) std::swap(r, b);
        reinterpret_cast<unsigned short*>(dest)[i] = (b >> 3)|((g & ~7) << 2)|((r & ~7) << 7);
    }
}

void MNNC3ToBGR565(const unsigned char* source, unsigned char* dest, size_t count, bool bgr) {
    int i = 0;
    for (; i < count; ++i) {
        int r = source[3 * i + 0];
        int g = source[3 * i + 1];
        int b = source[3 * i + 2];
        if (bgr) std::swap(r, b);
        reinterpret_cast<unsigned short*>(dest)[i] = (b >> 3)|((g&~3) << 3)|((r&~7) << 8);
    }
}

void MNNRGBToGRAY(const unsigned char* source, unsigned char* dest, size_t count) {
    int sta = 0;
#ifdef MNN_USE_NEON
    int countD8 = (int)count / 8;
    if (countD8 > 0) {
        auto rC = vdup_n_u8(19);
        auto gC = vdup_n_u8(38);
        auto bC = vdup_n_u8(7);
        for (int i = 0; i < countD8; ++i) {
            auto rgb   = vld3_u8(source + 24 * i);
            auto res   = vmull_u8(rC, rgb.val[0]) + vmull_u8(gC, rgb.val[1]) + vmull_u8(bC, rgb.val[2]);
            auto resU8 = vshrn_n_u16(res, 6);
            vst1_u8(dest + 8 * i, resU8);
        }
        sta = countD8 * 8;
    }
#endif
    for (int i = sta; i < count; ++i) {
        int r = source[3 * i + 0];
        int g = source[3 * i + 1];
        int b = source[3 * i + 2];

        int y = (19 * r + 38 * g + 7 * b) >> 6;
        // opencv impl: int y = (9798 * r + 19235 * g + 3735 * b + (1 << 14)) >> 15;

        dest[i] = y;
    }
}

void MNNBRGToGRAY(const unsigned char* source, unsigned char* dest, size_t count) {
    int sta = 0;
#ifdef MNN_USE_NEON
    int countD8 = (int)count / 8;
    if (countD8 > 0) {
        auto rC = vdup_n_u8(19);
        auto gC = vdup_n_u8(38);
        auto bC = vdup_n_u8(7);
        for (int i = 0; i < countD8; ++i) {
            auto rgb   = vld3_u8(source + 24 * i);
            auto res   = vmull_u8(rC, rgb.val[2]) + vmull_u8(gC, rgb.val[1]) + vmull_u8(bC, rgb.val[0]);
            auto resU8 = vshrn_n_u16(res, 6);
            vst1_u8(dest + 8 * i, resU8);
        }
        sta = countD8 * 8;
    }
#endif

    for (int i = sta; i < count; ++i) {
        int r = source[3 * i + 2];
        int g = source[3 * i + 1];
        int b = source[3 * i + 0];

        int y = (19 * r + 38 * g + 7 * b) >> 6;

        dest[i] = y;
    }
}

void MNNNV21ToRGBA(const unsigned char* source, unsigned char* dest, size_t count) {
    auto y   = source;
    auto uv  = source + count;
    auto dst = dest;
    int sta  = 0;
#ifdef MNN_USE_NEON
    const int unit   = 16;
    size_t countDiv8 = count / unit;
    if (countDiv8 > 0) {
        MNNNV21ToRGBAUnit(source, dest, countDiv8, uv);
        sta = (int)countDiv8 * unit;
    }
#endif
    for (int i = sta; i < count; ++i) {
        int Y = y[i];
        int U = (int)uv[(i / 2) * 2 + 1] - 128;
        int V = (int)uv[(i / 2) * 2 + 0] - 128;

        Y     = Y << 6;
        int R = (Y + 73 * V) >> 6;
        int G = (Y - 25 * U - 37 * V) >> 6;
        int B = (Y + 130 * U) >> 6;

        R = std::min(std::max(R, 0), 255);
        G = std::min(std::max(G, 0), 255);
        B = std::min(std::max(B, 0), 255);

        dst[4 * i + 0] = (uint8_t)R;
        dst[4 * i + 1] = (uint8_t)G;
        dst[4 * i + 2] = (uint8_t)B;
        dst[4 * i + 3] = 255;
    }
}

void MNNNV21ToRGB(const unsigned char* source, unsigned char* dest, size_t count) {
    auto y   = source;
    auto uv  = source + count;
    auto dst = dest;
    int sta  = 0;
#ifdef MNN_USE_NEON
    const int unit   = 16;
    size_t countDiv8 = count / unit;
    if (countDiv8 > 0) {
        MNNNV21ToRGBUnit(source, dest, countDiv8, uv);
        sta = (int)countDiv8 * unit;
    }
#endif
    for (int i = sta; i < count; ++i) {
        int Y = y[i];
        int U = (int)uv[(i / 2) * 2 + 1] - 128;
        int V = (int)uv[(i / 2) * 2 + 0] - 128;

/*
        OpenCV impl is as below:
        Y     = std::max(0, Y - 16) * 1220542;
        int R = (Y + (V * 1673527) + (1 << 19)) >> 20;
        int G = (Y + (-852492 * V + -409993 * U) + (1 << 19)) >> 20;
        int B = (Y + (2116026 * U) + (1 << 19)) >> 20;
*/
        Y     = Y << 6;
        int R = (Y + 73 * V) >> 6;
        int G = (Y - 25 * U - 37 * V) >> 6;
        int B = (Y + 130 * U) >> 6;

        R = std::min(std::max(R, 0), 255);
        G = std::min(std::max(G, 0), 255);
        B = std::min(std::max(B, 0), 255);

        dst[3 * i + 0] = (uint8_t)R;
        dst[3 * i + 1] = (uint8_t)G;
        dst[3 * i + 2] = (uint8_t)B;
    }
}

void MNNNV21ToBGRA(const unsigned char* source, unsigned char* dest, size_t count) {
    auto y   = source;
    auto uv  = source + count;
    auto dst = dest;
    int sta  = 0;
#ifdef MNN_USE_NEON
    const int unit   = 16;
    size_t countDiv8 = count / unit;
    if (countDiv8 > 0) {
        MNNNV21ToBGRAUnit(source, dest, countDiv8, uv);
        sta = (int)countDiv8 * unit;
    }
#endif
    for (int i = sta; i < count; ++i) {
        int Y = y[i];
        int U = (int)uv[(i / 2) * 2 + 1] - 128;
        int V = (int)uv[(i / 2) * 2 + 0] - 128;

        Y     = Y << 6;
        int R = (Y + 73 * V) >> 6;
        int G = (Y - 25 * U - 37 * V) >> 6;
        int B = (Y + 130 * U) >> 6;

        R = std::min(std::max(R, 0), 255);
        G = std::min(std::max(G, 0), 255);
        B = std::min(std::max(B, 0), 255);

        dst[4 * i + 0] = (uint8_t)B;
        dst[4 * i + 1] = (uint8_t)G;
        dst[4 * i + 2] = (uint8_t)R;
        dst[4 * i + 3] = 255;
    }
}

void MNNNV21ToBGR(const unsigned char* source, unsigned char* dest, size_t count) {
    auto y   = source;
    auto uv  = source + count;
    auto dst = dest;
    int sta  = 0;
#ifdef MNN_USE_NEON
    const int unit   = 16;
    size_t countDiv8 = count / unit;
    if (countDiv8 > 0) {
        MNNNV21ToBGRUnit(source, dest, countDiv8, uv);
        sta = (int)countDiv8 * unit;
    }
#endif
    for (int i = sta; i < count; ++i) {
        int Y = y[i];
        int U = (int)uv[(i / 2) * 2 + 1] - 128;
        int V = (int)uv[(i / 2) * 2 + 0] - 128;

        Y     = Y << 6;
        int R = (Y + 73 * V) >> 6;
        int G = (Y - 25 * U - 37 * V) >> 6;
        int B = (Y + 130 * U) >> 6;

        R = std::min(std::max(R, 0), 255);
        G = std::min(std::max(G, 0), 255);
        B = std::min(std::max(B, 0), 255);

        dst[3 * i + 0] = (uint8_t)B;
        dst[3 * i + 1] = (uint8_t)G;
        dst[3 * i + 2] = (uint8_t)R;
    }
}

void MNNC1ToFloatC1(const unsigned char* source, float* dest, const float* mean, const float* normal, size_t count) {
#ifdef MNN_USE_NEON
    unsigned long size  = count >> 4;
    float32x4_t cache   = vdupq_n_f32(0);
    float32x4_t _mean   = vdupq_n_f32(-mean[0]);
    float32x4_t _normal = vdupq_n_f32(normal[0]);
    for (int i = 0; i < size; i++, source += 16) {
        uint8x16_t v = vld1q_u8(source);
        int16x8_t vl = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v)));  // 0..7
        int16x8_t vh = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(v))); // 8..15
        // unpack to 32 bits
        float32x4_t vll = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vl))); // 0..3
        cache           = vaddq_f32(_mean, vll);
        cache           = vmulq_f32(cache, _normal);
        vst1q_f32(dest, cache);
        dest += 4;
        float32x4_t vlh = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vl))); // 4..7
        cache           = vaddq_f32(_mean, vlh);
        cache           = vmulq_f32(cache, _normal);
        vst1q_f32(dest, cache);
        dest += 4;
        float32x4_t vhl = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vh))); // 8..11
        cache           = vaddq_f32(_mean, vhl);
        cache           = vmulq_f32(cache, _normal);
        vst1q_f32(dest, cache);
        dest += 4;
        float32x4_t vhh = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vh))); // 12..15
        cache           = vaddq_f32(_mean, vhh);
        cache           = vmulq_f32(cache, _normal);
        vst1q_f32(dest, cache);
        dest += 4;
    }
    int left = count & 15;
    if (left == 0) {
        return;
    }
    for (int i = 0; i < left; ++i, ++dest, ++source) {
        *dest = normal[0] * (*source - mean[0]);
    }
#else
    for (int i = 0; i < count; ++i) {
        dest[i + 0] = normal[0] * (source[i + 0] - mean[0]);
    }
#endif
}

void MNNC3ToFloatC3(const unsigned char* source, float* dest, const float* mean, const float* normal,
                             size_t count) {
#ifdef MNN_USE_NEON
    int size              = (int)count / 16;
    float32x4x3_t cachell = {vmovq_n_f32(0), vmovq_n_f32(0), vmovq_n_f32(0)};
    float32x4x3_t cachelh = {vmovq_n_f32(0), vmovq_n_f32(0), vmovq_n_f32(0)};
    float32x4x3_t cachehl = {vmovq_n_f32(0), vmovq_n_f32(0), vmovq_n_f32(0)};
    float32x4x3_t cachehh = {vmovq_n_f32(0), vmovq_n_f32(0), vmovq_n_f32(0)};
    float32x4x3_t _mean;
    float32x4x3_t _normal;
    for (int c = 0; c < 3; c++) {
        _mean.val[c]   = vmovq_n_f32(-mean[c]);
        _normal.val[c] = vmovq_n_f32(normal[c]);
    }
    for (int i = 0; i < size; i++) {
        uint8x16x3_t v = vld3q_u8(source + 16 * 3 * i);
        int c          = 0;
        {
            int16x8_t vl = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v.val[c]))); // 0..7
            // unpack to 32 bits
            float32x4_t vll = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vl))); // 0..3
            cachell.val[c]  = vaddq_f32(_mean.val[c], vll);
            cachell.val[c]  = vmulq_f32(cachell.val[c], _normal.val[c]);

            float32x4_t vlh = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vl))); // 4..7
            cachelh.val[c]  = vaddq_f32(_mean.val[c], vlh);
            cachelh.val[c]  = vmulq_f32(cachelh.val[c], _normal.val[c]);

            int16x8_t vh = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(v.val[c]))); // 8..15
            // unpack to 32 bits
            float32x4_t vhl = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vh))); // 8..11
            cachehl.val[c]  = vaddq_f32(_mean.val[c], vhl);
            cachehl.val[c]  = vmulq_f32(cachehl.val[c], _normal.val[c]);

            float32x4_t vhh = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vh))); // 12..15
            cachehh.val[c]  = vaddq_f32(_mean.val[c], vhh);
            cachehh.val[c]  = vmulq_f32(cachehh.val[c], _normal.val[c]);
        }
        c = 1;
        {
            int16x8_t vl = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v.val[c]))); // 0..7
            // unpack to 32 bits
            float32x4_t vll = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vl))); // 0..3
            cachell.val[c]  = vaddq_f32(_mean.val[c], vll);
            cachell.val[c]  = vmulq_f32(cachell.val[c], _normal.val[c]);

            float32x4_t vlh = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vl))); // 4..7
            cachelh.val[c]  = vaddq_f32(_mean.val[c], vlh);
            cachelh.val[c]  = vmulq_f32(cachelh.val[c], _normal.val[c]);

            int16x8_t vh = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(v.val[c]))); // 8..15
            // unpack to 32 bits
            float32x4_t vhl = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vh))); // 8..11
            cachehl.val[c]  = vaddq_f32(_mean.val[c], vhl);
            cachehl.val[c]  = vmulq_f32(cachehl.val[c], _normal.val[c]);

            float32x4_t vhh = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vh))); // 12..15
            cachehh.val[c]  = vaddq_f32(_mean.val[c], vhh);
            cachehh.val[c]  = vmulq_f32(cachehh.val[c], _normal.val[c]);
        }
        c = 2;
        {
            int16x8_t vl = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v.val[c]))); // 0..7
            // unpack to 32 bits
            float32x4_t vll = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vl))); // 0..3
            cachell.val[c]  = vaddq_f32(_mean.val[c], vll);
            cachell.val[c]  = vmulq_f32(cachell.val[c], _normal.val[c]);

            float32x4_t vlh = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vl))); // 4..7
            cachelh.val[c]  = vaddq_f32(_mean.val[c], vlh);
            cachelh.val[c]  = vmulq_f32(cachelh.val[c], _normal.val[c]);

            int16x8_t vh = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(v.val[c]))); // 8..15
            // unpack to 32 bits
            float32x4_t vhl = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vh))); // 8..11
            cachehl.val[c]  = vaddq_f32(_mean.val[c], vhl);
            cachehl.val[c]  = vmulq_f32(cachehl.val[c], _normal.val[c]);

            float32x4_t vhh = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vh))); // 12..15
            cachehh.val[c]  = vaddq_f32(_mean.val[c], vhh);
            cachehh.val[c]  = vmulq_f32(cachehh.val[c], _normal.val[c]);
        }
        vst3q_f32(dest + 48 * i + 0 * 3, cachell);
        vst3q_f32(dest + 48 * i + 4 * 3, cachelh);
        vst3q_f32(dest + 48 * i + 8 * 3, cachehl);
        vst3q_f32(dest + 48 * i + 12 * 3, cachehh);
    }

    int remain = size * 16;
    for (int i = remain; i < count; i++) {
        dest[3 * i + 0] = normal[0] * (source[3 * i + 0] - mean[0]);
        dest[3 * i + 1] = normal[1] * (source[3 * i + 1] - mean[1]);
        dest[3 * i + 2] = normal[2] * (source[3 * i + 2] - mean[2]);
    }
#else
    for (int i = 0; i < count; ++i) {
        dest[3 * i + 0] = normal[0] * (source[3 * i + 0] - mean[0]);
        dest[3 * i + 1] = normal[1] * (source[3 * i + 1] - mean[1]);
        dest[3 * i + 2] = normal[2] * (source[3 * i + 2] - mean[2]);
    }
#endif
}

void MNNC4ToFloatC4(const unsigned char* source, float* dest, const float* mean, const float* normal, size_t count) {
    for (int i = 0; i < count; ++i) {
        dest[4 * i + 0] = normal[0] * (source[4 * i + 0] - mean[0]);
        dest[4 * i + 1] = normal[1] * (source[4 * i + 1] - mean[1]);
        dest[4 * i + 2] = normal[2] * (source[4 * i + 2] - mean[2]);
        dest[4 * i + 3] = normal[3] * (source[4 * i + 3] - mean[3]);
    }
}

void MNNC1ToFloatRGBA(const unsigned char* source, float* dest, const float* mean, const float* normal, size_t count) {
#ifdef MNN_USE_NEON
    MNNBlitC1ToFloatRGBA(source, dest, mean, normal, count);
#else
    // MNN_PRINT("normal = %f\n", normal[0]);
    ::memset(dest, 0, 4 * sizeof(float) * count);
    for (int i = 0; i < count; ++i) {
        dest[4 * i + 0] = normal[0] * (source[i + 0] - mean[0]);
    }
#endif
}

void MNNC3ToFloatRGBA(const unsigned char* source, float* dest, const float* mean, const float* normal, size_t count) {
#ifdef MNN_USE_NEON
    MNNBlitC3ToFloatRGBA(source, dest, mean, normal, count);
#else
    for (int i = 0; i < count; ++i) {
        dest[4 * i + 0] = normal[0] * (source[3 * i + 0] - mean[0]);
        dest[4 * i + 1] = normal[1] * (source[3 * i + 1] - mean[1]);
        dest[4 * i + 2] = normal[2] * (source[3 * i + 2] - mean[2]);
        dest[4 * i + 3] = 0.0f;
    }
#endif
}


static inline float __clamp(float v, float minV, float maxV) {
    return std::max(std::min(v, maxV), minV);
}

static void _sampleBilinearCommon(const unsigned char* source, unsigned char* dest, MNN::CV::Point* points, size_t count,
                                  size_t iw, size_t ih, size_t yStride, size_t bpp) {
    float dy   = points[1].fY;
    float dx   = points[1].fX;
    float xMax = iw - 1;
    float yMax = ih - 1;

    MNN::CV::Point curPoints;
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

void MNNSamplerC4Bilinear(const unsigned char* source, unsigned char* dest, MNN::CV::Point* points, size_t sta,
                          size_t count, size_t capacity, size_t iw, size_t ih, size_t yStride) {
#ifdef MNN_USE_NEON
    MNNSamplerC4BilinearOpt(source, dest + 4 * sta, reinterpret_cast<float*>(points), count, iw - 1, ih - 1, yStride);
#else
    _sampleBilinearCommon(source, dest + 4 * sta, points, count, iw, ih, yStride, 4);
#endif
}
void MNNSamplerC3Bilinear(const unsigned char* source, unsigned char* dest, MNN::CV::Point* points, size_t sta,
                          size_t count, size_t capacity, size_t iw, size_t ih, size_t yStride) {
    _sampleBilinearCommon(source, dest + 3 * sta, points, count, iw, ih, yStride, 3);
}
void MNNSamplerC1Bilinear(const unsigned char* source, unsigned char* dest, MNN::CV::Point* points, size_t sta,
                          size_t count, size_t capacity, size_t iw, size_t ih, size_t yStride) {
#ifdef MNN_USE_NEON
    MNNSamplerC1BilinearOpt(source, dest + sta, reinterpret_cast<float*>(points), count, iw - 1, ih - 1, yStride);
#else
    _sampleBilinearCommon(source, dest + sta, points, count, iw, ih, yStride, 1);
#endif
}
void MNNSamplerNearest(const unsigned char* source, unsigned char* dest, MNN::CV::Point* points, size_t sta, size_t count,
                       size_t iw, size_t ih, size_t yStride, int bpp) {
    dest = dest + bpp * sta;
    MNN::CV::Point curPoints;
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

void MNNSamplerC4Nearest(const unsigned char* source, unsigned char* dest, MNN::CV::Point* points, size_t sta,
                         size_t count, size_t capacity, size_t iw, size_t ih, size_t yStride) {
#ifdef MNN_USE_NEON
    MNNSamplerC4NearestOpt(source, dest + 4 * sta, (float*)points, count, iw - 1, ih - 1, yStride);
#else
    MNNSamplerNearest(source, dest, points, sta, count, iw, ih, yStride, 4);
#endif
}

void MNNSamplerC1Nearest(const unsigned char* source, unsigned char* dest, MNN::CV::Point* points, size_t sta,
                         size_t count, size_t capacity, size_t iw, size_t ih, size_t yStride) {
#ifdef MNN_USE_NEON
    MNNSamplerC1NearestOpt(source, dest + sta, (float*)points, count, iw - 1, ih - 1, yStride);
#else
    MNNSamplerNearest(source, dest, points, sta, count, iw, ih, yStride, 1);
#endif
}

void MNNSamplerC3Nearest(const unsigned char* source, unsigned char* dest, MNN::CV::Point* points, size_t sta,
                         size_t count, size_t capacity, size_t iw, size_t ih, size_t yStride) {
    MNNSamplerNearest(source, dest, points, sta, count, iw, ih, yStride, 3);
}

void MNNSamplerCopyCommon(const unsigned char* source, unsigned char* dest, MNN::CV::Point* points, size_t sta,
                          size_t count, size_t iw, size_t ih, size_t yStride, int bpp) {
    dest = dest + bpp * sta;
    MNN::CV::Point curPoints;
    curPoints.fX   = points[0].fX;
    curPoints.fY   = points[0].fY;
    float xMax     = iw - 1;
    float yMax     = ih - 1;
    int y          = (int)roundf(__clamp(curPoints.fY, 0, yMax));
    int x          = (int)roundf(__clamp(curPoints.fX, 0, xMax));
    auto sourcePos = y * yStride + bpp * x;
    ::memcpy(dest, source + sourcePos, bpp * count);
}

void MNNSamplerI420Copy(const unsigned char* source, unsigned char* dest, MNN::CV::Point* points, size_t sta,
                        size_t count, size_t capacity, size_t iw, size_t ih, size_t yStride) {
    MNN::CV::Point curPoints;
    curPoints.fX    = points[0].fX;
    curPoints.fY    = points[0].fY;
    float xMax      = iw - 1;
    float yMax      = ih - 1;
    int y           = (int)roundf(__clamp(curPoints.fY, 0, yMax));
    int x           = (int)roundf(__clamp(curPoints.fX, 0, xMax));
    auto uvPlane = (((int)iw + 1) / 2) * ((int(ih) + 1) / 2);
    int sourcePosY  = y * (int)iw + x;
    auto sourcePosU = source + (int)iw * (int)ih + (y / 2) * (((int)iw + 1) / 2) + (x / 2);
    auto sourcePosV = source + (int)iw * (int)ih + (y / 2) * (((int)iw + 1) / 2) + (x / 2) + uvPlane;
    auto uvCount = (count + 1) / 2;
    ::memcpy(dest + sta, source + sourcePosY, count);
    auto uDest = dest + (capacity) + (sta / 2) * 2;
    for (int i=0; i<uvCount; ++i) {
        uDest[2 * i + 0] = sourcePosV[i];
        uDest[2 * i + 1] = sourcePosU[i];
    }
}
void MNNSamplerI420Nearest(const unsigned char* source, unsigned char* dest, MNN::CV::Point* points, size_t sta,
                           size_t count, size_t capacity, size_t iw, size_t ih, size_t yStride) {
    auto srcY  = source;

    auto dstY  = dest + sta;
    auto dstUV = dest + (capacity) + (sta / 2) * 2;
    auto stride = yStride;
    if (yStride == 0) {
        stride = iw;
    }
    auto srcU = source + stride * ih;
    MNNSamplerC1Nearest(srcY, dstY, points, 0, count, capacity, iw, ih, stride);

    MNN::CV::Point uvPoints[2];
    uvPoints[0].fX = (points[0].fX - 0.01f) / 2.0f;
    uvPoints[0].fY = (points[0].fY - 0.01f) / 2.0f;
    uvPoints[1].fX = points[1].fX / 2.0f;
    uvPoints[1].fY = points[1].fY / 2.0f;
    if (yStride == 0) {
        stride =  ((iw + 1) / 2);
    }
    auto srcV = srcU + stride * ((ih + 1) / 2);
    auto uvCount = (count + 1) / 2;
    {
        MNN::CV::Point curPoints;
        curPoints.fX = uvPoints[0].fX;
        curPoints.fY = uvPoints[0].fY;
        float dy     = uvPoints[1].fY;
        float dx     = uvPoints[1].fX;
        float xMax   = ((iw + 1) / 2) - 1;
        float yMax   = ((ih + 1) / 2) - 1;

        for (int i = 0; i < uvCount; ++i) {
            int y = (int)roundf(__clamp(curPoints.fY, 0, yMax));
            int x = (int)roundf(__clamp(curPoints.fX, 0, xMax));
            curPoints.fY += dy;
            curPoints.fX += dx;
            auto offset = y * stride + x;
            dstUV[2 * i + 0] = srcV[offset];
            dstUV[2 * i + 1] = srcU[offset];
        }
    }
}

void MNNSamplerNV21Copy(const unsigned char* source, unsigned char* dest, MNN::CV::Point* points, size_t sta,
                        size_t count, size_t capacity, size_t iw, size_t ih, size_t yStride) {
    MNN::CV::Point curPoints;
    curPoints.fX    = points[0].fX;
    curPoints.fY    = points[0].fY;
    float xMax      = iw - 1;
    float yMax      = ih - 1;
    int y           = (int)roundf(__clamp(curPoints.fY, 0, yMax));
    int x           = (int)roundf(__clamp(curPoints.fX, 0, xMax));
    int stride = (int)yStride;
    int hstride = (int)yStride;
    if (yStride == 0) {
        stride = (int)iw;
        hstride = (((int)iw + 1) / 2) * 2;
    }

    int sourcePosY  = y * stride + x;
    int sourcePosUV = (int)stride * (int)ih + (y / 2) * hstride + (x / 2) * 2;

    ::memcpy(dest + sta, source + sourcePosY, count);
    ::memcpy(dest + (capacity) + (sta / 2) * 2, source + sourcePosUV, ((count + 1) / 2) * 2);
}

void MNNSamplerNV21Nearest(const unsigned char* source, unsigned char* dest, MNN::CV::Point* points, size_t sta,
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

    MNN::CV::Point uvPoints[2];
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

void MNNSamplerNV12Copy(const unsigned char* source, unsigned char* dest, MNN::CV::Point* points, size_t sta,
                        size_t count, size_t capacity, size_t iw, size_t ih, size_t yStride) {
    MNNSamplerNV21Copy(source, dest, points, sta, count, capacity, iw, ih, yStride);
    auto destUV = dest + (capacity) + (sta / 2) * 2;
    auto countC2 = ((count + 1) / 2);
    _swapUV(destUV, destUV, countC2);
}

void MNNSamplerNV12Nearest(const unsigned char* source, unsigned char* dest, MNN::CV::Point* points, size_t sta,
                           size_t count, size_t capacity, size_t iw, size_t ih, size_t yStride) {
    MNNSamplerNV21Nearest(source, dest, points, sta, count, capacity, iw, ih, yStride);
    auto destUV = dest + (capacity) + (sta / 2) * 2;
    auto countC2 = ((count + 1) / 2);
    _swapUV(destUV, destUV, countC2);
}

void MNNC3blitH(const unsigned char* source, unsigned char* dest, size_t count) {
    for (int i = 0; i < count; i++) {
        memcpy(dest + 3 * i, source, 3);
    }
}

void MNNC4blitH(const unsigned char* source, unsigned char* dest, size_t count) {
    for (int i = 0; i < count; i++) {
        memcpy(dest + 4 * i, source, 4);
    }
}

void MNNC1blitH(const unsigned char* source, unsigned char* dest, size_t count) {
    for (int i = 0; i < count; i++) {
        memcpy(dest + i, source, 1);
    }
}
