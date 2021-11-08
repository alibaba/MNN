//
//  ImageBlitter.cpp
//  MNN
//
//  Created by MNN on 2018/12/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//

/** x86 opt ref to https://skia.googlesource.com/skia/src/opts/SkSwizzler_opts.h */
#include "cv/ImageBlitter.hpp"
#include <string.h>
#include <mutex>
#include "core/Macro.h"
#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif
#ifdef MNN_USE_SSE
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#endif
#include <map>
extern "C" {
void MNNNV21ToRGBUnit(const unsigned char* source, unsigned char* dest, size_t countDiv8, const unsigned char* uv);
void MNNNV21ToBGRUnit(const unsigned char* source, unsigned char* dest, size_t countDiv8, const unsigned char* uv);
void MNNNV21ToRGBAUnit(const unsigned char* source, unsigned char* dest, size_t countDiv8, const unsigned char* uv);
void MNNNV21ToBGRAUnit(const unsigned char* source, unsigned char* dest, size_t countDiv8, const unsigned char* uv);
}

namespace MNN {
namespace CV {

static void _gray2C4(const unsigned char* source, unsigned char* dest, size_t count) {
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

static void _gray2C3(const unsigned char* source, unsigned char* dest, size_t count) {
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

static void _copyC1(const unsigned char* source, unsigned char* dest, size_t count) {
    ::memcpy(dest, source, count * sizeof(unsigned char));
}

static void _copyC4(const unsigned char* source, unsigned char* dest, size_t count) {
    ::memcpy(dest, source, 4 * count * sizeof(unsigned char));
}

static void _copyC3(const unsigned char* source, unsigned char* dest, size_t count) {
    ::memcpy(dest, source, 3 * count * sizeof(unsigned char));
}

static void _rgba2bgra(const unsigned char* source, unsigned char* dest, size_t count) {
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
#ifdef MNN_USE_SSE
    int countD8 = (int)count / 4;
    const auto swapRB = _mm_setr_epi8(2,1,0,3, 6,5,4,7, 10,9,8,11, 14,13,12,15);
    if (countD8 > 0) {
        for (int i = 0; i < countD8; ++i) {
            auto rgba = _mm_loadu_si128((const __m128i*)(source + 16 * i));
            auto bgra = _mm_shuffle_epi8(rgba, swapRB);
            _mm_storeu_si128((__m128i*)(dest + 16 * i), bgra);
        }
        sta = countD8 * 4;
    }
#endif
    for (int i = sta; i < count; ++i) {
        dest[4 * i + 0] = source[4 * i + 2];
        dest[4 * i + 1] = source[4 * i + 1];
        dest[4 * i + 2] = source[4 * i + 0];
        dest[4 * i + 3] = source[4 * i + 3];
    }
}
static void _rgba2bgr(const unsigned char* source, unsigned char* dest, size_t count) {
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
static void _rgb2bgr(const unsigned char* source, unsigned char* dest, size_t count) {
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
static void _bgra2bgr(const unsigned char* source, unsigned char* dest, size_t count) {
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

static void _bgra2gray(const unsigned char* source, unsigned char* dest, size_t count) {
    int sta = 0;
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
    for (int i = sta; i < count; ++i) {
        int r = source[4 * i + 2];
        int g = source[4 * i + 1];
        int b = source[4 * i + 0];

        int y = (19 * r + 38 * g + 7 * b) >> 6;

        dest[i] = y;
    }
}

static void _rgba2gray(const unsigned char* source, unsigned char* dest, size_t count) {
    int sta = 0;
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
    for (int i = sta; i < count; ++i) {
        int r = source[4 * i + 0];
        int g = source[4 * i + 1];
        int b = source[4 * i + 2];

        int y = (19 * r + 38 * g + 7 * b) >> 6;

        dest[i] = y;
    }
}

static void _rgb2gray(const unsigned char* source, unsigned char* dest, size_t count) {
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

        dest[i] = y;
    }
}

static void _bgr2gray(const unsigned char* source, unsigned char* dest, size_t count) {
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
#ifdef MNN_USE_SSE
#define MNN_SSE_YUV_INIT \
countUnit -= 1;\
const auto c_6 = _mm_set1_epi16((1 << 6));\
const auto c_10 = _mm_set1_epi16((1 << 10));\
const auto c_73 = _mm_set1_epi16(73);\
const auto c_25 = _mm_set1_epi16(25);\
const auto c_37 = _mm_set1_epi16(37);\
const auto c_130 = _mm_set1_epi16(130);\
const auto c_128 = _mm_set1_epi16(128);\
const auto zero = _mm_set1_epi8(0);\
const auto alpha = _mm_set1_epi8(-1);\
const auto crossMask = _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);\
const auto revertCrossMask = _mm_setr_epi8(0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15);\

#define MNN_SSE_YUV_CONVERT \
auto Y_ = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i*)(y + z * 16)), crossMask);\
auto UV = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i*)(uv + z * 16)), crossMask);\
auto y0 = _mm_mullo_epi16(_mm_unpacklo_epi8(Y_, zero), c_6);\
auto y1 = _mm_mullo_epi16(_mm_unpackhi_epi8(Y_, zero), c_6);\
auto U_ = _mm_sub_epi16(_mm_unpackhi_epi8(UV, zero), c_128);\
auto V_ = _mm_sub_epi16(_mm_unpacklo_epi8(UV, zero), c_128);\
auto r0 = _mm_add_epi16(y0, _mm_mullo_epi16(V_, c_73));\
auto r1 = _mm_add_epi16(y1, _mm_mullo_epi16(V_, c_73));\
auto g0 = _mm_sub_epi16(_mm_sub_epi16(y0, _mm_mullo_epi16(U_, c_25)), _mm_mullo_epi16(V_, c_37));\
auto g1 = _mm_sub_epi16(_mm_sub_epi16(y1, _mm_mullo_epi16(U_, c_25)), _mm_mullo_epi16(V_, c_37));\
auto b0 = _mm_add_epi16(y0, _mm_mullo_epi16(U_, c_130));\
auto b1 = _mm_add_epi16(y1, _mm_mullo_epi16(U_, c_130));\
r0 = _mm_mulhi_epi16(r0, c_10);\
r1 = _mm_mulhi_epi16(r1, c_10);\
g0 = _mm_mulhi_epi16(g0, c_10);\
g1 = _mm_mulhi_epi16(g1, c_10);\
b0 = _mm_mulhi_epi16(b0, c_10);\
b1 = _mm_mulhi_epi16(b1, c_10);\
auto dR = _mm_packus_epi16(r0, r1);\
auto dG = _mm_packus_epi16(g0, g1);\
auto dB = _mm_packus_epi16(b0, b1);\
dR = _mm_shuffle_epi8(dR, revertCrossMask);\
dG = _mm_shuffle_epi8(dG, revertCrossMask);\
dB = _mm_shuffle_epi8(dB, revertCrossMask);\
auto RG0 = _mm_unpacklo_epi8(dR, dG);\
auto RG1 = _mm_unpackhi_epi8(dR, dG);\
auto BA0 = _mm_unpacklo_epi8(dB, alpha);\
auto BA1 = _mm_unpackhi_epi8(dB, alpha);\
auto RGBA0 = _mm_unpacklo_epi16(RG0, BA0);\
auto RGBA1 = _mm_unpackhi_epi16(RG0, BA0);\
auto RGBA2 = _mm_unpacklo_epi16(RG1, BA1);\
auto RGBA3 = _mm_unpackhi_epi16(RG1, BA1);\

#endif

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
#ifdef MNN_USE_SSE
    const int unit   = 16;
    size_t countUnit = count / unit;
    if (countUnit > 0) {
        MNN_SSE_YUV_INIT;
        for (int z=0; z<countUnit; ++z) {
            MNN_SSE_YUV_CONVERT;

            // RGBA -> RGB
            _mm_storeu_si128((__m128i*)(dst + 64 * z + 16 * 0), RGBA0);
            _mm_storeu_si128((__m128i*)(dst + 64 * z + 16 * 1), RGBA1);
            _mm_storeu_si128((__m128i*)(dst + 64 * z + 16 * 2), RGBA2);
            _mm_storeu_si128((__m128i*)(dst + 64 * z + 16 * 3), RGBA3);
        }
        sta = (int)countUnit * unit;
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
#ifdef MNN_USE_SSE
    const int unit   = 16;
    size_t countUnit = count / unit;
    if (countUnit > 1) {
        countUnit -= 1;
        MNN_SSE_YUV_INIT;
        const auto rgbSelect = _mm_setr_epi8(0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, -1, -1, -1, -1);
        for (int z=0; z<countUnit; ++z) {
            MNN_SSE_YUV_CONVERT;

            // RGBA -> RGB
            _mm_storeu_si128((__m128i*)(dst + 48 * z + 12 * 0), _mm_shuffle_epi8(RGBA0, rgbSelect));
            _mm_storeu_si128((__m128i*)(dst + 48 * z + 12 * 1), _mm_shuffle_epi8(RGBA1, rgbSelect));
            _mm_storeu_si128((__m128i*)(dst + 48 * z + 12 * 2), _mm_shuffle_epi8(RGBA2, rgbSelect));
            _mm_storeu_si128((__m128i*)(dst + 48 * z + 12 * 3), _mm_shuffle_epi8(RGBA3, rgbSelect));
        }
        sta = (int)countUnit * unit;
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
#ifdef MNN_USE_SSE
    const int unit   = 16;
    size_t countUnit = count / unit;
    if (countUnit > 0) {
        MNN_SSE_YUV_INIT;
        const auto rgbaSelect = _mm_setr_epi8(2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15);
        for (int z=0; z<countUnit; ++z) {
            MNN_SSE_YUV_CONVERT;

            // RGBA -> RGB
            _mm_storeu_si128((__m128i*)(dst + 64 * z + 16 * 0), _mm_shuffle_epi8(RGBA0, rgbaSelect));
            _mm_storeu_si128((__m128i*)(dst + 64 * z + 16 * 1), _mm_shuffle_epi8(RGBA1, rgbaSelect));
            _mm_storeu_si128((__m128i*)(dst + 64 * z + 16 * 2), _mm_shuffle_epi8(RGBA2, rgbaSelect));
            _mm_storeu_si128((__m128i*)(dst + 64 * z + 16 * 3), _mm_shuffle_epi8(RGBA3, rgbaSelect));
        }
        sta = (int)countUnit * unit;
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
#ifdef MNN_USE_SSE
    const int unit   = 16;
    size_t countUnit = count / unit;
    if (countUnit > 1) {
        countUnit -= 1;
        MNN_SSE_YUV_INIT;
        const auto rgbSelect = _mm_setr_epi8(2, 1, 0, 6, 5, 4, 10, 9, 8, 14, 13, 12, -1, -1, -1, -1);
        for (int z=0; z<countUnit; ++z) {
            MNN_SSE_YUV_CONVERT;

            // RGBA -> RGB
            _mm_storeu_si128((__m128i*)(dst + 48 * z + 12 * 0), _mm_shuffle_epi8(RGBA0, rgbSelect));
            _mm_storeu_si128((__m128i*)(dst + 48 * z + 12 * 1), _mm_shuffle_epi8(RGBA1, rgbSelect));
            _mm_storeu_si128((__m128i*)(dst + 48 * z + 12 * 2), _mm_shuffle_epi8(RGBA2, rgbSelect));
            _mm_storeu_si128((__m128i*)(dst + 48 * z + 12 * 3), _mm_shuffle_epi8(RGBA3, rgbSelect));
        }
        sta = (int)countUnit * unit;
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

#define CHECKFORMAT(src, dst, func)\
if (source == src && dest == dst) return func

ImageBlitter::BLITTER ImageBlitter::choose(ImageFormat source, ImageFormat dest) {
    // YUV only different in sampler
    if (source == YUV_NV12) {
        source = YUV_NV21;
    }
    if (source == YUV_I420) {
        source = YUV_NV21;
    }
    CHECKFORMAT(RGBA, RGBA, _copyC4);
    CHECKFORMAT(RGBA, BGRA, _rgba2bgra);
    CHECKFORMAT(RGBA, BGR, _rgba2bgr);
    CHECKFORMAT(RGBA, RGB, _bgra2bgr);
    CHECKFORMAT(RGBA, GRAY, _rgba2gray);
    CHECKFORMAT(BGRA, RGBA, _rgba2bgra);
    CHECKFORMAT(BGRA, BGRA, _copyC4);
    CHECKFORMAT(BGRA, BGR, _bgra2bgr);
    CHECKFORMAT(BGRA, RGB, _rgba2bgr);
    CHECKFORMAT(BGRA, GRAY, _bgra2gray);
    CHECKFORMAT(RGB, RGB, _copyC3);
    CHECKFORMAT(RGB, BGR, _rgb2bgr);
    CHECKFORMAT(RGB, GRAY, _rgb2gray);

    CHECKFORMAT(BGR, BGR, _copyC3);
    CHECKFORMAT(BGR, RGB, _rgb2bgr);
    CHECKFORMAT(BGR, GRAY, _bgr2gray);

    CHECKFORMAT(GRAY, RGBA, _gray2C4);
    CHECKFORMAT(GRAY, BGRA, _gray2C4);
    CHECKFORMAT(GRAY, BGR, _gray2C3);
    CHECKFORMAT(GRAY, RGB, _gray2C3);
    CHECKFORMAT(GRAY, GRAY, _copyC1);

    CHECKFORMAT(YUV_NV21, GRAY, _copyC1);
    CHECKFORMAT(YUV_NV21, RGB, MNNNV21ToRGB);
    CHECKFORMAT(YUV_NV21, BGR, MNNNV21ToBGR);
    CHECKFORMAT(YUV_NV21, RGBA, MNNNV21ToRGBA);
    CHECKFORMAT(YUV_NV21, BGRA, MNNNV21ToBGRA);

    return nullptr;
}
} // namespace CV
} // namespace MNN
