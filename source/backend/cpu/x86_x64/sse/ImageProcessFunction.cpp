//
//  ImageProcessFunction.cpp
//  MNN
//
//  Created by MNN on 2021/11/01.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <algorithm>
#include "FunctionSummary.hpp"
#include "core/Macro.h"
#include "backend/cpu/x86_x64/cpu_id.h"
#include <MNN/ImageProcess.hpp>

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

static inline float __clamp(float v, float minV, float maxV) {
    return std::max(std::min(v, maxV), minV);
}

void _SSE_MNNRGBAToBGRA(const unsigned char* source, unsigned char* dest, size_t count) {
    int sta = 0;
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
    for (int i = sta; i < count; ++i) {
        dest[4 * i + 0] = source[4 * i + 2];
        dest[4 * i + 1] = source[4 * i + 1];
        dest[4 * i + 2] = source[4 * i + 0];
        dest[4 * i + 3] = source[4 * i + 3];
    }
}

void _SSE_MNNNV21ToRGBA(const unsigned char* source, unsigned char* dest, size_t count) {
    auto y   = source;
    auto uv  = source + count;
    auto dst = dest;
    int sta  = 0;
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

void _SSE_MNNNV21ToRGB(const unsigned char* source, unsigned char* dest, size_t count) {
    auto y   = source;
    auto uv  = source + count;
    auto dst = dest;
    int sta  = 0;
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

void _SSE_MNNNV21ToBGRA(const unsigned char* source, unsigned char* dest, size_t count) {
    auto y   = source;
    auto uv  = source + count;
    auto dst = dest;
    int sta  = 0;
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

void _SSE_MNNNV21ToBGR(const unsigned char* source, unsigned char* dest, size_t count) {
    auto y   = source;
    auto uv  = source + count;
    auto dst = dest;
    int sta  = 0;
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

// require SSE 4.1
void _SSE_MNNC1ToFloatC1(const unsigned char* source, float* dest, const float* mean, const float* normal, size_t count) {
    int remain = 0;
    int countC16 = count / 16;
    remain = countC16 * 16;
    const auto meanC4 = _mm_set1_ps(mean[0]);
    const auto normalC4 = _mm_set1_ps(normal[0]);
    const __m128i l1 = _mm_setr_epi8(4,5,6,7, 6,5,4,7, 10,9,8,11, 14,13,12,15);
    const __m128i l2 = _mm_setr_epi8(8,9,10,11, 6,5,4,7, 10,9,8,11, 14,13,12,15);
    const __m128i l3 = _mm_setr_epi8(12,13,14,15, 6,5,4,7, 10,9,8,11, 14,13,12,15);

    for (int i=0; i<countC16; ++i) {
        auto srcInt8 = _mm_loadu_si128((const __m128i*)(source + i * 16));
        auto int3200 = _mm_cvtepu8_epi32(srcInt8);
        auto int3201 = _mm_cvtepu8_epi32(_mm_shuffle_epi8(srcInt8, l1));
        auto int3210 = _mm_cvtepu8_epi32(_mm_shuffle_epi8(srcInt8, l2));
        auto int3211 = _mm_cvtepu8_epi32(_mm_shuffle_epi8(srcInt8, l3));
        auto float00 = _mm_cvtepi32_ps(int3200);
        auto float01 = _mm_cvtepi32_ps(int3201);
        auto float10 = _mm_cvtepi32_ps(int3210);
        auto float11 = _mm_cvtepi32_ps(int3211);
        _mm_storeu_ps(dest + 16 * i + 4 * 0, _mm_mul_ps(_mm_sub_ps(float00, meanC4), normalC4));
        _mm_storeu_ps(dest + 16 * i + 4 * 1, _mm_mul_ps(_mm_sub_ps(float01, meanC4), normalC4));
        _mm_storeu_ps(dest + 16 * i + 4 * 2, _mm_mul_ps(_mm_sub_ps(float10, meanC4), normalC4));
        _mm_storeu_ps(dest + 16 * i + 4 * 3, _mm_mul_ps(_mm_sub_ps(float11, meanC4), normalC4));
    }
    for (int i = remain; i < count; ++i) {
        dest[i + 0] = normal[0] * (source[i + 0] - mean[0]);
    }
}

// require SSE 4.1
void _SSE_MNNC3ToFloatC3(const unsigned char* source, float* dest, const float* mean, const float* normal,
                             size_t count) {
    int remain = 0;
    int countC4 = count / 4;
    if (countC4 > 1) {
        if ((count % 4) * 3 < 4) {
            //Avoid load extra memory
            countC4 -=1;
        }
        // RGBRGBRGBRGB -> RGBR , GBRG, BRGB
        auto alpha0 = _mm_setr_ps(normal[0], normal[1], normal[2], normal[0]);
        auto alpha1 = _mm_setr_ps(normal[1], normal[2], normal[0], normal[1]);
        auto alpha2 = _mm_setr_ps(normal[2], normal[0], normal[1], normal[2]);
        auto beta0 = _mm_setr_ps(mean[0], mean[1], mean[2], mean[0]);
        auto beta1 = _mm_setr_ps(mean[1], mean[2], mean[0], mean[1]);
        auto beta2 = _mm_setr_ps(mean[2], mean[0], mean[1], mean[2]);
        remain = countC4 * 4;
        const __m128i gM = _mm_setr_epi8(4, 5, 6, 7, 6,5,4,7, 10,9,8,11, 14,13,12,15);
        const __m128i bM = _mm_setr_epi8(8, 9,10,11, 6,5,4,7, 10,9,8,11, 14,13,12,15);

        for (int i = 0; i < countC4; ++i) {
            auto sInt8 = _mm_loadu_si128((const __m128i*)(source + 12 * i));
            auto s0 = _mm_cvtepu8_epi32(sInt8);
            auto s1 = _mm_cvtepu8_epi32(_mm_shuffle_epi8(sInt8, gM));
            auto s2 = _mm_cvtepu8_epi32(_mm_shuffle_epi8(sInt8, bM));
            
            auto f0 = _mm_cvtepi32_ps(s0);
            auto f1 = _mm_cvtepi32_ps(s1);
            auto f2 = _mm_cvtepi32_ps(s2);
            _mm_storeu_ps(dest + 12 * i + 4 * 0, _mm_mul_ps(_mm_sub_ps(f0, beta0), alpha0));
            _mm_storeu_ps(dest + 12 * i + 4 * 1, _mm_mul_ps(_mm_sub_ps(f1, beta1), alpha1));
            _mm_storeu_ps(dest + 12 * i + 4 * 2, _mm_mul_ps(_mm_sub_ps(f2, beta2), alpha2));
        }
    }
    for (int i = remain; i < count; ++i) {
        dest[3 * i + 0] = normal[0] * (source[3 * i + 0] - mean[0]);
        dest[3 * i + 1] = normal[1] * (source[3 * i + 1] - mean[1]);
        dest[3 * i + 2] = normal[2] * (source[3 * i + 2] - mean[2]);
    }
}

// require SSE 4.1
void _SSE_MNNC1ToFloatRGBA(const unsigned char* source, float* dest, const float* mean, const float* normal,
                          size_t count) {
    ::memset(dest, 0, 4 * sizeof(float) * count);
    int remain = 0;
    int countC16 = count / 16;
    remain = countC16 * 16;
    if (countC16 > 0) {
        const __m128i gM = _mm_setr_epi8(4, 5, 6, 7, 6,5,4,7, 10,9,8,11, 14,13,12,15);
        const __m128i bM = _mm_setr_epi8(8, 9,10,11, 6,5,4,7, 10,9,8,11, 14,13,12,15);
        const __m128i aM = _mm_setr_epi8(12,13,14,15, 6,5,4,7, 10,9,8,11, 14,13,12,15);
        auto normalC4 = _mm_set1_ps(normal[0]);
        auto meanC4 = _mm_set1_ps(mean[0]);

        for (int i=0; i<countC16; ++i) {
            auto sInt8 = _mm_loadu_si128((const __m128i*)(source + 16 * i));
            auto s0 = _mm_cvtepu8_epi32(sInt8);
            auto s1 = _mm_cvtepu8_epi32(_mm_shuffle_epi8(sInt8, gM));
            auto s2 = _mm_cvtepu8_epi32(_mm_shuffle_epi8(sInt8, bM));
            auto s3 = _mm_cvtepu8_epi32(_mm_shuffle_epi8(sInt8, aM));
            auto float00 = _mm_cvtepi32_ps(s0);
            auto float01 = _mm_cvtepi32_ps(s1);
            auto float10 = _mm_cvtepi32_ps(s2);
            auto float11 = _mm_cvtepi32_ps(s3);
            auto f0 = _mm_mul_ps(_mm_sub_ps(float00, meanC4), normalC4);
            auto f1 = _mm_mul_ps(_mm_sub_ps(float01, meanC4), normalC4);
            auto f2 = _mm_mul_ps(_mm_sub_ps(float10, meanC4), normalC4);
            auto f3 = _mm_mul_ps(_mm_sub_ps(float11, meanC4), normalC4);
            auto r1 = _mm_set1_ps(0.0f);
            auto r2 = _mm_set1_ps(0.0f);
            auto r3 = _mm_set1_ps(0.0f);

            auto curDst = dest +  4 * i * 16;
            
            _MM_TRANSPOSE4_PS(f0, r1, r2, r3);
            _mm_storeu_ps(curDst + 4 * 0, f0);
            _mm_storeu_ps(curDst + 4 * 1, r1);
            _mm_storeu_ps(curDst + 4 * 2, r2);
            _mm_storeu_ps(curDst + 4 * 3, r3);
            curDst += 16;

            _MM_TRANSPOSE4_PS(f1, r1, r2, r3);
            _mm_storeu_ps(curDst + 4 * 0, f1);
            _mm_storeu_ps(curDst + 4 * 1, r1);
            _mm_storeu_ps(curDst + 4 * 2, r2);
            _mm_storeu_ps(curDst + 4 * 3, r3);
            curDst += 16;

            _MM_TRANSPOSE4_PS(f2, r1, r2, r3);
            _mm_storeu_ps(curDst + 4 * 0, f2);
            _mm_storeu_ps(curDst + 4 * 1, r1);
            _mm_storeu_ps(curDst + 4 * 2, r2);
            _mm_storeu_ps(curDst + 4 * 3, r3);
            curDst += 16;

            _MM_TRANSPOSE4_PS(f3, r1, r2, r3);
            _mm_storeu_ps(curDst + 4 * 0, f3);
            _mm_storeu_ps(curDst + 4 * 1, r1);
            _mm_storeu_ps(curDst + 4 * 2, r2);
            _mm_storeu_ps(curDst + 4 * 3, r3);
        }
    }
    for (int i = remain; i < count; ++i) {
        dest[4 * i + 0] = normal[0] * (source[i + 0] - mean[0]);
    }
}

// require SSE 4.1
void _SSE_MNNC3ToFloatRGBA(const unsigned char* source, float* dest, const float* mean, const float* normal, size_t count) {
    int remain = 0;
    int countC4 = count / 4;
    if (countC4 > 1) {
        if ((count % 4) * 3 < 4) {
            //Avoid load extra memory
            countC4 -=1;
        }
        // RGBRGBRGBRGB -> RGB0 , RGB0, RGB0
        auto alpha0 = _mm_setr_ps(normal[0], normal[1], normal[2], 0.0f);
        auto beta0 = _mm_setr_ps(mean[0], mean[1], mean[2], 0.0f);
        remain = countC4 * 4;
        const __m128i rM = _mm_setr_epi8(0, 1, 2, 0, 6,5,4,7, 10,9,8,11, 14,13,12,15);
        const __m128i gM = _mm_setr_epi8(3, 4, 5, 3, 6,5,4,7, 10,9,8,11, 14,13,12,15);
        const __m128i bM = _mm_setr_epi8(6, 7, 8, 6, 6,5,4,7, 10,9,8,11, 14,13,12,15);
        const __m128i aM = _mm_setr_epi8(9, 10, 11, 9, 6,5,4,7, 10,9,8,11, 14,13,12,15);

        for (int i = 0; i < countC4; ++i) {
            auto sInt8 = _mm_loadu_si128((const __m128i*)(source + 12 * i));
            auto s0 = _mm_cvtepu8_epi32(_mm_shuffle_epi8(sInt8, rM));
            auto s1 = _mm_cvtepu8_epi32(_mm_shuffle_epi8(sInt8, gM));
            auto s2 = _mm_cvtepu8_epi32(_mm_shuffle_epi8(sInt8, bM));
            auto s3 = _mm_cvtepu8_epi32(_mm_shuffle_epi8(sInt8, aM));

            auto f0 = _mm_cvtepi32_ps(s0);
            auto f1 = _mm_cvtepi32_ps(s1);
            auto f2 = _mm_cvtepi32_ps(s2);
            auto f3 = _mm_cvtepi32_ps(s3);
            _mm_storeu_ps(dest + 16 * i + 4 * 0, _mm_mul_ps(_mm_sub_ps(f0, beta0), alpha0));
            _mm_storeu_ps(dest + 16 * i + 4 * 1, _mm_mul_ps(_mm_sub_ps(f1, beta0), alpha0));
            _mm_storeu_ps(dest + 16 * i + 4 * 2, _mm_mul_ps(_mm_sub_ps(f2, beta0), alpha0));
            _mm_storeu_ps(dest + 16 * i + 4 * 3, _mm_mul_ps(_mm_sub_ps(f3, beta0), alpha0));
        }
    }
    for (int i = remain; i < count; ++i) {
        dest[4 * i + 0] = normal[0] * (source[3 * i + 0] - mean[0]);
        dest[4 * i + 1] = normal[1] * (source[3 * i + 1] - mean[1]);
        dest[4 * i + 2] = normal[2] * (source[3 * i + 2] - mean[2]);
        dest[4 * i + 3] = 0.0f;
    }
}

// SSE 4.1
void _SSE_MNNSamplerNearest(const unsigned char* source, unsigned char* dest, MNN::CV::Point* points, size_t sta, size_t count,
                       size_t iw, size_t ih, size_t yStride, int bpp) {
    dest = dest + bpp * sta;
    MNN::CV::Point curPoints;
    curPoints.fX = points[0].fX;
    curPoints.fY = points[0].fY;
    float dy     = points[1].fY;
    float dx     = points[1].fX;
    float xMax   = iw - 1;
    float yMax   = ih - 1;
    int start = 0;
    int sizedQuad = count / 4;


    if (sizedQuad > 0 && bpp == 4) {
        auto yStride4 = _mm_set1_epi32(yStride);
        auto varBpp     = _mm_set1_epi32(bpp);
        auto varZero = _mm_set1_ps(0.f);
        // for roundf.
        auto zeroInt    = _mm_set1_epi32(0);
        __m128 plus = _mm_set1_ps(0.5f);
        __m128 minus = _mm_set1_ps(-0.5f);

        auto xmax4    = _mm_set1_ps(xMax);
        auto ymax4    = _mm_set1_ps(yMax);
        for (int i = 0; i < sizedQuad; ++i) {
            auto cury4 = _mm_set_ps(curPoints.fY + 3 * dy, curPoints.fY + 2 * dy, curPoints.fY + dy, curPoints.fY);
            auto curx4 = _mm_set_ps(curPoints.fX + 3 * dx, curPoints.fX + 2 * dx, curPoints.fX + dx, curPoints.fX);
            cury4 = _mm_max_ps(cury4, varZero);
            curx4 = _mm_max_ps(curx4, varZero);
            cury4 = _mm_min_ps(cury4, ymax4);
            curx4 = _mm_min_ps(curx4, xmax4);
            
            auto x0 = _mm_cmplt_ps(curx4, varZero);
            auto y0 = _mm_cmplt_ps(cury4, varZero);
            x0 = _mm_blendv_ps(plus, minus, x0);
            y0 = _mm_blendv_ps(plus, minus, y0);
            curx4 = _mm_add_ps(curx4, x0);
            cury4 = _mm_add_ps(cury4, y0);
            // __MM_FROUND_TO_ZERO
            auto ix0 = _mm_cvtps_epi32(_mm_round_ps(curx4, 3));
            auto iy0 = _mm_cvtps_epi32(_mm_round_ps(cury4, 3));
            
            int32_t posx[4], posy[4];
            _mm_store_si128((__m128i*)posx, ix0);
            _mm_store_si128((__m128i*)posy, iy0);

            curPoints.fY += 4 * dy;
            curPoints.fX += 4 * dx;

            auto sourcePos = _mm_add_epi32(_mm_mullo_epi32(iy0, yStride4), _mm_mullo_epi32(varBpp, ix0));
            int32_t pos4[4];
            _mm_store_si128((__m128i*)pos4, sourcePos);
            int iStart = 16 * i;
            auto w0 = *(int32_t*)(source + pos4[0]);
            auto w1 = *(int32_t*)(source + pos4[1]);
            auto w2 = *(int32_t*)(source + pos4[2]);
            auto w3 = *(int32_t*)(source + pos4[3]);
            *(int*)(dest + iStart) = w0;
            *(int*)(dest + iStart + 4) = w1;
            *(int*)(dest + iStart + 8) = w2;
            *(int*)(dest + iStart + 12) = w3;
            
        }
        start = sizedQuad * 4;
    }

    for (int i = start; i < count; ++i) {
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

void _SSE_MNNSampleBilinear(const unsigned char* source, unsigned char* dest, MNN::CV::Point* points, size_t count,
                            size_t iw, size_t ih, size_t yStride, size_t bpp) {
    float dy   = points[1].fY;
    float dx   = points[1].fX;
    float xMax = iw - 1;
    float yMax = ih - 1;

    MNN::CV::Point curPoints;
    curPoints.fX = points[0].fX;
    curPoints.fY = points[0].fY;
    int start = 0;

    if (count > 0 && bpp == 4) {
        __m128 minValue = _mm_set1_ps(0.f);
        __m128 maxValue = _mm_set1_ps(255.f);
        __m128i zero = _mm_set1_epi32(0);

        for (int i = 0; i < count; ++i) {
            float y  = __clamp(curPoints.fY, 0, yMax);
            float x  = __clamp(curPoints.fX, 0, xMax);
            int y0   = (int)y;
            int x0   = (int)x;
            int y1   = (int)ceilf(y);
            int x1   = (int)ceilf(x);
            float xF = x - (float)x0;
            float yF = y - (float)y0;
            
            int index0 = y0 * yStride + bpp * x0;
            int index1 = y0 * yStride + bpp * x1;
            int index2 = y1 * yStride + bpp * x0;
            int index3 = y1 * yStride + bpp * x1;
            
            auto f0 = _mm_set1_ps((1.0f - xF) * (1.0f - yF));
            auto f1 = _mm_set1_ps(xF * (1.0f - yF));
            auto f2 = _mm_set1_ps(yF * (1.0f - xF));
            auto f3 = _mm_set1_ps(xF * yF);

            if (bpp == 4) {
                auto c00_p0 = _mm_set_epi32(0, 0, 0, *(int32_t*)(source + index0));
                auto c01_p0 = _mm_set_epi32(0, 0, 0, *(int32_t*)(source + index1));
                auto c10_p0 = _mm_set_epi32(0, 0, 0, *(int32_t*)(source + index2));
                auto c11_p0 = _mm_set_epi32(0, 0, 0, *(int32_t*)(source + index3));
                // A
                auto c00_p0_16 = _mm_unpacklo_epi8(c00_p0, zero);
                auto c00_p0_32 = _mm_unpacklo_epi16(c00_p0_16, zero);
                auto c00_p0_f = _mm_cvtepi32_ps(c00_p0_32);
                
                auto c01_p0_16 = _mm_unpacklo_epi8(c01_p0, zero);
                auto c01_p0_32 = _mm_unpacklo_epi16(c01_p0_16, zero);
                auto c01_p0_f = _mm_cvtepi32_ps(c01_p0_32);
                
                auto c10_p0_16 = _mm_unpacklo_epi8(c10_p0, zero);
                auto c10_p0_32 = _mm_unpacklo_epi16(c10_p0_16, zero);
                auto c10_p0_f = _mm_cvtepi32_ps(c10_p0_32);
                
                auto c11_p0_16 = _mm_unpacklo_epi8(c11_p0, zero);
                auto c11_p0_32 = _mm_unpacklo_epi16(c11_p0_16, zero);
                auto c11_p0_f = _mm_cvtepi32_ps(c11_p0_32);
                
                auto v0 = _mm_mul_ps(f0, c00_p0_f);
                v0 = _mm_add_ps(v0, _mm_mul_ps(f1, c01_p0_f));
                v0 = _mm_add_ps(v0, _mm_mul_ps(f2, c10_p0_f));
                v0 = _mm_add_ps(v0, _mm_mul_ps(f3, c11_p0_f));

                v0 = _mm_min_ps(v0, maxValue);
                auto v0_m128i = _mm_cvtps_epi32(_mm_round_ps(_mm_max_ps(v0, minValue), 3));

                v0_m128i = _mm_packs_epi32(v0_m128i, v0_m128i);
                v0_m128i = _mm_packus_epi16(v0_m128i, v0_m128i);

                *((int*)(dest) + i) = _mm_cvtsi128_si32(v0_m128i);
            }
            curPoints.fY += dy;
            curPoints.fX += dx;
        }
        start = count;
    }

    for (int i = start; i < count; ++i) {
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

// requrie SSE 4.1
void _SSE_MNNSamplerC4Nearest(const unsigned char* source, unsigned char* dest, MNN::CV::Point* points, size_t sta,
                              size_t count, size_t capacity, size_t iw, size_t ih, size_t yStride){
    _SSE_MNNSamplerNearest(source, dest, points, sta, count, iw, ih, yStride, 4);
}

// requrie SSE 4.1
void _SSE_MNNSampleC4Bilinear(const unsigned char* source, unsigned char* dest, MNN::CV::Point* points, size_t sta,
                              size_t count, size_t capacity, size_t iw, size_t ih, size_t yStride) {
    _SSE_MNNSampleBilinear(source, dest + 4 * sta, points, count, iw, ih, yStride, 4);
}
