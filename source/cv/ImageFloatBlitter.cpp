                                                                                                                                                                                                                                                        //
//  ImageFloatBlitter.cpp
//  MNN
//
//  Created by MNN on 2018/12/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "cv/ImageFloatBlitter.hpp"
extern "C" {
void MNNBlitC1ToFloatRGBA(const unsigned char* source, float* dest, const float* mean, const float* normal,
                          size_t count);
void MNNBlitC3ToFloatRGBA(const unsigned char* source, float* dest, const float* mean, const float* normal,
                          size_t count);
void MNNBlitC4ToFloatC4(const unsigned char* source, float* dest, const float* mean, const float* normal, size_t count);
}
#ifdef MNN_USE_SSE
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#ifndef _MM_TRANSPOSE4_PS
#define _MM_TRANSPOSE4_PS(row0, row1, row2, row3) \
do { \
  __m128 tmp3, tmp2, tmp1, tmp0; \
  tmp0 = _mm_unpacklo_ps((row0), (row1)); \
  tmp2 = _mm_unpacklo_ps((row2), (row3)); \
  tmp1 = _mm_unpackhi_ps((row0), (row1)); \
  tmp3 = _mm_unpackhi_ps((row2), (row3)); \
  (row0) = _mm_movelh_ps(tmp0, tmp2); \
  (row1) = _mm_movehl_ps(tmp2, tmp0); \
  (row2) = _mm_movelh_ps(tmp1, tmp3); \
  (row3) = _mm_movehl_ps(tmp3, tmp1); \
} while (0)
#endif

#endif

#include "core/Macro.h"
#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif

namespace MNN {
namespace CV {
static void _blitC1ToFloatC1(const unsigned char* source, float* dest, const float* mean, const float* normal,
                             size_t count) {
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
    int remain = 0;
#ifdef MNN_USE_SSE
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
#endif
    for (int i = remain; i < count; ++i) {
        dest[i + 0] = normal[0] * (source[i + 0] - mean[0]);
    }
#endif
}

static void _blitC3ToFloatC3(const unsigned char* source, float* dest, const float* mean, const float* normal,
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
    int remain = 0;
#ifdef MNN_USE_SSE
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
#endif
    for (int i = remain; i < count; ++i) {
        dest[3 * i + 0] = normal[0] * (source[3 * i + 0] - mean[0]);
        dest[3 * i + 1] = normal[1] * (source[3 * i + 1] - mean[1]);
        dest[3 * i + 2] = normal[2] * (source[3 * i + 2] - mean[2]);
    }
#endif
}

void MNNBlitC4ToFloatC4(const unsigned char* source, float* dest, const float* mean, const float* normal,
                        size_t count) {
    for (int i = 0; i < count; ++i) {
        dest[4 * i + 0] = normal[0] * (source[4 * i + 0] - mean[0]);
        dest[4 * i + 1] = normal[1] * (source[4 * i + 1] - mean[1]);
        dest[4 * i + 2] = normal[2] * (source[4 * i + 2] - mean[2]);
        dest[4 * i + 3] = normal[3] * (source[4 * i + 3] - mean[3]);
    }
}
#ifndef MNN_USE_NEON
void MNNBlitC1ToFloatRGBA(const unsigned char* source, float* dest, const float* mean, const float* normal,
                          size_t count) {
    // MNN_PRINT("normal = %f\n", normal[0]);
    ::memset(dest, 0, 4 * sizeof(float) * count);
    int remain = 0;
#ifdef MNN_USE_SSE
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
            _mm_store_ps(curDst + 4 * 0, f0);
            _mm_store_ps(curDst + 4 * 1, r1);
            _mm_store_ps(curDst + 4 * 2, r2);
            _mm_store_ps(curDst + 4 * 3, r3);
            curDst += 16;

            _MM_TRANSPOSE4_PS(f1, r1, r2, r3);
            _mm_store_ps(curDst + 4 * 0, f1);
            _mm_store_ps(curDst + 4 * 1, r1);
            _mm_store_ps(curDst + 4 * 2, r2);
            _mm_store_ps(curDst + 4 * 3, r3);
            curDst += 16;

            _MM_TRANSPOSE4_PS(f2, r1, r2, r3);
            _mm_store_ps(curDst + 4 * 0, f2);
            _mm_store_ps(curDst + 4 * 1, r1);
            _mm_store_ps(curDst + 4 * 2, r2);
            _mm_store_ps(curDst + 4 * 3, r3);
            curDst += 16;

            _MM_TRANSPOSE4_PS(f3, r1, r2, r3);
            _mm_store_ps(curDst + 4 * 0, f3);
            _mm_store_ps(curDst + 4 * 1, r1);
            _mm_store_ps(curDst + 4 * 2, r2);
            _mm_store_ps(curDst + 4 * 3, r3);
        }
    }
#endif
    for (int i = remain; i < count; ++i) {
        dest[4 * i + 0] = normal[0] * (source[i + 0] - mean[0]);
    }
}

void MNNBlitC3ToFloatRGBA(const unsigned char* source, float* dest, const float* mean, const float* normal,
                          size_t count) {
int remain = 0;
#ifdef MNN_USE_SSE
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
#endif
    for (int i = remain; i < count; ++i) {
        dest[4 * i + 0] = normal[0] * (source[3 * i + 0] - mean[0]);
        dest[4 * i + 1] = normal[1] * (source[3 * i + 1] - mean[1]);
        dest[4 * i + 2] = normal[2] * (source[3 * i + 2] - mean[2]);
        dest[4 * i + 3] = 0.0f;
    }
}
#endif

ImageFloatBlitter::BLIT_FLOAT ImageFloatBlitter::choose(ImageFormat format, int dstBpp) {
    if (4 == dstBpp) {
        switch (format) {
            case GRAY:
                return MNNBlitC1ToFloatRGBA;
            case RGBA:
            case BGRA:
                return MNNBlitC4ToFloatC4;
            case RGB:
            case BGR:
                return MNNBlitC3ToFloatRGBA;
            default:
                break;
        }
    }
    switch (format) {
        case GRAY:
            return _blitC1ToFloatC1;
        case RGBA:
        case BGRA:
            return MNNBlitC4ToFloatC4;
        case RGB:
        case BGR:
            return _blitC3ToFloatC3;
        default:
            break;
    }
    return nullptr;
}

} // namespace CV
} // namespace MNN
