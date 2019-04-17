//
//  ImageFloatBlitter.cpp
//  MNN
//
//  Created by MNN on 2018/12/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ImageFloatBlitter.hpp"
extern "C" {
void MNNBlitC1ToFloatRGBA(const unsigned char* source, float* dest, const float* mean, const float* normal,
                          size_t count);
void MNNBlitC3ToFloatRGBA(const unsigned char* source, float* dest, const float* mean, const float* normal,
                          size_t count);
void MNNBlitC4ToFloatC4(const unsigned char* source, float* dest, const float* mean, const float* normal, size_t count);
}

#include "Macro.h"
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
    dest   = dest + left;
    source = source + left;
    for (int i = left; i > 0; --i, --dest, --source) {
        *dest = normal[0] * (*source - mean[0]);
    }
#else
    for (int i = 0; i < count; ++i) {
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
    for (int i = 0; i < count; ++i) {
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
    for (int i = 0; i < count; ++i) {
        dest[4 * i + 0] = normal[0] * (source[i + 0] - mean[0]);
    }
}

void MNNBlitC3ToFloatRGBA(const unsigned char* source, float* dest, const float* mean, const float* normal,
                          size_t count) {
    for (int i = 0; i < count; ++i) {
        dest[4 * i + 0] = normal[0] * (source[3 * i + 0] - mean[0]);
        dest[4 * i + 1] = normal[1] * (source[3 * i + 1] - mean[1]);
        dest[4 * i + 2] = normal[2] * (source[3 * i + 2] - mean[2]);
        dest[4 * i + 3] = 0.0f;
    }
}
#endif

ImageFloatBlitter::BLIT_FLOAT ImageFloatBlitter::choose(ImageFormat format, MNN_DATA_FORMAT dimensionformat) {
    if (dimensionformat == MNN_DATA_FORMAT_NC4HW4) {
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
