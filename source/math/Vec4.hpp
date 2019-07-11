//
//  Vec4.hpp
//  MNN
//
//  Created by MNN on 2019/04/01.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef Vec4_hpp
#define Vec4_hpp
#include "Macro.h"
#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif
namespace MNN {
namespace Math {
#ifdef MNN_USE_NEON
struct Vec4 {
    float32x4_t value;
    Vec4() {
    }
    Vec4(const float v) {
        value = vdupq_n_f32(v);
    }
    Vec4(const Vec4& lr) {
        value = lr.value;
    }
    Vec4(const Vec4&& lr) {
        value = std::move(lr.value);
    }
    float operator[](int i) {
        return value[i];
    }
    static Vec4 load(const float* addr) {
        Vec4 v;
        v.value = vld1q_f32(addr);
        return v;
    }
    static void save(float* addr, const Vec4& v) {
        vst1q_f32(addr, v.value);
    }
    Vec4 operator+(const Vec4& lr) {
        Vec4 dst;
        dst.value = value + lr.value;
        return dst;
    }
    Vec4 operator-(const Vec4& lr) {
        Vec4 dst;
        dst.value = value - lr.value;
        return dst;
    }
    Vec4 operator*(float lr) {
        Vec4 dst;
        dst.value = vmulq_n_f32(value, lr);
        return dst;
    }
    Vec4 operator*(const Vec4& lr) {
        Vec4 dst;
        dst.value = value * lr.value;
        return dst;
    }
    Vec4& operator=(const Vec4& lr) {
        value = lr.value;
        return *this;
    }
    Vec4& operator=(const Vec4&& lr) {
        value = std::move(lr.value);
        return *this;
    }
    Vec4 operator-() {
        Vec4 dst;
        dst.value = -value;
        return dst;
    }
};
#elif defined(MNN_USE_SSE)
#include <emmintrin.h>
struct Vec4 {
    __m128 value;
    Vec4 operator+(const Vec4& lr) {
        Vec4 dst;
        dst.value = _mm_add_ps(value, lr.value);
        return dst;
    }
    Vec4 operator-(const Vec4& lr) {
        Vec4 dst;
        dst.value = _mm_sub_ps(value, lr.value);
        return dst;
    }
    Vec4 operator*(const Vec4& lr) {
        Vec4 dst;
        dst.value = _mm_mul_ps(value, lr.value);
        return dst;
    }
    Vec4 operator*(float lr) {
        Vec4 dst;
        dst.value = _mm_mul_ps(value, _mm_set1_ps(lr));
        return dst;
    }

    Vec4& operator=(const Vec4& lr) {
        value = lr.value;
        return *this;
    }
    Vec4 operator-() {
        Vec4 dst;
#if defined(_MSC_VER)
        dst.value = _mm_xor_ps(value, _mm_set1_ps(-0.f)); // Using unary operation to SSE vec is GCC extension. We can not do this directly in MSVC.
#else
        dst.value = -value;
#endif
        return dst;
    }
    Vec4() {
    }
    Vec4(const float v) {
        value = _mm_set1_ps(v);
    }

    Vec4(const Vec4& lr) {
        value = lr.value;
    }
    float operator[](int i) {
#if defined(_MSC_VER)  // X64 native only mandatory support SSE and SSE2 extension, and we can not find intrinsic function to extract element directly by index in SSE and SSE2 extension.
        float temp[4];
        _mm_store_ps(temp, value);
        return temp[i];
#else
        return value[i];
#endif
    }
    static Vec4 load(const float* addr) {
        Vec4 v;
        v.value = _mm_load_ps(addr);
        return v;
    }
    static void save(float* addr, const Vec4& v) {
        _mm_store_ps(addr, v.value);
    }
};
#else
struct Vec4 {
    float value[4];
    Vec4 operator+(const Vec4& lr) {
        Vec4 dst;
        for (int i = 0; i < 4; ++i) {
            dst.value[i] = value[i] + lr.value[i];
        }
        return dst;
    }
    Vec4 operator-(const Vec4& lr) {
        Vec4 dst;
        for (int i = 0; i < 4; ++i) {
            dst.value[i] = value[i] - lr.value[i];
        }
        return dst;
    }
    Vec4 operator*(const Vec4& lr) {
        Vec4 dst;
        for (int i = 0; i < 4; ++i) {
            dst.value[i] = value[i] * lr.value[i];
        }
        return dst;
    }
    Vec4 operator*(float lr) {
        Vec4 dst;
        for (int i = 0; i < 4; ++i) {
            dst.value[i] = value[i] * lr;
        }
        return dst;
    }

    Vec4& operator=(const Vec4& lr) {
        for (int i = 0; i < 4; ++i) {
            value[i] = lr.value[i];
        }
        return *this;
    }
    Vec4 operator-() {
        Vec4 dst;
        for (int i = 0; i < 4; ++i) {
            dst.value[i] = -value[i];
        }
        return dst;
    }
    Vec4() {
    }
    Vec4(const float v) {
        for (int i = 0; i < 4; ++i) {
            value[i] = v;
        }
    }

    Vec4(const Vec4& lr) {
        for (int i = 0; i < 4; ++i) {
            value[i] = lr.value[i];
        }
    }
    float operator[](int i) {
        return value[i];
    }
    static Vec4 load(const float* addr) {
        Vec4 v;
        for (int i = 0; i < 4; ++i) {
            v.value[i] = addr[i];
        }
        return v;
    }
    static void save(float* addr, const Vec4& v) {
        for (int i = 0; i < 4; ++i) {
            addr[i] = v.value[i];
        }
    }
};
#endif
} // namespace Math
} // namespace MNN

#endif /* Vec4_hpp */
