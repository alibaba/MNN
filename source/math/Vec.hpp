//
//  Vec.hpp
//  MNN
//
//  Created by MNN on 2019/04/01.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef Vec_hpp
#define Vec_hpp
#include "core/Macro.h"
#include <array>
#include <algorithm>  // supply std::max and std::min
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
namespace MNN {
namespace Math {

template <typename T, int N>
struct Vec {
    using VecType = Vec<T, N>;
    std::array<T, N> value;
    VecType operator+(const VecType& lr) {
        VecType dst;
        for (int i = 0; i < N; ++i) {
            dst.value[i] = value[i] + lr.value[i];
        }
        return dst;
    }
    VecType operator-(const VecType& lr) const {
        VecType dst;
        for (int i = 0; i < N; ++i) {
            dst.value[i] = value[i] - lr.value[i];
        }
        return dst;
    }
    Vec(std::array<T, N>&& v) {
        value = std::move(v);
    }
    VecType operator*(const VecType& lr) const {
        VecType dst;
        for (int i = 0; i < N; ++i) {
            dst.value[i] = value[i] * lr.value[i];
        }
        return dst;
    }
    VecType operator*(T lr) const {
        VecType dst;
        for (int i = 0; i < N; ++i) {
            dst.value[i] = value[i] * lr;
        }
        return dst;
    }

    VecType& operator=(const VecType& lr) {
        for (int i = 0; i < N; ++i) {
            value[i] = lr.value[i];
        }
        return *this;
    }
    VecType operator-() {
        VecType dst;
        for (int i = 0; i < N; ++i) {
            dst.value[i] = -value[i];
        }
        return dst;
    }
    Vec() {
    }
    Vec(const T v) {
        for (int i = 0; i < N; ++i) {
            value[i] = v;
        }
    }

    Vec(const VecType& lr) {
        for (int i = 0; i < N; ++i) {
            value[i] = lr.value[i];
        }
    }
    T operator[](size_t i) {
        return value[i];
    }
    template<typename U>
    static VecType load(const U* addr) {
        VecType v;
        for (int i = 0; i < N; ++i) {
            v.value[i] = static_cast<T>(addr[i]);
        }
        return v;
    }
    static void save(T* addr, const VecType& v) {
        for (int i = 0; i < N; ++i) {
            addr[i] = v.value[i];
        }
    }
    static VecType max(const VecType& v1, const VecType& v2) {
        VecType dst;
        for (int i = 0; i < N; ++i) {
            dst.value[i] = std::max(v1.value[i], v2.value[i]);
        }
        return dst;
    }
    static VecType min(const VecType& v1, const VecType& v2) {
        VecType dst;
        for (int i = 0; i < N; ++i) {
            dst.value[i] = std::min(v1.value[i], v2.value[i]);
        }
        return dst;
    }
    static void mla(VecType& v1, const VecType& v2, const VecType& v3) {
        v1 = v1 + v2 * v3;
    }
    static void mls(VecType& v1, const VecType& v2, const VecType& v3) {
        v1 = v1 - v2 * v3;
    }
};

#ifdef MNN_USE_NEON
template<>
struct Vec<float, 4> {
    using VecType = Vec<float, 4>;
    float32x4_t value;
    Vec() {
    }
    Vec(const float v) {
        value = vdupq_n_f32(v);
    }
    Vec(const float32x4_t v) {
        value = v;
    }
    Vec(const VecType& lr) {
        value = lr.value;
    }
    Vec(const VecType&& lr) {
        value = std::move(lr.value);
    }
    float operator[](size_t i) {
        return value[i];
    }
    static VecType load(const float* addr) {
        VecType v = { vld1q_f32(addr) };
        return v;
    }
    static VecType load(const int32_t* addr) {
        VecType v = { vcvtq_f32_s32(vld1q_s32(addr)) };
        return v;
    }
    static void save(float* addr, const VecType& v) {
        vst1q_f32(addr, v.value);
    }
    static VecType max(const VecType& v1, const VecType& v2) {
        VecType dst = { vmaxq_f32(v1.value, v2.value) };
        return dst;
    }
    static VecType min(const VecType& v1, const VecType& v2) {
        VecType dst = { vminq_f32(v1.value, v2.value) };
        return dst;
    }
    static void mla(VecType& v1, const VecType& v2, const VecType& v3) {
#ifdef __aarch64__
        v1.value = vfmaq_f32(v1.value, v2.value, v3.value);
#else
        v1 = v1 + v2 * v3;
#endif
    }
    static void mls(VecType& v1, const VecType& v2, const VecType& v3) {
#ifdef __aarch64__
        v1.value = vfmsq_f32(v1.value, v2.value, v3.value);
#else
        v1 = v1 - v2 * v3;
#endif
    }
    VecType operator+(const VecType& lr) {
        VecType dst = { vaddq_f32(value, lr.value) };
        return dst;
    }
    VecType operator-(const VecType& lr) {
        VecType dst = { vsubq_f32(value, lr.value) };
        return dst;
    }
    VecType operator*(float lr) const {
        VecType dst = { vmulq_n_f32(value, lr) };
        return dst;
    }
    VecType operator*(const VecType& lr) const {
        VecType dst = { vmulq_f32(value, lr.value) };
        return dst;
    }
    VecType& operator=(const VecType& lr) {
        value = lr.value;
        return *this;
    }
    VecType& operator=(const VecType&& lr) {
        value = std::move(lr.value);
        return *this;
    }
    VecType operator-() {
        VecType dst = { vnegq_f32(value) };
        return dst;
    }
};

template<>
struct Vec<int8_t, 8> {
    using VecType = Vec<int8_t, 8>;
    int8x8_t value;
    
    VecType operator + (const VecType& lr) {
        VecType dst = { vqadd_s8(value, lr.value) };
        return dst;
    }
    
    VecType operator - (const VecType& lr) {
        VecType dst = { vqsub_s8(value, lr.value) };
        return dst;
    }
    
    VecType operator - () {
        VecType dst = { vqneg_s8(value) };
        return dst;
    }

    VecType& operator = (const VecType& lr) {
        value = lr.value;
        return *this;
    }
    Vec() {
    }
    Vec(const int8_t v) {
        value = vdup_n_s8(v);
    }
    Vec(int8x8_t&& v) {
        value = v;
    }
    Vec(const VecType& lr) {
        value = lr.value;
    }
    int8_t operator[](size_t i) {
        return value[i];
    }
    static VecType load(const int8_t* addr) {
        VecType v = { vld1_s8(addr) };
        return v;
    }
    static void save(int8_t* addr, const VecType& v) {
        vst1_s8(addr, v.value);
    }
    static VecType max(const VecType& v1, const VecType& v2) {
        VecType dst = { vmax_s8(v1.value, v2.value) };
        return dst;
    }
    static VecType min(const VecType& v1, const VecType& v2) {
        VecType dst = { vmin_s8(v1.value, v2.value) };
        return dst;
    }
};

template<>
struct Vec<int8_t, 16> {
    using VecType = Vec<int8_t, 16>;
    int8x16_t value;
    
    VecType operator + (const VecType& lr) {
        VecType dst = { vqaddq_s8(value, lr.value) };
        return dst;
    }
    
    VecType operator - (const VecType& lr) {
        VecType dst = { vqsubq_s8(value, lr.value) };
        return dst;
    }
    
    VecType operator - () {
        VecType dst = { vqnegq_s8(value) };
        return dst;
    }

    VecType operator*(int8_t lr) {
        MNN_ERROR("Vec[NEON]: int8_t multiply maybe overflow!");
        VecType dst = { vmulq_s8(value, vdupq_n_s8(lr)) };
        return dst;
    }
    VecType operator*(const VecType& lr) {
        MNN_ERROR("Vec[NEON]: int8_t multiply maybe overflow!");
        VecType dst = { vmulq_s8(value, lr.value) };
        return dst;
    }

    VecType& operator=(const VecType& lr) {
        value = lr.value;
        return *this;
    }
    Vec() {
    }
    Vec(const int8_t v) {
        value = vdupq_n_s8(v);
    }
    Vec(int8x16_t&& v) {
        value = v;
    }
    Vec(const VecType& lr) {
        value = lr.value;
    }
    int8_t operator[](size_t i) {
        return value[i];
    }
    static VecType load(const int8_t* addr) {
        VecType v = { vld1q_s8(addr) };
        return v;
    }
    static void save(int8_t* addr, const VecType& v) {
        vst1q_s8(addr, v.value);
    }
    static VecType max(const VecType& v1, const VecType& v2) {
        VecType dst = { vmaxq_s8(v1.value, v2.value) };
        return dst;
    }
    static VecType min(const VecType& v1, const VecType& v2) {
        VecType dst = { vminq_s8(v1.value, v2.value) };
        return dst;
    }
};
#elif defined(MNN_USE_SSE)
template<>
struct Vec<float, 4> {
    using VecType = Vec<float, 4>;
    __m128 value;
    VecType operator+(const VecType& lr) {
        VecType dst = { _mm_add_ps(value, lr.value) };
        return dst;
    }
    VecType operator-(const VecType& lr) {
        VecType dst = { _mm_sub_ps(value, lr.value) };
        return dst;
    }
    VecType operator*(const VecType& lr) const {
        VecType dst = { _mm_mul_ps(value, lr.value) };
        return dst;
    }
    VecType operator*(float lr) const {
        VecType dst = { _mm_mul_ps(value, _mm_set1_ps(lr)) };
        return dst;
    }

    VecType& operator=(const VecType& lr) {
        value = lr.value;
        return *this;
    }
    VecType operator-() {
        VecType dst;
#if defined(_MSC_VER)
        dst.value = _mm_xor_ps(value, _mm_set1_ps(-0.f)); // Using unary operation to SSE vec is GCC extension. We can not do this directly in MSVC.
#else
        dst.value = -value;
#endif
        return dst;
    }
    Vec() {
    }
    Vec(const float v) {
        value = _mm_set1_ps(v);
    }
    Vec(__m128&& v) {
        value = v;
    }
    Vec(const VecType& lr) {
        value = lr.value;
    }
    float operator[](size_t i) {
#if defined(_MSC_VER)  // X64 native only mandatory support SSE and SSE2 extension, and we can not find intrinsic function to extract element directly by index in SSE and SSE2 extension.
        float temp[4];
        _mm_storeu_ps(temp, value);
        return temp[i];
#else
        return value[i];
#endif
    }
    static VecType load(const float* addr) {
        VecType v = { _mm_loadu_ps(addr) };
        return v;
    }
    static VecType load(const int32_t* addr) {
        VecType v = { _mm_cvtepi32_ps(_mm_loadu_si128((__m128i const*)(addr))) };
        return v;
    }
    static void save(float* addr, const VecType& v) {
        _mm_storeu_ps(addr, v.value);
    }
    static VecType max(const VecType& v1, const VecType& v2) {
        VecType dst = { _mm_max_ps(v1.value, v2.value) };
        return dst;
    }
    static VecType min(const VecType& v1, const VecType& v2) {
        VecType dst = { _mm_min_ps(v1.value, v2.value) };
        return dst;
    }
    static void mla(VecType& v1, const VecType& v2, const VecType& v3) {
        v1 = v1 + v2 * v3; // TODO: use fma instruction
    }
    static void mls(VecType& v1, const VecType& v2, const VecType& v3) {
        v1 = v1 - v2 * v3; // TODO: use fma instruction
    }
};
template<>
struct Vec<int8_t, 16> {
    using VecType = Vec<int8_t, 16>;
    __m128i value;
    VecType operator+(const VecType& lr) {
        VecType dst = { _mm_add_epi8(value, lr.value) };
        return dst;
    }
    VecType operator-(const VecType& lr) {
        VecType dst = { _mm_sub_epi8(value, lr.value) };
        return dst;
    }
    VecType operator*(const VecType& lr) {
        MNN_ERROR("Vec[SSE]: int8_t multiply maybe overflow!");
        VecType dst = { _mul_epi8(value, lr.value) };
        return dst;
    }
    VecType operator*(float lr) {
        MNN_ERROR("Vec[SSE]: int8_t multiply maybe overflow!");
        VecType dst = { _mul_epi8(value, _mm_set1_epi8(lr)) };
        return dst;
    }

    VecType& operator=(const VecType& lr) {
        value = lr.value;
        return *this;
    }
    VecType operator-() {
        VecType dst;
#if defined(_MSC_VER)
        dst.value = _mm_sign_epi8(value, _mm_set1_epi8(-1)); // Using unary operation to SSE vec is GCC extension. We can not do this directly in MSVC.
#else
        dst.value = -value;
#endif
        return dst;
    }
    Vec() {
    }
    Vec(const int8_t v) {
        value = _mm_set1_epi8(v);
    }
    Vec(__m128i&& v) {
        value = v;
    }
    Vec(const VecType& lr) {
        value = lr.value;
    }
    float operator[](size_t i) {
#if defined(_MSC_VER)  // X64 native only mandatory support SSE and SSE2 extension, and we can not find intrinsic function to extract element directly by index in SSE and SSE2 extension.
        int8_t temp[16];
        _mm_storeu_ps((float*)temp, _mm_castsi128_ps(value));
        return temp[i];
#else
        return value[i];
#endif
    }
    static VecType load(const int8_t* addr) {
        VecType v = { _mm_castps_si128(_mm_loadu_ps((const float*)addr)) };
        return v;
    }
    static void save(int8_t* addr, const VecType& v) {
        _mm_storeu_ps((float*)addr, _mm_castsi128_ps(v.value));
    }
    static VecType max(const VecType& v1, const VecType& v2) {
        VecType dst = { _max_epi8(v1.value, v2.value) };
        return dst;
    }
    static VecType min(const VecType& v1, const VecType& v2) {
        VecType dst = { _min_epi8(v1.value, v2.value) };
        return dst;
    }
private:
    static __m128i _max_epi8(__m128i a, __m128i b) {
#ifdef __SSE4_1__
        return _mm_max_epi8(a, b);
#else
        auto mask0 = _mm_cmpgt_epi8(a, b);
        auto mask1 = _mm_xor_si128(mask0, _mm_cmpeq_epi8(mask0, mask0));
        return _mm_or_si128(_mm_and_si128(mask0, a), _mm_and_si128(mask1, b));
#endif
    }
    static __m128i _min_epi8(__m128i a, __m128i b) {
#ifdef __SSE4_1__
        return _mm_min_epi8(a, b);
#else
        auto mask0 = _mm_cmplt_epi8(a, b);
        auto mask1 = _mm_xor_si128(mask0, _mm_cmpeq_epi8(mask0, mask0));
        return _mm_or_si128(_mm_and_si128(mask0, a), _mm_and_si128(mask1, b));
#endif
    }
    __m128i _mul_epi8(__m128i a, __m128i b)
    {
        __m128i dst_even = _mm_mullo_epi16(a, b);
        __m128i dst_odd = _mm_mullo_epi16(_mm_srli_epi16(a, 8),_mm_srli_epi16(b, 8));
        return _mm_or_si128(_mm_slli_epi16(dst_odd, 8), _mm_srli_epi16(_mm_slli_epi16(dst_even,8), 8));
    }
};
#endif
} // namespace Math
} // namespace MNN

#endif /* Vec_hpp */
