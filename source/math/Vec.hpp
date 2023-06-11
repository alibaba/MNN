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
    VecType operator+(const VecType& lr) const {
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
    VecType operator+=(const VecType& lr) {
        for (int i = 0; i < N; ++i) {
            value[i] = value[i] + lr.value[i];
        }
        return *this;
    }
    VecType operator-=(const VecType& lr) {
        for (int i = 0; i < N; ++i) {
            value[i] = value[i] - lr.value[i];
        }
        return *this;
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
    template<typename U>
    static VecType broadcast(const U* addr) {
        VecType v;
        v.value[0] = static_cast<T>(addr[0]);
        for (int i = 1; i < N; ++i) {
            v.value[i] = v.value[0];
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
    static VecType fma(const VecType& v1, const VecType& v2, const VecType& v3) {
         return v1 + v2 * v3;
    }
    static VecType fms(const VecType& v1, const VecType& v2, const VecType& v3) {
        return v1 - v2 * v3;
    }
    static inline void transpose4(VecType& vec0, VecType& vec1, VecType& vec2, VecType& vec3) {
        VecType source[4] = {vec0, vec1, vec2, vec3};
        for (int i = 0; i < N; ++i) {
            vec0.value[i] = source[i % 4].value[i >> 2];
            vec1.value[i] = source[i % 4].value[(i + N)>> 2];
            vec2.value[i] = source[i % 4].value[(i + 2 * N)>> 2];
            vec3.value[i] = source[i % 4].value[(i + 3 * N)>> 2];
        }
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
    static VecType broadcast(const float* addr) {
        VecType dst = { vld1q_dup_f32(addr) };
        return dst;
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
    static VecType fma(const VecType& v1, const VecType& v2, const VecType& v3) {
        VecType dst = {vmlaq_f32(v1.value, v2.value, v3.value)};
        return dst;
    }
    static VecType fms(const VecType& v1, const VecType& v2, const VecType& v3) {
        VecType dst = {vmlsq_f32(v1.value, v2.value, v3.value)};
        return dst;
    }
    static inline void transpose4(VecType& vec0, VecType& vec1, VecType& vec2, VecType& vec3) {
#ifdef __aarch64__
        auto m0 = vtrn1q_s32(reinterpret_cast<int32x4_t>(vec0.value), reinterpret_cast<int32x4_t>(vec1.value));
        auto m1 = vtrn2q_s32(reinterpret_cast<int32x4_t>(vec0.value), reinterpret_cast<int32x4_t>(vec1.value));
        auto m2 = vtrn1q_s32(reinterpret_cast<int32x4_t>(vec2.value), reinterpret_cast<int32x4_t>(vec3.value));
        auto m3 = vtrn2q_s32(reinterpret_cast<int32x4_t>(vec2.value), reinterpret_cast<int32x4_t>(vec3.value));
        vec0.value = reinterpret_cast<float32x4_t>(vtrn1q_s64(reinterpret_cast<int64x2_t>(m0), reinterpret_cast<int64x2_t>(m2)));
        vec1.value = reinterpret_cast<float32x4_t>(vtrn1q_s64(reinterpret_cast<int64x2_t>(m1), reinterpret_cast<int64x2_t>(m3)));
        vec2.value = reinterpret_cast<float32x4_t>(vtrn2q_s64(reinterpret_cast<int64x2_t>(m0), reinterpret_cast<int64x2_t>(m2)));
        vec3.value = reinterpret_cast<float32x4_t>(vtrn2q_s64(reinterpret_cast<int64x2_t>(m1), reinterpret_cast<int64x2_t>(m3)));
#else

        auto m0m1 = vtrnq_s32(reinterpret_cast<int32x4_t>(vec0.value), reinterpret_cast<int32x4_t>(vec1.value));
        auto m2m3 = vtrnq_s32(reinterpret_cast<int32x4_t>(vec2.value), reinterpret_cast<int32x4_t>(vec3.value));
        vec0.value = reinterpret_cast<float32x4_t>(m0m1.val[0]);
        vec1.value = reinterpret_cast<float32x4_t>(m0m1.val[1]);
        vec2.value = reinterpret_cast<float32x4_t>(m2m3.val[0]);
        vec3.value = reinterpret_cast<float32x4_t>(m2m3.val[1]);
        vec0.value = reinterpret_cast<float32x4_t>(vsetq_lane_s64(vgetq_lane_s64(reinterpret_cast<int64x2_t>(m2m3.val[0]), 0), reinterpret_cast<int64x2_t>(vec0.value), 1));
        vec1.value = reinterpret_cast<float32x4_t>(vsetq_lane_s64(vgetq_lane_s64(reinterpret_cast<int64x2_t>(m2m3.val[1]), 0), reinterpret_cast<int64x2_t>(vec1.value), 1));
        vec2.value = reinterpret_cast<float32x4_t>(vsetq_lane_s64(vgetq_lane_s64(reinterpret_cast<int64x2_t>(m0m1.val[0]), 1), reinterpret_cast<int64x2_t>(vec2.value), 0));
        vec3.value = reinterpret_cast<float32x4_t>(vsetq_lane_s64(vgetq_lane_s64(reinterpret_cast<int64x2_t>(m0m1.val[1]), 1), reinterpret_cast<int64x2_t>(vec3.value), 0));
        /*
        generated arm32 assembly code is almost the same as:
            vtrn.32 d0, d2
            vtrn.32 d1, d3
            vtrn.32 d4, d6
            vtrn.32 d5, d7
            vswp d1, d4
            vswp d3, d6
        */

#endif
    }

    VecType operator+(const VecType& lr) const {
        VecType dst = { vaddq_f32(value, lr.value) };
        return dst;
    }
    VecType operator-(const VecType& lr) const {
        VecType dst = { vsubq_f32(value, lr.value) };
        return dst;
    }
    VecType operator+=(const VecType& lr) {
        value = vaddq_f32(value, lr.value);
        return *this;
    }
    VecType operator-=(const VecType& lr) {
        value = vsubq_f32(value, lr.value);
        return *this;
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
struct Vec<int8_t, 16> {
    using VecType = Vec<int8_t, 16>;
    int8x16_t value;
    Vec() {
    }
    Vec(const int8_t v) {
        value = vdupq_n_s8(v);
    }
    Vec(const int8x16_t v) {
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
    static VecType load(const int8_t* addr) {
        VecType v = { vld1q_s8(addr) };
        return v;
    }
    static VecType broadcast(const int8_t* addr) {
        VecType dst = { vld1q_dup_s8(addr) };
        return dst;
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
    static VecType fma(const VecType& v1, const VecType& v2, const VecType& v3) {
        VecType dst = {vmlaq_s8(v1.value, v2.value, v3.value)};
        return dst;
    }
    static VecType fms(const VecType& v1, const VecType& v2, const VecType& v3) {
        VecType dst = {vmlsq_s8(v1.value, v2.value, v3.value)};
        return dst;
    }
    static inline void transpose4(VecType& vec0, VecType& vec1, VecType& vec2, VecType& vec3) {
#ifdef __aarch64__
        auto m0 = vtrn1q_s8(reinterpret_cast<int8x16_t>(vec0.value), reinterpret_cast<int8x16_t>(vec1.value));
        auto m1 = vtrn2q_s8(reinterpret_cast<int8x16_t>(vec0.value), reinterpret_cast<int8x16_t>(vec1.value));
        auto m2 = vtrn1q_s8(reinterpret_cast<int8x16_t>(vec2.value), reinterpret_cast<int8x16_t>(vec3.value));
        auto m3 = vtrn2q_s8(reinterpret_cast<int8x16_t>(vec2.value), reinterpret_cast<int8x16_t>(vec3.value));
        vec0.value = reinterpret_cast<int8x16_t>(vtrn1q_s16(reinterpret_cast<int16x8_t>(m0), reinterpret_cast<int16x8_t>(m2)));
        vec1.value = reinterpret_cast<int8x16_t>(vtrn1q_s16(reinterpret_cast<int16x8_t>(m1), reinterpret_cast<int16x8_t>(m3)));
        vec2.value = reinterpret_cast<int8x16_t>(vtrn2q_s16(reinterpret_cast<int16x8_t>(m0), reinterpret_cast<int16x8_t>(m2)));
        vec3.value = reinterpret_cast<int8x16_t>(vtrn2q_s16(reinterpret_cast<int16x8_t>(m1), reinterpret_cast<int16x8_t>(m3)));
#else
        auto m0m1 = vtrnq_s8(reinterpret_cast<int8x16_t>(vec0.value), reinterpret_cast<int16x8_t>(vec1.value));
        auto m2m3 = vtrnq_s8(reinterpret_cast<int8x16_t>(vec2.value), reinterpret_cast<int16x8_t>(vec3.value));
        vec0.value = reinterpret_cast<int8x16_t>(m0m1.val[0]);
        vec1.value = reinterpret_cast<int8x16_t>(m0m1.val[1]);
        vec2.value = reinterpret_cast<int8x16_t>(m2m3.val[0]);
        vec3.value = reinterpret_cast<int8x16_t>(m2m3.val[1]);
        vec0.value = reinterpret_cast<int8x16_t>(vsetq_lane_s16(vgetq_lane_s16(reinterpret_cast<int16x8_t>(m2m3.val[0]), 0), reinterpret_cast<int16x8_t>(vec0.value), 1));
        vec1.value = reinterpret_cast<int8x16_t>(vsetq_lane_s16(vgetq_lane_s16(reinterpret_cast<int16x8_t>(m2m3.val[1]), 0), reinterpret_cast<int16x8_t>(vec1.value), 1));
        vec2.value = reinterpret_cast<int8x16_t>(vsetq_lane_s16(vgetq_lane_s16(reinterpret_cast<int16x8_t>(m0m1.val[0]), 1), reinterpret_cast<int16x8_t>(vec2.value), 0));
        vec3.value = reinterpret_cast<int8x16_t>(vsetq_lane_s16(vgetq_lane_s16(reinterpret_cast<int16x8_t>(m0m1.val[1]), 1), reinterpret_cast<int16x8_t>(vec3.value), 0));
        /*
        generated arm32 assembly code is almost the same as:
            vtrn.32 d0, d2
            vtrn.32 d1, d3
            vtrn.32 d4, d6
            vtrn.32 d5, d7
            vswp d1, d4
            vswp d3, d6
        */

#endif
    }

    VecType operator+(const VecType& lr) const {
        VecType dst = { vaddq_s8(value, lr.value) };
        return dst;
    }
    VecType operator-(const VecType& lr) const {
        VecType dst = { vsubq_s8(value, lr.value) };
        return dst;
    }
    VecType operator+=(const VecType& lr) {
        value = vaddq_s8(value, lr.value);
        return *this;
    }
    VecType operator-=(const VecType& lr) {
        value = vsubq_s8(value, lr.value);
        return *this;
    }
//    VecType operator*(int8_t lr) const {
//        VecType dst = { vmulq_n_s8(value, lr) };
//        return dst;
//    }
    VecType operator*(const VecType& lr) const {
        VecType dst = { vmulq_s8(value, lr.value) };
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
        VecType dst = { vnegq_s8(value) };
        return dst;
    }
};

#elif defined(MNN_USE_SSE)
template<>
struct Vec<float, 4> {
    using VecType = Vec<float, 4>;
    __m128 value;
    VecType operator+(const VecType& lr) const {
        VecType dst = { _mm_add_ps(value, lr.value) };
        return dst;
    }
    VecType operator-(const VecType& lr) const {
        VecType dst = { _mm_sub_ps(value, lr.value) };
        return dst;
    }
    VecType operator+=(const VecType& lr) {
        value = _mm_add_ps(value, lr.value);
        return *this;
    }
    VecType operator-=(const VecType& lr) {
        value = _mm_sub_ps(value, lr.value);
        return *this;
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
    static VecType broadcast(const float* addr) {
        VecType dst = { _mm_load_ss(addr) };
        return dst;
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
    static VecType fma(const VecType& v1, const VecType& v2, const VecType& v3) {
        return v1 + v2 * v3; // TODO: use fma instruction
    }
    static VecType fms(const VecType& v1, const VecType& v2, const VecType& v3) {
        return v1 - v2 * v3; // TODO: use fma instruction
    }
    static inline void transpose4(VecType& vec0, VecType& vec1, VecType& vec2, VecType& vec3) {
        __m128 tmp3, tmp2, tmp1, tmp0;
        tmp0   = _mm_unpacklo_ps((vec0.value), (vec1.value));
        tmp2   = _mm_unpacklo_ps((vec2.value), (vec3.value));
        tmp1   = _mm_unpackhi_ps((vec0.value), (vec1.value));
        tmp3   = _mm_unpackhi_ps((vec2.value), (vec3.value));
        vec0.value = _mm_movelh_ps(tmp0, tmp2);
        vec1.value = _mm_movehl_ps(tmp2, tmp0);
        vec2.value = _mm_movelh_ps(tmp1, tmp3);
        vec3.value = _mm_movehl_ps(tmp3, tmp1);
    }
};

#endif
} // namespace Math
} // namespace MNN

#endif /* Vec_hpp */
