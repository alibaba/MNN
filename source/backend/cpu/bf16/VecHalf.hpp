//
//  VecHalf.hpp
//  MNN
//
//  Created by MNN on 2021/01/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VecHalf_hpp
#define VecHalf_hpp
#include "core/Macro.h"
#include <stdint.h>
#include <array>
#include <algorithm>  // supply std::max and std::min
namespace MNN {
namespace Math {

template <int N>
struct VecHalf {
    using VecType = VecHalf<N>;
    std::array<float, N> value;
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
    VecType operator*(const VecType& lr) const {
        VecType dst;
        for (int i = 0; i < N; ++i) {
            dst.value[i] = value[i] * lr.value[i];
        }
        return dst;
    }
    VecType operator*(float lr) const {
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
    VecHalf() {
    }
    VecHalf(const float v) {
        for (int i = 0; i < N; ++i) {
            value[i] = v;
        }
    }
    VecHalf(std::array<float, N>&& v) {
        value = std::move(v);
    }
    VecHalf(const VecType& lr) {
        for (int i = 0; i < N; ++i) {
            value[i] = lr.value[i];
        }
    }
    float operator[](size_t i) {
        return value[i];
    }
    static VecType broadcast(int16_t val) {
        VecType v;
        auto tempV = (int32_t*)v.value.data();
        for (int i = 0; i < N; ++i) {
            tempV[i] = val << 16;
        }
        return v;
    }
    static VecType load(const int16_t* addr) {
        VecType v;
        auto tempV = (int32_t*)v.value.data();
        for (int i = 0; i < N; ++i) {
            tempV[i] = addr[i] << 16;
        }
        return v;
    }
    static void save(int16_t* addr, const VecType& v) {
        auto tempV = (int32_t*)v.value.data();
        for (int i = 0; i < N; ++i) {
            addr[i] = tempV[i] >> 16;
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
};

#if defined(MNN_USE_SSE)
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

template<>
struct VecHalf<4> {
    using VecType = VecHalf<4>;
    __m128 value;
    VecType operator+(const VecType& lr) const {
        VecType dst = { _mm_add_ps(value, lr.value) };
        return dst;
    }
    VecType operator-(const VecType& lr) const {
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
    VecHalf() {
    }
    VecHalf(const float v) {
        value = _mm_set1_ps(v);
    }
    VecHalf(__m128& v) {
        value = v;
    }
    VecHalf(__m128&& v) {
        value = std::move(v);
    }
    VecHalf(const VecType& lr) {
        value = lr.value;
    }
    VecHalf(VecType&& lr) {
        value = std::move(lr.value);
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
    static VecType broadcast(int16_t val) {
        auto temp = _mm_set1_epi16(val);
#ifndef MNN_SSE_USE_FP16_INSTEAD
        auto zero = _mm_xor_si128(temp, temp);
        auto res = _mm_castsi128_ps(_mm_unpacklo_epi16(zero, temp));
#else
        auto res = _mm_cvtph_ps(temp);
#endif
        VecType v = { std::move(res) };
        return v;
    }
    static VecType load(const int16_t* addr) {
        auto temp = _mm_loadl_epi64((__m128i*)addr);
#ifndef MNN_SSE_USE_FP16_INSTEAD
        auto zero = _mm_xor_si128(temp, temp);
        auto res = _mm_castsi128_ps(_mm_unpacklo_epi16(zero, temp));
#else
        auto res = _mm_cvtph_ps(temp);
#endif
        VecType v = { std::move(res) };
        return v;
    }
    static void save(int16_t* addr, const VecType& v) {
#ifndef MNN_SSE_USE_FP16_INSTEAD
        auto temp = _mm_castps_si128(v.value);
        temp = _mm_srai_epi32(temp, 16);
        temp = _mm_packs_epi32(temp, temp);
#else
        static __m128 gMinValue = _mm_set1_ps(-32768);
        static __m128 gMaxValue = _mm_set1_ps(32767);
        auto t = _mm_max_ps(v.value, gMinValue);
        t = _mm_min_ps(t, gMaxValue);
        auto temp = _mm_cvtps_ph(t, 0x8);
#endif
        _mm_storel_epi64((__m128i*)addr, temp);
    }
    static VecType max(const VecType& v1, const VecType& v2) {
        VecType dst = { _mm_max_ps(v1.value, v2.value) };
        return dst;
    }
    static VecType min(const VecType& v1, const VecType& v2) {
        VecType dst = { _mm_min_ps(v1.value, v2.value) };
        return dst;
    }
};
#endif

#if defined(MNN_USE_NEON)
#include <arm_neon.h>

template<>
struct VecHalf<4> {
    using VecType = VecHalf<4>;
    float32x4_t value;
    VecType operator+(const VecType& lr) const {
        VecType dst = { vaddq_f32(value, lr.value) };
        return dst;
    }
    VecType operator-(const VecType& lr) const {
        VecType dst = { vsubq_f32(value, lr.value) };
        return dst;
    }
    VecType operator*(const VecType& lr) const {
        VecType dst = { vmulq_f32(value, lr.value) };
        return dst;
    }
    VecType operator*(const float lr) const {
        VecType dst = { vmulq_f32(value, vdupq_n_f32(lr)) };
        return dst;
    }

    VecType& operator=(const VecType& lr) {
        value = lr.value;
        return *this;
    }
    VecType operator-() {
        VecType dst = { vnegq_f32(value) };
        return dst;
    }
    VecHalf() {
    }
    VecHalf(const float v) {
        value = vdupq_n_f32(v);
    }
    VecHalf(float32x4_t& v) {
        value = v;
    }
    VecHalf(float32x4_t&& v) {
        value = std::move(v);
    }
    VecHalf(const VecType& lr) {
        value = lr.value;
    }
    VecHalf(VecType&& lr) {
        value = std::move(lr.value);
    }
    float operator[](const int i) {
        // vgetq_lane_f32(value, i) does NOT work, i must be const number such as 0, 2,
        return value[i];
    }
    static VecType broadcast(int16_t val) {
        VecType dst = { vreinterpretq_f32_s32(vshll_n_s16(vdup_n_s16(val), 16)) };
        return dst;
    }
    static VecType load(const int16_t* addr) {

        // equivalent to this:
        // int16x4_t vec4s16 = vld1_s16(addr);                  // load bf16 data as fixed point data of 16-bit.
        // int32x4_t vec4s32 =vshll_n_s16(vec4s16, 16);         // shift left 16bit as 32-bit data.
        // float32x4_t vec4f32 = vreinterpretq_f32_s32(vec4s32);// treat 32-bit fix point result as float32 data
        // VecType dest = { vec4f32 };                          // construct a struct of VecType

        VecType dst = { vreinterpretq_f32_s32(vshll_n_s16(vld1_s16(addr), 16)) };
        return dst;
    }
    static void save(int16_t* addr, const VecType& v) {
        vst1_s16(addr, vshrn_n_s32(vreinterpretq_s32_f32(v.value), 16));
        return;
    }
    static VecType max(const VecType& v1, const VecType& v2) {
        VecType dst = { vmaxq_f32(v1.value, v2.value) };
        return dst;
    }
    static VecType min(const VecType& v1, const VecType& v2) {
        VecType dst = { vminq_f32(v1.value, v2.value) };
        return dst;
    }
};
#endif

}

}
#endif
