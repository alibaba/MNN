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

    VecHalf(float v0, float v1, float v2, float v3) {
        value[0] = v0;
        value[1] = v1;
        value[2] = v2;
        value[3] = v3;
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
    static VecType broadcast(int16_t* val) {
        VecType v;
        auto tempV = (int32_t*)v.value.data();
        tempV[0] = (*val) << 16;
        for (int i = 1; i < N; ++i) {
            tempV[i] = tempV[0];
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

    static inline void transpose12(int16_t* srcPtr, const size_t packCUnit) {

        MNN_ASSERT(false);
    }
};

#if defined(MNN_USE_SSE)

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
    VecType operator+=(const VecType& lr) {
        value = _mm_add_ps(value, lr.value);
        return *this;
    }
    VecType operator-=(const VecType& lr) {
        value = _mm_sub_ps(value, lr.value);
        return *this;
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
    VecHalf(const float f0, const float f1, const float f2, const float f3) {
        value = _mm_set_ps(f0, f1, f2, f3);
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
    static VecType broadcast(int16_t* val) {
        return broadcast(*val);
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
    static VecType fma(const VecType& v1, const VecType& v2, const VecType& v3) {
        return v1 + v2 * v3;
    }
    static VecType fms(const VecType& v1, const VecType& v2, const VecType& v3) {
        return v1 - v2 * v3;
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

    // x86 VecHalf transpose12 unused in any case
    static inline void transpose12(int16_t* srcPtr, const size_t packCUnit) {
        MNN_ASSERT(false);
    }
};
#endif

#if defined(MNN_USE_NEON)

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
    VecType operator+=(const VecType& lr) {
        value = vaddq_f32(value, lr.value);
        return *this;
    }
    VecType operator-=(const VecType& lr) {
        value = vsubq_f32(value, lr.value);
        return *this;
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
    VecHalf(const float f0, const float f1, const float f2, const float f3) {
         vsetq_lane_f32(f0, value, 0);
         vsetq_lane_f32(f1, value, 1);
         vsetq_lane_f32(f2, value, 2);
         vsetq_lane_f32(f3, value, 3);
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
    static VecType broadcast(int16_t* valPtr) {
        VecType dst = { vreinterpretq_f32_s32(vshll_n_s16(vld1_dup_s16(valPtr), 16)) };
        return dst;
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
    static inline void transpose4(int16x4_t& vec0, int16x4_t& vec1, int16x4_t& vec2, int16x4_t& vec3) {
        auto trans0 = vtrn_s16(vec0, vec1);
        auto m0 = trans0.val[0];
        auto m1 = trans0.val[1];
        auto trans1 = vtrn_s16(vec2, vec3);
        auto m2 = trans1.val[0];
        auto m3 = trans1.val[1];
        auto trans2 = vtrn_s32(reinterpret_cast<int32x2_t>(m0), reinterpret_cast<int32x2_t>(m2));
        vec0 = reinterpret_cast<int16x4_t>(trans2.val[0]);
        vec2 = reinterpret_cast<int16x4_t>(trans2.val[1]);
        auto trans3 = vtrn_s32(reinterpret_cast<int32x2_t>(m1), reinterpret_cast<int32x2_t>(m3));
        vec1 = reinterpret_cast<int16x4_t>(trans3.val[0]);
        vec3 = reinterpret_cast<int16x4_t>(trans3.val[1]);

    }
    static inline void transpose12(int16_t* srcPtr, const size_t packCUnit) {
        auto s0  = vld1_s16(srcPtr + 0 * packCUnit);
        auto s3  = vld1_s16(srcPtr + 1 * packCUnit);
        auto s6  = vld1_s16(srcPtr + 2 * packCUnit);
        auto s9  = vld1_s16(srcPtr + 3 * packCUnit);
        auto s1  = vld1_s16(srcPtr + 4 * packCUnit);
        auto s4  = vld1_s16(srcPtr + 5 * packCUnit);
        auto s7  = vld1_s16(srcPtr + 6 * packCUnit);
        auto s10 = vld1_s16(srcPtr + 7 * packCUnit);
        auto s2  = vld1_s16(srcPtr + 8 * packCUnit);
        auto s5  = vld1_s16(srcPtr + 9 * packCUnit);
        auto s8  = vld1_s16(srcPtr + 10 * packCUnit);
        auto s11 = vld1_s16(srcPtr + 11 * packCUnit);

        transpose4(s0, s3, s6, s9);
        transpose4(s1, s4, s7, s10);
        transpose4(s2, s5, s8, s11);

        vst1_s16(srcPtr + 0 * packCUnit, s0);
        vst1_s16(srcPtr + 1 * packCUnit, s1);
        vst1_s16(srcPtr + 2 * packCUnit, s2);
        vst1_s16(srcPtr + 3 * packCUnit, s3);
        vst1_s16(srcPtr + 4 * packCUnit, s4);
        vst1_s16(srcPtr + 5 * packCUnit, s5);
        vst1_s16(srcPtr + 6 * packCUnit, s6);
        vst1_s16(srcPtr + 7 * packCUnit, s7);
        vst1_s16(srcPtr + 8 * packCUnit, s8);
        vst1_s16(srcPtr + 9 * packCUnit, s9);
        vst1_s16(srcPtr + 10 * packCUnit, s10);
        vst1_s16(srcPtr + 11 * packCUnit, s11);

    }
};
#endif

}

}
#endif
