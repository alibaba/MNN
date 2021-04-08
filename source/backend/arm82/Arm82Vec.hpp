//
//  Arm82Vec.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#if defined(__ANDROID__) || defined(__aarch64__)

#ifndef Arm82Vec_hpp
#define Arm82Vec_hpp

#include "Arm82Backend.hpp"
#include "math/Vec.hpp"

#ifdef MNN_USE_NEON
namespace MNN {
namespace Math {
template<>
struct Vec<FLOAT16, 8> {
    using VecType = Vec<FLOAT16, 8>;
    float16x8_t value;
    Vec() {
    }
    Vec(const float v) {
        value = vdupq_n_f16(v);
    }
    Vec(const float16x8_t v) {
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
    static VecType load(const FLOAT16* addr) {
        VecType v = { vld1q_f16(addr) };
        return v;
    }
    static void save(FLOAT16* addr, const VecType& v) {
        vst1q_f16(addr, v.value);
    }
    static VecType max(const VecType& v1, const VecType& v2) {
        VecType dst = { vmaxq_f16(v1.value, v2.value) };
        return dst;
    }
    static VecType min(const VecType& v1, const VecType& v2) {
        VecType dst = { vminq_f16(v1.value, v2.value) };
        return dst;
    }
    static void mla(VecType& v1, const VecType& v2, const VecType& v3) {
        v1.value = vfmaq_f16(v1.value, v2.value, v3.value);
    }
    static void mls(VecType& v1, const VecType& v2, const VecType& v3) {
        v1.value = vfmsq_f16(v1.value, v2.value, v3.value);
    }
    VecType operator+(const VecType& lr) {
        VecType dst = { vaddq_f16(value, lr.value) };
        return dst;
    }
    VecType operator-(const VecType& lr) {
        VecType dst = { vsubq_f16(value, lr.value) };
        return dst;
    }
    VecType operator*(float lr) {
        VecType dst = { vmulq_n_f16(value, lr) };
        return dst;
    }
    VecType operator*(const VecType& lr) {
        VecType dst = { vmulq_f16(value, lr.value) };
        return dst;
    }
    VecType operator/(float lr) {
#if defined(__aarch64__)
        VecType dst = { vdivq_f16(value, vdupq_n_f16(lr)) };
#else
        VecType dst;
        for (int i = 0; i < 8; ++i) {
            dst.value[i] = value[i] / lr;
        }
#endif
        return dst;
    }
    VecType operator/(const VecType& lr) {
#if defined(__aarch64__)
        VecType dst = { vdivq_f16(value, lr.value) };
#else
        VecType dst;
        for (int i = 0; i < 8; ++i) {
            dst.value[i] = value[i] / lr.value[i];
        }
#endif
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
        VecType dst = { vnegq_f16(value) };
        return dst;
    }
};
} // namespace Math
} // namespace MNN
#endif /* MNN_USE_NEON */

#endif // Arm82Vec_hpp
#endif
