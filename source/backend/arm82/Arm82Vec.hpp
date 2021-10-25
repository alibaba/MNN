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
    static VecType fma(const VecType& v1, const VecType& v2, const VecType& v3) {
        VecType dst = { vfmaq_f16(v1.value, v2.value, v3.value) };
        return dst;
    }
    static VecType fms(const VecType& v1, const VecType& v2, const VecType& v3) {
        VecType dst = { vfmsq_f16(v1.value, v2.value, v3.value) };
        return dst;
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

    static inline void transpose12(VecType& vec0, VecType& vec1, VecType& vec2, VecType& vec3, VecType& vec4,
                                   VecType& vec5, VecType& vec6, VecType& vec7, VecType& vec8, VecType& vec9,
                                   VecType& vec10, VecType& vec11) {

#ifdef __aarch64__
        auto tmp1 = vzipq_s16(vec0.value, vec1.value); // tmp1 would disappear after compile
        auto v21 = tmp1.val[0];
        auto v22 = tmp1.val[1];
        auto tmp2 = vzipq_s16(vec2.value, vec3.value);
        auto v24 = tmp2.val[0];
        auto v25 = tmp2.val[1];
        auto tmp3 = vzipq_s16(vec4.value, vec5.value);
        auto v27 = tmp3.val[0];
        auto v28 = tmp3.val[1];
        auto tmp4 = vzipq_s16(vec6.value, vec7.value);
        auto v30 = tmp4.val[0];
        auto v31 = tmp4.val[1];

        auto tmp5 = vzipq_s32(v21, v24);
        vec0.value = tmp5.val[0];
        vec1.value = tmp5.val[1];
        auto tmp6 = vzipq_s32(v22, v25);
        vec2.value = tmp6.val[0];
        vec3.value = tmp6.val[1];
        auto tmp7 = vzipq_s32(v27, v30);
        vec4.value = tmp7.val[0];
        vec5.value = tmp7.val[1];
        auto tmp8 = vzipq_s32(v28, v31);
        vec6.value = tmp8.val[0];
        vec7.value = tmp8.val[1];
        auto v20 = vtrn1q_s64(vec0.value, vec4.value);
        auto v12 = vtrn2q_s64(vec0.value, vec4.value);
        auto v23 = vtrn1q_s64(vec1.value, vec5.value);
        auto v13 = vtrn2q_s64(vec1.value, vec5.value);
        auto v26 = vtrn1q_s64(vec2.value, vec6.value);
        auto v14 = vtrn2q_s64(vec2.value, vec6.value);
        auto v29 = vtrn1q_s64(vec3.value, vec7.value);
        auto v15 = vtrn2q_s64(vec3.value, vec7.value);

        auto tmp9 = vzipq_s16(vec8.value, vec9.value); // tmp9 would disappear after compile
        vec0.value = tmp9.val[0];
        vec1.value = tmp9.val[1];
        auto tmp10 = vzipq_s16(vec10.value, vec11.value);
        vec2.value = tmp10.val[0];
        vec3.value = tmp10.val[1];
        auto tmp11 = vzipq_s32(vec0.value, vec2.value);
        auto v16 = tmp11.val[0];
        auto v17 = tmp11.val[1];
        auto tmp12 = vzipq_s32(vec1.value, vec3.value);
        auto v18 = tmp12.val[0];
        auto v19 = tmp12.val[1];

        v21 = vtrn1q_s64(v16, v12);
        v22 = vtrn2q_s64(v12, v16);
        v24 = vtrn1q_s64(v17, v13);
        v25 = vtrn2q_s64(v13, v17);
        v27 = vtrn1q_s64(v18, v14);
        v28 = vtrn2q_s64(v14, v18);
        v30 = vtrn1q_s64(v19, v15);
        v31 = vtrn2q_s64(v15, v19);

        vec0.value  = v20;
        vec1.value  = v21;
        vec2.value  = v22;
        vec3.value  = v23;
        vec4.value  = v24;
        vec5.value  = v25;
        vec6.value  = v26;
        vec7.value  = v27;
        vec8.value  = v28;
        vec9.value  = v29;
        vec10.value = v30;
        vec11.value = v31;
#else

        auto tmp1 = vzipq_s16(vec0.value, vec1.value); // tmp1 would disappear after compile
        auto v21 = tmp1.val[0];
        auto v22 = tmp1.val[1];
        auto tmp2 = vzipq_s16(vec2.value, vec3.value);
        auto v24 = tmp2.val[0];
        auto v25 = tmp2.val[1];
        auto tmp3 = vzipq_s16(vec4.value, vec5.value);
        auto v27 = tmp3.val[0];
        auto v28 = tmp3.val[1];
        auto tmp4 = vzipq_s16(vec6.value, vec7.value);
        auto v30 = tmp4.val[0];
        auto v31 = tmp4.val[1];

        auto tmp5 = vzipq_s32(v21, v24);
        vec0.value = tmp5.val[0];
        vec1.value = tmp5.val[1];
        auto tmp6 = vzipq_s32(v22, v25);
        vec2.value = tmp6.val[0];
        vec3.value = tmp6.val[1];
        auto tmp7 = vzipq_s32(v27, v30);
        vec4.value = tmp7.val[0];
        vec5.value = tmp7.val[1];
        auto tmp8 = vzipq_s32(v28, v31);
        vec6.value = tmp8.val[0];
        vec7.value = tmp8.val[1];


        auto v20 = vec0.value;
        auto v12 = vec4.value;
        v20 = vsetq_lane_s64(vgetq_lane_s64(vec4.value, 0), v20, 1);
        v12 = vsetq_lane_s64(vgetq_lane_s64(vec0.value, 1), v12, 0);
        auto v23 = vec1.value;
        auto v13 = vec5.value;
        v23 = vsetq_lane_s64(vgetq_lane_s64(vec5.value, 0), v23, 1);
        v13 = vsetq_lane_s64(vgetq_lane_s64(vec1.value, 1), v13, 0);
        auto v26 = vec2.value;
        auto v14 = vec6.value;
        v26 = vsetq_lane_s64(vgetq_lane_s64(vec6.value, 0), v26, 1);
        v14 = vsetq_lane_s64(vgetq_lane_s64(vec2.value, 1), v14, 0);
        auto v29 = vec3.value;
        auto v15 = vec7.value;
        v29 = vsetq_lane_s64(vgetq_lane_s64(vec7.value, 0), v29, 1);
        v15 = vsetq_lane_s64(vgetq_lane_s64(vec3.value, 1), v15, 0);


        auto tmp9 = vzipq_s16(vec8.value, vec9.value); // tmp9 would disappear after compile
        vec0.value = tmp9.val[0];
        vec1.value = tmp9.val[1];
        auto tmp10 = vzipq_s16(vec10.value, vec11.value);
        vec2.value = tmp10.val[0];
        vec3.value = tmp10.val[1];
        auto tmp11 = vzipq_s32(vec0.value, vec2.value);
        auto v16 = tmp11.val[0];
        auto v17 = tmp11.val[1];
        auto tmp12 = vzipq_s32(vec1.value, vec3.value);
        auto v18 = tmp12.val[0];
        auto v19 = tmp12.val[1];

        v21 = v16;
        v22 = v16;
        v21 = vsetq_lane_s64(vgetq_lane_s64(v12, 0), v21, 1);
        v22 = vsetq_lane_s64(vgetq_lane_s64(v12, 1), v22, 0);
        v24 = v17;
        v25 = v17;
        v24 = vsetq_lane_s64(vgetq_lane_s64(v13, 0), v24, 1);
        v25 = vsetq_lane_s64(vgetq_lane_s64(v13, 1), v25, 0);
        v27 = v18;
        v28 = v18;
        v27 = vsetq_lane_s64(vgetq_lane_s64(v14, 0), v27, 1);
        v28 = vsetq_lane_s64(vgetq_lane_s64(v14, 1), v28, 0);
        v30 = v19;
        v31 = v19;
        v30 = vsetq_lane_s64(vgetq_lane_s64(v15, 0), v30, 1);
        v31 = vsetq_lane_s64(vgetq_lane_s64(v15, 1), v31, 0);

        vec0.value  = v20;
        vec1.value  = v21;
        vec2.value  = v22;
        vec3.value  = v23;
        vec4.value  = v24;
        vec5.value  = v25;
        vec6.value  = v26;
        vec7.value  = v27;
        vec8.value  = v28;
        vec9.value  = v29;
        vec10.value = v30;
        vec11.value = v31;

#endif

    }
};
} // namespace Math
} // namespace MNN
#endif /* MNN_USE_NEON */

#endif // Arm82Vec_hpp
#endif
