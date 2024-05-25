//
//  Vec8.hpp
//  MNN
//
//  Created by MNN on 2021/05/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef Vec8_hpp
#define Vec8_hpp
#include "FunctionSummary.hpp"

#define TRANSPOSE_8x8 \
t0 = _mm256_unpacklo_ps(r0, r1);\
t1 = _mm256_unpackhi_ps(r0, r1);\
t2 = _mm256_unpacklo_ps(r2, r3);\
t3 = _mm256_unpackhi_ps(r2, r3);\
t4 = _mm256_unpacklo_ps(r4, r5);\
t5 = _mm256_unpackhi_ps(r4, r5);\
t6 = _mm256_unpacklo_ps(r6, r7);\
t7 = _mm256_unpackhi_ps(r6, r7);\
\
r0 = _mm256_shuffle_ps(t0,t2,_MM_SHUFFLE(1,0,1,0));\
r1 = _mm256_shuffle_ps(t0,t2,_MM_SHUFFLE(3,2,3,2));\
r2 = _mm256_shuffle_ps(t1,t3,_MM_SHUFFLE(1,0,1,0));\
r3 = _mm256_shuffle_ps(t1,t3,_MM_SHUFFLE(3,2,3,2));\
r4 = _mm256_shuffle_ps(t4,t6,_MM_SHUFFLE(1,0,1,0));\
r5 = _mm256_shuffle_ps(t4,t6,_MM_SHUFFLE(3,2,3,2));\
r6 = _mm256_shuffle_ps(t5,t7,_MM_SHUFFLE(1,0,1,0));\
r7 = _mm256_shuffle_ps(t5,t7,_MM_SHUFFLE(3,2,3,2));\
\
t0 = _mm256_permute2f128_ps(r0, r4, 0x20);\
t1 = _mm256_permute2f128_ps(r1, r5, 0x20);\
t2 = _mm256_permute2f128_ps(r2, r6, 0x20);\
t3 = _mm256_permute2f128_ps(r3, r7, 0x20);\
t4 = _mm256_permute2f128_ps(r0, r4, 0x31);\
t5 = _mm256_permute2f128_ps(r1, r5, 0x31);\
t6 = _mm256_permute2f128_ps(r2, r6, 0x31);\
t7 = _mm256_permute2f128_ps(r3, r7, 0x31);\

#define TRANSPOSE_8x8_REPLACE(r0, r1, r2, r3, r4, r5, r6, r7) \
{\
auto t0 = _mm256_unpacklo_ps(r0, r1);\
auto t1 = _mm256_unpackhi_ps(r0, r1);\
auto t2 = _mm256_unpacklo_ps(r2, r3);\
auto t3 = _mm256_unpackhi_ps(r2, r3);\
auto t4 = _mm256_unpacklo_ps(r4, r5);\
auto t5 = _mm256_unpackhi_ps(r4, r5);\
auto t6 = _mm256_unpacklo_ps(r6, r7);\
auto t7 = _mm256_unpackhi_ps(r6, r7);\
\
r0 = _mm256_shuffle_ps(t0,t2,_MM_SHUFFLE(1,0,1,0));\
r1 = _mm256_shuffle_ps(t0,t2,_MM_SHUFFLE(3,2,3,2));\
r2 = _mm256_shuffle_ps(t1,t3,_MM_SHUFFLE(1,0,1,0));\
r3 = _mm256_shuffle_ps(t1,t3,_MM_SHUFFLE(3,2,3,2));\
r4 = _mm256_shuffle_ps(t4,t6,_MM_SHUFFLE(1,0,1,0));\
r5 = _mm256_shuffle_ps(t4,t6,_MM_SHUFFLE(3,2,3,2));\
r6 = _mm256_shuffle_ps(t5,t7,_MM_SHUFFLE(1,0,1,0));\
r7 = _mm256_shuffle_ps(t5,t7,_MM_SHUFFLE(3,2,3,2));\
\
t0 = _mm256_permute2f128_ps(r0, r4, 0x20);\
t1 = _mm256_permute2f128_ps(r1, r5, 0x20);\
t2 = _mm256_permute2f128_ps(r2, r6, 0x20);\
t3 = _mm256_permute2f128_ps(r3, r7, 0x20);\
t4 = _mm256_permute2f128_ps(r0, r4, 0x31);\
t5 = _mm256_permute2f128_ps(r1, r5, 0x31);\
t6 = _mm256_permute2f128_ps(r2, r6, 0x31);\
t7 = _mm256_permute2f128_ps(r3, r7, 0x31);\
r0 = t0, r1 = t1, r2 = t2, r3 = t3;\
r4 = t4, r5 = t5, r6 = t6, r7 = t7;\
}\

struct Vec8 {
    using VecType = Vec8;
    __m256 value;
    
    VecType operator+(const VecType& lr) const {
        VecType dst = { _mm256_add_ps(value, lr.value) };
        return dst;
    }
    VecType operator-(const VecType& lr) const {
        VecType dst = { _mm256_sub_ps(value, lr.value) };
        return dst;
    }
    VecType operator+=(const VecType& lr) {
        value = _mm256_add_ps(value, lr.value);
        return *this;
    }
    VecType operator-=(const VecType& lr) {
        value = _mm256_sub_ps(value, lr.value);
        return *this;
    }
    VecType operator*(const VecType& lr) const {
        VecType dst = { _mm256_mul_ps(value, lr.value) };
        return dst;
    }
    VecType operator*(float lr) {
        VecType dst = { _mm256_mul_ps(value, _mm256_set1_ps(lr)) };
        return dst;
    }

    VecType& operator=(const VecType& lr) {
        value = lr.value;
        return *this;
    }
    VecType operator==(const VecType& lr) const {
        __m256 one = _mm256_set1_ps(1.0f);
        __m256 mask = _mm256_cmp_ps(value, lr.value, 0);
        VecType dst =  { _mm256_and_ps(one, mask) } ;
        return dst;
    }
    VecType operator>(const VecType& lr) {
        __m256 one = _mm256_set1_ps(1.0f);
        __m256 mask = _mm256_cmp_ps(value, lr.value, 14);
        VecType dst =  { _mm256_and_ps(one, mask) } ;
        return dst;
    }
    VecType operator>=(const VecType& lr) {
        __m256 one = _mm256_set1_ps(1.0f);
        __m256 mask = _mm256_cmp_ps(value, lr.value, 0x0D);
        VecType dst =  { _mm256_and_ps(one, mask) } ;
        return dst;
    }
    VecType operator<(const VecType& lr) {
        __m256 one = _mm256_set1_ps(1.0f);
        __m256 mask = _mm256_cmp_ps(value, lr.value, 0x01);
        VecType dst =  { _mm256_and_ps(one, mask) } ;
        return dst;
    }
    VecType operator<=(const VecType& lr) {
        __m256 one = _mm256_set1_ps(1.0f);
        __m256 mask = _mm256_cmp_ps(value, lr.value, 0x02);
        VecType dst =  { _mm256_and_ps(one, mask) } ;
        return dst;
    }
    VecType operator-() {
        VecType dst;
#if defined(_MSC_VER)
        dst.value = _mm256_xor_ps(value, _mm256_set1_ps(-0.f)); // Using unary operation to SSE vec is GCC extension. We can not do this directly in MSVC.
#else
        dst.value = -value;
#endif
        return dst;
    }
    Vec8() {
    }
    Vec8(const float v) {
        value = _mm256_set1_ps(v);
    }
    Vec8(__m256&& v) {
        value = v;
    }
    Vec8(const VecType& lr) {
        value = lr.value;
    }
    float operator[](size_t i) {
#if defined(_MSC_VER)  // X64 native only mandatory support SSE and SSE2 extension, and we can not find intrinsic function to extract element directly by index in SSE and SSE2 extension.
        float temp[8];
        _mm256_storeu_ps(temp, value);
        return temp[i];
#else
        return value[i];
#endif
    }
    static VecType load(const float* addr) {
        VecType v = { _mm256_loadu_ps(addr) };
        return v;
    }
    static VecType broadcast(const float* addr) {
        VecType dst = { _mm256_broadcast_ss(addr) }; // compiled into 'vbroadcastss'
        return dst;
    }
    static void save(float* addr, const VecType& v) {
        _mm256_storeu_ps(addr, v.value);
    }
    static void save(int32_t* addr, const VecType& v) {
        _mm256_storeu_ps((float*)addr, v.value);
    }
    static VecType max(const VecType& v1, const VecType& v2) {
        VecType dst = { _mm256_max_ps(v1.value, v2.value) };
        return dst;
    }
    static VecType min(const VecType& v1, const VecType& v2) {
        VecType dst = { _mm256_min_ps(v1.value, v2.value) };
        return dst;
    }
    static VecType fma(const VecType& v0, const VecType& v1, const VecType& v2) {
        VecType dst = v0 + v1 * v2;// not sure support fmadd or not, { _mm256_fmadd_ps(v1.value, v2.value, v0.value) };
        return dst;
    }
    static VecType fms(const VecType& v0, const VecType& v1, const VecType& v2) {
        VecType dst = v0 - v1 * v2; // not sure support fnmadd or not, { _mm256_fnmadd_ps(v1.value, v2.value, v0.value) };
        return dst;
    }
    static void transpose8(VecType& v0, VecType& v1, VecType& v2, VecType& v3, VecType& v4, VecType& v5, VecType& v6, VecType& v7) {
        TRANSPOSE_8x8_REPLACE(v0.value, v1.value, v2.value, v3.value, v4.value, v5.value, v6.value, v7.value);
    }
};

#endif
