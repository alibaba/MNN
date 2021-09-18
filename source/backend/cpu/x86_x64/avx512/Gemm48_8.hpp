//
//  Gemm48_8.hpp
//  MNN
//
//  Created by MNN on 2021/05/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef __AVX512_GEMM48_8_HPP__
#define __AVX512_GEMM48_8_HPP__

#define MNNAVXFMA _mm256_fmadd_ps
#define MNNAVX512FMA _mm512_fmadd_ps

#define INIT_MAIN_4_8                                           \
        auto s0 = _mm256_loadu_ps(weight + 0 * 8);              \
        auto w0 = _mm256_broadcast_ss(A + 0 * aStride + 0);     \
        auto w1 = _mm256_broadcast_ss(A + 0 * aStride + 1);     \
        auto w2 = _mm256_broadcast_ss(A + 0 * aStride + 2);     \
        auto w3 = _mm256_broadcast_ss(A + 0 * aStride + 3);     \
        auto z0 = _mm256_mul_ps(s0, w0);                        \
        auto z1 = _mm256_mul_ps(s0, w1);                        \
        auto z2 = _mm256_mul_ps(s0, w2);                        \
        auto z3 = _mm256_mul_ps(s0, w3);                        \

#define COMPUTE_4_8                                             \
        s0 = _mm256_loadu_ps(weight + sy * 8);                  \
        w0 = _mm256_broadcast_ss(A + sy * aStride + 0);         \
        w1 = _mm256_broadcast_ss(A + sy * aStride + 1);         \
        w2 = _mm256_broadcast_ss(A + sy * aStride + 2);         \
        w3 = _mm256_broadcast_ss(A + sy * aStride + 3);         \
        z0 = MNNAVXFMA(s0, w0, z0);                             \
        z1 = MNNAVXFMA(s0, w1, z1);                             \
        z2 = MNNAVXFMA(s0, w2, z2);                             \
        z3 = MNNAVXFMA(s0, w3, z3);                             \

#define INIT_MAIN_5_8                                           \
        auto s0 = _mm256_loadu_ps(weight + 0 * 8);              \
        auto w0 = _mm256_broadcast_ss(A + 0 * aStride + 0);     \
        auto w1 = _mm256_broadcast_ss(A + 0 * aStride + 1);     \
        auto w2 = _mm256_broadcast_ss(A + 0 * aStride + 2);     \
        auto w3 = _mm256_broadcast_ss(A + 0 * aStride + 3);     \
        auto w4 = _mm256_broadcast_ss(A + 0 * aStride + 4);     \
        auto z0 = _mm256_mul_ps(s0, w0);                        \
        auto z1 = _mm256_mul_ps(s0, w1);                        \
        auto z2 = _mm256_mul_ps(s0, w2);                        \
        auto z3 = _mm256_mul_ps(s0, w3);                        \
        auto z4 = _mm256_mul_ps(s0, w4);                        \

#define COMPUTE_5_8                                             \
        s0 = _mm256_loadu_ps(weight + sy * 8);                  \
        w0 = _mm256_broadcast_ss(A + sy * aStride + 0);         \
        w1 = _mm256_broadcast_ss(A + sy * aStride + 1);         \
        w2 = _mm256_broadcast_ss(A + sy * aStride + 2);         \
        w3 = _mm256_broadcast_ss(A + sy * aStride + 3);         \
        w4 = _mm256_broadcast_ss(A + sy * aStride + 4);         \
        z0 = MNNAVXFMA(s0, w0, z0);                             \
        z1 = MNNAVXFMA(s0, w1, z1);                             \
        z2 = MNNAVXFMA(s0, w2, z2);                             \
        z3 = MNNAVXFMA(s0, w3, z3);                             \
        z4 = MNNAVXFMA(s0, w4, z4);                             \

#define INIT_MAIN_8_8                                           \
    auto s0 = _mm256_loadu_ps(A + 0 * aStride);                 \
    auto w0 = _mm256_broadcast_ss(weight + 0 * 8 + 0);          \
    auto w1 = _mm256_broadcast_ss(weight + 0 * 8 + 1);          \
    auto w2 = _mm256_broadcast_ss(weight + 0 * 8 + 2);          \
    auto w3 = _mm256_broadcast_ss(weight + 0 * 8 + 3);          \
    auto w4 = _mm256_broadcast_ss(weight + 0 * 8 + 4);          \
    auto w5 = _mm256_broadcast_ss(weight + 0 * 8 + 5);          \
    auto w6 = _mm256_broadcast_ss(weight + 0 * 8 + 6);          \
    auto w7 = _mm256_broadcast_ss(weight + 0 * 8 + 7);          \
    auto z0 = _mm256_mul_ps(s0, w0);                            \
    auto z1 = _mm256_mul_ps(s0, w1);                            \
    auto z2 = _mm256_mul_ps(s0, w2);                            \
    auto z3 = _mm256_mul_ps(s0, w3);                            \
    auto z4 = _mm256_mul_ps(s0, w4);                            \
    auto z5 = _mm256_mul_ps(s0, w5);                            \
    auto z6 = _mm256_mul_ps(s0, w6);                            \
    auto z7 = _mm256_mul_ps(s0, w7);

#define COMPUTE_8_8                                             \
    s0 = _mm256_loadu_ps(A + sy * aStride);                     \
    w0 = _mm256_broadcast_ss(weight + sy * 8 + 0);              \
    w1 = _mm256_broadcast_ss(weight + sy * 8 + 1);              \
    w2 = _mm256_broadcast_ss(weight + sy * 8 + 2);              \
    w3 = _mm256_broadcast_ss(weight + sy * 8 + 3);              \
    w4 = _mm256_broadcast_ss(weight + sy * 8 + 4);              \
    w5 = _mm256_broadcast_ss(weight + sy * 8 + 5);              \
    w6 = _mm256_broadcast_ss(weight + sy * 8 + 6);              \
    w7 = _mm256_broadcast_ss(weight + sy * 8 + 7);              \
    z0 = MNNAVXFMA(s0, w0, z0);                                 \
    z1 = MNNAVXFMA(s0, w1, z1);                                 \
    z2 = MNNAVXFMA(s0, w2, z2);                                 \
    z3 = MNNAVXFMA(s0, w3, z3);                                 \
    z4 = MNNAVXFMA(s0, w4, z4);                                 \
    z5 = MNNAVXFMA(s0, w5, z5);                                 \
    z6 = MNNAVXFMA(s0, w6, z6);                                 \
    z7 = MNNAVXFMA(s0, w7, z7);

#define INIT_MAIN_16_8                                  \
    auto s0  = _mm512_loadu_ps(A + 0 * aStride);        \
    auto wt  = _mm_load_ss(weight + 0 * 8 + 0);         \
    auto w0  = _mm512_broadcastss_ps(wt);               \
    auto z0  = _mm512_mul_ps(s0, w0);                   \
    wt = _mm_load_ss(weight + 0 * 8 + 1);               \
    auto w1  = _mm512_broadcastss_ps(wt);               \
    auto z1  = _mm512_mul_ps(s0, w1);                   \
    wt = _mm_load_ss(weight + 0 * 8 + 2);               \
    auto w2  = _mm512_broadcastss_ps(wt);               \
    auto z2  = _mm512_mul_ps(s0, w2);                   \
    wt = _mm_load_ss(weight + 0 * 8 + 3);               \
    auto w3  = _mm512_broadcastss_ps(wt);               \
    auto z3  = _mm512_mul_ps(s0, w3);                   \
    wt = _mm_load_ss(weight + 0 * 8 + 4);               \
    auto w4  = _mm512_broadcastss_ps(wt);               \
    auto z4  = _mm512_mul_ps(s0, w4);                   \
    wt = _mm_load_ss(weight + 0 * 8 + 5);               \
    auto w5  = _mm512_broadcastss_ps(wt);               \
    auto z5  = _mm512_mul_ps(s0, w5);                   \
    wt = _mm_load_ss(weight + 0 * 8 + 6);               \
    auto w6  = _mm512_broadcastss_ps(wt);               \
    auto z6  = _mm512_mul_ps(s0, w6);                   \
    wt = _mm_load_ss(weight + 0 * 8 + 7);               \
    auto w7  = _mm512_broadcastss_ps(wt);               \
    auto z7  = _mm512_mul_ps(s0, w7);

#define COMPUTE_16_8                                \
    s0  = _mm512_loadu_ps(A + sy * aStride);        \
    wt  = _mm_load_ss(weight + sy * 8 + 0);         \
    w0  = _mm512_broadcastss_ps(wt);                \
    z0  = MNNAVX512FMA(s0, w0, z0);                 \
    wt  = _mm_load_ss(weight + sy * 8 + 1);         \
    w1  = _mm512_broadcastss_ps(wt);                \
    z1  = MNNAVX512FMA(s0, w1, z1);                 \
    wt  = _mm_load_ss(weight + sy * 8 + 2);         \
    w2  = _mm512_broadcastss_ps(wt);                \
    z2  = MNNAVX512FMA(s0, w2, z2);                 \
    wt  = _mm_load_ss(weight + sy * 8 + 3);         \
    w3  = _mm512_broadcastss_ps(wt);                \
    z3  = MNNAVX512FMA(s0, w3, z3);                 \
    wt  = _mm_load_ss(weight + sy * 8 + 4);         \
    w4  = _mm512_broadcastss_ps(wt);                \
    z4  = MNNAVX512FMA(s0, w4, z4);                 \
    wt  = _mm_load_ss(weight + sy * 8 + 5);         \
    w5  = _mm512_broadcastss_ps(wt);                \
    z5  = MNNAVX512FMA(s0, w5, z5);                 \
    wt  = _mm_load_ss(weight + sy * 8 + 6);         \
    w6  = _mm512_broadcastss_ps(wt);                \
    z6  = MNNAVX512FMA(s0, w6, z6);                 \
    wt  = _mm_load_ss(weight + sy * 8 + 7);         \
    w7  = _mm512_broadcastss_ps(wt);                \
    z7  = MNNAVX512FMA(s0, w7, z7);

#define INIT_MAIN_32_8                                  \
    auto s0  = _mm512_loadu_ps(A + 0 * aStride);             \
    auto s1  = _mm512_loadu_ps(A + 0 * aStride + 16);        \
    auto wt  = _mm_load_ss(weight + 0 * 8 + 0);         \
    auto w0  = _mm512_broadcastss_ps(wt);               \
    auto z0  = _mm512_mul_ps(s0, w0);                   \
    auto z1  = _mm512_mul_ps(s1, w0);                   \
    wt = _mm_load_ss(weight + 0 * 8 + 1);               \
    auto w1  = _mm512_broadcastss_ps(wt);               \
    auto z3  = _mm512_mul_ps(s0, w1);                   \
    auto z4  = _mm512_mul_ps(s1, w1);                   \
    wt = _mm_load_ss(weight + 0 * 8 + 2);               \
    auto w2  = _mm512_broadcastss_ps(wt);               \
    auto z6  = _mm512_mul_ps(s0, w2);                   \
    auto z7  = _mm512_mul_ps(s1, w2);                   \
    wt = _mm_load_ss(weight + 0 * 8 + 3);               \
    auto w3  = _mm512_broadcastss_ps(wt);               \
    auto z9  = _mm512_mul_ps(s0, w3);                   \
    auto z10 = _mm512_mul_ps(s1, w3);                   \
    wt = _mm_load_ss(weight + 0 * 8 + 4);               \
    auto w4  = _mm512_broadcastss_ps(wt);               \
    auto z12 = _mm512_mul_ps(s0, w4);                   \
    auto z13 = _mm512_mul_ps(s1, w4);                   \
    wt = _mm_load_ss(weight + 0 * 8 + 5);               \
    auto w5  = _mm512_broadcastss_ps(wt);               \
    auto z15 = _mm512_mul_ps(s0, w5);                   \
    auto z16 = _mm512_mul_ps(s1, w5);                   \
    wt = _mm_load_ss(weight + 0 * 8 + 6);               \
    auto w6  = _mm512_broadcastss_ps(wt);               \
    auto z18 = _mm512_mul_ps(s0, w6);                   \
    auto z19 = _mm512_mul_ps(s1, w6);                   \
    wt = _mm_load_ss(weight + 0 * 8 + 7);               \
    auto w7  = _mm512_broadcastss_ps(wt);               \
    auto z21 = _mm512_mul_ps(s0, w7);                   \
    auto z22 = _mm512_mul_ps(s1, w7);                   \

#define COMPUTE_32_8                                \
    s0  = _mm512_loadu_ps(A + sy * aStride);             \
    s1  = _mm512_loadu_ps(A + sy * aStride + 16);        \
    wt  = _mm_load_ss(weight + sy * 8 + 0);         \
    w0  = _mm512_broadcastss_ps(wt);                \
    z0  = MNNAVX512FMA(s0, w0, z0);                 \
    z1  = MNNAVX512FMA(s1, w0, z1);                 \
    wt  = _mm_load_ss(weight + sy * 8 + 1);         \
    w1  = _mm512_broadcastss_ps(wt);                \
    z3  = MNNAVX512FMA(s0, w1, z3);                 \
    z4  = MNNAVX512FMA(s1, w1, z4);                 \
    wt  = _mm_load_ss(weight + sy * 8 + 2);         \
    w2  = _mm512_broadcastss_ps(wt);                \
    z6  = MNNAVX512FMA(s0, w2, z6);                 \
    z7  = MNNAVX512FMA(s1, w2, z7);                 \
    wt  = _mm_load_ss(weight + sy * 8 + 3);         \
    w3  = _mm512_broadcastss_ps(wt);                \
    z9  = MNNAVX512FMA(s0, w3, z9);                 \
    z10  = MNNAVX512FMA(s1, w3, z10);               \
    wt  = _mm_load_ss(weight + sy * 8 + 4);         \
    w4  = _mm512_broadcastss_ps(wt);                \
    z12  = MNNAVX512FMA(s0, w4, z12);               \
    z13  = MNNAVX512FMA(s1, w4, z13);               \
    wt  = _mm_load_ss(weight + sy * 8 + 5);         \
    w5  = _mm512_broadcastss_ps(wt);                \
    z15  = MNNAVX512FMA(s0, w5, z15);               \
    z16  = MNNAVX512FMA(s1, w5, z16);               \
    wt  = _mm_load_ss(weight + sy * 8 + 6);         \
    w6  = _mm512_broadcastss_ps(wt);                \
    z18  = MNNAVX512FMA(s0, w6, z18);               \
    z19  = MNNAVX512FMA(s1, w6, z19);               \
    wt  = _mm_load_ss(weight + sy * 8 + 7);         \
    w7  = _mm512_broadcastss_ps(wt);                \
    z21  = MNNAVX512FMA(s0, w7, z21);               \
    z22  = MNNAVX512FMA(s1, w7, z22);               \

#define INIT_MAIN_48_8                                  \
    auto s0  = _mm512_loadu_ps(A + 0 * 48);             \
    auto s1  = _mm512_loadu_ps(A + 0 * 48 + 16);        \
    auto s2  = _mm512_loadu_ps(A + 0 * 48 + 32);        \
    auto wt  = _mm_load_ss(weight + 0 * 8 + 0);         \
    auto w0  = _mm512_broadcastss_ps(wt);               \
    auto z0  = _mm512_mul_ps(s0, w0);                   \
    auto z1  = _mm512_mul_ps(s1, w0);                   \
    auto z2  = _mm512_mul_ps(s2, w0);                   \
    wt = _mm_load_ss(weight + 0 * 8 + 1);               \
    auto w1  = _mm512_broadcastss_ps(wt);               \
    auto z3  = _mm512_mul_ps(s0, w1);                   \
    auto z4  = _mm512_mul_ps(s1, w1);                   \
    auto z5  = _mm512_mul_ps(s2, w1);                   \
    wt = _mm_load_ss(weight + 0 * 8 + 2);               \
    auto w2  = _mm512_broadcastss_ps(wt);               \
    auto z6  = _mm512_mul_ps(s0, w2);                   \
    auto z7  = _mm512_mul_ps(s1, w2);                   \
    auto z8  = _mm512_mul_ps(s2, w2);                   \
    wt = _mm_load_ss(weight + 0 * 8 + 3);               \
    auto w3  = _mm512_broadcastss_ps(wt);               \
    auto z9  = _mm512_mul_ps(s0, w3);                   \
    auto z10 = _mm512_mul_ps(s1, w3);                   \
    auto z11 = _mm512_mul_ps(s2, w3);                   \
    wt = _mm_load_ss(weight + 0 * 8 + 4);               \
    auto w4  = _mm512_broadcastss_ps(wt);               \
    auto z12 = _mm512_mul_ps(s0, w4);                   \
    auto z13 = _mm512_mul_ps(s1, w4);                   \
    auto z14 = _mm512_mul_ps(s2, w4);                   \
    wt = _mm_load_ss(weight + 0 * 8 + 5);               \
    auto w5  = _mm512_broadcastss_ps(wt);               \
    auto z15 = _mm512_mul_ps(s0, w5);                   \
    auto z16 = _mm512_mul_ps(s1, w5);                   \
    auto z17 = _mm512_mul_ps(s2, w5);                   \
    wt = _mm_load_ss(weight + 0 * 8 + 6);               \
    auto w6  = _mm512_broadcastss_ps(wt);               \
    auto z18 = _mm512_mul_ps(s0, w6);                   \
    auto z19 = _mm512_mul_ps(s1, w6);                   \
    auto z20 = _mm512_mul_ps(s2, w6);                   \
    wt = _mm_load_ss(weight + 0 * 8 + 7);               \
    auto w7  = _mm512_broadcastss_ps(wt);               \
    auto z21 = _mm512_mul_ps(s0, w7);                   \
    auto z22 = _mm512_mul_ps(s1, w7);                   \
    auto z23 = _mm512_mul_ps(s2, w7);

#define COMPUTE_48_8                                \
    s0  = _mm512_loadu_ps(A + sy * 48);             \
    s1  = _mm512_loadu_ps(A + sy * 48 + 16);        \
    s2  = _mm512_loadu_ps(A + sy * 48 + 32);        \
    wt  = _mm_load_ss(weight + sy * 8 + 0);         \
    w0  = _mm512_broadcastss_ps(wt);                \
    z0  = MNNAVX512FMA(s0, w0, z0);                 \
    z1  = MNNAVX512FMA(s1, w0, z1);                 \
    z2  = MNNAVX512FMA(s2, w0, z2);                 \
    wt  = _mm_load_ss(weight + sy * 8 + 1);         \
    w1  = _mm512_broadcastss_ps(wt);                \
    z3  = MNNAVX512FMA(s0, w1, z3);                 \
    z4  = MNNAVX512FMA(s1, w1, z4);                 \
    z5  = MNNAVX512FMA(s2, w1, z5);                 \
    wt  = _mm_load_ss(weight + sy * 8 + 2);         \
    w2  = _mm512_broadcastss_ps(wt);                \
    z6  = MNNAVX512FMA(s0, w2, z6);                 \
    z7  = MNNAVX512FMA(s1, w2, z7);                 \
    z8  = MNNAVX512FMA(s2, w2, z8);                 \
    wt  = _mm_load_ss(weight + sy * 8 + 3);         \
    w3  = _mm512_broadcastss_ps(wt);                \
    z9  = MNNAVX512FMA(s0, w3, z9);                 \
    z10  = MNNAVX512FMA(s1, w3, z10);               \
    z11  = MNNAVX512FMA(s2, w3, z11);               \
    wt  = _mm_load_ss(weight + sy * 8 + 4);         \
    w4  = _mm512_broadcastss_ps(wt);                \
    z12  = MNNAVX512FMA(s0, w4, z12);               \
    z13  = MNNAVX512FMA(s1, w4, z13);               \
    z14  = MNNAVX512FMA(s2, w4, z14);               \
    wt  = _mm_load_ss(weight + sy * 8 + 5);         \
    w5  = _mm512_broadcastss_ps(wt);                \
    z15  = MNNAVX512FMA(s0, w5, z15);               \
    z16  = MNNAVX512FMA(s1, w5, z16);               \
    z17  = MNNAVX512FMA(s2, w5, z17);               \
    wt  = _mm_load_ss(weight + sy * 8 + 6);         \
    w6  = _mm512_broadcastss_ps(wt);                \
    z18  = MNNAVX512FMA(s0, w6, z18);               \
    z19  = MNNAVX512FMA(s1, w6, z19);               \
    z20  = MNNAVX512FMA(s2, w6, z20);               \
    wt  = _mm_load_ss(weight + sy * 8 + 7);         \
    w7  = _mm512_broadcastss_ps(wt);                \
    z21  = MNNAVX512FMA(s0, w7, z21);               \
    z22  = MNNAVX512FMA(s1, w7, z22);               \
    z23  = MNNAVX512FMA(s2, w7, z23);

#define AVX512_TRANSPOSE_SAVE(u, v, z0, z3, z6, z9, z12, z15, z18, z21) \
    {                                                                   \
        auto m0 = _mm512_extractf32x4_ps(z0, u);                        \
        auto m1 = _mm512_extractf32x4_ps(z3, u);                        \
        auto m2 = _mm512_extractf32x4_ps(z6, u);                        \
        auto m3 = _mm512_extractf32x4_ps(z9, u);                        \
        auto m4 = _mm512_extractf32x4_ps(z12, u);                       \
        auto m5 = _mm512_extractf32x4_ps(z15, u);                       \
        auto m6 = _mm512_extractf32x4_ps(z18, u);                       \
        auto m7 = _mm512_extractf32x4_ps(z21, u);                       \
        _MM_TRANSPOSE4_PS(m0, m1, m2, m3);                              \
        _MM_TRANSPOSE4_PS(m4, m5, m6, m7);                              \
        auto tmp0 = _mm256_castps128_ps256(m0);                         \
        auto tmp4 = _mm256_castps128_ps256(m4);                         \
        auto s0 = _mm256_permute2f128_ps(tmp0, tmp4, 0x20);             \
        auto tmp1 = _mm256_castps128_ps256(m1);                         \
        auto tmp5 = _mm256_castps128_ps256(m5);                         \
        auto s1 = _mm256_permute2f128_ps(tmp1, tmp5, 0x20);             \
        auto tmp2 = _mm256_castps128_ps256(m2);                         \
        auto tmp6 = _mm256_castps128_ps256(m6);                         \
        auto s2 = _mm256_permute2f128_ps(tmp2, tmp6, 0x20);             \
        auto tmp3 = _mm256_castps128_ps256(m3);                         \
        auto tmp7 = _mm256_castps128_ps256(m7);                         \
        auto s3 = _mm256_permute2f128_ps(tmp3, tmp7, 0x20);             \
        _mm256_storeu_ps(dst + 256 * v + 64 * u, s0);                   \
        _mm256_storeu_ps(dst + 256 * v + 64 * u + 16, s1);               \
        _mm256_storeu_ps(dst + 256 * v + 64 * u + 32, s2);              \
        _mm256_storeu_ps(dst + 256 * v + 64 * u + 48, s3);              \
    }

#define AVX512_TRANSPOSE_SAVE_HALF(u, v, z0, z3, z6, z9, z12, z15, z18, z21) \
    {                                                                        \
        auto m0 = _mm512_extractf32x4_ps(z0, u);                             \
        auto m1 = _mm512_extractf32x4_ps(z3, u);                             \
        auto m2 = _mm512_extractf32x4_ps(z6, u);                             \
        auto m3 = _mm512_extractf32x4_ps(z9, u);                             \
        _MM_TRANSPOSE4_PS(m0, m1, m2, m3);                                   \
        auto tmp0 = _mm256_castps128_ps256(m0);                              \
        auto tmp1 = _mm256_castps128_ps256(m1);                              \
        auto s0 = _mm256_permute2f128_ps(tmp0, tmp1, 0x20);                  \
        auto tmp2 = _mm256_castps128_ps256(m2);                              \
        auto tmp3 = _mm256_castps128_ps256(m3);                              \
        auto s1 = _mm256_permute2f128_ps(tmp2, tmp3, 0x20);                  \
        _mm256_storeu_ps((dst + 256 * v + 64 * u), s0);                      \
        _mm256_storeu_ps((dst + 256 * v + 64 * u + 16), s1);                  \
    }

#define AVX2_TRANSPOSE_SAVE(u, z0, z3, z6, z9, z12, z15, z18, z21)   \
    {                                                                \
        auto m0 = _mm256_extractf128_ps(z0, u);                      \
        auto m1 = _mm256_extractf128_ps(z3, u);                      \
        auto m2 = _mm256_extractf128_ps(z6, u);                      \
        auto m3 = _mm256_extractf128_ps(z9, u);                      \
        auto m4 = _mm256_extractf128_ps(z12, u);                     \
        auto m5 = _mm256_extractf128_ps(z15, u);                     \
        auto m6 = _mm256_extractf128_ps(z18, u);                     \
        auto m7 = _mm256_extractf128_ps(z21, u);                     \
        _MM_TRANSPOSE4_PS(m0, m1, m2, m3);                           \
        _MM_TRANSPOSE4_PS(m4, m5, m6, m7);                           \
        auto tmp0 = _mm256_castps128_ps256(m0);                      \
        auto tmp4 = _mm256_castps128_ps256(m4);                      \
        auto s0 = _mm256_permute2f128_ps(tmp0, tmp4, 0x20);          \
        auto tmp1 = _mm256_castps128_ps256(m1);                      \
        auto tmp5 = _mm256_castps128_ps256(m5);                      \
        auto s1 = _mm256_permute2f128_ps(tmp1, tmp5, 0x20);          \
        auto tmp2 = _mm256_castps128_ps256(m2);                      \
        auto tmp6 = _mm256_castps128_ps256(m6);                      \
        auto s2 = _mm256_permute2f128_ps(tmp2, tmp6, 0x20);          \
        auto tmp3 = _mm256_castps128_ps256(m3);                      \
        auto tmp7 = _mm256_castps128_ps256(m7);                      \
        auto s3 = _mm256_permute2f128_ps(tmp3, tmp7, 0x20);          \
        _mm256_storeu_ps(dst + 64 * u, s0);                          \
        _mm256_storeu_ps(dst + 64 * u + 16, s1);                      \
        _mm256_storeu_ps(dst + 64 * u + 32, s2);                     \
        _mm256_storeu_ps(dst + 64 * u + 48, s3);                     \
    }

#define AVX2_TRANSPOSE_SAVE_HALF(u, z0, z3, z6, z9, z12, z15, z18, z21)     \
    {                                                                       \
        auto m0 = _mm256_extractf128_ps(z0, u);                             \
        auto m1 = _mm256_extractf128_ps(z3, u);                             \
        auto m2 = _mm256_extractf128_ps(z6, u);                             \
        auto m3 = _mm256_extractf128_ps(z9, u);                             \
        _MM_TRANSPOSE4_PS(m0, m1, m2, m3);                                  \
        auto tmp0 = _mm256_castps128_ps256(m0);                             \
        auto tmp1 = _mm256_castps128_ps256(m1);                             \
        auto s0 = _mm256_permute2f128_ps(tmp0, tmp1, 0x20);                 \
        auto tmp2 = _mm256_castps128_ps256(m2);                             \
        auto tmp3 = _mm256_castps128_ps256(m3);                             \
        auto s1 = _mm256_permute2f128_ps(tmp2, tmp3, 0x20);                 \
        _mm256_storeu_ps(dst + 16 * (0 + 4 * u), s0);                        \
        _mm256_storeu_ps(dst + 16 * (1 + 4 * u), s1);                        \
    }

#endif
