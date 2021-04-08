//
//  GemmFunction.hpp
//  MNN
//
//  Created by MNN on b'2021/03/01'.
//  Copyright © 2018, Alibaba Group Holding Limited

#define LOAD16(x) _mm512_loadu_ps(x)
#define LOAD4_16(x) _mm512_broadcast_f32x4(_mm_loadu_ps(x))

#define _AVX512_HADD_SAVE(u, z0, z1, z2, z3) \
    {                                            \
        auto z0h = _mm512_extractf32x8_ps(z0, 0); \
        auto z0l = _mm512_extractf32x8_ps(z0, 1); \
        auto z1h = _mm512_extractf32x8_ps(z1, 0); \
        auto z1l = _mm512_extractf32x8_ps(z1, 1); \
        auto z2h = _mm512_extractf32x8_ps(z2, 0); \
        auto z2l = _mm512_extractf32x8_ps(z2, 1); \
        auto z3h = _mm512_extractf32x8_ps(z3, 0); \
        auto z3l = _mm512_extractf32x8_ps(z3, 1); \
        auto tmp0 = _mm256_hadd_ps(z0h, z1h);  \
        auto tmp1 = _mm256_hadd_ps(z2h, z3h);  \
        auto sum0 = _mm256_hadd_ps(tmp0, tmp1); \
        auto tmp2 = _mm256_hadd_ps(z0l, z1l);  \
        auto tmp3 = _mm256_hadd_ps(z2l, z3l);  \
        auto sum1 = _mm256_hadd_ps(tmp2, tmp3);  \
        _mm256_storeu_ps(dst + u * 16, sum0); \
        _mm256_storeu_ps(dst + u * 16 + 8, sum1);  \
    }

#define MUL_BLOCK(u) \
    auto z##u##0 = _mm512_mul_ps(a##u, w0); \
    auto z##u##1 = _mm512_mul_ps(a##u, w1); \
    auto z##u##2 = _mm512_mul_ps(a##u, w2); \
    auto z##u##3 = _mm512_mul_ps(a##u, w3);

#define FMADD_BLOCK(u) \
    z##u##0 = _mm512_fmadd_ps(a##u, w0, z##u##0); \
    z##u##1 = _mm512_fmadd_ps(a##u, w1, z##u##1); \
    z##u##2 = _mm512_fmadd_ps(a##u, w2, z##u##2); \
    z##u##3 = _mm512_fmadd_ps(a##u, w3, z##u##3);

#define INIT_WEIGHTS \
    auto w0 = LOAD4_16(weight);         \
    auto w1 = LOAD4_16(weight + 4); \
    auto w2 = LOAD4_16(weight + 8);         \
    auto w3 = LOAD4_16(weight + 12); \

#define LOAD_WEIGHTS(sy) \
w0 = LOAD4_16(weight + sy * 16 + 4 * 0);         \
w1 = LOAD4_16(weight + sy * 16 + 4 * 1);         \
w2 = LOAD4_16(weight + sy * 16 + 4 * 2);         \
w3 = LOAD4_16(weight + sy * 16 + 4 * 3);         \

#define INIT_MAIN_24_4                                \
    auto a0  = _mm512_loadu_ps(Ay);             \
    auto a1  = _mm512_loadu_ps(Ay + 16);         \
    auto a2  = _mm512_loadu_ps(Ay + 32);             \
    auto a3  = _mm512_loadu_ps(Ay + 48);         \
    auto a4  = _mm512_loadu_ps(Ay + 64);             \
    auto a5  = _mm512_loadu_ps(Ay + 80);         \
    INIT_WEIGHTS; \
    MUL_BLOCK(0);  \
    MUL_BLOCK(1);  \
    MUL_BLOCK(2);  \
    MUL_BLOCK(3);  \
    MUL_BLOCK(4);  \
    MUL_BLOCK(5);

#define COMPUTE_24_4                               \
    a0  = _mm512_loadu_ps(Ay);             \
    a1  = _mm512_loadu_ps(Ay + 16);         \
    a2  = _mm512_loadu_ps(Ay + 32);             \
    a3  = _mm512_loadu_ps(Ay + 48);         \
    a4  = _mm512_loadu_ps(Ay + 64);             \
    a5  = _mm512_loadu_ps(Ay + 80);         \
    LOAD_WEIGHTS(sy); \
    FMADD_BLOCK(0); \
    FMADD_BLOCK(1); \
    FMADD_BLOCK(2); \
    FMADD_BLOCK(3); \
    FMADD_BLOCK(4); \
    FMADD_BLOCK(5);

#define LOAD16(x) _mm512_loadu_ps(x)
#define LOAD4_16(x) _mm512_broadcast_f32x4(_mm_loadu_ps(x))

#define AVX512_TRANPOSE_SAVE(u, v, z0, z2, z4, z6)              \
    {                                                    \
        auto m0 = _mm512_extractf32x4_ps(z0, v);          \
        auto m1 = _mm512_extractf32x4_ps(z2, v);          \
        auto m2 = _mm512_extractf32x4_ps(z4, v);          \
        auto m3 = _mm512_extractf32x4_ps(z6, v);          \
        _MM_TRANSPOSE4_PS(m0, m1, m2, m3);               \
        _mm_store_ps(dst + 4 * (u * 16 + v * 4), m0);     \
        _mm_store_ps(dst + 4 * (u * 16 + v * 4 + 1), m1); \
        _mm_store_ps(dst + 4 * (u * 16 + v * 4 + 2), m2); \
        _mm_store_ps(dst + 4 * (u * 16 + v * 4 + 3), m3); \
    }

#define AVX2_TRANPOSE_SAVE(u, v, z0, z3, z6, z9)              \
    {                                                    \
        auto m0 = _mm256_extractf128_ps(z0, u);          \
        auto m1 = _mm256_extractf128_ps(z3, u);          \
        auto m2 = _mm256_extractf128_ps(z6, u);          \
        auto m3 = _mm256_extractf128_ps(z9, u);          \
        _MM_TRANSPOSE4_PS(m0, m1, m2, m3);               \
        _mm_store_ps(dst + 4 * (0 + 4 * u + 8 * v), m0); \
        _mm_store_ps(dst + 4 * (1 + 4 * u + 8 * v), m1); \
        _mm_store_ps(dst + 4 * (2 + 4 * u + 8 * v), m2); \
        _mm_store_ps(dst + 4 * (3 + 4 * u + 8 * v), m3); \
    }

#define INIT_BLOCK(u)  \
auto z##u##0 = _mm512_mul_ps(a##u, w0);\
auto z##u##1 = _mm512_mul_ps(a##u, w1);\
auto z##u##2 = _mm512_mul_ps(a##u, w2);\
auto z##u##3 = _mm512_mul_ps(a##u, w3);\

#define INIT_MAIN_24_4_4                                \
    auto a0  = LOAD16(Ay);             \
    auto a1  = LOAD16(Ay + 16);         \
    auto a2  = LOAD16(Ay + 32);             \
    auto a3  = LOAD16(Ay + 48);         \
    auto a4  = LOAD16(Ay + 64);             \
    auto a5  = LOAD16(Ay + 80);         \
    auto w0   = LOAD4_16(weight + 0);         \
    auto w1   = LOAD4_16(weight + 4);         \
    auto w2   = LOAD4_16(weight + 8);         \
    auto w3   = LOAD4_16(weight + 12);         \
    INIT_BLOCK(0); \
    INIT_BLOCK(1); \
    INIT_BLOCK(2); \
    INIT_BLOCK(3); \
    INIT_BLOCK(4); \
    INIT_BLOCK(5);

#define COMPUTE_BLOCK(u) \
z##u##0 = _mm512_fmadd_ps(a##u, w0, z##u##0);\
z##u##1 = _mm512_fmadd_ps(a##u, w1, z##u##1);\
z##u##2 = _mm512_fmadd_ps(a##u, w2, z##u##2);\
z##u##3 = _mm512_fmadd_ps(a##u, w3, z##u##3);\

#define COMPUTE_24_4_4                                \
    a0  = LOAD16(Ay);             \
    a1  = LOAD16(Ay + 16);         \
    a2  = LOAD16(Ay + 32);             \
    a3  = LOAD16(Ay + 48);         \
    a4  = LOAD16(Ay + 64);             \
    a5  = LOAD16(Ay + 80);         \
    w0  = LOAD4_16(weight + sy * 16 + 0);         \
    w1  = LOAD4_16(weight + sy * 16 + 4);         \
    w2  = LOAD4_16(weight + sy * 16 + 8);         \
    w3  = LOAD4_16(weight + sy * 16 + 12);         \
    COMPUTE_BLOCK(0); \
    COMPUTE_BLOCK(1); \
    COMPUTE_BLOCK(2); \
    COMPUTE_BLOCK(3); \
    COMPUTE_BLOCK(4); \
    COMPUTE_BLOCK(5);

#define _AVX512_SHUFFLE_SAVE(u, m0, m1, m2, m3) \
    {                                   \
        auto tmp0 = _mm512_shuffle_f32x4(m0, m1, 0x44); \
        auto tmp1 = _mm512_shuffle_f32x4(m2, m3, 0x44); \
        auto tmp2 = _mm512_shuffle_f32x4(m0, m1, 0xEE); \
        auto tmp3 = _mm512_shuffle_f32x4(m2, m3, 0xEE); \
        m0 = _mm512_shuffle_f32x4(tmp0, tmp1, 0x88); \
        m1 = _mm512_shuffle_f32x4(tmp0, tmp1, 0xDD); \
        m2 = _mm512_shuffle_f32x4(tmp2, tmp3, 0x88); \
        m3 = _mm512_shuffle_f32x4(tmp2, tmp3, 0xDD); \
        auto sum = _mm512_add_ps(m0, m1); \
        sum = _mm512_add_ps(sum, m2); \
        sum = _mm512_add_ps(sum, m3); \
        _mm512_storeu_ps(dst + u * 16, sum);          \
    }

static void _AVX512_MNNPackedMatMul_24(float* C, const float* A, const float* B, const size_t* parameter) {
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(float);
    auto bExtraStride = parameter[5] / sizeof(float);
    auto hC4          = UP_DIV(h, 4);
    auto lC4          = UP_DIV(l, 4);
    auto bStride      = bExtraStride + lC4 * 16;
    for (int y = 0; y < hC4; ++y) {
        auto Ay = A;
        auto weight = B + y * bStride;
        auto dst    = C + y * cStride;
        INIT_MAIN_24_4_4;

        for (int sy = 1; sy < lC4; ++sy) {
            Ay = A + sy * 96;
            COMPUTE_24_4_4;
        }
        _AVX512_HADD_SAVE(0, z00, z01, z02, z03);
        _AVX512_HADD_SAVE(1, z10, z11, z12, z13);
        _AVX512_HADD_SAVE(2, z20, z21, z22, z23);
        _AVX512_HADD_SAVE(3, z30, z31, z32, z33);
        _AVX512_HADD_SAVE(4, z40, z41, z42, z43);
        _AVX512_HADD_SAVE(5, z50, z51, z52, z53);
    }
}


#define INIT_MAIN_20_4                                \
    auto a0  = _mm512_loadu_ps(Ay);             \
    auto a1  = _mm512_loadu_ps(Ay + 16);         \
    auto a2  = _mm512_loadu_ps(Ay + 32);             \
    auto a3  = _mm512_loadu_ps(Ay + 48);         \
    auto a4  = _mm512_loadu_ps(Ay + 64);             \
    INIT_WEIGHTS; \
    MUL_BLOCK(0); \
    MUL_BLOCK(1); \
    MUL_BLOCK(2); \
    MUL_BLOCK(3); \
    MUL_BLOCK(4);

#define COMPUTE_20_4                                \
    a0  = _mm512_loadu_ps(Ay);             \
    a1  = _mm512_loadu_ps(Ay + 16);         \
    a2  = _mm512_loadu_ps(Ay + 32);             \
    a3  = _mm512_loadu_ps(Ay + 48);         \
    a4  = _mm512_loadu_ps(Ay + 64);             \
    LOAD_WEIGHTS(sy);    \
    FMADD_BLOCK(0);  \
    FMADD_BLOCK(1);  \
    FMADD_BLOCK(2);  \
    FMADD_BLOCK(3);  \
    FMADD_BLOCK(4);

static void _AVX512_MNNPackedMatMul_20_4(float* C, const float* A, const float* B, const size_t* parameter) {
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(float);
    auto bExtraStride = parameter[5] / sizeof(float);
    auto hC4          = UP_DIV(h, 4);
    auto lC4          = UP_DIV(l, 4);
    auto bStride      = bExtraStride + lC4 * 16;
    auto xCount       = parameter[0] / sizeof(float);

    for (int y = 0; y < hC4; ++y) {
        auto Ay = A;
        auto weight = B + y * bStride;
        auto dst    = C + y * cStride;
        INIT_MAIN_20_4;

        for (int sy = 1; sy < lC4; ++sy) {
            Ay = A + sy * xCount * 4;
            COMPUTE_20_4;
        }

        _AVX512_HADD_SAVE(0, z00, z01, z02, z03);
        _AVX512_HADD_SAVE(1, z10, z11, z12, z13);
        _AVX512_HADD_SAVE(2, z20, z21, z22, z23);
        _AVX512_HADD_SAVE(3, z30, z31, z32, z33);
        _AVX512_HADD_SAVE(4, z40, z41, z42, z43);
    }
}

#define INIT_MAIN_16_4                                  \
    auto a0  = _mm512_loadu_ps(Ay);             \
    auto a1  = _mm512_loadu_ps(Ay + 16);         \
    auto a2  = _mm512_loadu_ps(Ay + 32);             \
    auto a3  = _mm512_loadu_ps(Ay + 48);         \
    INIT_WEIGHTS; \
    MUL_BLOCK(0); \
    MUL_BLOCK(1); \
    MUL_BLOCK(2); \
    MUL_BLOCK(3);

#define COMPUTE_16_4                               \
    a0  = _mm512_loadu_ps(Ay);             \
    a1  = _mm512_loadu_ps(Ay + 16);         \
    a2  = _mm512_loadu_ps(Ay + 32);             \
    a3  = _mm512_loadu_ps(Ay + 48);         \
    LOAD_WEIGHTS(sy);    \
    FMADD_BLOCK(0);  \
    FMADD_BLOCK(1);  \
    FMADD_BLOCK(2);  \
    FMADD_BLOCK(3);

static void _AVX512_MNNPackedMatMul_16_4(float* C, const float* A, const float* B, const size_t* parameter) {
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(float);
    auto bExtraStride = parameter[5] / sizeof(float);
    auto hC4          = UP_DIV(h, 4);
    auto lC4          = UP_DIV(l, 4);
    auto bStride      = bExtraStride + lC4 * 16;
    auto xCount       = parameter[0] / sizeof(float);

    for (int y = 0; y < hC4; ++y) {
        auto Ay = A;
        auto weight = B + y * bStride;
        auto dst    = C + y * cStride;
        INIT_MAIN_16_4;

        for (int sy = 1; sy < lC4; ++sy) {
            Ay = A + sy * xCount * 4;
            COMPUTE_16_4;
        }
        _AVX512_HADD_SAVE(0, z00, z01, z02, z03);
        _AVX512_HADD_SAVE(1, z10, z11, z12, z13);
        _AVX512_HADD_SAVE(2, z20, z21, z22, z23);
        _AVX512_HADD_SAVE(3, z30, z31, z32, z33);
    }
}

#define INIT_MAIN_12_4                                  \
    auto a0  = _mm512_loadu_ps(Ay);             \
    auto a1  = _mm512_loadu_ps(Ay + 16);         \
    auto a2  = _mm512_loadu_ps(Ay + 32);             \
    INIT_WEIGHTS; \
    MUL_BLOCK(0); \
    MUL_BLOCK(1); \
    MUL_BLOCK(2);

#define COMPUTE_12_4                               \
    a0  = _mm512_loadu_ps(Ay);             \
    a1  = _mm512_loadu_ps(Ay + 16);         \
    a2  = _mm512_loadu_ps(Ay + 32);             \
    LOAD_WEIGHTS(sy);    \
    FMADD_BLOCK(0);  \
    FMADD_BLOCK(1);  \
    FMADD_BLOCK(2);

static void _AVX512_MNNPackedMatMul_12_4(float* C, const float* A, const float* B, const size_t* parameter) {
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(float);
    auto bExtraStride = parameter[5] / sizeof(float);
    auto hC4          = UP_DIV(h, 4);
    auto lC4          = UP_DIV(l, 4);
    auto bStride      = bExtraStride + lC4 * 16;
    auto xCount       = parameter[0] / sizeof(float);

    for (int y = 0; y < hC4; ++y) {
        auto Ay = A;
        auto weight = B + y * bStride;
        auto dst    = C + y * cStride;
        INIT_MAIN_12_4; \

        for (int sy = 1; sy < lC4; ++sy) {
            Ay = A + sy * xCount * 4;
            COMPUTE_12_4;
        }
        _AVX512_HADD_SAVE(0, z00, z01, z02, z03);
        _AVX512_HADD_SAVE(1, z10, z11, z12, z13);
        _AVX512_HADD_SAVE(2, z20, z21, z22, z23);
    }
}

#define INIT_MAIN_8_4                         \
    auto a0  = _mm512_loadu_ps(Ay);             \
    auto a1  = _mm512_loadu_ps(Ay + 16);        \
    INIT_WEIGHTS; \
    MUL_BLOCK(0); \
    MUL_BLOCK(1);

#define COMPUTE_8_4                                \
    a0  = _mm512_loadu_ps(Ay);             \
    a1  = _mm512_loadu_ps(Ay + 16);         \
    LOAD_WEIGHTS(sy);    \
    FMADD_BLOCK(0);  \
    FMADD_BLOCK(1);

static void _AVX512_MNNPackedMatMul_8_4(float* C, const float* A, const float* B, const size_t* parameter) {
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(float);
    auto bExtraStride = parameter[5] / sizeof(float);
    auto hC4          = UP_DIV(h, 4);
    auto lC4          = UP_DIV(l, 4);
    auto bStride      = bExtraStride + lC4 * 16;
    auto xCount       = parameter[0] / sizeof(float);

    for (int y = 0; y < hC4; ++y) {
        auto Ay = A;
        auto weight = B + y * bStride;
        auto dst    = C + y * cStride;
        INIT_MAIN_8_4;

        for (int sy = 1; sy < lC4; ++sy) {
            Ay = A + sy * xCount * 4;
            COMPUTE_8_4;
        }
        _AVX512_HADD_SAVE(0, z00, z01, z02, z03);
        _AVX512_HADD_SAVE(1, z10, z11, z12, z13);
    }
}

static void _AVX512_MNNPackedMatMul_5_4(float* C, const float* A, const float* B, const size_t* parameter) {
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(float);
    auto bExtraStride = parameter[5] / sizeof(float);
    auto hC4          = UP_DIV(h, 4);
    auto lC4          = UP_DIV(l, 4);
    auto bStride      = bExtraStride + lC4 * 16;
    auto xCount       = parameter[0] / sizeof(float);

    for (int y = 0; y < hC4; ++y) {
        auto Ay = A;
        auto weight = B + y * bStride;
        auto dst    = C + y * cStride;
        auto a0 = _mm512_loadu_ps(Ay);
        auto a1 = _mm_loadu_ps(Ay + 16);
        auto wt0 = _mm_loadu_ps(weight);
        auto w0 = _mm512_broadcast_f32x4(wt0);
        auto wt1 = _mm_loadu_ps(weight + 4);
        auto w1 = _mm512_broadcast_f32x4(wt1);
        auto wt2 = _mm_loadu_ps(weight + 8);
        auto w2 = _mm512_broadcast_f32x4(wt2);
        auto wt3 = _mm_loadu_ps(weight + 12);
        auto w3 = _mm512_broadcast_f32x4(wt3);
        auto z00 = _mm512_mul_ps(a0, w0);
        auto z01 = _mm512_mul_ps(a0, w1);
        auto z02 = _mm512_mul_ps(a0, w2);
        auto z03 = _mm512_mul_ps(a0, w3);
        auto z10 = _mm_mul_ps(a1, wt0);
        auto z11 = _mm_mul_ps(a1, wt1);
        auto z12 = _mm_mul_ps(a1, wt2);
        auto z13 = _mm_mul_ps(a1, wt3);

        for (int sy = 1; sy < lC4; ++sy) {
            Ay = A + sy * xCount * 4;
            a0 = _mm512_loadu_ps(Ay);
            a1 = _mm_loadu_ps(Ay + 16);
            wt0 = _mm_loadu_ps(weight + sy * 16);
            w0 = _mm512_broadcast_f32x4(wt0);
            wt1 = _mm_loadu_ps(weight + sy * 16 + 4);
            w1 = _mm512_broadcast_f32x4(wt1);
            wt2 = _mm_loadu_ps(weight + sy * 16 + 8);
            w2 = _mm512_broadcast_f32x4(wt2);
            wt3 = _mm_loadu_ps(weight + sy * 16 + 12);
            w3 = _mm512_broadcast_f32x4(wt3);
            z00 = _mm512_fmadd_ps(a0, w0, z00);
            z01 = _mm512_fmadd_ps(a0, w1, z01);
            z02 = _mm512_fmadd_ps(a0, w2, z02);
            z03 = _mm512_fmadd_ps(a0, w3, z03);
            z10 = _mm_fmadd_ps(a1, wt0, z10);
            z11 = _mm_fmadd_ps(a1, wt1, z11);
            z12 = _mm_fmadd_ps(a1, wt2, z12);
            z13 = _mm_fmadd_ps(a1, wt3, z13);
        }
        _AVX512_HADD_SAVE(0, z00, z01, z02, z03);
        auto tmp4 = _mm_hadd_ps(z10, z11);
        auto tmp5 = _mm_hadd_ps(z12, z13);
        auto sum  = _mm_hadd_ps(tmp4, tmp5);
        _mm_storeu_ps(dst + 16, sum);
    }
}

#define INIT_MAIN_4_4                                  \
    auto a0  = _mm512_loadu_ps(Ay);             \
    INIT_WEIGHTS; \
    MUL_BLOCK(0);

#define COMPUTE_4_4                                \
    a0  = _mm512_loadu_ps(Ay);             \
    LOAD_WEIGHTS(sy);    \
    FMADD_BLOCK(0);

static void _AVX512_MNNPackedMatMul_4_4(float* C, const float* A, const float* B, const size_t* parameter) {
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(float);
    auto bExtraStride = parameter[5] / sizeof(float);
    auto hC4          = UP_DIV(h, 4);
    auto lC4          = UP_DIV(l, 4);
    auto xCount       = parameter[0] / sizeof(float);
    auto bStride      = bExtraStride + lC4 * 16;

    for (int y = 0; y < hC4; ++y) {
        auto Ay = A;
        auto weight = B + y * bStride;
        auto dst    = C + y * cStride;
        INIT_MAIN_4_4;

        for (int sy = 1; sy < lC4; ++sy) {
            Ay = A + sy * xCount * 4;
            COMPUTE_4_4;
        }
        _AVX512_HADD_SAVE(0, z00, z01, z02, z03);
    }
}

static void _AVX512_MNNPackedMatMul_3_4(float* C, const float* A, const float* B, const size_t* parameter) {
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(float);
    auto bExtraStride = parameter[5] / sizeof(float);
    auto hC4          = UP_DIV(h, 4);
    auto lC4          = UP_DIV(l, 4);
    auto xCount       = parameter[0] / sizeof(float);
    auto bStride      = bExtraStride + lC4 * 16;

    for (int y = 0; y < hC4; ++y) {
        auto Ay = A;
        auto weight = B + y * bStride;
        auto dst    = C + y * cStride;
        auto a0  = _mm512_maskz_loadu_ps(0x0FFF, Ay);
        INIT_WEIGHTS;
        MUL_BLOCK(0);

        for (int sy = 1; sy < lC4; ++sy) {
            Ay = A + sy * xCount * 4;
            a0  = _mm512_maskz_loadu_ps(0x0FFF, Ay);
            LOAD_WEIGHTS(sy);
            FMADD_BLOCK(0);
        }
        auto z0h = _mm512_extractf32x8_ps(z00, 0);
        auto z0l = _mm512_extractf32x4_ps(z00, 2);
        auto z1h = _mm512_extractf32x8_ps(z01, 0);
        auto z1l = _mm512_extractf32x4_ps(z01, 2);
        auto z2h = _mm512_extractf32x8_ps(z02, 0);
        auto z2l = _mm512_extractf32x4_ps(z02, 2);
        auto z3h = _mm512_extractf32x8_ps(z03, 0);
        auto z3l = _mm512_extractf32x4_ps(z03, 2);
        auto tmp0 = _mm256_hadd_ps(z0h, z1h);
        auto tmp1 = _mm256_hadd_ps(z2h, z3h);
        auto sum0 = _mm256_hadd_ps(tmp0, tmp1);
        auto tmp3 = _mm_hadd_ps(z0l, z1l);
        auto tmp4 = _mm_hadd_ps(z2l, z3l);
        auto sum1 = _mm_hadd_ps(tmp3, tmp4);
        _mm256_storeu_ps(dst, sum0);
        _mm_storeu_ps(dst + 8, sum1);
    }
}

static void _AVX512_MNNPackedMatMul_2_4(float* C, const float* A, const float* B, const size_t* parameter) {
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(float);
    auto bExtraStride = parameter[5] / sizeof(float);
    auto hC4          = UP_DIV(h, 4);
    auto lC4          = UP_DIV(l, 4);
    auto xCount       = parameter[0] / sizeof(float);
    auto bStride      = bExtraStride + lC4 * 16;

    for (int y = 0; y < hC4; ++y) {
        auto Ay = A;
        auto weight = B + y * bStride;
        auto dst    = C + y * cStride;
        auto a0  = _mm256_loadu_ps(Ay);
        auto wt = _mm_loadu_ps(weight);
        auto w0 = _mm256_broadcast_f32x4(wt);
        wt = _mm_loadu_ps(weight + 4);
        auto w1 = _mm256_broadcast_f32x4(wt);
        wt = _mm_loadu_ps(weight + 8);
        auto w2 = _mm256_broadcast_f32x4(wt);
        wt = _mm_loadu_ps(weight + 12);
        auto w3 = _mm256_broadcast_f32x4(wt);
        auto z0 = _mm256_mul_ps(a0, w0);
        auto z1 = _mm256_mul_ps(a0, w1);
        auto z2 = _mm256_mul_ps(a0, w2);
        auto z3 = _mm256_mul_ps(a0, w3);

        for (int sy = 1; sy < lC4; ++sy) {
            Ay = A + sy * xCount * 4;
            a0  = _mm256_loadu_ps(Ay);;
            wt = _mm_loadu_ps(weight + sy * 16);
            w0 = _mm256_broadcast_f32x4(wt);
            wt = _mm_loadu_ps(weight + sy * 16 + 4);
            w1 = _mm256_broadcast_f32x4(wt);
            wt = _mm_loadu_ps(weight + sy * 16 + 8);
            w2 = _mm256_broadcast_f32x4(wt);
            wt = _mm_loadu_ps(weight + sy * 16 + 12);
            w3 = _mm256_broadcast_f32x4(wt);
            z0 = _mm256_fmadd_ps(a0, w0, z0);
            z1 = _mm256_fmadd_ps(a0, w1, z1);
            z2 = _mm256_fmadd_ps(a0, w2, z2);
            z3 = _mm256_fmadd_ps(a0, w3, z3);
        }
        auto z0h = _mm256_extractf32x4_ps(z0, 0);
        auto z0l = _mm256_extractf32x4_ps(z0, 1);
        auto z1h = _mm256_extractf32x4_ps(z1, 0);
        auto z1l = _mm256_extractf32x4_ps(z1, 1);
        auto z2h = _mm256_extractf32x4_ps(z2, 0);
        auto z2l = _mm256_extractf32x4_ps(z2, 1);
        auto z3h = _mm256_extractf32x4_ps(z3, 0);
        auto z3l = _mm256_extractf32x4_ps(z3, 1);
        auto tmp0 = _mm_hadd_ps(z0h, z1h);
        auto tmp1 = _mm_hadd_ps(z2h, z3h);
        auto sum0 = _mm_hadd_ps(tmp0, tmp1);
        auto tmp3 = _mm_hadd_ps(z0l, z1l);
        auto tmp4 = _mm_hadd_ps(z2l, z3l);
        auto sum1 = _mm_hadd_ps(tmp3, tmp4);
        _mm_storeu_ps(dst, sum0);
        _mm_storeu_ps(dst + 4, sum1);
    }
}

static void _AVX512_MNNPackedMatMul_1_4(float* C, const float* A, const float* B, const size_t* parameter) {
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(float);
    auto bExtraStride = parameter[5] / sizeof(float);
    auto hC4          = UP_DIV(h, 4);
    auto lC4          = UP_DIV(l, 4);
    auto xCount       = parameter[0] / sizeof(float);
    auto bStride      = bExtraStride + lC4 * 16;

    for (int y = 0; y < hC4; ++y) {
        auto Ay = A;
        auto weight = B + y * bStride;
        auto dst    = C + y * cStride;
        auto a0  = _mm_loadu_ps(Ay);
        auto w0 = _mm_loadu_ps(weight);
        auto w1 = _mm_loadu_ps(weight + 4);
        auto w2 = _mm_loadu_ps(weight + 8);
        auto w3 = _mm_loadu_ps(weight + 12);
        auto z0 = _mm_mul_ps(a0, w0);
        auto z1 = _mm_mul_ps(a0, w1);
        auto z2 = _mm_mul_ps(a0, w2);
        auto z3 = _mm_mul_ps(a0, w3);

        for (int sy = 1; sy < lC4; ++sy) {
            Ay = A + sy * xCount * 4;
            a0  = _mm_loadu_ps(Ay);;
            w0 = _mm_loadu_ps(weight + sy * 16);
            w1 = _mm_loadu_ps(weight + sy * 16 + 4);
            w2 = _mm_loadu_ps(weight + sy * 16 + 8);
            w3 = _mm_loadu_ps(weight + sy * 16 + 12);
            z0 = _mm_fmadd_ps(a0, w0, z0);
            z1 = _mm_fmadd_ps(a0, w1, z1);
            z2 = _mm_fmadd_ps(a0, w2, z2);
            z3 = _mm_fmadd_ps(a0, w3, z3);
        }
        auto tmp0 = _mm_hadd_ps(z0, z1);
        auto tmp1 = _mm_hadd_ps(z2, z3);
        auto sum  = _mm_hadd_ps(tmp0, tmp1);
        _mm_storeu_ps(dst, sum);
    }
}

static void _AVX512_MNNPackednMatMulRemainCommon_4(float* C, const float* A, const float* B, size_t eSize,
                                              const size_t* parameter, const float* postParameters,
                                              const float* bias) {
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(float);
    auto bExtraStride = parameter[5] / sizeof(float);
    auto es           = eSize;
    auto oC           = C;

    if (eSize >= 20) {
        _AVX512_MNNPackedMatMul_20_4(C, A, B, parameter);
        eSize -= 20;
        C += 20 * 4;
        A += 20 * 4;
    }
    if (eSize >= 16) {
        _AVX512_MNNPackedMatMul_16_4(C, A, B, parameter);
        eSize -= 16;
        C += 16 * 4;
        A += 16 * 4;
    }
    if (eSize >= 12) {
        _AVX512_MNNPackedMatMul_12_4(C, A, B, parameter);
        eSize -= 12;
        C += 12 * 4;
        A += 12 * 4;
    }
    if (eSize >= 8) {
        _AVX512_MNNPackedMatMul_8_4(C, A, B, parameter);
        eSize -= 8;
        C += 8 * 4;
        A += 8 * 4;
    }
    if (eSize >= 5) {
        _AVX512_MNNPackedMatMul_5_4(C, A, B, parameter);
        eSize -= 5;
        C += 5 * 4;
        A += 5 * 4;
    }
    if (eSize >= 4) {
        _AVX512_MNNPackedMatMul_4_4(C, A, B, parameter);
        eSize -= 4;
        C += 4 * 4;
        A += 4 * 4;
    }
    if (eSize >= 3) {
        _AVX512_MNNPackedMatMul_3_4(C, A, B, parameter);
        eSize -= 3;
        C += 3 * 4;
        A += 3 * 4;
    }
    if (eSize >= 2) {
        _AVX512_MNNPackedMatMul_2_4(C, A, B, parameter);
        eSize -= 2;
        C += 2 * 4;
        A += 2 * 4;
    }
    if (eSize >= 1) {
        _AVX512_MNNPackedMatMul_1_4(C, A, B, parameter);
        eSize -= 1;
        return;
    }
}
