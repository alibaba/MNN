//
//  GemmFunction.hpp
//  MNN
//
//  Created by MNN on 2020/09/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#define INIT_MAIN_12_4                               \
    auto s0  = _mm_loadu_ps(A + 0 * 12);             \
    auto s1  = _mm_loadu_ps(A + 0 * 12 + 4);         \
    auto s2  = _mm_loadu_ps(A + 0 * 12 + 8);         \
    auto w0  = _mm_set1_ps(weight[0 * 4 + 0]); \
    auto z0  = _mm_mul_ps(s0, w0);                   \
    auto z1  = _mm_mul_ps(s1, w0);                   \
    auto z2  = _mm_mul_ps(s2, w0);                   \
    auto w1  = _mm_set1_ps(weight[0 * 4 + 1]); \
    auto z3  = _mm_mul_ps(s0, w1);                   \
    auto z4  = _mm_mul_ps(s1, w1);                   \
    auto z5  = _mm_mul_ps(s2, w1);                   \
    auto w2  = _mm_set1_ps(weight[0 * 4 + 2]); \
    auto z6  = _mm_mul_ps(s0, w2);                   \
    auto z7  = _mm_mul_ps(s1, w2);                   \
    auto z8  = _mm_mul_ps(s2, w2);                   \
    auto w3  = _mm_set1_ps(weight[0 * 4 + 3]); \
    auto z9  = _mm_mul_ps(s0, w3);                   \
    auto z10 = _mm_mul_ps(s1, w3);                   \
    auto z11 = _mm_mul_ps(s2, w3);

#define COMPUTE_12_4                             \
    s0  = _mm_loadu_ps(A + sy * 12);             \
    s1  = _mm_loadu_ps(A + sy * 12 + 4);         \
    s2  = _mm_loadu_ps(A + sy * 12 + 8);         \
    w0  = _mm_set1_ps(weight[sy * 4 + 0]); \
    z0  = MNNSSEFMA(s0, w0, z0);                 \
    z1  = MNNSSEFMA(s1, w0, z1);                 \
    z2  = MNNSSEFMA(s2, w0, z2);                 \
    w1  = _mm_set1_ps(weight[sy * 4 + 1]); \
    z3  = MNNSSEFMA(s0, w1, z3);                 \
    z4  = MNNSSEFMA(s1, w1, z4);                 \
    z5  = MNNSSEFMA(s2, w1, z5);                 \
    w2  = _mm_set1_ps(weight[sy * 4 + 2]); \
    z6  = MNNSSEFMA(s0, w2, z6);                 \
    z7  = MNNSSEFMA(s1, w2, z7);                 \
    z8  = MNNSSEFMA(s2, w2, z8);                 \
    w3  = _mm_set1_ps(weight[sy * 4 + 3]); \
    z9  = MNNSSEFMA(s0, w3, z9);                 \
    z10 = MNNSSEFMA(s1, w3, z10);                \
    z11 = MNNSSEFMA(s2, w3, z11);

static void _SSE_MNNPackedMatMul_12(float* C, const float* A, const float* B, const size_t* parameter) {
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(float);
    auto bExtraStride = parameter[5] / sizeof(float);
    auto bStride      = bExtraStride + l * 4;
    auto hC4          = UP_DIV(h, 4);
    for (int y = 0; y < hC4; ++y) {
        auto weight = B + y * bStride;
        auto dst    = C + y * cStride;
        INIT_MAIN_12_4;

        for (int sy = 1; sy < l; ++sy) {
            COMPUTE_12_4;
        }
        TRANPOSE_SAVE(0, 0, z0, z3, z6, z9);
        TRANPOSE_SAVE(0, 1, z1, z4, z7, z10);
        TRANPOSE_SAVE(0, 2, z2, z5, z8, z11);
    }
}

#define INIT_MAIN_8_4                                \
    auto s0  = _mm_loadu_ps(A + 0 * aStride);        \
    auto s1  = _mm_loadu_ps(A + 0 * aStride + 4);    \
    auto w0  = _mm_set1_ps(weight[0 * 4 + 0]); \
    auto w1  = _mm_set1_ps(weight[0 * 4 + 1]); \
    auto w2  = _mm_set1_ps(weight[0 * 4 + 2]); \
    auto w3  = _mm_set1_ps(weight[0 * 4 + 3]); \
    auto z0  = _mm_mul_ps(s0, w0);                   \
    auto z3  = _mm_mul_ps(s0, w1);                   \
    auto z6  = _mm_mul_ps(s0, w2);                   \
    auto z9  = _mm_mul_ps(s0, w3);                   \
    auto z1  = _mm_mul_ps(s1, w0);                   \
    auto z4  = _mm_mul_ps(s1, w1);                   \
    auto z7  = _mm_mul_ps(s1, w2);                   \
    auto z10 = _mm_mul_ps(s1, w3);

#define COMPUTE_8_4                              \
    s0  = _mm_loadu_ps(A + sy * aStride);        \
    s1  = _mm_loadu_ps(A + sy * aStride + 4);    \
    w0  = _mm_set1_ps(weight[sy * 4 + 0]); \
    w1  = _mm_set1_ps(weight[sy * 4 + 1]); \
    w2  = _mm_set1_ps(weight[sy * 4 + 2]); \
    w3  = _mm_set1_ps(weight[sy * 4 + 3]); \
    z0  = MNNSSEFMA(s0, w0, z0);                 \
    z3  = MNNSSEFMA(s0, w1, z3);                 \
    z6  = MNNSSEFMA(s0, w2, z6);                 \
    z9  = MNNSSEFMA(s0, w3, z9);                 \
    z1  = MNNSSEFMA(s1, w0, z1);                 \
    z4  = MNNSSEFMA(s1, w1, z4);                 \
    z7  = MNNSSEFMA(s1, w2, z7);                 \
    z10 = MNNSSEFMA(s1, w3, z10);

static void _SSE_MNNPackedMatMul_8(float* C, const float* A, const float* B, const size_t* parameter) {
    auto aStride      = parameter[0] / sizeof(float);
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(float);
    auto bExtraStride = parameter[5] / sizeof(float);
    auto bStride      = bExtraStride + l * 4;
    auto hC4          = UP_DIV(h, 4);
    for (int y = 0; y < hC4; ++y) {
        auto weight = B + y * bStride;
        auto dst    = C + y * cStride;
        INIT_MAIN_8_4;

        for (int sy = 1; sy < l; ++sy) {
            COMPUTE_8_4;
        }
        TRANPOSE_SAVE(0, 0, z0, z3, z6, z9);
        TRANPOSE_SAVE(0, 1, z1, z4, z7, z10);
    }
}

static void _SSE_MNNPackedMatMul_4(float* C, const float* A, const float* B, const size_t* parameter) {
    auto aStride      = parameter[0] / sizeof(float);
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(float);
    auto bExtraStride = parameter[5] / sizeof(float);
    auto bStride      = bExtraStride + l * 4;
    auto hC4          = UP_DIV(h, 4);
    for (int y = 0; y < hC4; ++y) {
        auto weight = B + y * bStride;
        auto dst    = C + y * cStride;
        auto s0     = _mm_loadu_ps(A + 0 * aStride);
        auto w0     = _mm_set1_ps(weight[0 * 4 + 0]);
        auto w1     = _mm_set1_ps(weight[0 * 4 + 1]);
        auto w2     = _mm_set1_ps(weight[0 * 4 + 2]);
        auto w3     = _mm_set1_ps(weight[0 * 4 + 3]);
        auto z0     = _mm_mul_ps(s0, w0);
        auto z3     = _mm_mul_ps(s0, w1);
        auto z6     = _mm_mul_ps(s0, w2);
        auto z9     = _mm_mul_ps(s0, w3);

        for (int sy = 1; sy < l; ++sy) {
            s0 = _mm_loadu_ps(A + sy * aStride);
            w0 = _mm_set1_ps(weight[sy * 4 + 0]);
            w1 = _mm_set1_ps(weight[sy * 4 + 1]);
            w2 = _mm_set1_ps(weight[sy * 4 + 2]);
            w3 = _mm_set1_ps(weight[sy * 4 + 3]);
            z0 = MNNSSEFMA(s0, w0, z0);
            z3 = MNNSSEFMA(s0, w1, z3);
            z6 = MNNSSEFMA(s0, w2, z6);
            z9 = MNNSSEFMA(s0, w3, z9);
        }
        _MM_TRANSPOSE4_PS(z0, z3, z6, z9);
        _mm_storeu_ps(dst + 4 * 0, z0);
        _mm_storeu_ps(dst + 4 * 1, z3);
        _mm_storeu_ps(dst + 4 * 2, z6);
        _mm_storeu_ps(dst + 4 * 3, z9);
    }
}
static void _SSE_MNNPackednMatMulRemainCommon(float* C, const float* A, const float* B, size_t eSize,
                                              const size_t* parameter, const float* postParameters,
                                              const float* bias) {
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(float);
    auto bExtraStride = parameter[5] / sizeof(float);
    auto bStride      = bExtraStride + l * 4;
    auto hC4          = UP_DIV(h, 4);
    auto es           = eSize;
    auto oC           = C;
    auto aStride      = parameter[0] / sizeof(float);
    if (eSize >= 8) {
        _SSE_MNNPackedMatMul_8(C, A, B, parameter);
        eSize -= 8;
        C += 8 * 4;
        A += 8;
    }
    if (eSize >= 4) {
        _SSE_MNNPackedMatMul_4(C, A, B, parameter);
        eSize -= 4;
        C += 4 * 4;
        A += 4;
    }
    for (int x = 0; x < eSize; ++x) {
        auto src = A + x;
        for (int y = 0; y < hC4; ++y) {
            auto weight = B + y * bStride;
            auto dst    = C + y * cStride + x * 4;
            auto sum    = _mm_set1_ps(0.0f);
            for (int sy = 0; sy < l; ++sy) {
                auto s = _mm_set1_ps(src[sy * aStride]);
                auto w = _mm_loadu_ps(weight + 4 * sy);
                sum    = MNNSSEFMA(s, w, sum);
            }
            _mm_storeu_ps(dst, sum);
        }
    }
}

#ifdef MNN_LOW_MEMORY
//----------------------- MatMul(float, int4) Functions ---------------------------//
static inline __m128 _load_int4x4(const uint8_t* src, __m128 alpha, __m128 bias) {
    auto w01    = src[0];
    auto w23    = src[1];
    int iw01    = w01;
    int iw23    = w23;
    int iw0     = iw01 / 16;
    int iw1     = iw01 % 16;
    int iw2     = iw23 / 16;
    int iw3     = iw23 % 16;
    auto ws     = _mm_set_ps(iw3, iw2, iw1, iw0);
    ws          = _mm_sub_ps(ws, _mm_set1_ps(8));
    ws          = _mm_add_ps(_mm_mul_ps(ws, alpha), bias);
    return ws;
}

static void _SSE_MNNPackedMatMul_12_int4(float* C, const float* A, const float* fB, const size_t* parameter, const float* k, const float* b) {
    auto B            = reinterpret_cast<const uint8_t*>(fB);
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(float);
    auto bExtraStride = parameter[5] / sizeof(float);
    auto bStride      = bExtraStride + l * 4;
    auto hC4          = UP_DIV(h, 4);
    float ws_tmp[4];
    for (int y = 0; y < hC4; ++y) {
        auto weight = B + y * bStride / 2;
        auto dst    = C + y * cStride;
        auto alpha  = _mm_loadu_ps(k + y * 4);
        auto bias   = _mm_loadu_ps(b + y * 4);
        auto s0  = _mm_loadu_ps(A + 0 * 12);
        auto s1  = _mm_loadu_ps(A + 0 * 12 + 4);
        auto s2  = _mm_loadu_ps(A + 0 * 12 + 8);
        auto ws  = _load_int4x4(weight, alpha, bias);
        _mm_storeu_ps(ws_tmp, ws);
        auto w0  = _mm_set1_ps(ws_tmp[0]);
        auto w1  = _mm_set1_ps(ws_tmp[1]);
        auto w2  = _mm_set1_ps(ws_tmp[2]);
        auto w3  = _mm_set1_ps(ws_tmp[3]);
        auto z0  = _mm_mul_ps(s0, w0);
        auto z1  = _mm_mul_ps(s1, w0);
        auto z2  = _mm_mul_ps(s2, w0);
        auto z3  = _mm_mul_ps(s0, w1);
        auto z4  = _mm_mul_ps(s1, w1);
        auto z5  = _mm_mul_ps(s2, w1);
        auto z6  = _mm_mul_ps(s0, w2);
        auto z7  = _mm_mul_ps(s1, w2);
        auto z8  = _mm_mul_ps(s2, w2);
        auto z9  = _mm_mul_ps(s0, w3);
        auto z10 = _mm_mul_ps(s1, w3);
        auto z11 = _mm_mul_ps(s2, w3);

        for (int sy = 1; sy < l; ++sy) {
            s0  = _mm_loadu_ps(A + sy * 12);
            s1  = _mm_loadu_ps(A + sy * 12 + 4);
            s2  = _mm_loadu_ps(A + sy * 12 + 8);
            ws = _load_int4x4(weight + sy * 2, alpha, bias);
            _mm_storeu_ps(ws_tmp, ws);
            w0 = _mm_set1_ps(ws_tmp[0]);
            w1 = _mm_set1_ps(ws_tmp[1]);
            w2 = _mm_set1_ps(ws_tmp[2]);
            w3 = _mm_set1_ps(ws_tmp[3]);
            z0  = MNNSSEFMA(s0, w0, z0);
            z1  = MNNSSEFMA(s1, w0, z1);
            z2  = MNNSSEFMA(s2, w0, z2);
            z3  = MNNSSEFMA(s0, w1, z3);
            z4  = MNNSSEFMA(s1, w1, z4);
            z5  = MNNSSEFMA(s2, w1, z5);
            z6  = MNNSSEFMA(s0, w2, z6);
            z7  = MNNSSEFMA(s1, w2, z7);
            z8  = MNNSSEFMA(s2, w2, z8);
            z9  = MNNSSEFMA(s0, w3, z9);
            z10 = MNNSSEFMA(s1, w3, z10);
            z11 = MNNSSEFMA(s2, w3, z11);
        }
        TRANPOSE_SAVE(0, 0, z0, z3, z6, z9);
        TRANPOSE_SAVE(0, 1, z1, z4, z7, z10);
        TRANPOSE_SAVE(0, 2, z2, z5, z8, z11);
    }
}

static void _SSE_MNNPackedMatMul_8_int4(float* C, const float* A, const uint8_t* B, const size_t* parameter, const float* k, const float* b) {
    auto aStride      = parameter[0] / sizeof(float);
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(float);
    auto bExtraStride = parameter[5] / sizeof(float);
    auto bStride      = bExtraStride + l * 4;
    auto hC4          = UP_DIV(h, 4);
    float ws_tmp[4];
    for (int y = 0; y < hC4; ++y) {
        auto weight = B + y * bStride / 2;
        auto dst    = C + y * cStride;
        auto alpha  = _mm_loadu_ps(k + y * 4);
        auto bias   = _mm_loadu_ps(b + y * 4);
        auto s0  = _mm_loadu_ps(A + 0 * aStride);
        auto s1  = _mm_loadu_ps(A + 0 * aStride + 4);
        auto ws  = _load_int4x4(weight, alpha, bias);
        _mm_storeu_ps(ws_tmp, ws);
        auto w0  = _mm_set1_ps(ws_tmp[0]);
        auto w1  = _mm_set1_ps(ws_tmp[1]);
        auto w2  = _mm_set1_ps(ws_tmp[2]);
        auto w3  = _mm_set1_ps(ws_tmp[3]);
        auto z0  = _mm_mul_ps(s0, w0);
        auto z3  = _mm_mul_ps(s0, w1);
        auto z6  = _mm_mul_ps(s0, w2);
        auto z9  = _mm_mul_ps(s0, w3);
        auto z1  = _mm_mul_ps(s1, w0);
        auto z4  = _mm_mul_ps(s1, w1);
        auto z7  = _mm_mul_ps(s1, w2);
        auto z10 = _mm_mul_ps(s1, w3);

        for (int sy = 1; sy < l; ++sy) {
            s0  = _mm_loadu_ps(A + sy * aStride);
            s1  = _mm_loadu_ps(A + sy * aStride + 4);
            ws = _load_int4x4(weight + sy * 2, alpha, bias);
            _mm_storeu_ps(ws_tmp, ws);
            w0 = _mm_set1_ps(ws_tmp[0]);
            w1 = _mm_set1_ps(ws_tmp[1]);
            w2 = _mm_set1_ps(ws_tmp[2]);
            w3 = _mm_set1_ps(ws_tmp[3]);
            z0  = MNNSSEFMA(s0, w0, z0);
            z3  = MNNSSEFMA(s0, w1, z3);
            z6  = MNNSSEFMA(s0, w2, z6);
            z9  = MNNSSEFMA(s0, w3, z9);
            z1  = MNNSSEFMA(s1, w0, z1);
            z4  = MNNSSEFMA(s1, w1, z4);
            z7  = MNNSSEFMA(s1, w2, z7);
            z10 = MNNSSEFMA(s1, w3, z10);
        }
        TRANPOSE_SAVE(0, 0, z0, z3, z6, z9);
        TRANPOSE_SAVE(0, 1, z1, z4, z7, z10);
    }
}

static void _SSE_MNNPackedMatMul_4_int4(float* C, const float* A, const uint8_t* B, const size_t* parameter, const float* k, const float* b) {
    auto aStride      = parameter[0] / sizeof(float);
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(float);
    auto bExtraStride = parameter[5] / sizeof(float);
    auto bStride      = bExtraStride + l * 4;
    auto hC4          = UP_DIV(h, 4);
    float ws_tmp[4];
    for (int y = 0; y < hC4; ++y) {
        auto weight = B + y * bStride / 2;
        auto dst    = C + y * cStride;
        auto alpha  = _mm_loadu_ps(k + y * 4);
        auto bias   = _mm_loadu_ps(b + y * 4);
        auto s0     = _mm_loadu_ps(A + 0 * aStride);
        auto ws  = _load_int4x4(weight, alpha, bias);
        _mm_storeu_ps(ws_tmp, ws);
        auto w0  = _mm_set1_ps(ws_tmp[0]);
        auto w1  = _mm_set1_ps(ws_tmp[1]);
        auto w2  = _mm_set1_ps(ws_tmp[2]);
        auto w3  = _mm_set1_ps(ws_tmp[3]);
        auto z0     = _mm_mul_ps(s0, w0);
        auto z3     = _mm_mul_ps(s0, w1);
        auto z6     = _mm_mul_ps(s0, w2);
        auto z9     = _mm_mul_ps(s0, w3);

        for (int sy = 1; sy < l; ++sy) {
            s0 = _mm_loadu_ps(A + sy * aStride);
            ws = _load_int4x4(weight + sy * 2, alpha, bias);
            _mm_storeu_ps(ws_tmp, ws);
            w0 = _mm_set1_ps(ws_tmp[0]);
            w1 = _mm_set1_ps(ws_tmp[1]);
            w2 = _mm_set1_ps(ws_tmp[2]);
            w3 = _mm_set1_ps(ws_tmp[3]);
            z0 = MNNSSEFMA(s0, w0, z0);
            z3 = MNNSSEFMA(s0, w1, z3);
            z6 = MNNSSEFMA(s0, w2, z6);
            z9 = MNNSSEFMA(s0, w3, z9);
        }
        _MM_TRANSPOSE4_PS(z0, z3, z6, z9);
        _mm_storeu_ps(dst + 4 * 0, z0);
        _mm_storeu_ps(dst + 4 * 1, z3);
        _mm_storeu_ps(dst + 4 * 2, z6);
        _mm_storeu_ps(dst + 4 * 3, z9);
    }
}

static void _SSE_MNNPackednMatMulRemainCommon_int4(float* C, const float* A, const float* fB, size_t eSize, const size_t* parameter,
                                                   const float* postParameters, const float* bias, const float* k, const float* b) {
    auto B            = reinterpret_cast<const uint8_t*>(fB);
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(float);
    auto bExtraStride = parameter[5] / sizeof(float);
    auto bStride      = bExtraStride + l * 4;
    auto hC4          = UP_DIV(h, 4);
    auto es           = eSize;
    auto oC           = C;
    auto aStride      = parameter[0] / sizeof(float);
    if (eSize >= 8) {
        _SSE_MNNPackedMatMul_8_int4(C, A, B, parameter, k, b);
        eSize -= 8;
        C += 8 * 4;
        A += 8;
    }
    if (eSize >= 4) {
        _SSE_MNNPackedMatMul_4_int4(C, A, B, parameter, k, b);
        eSize -= 4;
        C += 4 * 4;
        A += 4;
    }
    for (int x = 0; x < eSize; ++x) {
        auto src = A + x;
        for (int y = 0; y < hC4; ++y) {
            auto weight = B + y * bStride / 2;
            auto dst    = C + y * cStride + x * 4;
            auto alpha  = _mm_loadu_ps(k + y * 4);
            auto bias   = _mm_loadu_ps(b + y * 4);
            auto sum    = _mm_set1_ps(0.0f);
            for (int sy = 0; sy < l; ++sy) {
                auto s = _mm_set1_ps(src[sy * aStride]);
                auto w = _load_int4x4(weight + sy * 2, alpha, bias);
                sum    = MNNSSEFMA(s, w, sum);
            }
            _mm_storeu_ps(dst, sum);
        }
    }
}
//----------------------- MatMul(float, int8) Functions ---------------------------//
static inline __m128 _load_int8x4(const int8_t* src, __m128 alpha, __m128 bias) {
    int iw0     = src[0];
    int iw1     = src[1];
    int iw2     = src[2];
    int iw3     = src[3];
    auto ws     = _mm_set_ps(iw3, iw2, iw1, iw0);
    ws          = _mm_add_ps(_mm_mul_ps(ws, alpha), bias);
    return ws;
}

static void _SSE_MNNPackedMatMul_12_int8(float* C, const float* A, const float* fB, const size_t* parameter, const float* k, const float* b) {
    auto B            = reinterpret_cast<const int8_t*>(fB);
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(float);
    auto bExtraStride = parameter[5] / sizeof(float);
    auto bStride      = bExtraStride + l * 4;
    auto hC4          = UP_DIV(h, 4);
    float ws_tmp[4];
    for (int y = 0; y < hC4; ++y) {
        auto weight = B + y * bStride;
        auto dst    = C + y * cStride;
        auto alpha  = _mm_loadu_ps(k + y * 4);
        auto bias   = _mm_loadu_ps(b + y * 4);
        auto s0  = _mm_loadu_ps(A + 0 * 12);
        auto s1  = _mm_loadu_ps(A + 0 * 12 + 4);
        auto s2  = _mm_loadu_ps(A + 0 * 12 + 8);
        auto ws  = _load_int8x4(weight, alpha, bias);
        _mm_storeu_ps(ws_tmp, ws);
        auto w0  = _mm_set1_ps(ws_tmp[0]);
        auto w1  = _mm_set1_ps(ws_tmp[1]);
        auto w2  = _mm_set1_ps(ws_tmp[2]);
        auto w3  = _mm_set1_ps(ws_tmp[3]);
        auto z0  = _mm_mul_ps(s0, w0);
        auto z1  = _mm_mul_ps(s1, w0);
        auto z2  = _mm_mul_ps(s2, w0);
        auto z3  = _mm_mul_ps(s0, w1);
        auto z4  = _mm_mul_ps(s1, w1);
        auto z5  = _mm_mul_ps(s2, w1);
        auto z6  = _mm_mul_ps(s0, w2);
        auto z7  = _mm_mul_ps(s1, w2);
        auto z8  = _mm_mul_ps(s2, w2);
        auto z9  = _mm_mul_ps(s0, w3);
        auto z10 = _mm_mul_ps(s1, w3);
        auto z11 = _mm_mul_ps(s2, w3);

        for (int sy = 1; sy < l; ++sy) {
            s0  = _mm_loadu_ps(A + sy * 12);
            s1  = _mm_loadu_ps(A + sy * 12 + 4);
            s2  = _mm_loadu_ps(A + sy * 12 + 8);
            ws = _load_int8x4(weight + sy * 4, alpha, bias);
            _mm_storeu_ps(ws_tmp, ws);
            w0 = _mm_set1_ps(ws_tmp[0]);
            w1 = _mm_set1_ps(ws_tmp[1]);
            w2 = _mm_set1_ps(ws_tmp[2]);
            w3 = _mm_set1_ps(ws_tmp[3]);
            z0  = MNNSSEFMA(s0, w0, z0);
            z1  = MNNSSEFMA(s1, w0, z1);
            z2  = MNNSSEFMA(s2, w0, z2);
            z3  = MNNSSEFMA(s0, w1, z3);
            z4  = MNNSSEFMA(s1, w1, z4);
            z5  = MNNSSEFMA(s2, w1, z5);
            z6  = MNNSSEFMA(s0, w2, z6);
            z7  = MNNSSEFMA(s1, w2, z7);
            z8  = MNNSSEFMA(s2, w2, z8);
            z9  = MNNSSEFMA(s0, w3, z9);
            z10 = MNNSSEFMA(s1, w3, z10);
            z11 = MNNSSEFMA(s2, w3, z11);
        }
        TRANPOSE_SAVE(0, 0, z0, z3, z6, z9);
        TRANPOSE_SAVE(0, 1, z1, z4, z7, z10);
        TRANPOSE_SAVE(0, 2, z2, z5, z8, z11);
    }
}

static void _SSE_MNNPackedMatMul_8_int8(float* C, const float* A, const int8_t* B, const size_t* parameter, const float* k, const float* b) {
    auto aStride      = parameter[0] / sizeof(float);
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(float);
    auto bExtraStride = parameter[5] / sizeof(float);
    auto bStride      = bExtraStride + l * 4;
    auto hC4          = UP_DIV(h, 4);
    float ws_tmp[4];
    for (int y = 0; y < hC4; ++y) {
        auto weight = B + y * bStride;
        auto dst    = C + y * cStride;
        auto alpha  = _mm_loadu_ps(k + y * 4);
        auto bias   = _mm_loadu_ps(b + y * 4);
        auto s0  = _mm_loadu_ps(A + 0 * aStride);
        auto s1  = _mm_loadu_ps(A + 0 * aStride + 4);
        auto ws  = _load_int8x4(weight, alpha, bias);
        _mm_storeu_ps(ws_tmp, ws);
        auto w0  = _mm_set1_ps(ws_tmp[0]);
        auto w1  = _mm_set1_ps(ws_tmp[1]);
        auto w2  = _mm_set1_ps(ws_tmp[2]);
        auto w3  = _mm_set1_ps(ws_tmp[3]);
        auto z0  = _mm_mul_ps(s0, w0);
        auto z3  = _mm_mul_ps(s0, w1);
        auto z6  = _mm_mul_ps(s0, w2);
        auto z9  = _mm_mul_ps(s0, w3);
        auto z1  = _mm_mul_ps(s1, w0);
        auto z4  = _mm_mul_ps(s1, w1);
        auto z7  = _mm_mul_ps(s1, w2);
        auto z10 = _mm_mul_ps(s1, w3);

        for (int sy = 1; sy < l; ++sy) {
            s0  = _mm_loadu_ps(A + sy * aStride);
            s1  = _mm_loadu_ps(A + sy * aStride + 4);
            ws = _load_int8x4(weight + sy * 4, alpha, bias);
            _mm_storeu_ps(ws_tmp, ws);
            w0 = _mm_set1_ps(ws_tmp[0]);
            w1 = _mm_set1_ps(ws_tmp[1]);
            w2 = _mm_set1_ps(ws_tmp[2]);
            w3 = _mm_set1_ps(ws_tmp[3]);
            z0  = MNNSSEFMA(s0, w0, z0);
            z3  = MNNSSEFMA(s0, w1, z3);
            z6  = MNNSSEFMA(s0, w2, z6);
            z9  = MNNSSEFMA(s0, w3, z9);
            z1  = MNNSSEFMA(s1, w0, z1);
            z4  = MNNSSEFMA(s1, w1, z4);
            z7  = MNNSSEFMA(s1, w2, z7);
            z10 = MNNSSEFMA(s1, w3, z10);
        }
        TRANPOSE_SAVE(0, 0, z0, z3, z6, z9);
        TRANPOSE_SAVE(0, 1, z1, z4, z7, z10);
    }
}

static void _SSE_MNNPackedMatMul_4_int8(float* C, const float* A, const int8_t* B, const size_t* parameter, const float* k, const float* b) {
    auto aStride      = parameter[0] / sizeof(float);
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(float);
    auto bExtraStride = parameter[5] / sizeof(float);
    auto bStride      = bExtraStride + l * 4;
    auto hC4          = UP_DIV(h, 4);
    float ws_tmp[4];
    for (int y = 0; y < hC4; ++y) {
        auto weight = B + y * bStride;
        auto dst    = C + y * cStride;
        auto alpha  = _mm_loadu_ps(k + y * 4);
        auto bias   = _mm_loadu_ps(b + y * 4);
        auto s0     = _mm_loadu_ps(A + 0 * aStride);
        auto ws  = _load_int8x4(weight, alpha, bias);
        _mm_storeu_ps(ws_tmp, ws);
        auto w0  = _mm_set1_ps(ws_tmp[0]);
        auto w1  = _mm_set1_ps(ws_tmp[1]);
        auto w2  = _mm_set1_ps(ws_tmp[2]);
        auto w3  = _mm_set1_ps(ws_tmp[3]);
        auto z0     = _mm_mul_ps(s0, w0);
        auto z3     = _mm_mul_ps(s0, w1);
        auto z6     = _mm_mul_ps(s0, w2);
        auto z9     = _mm_mul_ps(s0, w3);

        for (int sy = 1; sy < l; ++sy) {
            s0 = _mm_loadu_ps(A + sy * aStride);
            ws = _load_int8x4(weight + sy * 4, alpha, bias);
            _mm_storeu_ps(ws_tmp, ws);
            w0 = _mm_set1_ps(ws_tmp[0]);
            w1 = _mm_set1_ps(ws_tmp[1]);
            w2 = _mm_set1_ps(ws_tmp[2]);
            w3 = _mm_set1_ps(ws_tmp[3]);
            z0 = MNNSSEFMA(s0, w0, z0);
            z3 = MNNSSEFMA(s0, w1, z3);
            z6 = MNNSSEFMA(s0, w2, z6);
            z9 = MNNSSEFMA(s0, w3, z9);
        }
        _MM_TRANSPOSE4_PS(z0, z3, z6, z9);
        _mm_storeu_ps(dst + 4 * 0, z0);
        _mm_storeu_ps(dst + 4 * 1, z3);
        _mm_storeu_ps(dst + 4 * 2, z6);
        _mm_storeu_ps(dst + 4 * 3, z9);
    }
}

static void _SSE_MNNPackednMatMulRemainCommon_int8(float* C, const float* A, const float* fB, size_t eSize, const size_t* parameter,
                                                   const float* postParameters, const float* bias, const float* k, const float* b) {
    auto B            = reinterpret_cast<const int8_t*>(fB);
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(float);
    auto bExtraStride = parameter[5] / sizeof(float);
    auto bStride      = bExtraStride + l * 4;
    auto hC4          = UP_DIV(h, 4);
    auto es           = eSize;
    auto oC           = C;
    auto aStride      = parameter[0] / sizeof(float);
    if (eSize >= 8) {
        _SSE_MNNPackedMatMul_8_int8(C, A, B, parameter, k, b);
        eSize -= 8;
        C += 8 * 4;
        A += 8;
    }
    if (eSize >= 4) {
        _SSE_MNNPackedMatMul_4_int8(C, A, B, parameter, k, b);
        eSize -= 4;
        C += 4 * 4;
        A += 4;
    }
    for (int x = 0; x < eSize; ++x) {
        auto src = A + x;
        for (int y = 0; y < hC4; ++y) {
            auto weight = B + y * bStride;
            auto dst    = C + y * cStride + x * 4;
            auto alpha  = _mm_loadu_ps(k + y * 4);
            auto bias   = _mm_loadu_ps(b + y * 4);
            auto sum    = _mm_set1_ps(0.0f);
            for (int sy = 0; sy < l; ++sy) {
                auto s = _mm_set1_ps(src[sy * aStride]);
                auto w = _load_int8x4(weight + sy * 4, alpha, bias);
                sum    = MNNSSEFMA(s, w, sum);
            }
            _mm_storeu_ps(dst, sum);
        }
    }
}
// int4 -> int8
static inline __m128i _load_int4_to_int8(const uint8_t* src) {
    uint8_t c = 0xf;
    int32_t data[4];
    int8_t temp[16];
    for (int i = 0; i < 8; ++i) {
        temp[2 * i] = (src[i] >> 4) - 8;
        temp[2 * i +1] = (src[i] & c) - 8;
    }
    auto int8_tx16 = _mm_loadu_si128((const __m128i*)temp);
    return int8_tx16;
}
static void _SSE_MNNGemmHybrid_int4(float* C, const int8_t* A, const int8_t* B, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad, size_t realSize, const float** param) {
    // C:(oc/4,N,4) A:(ic/4,N,4) B:(oc/4,ic/4,4,4)
    int pack = 4;
    __m128i zero_128i = _mm_set1_epi32(0);
    size_t weight_step = src_depth_quad * pack * pack * 0.5;
    size_t weight_stride = pack * pack * 0.5;
    const float* alpha_ptr = param[0];
    const float* zero_ptr = param[1];
    const float* bias_ptr = param[2];
    const float* sums_ptr = param[3];
    const float* scale_ptr = param[4];
    std::vector<int8_t> tmpsrc(16, 0);
    
    for (int ci = 0; ci < dst_depth_quad; ++ci) {
        float* dstZ = C + ci * pack * realSize;
        const int8_t*    weight = B + ci * weight_step;
        auto alpha = alpha_ptr + ci * pack;
        auto zero  = zero_ptr + ci * pack;
        auto bias  = bias_ptr + ci * pack;
        __m128 alphaValue = _mm_load_ps(alpha);
        //const float* sums = param[2];
        for (int j = 0; j < realSize; ++j) {
            const float* sums = sums_ptr + j;
            const float* scale = scale_ptr + j;
            float* dstX = dstZ + j * pack;
            __m128i sum4 = _mm_set1_epi32(0);
            __m128  scaleValue = _mm_set1_ps(scale[0]);
            __m128 biasValue  = _mm_add_ps(_mm_load_ps(bias), _mm_mul_ps(_mm_load_ps(zero), _mm_set1_ps(sums[0])));
            const int8_t* srcBatch = A + j * pack;
            for (int k = 0; k < src_depth_quad; ++k) {
                const int8_t* srcZ = srcBatch + k * pack * realSize;
                const uint8_t* weightZ = (uint8_t*)weight + k * weight_stride;
                auto w0 = _load_int4_to_int8(weightZ);
                
                ::memcpy(tmpsrc.data(), srcZ, 4 * sizeof(int8_t));
                auto s0 = _mm_loadu_si128((const __m128i*)tmpsrc.data());
                // src,weight: int8->int16
                auto s0_16 = _mm_srai_epi16(_mm_unpacklo_epi8(zero_128i, s0), 8);
                auto w0_16 = _mm_srai_epi16(_mm_unpacklo_epi8(zero_128i, w0), 8);
                auto w1_16 = _mm_srai_epi16(_mm_unpackhi_epi8(zero_128i, w0), 8);
                auto w2_16 = _mm_unpackhi_epi64(w0_16, zero_128i);
                auto w3_16 = _mm_unpackhi_epi64(w1_16, zero_128i);
                
                auto oc0 = _mm_madd_epi16(s0_16, w0_16);
                auto oc1 = _mm_madd_epi16(s0_16, w2_16);
                auto oc2 = _mm_madd_epi16(s0_16, w1_16);
                auto oc3 = _mm_madd_epi16(s0_16, w3_16);
                
                auto d0 = _mm_unpacklo_epi32(oc0, oc1);
                auto d1 = _mm_unpackhi_epi32(oc0, oc1);
                auto d2 = _mm_unpacklo_epi32(oc2, oc3);
                auto d3 = _mm_unpackhi_epi32(oc2, oc3);
                
                auto e0 = _mm_unpacklo_epi64(d0, d2);
                auto e1 = _mm_unpackhi_epi64(d0, d2);
                auto e2 = _mm_unpacklo_epi64(d1, d3);
                auto e3 = _mm_unpackhi_epi64(d1, d3);
                
                e0 = _mm_add_epi32(e0, e1);
                e2 = _mm_add_epi32(e2, e3);
                e0 = _mm_add_epi32(e0, e2);
                
                sum4 = _mm_add_epi32(e0, sum4);
                
            }
            __m128 f0 = _mm_cvtepi32_ps(sum4);
            __m128 fs = _mm_mul_ps(_mm_mul_ps(f0, scaleValue), alphaValue);
            fs = _mm_add_ps(biasValue, fs);
            _mm_storeu_ps(dstX, fs);
        }
    }
}
static void _SSE_MNNGemmHybrid_int8(float* C, const int8_t* A, const int8_t* B, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad, size_t realSize, const float** param) {
    // C:(oc/4,N,4) A:(ic/4,N,4) B:(oc/4,ic/4,4,4)
    int pack = 4;
    __m128i zero_128i = _mm_set1_epi32(0);
    size_t weight_step = src_depth_quad * pack * pack;
    size_t weight_stride = pack * pack;
    const float* alpha_ptr = param[0];
    const float* zero_ptr = param[1];
    const float* bias_ptr = param[2];
    const float* sums_ptr = param[3];
    const float* scale_ptr = param[4];
    std::vector<int8_t> tmpsrc(16, 0);
    
    for (int ci = 0; ci < dst_depth_quad; ++ci) {
        float* dstZ = C + ci * pack * realSize;
        const int8_t*    weight = B + ci * weight_step;
        auto alpha = alpha_ptr + ci * pack;
        auto zero  = zero_ptr + ci * pack;
        auto bias  = bias_ptr + ci * pack;
        __m128 alphaValue = _mm_load_ps(alpha);
        //const float* sums = param[2];
        for (int j = 0; j < realSize; ++j) {
            const float* sums = sums_ptr + j;
            const float* scale = scale_ptr + j;
            float* dstX = dstZ + j * pack;

            __m128i sum4 = _mm_set1_epi32(0);
            __m128  scaleValue = _mm_set1_ps(scale[0]);
            __m128 biasValue  = _mm_add_ps(_mm_load_ps(bias), _mm_mul_ps(_mm_load_ps(zero), _mm_set1_ps(sums[0])));
            const int8_t* srcBatch = A + j * pack;
            for (int k = 0; k < src_depth_quad; ++k) {
                const int8_t* srcZ = srcBatch + k * pack * realSize;
                const int8_t* weightZ = weight + k * weight_stride;
                auto w0 = _mm_loadu_si128((__m128i*)(weightZ)); // 16xint8_t weight
                
                ::memcpy(tmpsrc.data(), srcZ, 4 * sizeof(int8_t));
                auto s0 = _mm_loadu_si128((const __m128i*)tmpsrc.data());
                // src,weight: int8->int16
//                auto s0_16 = _mm_unpacklo_epi8(s0, zero_128i);
                auto s0_16 = _mm_srai_epi16(_mm_unpacklo_epi8(zero_128i, s0), 8);
                auto w0_16 = _mm_srai_epi16(_mm_unpacklo_epi8(zero_128i, w0), 8);
                auto w1_16 = _mm_srai_epi16(_mm_unpackhi_epi8(zero_128i, w0), 8);
                auto w2_16 = _mm_unpackhi_epi64(w0_16, zero_128i);
                auto w3_16 = _mm_unpackhi_epi64(w1_16, zero_128i);

                auto oc0 = _mm_madd_epi16(s0_16, w0_16);
                auto oc1 = _mm_madd_epi16(s0_16, w2_16);
                auto oc2 = _mm_madd_epi16(s0_16, w1_16);
                auto oc3 = _mm_madd_epi16(s0_16, w3_16);
                
                auto d0 = _mm_unpacklo_epi32(oc0, oc1);
                auto d1 = _mm_unpackhi_epi32(oc0, oc1);
                auto d2 = _mm_unpacklo_epi32(oc2, oc3);
                auto d3 = _mm_unpackhi_epi32(oc2, oc3);
                
                auto e0 = _mm_unpacklo_epi64(d0, d2);
                auto e1 = _mm_unpackhi_epi64(d0, d2);
                auto e2 = _mm_unpacklo_epi64(d1, d3);
                auto e3 = _mm_unpackhi_epi64(d1, d3);
                
                e0 = _mm_add_epi32(e0, e1);
                e2 = _mm_add_epi32(e2, e3);
                e0 = _mm_add_epi32(e0, e2);

                sum4 = _mm_add_epi32(e0, sum4);
                
            }
            __m128 f0 = _mm_cvtepi32_ps(sum4);
            __m128 fs = _mm_mul_ps(_mm_mul_ps(f0, scaleValue), alphaValue);
            fs = _mm_add_ps(biasValue, fs);
            _mm_storeu_ps(dstX, fs);
        }
    }
}
#endif
