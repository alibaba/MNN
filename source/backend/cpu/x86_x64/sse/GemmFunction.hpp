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
    ws          = _mm_add_ps(_mm_mul_ps(ws, alpha), bias);
    return ws;
}

static void _SSE_MNNPackedMatMul_12_int4(float* C, const float* A, const float* fB, const size_t* parameter, const float* k, const float* b) {
    auto B            = reinterpret_cast<const uint8_t*>(fB);
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(float);
    float weightBytes = 0.5; // sizeof(int4_t)
    auto bExtraStride = static_cast<int32_t>(parameter[5] / weightBytes);
    auto bStride      = bExtraStride + 4 * l;
    auto hC4          = UP_DIV(h, 4);
    auto blockId      = parameter[6];
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
        if (0 == blockId) {
            TRANPOSE_SAVE(0, 0, z0, z3, z6, z9);
            TRANPOSE_SAVE(0, 1, z1, z4, z7, z10);
            TRANPOSE_SAVE(0, 2, z2, z5, z8, z11);
        } else {
            FMLA_TRANPOSE_SAVE(0, 0, z0, z3, z6, z9);
            FMLA_TRANPOSE_SAVE(0, 1, z1, z4, z7, z10);
            FMLA_TRANPOSE_SAVE(0, 2, z2, z5, z8, z11);
        }
    }
}

static void _SSE_MNNPackedMatMul_8_int4(float* C, const float* A, const uint8_t* B, const size_t* parameter, const float* k, const float* b) {
    auto aStride      = parameter[0] / sizeof(float);
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(float);
    float weightBytes = 0.5; // sizeof(int4_t)
    auto bExtraStride = static_cast<int32_t>(parameter[5] / weightBytes);
    auto bStride      = bExtraStride + 4 * l;
    auto blockId      = parameter[6];
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
        if (0 == blockId) {
            TRANPOSE_SAVE(0, 0, z0, z3, z6, z9);
            TRANPOSE_SAVE(0, 1, z1, z4, z7, z10);
        } else {
            FMLA_TRANPOSE_SAVE(0, 0, z0, z3, z6, z9);
            FMLA_TRANPOSE_SAVE(0, 1, z1, z4, z7, z10);
        }
    }
}

static void _SSE_MNNPackedMatMul_4_int4(float* C, const float* A, const uint8_t* B, const size_t* parameter, const float* k, const float* b) {
    auto aStride      = parameter[0] / sizeof(float);
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(float);
    float weightBytes = 0.5;
    auto bExtraStride = static_cast<int32_t>(parameter[5] / weightBytes);
    auto bStride      = bExtraStride + 4 * l;
    auto blockId      = parameter[6];
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
        if (0 == blockId) {
            _mm_storeu_ps(dst + 4 * 0, z0);
            _mm_storeu_ps(dst + 4 * 1, z3);
            _mm_storeu_ps(dst + 4 * 2, z6);
            _mm_storeu_ps(dst + 4 * 3, z9);
        } else {
            auto t0 = _mm_loadu_ps(dst + 4 * 0);
            auto t1 = _mm_loadu_ps(dst + 4 * 1);
            auto t2 = _mm_loadu_ps(dst + 4 * 2);
            auto t3 = _mm_loadu_ps(dst + 4 * 3);

            z0 = _mm_add_ps(z0, t0);
            z3 = _mm_add_ps(z3, t1);
            z6 = _mm_add_ps(z6, t2);
            z9 = _mm_add_ps(z9, t3);
            _mm_storeu_ps(dst + 4 * 0, z0);
            _mm_storeu_ps(dst + 4 * 1, z3);
            _mm_storeu_ps(dst + 4 * 2, z6);
            _mm_storeu_ps(dst + 4 * 3, z9);
        }
    }
}

static void _SSE_MNNPackednMatMulRemainCommon_int4(float* C, const float* A, const float* fB, size_t eSize, const size_t* parameter,
                                                   const float* postParameters, const float* bias, const float* k, const float* b) {
    auto B            = reinterpret_cast<const uint8_t*>(fB);
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(float);
    float weightBytes = 0.5; // sizeof(int4_t)
    auto bExtraStride = static_cast<int32_t>(parameter[5] / weightBytes); // parameter[5]/weightBytes
    auto bStride      = bExtraStride + 4 * l;
    auto blockId      = parameter[6];
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
            if (0 == blockId) {
                _mm_storeu_ps(dst, sum);
            } else {
                auto tmp = _mm_loadu_ps(dst);
                sum = _mm_add_ps(sum, tmp);
                _mm_storeu_ps(dst, sum);
            }
            

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
    float weightBytes = 1; // sizeof(int8_t)
    auto bExtraStride = static_cast<int32_t>(parameter[5] / weightBytes);
    auto bStride      = bExtraStride + 4 * l;
    auto blockId      = parameter[6];
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
        if (0 == blockId) {
            TRANPOSE_SAVE(0, 0, z0, z3, z6, z9);
            TRANPOSE_SAVE(0, 1, z1, z4, z7, z10);
            TRANPOSE_SAVE(0, 2, z2, z5, z8, z11);
        } else {
            FMLA_TRANPOSE_SAVE(0, 0, z0, z3, z6, z9);
            FMLA_TRANPOSE_SAVE(0, 1, z1, z4, z7, z10);
            FMLA_TRANPOSE_SAVE(0, 2, z2, z5, z8, z11);
        }
        
    }
}

static void _SSE_MNNPackedMatMul_8_int8(float* C, const float* A, const int8_t* B, const size_t* parameter, const float* k, const float* b) {
    auto aStride      = parameter[0] / sizeof(float);
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(float);
    float weightBytes = 1; // sizeof(int8_t)
    auto bExtraStride = static_cast<int32_t>(parameter[5] / weightBytes);
    auto bStride      = bExtraStride + 4 * l;
    auto blockId      = parameter[6];
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
        if (0 == blockId) {
            TRANPOSE_SAVE(0, 0, z0, z3, z6, z9);
            TRANPOSE_SAVE(0, 1, z1, z4, z7, z10);
        } else {
            FMLA_TRANPOSE_SAVE(0, 0, z0, z3, z6, z9);
            FMLA_TRANPOSE_SAVE(0, 1, z1, z4, z7, z10);
        }
        
    }
}

static void _SSE_MNNPackedMatMul_4_int8(float* C, const float* A, const int8_t* B, const size_t* parameter, const float* k, const float* b) {
    auto aStride      = parameter[0] / sizeof(float);
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(float);
    float weightBytes = 1; // sizeof(int8_t)
    auto bExtraStride = static_cast<int32_t>(parameter[5] / weightBytes);
    auto bStride      = bExtraStride + 4 * l;
    auto blockId      = parameter[6];
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
        if (0 == blockId) {
            _mm_storeu_ps(dst + 4 * 0, z0);
            _mm_storeu_ps(dst + 4 * 1, z3);
            _mm_storeu_ps(dst + 4 * 2, z6);
            _mm_storeu_ps(dst + 4 * 3, z9);
        } else {
            auto t0 = _mm_loadu_ps(dst + 4 * 0);
            auto t1 = _mm_loadu_ps(dst + 4 * 1);
            auto t2 = _mm_loadu_ps(dst + 4 * 2);
            auto t3 = _mm_loadu_ps(dst + 4 * 3);
            
            z0 = _mm_add_ps(t0, z0);
            z3 = _mm_add_ps(t1, z3);
            z6 = _mm_add_ps(t2, z6);
            z9 = _mm_add_ps(t3, z9);
            
            _mm_storeu_ps(dst + 4 * 0, z0);
            _mm_storeu_ps(dst + 4 * 1, z3);
            _mm_storeu_ps(dst + 4 * 2, z6);
            _mm_storeu_ps(dst + 4 * 3, z9);
        }
    }
}

static void _SSE_MNNPackednMatMulRemainCommon_int8(float* C, const float* A, const float* fB, size_t eSize, const size_t* parameter,
                                                   const float* postParameters, const float* bias, const float* k, const float* b) {
    auto B            = reinterpret_cast<const int8_t*>(fB);
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(float);
    float weightBytes = 1; // sizeof(int8_t)
    auto bExtraStride = static_cast<int32_t>(parameter[5] / weightBytes);
    auto bStride      = bExtraStride + 4 * l;
    auto hC4          = UP_DIV(h, 4);
    auto es           = eSize;
    auto oC           = C;
    auto aStride      = parameter[0] / sizeof(float);
    auto blockId      = parameter[6];
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
            if (blockId == 0) {
                _mm_storeu_ps(dst, sum);
            } else {
                auto t = _mm_loadu_ps(dst);
                sum = _mm_add_ps(sum, t);
                _mm_storeu_ps(dst, sum);
            }
        }
    }
}
#endif
