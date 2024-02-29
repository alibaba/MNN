//
//  calib3d.cpp
//  MNN
//
//  Created by MNN on 2021/08/26.
//  Copyright © 2018][Alibaba Group Holding Limited
//

#include <math/Matrix.hpp>
#include "cv/calib3d.hpp"
#include "cv/imgproc/geometric.hpp"
#include <MNN/expr/NeuralNetWorkOp.hpp>
#include <MNN/expr/MathOp.hpp>
#include <cmath>
#include <limits>
#define DUMP(x)\
{\
    printf(#x "\n");\
    Math::Matrix::print(x.get());\
}
namespace MNN {
namespace CV {
// helper functions
inline static float det3x3(const float* ptr) {
    #define M(r, c) ptr[r * 3 + c]
    return M(0, 0) * (M(1, 1) * M(2, 2) - M(1, 2) * M(2, 1)) -
           M(0, 1) * (M(1, 0) * M(2, 2) - M(1, 2) * M(2, 0)) +
           M(0, 2) * (M(1, 0) * M(2, 1) - M(1, 1) * M(2, 0));
}
inline static float det9x1(const float* r) {
    return r[0]*r[4]*r[8] + r[1]*r[5]*r[6] + r[2]*r[3]*r[7] - r[6]*r[4]*r[2] - r[7]*r[5]*r[0] - r[8]*r[3]*r[1];
}
inline static float orthogonalityError(const float a[9]) {
    float sq_norm_a1 = a[0] * a[0] + a[1] * a[1] + a[2] * a[2],
          sq_norm_a2 = a[3] * a[3] + a[4] * a[4] + a[5] * a[5],
          sq_norm_a3 = a[6] * a[6] + a[7] * a[7] + a[8] * a[8];
    float dot_a1a2 = a[0] * a[3] + a[1] * a[4] + a[2] * a[5],
          dot_a1a3 = a[0] * a[6] + a[1] * a[7] + a[2] * a[8],
          dot_a2a3 = a[3] * a[6] + a[4] * a[7] + a[5] * a[8];
    return (sq_norm_a1 - 1) * (sq_norm_a1 - 1) + (sq_norm_a2 - 1) * (sq_norm_a2 - 1) + (sq_norm_a3 - 1) * (sq_norm_a3 - 1) +
           2 * (dot_a1a2*dot_a1a2 + dot_a1a3*dot_a1a3 + dot_a2a3*dot_a2a3);
}

static void orthogonal(float* at, float* vt, int i, int j, int row, int col, bool& pass) {
    auto ai = at + i * row;
    auto aj = at + j * row;
    auto vi = vt + i * col;
    auto vj = vt + j * col;
    float norm = 0.f, normi = 0.f, normj = 0.f;
    for (int i = 0; i < col; i++) {
        norm += ai[i] * aj[i];
        normi += ai[i] * ai[i];
        normj += aj[i] * aj[i];
    }
    constexpr float eps = std::numeric_limits<float>::epsilon() * 2;
    if (std::abs(norm) < eps * std::sqrt(normi * normj)) {
        return;
    }
    pass = false;
    float tao = (normi - normj) / (2.0 * norm);
    float tan = (tao < 0 ? -1 : 1) / (fabs(tao) + sqrt(1 + pow(tao, 2)));
    float cos = 1 / sqrt(1 + pow(tan, 2));
    float sin = cos * tan;
    bool swap = normi < normj;
    for (int i = 0; i < col; i++) {
        float nai = ai[i];
        float naj = aj[i];
        float nvi = vi[i];
        float nvj = vj[i];
        if (swap) {
            std::swap(nai, naj);
            std::swap(nvi, nvj);
        }
        ai[i] = nai * cos + naj * sin;
        aj[i] = naj * cos - nai * sin;
        vi[i] = nvi * cos + nvj * sin;
        vj[i] = nvj * cos - nvi * sin;
    }
}
inline static void svdMatrix(float* w, float* u, float* vt, float* a, int M, int N) {
    int size = M * N;
    std::vector<float> AT_(size);
    float* at = AT_.data();
    // init at
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            at[i * M + j] = a[j * N + i];
        }
    }
    // init vt
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            vt[i * N + j] = (i == j);
        }
    }
    constexpr int max_iteration = 30;
    for (int iter = 0; iter < max_iteration; iter++) {
        bool pass = true;
        for (int i = 0; i < N; i++) {
            for (int j = i + 1; j < N; j++) {
                orthogonal(at, vt, i, j, M, N, pass);
            }
        }
        if (pass) break;
    }
    for (int i = 0; i < N; i++) {
        float norm = 0.f;
        for (int j = 0; j < N; j++) {
            auto tmp = at[i * N + j];
            norm += tmp * tmp;
        }
        norm = sqrt(norm);
        w[i] = norm;
    }
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            u[i * N + j] = at[j * N + i] / w[j];
        }
    }
}

void Rodrigues(float* dst, float* src) {
    float w_[9], u_[9], vt_[9];
    svdMatrix(w_, u_, vt_, src, 3, 3);
    float R[9];
    Math::Matrix::multi(R, u_, vt_, 3, 3, 3);
    float x = R[7] - R[5], y = R[2] - R[6], z = R[3] - R[1];
    float s = sqrt((x * x + y * y + z * z) * 0.25);
    float c = (R[0] + R[4] + R[8] - 1) * 0.5;
    c = c > 1. ? 1. : c < -1. ? -1. : c;
    float theta = acos(c);
    if (s < 1e-5) {
        if (c > 0) {
            x = y = z = 0;
        } else {
            x = sqrt(fmax((R[0] + 1) * 0.5, 0));
            y = sqrt(fmax((R[4] + 1) * 0.5, 0)) * (R[1] < 0 ? -1. : 1.);
            z = sqrt(fmax((R[8] + 1) * 0.5, 0)) * (R[2] < 0 ? -1. : 1.);
            if (fabs(x) < fabs(y) && fabs(x) < fabs(z) && (R[5] > 0) != (y * z > 0)) {
                z = -z;
            }
            theta /= sqrt(x * x + y * y + z * z);
            x *= theta;
            y *= theta;
            z *= theta;
        }
    } else {
        float vth = 1 / (2 * s);
        vth *= theta;
        x *= vth;
        y *= vth;
        z *= vth;
    }
    dst[0] = x;
    dst[1] = y;
    dst[2] = z;
}

void nearestRotationMatrix(float* r, float* e) {
    // VARP e = Express::Variable::create(Express::Expr::create(e_, false));
    float w[9] = {0};
    float u[9] = {0};
    float vt[9] = {0};
    float v[9] = {0};
    float e_t[9] = {0};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0;j < 3; ++j) {
            e_t[i * 3 + j] = e[j * 3 + i];
        }
    }
    svdMatrix(w, u, vt, e_t, 3, 3);
    for(int i = 0; i < 3; ++i) {
        for(int j = 0; j < 3; ++j) {
            v[i * 3 + j] = vt[j * 3 + i];
        }
    }
    float detuv[9] = {1, 0, 0, 0, 1, 0, 0, 0, det3x3(u) * det3x3(v)};
    float udetuv_[9] = {0};
    float R_[9] = {0};
    Math::Matrix::multi(udetuv_, u, detuv, 3, 3, 3);
    Math::Matrix::multi(R_, udetuv_, vt, 3, 3, 3);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            r[i * 3 + j] = R_[j * 3 + i];
        }
    }
}
void solveSQPSystem(float* delta_, float* r, float* omega) {
    float sqnorm_r1 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2],
           sqnorm_r2 = r[3] * r[3] + r[4] * r[4] + r[5] * r[5],
           sqnorm_r3 = r[6] * r[6] + r[7] * r[7] + r[8] * r[8];
    float dot_r1r2 = r[0] * r[3] + r[1] * r[4] + r[2] * r[5],
           dot_r1r3 = r[0] * r[6] + r[1] * r[7] + r[2] * r[8],
           dot_r2r3 = r[3] * r[6] + r[4] * r[7] + r[5] * r[8];
    std::vector<float> h(54), k(36), n(27), h_t(54);
#define H(r, c) h[r * 6 + c]
#define K(r, c) k[r * 6 + c]
#define N(r, c) n[r * 3 + c]
    // RowAndNullSpace start
    // 1. q1
    float norm_r1 = sqrt(sqnorm_r1);
    float inv_norm_r1 = norm_r1 > 1e-5 ? 1.0 / norm_r1 : 0.0;
    H(0, 0) = r[0] * inv_norm_r1;
    H(1, 0) = r[1] * inv_norm_r1;
    H(2, 0) = r[2] * inv_norm_r1;
    K(0, 0) = 2 * norm_r1;

    // 2. q2
    float norm_r2 = sqrt(sqnorm_r2);
    float inv_norm_r2 = 1.0 / norm_r2;
    H(3, 1) = r[3] * inv_norm_r2;
    H(4, 1) = r[4] * inv_norm_r2;
    H(5, 1) = r[5] * inv_norm_r2;
    K(1, 0) = 0;
    K(1, 1) = 2 * norm_r2;

    // 3. q3 = (r3'*q2)*q2 - (r3'*q1)*q1 ; q3 = q3/norm(q3)
    float norm_r3 = sqrt(sqnorm_r3);
    float inv_norm_r3 = 1.0 / norm_r3;
    H(6, 2) = r[6] * inv_norm_r3;
    H(7, 2) = r[7] * inv_norm_r3;
    H(8, 2) = r[8] * inv_norm_r3;
    K(2, 0) = K(2, 1) = 0;
    K(2, 2) = 2 * norm_r3;

    // 4. q4
    float dot_j4q1 = r[3] * H(0, 0) + r[4] * H(1, 0) + r[5] * H(2, 0),
           dot_j4q2 = r[0] * H(3, 1) + r[1] * H(4, 1) + r[2] * H(5, 1);

    H(0, 3) = r[3] - dot_j4q1 * H(0, 0);
    H(1, 3) = r[4] - dot_j4q1 * H(1, 0);
    H(2, 3) = r[5] - dot_j4q1 * H(2, 0);
    H(3, 3) = r[0] - dot_j4q2 * H(3, 1);
    H(4, 3) = r[1] - dot_j4q2 * H(4, 1);
    H(5, 3) = r[2] - dot_j4q2 * H(5, 1);
    float inv_norm_j4 = 1.0 / sqrt(H(0, 3) * H(0, 3) + H(1, 3) * H(1, 3) + H(2, 3) * H(2, 3) +
                                    H(3, 3) * H(3, 3) + H(4, 3) * H(4, 3) + H(5, 3) * H(5, 3));

    H(0, 3) *= inv_norm_j4;
    H(1, 3) *= inv_norm_j4;
    H(2, 3) *= inv_norm_j4;
    H(3, 3) *= inv_norm_j4;
    H(4, 3) *= inv_norm_j4;
    H(5, 3) *= inv_norm_j4;

    K(3, 0) = r[3] * H(0, 0) + r[4] * H(1, 0) + r[5] * H(2, 0);
    K(3, 1) = r[0] * H(3, 1) + r[1] * H(4, 1) + r[2] * H(5, 1);
    K(3, 2) = 0;
    K(3, 3) = r[3] * H(0, 3) + r[4] * H(1, 3) + r[5] * H(2, 3) + r[0] * H(3, 3) + r[1] * H(4, 3) + r[2] * H(5, 3);

    // 5. q5
    float dot_j5q2 = r[6] * H(3, 1) + r[7] * H(4, 1) + r[8] * H(5, 1),
           dot_j5q3 = r[3] * H(6, 2) + r[4] * H(7, 2) + r[5] * H(8, 2),
           dot_j5q4 = r[6] * H(3, 3) + r[7] * H(4, 3) + r[8] * H(5, 3);

    H(0, 4) = -dot_j5q4 * H(0, 3);
    H(1, 4) = -dot_j5q4 * H(1, 3);
    H(2, 4) = -dot_j5q4 * H(2, 3);
    H(3, 4) = r[6] - dot_j5q2 * H(3, 1) - dot_j5q4 * H(3, 3);
    H(4, 4) = r[7] - dot_j5q2 * H(4, 1) - dot_j5q4 * H(4, 3);
    H(5, 4) = r[8] - dot_j5q2 * H(5, 1) - dot_j5q4 * H(5, 3);
    H(6, 4) = r[3] - dot_j5q3 * H(6, 2);
    H(7, 4) = r[4] - dot_j5q3 * H(7, 2);
    H(8, 4) = r[5] - dot_j5q3 * H(8, 2);

    auto norm_4 = 0.f;
    for (int i = 0; i < 9; i++) {
        norm_4 += H(i, 4) * H(i, 4);
    }
    norm_4 = sqrt(norm_4);
    for (int i = 0; i < 9; i++) {
        H(i, 4) = H(i, 4) / norm_4;
    }

    K(4, 0) = 0;
    K(4, 1) = r[6] * H(3, 1) + r[7] * H(4, 1) + r[8] * H(5, 1);
    K(4, 2) = r[3] * H(6, 2) + r[4] * H(7, 2) + r[5] * H(8, 2);
    K(4, 3) = r[6] * H(3, 3) + r[7] * H(4, 3) + r[8] * H(5, 3);
    K(4, 4) = r[6] * H(3, 4) + r[7] * H(4, 4) + r[8] * H(5, 4) + r[3] * H(6, 4) + r[4] * H(7, 4) + r[5] * H(8, 4);

    // 4. q6
    float dot_j6q1 = r[6] * H(0, 0) + r[7] * H(1, 0) + r[8] * H(2, 0),
           dot_j6q3 = r[0] * H(6, 2) + r[1] * H(7, 2) + r[2] * H(8, 2),
           dot_j6q4 = r[6] * H(0, 3) + r[7] * H(1, 3) + r[8] * H(2, 3),
           dot_j6q5 = r[0] * H(6, 4) + r[1] * H(7, 4) + r[2] * H(8, 4) + r[6] * H(0, 4) + r[7] * H(1, 4) + r[8] * H(2, 4);

    H(0, 5) = r[6] - dot_j6q1 * H(0, 0) - dot_j6q4 * H(0, 3) - dot_j6q5 * H(0, 4);
    H(1, 5) = r[7] - dot_j6q1 * H(1, 0) - dot_j6q4 * H(1, 3) - dot_j6q5 * H(1, 4);
    H(2, 5) = r[8] - dot_j6q1 * H(2, 0) - dot_j6q4 * H(2, 3) - dot_j6q5 * H(2, 4);

    H(3, 5) = -dot_j6q5 * H(3, 4) - dot_j6q4 * H(3, 3);
    H(4, 5) = -dot_j6q5 * H(4, 4) - dot_j6q4 * H(4, 3);
    H(5, 5) = -dot_j6q5 * H(5, 4) - dot_j6q4 * H(5, 3);

    H(6, 5) = r[0] - dot_j6q3 * H(6, 2) - dot_j6q5 * H(6, 4);
    H(7, 5) = r[1] - dot_j6q3 * H(7, 2) - dot_j6q5 * H(7, 4);
    H(8, 5) = r[2] - dot_j6q3 * H(8, 2) - dot_j6q5 * H(8, 4);

    auto norm_5 = 0.f;
    for (int i = 0; i < 9; i++) {
        norm_5 += H(i, 5) * H(i, 5);
    }
    norm_5 = sqrt(norm_5);
    for (int i = 0; i < 9; i++) {
        H(i, 5) = H(i, 5) / norm_5;
    }

    K(5, 0) = r[6] * H(0, 0) + r[7] * H(1, 0) + r[8] * H(2, 0);
    K(5, 1) = 0;
    K(5, 2) = r[0] * H(6, 2) + r[1] * H(7, 2) + r[2] * H(8, 2);
    K(5, 3) = r[6] * H(0, 3) + r[7] * H(1, 3) + r[8] * H(2, 3);
    K(5, 4) = r[6] * H(0, 4) + r[7] * H(1, 4) + r[8] * H(2, 4) + r[0] * H(6, 4) + r[1] * H(7, 4) + r[2] * H(8, 4);
    K(5, 5) = r[6] * H(0, 5) + r[7] * H(1, 5) + r[8] * H(2, 5) + r[0] * H(6, 5) + r[1] * H(7, 5) + r[2] * H(8, 5);

    std::vector<float> pn(81, 0);
#define Pn(r, c) pn[r + 9 * c]
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            Pn(i, j) = (i == j);
        }
    }
    for (int i = 0; i < 6; ++i) {
        for (int j = 0;j < 9; ++j) {
            h_t[i * 9 + j] = h[j * 6 + i];
        }
    }
    std::vector<float> HHT(81); // h:(9,6),h_t(6,9),HHT(9,9)
    Math::Matrix::multi(HHT.data(), h.data(), h_t.data(), 9, 6, 9);
    for (int i = 0; i < 9; ++i) {
        for (int j = 0; j < 9; ++j) {
            pn[j + 9 * i] = pn[j + 9 * i] - HHT[j + 9 * i];
        }
    }

    float norm_threshold = 0.1;
    int index1 = -1,
        index2 = -1,
        index3 = -1;
    float max_norm1 = std::numeric_limits<float>::min(),
           min_dot12 = std::numeric_limits<float>::max(),
           min_dot1323 = std::numeric_limits<float>::max();
    float col_norms[9] = { 0 };
    
    for (int c = 0; c < 9; c++) {
        for (int r = 0; r < 9; r++) {
            col_norms[c] += pn[9 * c + r] * pn[9 * c + r];
        }
        col_norms[c] = sqrt(col_norms[c]);
    }
    for (int i = 0; i < 9; i++) {
        if (col_norms[i] >= norm_threshold) {
            if (max_norm1 < col_norms[i]) {
                max_norm1 = col_norms[i];
                index1 = i;
            }
        }
    }
    for (int i = 0; i < 9; i++) {
        N(i, 0) = Pn(i, index1) / max_norm1;
    }
    for (int i = 0; i < 9; i++) {
        if (i == index1) continue;
        if (col_norms[i] >= norm_threshold) {
            float cos_v1_x_col = 0.f;
            for (int j = 0; j < 9; j++) {
                cos_v1_x_col += Pn(j, i) * Pn(j, index1);
            }
            cos_v1_x_col = fabs(cos_v1_x_col / col_norms[i]);
            if (cos_v1_x_col <= min_dot12) {
                index2 = i;
                min_dot12 = cos_v1_x_col;
            }
        }
    }
    float v2dotN0 = 0.f;
    for (int i = 0; i < 9; i++) {
        v2dotN0 += Pn(i, index2) * N(i, 0);
    }
    float norm_N1 = 0.f;
    for (int i = 0; i < 9; i++) {
        N(i, 1) = Pn(i, index2) - v2dotN0 * N(i, 0);
        norm_N1 += N(i, 1) * N(i, 1);
    }
    norm_N1 = sqrt(norm_N1);
    for (int i = 0; i < 9; i++) {
        N(i, 1) /= norm_N1;
    }
    for (int i = 0; i < 9; i++) {
        if (i == index2 || i == index1) continue;
        if (col_norms[i] >= norm_threshold) {
            float cos_v1_x_col = 0.f, cos_v2_x_col = 0.f;
            for (int j = 0; j < 9; j++) {
                cos_v1_x_col += Pn(j, i) * Pn(j, index1);
                cos_v2_x_col += Pn(j, i) * Pn(j, index2);
            }
            cos_v1_x_col = fabs(cos_v1_x_col / col_norms[i]);
            cos_v2_x_col = fabs(cos_v2_x_col / col_norms[i]);
            if (cos_v1_x_col + cos_v2_x_col <= min_dot1323) {
                index3 = i;
                min_dot1323 = cos_v2_x_col + cos_v2_x_col;
            }
        }
    }
    float v3dotN1 = 0.f, v3dotN0 = 0.f;
    for (int i = 0; i < 9; i++) {
        v3dotN0 += Pn(i, index3) * N(i, 0);
        v3dotN1 += Pn(i, index3) * N(i, 1);
    }
    float norm_N2 = 0.f;
    for (int i = 0; i < 9; i++) {
        N(i, 2) = Pn(i, index3) - v3dotN1 * N(i, 1) - v3dotN0 * N(i, 0);
        norm_N2 += N(i, 2) * N(i, 2);
    }
    norm_N2 = sqrt(norm_N2);
    for (int i = 0; i < 9; i++) {
        N(i, 2) /= norm_N2;
    }
    // RowAndNullSpace end
    float g[6];
    g[0] = 1 - sqnorm_r1;
    g[1] = 1 - sqnorm_r2;
    g[2] = 1 - sqnorm_r3;
    g[3] = -dot_r1r2;
    g[4] = -dot_r2r3;
    g[5] = -dot_r1r3;

    float x[6];
    x[0] = g[0] / K(0, 0);
    x[1] = g[1] / K(1, 1);
    x[2] = g[2] / K(2, 2);
    x[3] = (g[3] - K(3, 0) * x[0] - K(3, 1) * x[1]) / K(3, 3);
    x[4] = (g[4] - K(4, 1) * x[1] - K(4, 2) * x[2] - K(4, 3) * x[3]) / K(4, 4);
    x[5] = (g[5] - K(5, 0) * x[0] - K(5, 2) * x[2] - K(5, 3) * x[3] - K(5, 4) * x[4]) / K(5, 5);

    std::vector<float> NtOmega_(27); // (3,9)
    float W_[9]        = {0}; // (3,3)
    float WInverse_[9] = {0}; // (3,3)
    std::vector<float> WInverseOmega(27);
    float delta_r_[9] = {0}; // (9,1)
    float y_[3] = {0};       // (3,1)
    float ny_[9] = {0};      // (9,1)
    Matrix winv;

    Math::Matrix::multi(delta_, h.data(), x, 9, 6, 1);
    // n:(9,3), n_t:(3,9)
    std::vector<float> n_t(27);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0;j < 9; ++j) {
            n_t[i * 9 + j] = n[j * 3 + i];
        }
    }
    Math::Matrix::multi(NtOmega_.data(), n_t.data(), omega, 3, 9, 9); // n_t * omega
    Math::Matrix::multi(W_, NtOmega_.data(), n.data(), 3, 9, 3);

    winv.set9(W_);
    winv.invert(&winv);
    winv.get9(WInverse_);
    
    for (int i = 0; i < 9; ++i) {
        WInverse_[i] = -1.0f * WInverse_[i];
    }
    Math::Matrix::multi(WInverseOmega.data(), WInverse_, NtOmega_.data(), 3, 3, 9);
    Math::Matrix::add(delta_r_, delta_, r, 9);
    Math::Matrix::multi(y_, WInverseOmega.data(), delta_r_, 3, 9, 1);
    Math::Matrix::multi(ny_, n.data(), y_, 9, 3, 1);
    Math::Matrix::add(delta_, delta_, ny_, 9);
}
void runSQP(float* solution_r_hat_, float* r_, float* omega_) {
    float delta_squared_norm = std::numeric_limits<float>::max();
    int step = 0;
    while (delta_squared_norm > 1e-10 && step++ < 15) {
        float delta[9] = {0};
        solveSQPSystem(delta, r_, omega_);
        for (int i = 0; i < 9; i++) {
            auto d = delta[i];
            delta_squared_norm += d * d;
            r_[i] += d;
        }
    }
    float solution_r_[9] = {0}; // (9,1)
//    std::unique_ptr<Tensor> solution_r_(Math::Matrix::create(1, 9));
    ::memcpy(solution_r_, r_, 36);
    float det_r = det9x1(r_);
    if (det_r < 0) {
        for (int i = 0; i < 9; ++i) {
            r_[i] = (-1.f) * r_[i];
        }
        det_r = -det_r;
    }
    if (det_r > 1.001) {
        nearestRotationMatrix(solution_r_hat_, solution_r_);
    } else {
        ::memcpy(solution_r_hat_, solution_r_, 36);
    }
}

void handleSolution(float* solution_r_hat_, float* solution_t, float* omega_,
                    float mean_x, float mean_y, float mean_z, const float* optr, int n,
                    float* rvec, float* tvec, float& min_sq_error) {
    auto r = solution_r_hat_;
    auto t = solution_t;
    bool cheirok = (r[6] * mean_x + r[7] * mean_y + r[8] * mean_z + t[2]) > 0;
    if (!cheirok) {
        int npos = 0, nneg = 0;
        for (size_t i = 0; i < n; i++) {
            if (r[6] * optr[0] + r[7] * optr[1] + r[8] * optr[2] + t[2] > 0) {
                ++npos;
            } else {
                ++nneg;
            }
        }
        cheirok = (npos >= nneg);
    }
    if (cheirok) {
        float sq_error = 0.f;
        float omega_r_[9] = {0}; // (9,1)
        Math::Matrix::multi(omega_r_, omega_, solution_r_hat_, 9, 9, 1);

        for (int i = 0; i < 9; i++) {
            sq_error += omega_r_[i] * solution_r_hat_[i];
        }
        if (min_sq_error - sq_error > 1e-6) {
            min_sq_error = sq_error;
            memcpy(rvec, r, 36);
            memcpy(tvec, t, 12);
        }
    }
}
// helper functions
std::pair<VARP, VARP> solvePnP(VARP objectPoints, VARP imagePoints, VARP cameraMatrix, VARP distCoeffs, bool useExtrinsicGuess) {
    imagePoints = undistortPoints(imagePoints, cameraMatrix, distCoeffs);
    int n = objectPoints->getInfo()->dim[0];
    auto optr = objectPoints->readMap<float>();
    auto iptr = imagePoints->readMap<float>();
    // computeOmega start
    std::vector<float> omega(81);  // (9,9)
    std::vector<float> qa_sum(27); // (3,9)
#define omega(i,j) omega[i * 9 + j]
#define qa_sum(i,j) qa_sum[i * 9 + j]
    float sq_norm_sum = 0, sum_img_x = 0, sum_img_y = 0,
          sum_obj_x = 0, sum_obj_y = 0, sum_obj_z = 0;
    for (int i = 0; i < n; i++) {
        auto X = optr[i * 3], Y = optr[i * 3 + 1], Z = optr[i * 3 + 2];
        auto x = iptr[i * 2], y = iptr[i * 2 + 1];
        float sq_norm = x * x + y * y;
        sq_norm_sum += sq_norm;
        sum_img_x += x;
        sum_img_y += y;
        sum_obj_x += X;
        sum_obj_y += Y;
        sum_obj_z += Z;
        float X2 = X * X;
        float XY = X * Y;
        float XZ = X * Z;
        float Y2 = Y * Y;
        float YZ = Y * Z;
        float Z2 = Z * Z;
        omega(0,0) += X2;
        omega(0,1) += XY;
        omega(0,2) += XZ;
        omega(1,1) += Y2;
        omega(1,2) += YZ;
        omega(2,2) += Z2;
        omega(0,6) += -x * X2; omega(0,7) += -x * XY; omega(0,8) += -x * XZ;
        omega(1,7) += -x * Y2; omega(1,8) += -x * YZ;
        omega(2,8) += -x * Z2;
        omega(3,6) += -y * X2; omega(3,7) += -y * XY; omega(3,8) += -y * XZ;
        omega(4,7) += -y * Y2; omega(4,8) += -y * YZ;
        omega(5,8) += -y * Z2;
        omega(6,6) += sq_norm * X2; omega(6,7) += sq_norm * XY; omega(6,8) += sq_norm * XZ;
        omega(7,7) += sq_norm * Y2; omega(7,8) += sq_norm * YZ;
        omega(8,8) += sq_norm * Z2;
        qa_sum(0,0) += X; qa_sum(0,1) += Y; qa_sum(0,2) += Z;
        qa_sum(1,3) += X; qa_sum(1,4) += Y; qa_sum(1,5) += Z;
        qa_sum(0,6) += -x * X; qa_sum(0,7) += -x * Y; qa_sum(0,8) += -x * Z;
        qa_sum(1,6) += -y * X; qa_sum(1,7) += -y * Y; qa_sum(1,8) += -y * Z;
        qa_sum(2,0) += -x * X; qa_sum(2,1) += -x * Y; qa_sum(2,2) += -x * Z;
        qa_sum(2,3) += -y * X; qa_sum(2,4) += -y * Y; qa_sum(2,5) += -y * Z;
        qa_sum(2,6) += sq_norm * X; qa_sum(2,7) += sq_norm * Y; qa_sum(2,8) += sq_norm * Z;
    }
    omega(1,6) = omega(0,7); omega(2,6) = omega(0,8); omega(2,7) = omega(1,8);
    omega(4,6) = omega(3,7); omega(5,6) = omega(3,8); omega(5,7) = omega(4,8);
    omega(7,6) = omega(6,7); omega(8,6) = omega(6,8); omega(8,7) = omega(7,8);
    omega(3,3) = omega(0,0); omega(3,4) = omega(0,1); omega(3,5) = omega(0,2);
    omega(4,4) = omega(1,1); omega(4,5) = omega(1,2);
    omega(5,5) = omega(2,2);
    for (int r = 0; r < 9; r++) {
        for (int c = 0; c < r; c++) {
            omega(r,c) = omega(c,r);
        }
    }
    float qinv[9]; // (3,3)
    std::vector<float> p(27);   // (3,9)
    CV::Matrix q;
    q.setAll(n, 0, -sum_img_x, 0, n, -sum_img_y, -sum_img_x, -sum_img_y, sq_norm_sum);
    q.invert(&q);
    q.get9(qinv);

    std::vector<float> qa_sum_t(27);   // (9,3)
    std::vector<float> omega_add_(81); // (9,9)
    
    for (int i = 0; i < 9; ++i) {
        qinv[i] = qinv[i] * (-1.f);
    }
    Math::Matrix::multi(p.data(), qinv, qa_sum.data(), 3, 3, 9);
    for (int i = 0; i < 9; ++i) {
        for (int j = 0; j < 3; ++j) {
            qa_sum_t[i * 3 + j] = qa_sum[j * 9 + i];
        }
    }
    Math::Matrix::multi(omega_add_.data(), qa_sum_t.data(), p.data(), 9, 3, 9);
    Math::Matrix::add(omega.data(), omega.data(), omega_add_.data(), 81);
    std::vector<float> s_(81), u(81), vt_(81);
    svdMatrix(s_.data(), u.data(), vt_.data(), omega.data(), 9, 9);

    int num_null_vectors_ = -1;
    while (s_[7 - num_null_vectors_] < 1e-7) num_null_vectors_++;
    float mean_x = sum_obj_x / n, mean_y = sum_obj_y / n, mean_z = sum_obj_z / n;
    // computeOmega end
    // solveInternal start
    int num_eigen_points = num_null_vectors_ > 0 ? num_null_vectors_ : 1;
    float min_sq_error = std::numeric_limits<float>::max();
    float e[9]; // (9,1)
    float solution_r_hat_[9] = {0}; // (9,1)
    float solution_t_[3] = {0};      // (3,1)
    float rvec[9] = {0}, tvec[3] = {0};
    for (int i = 9 - num_eigen_points; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            e[j] = vt_[i * 9 + j] * sqrt(3);
        }
        float orthogonality_sq_error = orthogonalityError(e);
        if (orthogonality_sq_error < 1e-8) {
            float det9x1e = det9x1(e);
            Math::Matrix::multi(solution_r_hat_, e, &det9x1e, 9, 1, 1);
            Math::Matrix::multi(solution_t_, p.data(), solution_r_hat_, 3, 9, 1);
            handleSolution(solution_r_hat_, solution_t_, omega.data(), mean_x, mean_y, mean_z, optr, n, rvec, tvec, min_sq_error);
        } else {
            float r0[9] = {0};
            nearestRotationMatrix(r0, e);
            runSQP(solution_r_hat_, r0, omega.data());
            Math::Matrix::multi(solution_t_, p.data(), solution_r_hat_, 3, 9, 1);
            handleSolution(solution_r_hat_, solution_t_, omega.data(), mean_x, mean_y, mean_z, optr, n, rvec, tvec, min_sq_error);
            for (int ix = 0; ix < 9; ++ix) {
                e[ix] = (-1.0f) * e[ix];
            }
            float r1_[9] = {0};
            nearestRotationMatrix(r1_, e);
            runSQP(solution_r_hat_, r1_, omega.data());
            Math::Matrix::multi(solution_t_, p.data(), solution_r_hat_, 3, 9, 1);
            handleSolution(solution_r_hat_, solution_t_, omega.data(), mean_x, mean_y, mean_z, optr, n, rvec, tvec, min_sq_error);
        }
    }
    int index, c = 1;
    while ((index = 9 - num_eigen_points - c) > 0 && min_sq_error > 3 * s_[index]) {
        for (int j = 0; j < 9; j++) {
            e[j] = vt_[index * 9 + j];
        }
        float r0_[9] = {0};
        nearestRotationMatrix(r0_, e);
        runSQP(solution_r_hat_, r0_, omega.data());
        Math::Matrix::multi(solution_t_, p.data(), solution_r_hat_, 3, 9, 1);
        handleSolution(solution_r_hat_, solution_t_, omega.data(), mean_x, mean_y, mean_z, optr, n, rvec, tvec, min_sq_error);
        for (int ix = 0; ix < 9; ++ix) {
            e[ix] = (-1.0f) * e[ix];
        }
        float r1_[9] = {0};
        nearestRotationMatrix(r1_, e);
        runSQP(solution_r_hat_, r1_, omega.data());
        Math::Matrix::multi(solution_t_, p.data(), solution_r_hat_, 3, 9, 1);
        handleSolution(solution_r_hat_, solution_t_, omega.data(), mean_x, mean_y, mean_z, optr, n, rvec, tvec, min_sq_error);
        c++;
    }
    // solveInternal end
    float res[3];
    Rodrigues(res, rvec);
    VARP tvecvarp = _Input({3, 1}, NCHW);
    VARP rvec_ = _Const(res, {3, 1}, NCHW);
    memcpy(tvecvarp->writeMap<float>(), tvec, 12);
    return std::make_pair(rvec_, tvecvarp);
}

VARP Rodrigues(VARP src) {
    auto res = _Svd(src);
    auto w_ = res[0];
    auto u_ = res[1];
    auto vt_ = res[2];
    auto R_ = _MatMul(u_, vt_);
    R_.fix(Express::VARP::CONSTANT);
    auto R = R_->readMap<float>();
    float x = R[7] - R[5], y = R[2] - R[6], z = R[3] - R[1];
    float s = sqrt((x * x + y * y + z * z) * 0.25);
    float c = (R[0] + R[4] + R[8] - 1) * 0.5;
    c = c > 1. ? 1. : c < -1. ? -1. : c;
    float theta = acos(c);
    if (s < 1e-5) {
        if (c > 0) {
            x = y = z = 0;
        } else {
            x = sqrt(fmax((R[0] + 1) * 0.5, 0));
            y = sqrt(fmax((R[4] + 1) * 0.5, 0)) * (R[1] < 0 ? -1. : 1.);
            z = sqrt(fmax((R[8] + 1) * 0.5, 0)) * (R[2] < 0 ? -1. : 1.);
            if (fabs(x) < fabs(y) && fabs(x) < fabs(z) && (R[5] > 0) != (y * z > 0)) {
                z = -z;
            }
            theta /= sqrt(x * x + y * y + z * z);
            x *= theta;
            y *= theta;
            z *= theta;
        }
    } else {
        float vth = 1 / (2 * s);
        vth *= theta;
        x *= vth;
        y *= vth;
        z *= vth;
    }
    float data[3] = { x, y, z };
    return _Const(data, {3, 1}, NCHW);
}
} // CV
} // MNN
