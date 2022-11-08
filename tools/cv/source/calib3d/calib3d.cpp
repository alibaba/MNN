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
std::unique_ptr<Tensor> nearestRotationMatrix(Tensor* e_) {
    VARP e = Express::Variable::create(Express::Expr::create(e_, false));
    e = _Transpose(_Reshape(e, {3, 3}), {1, 0});
    auto res = _Svd(e);
    auto u = res[1];
    auto vt = res[2];
    std::unique_ptr<Tensor> u_(Math::Matrix::create(3, 3));
    std::unique_ptr<Tensor> vt_(Math::Matrix::create(3, 3));
    std::unique_ptr<Tensor> v_(Math::Matrix::create(3, 3));
    for (int i = 0; i < 9; i++) {
        u_->host<float>()[i] = u->readMap<float>()[i];
        vt_->host<float>()[i] = vt->readMap<float>()[i];
    }
    Math::Matrix::transpose(v_.get(), vt_.get());
    float detuv[9] = {1, 0, 0, 0, 1, 0, 0, 0, det3x3(u_->host<float>()) * det3x3(v_->host<float>())};
    std::unique_ptr<Tensor> detuv_(Math::Matrix::createShape(3, 3, detuv));
    std::unique_ptr<Tensor> udetuv_(Math::Matrix::create(3, 3));
    std::unique_ptr<Tensor> R_(Math::Matrix::create(3, 3));
    std::unique_ptr<Tensor> r_(Math::Matrix::create(3, 3));
    Math::Matrix::multi(udetuv_.get(), u_.get(), detuv_.get());
    Math::Matrix::multi(R_.get(), udetuv_.get(), vt_.get());
    Math::Matrix::transpose(r_.get(), R_.get());
    r_->buffer().dim[0].extent = 9;
    r_->buffer().dim[0].stride = 1;
    r_->buffer().dim[1].extent = 1;
    r_->buffer().dim[1].stride = 1;
    return r_;
}
std::unique_ptr<Tensor> solveSQPSystem(const Tensor* r_, const Tensor* omega_) {
    auto r = r_->host<float>();
    float sqnorm_r1 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2],
           sqnorm_r2 = r[3] * r[3] + r[4] * r[4] + r[5] * r[5],
           sqnorm_r3 = r[6] * r[6] + r[7] * r[7] + r[8] * r[8];
    float dot_r1r2 = r[0] * r[3] + r[1] * r[4] + r[2] * r[5],
           dot_r1r3 = r[0] * r[6] + r[1] * r[7] + r[2] * r[8],
           dot_r2r3 = r[3] * r[6] + r[4] * r[7] + r[5] * r[8];
    float h[54] = { 0 }, k[36] = { 0 }, n[27] = { 0 };
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

    std::unique_ptr<Tensor> h_(Math::Matrix::createShape(6, 9, h));
    std::unique_ptr<Tensor> k_(Math::Matrix::createShape(6, 6, k));
    std::unique_ptr<Tensor> n_(Math::Matrix::createShape(3, 9, n));
    std::unique_ptr<Tensor> pn_(Math::Matrix::create(9, 9));
    auto pn  = pn_->host<float>();
#define Pn(r, c) pn[r + 9 * c]
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            Pn(i, j) = (i == j);
        }
    }
    std::unique_ptr<Tensor> h_t_(Math::Matrix::create(9, 6));
    std::unique_ptr<Tensor> h_h_t_(Math::Matrix::create(9, 9));
    Math::Matrix::transpose(h_t_.get(), h_.get());
    Math::Matrix::multi(h_h_t_.get(), h_.get(), h_t_.get());
    Math::Matrix::sub(pn_.get(), pn_.get(), h_h_t_.get());
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
            col_norms[c] += Pn(r, c) * Pn(r, c);
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
    std::unique_ptr<Tensor> x_(Math::Matrix::createShape(1, 6, x));
    std::unique_ptr<Tensor> delta_(Math::Matrix::create(1, 9));
    std::unique_ptr<Tensor> n_t_(Math::Matrix::create(9, 3));
    std::unique_ptr<Tensor> NtOmega_(Math::Matrix::create(9, 3));
    std::unique_ptr<Tensor> W_(Math::Matrix::create(3, 3));
    std::unique_ptr<Tensor> Winv_(Math::Matrix::create(3, 3));
    std::unique_ptr<Tensor> WinvNtOmega_(Math::Matrix::create(9, 3));
    std::unique_ptr<Tensor> delta_r_(Math::Matrix::create(1, 9));
    std::unique_ptr<Tensor> y_(Math::Matrix::create(1, 3));
    std::unique_ptr<Tensor> ny_(Math::Matrix::create(1, 9));
    Math::Matrix::multi(delta_.get(), h_.get(), x_.get());
    Math::Matrix::transpose(n_t_.get(), n_.get());
    Math::Matrix::multi(NtOmega_.get(), n_t_.get(), omega_);
    Math::Matrix::multi(W_.get(), NtOmega_.get(), n_.get());
    Matrix winv;
    winv.set9(W_->host<float>());
    winv.invert(&winv);
    winv.get9(Winv_->host<float>());
    Math::Matrix::mul(Winv_.get(), Winv_.get(), -1.f);
    Math::Matrix::multi(WinvNtOmega_.get(), Winv_.get(), NtOmega_.get());
    Math::Matrix::add(delta_r_.get(), delta_.get(), r_);
    Math::Matrix::multi(y_.get(), WinvNtOmega_.get(), delta_r_.get());
    Math::Matrix::multi(ny_.get(), n_.get(), y_.get());
    Math::Matrix::add(delta_.get(), delta_.get(), ny_.get());
    return delta_;
}
std::unique_ptr<Tensor> runSQP(Tensor* r_, Tensor* omega_) {
    float delta_squared_norm = std::numeric_limits<float>::max();
    int step = 0;
    while (delta_squared_norm > 1e-10 && step++ < 15) {
        auto delta = solveSQPSystem(r_, omega_);
        for (int i = 0; i < 9; i++) {
            auto d = delta->host<float>()[i];
            delta_squared_norm += d * d;
            r_->host<float>()[i] += d;
        }
    }
    std::unique_ptr<Tensor> solution_r_(Math::Matrix::create(1, 9));
    ::memcpy(solution_r_->host<float>(), r_->host<float>(), 36);
    std::unique_ptr<Tensor> solution_r_hat_;
    float det_r = det9x1(r_->host<float>());
    if (det_r < 0) {
        Math::Matrix::mul(r_, r_, -1.f);
        det_r = -det_r;
    }
    if (det_r > 1.001) {
        solution_r_hat_ = nearestRotationMatrix(solution_r_.get());
    } else {
        solution_r_hat_ = std::move(solution_r_);
    }
    return solution_r_hat_;
}
void handleSolution(Tensor* solution_r_hat_, Tensor* solution_t, Tensor* omega_,
                    float mean_x, float mean_y, float mean_z, const float* optr, int n,
                    VARP& rvec, VARP& tvec, float& min_sq_error) {
    auto r = solution_r_hat_->host<float>();
    auto t = solution_t->host<float>();
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
        std::unique_ptr<Tensor> omega_r_(Math::Matrix::create(1, 9));
        Math::Matrix::multi(omega_r_.get(), omega_, solution_r_hat_);
        for (int i = 0; i < 9; i++) {
            sq_error += omega_r_->host<float>()[i] * solution_r_hat_->host<float>()[i];
        }
        if (min_sq_error - sq_error > 1e-6) {
            min_sq_error = sq_error;
            memcpy(rvec->writeMap<float>(), r, 36);
            memcpy(tvec->writeMap<float>(), t, 12);
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
    float omega[9][9] = { 0 };
    float qa_sum[3][9] = { 0 };
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
        omega[0][0] += X2;
        omega[0][1] += XY;
        omega[0][2] += XZ;
        omega[1][1] += Y2;
        omega[1][2] += YZ;
        omega[2][2] += Z2;
        omega[0][6] += -x * X2; omega[0][7] += -x * XY; omega[0][8] += -x * XZ;
        omega[1][7] += -x * Y2; omega[1][8] += -x * YZ;
        omega[2][8] += -x * Z2;
        omega[3][6] += -y * X2; omega[3][7] += -y * XY; omega[3][8] += -y * XZ;
        omega[4][7] += -y * Y2; omega[4][8] += -y * YZ;
        omega[5][8] += -y * Z2;
        omega[6][6] += sq_norm * X2; omega[6][7] += sq_norm * XY; omega[6][8] += sq_norm * XZ;
        omega[7][7] += sq_norm * Y2; omega[7][8] += sq_norm * YZ;
        omega[8][8] += sq_norm * Z2;
        qa_sum[0][0] += X; qa_sum[0][1] += Y; qa_sum[0][2] += Z;
        qa_sum[1][3] += X; qa_sum[1][4] += Y; qa_sum[1][5] += Z;
        qa_sum[0][6] += -x * X; qa_sum[0][7] += -x * Y; qa_sum[0][8] += -x * Z;
        qa_sum[1][6] += -y * X; qa_sum[1][7] += -y * Y; qa_sum[1][8] += -y * Z;
        qa_sum[2][0] += -x * X; qa_sum[2][1] += -x * Y; qa_sum[2][2] += -x * Z;
        qa_sum[2][3] += -y * X; qa_sum[2][4] += -y * Y; qa_sum[2][5] += -y * Z;
        qa_sum[2][6] += sq_norm * X; qa_sum[2][7] += sq_norm * Y; qa_sum[2][8] += sq_norm * Z;
    }
    omega[1][6] = omega[0][7]; omega[2][6] = omega[0][8]; omega[2][7] = omega[1][8];
    omega[4][6] = omega[3][7]; omega[5][6] = omega[3][8]; omega[5][7] = omega[4][8];
    omega[7][6] = omega[6][7]; omega[8][6] = omega[6][8]; omega[8][7] = omega[7][8];
    omega[3][3] = omega[0][0]; omega[3][4] = omega[0][1]; omega[3][5] = omega[0][2];
    omega[4][4] = omega[1][1]; omega[4][5] = omega[1][2];
    omega[5][5] = omega[2][2];
    for (int r = 0; r < 9; r++) {
        for (int c = 0; c < r; c++) {
            omega[r][c] = omega[c][r];
        }
    }
    float qinv[9], p[3][9];
    CV::Matrix q;
    q.setAll(n, 0, -sum_img_x, 0, n, -sum_img_y, -sum_img_x, -sum_img_y, sq_norm_sum);
    q.invert(&q);
    q.get9(qinv);
    std::unique_ptr<Tensor> q_inv_(Math::Matrix::createShape(3, 3, static_cast<void*>(qinv)));
    std::unique_ptr<Tensor> qa_sum_(Math::Matrix::createShape(9, 3, static_cast<void*>(qa_sum)));
    std::unique_ptr<Tensor> omega_(Math::Matrix::createShape(9, 9, static_cast<void*>(omega)));
    std::unique_ptr<Tensor> p_(Math::Matrix::createShape(9, 3, static_cast<void*>(p)));
    std::unique_ptr<Tensor> qa_sum_t_(Math::Matrix::create(3, 9));
    std::unique_ptr<Tensor> omega_add_(Math::Matrix::create(9, 9));
    Math::Matrix::mul(q_inv_.get(), q_inv_.get(), -1.f);
    Math::Matrix::multi(p_.get(), q_inv_.get(), qa_sum_.get());
    Math::Matrix::transpose(qa_sum_t_.get(), qa_sum_.get());
    Math::Matrix::multi(omega_add_.get(), qa_sum_t_.get(), p_.get());
    Math::Matrix::add(omega_.get(), omega_.get(), omega_add_.get());
    auto res = _Svd(Express::Variable::create(Express::Expr::create(omega_.get(), false)));
    auto s_ = res[0];
    auto vt_ = res[2];
    auto u_ = _Transpose(vt_, {1, 0});
    int num_null_vectors_ = -1;
    while (s_->readMap<float>()[7 - num_null_vectors_] < 1e-7) num_null_vectors_++;
    float mean_x = sum_obj_x / n, mean_y = sum_obj_y / n, mean_z = sum_obj_z / n;
    // computeOmega end
    // solveInternal start
    int num_eigen_points = num_null_vectors_ > 0 ? num_null_vectors_ : 1;
    float min_sq_error = std::numeric_limits<float>::max();
    float e[9];
    std::unique_ptr<Tensor> solution_r_hat_(Math::Matrix::create(1, 9));
    std::unique_ptr<Tensor> solution_t_(Math::Matrix::create(1, 3));
    std::unique_ptr<Tensor> e_(Math::Matrix::createShape(1, 9, e));
    VARP rvec = _Input({3, 3}, NCHW), tvec = _Input({3, 1}, NCHW);
    for (int i = 9 - num_eigen_points; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            e[j] = vt_->readMap<float>()[i * 9 + j] * sqrt(3);
        }
        float orthogonality_sq_error = orthogonalityError(e);
        if (orthogonality_sq_error < 1e-8) {
            Math::Matrix::mul(solution_r_hat_.get(), e_.get(), det9x1(e));
            Math::Matrix::multi(solution_t_.get(), p_.get(), solution_r_hat_.get());
            handleSolution(solution_r_hat_.get(), solution_t_.get(), omega_.get(), mean_x, mean_y, mean_z, optr, n, rvec, tvec, min_sq_error);
        } else {
            auto r0_ = nearestRotationMatrix(e_.get());
            solution_r_hat_ = runSQP(r0_.get(), omega_.get());
            Math::Matrix::multi(solution_t_.get(), p_.get(), solution_r_hat_.get());
            handleSolution(solution_r_hat_.get(), solution_t_.get(), omega_.get(), mean_x, mean_y, mean_z, optr, n, rvec, tvec, min_sq_error);
            Math::Matrix::mul(e_.get(), e_.get(), -1.f);
            auto r1_ = nearestRotationMatrix(e_.get());
            solution_r_hat_ = runSQP(r1_.get(), omega_.get());
            Math::Matrix::multi(solution_t_.get(), p_.get(), solution_r_hat_.get());
            handleSolution(solution_r_hat_.get(), solution_t_.get(), omega_.get(), mean_x, mean_y, mean_z, optr, n, rvec, tvec, min_sq_error);
        }
    }
    int index, c = 1;
    while ((index = 9 - num_eigen_points - c) > 0 && min_sq_error > 3 * s_->readMap<float>()[index]) {
        for (int j = 0; j < 9; j++) {
            e[j] = vt_->readMap<float>()[index * 9 + j];
        }
        auto r0_ = nearestRotationMatrix(e_.get());
        solution_r_hat_ = runSQP(r0_.get(), omega_.get());
        Math::Matrix::multi(solution_t_.get(), p_.get(), solution_r_hat_.get());
        handleSolution(solution_r_hat_.get(), solution_t_.get(), omega_.get(), mean_x, mean_y, mean_z, optr, n, rvec, tvec, min_sq_error);
        Math::Matrix::mul(e_.get(), e_.get(), -1.f);
        auto r1_ = nearestRotationMatrix(e_.get());
        solution_r_hat_ = runSQP(r1_.get(), omega_.get());
        Math::Matrix::multi(solution_t_.get(), p_.get(), solution_r_hat_.get());
        handleSolution(solution_r_hat_.get(), solution_t_.get(), omega_.get(), mean_x, mean_y, mean_z, optr, n, rvec, tvec, min_sq_error);
        c++;
    }
    // solveInternal end
    rvec = Rodrigues(rvec);
    return std::make_pair(rvec, tvec);
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
