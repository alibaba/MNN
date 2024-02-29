//
//  core.cpp
//  MNN
//
//  Created by MNN on 2023/04/18.
//  Copyright © 2018][Alibaba Group Holding Limited
//

#include <math/Matrix.hpp>
#include "cv/core.hpp"
#include "cv/imgproc/geometric.hpp"
#include <MNN/expr/NeuralNetWorkOp.hpp>
#include <MNN/expr/MathOp.hpp>


namespace MNN {
namespace CV {

#ifndef FLT_EPSILON
#define FLT_EPSILON 1.19209290E-07F
#endif
#define det2(m)   ((double)m(0,0)*m(1,1) - (double)m(0,1)*m(1,0))
#define det3(m)   (m(0,0)*((double)m(1,1)*m(2,2) - (double)m(1,2)*m(2,1)) -  \
                   m(0,1)*((double)m(1,0)*m(2,2) - (double)m(1,2)*m(2,0)) +  \
                   m(0,2)*((double)m(1,0)*m(2,1) - (double)m(1,1)*m(2,0)))

int LUImpl(float* A, int astep, int m, float* b, int bstep, int n, float eps) {
    int i, j, k, p = 1;

    for (i = 0; i < m; i++) {
        k = i;
        for (j = i+1; j < m; j++) {
            if (fabs(A[j*astep + i]) > fabs(A[k*astep + i])) {
                k = j;
            }
        }
        if (fabs(A[k*astep + i]) < eps) {
            return 0;
        }
        if (k != i) {
            for (j = i; j < m; j++) {
                std::swap(A[i*astep + j], A[k*astep + j]);
            }
            if (b) {
                for (j = 0; j < n; j++) {
                    std::swap(b[i*bstep + j], b[k*bstep + j]);
                }
            }
            p = -p;
        }

        float d = -1/A[i*astep + i];

        for (j = i+1; j < m; j++) {
            float alpha = A[j*astep + i]*d;
            for (k = i+1; k < m; k++) {
                A[j*astep + k] += alpha*A[i*astep + k];
            }
            if (b) {
                for (k = 0; k < n; k++) {
                    b[j*bstep + k] += alpha*b[i*bstep + k];
                }
            }
        }
    }

    if (b) {
        for (i = m-1; i >= 0; i--) {
            for (j = 0; j < n; j++) {
                float s = b[i*bstep + j];
                for (k = i+1; k < m; k++) {
                    s -= A[i*astep + k]*b[k*bstep + j];
                }
                b[i*bstep + j] = s/A[i*astep + i];
            }
        }
    }

    return p;
}

std::pair<bool, VARP> solve(VARP src1, VARP src2, int method) {
    method = DECOMP_LU;
    int row1, col1, channel1, row2, col2, channel2;
    getVARPSize(src1, &row1, &col1, &channel1);
    getVARPSize(src2, &row2, &col2, &channel2);
    auto dst = _Input({col1, col2});
    bool is_normal = (method == DECOMP_NORMAL);
    bool result = true;
    // check case of a single equation and small matrix
    if ((method == DECOMP_LU || method == DECOMP_CHOLESKY) &&
        row1 <= 3 && row1 == col1 && col2 == 1) {
        auto ptr1 = src1->readMap<float>();
        auto ptr2 = src2->readMap<float>();
        auto dstptr = dst->writeMap<float>();
#define Sf(y, x) ptr1[y * col1 + x]
#define bf(y) ptr2[y * col2]
#define Df(y, x) dstptr[y * col2 + x]
        if (row1 == 2) {
            double d = det2(Sf);
            if (d != 0.) {
                double t;
                d = 1./d;
                t = (float)(((double)bf(0) * Sf(1,1) - (double)bf(1) * Sf(0,1)) * d);
                Df(1,0) = (float)(((double)bf(1) * Sf(0,0) - (double)bf(0) * Sf(1,0)) * d);
                Df(0,0) = (float)t;
            } else {
                result = false;
            }
        } else if (row1 == 3) {
            double d = det3(Sf);
            if (d != 0.) {
                float t[3];
                d = 1./d;
                t[0] = (float)(d*
                       (bf(0)*((double)Sf(1,1)*Sf(2,2) - (double)Sf(1,2)*Sf(2,1)) -
                        Sf(0,1)*((double)bf(1)*Sf(2,2) - (double)Sf(1,2)*bf(2)) +
                        Sf(0,2)*((double)bf(1)*Sf(2,1) - (double)Sf(1,1)*bf(2))));

                t[1] = (float)(d*
                       (Sf(0,0)*(double)(bf(1)*Sf(2,2) - (double)Sf(1,2)*bf(2)) -
                        bf(0)*((double)Sf(1,0)*Sf(2,2) - (double)Sf(1,2)*Sf(2,0)) +
                        Sf(0,2)*((double)Sf(1,0)*bf(2) - (double)bf(1)*Sf(2,0))));

                t[2] = (float)(d*
                       (Sf(0,0)*((double)Sf(1,1)*bf(2) - (double)bf(1)*Sf(2,1)) -
                        Sf(0,1)*((double)Sf(1,0)*bf(2) - (double)bf(1)*Sf(2,0)) +
                        bf(0)*((double)Sf(1,0)*Sf(2,1) - (double)Sf(1,1)*Sf(2,0))));

                Df(0,0) = t[0];
                Df(1,0) = t[1];
                Df(2,0) = t[2];
            } else {
                result = false;
            }
        } else {
            double d = Sf(0,0);
            if (d != 0.) {
                Df(0,0) = (float)(bf(0) / d);
            } else {
                result = false;
            }
        }
        return std::make_pair(result, dst);
    }
    // other matrix
    if (row1 < col1) {
        MNN_ERROR("The function can not solve under-determined linear systems.");
        return std::make_pair(false, dst);
    }
    VARP a;
    if (is_normal) {
    } else if (method != DECOMP_SVD) {
        a = _Clone(src1, true);
    } else {
        a = _Transpose(src1, {1, 0});
    }
    if (!is_normal) {
        if( method == DECOMP_LU || method == DECOMP_CHOLESKY ) {
            dst = _Clone(src2);
        }
    } else {
        if (method == DECOMP_LU || method == DECOMP_CHOLESKY) {
            dst = _MatMul(src1, src2);
        } else {
            src2 = _MatMul(src1, src2);
        }
    }
    a.fix(Express::VARP::CONSTANT);
    dst.fix(Express::VARP::CONSTANT);
    if (method == DECOMP_LU) {
        result = LUImpl(a->writeMap<float>(), row1, col1, dst->writeMap<float>(), col2, col2, FLT_EPSILON * 10);
    }
    return std::make_pair(result, dst);
}

} // CV
} // MNN
