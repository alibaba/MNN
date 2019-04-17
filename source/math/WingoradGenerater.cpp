//
//  WingoradGenerater.cpp
//  MNN
//
//  Created by MNN on 2018/08/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "WingoradGenerater.hpp"
#include <math.h>
#include "AutoStorage.h"
#include "Macro.h"

namespace MNN {
namespace Math {

static std::shared_ptr<Tensor> computeF(const float* a, int alpha) {
    std::shared_ptr<Tensor> res;
    res.reset(Matrix::create(alpha, 1));
    auto diagData = res->host<float>();
    for (int x = 0; x < alpha; ++x) {
        float product = 1.0f;
        for (int i = 0; i < alpha; ++i) {
            if (x == i) {
                continue;
            }
            product *= (a[x] - a[i]);
        }
        diagData[x] = product;
    }
    return res;
}

static std::shared_ptr<Tensor> computeT(const float* a, int n) {
    std::shared_ptr<Tensor> result(Matrix::create(n + 1, n));
    for (int y = 0; y < n; ++y) {
        auto line = result->host<float>() + result->stride(0) * y;
        ::memset(line, 0, result->length(0) * sizeof(float));
        line[y] = 1.0f;
        line[n] = -::powf(a[y], (float)n);
    }
    // Matrix::print(result.get());
    return result;
}

static std::shared_ptr<Tensor> computeL(const float* a, int n) {
    MNN_ASSERT(n >= 1);
    std::shared_ptr<Tensor> result(Matrix::create(n, n));
    for (int k = 0; k < n; ++k) {
        std::shared_ptr<Tensor> poly(Matrix::create(1, 1));
        auto p = poly->host<float>();
        p[0]   = 1.0f;
        std::shared_ptr<Tensor> poly2(Matrix::create(2, 1));
        auto p2 = poly2->host<float>();
        for (int i = 0; i < n; ++i) {
            if (i == k) {
                continue;
            }
            p2[0] = -a[i];
            p2[1] = 1.0f;
            poly  = Matrix::polyMulti(poly, poly2);
        }
        ::memcpy(result->host<float>() + result->buffer().dim[0].stride * k, poly->host<float>(), n * sizeof(float));
    }
    return result;
}

static std::shared_ptr<Tensor> computeB(const float* a, int alpha) {
    std::shared_ptr<Tensor> res;
    auto LT    = computeL(a, alpha - 1);
    auto fdiag = computeF(a, alpha - 1);
    Matrix::divPerLine(LT.get(), LT.get(), fdiag.get());

    std::shared_ptr<Tensor> L(Matrix::create(alpha - 1, alpha - 1));
    Matrix::transpose(L.get(), LT.get());

    auto T = computeT(a, alpha - 1);
    std::shared_ptr<Tensor> BT(Matrix::create(alpha, alpha - 1));
    Matrix::multi(BT.get(), L.get(), T.get());

    std::shared_ptr<Tensor> B(Matrix::create(alpha, alpha));
    for (int y = 0; y < alpha - 1; ++y) {
        ::memcpy(B->host<float>() + B->stride(0) * y, BT->host<float>() + BT->stride(0) * y, alpha * sizeof(float));
    }
    auto BLast = B->host<float>() + B->stride(0) * (alpha - 1);
    for (int x = 0; x < alpha - 1; ++x) {
        BLast[x] = 0;
    }
    BLast[alpha - 1] = 1.0f;

    return B;
}

static std::shared_ptr<Tensor> computeA(const float* a, int m, int n) {
    std::shared_ptr<Tensor> res;
    res.reset(Matrix::create(m, n));
    for (int y = 0; y < n; ++y) {
        auto line = res->host<float>() + res->buffer().dim[0].stride * y;
        for (int x = 0; x < m - 1; ++x) {
            if (x == 0 && y == 0) {
                line[x] = 1.0f;
            } else {
                line[x] = ::powf(a[x], (float)y);
            }
        }
        if (y == n - 1) {
            line[m - 1] = 1.0f;
        } else {
            line[m - 1] = 0.0f;
        }
    }
    return res;
}

static std::shared_ptr<Tensor> computeFDiag(const float* a, int alpha) {
    std::shared_ptr<Tensor> res;
    res.reset(Matrix::create(alpha, 1));
    auto diagData = res->host<float>();
    for (int x = 0; x < alpha - 1; ++x) {
        float product = 1.0f;
        for (int i = 0; i < alpha - 1; ++i) {
            if (x == i) {
                continue;
            }
            product *= (a[x] - a[i]);
        }
        diagData[x] = product;
    }
    diagData[alpha - 1] = 1.0f;
    if (diagData[0] < 0) {
        diagData[0] = -diagData[0];
    }
    return res;
}

WinogradGenerater::WinogradGenerater(int computeUnit, int kernelSize, float interp) {
    MNN_ASSERT(computeUnit > 0 && kernelSize > 0);
    mUnit       = computeUnit;
    mKernelSize = kernelSize;

    int n     = computeUnit;
    int r     = kernelSize;
    int alpha = n + r - 1;
    mG.reset(Matrix::create(r, alpha));
    mB.reset(Matrix::create(alpha, alpha));
    mA.reset(Matrix::create(n, alpha));

    std::shared_ptr<Tensor> polyBuffer(Matrix::create(alpha, 1));

    auto a   = polyBuffer->host<float>();
    a[0]     = 0.0f;
    int sign = 1;
    for (int i = 0; i < alpha - 1; ++i) {
        int value = 1 + i / 2;
        a[i + 1]  = sign * value * interp;
        sign *= -1;
    }
    // Matrix::print(polyBuffer.get());
    {
        auto A = computeA(a, alpha, n);
        Matrix::transpose(mA.get(), A.get());
    }
    auto fdiag = computeFDiag(a, alpha);
    // Matrix::print(fdiag.get());
    {
        auto A = computeA(a, alpha, r);
        Matrix::transpose(mG.get(), A.get());
    }
    {
        auto B = computeB(a, alpha);
        Matrix::transpose(mB.get(), B.get());
        Matrix::transpose(B.get(), mB.get());
        mB = B;
    }
}
std::shared_ptr<Tensor> WinogradGenerater::allocTransformWeight(const Tensor* source, int unitCi, int unitCo, bool alloc) {
    int ci = source->channel();
    int co = source->batch();
    MNN_ASSERT(source->width() == source->height() && source->width() == mG->length(1));
    int ciC4 = UP_DIV(ci, unitCi);
    int coC4 = UP_DIV(co, unitCo);
    if (alloc) {
        return std::shared_ptr<Tensor>(Tensor::create<float>({mB->length(0) * mB->length(1), coC4, ciC4, unitCi, unitCo}));
    }
    return std::shared_ptr<Tensor>(Tensor::createDevice<float>({mB->length(0) * mB->length(1), coC4, ciC4, unitCi, unitCo}));
}

void WinogradGenerater::transformWeight(const Tensor* weightDest, const Tensor* source) {
    std::shared_ptr<Tensor> GT(Math::Matrix::create(mG->length(0), mG->length(1)));
    Math::Matrix::transpose(GT.get(), mG.get());
    int ci          = source->length(1);
    int co          = source->length(0);
    int kernelCount = source->length(2);
    int unitCi      = weightDest->length(3);
    int unitCo      = weightDest->length(4);
    auto alpha      = mB->length(0);

    if (ci % unitCi != 0 || co % unitCo != 0) {
        ::memset(weightDest->host<float>(), 0, weightDest->size());
    }
    std::shared_ptr<Tensor> M(Math::Matrix::create(kernelCount, alpha));
    std::shared_ptr<Tensor> K(Math::Matrix::createShape(kernelCount, kernelCount));
    std::shared_ptr<Tensor> K_Transform(Math::Matrix::create(alpha, alpha));
    auto weightPtr      = source->host<float>();
    auto KTransformData = K_Transform->host<float>();
    for (int oz = 0; oz < co; ++oz) {
        auto srcOz = weightPtr + oz * ci * kernelCount * kernelCount;

        int ozC4 = oz / unitCo;
        int mx   = oz % unitCo;

        auto dstOz = weightDest->host<float>() + weightDest->stride(1) * ozC4 + mx;
        for (int sz = 0; sz < ci; ++sz) {
            int szC4         = sz / unitCi;
            int my           = sz % unitCi;
            auto srcSz       = srcOz + kernelCount * kernelCount * sz;
            K->buffer().host = (uint8_t*)srcSz;
            // M = G * K
            Math::Matrix::multi(M.get(), mG.get(), K.get());
            // K_Transform = M*GT
            Math::Matrix::multi(K_Transform.get(), M.get(), GT.get());

            auto dstSz = dstOz + szC4 * weightDest->stride(2) + unitCo * my;

            for (int i = 0; i < alpha * alpha; ++i) {
                *(dstSz + i * weightDest->stride(0)) = KTransformData[i];
            }
        }
    }
}
} // namespace Math
} // namespace MNN
