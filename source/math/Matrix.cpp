//
//  Matrix.cpp
//  MNN
//
//  Created by MNN on 2018/08/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "math/Matrix.hpp"
#include "core/MNNMemoryUtils.h"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "backend/cpu/compute/ConvOpt.h"
#include <cmath>

#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif

namespace MNN {
namespace Math {
Tensor* Matrix::createShape(int w, int h, void* data) {
    auto shape                    = new Tensor(2);
    shape->buffer().dim[0].extent = h;
    shape->buffer().dim[1].extent = w;
    TensorUtils::setLinearLayout(shape);
    shape->buffer().host = (uint8_t*)data;
    return shape;
}

Tensor* Matrix::create(int w, int h) {
    Tensor shape(2);
    shape.buffer().dim[0].extent = h;
    shape.buffer().dim[1].extent = w;
    auto result                  = new Tensor(&shape);
    TensorUtils::setLinearLayout(result);
    return result;
}

void Matrix::multi(Tensor* C, const Tensor* A, const Tensor* B) {
    MNN_ASSERT(NULL != C);
    MNN_ASSERT(NULL != B);
    MNN_ASSERT(NULL != A);

    MNN_ASSERT(2 == C->dimensions());
    MNN_ASSERT(2 == B->dimensions());
    MNN_ASSERT(2 == A->dimensions());

    const auto a = A->host<float>();
    const auto b = B->host<float>();
    auto c       = C->host<float>();

    const int h = A->length(0);
    const int k = A->length(1);
    const int w = B->length(1);

    const int aw = A->stride(0);
    const int bw = B->stride(0);
    const int cw = C->stride(0);

    MNN_ASSERT(k == B->length(0));

    int y = 0;
    for (; y < h; ++y) {
        int x            = 0;
        const auto aLine = a + y * aw;
        auto cLine       = c + y * cw;
#ifdef MNN_USE_NEON
        // firstly, compute 16 together
        for (; x <= w - 16; x += 16) {
            auto bColumn     = b + x;
            float32x4_t sum0 = vdupq_n_f32(0.0);
            float32x4_t sum1 = vdupq_n_f32(0.0);
            float32x4_t sum2 = vdupq_n_f32(0.0);
            float32x4_t sum3 = vdupq_n_f32(0.0);
            for (int i = 0; i < k; ++i) {
                const auto bLine = bColumn + i * bw;
                float32x4_t a0   = vdupq_n_f32(aLine[i]);
                float32x4_t b0   = vld1q_f32(bLine);
                float32x4_t b1   = vld1q_f32(bLine + 4);
                float32x4_t b2   = vld1q_f32(bLine + 8);
                float32x4_t b3   = vld1q_f32(bLine + 12);
                sum0             = vmlaq_f32(sum0, a0, b0);
                sum1             = vmlaq_f32(sum1, a0, b1);
                sum2             = vmlaq_f32(sum2, a0, b2);
                sum3             = vmlaq_f32(sum3, a0, b3);
            }
            vst1q_f32(cLine + x, sum0);
            vst1q_f32(cLine + x + 4, sum1);
            vst1q_f32(cLine + x + 8, sum2);
            vst1q_f32(cLine + x + 12, sum3);
        }
        // secondly, compute 4 together
        for (; x <= w - 4; x += 4) {
            auto bColumn    = b + x;
            float32x4_t sum = vdupq_n_f32(0.0);
            for (int i = 0; i < k; ++i) {
                const auto bLine = bColumn + i * bw;
                float32x4_t a4   = vdupq_n_f32(aLine[i]);
                float32x4_t b4   = vld1q_f32(bLine);
                sum              = vmlaq_f32(sum, a4, b4);
            }
            vst1q_f32(cLine + x, sum);
        }
#endif
        for (; x < w; ++x) {
            auto bColumn = b + x;
            float sum    = 0.0f;
            for (int i = 0; i < k; ++i) {
                sum += aLine[i] * bColumn[i * bw];
            }
            cLine[x] = sum;
        }
    }
}

void Matrix::multi (float* C, float* A, float* B, int M, int K, int N, bool A_needTranspose, bool B_needTranspose) {
    if (N == 0) {
        // step1: dst->shape()=(M,M), src->shape()=(M,K), dst=src*src_T, dst is a symmetric matrix.
        // step2: (E-dst)*2
        int y = 0;
        for (; y < M; ++y) { // C:row
            int x = 0;
            const auto aLineRow = B + y * K;
            for (; x < y; ++x) { // C:column
                // half bottom coordinate (y,x), half top (x,y)
                int indexBottom = y * M + x;
                int indexTop    = x * M + y;
                const auto aLineColumn = B + x * K;
                float sum    = 0.0f;
                for (int i = 0; i < K; ++i) {
                    sum += aLineRow[i] * aLineColumn[i];
                }
                C[indexBottom] = sum * sum;
                C[indexTop]    = sum * sum;
                A[indexBottom]  = -sum;
                A[indexTop]     = -sum;
            }
            // diagonal
            int index = y * M + x;
            const auto aLineColumn = B + x * K;
            float sum = 0.f;
            for (int i = 0; i < K; ++i) {
                sum += aLineRow[i] * aLineColumn[i];
            }
            C[index] = (1 - sum) * (1 - sum);
            A[index]  = 1 - sum;
        } // Finish compute src*src_T
        return;
    }
    int y = 0;
    for (; y < M; ++y) {
        int x            = 0;
        const auto aLine = A + y * K;
        auto cLine       = C + y * N;
        for (; x < N; ++x) {
            auto bColumn = B + x;
            float sum    = 0.0f;
            for (int i = 0; i < K; ++i) {
                sum += aLine[i] * bColumn[i * N];
            }
            cLine[x] = sum;
        }
    }
}

void Matrix::add (float* C, float* A, float* B, int size) {
    MNNMatrixAddCommon(C, A, B, size, 0, 0, 0, 1);
}

void Matrix::add(Tensor* C, const Tensor* A, const Tensor* B) {
    MNN_ASSERT(NULL != C);
    MNN_ASSERT(NULL != B);
    MNN_ASSERT(NULL != A);

    MNN_ASSERT(A->size() == C->size());
    auto height = A->length(0);
    auto width = A->length(1);
    int bOffset = 0;
    if (B->dimensions() == A->dimensions()) {
        bOffset = B->stride(0);
        MNN_ASSERT(B->length(1) == A->length(1));
        MNN_ASSERT(B->length(0) == A->length(0));
    } else {
        bOffset = 0;
        MNN_ASSERT(B->length(0) == A->length(1));
    }
    MNNMatrixAddCommon(C->host<float>(), A->host<float>(), B->host<float>(), width, C->stride(0), A->stride(0), bOffset, height);
    return;
}

void Matrix::sub(Tensor* C, const Tensor* A, const Tensor* B) {
    MNN_ASSERT(NULL != C);
    MNN_ASSERT(NULL != B);
    MNN_ASSERT(NULL != A);

    MNN_ASSERT(A->size() == C->size());
    auto height = A->length(0);
    auto width = A->length(1);
    int bOffset = 0;
    if (B->dimensions() == A->dimensions()) {
        bOffset = B->stride(0);
        MNN_ASSERT(B->length(1) == A->length(1));
        MNN_ASSERT(B->length(0) == A->length(0));
    } else {
        bOffset = 0;
        MNN_ASSERT(B->length(0) == A->length(1));
    }
    MNNMatrixSubCommon(C->host<float>(), A->host<float>(), B->host<float>(), width, C->stride(0), A->stride(0), bOffset, height);
}

void Matrix::dot(Tensor* C, const Tensor* A, const Tensor* B) {
    MNN_ASSERT(NULL != C);
    MNN_ASSERT(NULL != B);
    MNN_ASSERT(NULL != A);
    MNN_ASSERT(2 == C->dimensions());
    MNN_ASSERT(2 == B->dimensions());
    MNN_ASSERT(2 == A->dimensions());
    MNN_ASSERT(A->shape() == B->shape());
    MNN_ASSERT(A->shape() == C->shape());
    const int height = A->length(0);
    const int width = A->length(1);

    const int aw = A->stride(0);
    const int bw = B->stride(0);
    const int cw = C->stride(0);
    MNNMatrixProdCommon(C->host<float>(), A->host<float>(), B->host<float>(), width, cw, aw, bw, height);
}

void Matrix::invert(Tensor* dst, const Tensor* src) {
    MNN_ASSERT(2 == src->buffer().dimensions);
    const int N0 = src->buffer().dim[0].extent;
    MNN_ASSERT(N0 == src->buffer().dim[1].extent);

    int i, j, k;
    float max, temp;
    std::shared_ptr<Tensor> tempMat(Matrix::create(N0, N0));
    ::memcpy(tempMat->buffer().host, src->buffer().host, src->size());
    const auto tempData = tempMat->host<float>();
    const auto dstData  = dst->host<float>();
    for (i = 0; i < N0; ++i) {
        for (j = 0; j < N0; ++j) {
            *(dstData + i * N0 + j) = (i == j) ? 1.0f : 0.0f;
        }
    }

    for (i = 0; i < N0; ++i) {
        max = *(tempData + i * N0 + i);
        k   = i;
        for (j = i + 1; j < N0; ++j) {
            auto data1 = *(tempData + j * N0 + i);
            if (fabs(data1) > fabs(max)) {
                max = data1;
                k   = j;
            }
        }
        if (k != i) {
            for (j = 0; j < N0; ++j) {
                temp                     = *(tempData + i * N0 + j);
                *(tempData + i * N0 + j) = *(tempData + k * N0 + j);
                *(tempData + k * N0 + j) = temp;
                temp                     = *(dstData + i * N0 + j);
                *(dstData + i * N0 + j)  = *(dstData + k * N0 + j);
                *(dstData + k * N0 + j)  = temp;
            }
        }
        if (*(tempData + i * N0 + i) == 0) {
            MNN_PRINT("This matrix have no inverse!\n");
            return;
        }
        temp = *(tempData + i * N0 + i);

        for (j = 0; j < N0; ++j) {
            *(tempData + i * N0 + j) = *(tempData + i * N0 + j) / temp;
            *(dstData + i * N0 + j)  = *(dstData + i * N0 + j) / temp;
        }

        for (j = 0; j < N0; ++j) {
            if (j != i) {
                temp = *(tempData + j * N0 + i);
                for (k = 0; k < N0; ++k) {
                    *(tempData + j * N0 + k) = *(tempData + j * N0 + k) - *(tempData + i * N0 + k) * temp;
                    *(dstData + j * N0 + k)  = *(dstData + j * N0 + k) - *(dstData + i * N0 + k) * temp;
                }
            }
        }
    }
}

void Matrix::transpose(Tensor* dst, const Tensor* src) {
    auto a = src->host<float>();
    auto b = dst->host<float>();
    int as = src->buffer().dim[0].stride;
    int bs = dst->buffer().dim[0].stride;

    int w = dst->buffer().dim[1].extent;
    int h = dst->buffer().dim[0].extent;

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            b[bs * y + x] = a[as * x + y];
        }
    }
}
void Matrix::print(const Tensor* C, const char* head) {
    auto c      = C->host<float>();
    auto w      = C->buffer().dim[1].extent;
    for (int i=2; i<C->dimensions(); ++i) {
        w *= C->length(i);
    }
    auto h      = C->buffer().dim[0].extent;
    auto stride = C->buffer().dim[0].stride;

    MNN_PRINT("%s\n", head);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            MNN_PRINT("%.7f\t", c[x + y * stride]);
        }
        MNN_PRINT("\n");
    }
}

void Matrix::mul(Tensor* dst, const Tensor* src, const float scale) {
    MNN_ASSERT(NULL != dst);
    MNN_ASSERT(NULL != src);
    MNN_ASSERT(2 == dst->dimensions());
    MNN_ASSERT(2 == src->dimensions());
    MNN_ASSERT(src->shape() == dst->shape());
    const int height = src->length(0);
    const int width = src->length(1);

    const int sw = src->stride(0);
    const int dw = dst->stride(0);
#ifdef MNN_USE_NEON
    float32x4_t scale_ = vdupq_n_f32(scale);
#endif
    for(int y = 0; y < height; y++) {
        auto s = src->host<float>() + y * sw;
        auto d = dst->host<float>() + y * dw;
        int i = 0;
#ifdef MNN_USE_NEON
        for (; i <= width - 8; i += 8) {
            float32x4_t s0 = vld1q_f32(s + i);
            float32x4_t s1 = vld1q_f32(s + i + 4);
            float32x4_t d0 = vmulq_f32(s0, scale_);
            float32x4_t d1 = vmulq_f32(s1, scale_);
            vst1q_f32(d + i, d0);
            vst1q_f32(d + i + 4, d1);
        }
        for (; i <= width - 4; i += 4) {
            float32x4_t ss  = vld1q_f32(s + i);
            float32x4_t dd = vmulq_f32(ss, scale_);
            vst1q_f32(d + i, dd);
        }
#endif
        for (; i < width; ++i) {
            d[i] = s[i] * scale;
        }
    }
}

void Matrix::mulPerLine(Tensor* C, const Tensor* A, const Tensor* Line) {
    auto c         = C->host<float>();
    auto a         = A->host<float>();
    auto l         = Line->host<float>();
    auto w         = C->buffer().dim[1].extent;
    auto h         = C->buffer().dim[0].extent;
    auto stride    = C->buffer().dim[0].stride;
    auto srcStride = A->buffer().dim[0].stride;
    MNN_ASSERT(Line->buffer().dim[1].extent >= h);
    MNN_ASSERT(A->buffer().dim[0].extent == h);
    MNN_ASSERT(A->buffer().dim[1].extent == w);
    MNN_ASSERT(Line->buffer().dim[0].extent == 1);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            c[x + y * stride] = a[x + y * srcStride] * l[y];
        }
    }
}

void Matrix::divPerLine(Tensor* C, const Tensor* A, const Tensor* Line) {
    auto c         = C->host<float>();
    auto a         = A->host<float>();
    auto l         = Line->host<float>();
    auto w         = C->buffer().dim[1].extent;
    auto h         = C->buffer().dim[0].extent;
    auto stride    = C->buffer().dim[0].stride;
    auto srcStride = A->buffer().dim[0].stride;
    MNN_ASSERT(Line->buffer().dim[1].extent >= h);
    MNN_ASSERT(A->buffer().dim[0].extent == h);
    MNN_ASSERT(A->buffer().dim[1].extent == w);
    MNN_ASSERT(Line->buffer().dim[0].extent == 1);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            c[x + y * stride] = a[x + y * srcStride] / l[y];
        }
    }
}

std::shared_ptr<Tensor> Matrix::polyMulti(std::shared_ptr<Tensor> A, std::shared_ptr<Tensor> B) {
    MNN_ASSERT(A->buffer().dim[0].extent == 1);
    MNN_ASSERT(B->buffer().dim[0].extent == 1);
    auto aw = A->buffer().dim[1].extent;
    auto bw = B->buffer().dim[1].extent;

    std::shared_ptr<Tensor> result(Matrix::create(aw + bw - 1, 1));

    auto a = A->host<float>();
    auto b = B->host<float>();

    auto c = result->host<float>();
    for (int i = 0; i < aw + bw - 1; ++i) {
        c[i] = 0.0f;
    }
    for (int y = 0; y < bw; ++y) {
        auto bValue = b[y];
        for (int x = 0; x < aw; ++x) {
            auto aValue = a[x];
            c[x + y] += bValue * aValue;
        }
    }
    return result;
}

float Matrix::matDet(const Tensor* A) {
    MNN_ASSERT(2 == A->buffer().dimensions);
    const int n0 = A->buffer().dim[0].extent;
    MNN_ASSERT(n0 == A->buffer().dim[1].extent);
    auto dataPtr = A->host<float>();
    int r, c, m;
    int lop      = 0;
    float result = 0;
    float mid    = 1;
    if (n0 != 1) {
        if (2 == n0) {
            lop = 1;
        } else {
            lop = n0;
        }

        for (m = 0; m < lop; ++m) {
            mid = 1;
            for (r = 0, c = m; r < n0; ++r, ++c) {
                mid = mid * (*(dataPtr + r * n0 + c % n0));
            }
            result += mid;
        }

        for (m = 0; m < lop; ++m) {
            mid = 1;
            for (r = 0, c = n0 - 1 - m + n0; r < n0; ++r, --c) {
                mid = mid * (*(dataPtr + r * n0 + c % n0));
            }
            result -= mid;
        }
    }

    return result;
}
} // namespace Math
} // namespace MNN
