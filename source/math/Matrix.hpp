//
//  Matrix.hpp
//  MNN
//
//  Created by MNN on 2018/08/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef Matrix_hpp
#define Matrix_hpp

#include <stdio.h>
#include <memory>
#include <MNN/Tensor.hpp>
namespace MNN {
namespace Math {
class MNN_PUBLIC Matrix {
public:
    static Tensor* createShape(int w, int h, void* data = nullptr);
    static Tensor* create(int w, int h);

    static void multi(Tensor* C, const Tensor* A, const Tensor* B);
    static void multi(float* C, float* A, float* B, int M, int K, int N = 0, bool A_needTranspose=false, bool B_needTranspose=false);
    static void add(Tensor* C, const Tensor* A, const Tensor* B);
    static void add(float* C, float* A, float* B, int size);
    static void sub(Tensor* C, const Tensor* A, const Tensor* B);
    static void dot(Tensor* C, const Tensor* A, const Tensor* B);
    static void divPerLine(Tensor* C, const Tensor* A, const Tensor* Line);
    static void invert(Tensor* dst, const Tensor* src);
    static void transpose(Tensor* dst, const Tensor* src);
    static void print(const Tensor* C, const char* head = "Matrix:");
    static void mul(Tensor* dst, const Tensor* src, const float scale);

    static void mulPerLine(Tensor* C, const Tensor* A, const Tensor* Line);

    static std::shared_ptr<Tensor> polyMulti(std::shared_ptr<Tensor> A, std::shared_ptr<Tensor> B);

    // the determinant of the matrix
    static float matDet(const Tensor* A);
};
} // namespace Math
} // namespace MNN

#endif /* Matrix_hpp */
