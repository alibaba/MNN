#include "WinogradHelper.hpp"
#include "math/Vec.hpp"

namespace MNN {
namespace WinogradHelper {

namespace L2K3 {


template <typename T, int VecSize>
void sourceTransformUnit1D(const T* srcStart, T* dstStart, size_t srcStep, size_t dstStep, size_t blockSize) {
    using VecType = MNN::Math::Vec<T, VecSize>;
    for (int b = 0; b < blockSize; ++b) {
        auto _x = srcStart + b * VecSize;
        auto _y = dstStart + b * VecSize;
        
        auto x0 = VecType::load(_x + srcStep * 0);
        auto x1 = VecType::load(_x + srcStep * 1);
        auto x2 = VecType::load(_x + srcStep * 2);
        auto x3 = VecType::load(_x + srcStep * 3);
        
        VecType::save(_y + dstStep * 0, x0 - x2);
        VecType::save(_y + dstStep * 1, x1 + x2);
        VecType::save(_y + dstStep * 2, x2 - x1);
        VecType::save(_y + dstStep * 3, x3 - x1);
    }
}

template <typename T, int VecSize>
void sourceTransformUnit2D(const T* srcStart, T* dstStart, size_t srcStep, size_t dstStep, size_t blockSize) {
    using VecType = MNN::Math::Vec<T, VecSize>;
    for (int b = 0; b < blockSize; ++b) {
        auto _x = srcStart + b * VecSize;
        auto _y = dstStart + b * VecSize;
        
        auto m00 = VecType::load(_x + srcStep * 0) - VecType::load(_x + srcStep * 8);
        auto m01 = VecType::load(_x + srcStep * 1) - VecType::load(_x + srcStep * 9);
        auto m02 = VecType::load(_x + srcStep * 2) - VecType::load(_x + srcStep * 10);
        auto m03 = VecType::load(_x + srcStep * 3) - VecType::load(_x + srcStep * 11);
        auto m10 = VecType::load(_x + srcStep * 4) + VecType::load(_x + srcStep * 8);
        auto m11 = VecType::load(_x + srcStep * 5) + VecType::load(_x + srcStep * 9);
        auto m12 = VecType::load(_x + srcStep * 6) + VecType::load(_x + srcStep * 10);
        auto m13 = VecType::load(_x + srcStep * 7) + VecType::load(_x + srcStep * 11);
        auto m20 = VecType::load(_x + srcStep * 8) - VecType::load(_x + srcStep * 4);
        auto m21 = VecType::load(_x + srcStep * 9) - VecType::load(_x + srcStep * 5);
        auto m22 = VecType::load(_x + srcStep * 10) - VecType::load(_x + srcStep * 6);
        auto m23 = VecType::load(_x + srcStep * 11) - VecType::load(_x + srcStep * 7);
        auto m30 = VecType::load(_x + srcStep * 12) - VecType::load(_x + srcStep * 4);
        auto m31 = VecType::load(_x + srcStep * 13) - VecType::load(_x + srcStep * 5);
        auto m32 = VecType::load(_x + srcStep * 14) - VecType::load(_x + srcStep * 6);
        auto m33 = VecType::load(_x + srcStep * 15) - VecType::load(_x + srcStep * 7);

        VecType::save(_y + dstStep * 0, m00 - m02);
        VecType::save(_y + dstStep * 1, m01 + m02);
        VecType::save(_y + dstStep * 2, m02 - m01);
        VecType::save(_y + dstStep * 3, m03 - m01);
        VecType::save(_y + dstStep * 4, m10 - m12);
        VecType::save(_y + dstStep * 5, m11 + m12);
        VecType::save(_y + dstStep * 6, m12 - m11);
        VecType::save(_y + dstStep * 7, m13 - m11);
        VecType::save(_y + dstStep * 8, m20 - m22);
        VecType::save(_y + dstStep * 9, m21 + m22);
        VecType::save(_y + dstStep * 10, m22 - m21);
        VecType::save(_y + dstStep * 11, m23 - m21);
        VecType::save(_y + dstStep * 12, m30 - m32);
        VecType::save(_y + dstStep * 13, m31 + m32);
        VecType::save(_y + dstStep * 14, m32 - m31);
        VecType::save(_y + dstStep * 15, m33 - m31);
    }
}

template <typename T, int VecSize>
void weightTransform1D(const T* srcStart, T* dstStart, size_t srcStep, size_t dstStep, size_t blockSize) {
    using VecType = MNN::Math::Vec<T, VecSize>;
    for (int b = 0; b < blockSize; ++b) {
        auto y = dstStart + b * VecSize;
        auto x = srcStart + b * VecSize;
        
        auto x0 = VecType::load(x + srcStep * 0);
        auto x1 = VecType::load(x + srcStep * 1);
        auto x2 = VecType::load(x + srcStep * 2);
        
        VecType::save(y + dstStep * 0, x0);
        VecType::save(y + dstStep * 1, x0 + x1 + x2);
        VecType::save(y + dstStep * 2, x0 - x1 + x2);
        VecType::save(y + dstStep * 3, x2);
    }
}

template <typename T, int VecSize>
void weightTransform2D(const T* srcStart, T* dstStart, size_t srcStep, size_t dstStep, size_t blockSize) {
    using VecType = MNN::Math::Vec<T, VecSize>;
    for (int b = 0; b < blockSize; ++b) {
        auto y = dstStart + b * VecSize;
        auto x = srcStart + b * VecSize;
        
        auto m00 = VecType::load(x + srcStep * 0);
        auto m01 = VecType::load(x + srcStep * 1);
        auto m02 = VecType::load(x + srcStep * 2);
        auto m10 = VecType::load(x + srcStep * 0) + VecType::load(x + srcStep * 3) + VecType::load(x + srcStep * 6);
        auto m11 = VecType::load(x + srcStep * 1) + VecType::load(x + srcStep * 4) + VecType::load(x + srcStep * 7);
        auto m12 = VecType::load(x + srcStep * 2) + VecType::load(x + srcStep * 5) + VecType::load(x + srcStep * 8);
        auto m20 = VecType::load(x + srcStep * 0) - VecType::load(x + srcStep * 3) + VecType::load(x + srcStep * 6);
        auto m21 = VecType::load(x + srcStep * 1) - VecType::load(x + srcStep * 4) + VecType::load(x + srcStep * 7);
        auto m22 = VecType::load(x + srcStep * 2) - VecType::load(x + srcStep * 5) + VecType::load(x + srcStep * 8);
        auto m30 = VecType::load(x + srcStep * 6);
        auto m31 = VecType::load(x + srcStep * 7);
        auto m32 = VecType::load(x + srcStep * 8);
        
        VecType::save(y + dstStep * 0, m00);
        VecType::save(y + dstStep * 1, m00 + m01 + m02);
        VecType::save(y + dstStep * 2, m00 - m01 + m02);
        VecType::save(y + dstStep * 3, m02);
        VecType::save(y + dstStep * 4, m10);
        VecType::save(y + dstStep * 5, m10 + m11 + m12);
        VecType::save(y + dstStep * 6, m10 - m11 + m12);
        VecType::save(y + dstStep * 7, m12);
        VecType::save(y + dstStep * 8, m20);
        VecType::save(y + dstStep * 9, m20 + m21 + m22);
        VecType::save(y + dstStep * 10, m20 - m21 + m22);
        VecType::save(y + dstStep * 11, m22);
        VecType::save(y + dstStep * 12, m30);
        VecType::save(y + dstStep * 13, m30 + m31 + m32);
        VecType::save(y + dstStep * 14, m30 - m31 + m32);
        VecType::save(y + dstStep * 15, m32);
    }
}

template <>
void weightTransform2D<float, 4>(const float* srcStart, float* dstStart, size_t srcStep, size_t dstStep, size_t blockSize) {
    using VecType = MNN::Math::Vec<float, 4>;
    for (int b = 0; b < blockSize; ++b) {
        auto y = dstStart + b * 4;
        auto x = srcStart + b * 4;
        
        auto m00 = VecType::load(x + srcStep * 0);
        auto m01 = VecType::load(x + srcStep * 1);
        auto m02 = VecType::load(x + srcStep * 2);
        auto m10 = (VecType::load(x + srcStep * 0) + VecType::load(x + srcStep * 3) + VecType::load(x + srcStep * 6)) * 0.5f;
        auto m11 = (VecType::load(x + srcStep * 1) + VecType::load(x + srcStep * 4) + VecType::load(x + srcStep * 7)) * 0.5f;
        auto m12 = (VecType::load(x + srcStep * 2) + VecType::load(x + srcStep * 5) + VecType::load(x + srcStep * 8)) * 0.5f;
        auto m20 = (VecType::load(x + srcStep * 0) - VecType::load(x + srcStep * 3) + VecType::load(x + srcStep * 6)) * 0.5f;
        auto m21 = (VecType::load(x + srcStep * 1) - VecType::load(x + srcStep * 4) + VecType::load(x + srcStep * 7)) * 0.5f;
        auto m22 = (VecType::load(x + srcStep * 2) - VecType::load(x + srcStep * 5) + VecType::load(x + srcStep * 8)) * 0.5f;
        auto m30 = VecType::load(x + srcStep * 6);
        auto m31 = VecType::load(x + srcStep * 7);
        auto m32 = VecType::load(x + srcStep * 8);
        
        VecType::save(y + dstStep * 0, m00);
        VecType::save(y + dstStep * 1, (m00 + m01 + m02) * 0.5f);
        VecType::save(y + dstStep * 2, (m00 - m01 + m02) * 0.5f);
        VecType::save(y + dstStep * 3, m02);
        VecType::save(y + dstStep * 4, m10);
        VecType::save(y + dstStep * 5, (m10 + m11 + m12) * 0.5f);
        VecType::save(y + dstStep * 6, (m10 - m11 + m12) * 0.5f);
        VecType::save(y + dstStep * 7, m12);
        VecType::save(y + dstStep * 8, m20);
        VecType::save(y + dstStep * 9, (m20 + m21 + m22) * 0.5f);
        VecType::save(y + dstStep * 10, (m20 - m21 + m22) * 0.5f);
        VecType::save(y + dstStep * 11, m22);
        VecType::save(y + dstStep * 12, m30);
        VecType::save(y + dstStep * 13, (m30 + m31 + m32) * 0.5f);
        VecType::save(y + dstStep * 14, (m30 - m31 + m32) * 0.5f);
        VecType::save(y + dstStep * 15, m32);
    }
}

template <>
void destTransform2D<FractionsInG>(const float* srcStart, float* dstStart, size_t srcStep, size_t dstStep, size_t blockSize) {
    using VecType = MNN::Math::Vec<float, 4>;
    for (int b = 0; b < blockSize; ++b) {
        auto y = dstStart + b * 4;
        auto x = srcStart + b * 4;
        
        auto m00 = VecType::load(x + srcStep * 0) + VecType::load(x + srcStep * 4) + VecType::load(x + srcStep * 8);
        auto m01 = VecType::load(x + srcStep * 1) + VecType::load(x + srcStep * 5) + VecType::load(x + srcStep * 9);
        auto m02 = VecType::load(x + srcStep * 2) + VecType::load(x + srcStep * 6) + VecType::load(x + srcStep * 10);
        auto m03 = VecType::load(x + srcStep * 3) + VecType::load(x + srcStep * 7) + VecType::load(x + srcStep * 11);
        auto m10 = VecType::load(x + srcStep * 4) - VecType::load(x + srcStep * 8) + VecType::load(x + srcStep * 12);
        auto m11 = VecType::load(x + srcStep * 5) - VecType::load(x + srcStep * 9) + VecType::load(x + srcStep * 13);
        auto m12 = VecType::load(x + srcStep * 6) - VecType::load(x + srcStep * 10) + VecType::load(x + srcStep * 14);
        auto m13 = VecType::load(x + srcStep * 7) - VecType::load(x + srcStep * 11) + VecType::load(x + srcStep * 15);
        
        VecType::save(y + dstStep * 0, m00 + m01 + m02);
        VecType::save(y + dstStep * 1, m01 - m02 + m03);
        VecType::save(y + dstStep * 2, m10 + m11 + m12);
        VecType::save(y + dstStep * 3, m11 - m12 + m13);
    }
}

template <>
void destTransform2D<FractionsInA>(const float* srcStart, float* dstStart, size_t srcStep, size_t dstStep, size_t blockSize) {
    using VecType = MNN::Math::Vec<float, 4>;
    for (int b = 0; b < blockSize; ++b) {
        auto y = dstStart + b * 4;
        auto x = srcStart + b * 4;
        
        auto m00 = VecType::load(x + srcStep * 0) + (VecType::load(x + srcStep * 4) + VecType::load(x + srcStep * 8)) * 0.5f;
        auto m01 = VecType::load(x + srcStep * 1) + (VecType::load(x + srcStep * 5) + VecType::load(x + srcStep * 9)) * 0.5f;
        auto m02 = VecType::load(x + srcStep * 2) + (VecType::load(x + srcStep * 6) + VecType::load(x + srcStep * 10)) * 0.5f;
        auto m03 = VecType::load(x + srcStep * 3) + (VecType::load(x + srcStep * 7) + VecType::load(x + srcStep * 11)) * 0.5f;
        auto m10 = (VecType::load(x + srcStep * 4) - VecType::load(x + srcStep * 8)) * 0.5f + VecType::load(x + srcStep * 12);
        auto m11 = (VecType::load(x + srcStep * 5) - VecType::load(x + srcStep * 9)) * 0.5f + VecType::load(x + srcStep * 13);
        auto m12 = (VecType::load(x + srcStep * 6) - VecType::load(x + srcStep * 10)) * 0.5f + VecType::load(x + srcStep * 14);
        auto m13 = (VecType::load(x + srcStep * 7) - VecType::load(x + srcStep * 11)) * 0.5f + VecType::load(x + srcStep * 15);
        
        VecType::save(y + dstStep * 0, m00 + (m01 + m02) * 0.5f);
        VecType::save(y + dstStep * 1, (m01 - m02) * 0.5f + m03);
        VecType::save(y + dstStep * 2, m10 + (m11 + m12) * 0.5f);
        VecType::save(y + dstStep * 3, (m11 - m12) * 0.5f + m13);
    }
}

template <>
void destTransform1D<FractionsInA>(const float* srcStart, float* dstStart, size_t srcStep, size_t dstStep, size_t blockSize) {
    using VecType = MNN::Math::Vec<float, 4>;
    for (int b = 0; b < blockSize; ++b) {
        auto y = dstStart + b * 4;
        auto x = srcStart + b * 4;
        
        auto x0 = VecType::load(x + srcStep * 0);
        auto x1 = VecType::load(x + srcStep * 1);
        auto x2 = VecType::load(x + srcStep * 2);
        auto x3 = VecType::load(x + srcStep * 3);
        
        VecType::save(y + dstStep * 0, x0 + (x1 + x2) * 0.5f);
        VecType::save(y + dstStep * 1, (x1 - x2) * 0.5f + x3);
    }
}

template void sourceTransformUnit1D<int8_t, 8>(const int8_t*, int8_t*, size_t, size_t, size_t);
template void sourceTransformUnit2D<int8_t, 8>(const int8_t*, int8_t*, size_t, size_t, size_t);
template void sourceTransformUnit2D<float, 4>(const float*, float*, size_t, size_t, size_t);

template void weightTransform1D<int8_t, 16>(const int8_t*, int8_t*, size_t, size_t, size_t);
template void weightTransform2D<int8_t, 16>(const int8_t*, int8_t*, size_t, size_t, size_t);

}

}
}
