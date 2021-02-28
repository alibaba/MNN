//
//  WinogradHelper.hpp
//  MNN
//
//  Created by MNN on 2018/07/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef WinogradHelper_hpp
#define WinogradHelper_hpp

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace MNN {
namespace WinogradHelper {

// Winograd Fraction Matrix (for G and for A)
enum WinogradFractionEnum {
    FractionsInG,
    FractionsInA
};

namespace L2K3 {
inline int blockUnit() { return 4;}
inline int dstUnit() { return 2;}
inline int srcUnit() { return 4;}
template <typename T, int VecSize>
void sourceTransformUnit1D(const T* srcStart, T* dstStart, size_t srcStep, size_t dstStep, size_t blockSize);

template <typename T, int VecSize>
void sourceTransformUnit2D(const T* srcStart, T* dstStart, size_t srcStep, size_t dstStep, size_t blockSize);

template <typename T, int VecSize>
void weightTransform1D(const T* srcStart, T* dstStart, size_t srcStep, size_t dstStep, size_t blockSize);

template <typename T, int VecSize>
void weightTransform2D(const T* srcStart, T* dstStart, size_t srcStep, size_t dstStep, size_t blockSize);

template <WinogradFractionEnum WFE>
void destTransform1D(const float* srcStart, float* dstBlock, size_t srcStep, size_t dstStep, size_t blockSize);

template <WinogradFractionEnum WFE>
void destTransform2D(const float* srcStart, float* dstBlock, size_t srcStep, size_t dstStep, size_t blockSize);

}

}
}

#endif // WinogradHelper_hpp
