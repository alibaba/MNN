//
//  CommonCompute.hpp
//  MNN
//
//  Created by MNN on 2021/07/23.
//  Copyright © 2018 - 2021, Alibaba Group Holding Limited
//

#ifndef CommonCompute_hpp
#define CommonCompute_hpp
#include <random>

namespace MNN {
class MNN_PUBLIC CommonCompute {
public:
    // sparse common functions
    template <typename ElementType>
    static void statisticWeightSparsity(size_t& weightNNZElement, size_t& weightBlockNumber, const ElementType* data, size_t h, size_t l,  int sparseBlockOC) {

        size_t nnzBlock = 0;
        size_t nnzTail = 0;
        int i = 0;
        for (; i + sparseBlockOC <= h; i += sparseBlockOC) {
            for(int j = 0; j < l; j += 1) {
                nnzBlock += !checkAllZeros(data, l, sparseBlockOC, 1);
                data++;
            }
            data += l * (sparseBlockOC - 1);
        }
        for (; i < h; i++) {
            for(int j = 0; j < l; j++) {
                nnzTail += (*data != 0);
                data++;
            }
        }
        weightNNZElement = nnzBlock * sparseBlockOC + nnzTail;
        weightBlockNumber = nnzBlock + nnzTail;
        return;
    }

    template <typename ElementType>
    static void fillRandValueAsSparsity(size_t& weightNNZElement, size_t& weightBlockNumber, ElementType* data, int oc, int reduceDimLength, float sparsity, int sparseBlockOC, ElementType minValue = 0, ElementType maxValue = 1) {
        unsigned int seed = 1000;
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> uniform_dist(0, 1);
        std::uniform_real_distribution<ElementType> uniform_value(minValue, maxValue);
        float* data_ptr = data;

        size_t nnzBlock = 0;
        size_t nnzTail = 0;
        int ocEven = (oc / sparseBlockOC) * sparseBlockOC;


        size_t ioc = 0;
        for (; ioc < ocEven; ioc += sparseBlockOC) {
        for (size_t i = 0; i < reduceDimLength; i++) {
            bool isZero = uniform_dist(rng) <= sparsity;
            for (int iblock = 0; iblock < sparseBlockOC; iblock++) {
                *(data + iblock * reduceDimLength) = isZero ? 0.f : uniform_value(rng);
            }
            data++;
            nnzBlock += !isZero;
            }
            data += (sparseBlockOC - 1) * reduceDimLength;
        }
        for (; ioc < oc; ioc++) {
            for (size_t i = 0; i < reduceDimLength; i++) {
                bool isZero = uniform_dist(rng) <= sparsity;
                *data++ = isZero ? 0.f : uniform_value(rng);
                nnzTail += !isZero;
            }
        }
        weightNNZElement = nnzBlock * sparseBlockOC + nnzTail;
        weightBlockNumber = nnzBlock + nnzTail;
    }
    template <typename ElementType>
    bool static checkAllZeros(const ElementType * source, size_t rowDimLength, int blockRow, int blockCol) {
        for (int i = 0; i < blockRow; i++) {
            for (int j = 0; j < blockCol; j++) {
                if (*(source + i * rowDimLength + j) != 0) {
                    return false;
                }
            }
        }
        return true;
    }

};
} // namespace MNN

#endif
