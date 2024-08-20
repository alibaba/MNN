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
    static bool compressFloatWeightToSparse(MNN::OpT* op) {
        auto opType = op->type;
        auto param = op->main.AsConvolution2D();
        if (param->sparseParameter.get() == nullptr) {
            return false;
        }
        // Encode for sparse float weight
        size_t weightSize = param->weight.size();

        if (weightSize > std::numeric_limits<uint32_t>().max()) {
            MNN_ERROR("The weightSize exceed uint32_t, can't compress the sparse weight\n");
            return false;
        }
        param->quanParameter.reset(new IDSTQuanT);
        size_t validSize = 0;
        std::vector<uint32_t> indexes;
        std::vector<float> newWeights;

        for (size_t i=0; i<weightSize; ++i) {
            if (param->weight[i] != 0.0f) {
                indexes.emplace_back(i);
                newWeights.emplace_back(param->weight[i]);
            }
        }
        // If empty, Add Single weight to avoid error, runtime can't extract full sparse convolution
        if (indexes.empty()) {
            indexes.emplace_back(0);
            newWeights.emplace_back(0.0f);
        }
        param->weight.clear();
        param->quanParameter->alpha = std::move(newWeights);
        param->quanParameter->weightSize = (uint32_t)weightSize;
        param->quanParameter->index = std::move(indexes);
        return true;
    }
};
} // namespace MNN

#endif
