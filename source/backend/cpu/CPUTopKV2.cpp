//
//  CPUTopKV2.cpp
//  MNN
//
//  Created by MNN on 2018/08/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUTopKV2.hpp"
#include "CPUBackend.hpp"
#include "Macro.h"

namespace MNN {

template <typename T>
class TopContainer {
public:
    TopContainer() = delete;
    TopContainer(int32_t k, int32_t rowSize) : mK(k) {
        mContainer.reserve(std::min(k, rowSize) + 1);
    }

    void startCollecting(const T* values) {
        mValues = values;
        mContainer.clear();
    }
    void push(int32_t a) {
        auto comparator = [this](int32_t a, int32_t b) { return compareFunc(a, b); };
        if (mContainer.size() <= mK) {
            mContainer.push_back(a);
            if (mContainer.size() == mK + 1) {
                std::make_heap(mContainer.begin(), mContainer.end(), comparator);
                std::pop_heap(mContainer.begin(), mContainer.end(), comparator);
            }
        } else if (comparator(a, mContainer.front())) {
            mContainer.back() = a;
            std::push_heap(mContainer.begin(), mContainer.end(), comparator);
            std::pop_heap(mContainer.begin(), mContainer.end(), comparator);
        }
    }

    const std::vector<int32_t>& sortedResult() {
        auto comparator = [this](int32_t a, int32_t b) { return compareFunc(a, b); };
        if (mContainer.size() <= mK) {
            std::sort(mContainer.begin(), mContainer.end(), comparator);
        } else {
            std::sort_heap(mContainer.begin(), mContainer.end() - 1, comparator);
            mContainer.resize(mK);
        }
        return mContainer;
    }

private:
    int32_t mK;
    std::vector<int32_t> mContainer;
    const T* mValues = nullptr;

    bool compareFunc(int32_t a, int32_t b) const {
        if (mValues[b] < mValues[a]) {
            return true;
        } else if (mValues[b] > mValues[a]) {
            return false;
        } else {
            return a < b;
        }
    }
};

template <typename T>
void findTopK(int32_t rowSize, int32_t numRows, const T* data, int32_t k, int32_t* outputIndexes, T* outputValues) {
    TopContainer<T> topc(k, rowSize);
    for (int row = 0; row < numRows; row++) {
        const T* valuesRow = data + row * rowSize;
        topc.startCollecting(valuesRow);
        for (int c = 0; c < rowSize; c++) {
            topc.push(c);
        }

        int32_t* indexesRow = outputIndexes + row * k;
        T* ouputRow         = outputValues + row * k;

        const auto& topK = topc.sortedResult();
        std::copy(topK.begin(), topK.end(), indexesRow);
        std::transform(topK.begin(), topK.end(), ouputRow, [valuesRow](const int32_t loc) { return valuesRow[loc]; });
    }
}

CPUTopKV2::CPUTopKV2(Backend* b, const TopKV2* TopKV2Param) : MNN::Execution(b), mTopKV2Param(TopKV2Param) {
    // nothing to do
}

ErrorCode CPUTopKV2::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    const int k        = inputs[1]->host<int32_t>()[0];
    auto inputTensor   = inputs[0];
    auto outputData    = outputs[0];
    auto outputIndices = outputs[1];

    const int inputDimension = inputTensor->buffer().dimensions;

    const int rowSize = inputTensor->buffer().dim[inputDimension - 1].extent;
    MNN_ASSERT(k <= rowSize);
    const int numRows = inputTensor->elementSize() / rowSize;
    if (halide_type_float == inputTensor->getType().code) {
        auto inputData   = inputTensor->host<float>();
        auto topkData    = outputData->host<float>();
        int* indicesData = outputIndices->host<int32_t>();
        findTopK<float>(rowSize, numRows, inputData, k, indicesData, topkData);
    } else {
        MNN_PRINT("TODO\n");
        MNN_ASSERT(false);
    }
    return NO_ERROR;
}

class CPUTopKV2Creator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        return new CPUTopKV2(backend, op->main_as_TopKV2());
    }
};

REGISTER_CPU_OP_CREATOR(CPUTopKV2Creator, OpType_TopKV2);

} // namespace MNN
