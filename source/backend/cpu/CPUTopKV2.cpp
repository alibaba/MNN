//
//  CPUTopKV2.cpp
//  MNN
//
//  Created by MNN on 2018/08/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUTopKV2.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "core/Macro.h"
#include "core/Concurrency.h"
#include "backend/cpu/compute/CommonOptFunction.h"
#include <algorithm>
namespace MNN {

template <typename T>
void findTopK(int32_t rowSize, int32_t numRows, const T* data, int32_t k, int32_t* outputIndexes, T* outputValues, bool largest) {
    struct DataType {
        T value;
        int index;
    };
    std::vector<DataType> cacheData(rowSize);
    auto compareL = [](const DataType& A, const DataType& B) {
        return A.value > B.value;
    };
    auto compareM = [](const DataType& A, const DataType& B) {
        return A.value < B.value;
    };
    for (int row = 0; row < numRows; row++) {
        const T* valuesRow = data + row * rowSize;
        int32_t* indexesRow = outputIndexes + row * k;
        T* outputRow         = outputValues + row * k;
        for (int i=0; i<rowSize; ++i) {
            cacheData[i].value = valuesRow[i];
            cacheData[i].index = i;
        }
        if (largest) {
            std::partial_sort(cacheData.begin(), cacheData.begin() + k, cacheData.end(), compareL);
        } else {
            std::partial_sort(cacheData.begin(), cacheData.begin() + k, cacheData.end(), compareM);
        }
        for (int i=0; i<k; ++i) {
            outputRow[i] = cacheData[i].value;
            indexesRow[i] = cacheData[i].index;
        }
    }
}

CPUTopKV2::CPUTopKV2(Backend* b, const Op* op) : MNN::Execution(b) {
    auto param = op->main_as_TopKV2();
    if (param != nullptr) {
        mLargest = param->largest();
    }
}

ErrorCode CPUTopKV2::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    const int k        = inputs[1]->host<int32_t>()[0];
    auto inputTensor   = inputs[0];
    auto outputData    = outputs[0];
    auto outputIndices = outputs[1];

    const int inputDimension = inputTensor->buffer().dimensions;

    const int rowSize = inputTensor->buffer().dim[inputDimension - 1].extent;
    const int rowC4Blocks = rowSize / 4;
    const int rowRemain = rowSize % 4;
    const int rowC4ElementSize = rowC4Blocks * 4;
    MNN_ASSERT(k <= rowSize);
    const int numRows = inputTensor->elementSize() / rowSize;

    if (k == 1 && mLargest) {
        if (halide_type_float == inputTensor->getType().code) {
            float* inputData   = inputTensor->host<float>();
            float* topkData    = outputData->host<float>();
            int32_t* indicesData = outputIndices->host<int32_t>();

            MNN_CONCURRENCY_BEGIN(i, numRows) {
                float* inputRowData = inputData + i * rowSize;
                float* rowTopkData = topkData + i * k;
                int32_t* rowTopkIndexData = indicesData + i * k;
                MNNVectorTop1Float(inputRowData, rowTopkData, rowTopkIndexData, rowC4Blocks);
                for (int j = 0; j < rowRemain; j++) {
                    int index = rowC4ElementSize + j;
                    float value = inputRowData[index];
                    if (value > rowTopkData[0]) {
                        rowTopkData[0] = value;
                        rowTopkIndexData[0] = index;
                    }
                }
            }
            MNN_CONCURRENCY_END();
        } else if (halide_type_int == inputTensor->getType().code && 32 == inputTensor->getType().bits) {
            int32_t* inputData   = inputTensor->host<int32_t>();
            int32_t* topkData    = outputData->host<int32_t>();
            int32_t* indicesData = outputIndices->host<int32_t>();
            MNN_CONCURRENCY_BEGIN(i, numRows) {
                int32_t* inputRowData = inputData + i * rowSize;
                int32_t* rowTopkData = topkData + i * k;
                int32_t* rowTopkIndexData = indicesData + i * k;
                MNNVectorTop1Int32(inputRowData, rowTopkData, rowTopkIndexData, rowC4Blocks);
                for (int j = 0; j < rowRemain; j++) {
                    int index = rowC4ElementSize + j;
                    int32_t value = inputRowData[index];
                    if (value > rowTopkData[0]) {
                        rowTopkData[0] = value;
                        rowTopkIndexData[0] = index;
                    }
                }
            }
            MNN_CONCURRENCY_END();
        } else {
            MNN_PRINT("TopKV2 data type not supported\n");
            MNN_ASSERT(false);
        }

        return NO_ERROR;
    }

    if (halide_type_float == inputTensor->getType().code) {
        auto inputData   = inputTensor->host<float>();
        auto topkData    = outputData->host<float>();
        int* indicesData = outputIndices->host<int32_t>();
        findTopK<float>(rowSize, numRows, inputData, k, indicesData, topkData, mLargest);
    } else if(halide_type_int == inputTensor->getType().code && 32 == inputTensor->getType().bits) {
        auto inputData   = inputTensor->host<int32_t>();
        auto topkData    = outputData->host<int32_t>();
        int* indicesData = outputIndices->host<int32_t>();
        findTopK<int32_t>(rowSize, numRows, inputData, k, indicesData, topkData, mLargest);
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
        return new CPUTopKV2(backend, op);
    }
};

REGISTER_CPU_OP_CREATOR(CPUTopKV2Creator, OpType_TopKV2);

} // namespace MNN
