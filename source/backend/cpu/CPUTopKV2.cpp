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
#include <cmath>

#ifdef MNN_USE_ARMV82
extern "C" {
void MNNRouteSigmoidFP16(const __fp16* src, const __fp16* bias, __fp16* probs, float* scores, int rowSize,
                         int numRows);
void MNNVectorTop1FP16(const __fp16* input, __fp16* maxValue, int32_t* maxIndex, size_t count);
void findTopKFP16(int32_t rowSize, int32_t numRows, const __fp16* data, int32_t k, int32_t* outputIndexes,
                  __fp16* outputValues, bool largest);
}
#endif
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
        mRouteSigmoid = param->routeSigmoid();
    }
}

ErrorCode CPUTopKV2::routeSigmoid(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input = inputs[0];
    const int rowSize = input->length(input->dimensions() - 1);
    const int numRows = input->elementSize() / rowSize;
    const int k = inputs[1]->host<int32_t>()[0];
    if (inputs.size() != 4 || input->getType().code != halide_type_float || k <= 0 || k > rowSize ||
        inputs[3]->elementSize() != rowSize) {
        return NOT_SUPPORT;
    }
    int* indices = outputs[1]->host<int32_t>();
    struct Cand { float score; int index; };
    std::vector<float> scores(input->elementSize());
#if defined(__aarch64__)
    auto core = static_cast<CPUBackend*>(backend())->functions();
#ifdef MNN_USE_ARMV82
    if (core->bytes == 2) {
        MNNRouteSigmoidFP16(reinterpret_cast<const __fp16*>(input->host<void>()),
                            reinterpret_cast<const __fp16*>(inputs[3]->host<void>()),
                            reinterpret_cast<__fp16*>(outputs[0]->host<void>()), scores.data(), rowSize, numRows);
    } else
#endif
    {
        MNNRouteSigmoid(input->host<float>(), inputs[3]->host<float>(), outputs[0]->host<float>(), scores.data(),
                        rowSize, numRows);
    }
#else
    const float* logits = input->host<float>();
    const float* bias = inputs[3]->host<float>();
    float* probs = outputs[0]->host<float>();
    for (int row = 0; row < numRows; ++row) {
        for (int i = 0; i < rowSize; ++i) {
            float p = 1.0f / (1.0f + expf(-logits[row * rowSize + i]));
            probs[row * rowSize + i] = p;
            scores[row * rowSize + i] = p + bias[i];
        }
    }
#endif
    MNN_CONCURRENCY_BEGIN(row, numRows) {
        std::vector<Cand> cand(rowSize);
        for (int i = 0; i < rowSize; ++i) {
            cand[i] = {scores[row * rowSize + i], i};
        }
        auto better = [](const Cand& a, const Cand& b) {
            return a.score > b.score || (a.score == b.score && a.index < b.index);
        };
        std::partial_sort(cand.begin(), cand.begin() + k, cand.end(), better);
        for (int j = 0; j < k; ++j) {
            indices[row * k + j] = cand[j].index;
        }
    }
    MNN_CONCURRENCY_END();
    return NO_ERROR;
}

ErrorCode CPUTopKV2::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    if (mRouteSigmoid) {
        return routeSigmoid(inputs, outputs);
    }
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
#ifdef MNN_USE_ARMV82
        auto core = static_cast<CPUBackend*>(backend())->functions();
        if (core->bytes == 2 && halide_type_float == inputTensor->getType().code) {
            auto inputData = reinterpret_cast<const __fp16*>(inputTensor->host<void>());
            auto topkData = reinterpret_cast<__fp16*>(outputData->host<void>());
            auto indicesData = outputIndices->host<int32_t>();
            MNN_CONCURRENCY_BEGIN(i, numRows) {
                auto in = inputData + i * rowSize;
                auto outV = topkData + i * k;
                auto outI = indicesData + i * k;
                MNNVectorTop1FP16(in, outV, outI, rowSize);
            }
            MNN_CONCURRENCY_END();
            return NO_ERROR;
        }
#endif
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

#ifdef MNN_USE_ARMV82
    auto core = static_cast<CPUBackend*>(backend())->functions();
    if (core->bytes == 2 && halide_type_float == inputTensor->getType().code) {
        auto inputData = reinterpret_cast<const __fp16*>(inputTensor->host<void>());
        auto topkData = reinterpret_cast<__fp16*>(outputData->host<void>());
        auto indicesData = outputIndices->host<int32_t>();
        findTopKFP16(rowSize, numRows, inputData, k, indicesData, topkData, mLargest);
        return NO_ERROR;
    }
#endif
    if (halide_type_float == inputTensor->getType().code) {
        auto inputData = inputTensor->host<float>();
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
