//
//  TopKV2Execution.hpp
//  MNN
//
//  Created by MNN on 2023/07/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/AutoTime.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"
#include <cstring>
#include <memory>
#include <random>
#include <vector>

using namespace MNN::Express;

template <typename valueT, typename indexT>
void MinHeapify(valueT* arr, indexT* index, int size, int i) {
    int l = 2 * i + 1;
    int r = 2 * i + 2;
    int smallest = i;
    if (l < size && arr[l] < arr[smallest]) {
        smallest = l;
    }
    if (r < size && arr[r] < arr[smallest]) {
        smallest = r;
    }
    if (smallest != i) {
        std::swap(arr[i], arr[smallest]);
        std::swap(index[i], index[smallest]);
        MinHeapify<valueT, indexT>(arr, index, size, smallest);
    }

    return;
}

template <typename valueT, typename indexT>
void BuildMinHeap(valueT* arr, indexT* index, int size) {
    for (int i = size / 2 - 1; i >= 0; i--) {
        MinHeapify<valueT, indexT>(arr, index, size, i);
    }
}

template <typename valueT, typename indexT>
void Sort(valueT* values, indexT* indices, const int num) {
    valueT* _values = static_cast<valueT*>(values);
    indexT* _indices = static_cast<indexT*>(indices);
    for (int i = 0; i < num - 1; i++) {
        for (int j = 0; j < num - i - 1; j++) {
            if (_values[j] < _values[j + 1]) {
                std::swap(_values[j], _values[j + 1]);
                std::swap(_indices[j], _indices[j + 1]);
            }
        }
    }

    return;
}

template <typename valueT, typename indexT>
void CpuKernelOneRow(const valueT* input, indexT* outputIndices, valueT* outputValues, const int K, const int length) {
    for (int i = 0; i < K; i++) {
        outputIndices[i] = i;
        outputValues[i] = input[i];
    }
    BuildMinHeap<valueT, indexT>(outputValues, outputIndices, K);
    for (int i = K; i < length; i++) {
        if (input[i] > outputValues[0]) {
            outputValues[0] = input[i];
            outputIndices[0] = i;
            MinHeapify<valueT, indexT>(outputValues, outputIndices, K, 0);
        }
    }
    Sort<valueT, indexT>(outputValues, outputIndices, K);

    return;
}

template <typename indexT, typename valueT>
void CpuKernelAllRows(valueT* input, indexT* outputIndices, valueT* outputValues, const int K, const int lengthRow,
                      const int numRow, int descendFlag) {
    for (int i = 0; i < lengthRow * numRow; i++) {
        input[i] = input[i] * descendFlag;
    }

    for (int i = 0; i < numRow; i++) {
        const valueT* inputThisRow = input + lengthRow * i;
        indexT* outputIndicesThisRow = outputIndices + K * i;
        valueT* outputValuesThisRow = outputValues + K * i;
        CpuKernelOneRow(inputThisRow, outputIndicesThisRow, outputValuesThisRow, K, lengthRow);
    }

    for (int i = 0; i < lengthRow * numRow; i++) {
        input[i] = input[i] * descendFlag;
    }

    for (int i = 0; i < numRow * K; i++) {
        outputValues[i] = outputValues[i] * descendFlag;
    }

    return;
}

void RandomInitFloat(float* array, const int& numEle) {
    std::mt19937 rng(4);
    std::uniform_real_distribution<float> dist(0.0, 1.0);

    for (int i = 0; i < numEle; i++) {
        array[i] = dist(rng);
    }
    return;
}

void SetK(int* valuePtr, const int K) {
    *valuePtr = K;
}

static std::vector<VARP> _TopKV2WithLargest(VARP input, VARP k, bool largest) {
    std::unique_ptr<MNN::TopKV2T> topk(new MNN::TopKV2T);
    topk->largest = largest;

    std::unique_ptr<MNN::OpT> op(new MNN::OpT);
    op->type = MNN::OpType_TopKV2;
    op->main.type = MNN::OpParameter_TopKV2;
    op->main.value = topk.release();

    auto expr = Expr::create(op.get(), {input, k}, 2);
    auto values = Variable::create(expr, 0);
    auto indices = Variable::create(expr, 1);
    return {values, indices};
}

bool checkIndicesHalf(const float* input, const float* expectedOutput0, const int* gotOutput1, const int K,
                      const int numRow, const int lengthRow) {
    for (int i = 0; i < numRow; i++) {
        for (int j = 0; j < K; j++) {
            bool condition =
                (fabs((expectedOutput0[i * K + j]) - input[gotOutput1[i * K + j] + i * lengthRow]) > 0.02f);
            if (condition) {
                MNN_PRINT("Conflict: Number %d. Value Correct is %f. Value Computed is %f.\n", i * K + j,
                          convertFP32ToFP16(expectedOutput0[i * K + j]),
                          convertFP32ToFP16(input[gotOutput1[i * K + j] + i * lengthRow]));
                return false;
            }
        }
    }

    return true;
}

bool checkIndicesFloat(const float* input, const float* expectedOutput0, const int* gotOutput1, const int K,
                       const int numRow, const int lengthRow) {
    for (int i = 0; i < numRow; i++) {
        for (int j = 0; j < K; j++) {
            bool condition = (expectedOutput0[i * K + j] != input[gotOutput1[i * K + j] + i * lengthRow]);
            if (condition) {
                MNN_PRINT("Conflict: Number %d. Value Correct is %f. Value Computed is %f.\n", i * K + j,
                          expectedOutput0[i * K + j], input[gotOutput1[i * K + j] + i * lengthRow]);
                return false;
            }
        }
    }

    return true;
}

void printTimeCost(uint64_t timeCost) {
    uint64_t seconds = timeCost / 1000000;
    uint64_t microseconds = timeCost % 1000000;
    MNN_PRINT("%lu s %lu ms\n", seconds, microseconds / 1000);

    return;
}

class TopKV2Test : public MNNTestCase {
public:
    virtual ~TopKV2Test() = default;

    bool runLargestFlagCase() {
        const int rowCount = 1;
        const int rowLength = 8;
        const int k = 4;
        const std::vector<float> inputData = {3.0f, -1.0f, 2.0f, -4.0f, 0.5f, -2.0f, 1.0f, 4.0f};
        const std::vector<float> expectedValues = {4.0f, 3.0f, 2.0f, 1.0f};
        const std::vector<int> expectedIndices = {7, 0, 2, 6};

        auto input = _Input({rowCount, rowLength}, NCHW, halide_type_of<float>());
        auto kVar = _Input({1}, NCHW, halide_type_of<int>());
        ::memcpy(input->writeMap<float>(), inputData.data(), inputData.size() * sizeof(float));
        input->unMap();
        kVar->writeMap<int>()[0] = k;
        kVar->unMap();

        auto outputs = _TopKV2(input, kVar);
        auto values = outputs[0]->readMap<float>();
        auto indices = outputs[1]->readMap<int>();
        if (!checkVectorByRelativeError<float>(values, expectedValues.data(), rowCount * k, 0.001f)) {
            MNN_ERROR("TopKV2 largest value test failed\n");
            return false;
        }
        if (!checkVector<int>(indices, expectedIndices.data(), rowCount * k, 0)) {
            MNN_ERROR("TopKV2 largest index test failed\n");
            return false;
        }
        return true;
    }

    bool runSmallestFlagCase() {
        const int rowCount = 1;
        const int rowLength = 8;
        const int k = 4;
        const std::vector<float> inputData = {3.0f, -1.0f, 2.0f, -4.0f, 0.5f, -2.0f, 1.0f, 4.0f};
        const std::vector<float> expectedValues = {-4.0f, -2.0f, -1.0f, 0.5f};
        const std::vector<int> expectedIndices = {3, 5, 1, 4};

        auto input = _Input({rowCount, rowLength}, NCHW, halide_type_of<float>());
        auto kVar = _Input({1}, NCHW, halide_type_of<int>());
        ::memcpy(input->writeMap<float>(), inputData.data(), inputData.size() * sizeof(float));
        input->unMap();
        kVar->writeMap<int>()[0] = k;
        kVar->unMap();

        auto outputs = _TopKV2WithLargest(input, kVar, false);
        auto values = outputs[0]->readMap<float>();
        auto indices = outputs[1]->readMap<int>();
        if (!checkVectorByRelativeError<float>(values, expectedValues.data(), rowCount * k, 0.001f)) {
            MNN_ERROR("TopKV2 smallest value test failed\n");
            return false;
        }
        if (!checkVector<int>(indices, expectedIndices.data(), rowCount * k, 0)) {
            MNN_ERROR("TopKV2 smallest index test failed\n");
            return false;
        }
        return true;
    }

    virtual bool run(int precision) {
        if (!runLargestFlagCase()) {
            return false;
        }
        if (!runSmallestFlagCase()) {
            return false;
        }
        // set params
        const int K = 300;
        const int numRow = 180;

        const int lengthRow = 21491;

        // set input
        VARP input0 = _Input({numRow, lengthRow}, NCHW, halide_type_of<float>());
        VARP input1 = _Input({1}, NCHW, halide_type_of<int>());
        RandomInitFloat(input0->writeMap<float>(), numRow * lengthRow);
        SetK(input1->writeMap<int>(), K);
        MNN::Timer _t;

        // calculate gotOutput
        auto res = _TopKV2(input0, input1);
        VARP output0 = res[0];
        VARP output1 = res[1];
        auto gotOutput0 = output0->readMap<float>();
        auto gotOutput1 = output1->readMap<int>();
        auto timeCost = _t.durationInUs();

        // calculate expectedOutput
        std::vector<float> expectedOutput0(numRow * K);
        std::vector<int> expectedOutput1(numRow * K);
        CpuKernelAllRows<int, float>(input0->writeMap<float>(), expectedOutput1.data(), expectedOutput0.data(), K,
                                     lengthRow, numRow, 1);

        printTimeCost(timeCost);

        // check values
        float errorScale = precision <= MNN::BackendConfig::Precision_High ? 1 : 20;
        if (!checkVectorByRelativeError<float>(gotOutput0, expectedOutput0.data(), numRow * K, 0.001 * errorScale)) {
            MNN_ERROR("TopKV2 test failed!\n");
            return false;
        }

        if (precision <= 1) {
            if (!checkVectorByRelativeError<int>(gotOutput1, expectedOutput1.data(), K, 1 * errorScale)) {
                MNN_ERROR("TopKV2 index test failed!\n");
                return false;
            }
        } else if (precision == 2) {
            if (!checkIndicesHalf(input0->readMap<float>(), expectedOutput0.data(), gotOutput1, K, numRow, lengthRow)) {
                MNN_ERROR("TopKV2 test failed!\n");
                return false;
            }
        }

        return true;
    }
};

MNNTestSuiteRegister(TopKV2Test, "op/TopKV2");
