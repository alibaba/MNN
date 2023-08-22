//
//  TopKV2Execution.hpp
//  MNN
//
//  Created by MNN on 2023/07/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"
#include <random>
#include <vector>

using namespace MNN::Express;


template<typename valueT, typename indexT>
void MinHeapify(valueT * arr, indexT * index, int size, int i) {
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


template<typename valueT, typename indexT>
void BuildMinHeap(valueT * arr, indexT * index, int size) {
    for (int i = size / 2 - 1; i >= 0; i--) {
        MinHeapify<valueT, indexT>(arr, index, size, i);
    }
}


template<typename valueT, typename indexT>
void Sort(valueT * values, indexT * indices, const int num) {
    valueT * _values = static_cast<valueT *>(values);
    indexT * _indices = static_cast<indexT *>(indices);
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


template<typename valueT, typename indexT>
void CpuKernelOneRow(const valueT * input, indexT * outputIndices, valueT * outputValues, const int K, const int length) {
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


template<typename indexT, typename valueT>
void CpuKernelAllRows(valueT * input, indexT * outputIndices, valueT * outputValues, const int K, const int lengthRow, const int numRow, int descendFlag) {
    for (int i = 0; i < lengthRow * numRow; i++) {
        input[i] = input[i] * descendFlag;
    }

    for (int i = 0; i < numRow; i++) {
        const valueT * inputThisRow = input + lengthRow * i;
        indexT * outputIndicesThisRow = outputIndices + K * i;
        valueT * outputValuesThisRow = outputValues + K * i;
        CpuKernelOneRow(inputThisRow, outputIndicesThisRow, outputValuesThisRow, K, lengthRow);
    }

    for (int i = 0; i < lengthRow * numRow; i++) {
        input[i] = input[i] * descendFlag;
    }

    for (int i = 0 ; i < numRow * K; i++) {
        outputValues[i] = outputValues[i] * descendFlag;
    }

    return;
}


void RandomInitFloat(float * array, const int & numEle) {
    std::mt19937 rng(4);
    std::uniform_real_distribution<float> dist(0.0, 1.0);

    for (int i = 0; i < numEle; i++) {
        array[i] = dist(rng);
    }
    return;
}


void SetK(int * valuePtr, const int K) {
    *valuePtr = K;
}


bool checkIndicesHalf(const float * input, const float * expectedOutput0, const int * gotOutput1, const int K, const int numRow, const int lengthRow) {
    for (int i = 0; i < numRow; i++) {
        for (int j = 0; j < K; j++) {
            bool condition = (convertFP32ToFP16(expectedOutput0[i * K + j]) != convertFP32ToFP16(input[gotOutput1[i * K + j] + i * lengthRow]));
            if (condition) {
                    MNN_PRINT("Conflict: Number %d. Value Correct is %f. Value Computed is %f.\n", i * K + j, convertFP32ToFP16(expectedOutput0[i * K + j]), convertFP32ToFP16(input[gotOutput1[i * K + j] + i * lengthRow]));
                    return false;
            }
        }
    }

    return true;
}


bool checkIndicesFloat(const float * input, const float * expectedOutput0, const int * gotOutput1, const int K, const int numRow, const int lengthRow) {
    for (int i = 0; i < numRow; i++) {
        for (int j = 0; j < K; j++) {
            bool condition = (expectedOutput0[i * K + j] != input[gotOutput1[i * K + j] + i * lengthRow]);
            if (condition) {
                    MNN_PRINT("Conflict: Number %d. Value Correct is %f. Value Computed is %f.\n", i * K + j, expectedOutput0[i * K + j], input[gotOutput1[i * K + j] + i * lengthRow]);
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

    virtual bool run(int precision) {
        // set params
        const int K = 10;
        const int numRow = 180;
        
        const int lengthRow = 21491;

        // set input
        VARP input0 = _Input({numRow, lengthRow}, NCHW, halide_type_of<float>());
        VARP input1 = _Input({1}, NCHW, halide_type_of<int>());
        RandomInitFloat(input0->writeMap<float>(), numRow * lengthRow);
        SetK(input1->writeMap<int>(), K);

        auto timeStart = getTimeInUs();
        // calculate gotOutput
        auto res = _TopKV2(input0, input1);
        VARP output0 = res[0];
        VARP output1 = res[1];
        auto gotOutput0                        = output0->readMap<float>();
        auto gotOutput1                        = output1->readMap<int>();
        auto timeEnd = getTimeInUs();
        auto timeCost = timeEnd - timeStart;

        // calculate expectedOutput
        std::vector<float> expectedOutput0(numRow * K);
        std::vector<int> expectedOutput1(numRow * K);
        CpuKernelAllRows<int, float>(input0->writeMap<float>(), expectedOutput1.data(), expectedOutput0.data(), K, lengthRow, numRow, 1);

        printTimeCost(timeCost);

        // check values
        float errorScale = precision <= MNN::BackendConfig::Precision_High ? 1 : 20;
        if (!checkVectorByRelativeError<float>(gotOutput0, expectedOutput0.data(), numRow * K, 0.001 * errorScale)) {
            MNN_ERROR("TopKV2 test failed!\n");
            return false;
        }

        // check indices
        if (precision <= 1) {
            if (!checkIndicesFloat(input0->readMap<float>(), expectedOutput0.data(), gotOutput1, K, numRow, lengthRow)) {
                MNN_ERROR("TopKV2 test failed!\n");
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
