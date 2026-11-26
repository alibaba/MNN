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
#include <algorithm>
#include <cmath>
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

static std::vector<VARP> _MoeRoute(VARP logits, VARP k, VARP axis, VARP bias) {
    std::unique_ptr<MNN::TopKV2T> topk(new MNN::TopKV2T);
    topk->largest = true;
    topk->routeSigmoid = true;

    std::unique_ptr<MNN::OpT> op(new MNN::OpT);
    op->type = MNN::OpType_TopKV2;
    op->main.type = MNN::OpParameter_TopKV2;
    op->main.value = topk.release();

    auto expr = Expr::create(op.get(), {logits, k, axis, bias}, 2);
    return {Variable::create(expr, 0), Variable::create(expr, 1)};
}

static bool runRouteSigmoid(int precision) {
    const float tol = precision == 2 ? 0.002f : 1e-5f;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-6.0f, 6.0f);
    for (int width : {8, 60, 64, 129}) {
        for (int rows : {1, 3, 17}) {
            for (int k : {1, std::min(width, 4), std::min(width, 16)}) {
                for (bool constBias : {false, true}) {
                auto logits = _Input({rows, width}, NCHW, halide_type_of<float>());
                std::vector<float> biasData(width);
                for (int i = 0; i < width; ++i) {
                    biasData[i] = dist(rng) * 0.5f;
                }
                VARP bias;
                if (constBias) {
                    bias = _Const(biasData.data(), {width}, NCHW);
                } else {
                    bias = _Input({width}, NCHW, halide_type_of<float>());
                    ::memcpy(bias->writeMap<float>(), biasData.data(), width * sizeof(float));
                }
                const float* biasPtr = biasData.data();
                auto x = logits->writeMap<float>();
                for (int i = 0; i < rows * width; ++i) {
                    x[i] = dist(rng);
                }
                auto outputs = _MoeRoute(logits, _Scalar<int>(k), _Scalar<int>(-1), bias);
                auto probs = outputs[0]->readMap<float>();
                auto ids = outputs[1]->readMap<int>();
                if (probs == nullptr || ids == nullptr) {
                    MNN_ERROR("MoeRoute map failed: width=%d rows=%d k=%d\n", width, rows, k);
                    return false;
                }
                for (int r = 0; r < rows; ++r) {
                    std::vector<float> refP(width), refS(width);
                    std::vector<int> order(width);
                    for (int i = 0; i < width; ++i) {
                        refP[i] = 1.0f / (1.0f + expf(-x[r * width + i]));
                        refS[i] = refP[i] + biasPtr[i];
                        order[i] = i;
                        if (fabsf(probs[r * width + i] - refP[i]) > tol) {
                            MNN_ERROR("MoeRoute prob mismatch: width=%d rows=%d k=%d [%d,%d] %f vs %f\n",
                                      width, rows, k, r, i, probs[r * width + i], refP[i]);
                            return false;
                        }
                    }
                    std::sort(order.begin(), order.end(), [&](int a, int b) {
                        return refS[a] > refS[b] || (refS[a] == refS[b] && a < b);
                    });
                    std::vector<bool> seen(width, false);
                    for (int j = 0; j < k; ++j) {
                        int idx = ids[r * k + j];
                        if (idx < 0 || idx >= width || seen[idx]) {
                            MNN_ERROR("MoeRoute bad index: width=%d rows=%d k=%d idx=%d\n", width, rows, k, idx);
                            return false;
                        }
                        seen[idx] = true;
                        if (fabsf(refS[idx] - refS[order[j]]) > tol) {
                            MNN_ERROR("MoeRoute selection mismatch: width=%d rows=%d k=%d row=%d j=%d\n",
                                      width, rows, k, r, j);
                            return false;
                        }
                    }
                }
                }
            }
        }
    }
    {
        const int width = 60, k = 4;
        auto logits = _Input({1, width}, NCHW, halide_type_of<float>());
        auto x = logits->writeMap<float>();
        for (int i = 0; i < width; ++i) {
            x[i] = 0.0f;
        }
        auto zeroBias = _Input({width}, NCHW, halide_type_of<float>());
        ::memset(zeroBias->writeMap<float>(), 0, width * sizeof(float));
        auto outputs = _MoeRoute(logits, _Scalar<int>(k), _Scalar<int>(-1), zeroBias);
        auto ids = outputs[1]->readMap<int>();
        for (int j = 0; j < k; ++j) {
            if (ids[j] != j) {
                MNN_ERROR("MoeRoute tie-break failed: j=%d idx=%d\n", j, ids[j]);
                return false;
            }
        }
    }
    return true;
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

    template <typename T>
    bool runSmallRows() {
        for (int width : {1, 3, 31, 32, 60, 64, 65, 127, 128, 129}) {
            for (int rows : {1, 3}) {
                for (int k : {1, std::min(width, 4), std::min(width, 16)}) {
                    for (bool largest : {false, true}) {
                        auto input = _Input({rows, width}, NCHW, halide_type_of<T>());
                        std::vector<T> data(rows * width);
                        for (int i = 0; i < data.size(); ++i) {
                            data[i] = static_cast<T>((i * 17) % 11 - 5);
                        }
                        ::memcpy(input->template writeMap<T>(), data.data(), data.size() * sizeof(T));
                        auto outputs = _TopKV2WithLargest(input, _Scalar<int>(k), largest);
                        auto values = outputs[0]->template readMap<T>();
                        auto indices = outputs[1]->template readMap<int>();
                        if (values == nullptr || indices == nullptr) {
                            return false;
                        }
                        for (int row = 0; row < rows; ++row) {
                            std::vector<T> expected(data.begin() + row * width, data.begin() + (row + 1) * width);
                            std::sort(expected.begin(), expected.end());
                            if (largest) {
                                std::reverse(expected.begin(), expected.end());
                            }
                            std::vector<bool> seen(width, false);
                            for (int j = 0; j < k; ++j) {
                                int index = indices[row * k + j];
                                if (index < 0 || index >= width || seen[index] || values[row * k + j] != expected[j] ||
                                    values[row * k + j] != data[row * width + index]) {
                                    MNN_ERROR("TopKV2 small row failed: width=%d rows=%d k=%d largest=%d\n", width,
                                              rows, k, largest);
                                    return false;
                                }
                                seen[index] = true;
                            }
                        }
                    }
                }
            }
        }
        return true;
    }

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
        if (!runRouteSigmoid(precision)) {
            return false;
        }
        if (!runSmallRows<float>() || !runSmallRows<int>()) {
            return false;
        }
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
