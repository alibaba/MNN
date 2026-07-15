//
//  CUDATopKV2LargestFlagTest.cpp
//  MNNTests
//
//  Created by MNN on 2026/07/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <cstring>
#include <memory>
#include <vector>

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>

#include "MNNTestSuite.h"
#include "TestUtils.h"

using namespace MNN;
using namespace MNN::Express;

static std::vector<VARP> _TopKV2WithLargest(VARP input, VARP k, bool largest) {
    std::unique_ptr<TopKV2T> topk(new TopKV2T);
    topk->largest = largest;

    std::unique_ptr<OpT> op(new OpT);
    op->type       = OpType_TopKV2;
    op->main.type  = OpParameter_TopKV2;
    op->main.value = topk.release();

    auto expr    = Expr::create(op.get(), {input, k}, 2);
    auto values  = Variable::create(expr, 0);
    auto indices = Variable::create(expr, 1);
    return {values, indices};
}

class CUDATopKV2LargestFlagTest : public MNNTestCase {
public:
    virtual ~CUDATopKV2LargestFlagTest() = default;

    bool runLargestCase() {
        const int rowCount  = 1;
        const int rowLength = 8;
        const int k         = 4;
        const std::vector<float> inputData = {3.0f, -1.0f, 2.0f, -4.0f, 0.5f, -2.0f, 1.0f, 4.0f};
        const std::vector<float> expectedValues = {4.0f, 3.0f, 2.0f, 1.0f};
        const std::vector<int> expectedIndices  = {7, 0, 2, 6};

        auto input = _Input({rowCount, rowLength}, NCHW, halide_type_of<float>());
        auto kVar  = _Input({1}, NCHW, halide_type_of<int>());
        ::memcpy(input->writeMap<float>(), inputData.data(), inputData.size() * sizeof(float));
        input->unMap();
        kVar->writeMap<int>()[0] = k;
        kVar->unMap();

        auto outputs = _TopKV2(input, kVar);
        auto values  = outputs[0]->readMap<float>();
        auto indices = outputs[1]->readMap<int>();
        if (!checkVectorByRelativeError<float>(values, expectedValues.data(), rowCount * k, 0.001f)) {
            MNN_ERROR("CUDATopKV2LargestFlag largest value test failed\n");
            return false;
        }
        if (!checkVector<int>(indices, expectedIndices.data(), rowCount * k, 0)) {
            MNN_ERROR("CUDATopKV2LargestFlag largest index test failed\n");
            return false;
        }
        return true;
    }

    bool runSmallestCase() {
        const int rowCount  = 1;
        const int rowLength = 8;
        const int k         = 4;
        const std::vector<float> inputData = {3.0f, -1.0f, 2.0f, -4.0f, 0.5f, -2.0f, 1.0f, 4.0f};
        const std::vector<float> expectedValues = {-4.0f, -2.0f, -1.0f, 0.5f};
        const std::vector<int> expectedIndices  = {3, 5, 1, 4};

        auto input = _Input({rowCount, rowLength}, NCHW, halide_type_of<float>());
        auto kVar  = _Input({1}, NCHW, halide_type_of<int>());
        ::memcpy(input->writeMap<float>(), inputData.data(), inputData.size() * sizeof(float));
        input->unMap();
        kVar->writeMap<int>()[0] = k;
        kVar->unMap();

        auto outputs = _TopKV2WithLargest(input, kVar, false);
        auto values  = outputs[0]->readMap<float>();
        auto indices = outputs[1]->readMap<int>();
        if (!checkVectorByRelativeError<float>(values, expectedValues.data(), rowCount * k, 0.001f)) {
            MNN_ERROR("CUDATopKV2LargestFlag smallest value test failed\n");
            return false;
        }
        if (!checkVector<int>(indices, expectedIndices.data(), rowCount * k, 0)) {
            MNN_ERROR("CUDATopKV2LargestFlag smallest index test failed\n");
            return false;
        }
        return true;
    }

    virtual bool run(int precision) {
        if (!runLargestCase()) {
            return false;
        }
        return runSmallestCase();
    }
};

MNNTestSuiteRegister(CUDATopKV2LargestFlagTest, "op/CUDATopKV2LargestFlag");
