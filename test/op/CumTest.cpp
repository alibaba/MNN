//
//  CumTest.cpp
//  MNNTests
//
//  Created by MNN on 2022/05/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"

using namespace MNN::Express;
class CumProdTest : public MNNTestCase {
public:
    virtual ~CumProdTest() = default;
    virtual bool run(int precision) {
        auto input = _Input({2, 2, 2},NCHW);
        input->setName("input_tensor");
        const float inpudata[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 8 * sizeof(float));
        auto output0                             = _CumProd(input, 0);
        const std::vector<float> expectedOutput0 = {1., 2., 3., 4., 5., 12., 21., 32.};
        auto gotOutput0                          = output0->readMap<float>();
        if (!checkVector<float>(gotOutput0, expectedOutput0.data(), 8, 0.01)) {
            MNN_ERROR("CumProdTest axis=0 test failed!\n");
            return false;
        }
        auto output1                             = _CumProd(input, 1);
        const std::vector<float> expectedOutput1 = {1., 2., 3., 8., 5., 6., 35., 48.};
        auto gotOutput1                          = output1->readMap<float>();
        if (!checkVector<float>(gotOutput1, expectedOutput1.data(), 8, 0.01)) {
            MNN_ERROR("CumProdTest axis=1 test failed!\n");
            return false;
        }
        auto output2                             = _CumProd(input, 2);
        const std::vector<float> expectedOutput2 = {1., 2., 3., 12., 5., 30., 7., 56.};
        auto gotOutput2                          = output2->readMap<float>();
        if (!checkVector<float>(gotOutput2, expectedOutput2.data(), 8, 0.01)) {
            MNN_ERROR("CumProdTest axis=2 test failed!\n");
            return false;
        }
        return true;
    }
};
MNNTestSuiteRegister(CumProdTest, "op/cumprod");

class CumSumTest : public MNNTestCase {
public:
    virtual ~CumSumTest() = default;
    virtual bool run(int precision) {
        auto input = _Input({2, 2, 2},NCHW);
        input->setName("input_tensor");
        const float inpudata[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 8 * sizeof(float));
        auto output0                             = _CumSum(input, 0);
        const std::vector<float> expectedOutput0 = {1., 2., 3., 4., 6., 8., 10., 12.};
        auto gotOutput0                          = output0->readMap<float>();
        if (!checkVector<float>(gotOutput0, expectedOutput0.data(), 8, 0.01)) {
            MNN_ERROR("CumSumTest axis=0 test failed!\n");
            return false;
        }
        auto output1                             = _CumSum(input, 1);
        const std::vector<float> expectedOutput1 = {1., 2., 4., 6., 5., 6., 12., 14.};
        auto gotOutput1                          = output1->readMap<float>();
        if (!checkVector<float>(gotOutput1, expectedOutput1.data(), 8, 0.01)) {
            MNN_ERROR("CumSumTest axis=1 test failed!\n");
            return false;
        }
        auto output2                             = _CumSum(input, 2);
        const std::vector<float> expectedOutput2 = {1., 3., 3., 7., 5., 11., 7., 15.};
        auto gotOutput2                          = output2->readMap<float>();
        if (!checkVector<float>(gotOutput2, expectedOutput2.data(), 8, 0.01)) {
            MNN_ERROR("CumSumTest axis=2 test failed!\n");
            return false;
        }
        return true;
    }
};
MNNTestSuiteRegister(CumSumTest, "op/cumsum");
