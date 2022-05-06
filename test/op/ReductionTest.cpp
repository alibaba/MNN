//
//  ReductionTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/01/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"
using namespace MNN::Express;
class ReduceSumTest : public MNNTestCase {
public:
    virtual ~ReduceSumTest() = default;
    virtual bool run(int precision) {
        auto input = _Input(
            {
                4,
            },
            NCHW);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {-1.0, -2.0, 3.0, 4.0};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 4 * sizeof(float));
        input->unMap();
        auto output                             = _ReduceSum(input);
        const std::vector<float> expectedOutput = {4.0};
        auto gotOutput                          = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 1, 0.01)) {
            MNN_ERROR("ReduceSumTest test failed!\n");
            return false;
        }
        return true;
    }
};
class ReduceSumMultiTest : public MNNTestCase {
public:
    virtual ~ReduceSumMultiTest() = default;
    virtual bool run(int precision) {
        auto input = _Input({4, 10, 1, 4}, NCHW, halide_type_of<float>());
        // set input data
        auto inputPtr  = input->writeMap<float>();
        auto inputInfo = input->getInfo();
        std::vector<float> inputData(inputInfo->size);
        for (int i = 0; i < inputData.size(); ++i) {
            inputData[i] = (float)((10.3 - i) * (i + 0.2));
        }
        memcpy(inputPtr, inputData.data(), inputData.size() * sizeof(float));
        input->unMap();
        auto output = _ReduceSum(input, {0, 2, 3});
        std::vector<float> expectedOutput(10);
        auto func = FP32Converter[precision];
        for (int i = 0; i < 10; ++i) {
            float sumValue = 0.0f;
            for (int j = 0; j < 4; ++j) {
                for (int k = 0; k < 4; ++k) {
                    sumValue = func(func(inputData[i * 4 + k + j * 40]) + sumValue);
                }
            }
            expectedOutput[i] = sumValue;
        }
        auto gotOutput = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 1, 0.01)) {
            MNN_ERROR("ReduceSumMultiTest test failed!\n");
            return false;
        }
        return true;
    }
};
class ReduceMeanTest : public MNNTestCase {
public:
    virtual ~ReduceMeanTest() = default;
    virtual bool run(int precision) {
        auto input = _Input(
            {
                4,
            },
            NCHW);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {-1.0, -2.0, 3.0, 4.0};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 4 * sizeof(float));
        input->unMap();
        auto output                             = _ReduceMean(input);
        const std::vector<float> expectedOutput = {1.0};
        auto gotOutput                          = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 1, 0.01)) {
            MNN_ERROR("ReduceMeanTest test failed!\n");
            return false;
        }
        return true;
    }
};
class ReduceMaxTest : public MNNTestCase {
public:
    virtual ~ReduceMaxTest() = default;
    virtual bool run(int precision) {
        auto input = _Input(
            {
                4,
            },
            NCHW);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {-1.0, -2.0, 3.0, 4.0};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 4 * sizeof(float));
        input->unMap();
        auto output                             = _ReduceMax(input);
        const std::vector<float> expectedOutput = {4.0};
        auto gotOutput                          = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 1, 0.01)) {
            MNN_ERROR("ReduceMaxTest test failed!\n");
            return false;
        }
        return true;
    }
};
class ReduceMinTest : public MNNTestCase {
public:
    virtual ~ReduceMinTest() = default;
    virtual bool run(int precision) {
        auto input = _Input(
            {
                4,
            },
            NCHW);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {-1.0, -2.0, 3.0, 4.0};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 4 * sizeof(float));
        input->unMap();
        auto output                             = _ReduceMin(input);
        const std::vector<float> expectedOutput = {-2.0};
        auto gotOutput                          = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 1, 0.01)) {
            MNN_ERROR("ReduceMinTest test failed!\n");
            return false;
        }
        return true;
    }
};
class ReduceProdTest : public MNNTestCase {
public:
    virtual ~ReduceProdTest() = default;
    virtual bool run(int precision) {
        auto input = _Input(
            {
                4,
            },
            NCHW);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {-1.0, -2.0, 3.0, 4.0};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 4 * sizeof(float));
        input->unMap();
        auto output                             = _ReduceProd(input);
        const std::vector<float> expectedOutput = {24.0};
        auto gotOutput                          = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 1, 0.01)) {
            MNN_ERROR("ReduceProdTest test failed!\n");
            return false;
        }
        return true;
    }
};
MNNTestSuiteRegister(ReduceSumTest, "op/reduction/reduce_sum");
MNNTestSuiteRegister(ReduceSumMultiTest, "op/reduction/reduce_sum_multi");
MNNTestSuiteRegister(ReduceMeanTest, "op/reduction/reduce_mean");
MNNTestSuiteRegister(ReduceMaxTest, "op/reduction/reduce_max");
MNNTestSuiteRegister(ReduceMinTest, "op/reduction/reduce_min");
MNNTestSuiteRegister(ReduceProdTest, "op/reduction/reduce_prod");
