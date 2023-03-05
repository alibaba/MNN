//
//  GatherTest.cpp
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
class GatherNDTest : public MNNTestCase {
public:
    virtual ~GatherNDTest() = default;
    bool _run(int precision, bool lazy) {
        {
            const float inpudata[]                  = {-1.0, -2.0, 3.0, 4.0};
            const int indices_data[]                = {0, 0, 1, 1};
            auto params                             = _Const(inpudata, {2, 2}, NCHW, halide_type_of<float>());
            auto indices                            = _Const(indices_data, {2, 2}, NCHW, halide_type_of<int>());
            auto output                             = _GatherND(params, indices);
            const std::vector<float> expectedOutput = {-1.0, 4.0};
            auto gotOutput                          = output->readMap<float>();
            if (!checkVectorByRelativeError<float>(gotOutput, expectedOutput.data(), 2, 0.001)) {
                MNN_ERROR("GatherNDTest test failed!\n");
                return false;
            }
        }
        {
            const float inpudata[]                  = {1, 2, 3, 4, 5, 6, 7, 8};
            const int indices_data[]                = {0, 0, 1, 1, 1, 0};
            auto params                             = _Const(inpudata, {2, 2, 2}, NCHW, halide_type_of<float>());
            auto indices                            = _Const(indices_data, {2, 3}, NCHW, halide_type_of<int>());
            auto output                             = _GatherND(params, indices);
            const std::vector<float> expectedOutput = {2, 7};
            auto gotOutput                          = output->readMap<float>();
            if (!checkVectorByRelativeError<float>(gotOutput, expectedOutput.data(), 2, 0.001)) {
                MNN_ERROR("GatherNDTest test failed!\n");
                return false;
            }
        }
        return true;
    }
    virtual bool run(int precision) {
        ExecutorScope::Current()->lazyEval = false;
        auto res = _run(precision, false);
        if (!res) {
            FUNC_PRINT(1);
            return false;
        }
        ExecutorScope::Current()->lazyEval = true;
        ExecutorScope::Current()->setLazyComputeMode(MNN::Express::Executor::LAZY_CONTENT);
        res = _run(precision, true);
        if (!res) {
            FUNC_PRINT(1);
            return false;
        }
        ExecutorScope::Current()->setLazyComputeMode(MNN::Express::Executor::LAZY_FULL);
        res = _run(precision, true);
        return res;
    }
};
class GatherTest : public MNNTestCase {
public:
    virtual ~GatherTest() = default;
    bool _run(int precision, bool lazy) {
        auto params = _Input({4, 3, 2}, NCHW);
        params->setName("input_tensor");
        // set input data
        const float inpudata[] = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
                                  14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21,  0,   22.0, 23.0, 24.0};
        auto inputPtr          = params->writeMap<float>();
        memcpy(inputPtr, inpudata, 24 * sizeof(float));
        params->unMap();
        const int indices_data[]                = {1, 0, 1, 0};
        auto indices                            = _Const(indices_data, {4}, NCHW, halide_type_of<int>());
        auto output                             = _Gather(params, indices);
        const std::vector<float> expectedOutput = {7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                                                   7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        auto gotOutput                          = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 24, 0.001)) {
            MNN_ERROR("GatherTest test failed!\n");
            return false;
        }
        return true;
    }
    virtual bool run(int precision) {
        ExecutorScope::Current()->lazyEval = false;
        auto res = _run(precision, false);
        if (!res) {
            FUNC_PRINT(1);
            return false;
        }
        ExecutorScope::Current()->lazyEval = true;
        ExecutorScope::Current()->setLazyComputeMode(MNN::Express::Executor::LAZY_CONTENT);
        res = _run(precision, true);
        if (!res) {
            FUNC_PRINT(1);
            return false;
        }
        ExecutorScope::Current()->setLazyComputeMode(MNN::Express::Executor::LAZY_FULL);
        res = _run(precision, true);
        return res;
    }
};
MNNTestSuiteRegister(GatherNDTest, "op/gather_nd");
MNNTestSuiteRegister(GatherTest, "op/gather");
