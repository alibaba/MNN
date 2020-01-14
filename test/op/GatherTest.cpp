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
    virtual bool run() {
        auto params = _Input({2,2}, NCHW);
        params->setName("input_tensor");
        // set input data
        const float inpudata[] = {-1.0, -2.0, 3.0, 4.0};
        auto inputPtr          = params->writeMap<float>();
        memcpy(inputPtr, inpudata, 4 * sizeof(float));
        params->unMap();
	const int indices_data[] = {0, 0, 1, 1};
        auto indices = _Const(indices_data, {2, 2}, NCHW, halide_type_of<int>()); 
        auto output = _GatherND(params, indices);
        const std::vector<float> expectedOutput = {-1.0, 4.0};
        auto gotOutput = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 2, 0.01)) {
            MNN_ERROR("GatherNDTest test failed!\n");
            return false;
        }
        return true;
    }
};
class GatherTest : public MNNTestCase {
public:
    virtual ~GatherTest() = default;
    virtual bool run() {
        auto params = _Input({4,3,2}, NCHW);
        params->setName("input_tensor");
        // set input data
        const float inpudata[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21,0, 22.0, 23.0, 24.0};
        auto inputPtr          = params->writeMap<float>();
        memcpy(inputPtr, inpudata, 24 * sizeof(float));
        params->unMap();
        const int indices_data[] = {1, 0, 1, 0};
        auto indices = _Const(indices_data, {4}, NCHW, halide_type_of<int>());
        auto output = _Gather(params, indices);
        const std::vector<float> expectedOutput = {7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        auto gotOutput = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 24, 0.01)) {
            MNN_ERROR("GatherTest test failed!\n");
            return false;
        }
        return true;
    }
};
MNNTestSuiteRegister(GatherNDTest, "op/gather_nd");
MNNTestSuiteRegister(GatherTest, "op/gather");
