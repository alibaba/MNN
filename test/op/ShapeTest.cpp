//
//  ShapeTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/12/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"

using namespace MNN::Express;
class ShapeTest : public MNNTestCase {
public:
    virtual ~ShapeTest() = default;
    virtual bool run(int precision) {
        auto input = _Input({1, 1, 1, 4}, NCHW);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {-1.0, -2.0, 3.0, 4.0};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 4 * sizeof(float));
        input->unMap();
        auto output                           = _Shape(input);
        const std::vector<int> expectedOutput = {1, 1, 1, 4};
        auto gotOutput                        = output->readMap<int>();
        if (!checkVector<int>(gotOutput, expectedOutput.data(), 4, 0)) {
            MNN_ERROR("ShapeTest test failed!\n");
            return false;
        }
        return true;
    }
};
MNNTestSuiteRegister(ShapeTest, "op/shape");
