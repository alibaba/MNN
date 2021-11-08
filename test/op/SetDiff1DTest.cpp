//
//  SetDiff1DTest.cpp
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
class SetDiff1DTest : public MNNTestCase {
public:
    virtual ~SetDiff1DTest() = default;
    virtual bool run(int precision) {
        auto input_x = _Input({16}, NHWC, halide_type_of<int>());
        auto input_y = _Input({8}, NHWC, halide_type_of<int>());
        input_x->setName("input_x");
        input_y->setName("input_y");
        // set input data
        const int x_data[] = {-1, 2, -3, 4, 5, -6, 7, -8, -9, -10, 11, 12, 13, 14, -15, -16};
        const int y_data[] = {-1, 2, -3, 4, 5, -6, 7, -8};
        auto xPtr          = input_x->writeMap<int>();
        auto yPtr          = input_y->writeMap<int>();
        memcpy(xPtr, x_data, 16 * sizeof(int));
        memcpy(yPtr, y_data, 8 * sizeof(int));
        input_x->unMap();
        input_y->unMap();
        auto output                           = _SetDiff1D(input_x, input_y);
        const std::vector<int> expectedOutput = {-9, -10, 11, 12, 13, 14, -15, -16};
        auto gotOutput                        = output->readMap<int>();
        if (!checkVector<int>(gotOutput, expectedOutput.data(), 8, 0)) {
            MNN_ERROR("SetDiff1DTest test failed!\n");
            return false;
        }
        return true;
    }
};
MNNTestSuiteRegister(SetDiff1DTest, "op/setdiff1d");
