//
//  LinSpaceTest.cpp
//  MNNTests
//
//  Created by MNN on 2020/02/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"

using namespace MNN::Express;

class LinSpaceTest : public MNNTestCase {
public:
    virtual ~LinSpaceTest() = default;
    virtual bool run(int precision) {
        auto start                              = _Scalar<float>(0.1);
        auto end                                = _Scalar<float>(20);
        auto steps                              = _Scalar<int>(10);
        auto output                             = _LinSpace(start, end, steps);
        const std::vector<float> expectedOutput = {0.1,       2.311111,  4.522222,  6.733333,  8.944445,
                                                   11.155556, 13.366667, 15.577778, 17.788889, 20.};
        auto gotOutput                          = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 10, 0.01)) {
            MNN_ERROR("LinSpaceTest test failed!\n");
            return false;
        }
        return true;
    }
};
MNNTestSuiteRegister(LinSpaceTest, "op/linspace");
