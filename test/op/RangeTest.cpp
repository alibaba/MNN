//
//  RangeTest.cpp
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
class RangeTest : public MNNTestCase {
public:
    virtual ~RangeTest() = default;
    virtual bool run(int precision) {
        // Float
        {
            auto start                              = _Const(0.0);
            auto limit                              = _Const(1.0);
            auto delta                              = _Const(0.3);
            auto output                             = _Range(start, limit, delta);
            const std::vector<float> expectedOutput = {0.0f, 0.3f, 0.6f, 0.9f};
            auto gotOutput                          = output->readMap<float>();
            if (!checkVector<float>(gotOutput, expectedOutput.data(), 4, 0.01)) {
                MNN_ERROR("RangeTest test failed!\n");
                return false;
            }
            auto dims = output->getInfo()->dim;
            if (dims.size() != 1) {
                MNN_ERROR("RangeTest test failed!\n");
                return false;
            }
            if (dims[0] != 4) {
                MNN_ERROR("RangeTest test failed!\n");
                return false;
            }
        }
        // Int
        {
            auto start                              = _Scalar<int>(0);
            auto limit                              = _Scalar<int>(10);
            auto delta                              = _Scalar<int>(2);
            auto output                             = _Range(start, limit, delta);
            const std::vector<int> expectedOutput = {0, 2, 4, 6, 8};
            auto gotOutput                          = output->readMap<int>();
            if (!checkVector<int>(gotOutput, expectedOutput.data(), 5, 0)) {
                MNN_ERROR("RangeTest test failed!\n");
                return false;
            }
            auto dims = output->getInfo()->dim;
            if (dims.size() != 1) {
                MNN_ERROR("RangeTest test failed!\n");
                return false;
            }
            if (dims[0] != 5) {
                MNN_ERROR("RangeTest test failed!\n");
                return false;
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(RangeTest, "op/range");
