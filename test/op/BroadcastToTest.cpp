//
//  BroadcastToTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/12/3.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"

using namespace MNN::Express;

class BroadcastToTest : public MNNTestCase {
    virtual ~BroadcastToTest() = default;

    virtual bool run() {
        {
            const float tensorData[]   = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
            const int shapeData[]      = {2, 3, 2, 2};
            const float expectedData[] = {
                1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 6.0, 5.0, 6.0,
                1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 6.0, 5.0, 6.0,
            };

            auto tensor = _Const(tensorData, {1, 3, 1, 2}, NHWC, halide_type_of<float>());
            auto shape  = _Const(shapeData, {4}, NHWC, halide_type_of<int>());
            auto result = _BroadcastTo(tensor, shape);

            const int size  = result->getInfo()->size;
            auto resultData = result->readMap<float>();
            if (!checkVector<float>(resultData, expectedData, size, 0.0)) {
                return false;
            }
        }

        {
            const float tensorData[]   = {1.0, 2.0, 3.0};
            const int shapeData[]      = {3, 3};
            const float expectedData[] = {1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0};

            auto tensor = _Const(tensorData, {1, 3}, NHWC, halide_type_of<float>());
            auto shape  = _Const(shapeData, {2}, NHWC, halide_type_of<int>());
            auto result = _BroadcastTo(tensor, shape);

            const int size  = result->getInfo()->size;
            auto resultData = result->readMap<float>();
            if (!checkVector<float>(resultData, expectedData, size, 0.0)) {
                return false;
            }
        }

        {
            const float tensorData[]   = {1.0, 2.0, 3.0};
            const int shapeData[]      = {3, 3};
            const float expectedData[] = {1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0};

            auto tensor = _Const(tensorData, {3, 1}, NHWC, halide_type_of<float>());
            auto shape  = _Const(shapeData, {2}, NHWC, halide_type_of<int>());
            auto result = _BroadcastTo(tensor, shape);

            const int size  = result->getInfo()->size;
            auto resultData = result->readMap<float>();
            if (!checkVector<float>(resultData, expectedData, size, 0.0)) {
                return false;
            }
        }

        {
            const float tensorData[]   = {1.0, 2.0};
            const int shapeData[]      = {2, 3, 2, 2};
            const float expectedData[] = {1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0,
                                          1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0};

            auto tensor = _Const(tensorData, {1, 1, 1, 2}, NHWC, halide_type_of<float>());
            auto shape  = _Const(shapeData, {4}, NHWC, halide_type_of<int>());
            auto result = _BroadcastTo(tensor, shape);

            const int size  = result->getInfo()->size;
            auto resultData = result->readMap<float>();
            if (!checkVector<float>(resultData, expectedData, size, 0.0)) {
                return false;
            }
        }

        return true;
    }
};

MNNTestSuiteRegister(BroadcastToTest, "op/BroadcastToTest");
