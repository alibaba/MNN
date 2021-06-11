//
//  ScatterNdTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/11/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"

using namespace MNN::Express;

class ScatterNdTest : public MNNTestCase {
    virtual ~ScatterNdTest() = default;

    virtual bool run(int precision) {
        {
            const int indicesData[]      = {4, 3, 1, 7};
            const float updatesData[]    = {9, 10, 11, 12};
            const int shapeData[]        = {8};
            const float expectedResult[] = {0, 11, 0, 10, 9, 0, 0, 12};

            auto indices = _Const(indicesData, {4, 1}, NHWC, halide_type_of<int>());
            auto updates = _Const(updatesData, {4}, NHWC, halide_type_of<float>());
            auto shape   = _Const(shapeData, {1}, NHWC, halide_type_of<float>());
            auto result  = _ScatterNd(indices, updates, shape);

            auto resultData = result->readMap<float>();
            const int size  = result->getInfo()->size;
            if (!checkVector<float>(resultData, expectedResult, size, 0.001)) {
                return false;
            }
        }

        {
            const int indicesData[]      = {0, 2};
            const float updatesData[]    = {5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8,
                                         5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8};
            const int shapeData[]        = {4, 4, 4};
            auto indices                 = _Const(indicesData, {2, 1}, NHWC, halide_type_of<int>());
            auto updates                 = _Const(updatesData, {2, 4, 4}, NHWC, halide_type_of<float>());
            auto shape                   = _Const(shapeData, {3}, NHWC, halide_type_of<int>());
            const float expectedResult[] = {5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7,
                                            8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
            auto result                  = _ScatterNd(indices, updates, shape);

            auto resultData = result->readMap<float>();
            const int size  = result->getInfo()->size;
            if (!checkVector<float>(resultData, expectedResult, size, 0.001)) {
                return false;
            }
        }

        return true;
    }
};

MNNTestSuiteRegister(ScatterNdTest, "op/ScatterNdTest");
