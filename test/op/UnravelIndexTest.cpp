//
//  UnravelIndexTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/11/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"

using namespace MNN::Express;

class UnravelIndexTest : public MNNTestCase {
    virtual ~UnravelIndexTest() = default;

    virtual bool run(int precision) {
        {
            const int indicesData[] = {22, 41, 37};
            const int shapeData[]   = {7, 6};
            auto indices            = _Const(indicesData, {3}, NHWC, halide_type_of<int>());
            auto dims               = _Const(shapeData, {2}, NHWC, halide_type_of<int>());
            auto result             = _UnravelIndex(indices, dims);

            const int expectedData[] = {3, 6, 6, 4, 5, 1};

            auto resultData = result->readMap<int32_t>();
            const int size  = result->getInfo()->size;
            if (!checkVector<int>(resultData, expectedData, size, 0)) {
                return false;
            }
        }
        {
            const int indicesData[] = {1621};
            const int shapeData[]   = {6, 7, 8, 9};
            auto indices            = _Const(indicesData, {1}, NHWC, halide_type_of<int>());
            auto dims               = _Const(shapeData, {4}, NHWC, halide_type_of<int>());
            auto result             = _UnravelIndex(indices, dims);

            const int expectedData[] = {3, 1, 4, 1};

            auto resultData = result->readMap<int32_t>();
            const int size  = result->getInfo()->size;
            if (!checkVector<int>(resultData, expectedData, size, 0)) {
                return false;
            }
        }

        return true;
    }
};

MNNTestSuiteRegister(UnravelIndexTest, "op/UnravelIndexTest");
