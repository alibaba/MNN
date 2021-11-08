//
//  OneHotTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/11/29.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"

using namespace MNN::Express;

class OneHotTest : public MNNTestCase {
    virtual ~OneHotTest() = default;

    virtual bool run(int precision) {
        {
            const int indicesData[]    = {0, 1, 2};
            const int depthData[]      = {3};
            const float onValueData[]  = {1.0};
            const float offValueData[] = {0.0};
            const int axis             = -1;

            const float expectedValue[] = {
                1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
            };

            auto indices  = _Const(indicesData, {3}, NHWC, halide_type_of<int>());
            auto depth    = _Const(depthData, {1}, NHWC, halide_type_of<int>());
            auto onValue  = _Const(onValueData, {1}, NHWC, halide_type_of<float>());
            auto offValue = _Const(offValueData, {1}, NHWC, halide_type_of<float>());

            auto result     = _OneHot(indices, depth, onValue, offValue, axis);
            auto resultdata = result->readMap<float>();
            const int size  = result->getInfo()->size;
            if (!checkVector<float>(resultdata, expectedValue, size, 0.0)) {
                return false;
            }
        }

        {
            const int indicesData[]    = {0, 2, -1, 1};
            const int depthData[]      = {3};
            const float onValueData[]  = {5.0};
            const float offValueData[] = {0.0};
            const int axis             = -1;

            const float expectedValue[] = {
                5.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0,
            };

            auto indices  = _Const(indicesData, {4}, NHWC, halide_type_of<int>());
            auto depth    = _Const(depthData, {1}, NHWC, halide_type_of<int>());
            auto onValue  = _Const(onValueData, {1}, NHWC, halide_type_of<float>());
            auto offValue = _Const(offValueData, {1}, NHWC, halide_type_of<float>());

            auto result     = _OneHot(indices, depth, onValue, offValue, axis);
            auto resultdata = result->readMap<float>();
            const int size  = result->getInfo()->size;
            if (!checkVector<float>(resultdata, expectedValue, size, 0.0)) {
                return false;
            }
        }

        {
            const int indicesData[]    = {0, 2, 1, -1};
            const int depthData[]      = {3};
            const float onValueData[]  = {1.0};
            const float offValueData[] = {0.0};
            const int axis             = -1;

            const float expectedValue[] = {1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0};

            auto indices  = _Const(indicesData, {2, 2}, NHWC, halide_type_of<int>());
            auto depth    = _Const(depthData, {1}, NHWC, halide_type_of<int>());
            auto onValue  = _Const(onValueData, {1}, NHWC, halide_type_of<float>());
            auto offValue = _Const(offValueData, {1}, NHWC, halide_type_of<float>());

            auto result     = _OneHot(indices, depth, onValue, offValue, axis);
            auto resultdata = result->readMap<float>();
            const int size  = result->getInfo()->size;
            if (!checkVector<float>(resultdata, expectedValue, size, 0.0)) {
                return false;
            }
        }

        return true;
    }
};

MNNTestSuiteRegister(OneHotTest, "op/OneHotTest");
