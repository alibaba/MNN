//
//  ScatterElementsTest.cpp
//  MNNTests
//
//  Created by MNN on 2022/06/29.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"

using namespace MNN::Express;

class ScatterElementsTest : public MNNTestCase {
    virtual ~ScatterElementsTest() = default;

    virtual bool run(int precision) {
        {
            const float dataData[]    = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            const int indicesData[]      = {1, 0, 2, 0, 2, 1};
            const float updatesData[]    = {1.0, 1.1, 1.2, 2.0, 2.1, 2.2};
            const float expectedData[] = {2.0, 1.1, 0.0, 1.0, 0.0, 2.2, 0.0, 2.1, 1.2};
            {
                auto data    = _Const(dataData, {3, 3}, NHWC, halide_type_of<float>());
                auto indices = _Const(indicesData, {2, 3}, NHWC, halide_type_of<int>());
                auto updates = _Const(updatesData, {2, 3}, NHWC, halide_type_of<float>());
                auto output  = _ScatterElements(data, indices, updates);

                auto outputData = output->readMap<float>();
                const int size  = output->getInfo()->size;
                if (!checkVector<float>(outputData, expectedData, size, 0.001)) {
                    FUNC_PRINT(1);
                    return false;
                }
            }
            {
                auto data    = _Const(dataData, {3, 3}, NHWC, halide_type_of<float>());
                auto indices = _Const(nullptr, {0, 3}, NHWC, halide_type_of<int>());
                auto updates = _Const(updatesData, {2, 3}, NHWC, halide_type_of<float>());
                auto output  = _ScatterElements(data, indices, updates);

                auto outputData = output->readMap<float>();
                const int size  = output->getInfo()->size;
                if (!checkVector<float>(outputData, dataData, size, 0.001)) {
                    FUNC_PRINT(1);
                    return false;
                }
            }
        }
        {
            const float dataData[]    = {1.0, 2.0, 3.0, 4.0, 5.0};
            const int indicesData[]      = {1, 3};
            const float updatesData[]    = {1.1, 2.1};
            const float expectedData[] = {1.0, 1.1, 3.0, 2.1, 5.0};

            auto data    = _Const(dataData, {1, 5}, NHWC, halide_type_of<float>());
            auto indices = _Const(indicesData, {1, 2}, NHWC, halide_type_of<int>());
            auto updates = _Const(updatesData, {1, 2}, NHWC, halide_type_of<float>());
            auto output  = _ScatterElements(data, indices, updates, _Scalar(1));

            auto outputData = output->readMap<float>();
            const int size  = output->getInfo()->size;
            if (!checkVector<float>(outputData, expectedData, size, 0.001)) {
                return false;
            }
        }
        {
            const float dataData[]    = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
            const int indicesData[]      = {0, 1, 2, 0, 1, 4};
            const float updatesData[]    = {1, 2, 3, 6, 7, 8};
            const float expectedData[] = {1, 2, 3, 0, 0, 6, 7, 0, 0, 8, 0, 0, 0, 0, 0};

            auto data    = _Const(dataData, {3, 5}, NHWC, halide_type_of<float>());
            auto indices = _Const(indicesData, {2, 3}, NHWC, halide_type_of<int>());
            auto updates = _Const(updatesData, {2, 3}, NHWC, halide_type_of<float>());
            auto output  = _ScatterElements(data, indices, updates, _Scalar(1));

            auto outputData = output->readMap<float>();
            const int size  = output->getInfo()->size;
            if (!checkVector<float>(outputData, expectedData, size, 0.001)) {
                return false;
            }
        }
        {
            const float dataData[]    = {1.0, 2.0, 3.0, 4.0, 5.0};
            const int indicesData[]      = {1, 1};
            const float updatesData[]    = {1.1, 2.1};
            const float expectedData[] = {1.0, 5.2, 3.0, 4.0, 5.0};

            auto data    = _Const(dataData, {1, 5}, NHWC, halide_type_of<float>());
            auto indices = _Const(indicesData, {1, 2}, NHWC, halide_type_of<int>());
            auto updates = _Const(updatesData, {1, 2}, NHWC, halide_type_of<float>());
            auto output  = _ScatterElements(data, indices, updates, _Scalar(1), MNN::BinaryOpOperation_ADD);

            auto outputData = output->readMap<float>();
            const int size  = output->getInfo()->size;
            if (!checkVector<float>(outputData, expectedData, size, 0.001)) {
                return false;
            }
        }
        {
            const float dataData[]    = {1.0, 2.0, 3.0, 4.0, 5.0};
            const int indicesData[]      = {1, 1};
            const float updatesData[]    = {1.1, 2.1};
            const float expectedData[] = {1.0, 4.62, 3.0, 4.0, 5.0};

            auto data    = _Const(dataData, {1, 5}, NHWC, halide_type_of<float>());
            auto indices = _Const(indicesData, {1, 2}, NHWC, halide_type_of<int>());
            auto updates = _Const(updatesData, {1, 2}, NHWC, halide_type_of<float>());
            auto output  = _ScatterElements(data, indices, updates, _Scalar(1), MNN::BinaryOpOperation_MUL);

            auto outputData = output->readMap<float>();
            const int size  = output->getInfo()->size;
            if (!checkVector<float>(outputData, expectedData, size, 0.001)) {
                return false;
            }
        }
        return true;
    }
};

MNNTestSuiteRegister(ScatterElementsTest, "op/ScatterElementsTest");
