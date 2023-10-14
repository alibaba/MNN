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
            const float dataData[]       = {1, 2, 3, 4, 5, 6, 7, 8};
            const float expectedResult[] = {1, 11, 3, 10, 9, 6, 7, 12};

            auto indices = _Const(indicesData, {0, 1}, NHWC, halide_type_of<int>());
            auto updates = _Const(nullptr, {0}, NHWC, halide_type_of<float>());
            auto shape   = _Const(shapeData, {1}, NHWC, halide_type_of<float>());
            auto data    = _Const(dataData, {8}, NHWC, halide_type_of<float>());
            auto result  = _ScatterNd(indices, updates, shape, data);

            auto resultData = result->readMap<float>();
            const int size  = result->getInfo()->size;
            if (!checkVector<float>(resultData, dataData, size, 0.001)) {
                FUNC_PRINT(1);
                return false;
            }
        }
        {
            const int indicesData[]      = {4, 3, 1, 7};
            const float updatesData[]    = {9, 10, 11, 12};
            const int shapeData[]        = {8};
            const float dataData[]       = {1, 2, 3, 4, 5, 6, 7, 8};
            const float expectedResult[] = {1, 11, 3, 10, 9, 6, 7, 12};

            auto indices = _Const(indicesData, {4, 1}, NHWC, halide_type_of<int>());
            auto updates = _Const(updatesData, {4}, NHWC, halide_type_of<float>());
            auto shape   = _Const(shapeData, {1}, NHWC, halide_type_of<float>());
            auto data    = _Const(dataData, {8}, NHWC, halide_type_of<float>());
            auto result  = _ScatterNd(indices, updates, shape, data);

            auto resultData = result->readMap<float>();
            const int size  = result->getInfo()->size;
            if (!checkVector<float>(resultData, expectedResult, size, 0.001)) {
                return false;
            }
        }
        {
            const int indicesData[]      = {2, 0};
            const int updatesData[]    = {33, 55};
            const int shapeData[]        = {8};
            const int dataData[]       = {1, 2, 3, 4, 5, 6, 7, 8};
            const int expectedResult[] = {55, 2, 33, 4, 5, 6, 7, 8};

            auto indices = _Const(indicesData, {2, 1}, NHWC, halide_type_of<int>());
            auto updates = _Const(updatesData, {2}, NHWC, halide_type_of<int>());
            auto shape   = _Const(shapeData, {1}, NHWC, halide_type_of<int>());
            auto data    = _Const(dataData, {8}, NHWC, halide_type_of<int>());
            auto result  = _ScatterNd(indices, updates, shape, data);

            auto resultData = result->readMap<int>();
            const int size  = result->getInfo()->size;
            if (!checkVector<int>(resultData, expectedResult, size, 1)) {
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

        {
            const int indicesData[]      = {0, 2};
            const float updatesData[]    = {5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8,
                                            5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8};
            const float dataData[]       = {1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1,
                                            1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1,
                                            8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8,
                                            8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8};
            const int shapeData[]        = {4, 4, 4};
            auto indices                 = _Const(indicesData, {2, 1}, NHWC, halide_type_of<int>());
            auto updates                 = _Const(updatesData, {2, 4, 4}, NHWC, halide_type_of<float>());
            auto shape                   = _Const(shapeData, {3}, NHWC, halide_type_of<int>());
            auto data                    = _Const(dataData, {4, 4, 4}, NHWC, halide_type_of<float>());

            const float expectedResult[] = {5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8,
                                            1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1,
                                            5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8,
                                            8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8};
            auto result                  = _ScatterNd(indices, updates, shape, data);

            auto resultData = result->readMap<float>();
            const int size  = result->getInfo()->size;
            if (!checkVector<float>(resultData, expectedResult, size, 0.001)) {
                return false;
            }
        }
        {
            const int indicesData[]      = {0, 0};
            const float updatesData[]    = {5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8,
                                            1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4};
            const float dataData[]       = {1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1,
                                            1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1,
                                            8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8,
                                            8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8};
            const int shapeData[]        = {4, 4, 4};
            auto indices                 = _Const(indicesData, {2, 1}, NHWC, halide_type_of<int>());
            auto updates                 = _Const(updatesData, {2, 4, 4}, NHWC, halide_type_of<float>());
            auto shape                   = _Const(shapeData, {3}, NHWC, halide_type_of<int>());
            auto data                    = _Const(dataData, {4, 4, 4}, NHWC, halide_type_of<float>());

            const float expectedResult[] = {7, 8, 9, 10, 13, 14, 15, 16, 18, 17, 16, 15, 16, 15, 14, 13,
                                            1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1,
                                            8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8,
                                            8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8};
            auto result                  = _ScatterNd(indices, updates, shape, data, MNN::BinaryOpOperation_ADD);

            auto resultData = result->readMap<float>();
            const int size  = result->getInfo()->size;
            if (!checkVector<float>(resultData, expectedResult, size, 0.001)) {
                return false;
            }
        }
        {
            const int indicesData[]      = {0, 0};
            const float updatesData[]    = {5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8,
                                            1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4};
            const float dataData[]       = {1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1,
                                            1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1,
                                            8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8,
                                            8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8};
            const int shapeData[]        = {4, 4, 4};
            auto indices                 = _Const(indicesData, {2, 1}, NHWC, halide_type_of<int>());
            auto updates                 = _Const(updatesData, {2, 4, 4}, NHWC, halide_type_of<float>());
            auto shape                   = _Const(shapeData, {3}, NHWC, halide_type_of<int>());
            auto data                    = _Const(dataData, {4, 4, 4}, NHWC, halide_type_of<float>());

            const float expectedResult[] = {5, 10, 15, 20, 60, 72, 84, 96, 168, 147, 126, 105, 128, 96, 64, 32,
                                            1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1,
                                            8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8,
                                            8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8};
            auto result                  = _ScatterNd(indices, updates, shape, data, MNN::BinaryOpOperation_MUL);

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
