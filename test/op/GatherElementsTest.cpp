//
//  GatherElementsTest.cpp
//  MNNTests
//
//  Created by MNN on 2022/07/01.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"
using namespace MNN::Express;
class GatherElementsTest : public MNNTestCase {
public:
    virtual ~GatherElementsTest() = default;
    virtual bool run(int precision) {
        {
            const float dataData[]  = {1, 2, 3, 4, 5, 6, 7, 8, 9};
            const int indicesData[] = {1, 2, 0, 2, 0, 0};
            const float expectedData[] = {4, 8, 3, 7, 2, 3};
            auto data    = _Const(dataData, {3, 3}, NHWC, halide_type_of<float>());
            auto indices = _Const(indicesData, {2, 3}, NHWC, halide_type_of<int>());
            auto output  = _GatherElements(data, indices);
            auto outputData = output->readMap<float>();
            const int size  = output->getInfo()->size;
            if (!checkVector<float>(outputData, expectedData, size, 0.001)) {
                return false;
            }
        }
        {
            const float dataData[]  = {1, 2, 3, 4};
            const int indicesData[] = {0, 0, 1, 0};
            const float expectedData[] = {1, 1, 4, 3};
            auto data    = _Const(dataData, {2, 2}, NHWC, halide_type_of<float>());
            auto indices = _Const(indicesData, {2, 2}, NHWC, halide_type_of<int>());
            auto output  = _GatherElements(data, indices, _Scalar(1));
            auto outputData = output->readMap<float>();
            const int size  = output->getInfo()->size;
            if (!checkVector<float>(outputData, expectedData, size, 0.001)) {
                return false;
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(GatherElementsTest, "op/GatherElements");
