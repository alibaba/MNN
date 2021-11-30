//
//  RandomUniformTest.cpp
//  MNNTests
//
//  Created by MNN on 2021/10/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"
using namespace MNN::Express;
class RandomUniformTest : public MNNTestCase {
public:
    virtual ~RandomUniformTest() = default;
    virtual bool run(int precision) {
        std::vector<int> shapeValue = {2, 4};
        auto input = _Const(shapeValue.data(), {2}, NHWC, halide_type_of<int>());
        auto random0 = _RandomUnifom(input, halide_type_of<float>(), -1.0f, 1.0f);
        auto random1 = _RandomUnifom(input, halide_type_of<float>(), 0.0f, 1.0f);
        auto size = random0->getInfo()->size;
        auto p0 = random0->readMap<float>();
        for (int i=0; i<size; ++i) {
            if (p0[i] < -1.0f || p0[i] > 1.0f) {
                FUNC_PRINT(1);
                return false;
            }
        }
        auto p1 = random1->readMap<float>();
        for (int i=0; i<size; ++i) {
            if (p1[i] < 0.0f || p1[i] > 1.0f) {
                FUNC_PRINT(1);
                return false;
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(RandomUniformTest, "op/randomuniform");
