//
//  SoftmaxSpeed.cpp
//  MNNTests
//
//  Created by MNN on 2019/09/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <math.h>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <random>
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include "MNNTestSuite.h"
using namespace MNN::Express;
#define TIME 100
class SoftmaxSpeed : public MNNTestCase {
public:
    void SoftmaxTest() {
        std::vector<std::tuple<int, int, int>> oci = {
            {4096, 1024, 1},
            {4096, 2, 1024},
            {4096, 32, 32},
        };
        for (auto& iter : oci) {
            auto outside = std::get<0>(iter);
            auto axis = std::get<1>(iter);
            auto inside = std::get<2>(iter);
            auto x      = _Input({outside, axis, inside}, NCHW);
            auto output = _Softmax(x, 1);
            MNN::Timer _t;
            for (int i = 0; i < TIME; ++i) {
                x->writeMap<float>();
                output->readMap<float>();
            }
            float cost = (float)_t.durationInUs()/1000.0f / (float)TIME;
            MNN_PRINT("Test Speed for softmax outside:%d, axis:%d, inside:%d, run %d, avgtime: %f ms\n", outside, axis, inside, TIME, cost);

        }
    }
    virtual bool run(int precision) {
        SoftmaxTest();
        return true;
    }
};
MNNTestSuiteRegister(SoftmaxSpeed, "speed/Softmax");
