//
//  MomentsTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/12/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"

using namespace MNN::Express;
class MomentsTest : public MNNTestCase {
public:
    virtual ~MomentsTest() = default;
    virtual bool run(int precision) {
        auto input = _Input({1, 4, 4, 1}, NCHW);
        input->setName("input_tensor");
        // set input data
        const float inpudata[16] = {0.0,  1.0,  2.0, 3.0, -1.0, 0.0,  1.0,  2.0,
                                    -2.0, -1.0, 0.0, 1.0, -3.0, -2.0, -1.0, 0.0};
        auto inputPtr            = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 16 * sizeof(float));
        input->unMap();
        input                                 = _Convert(input, NC4HW4);
        auto notused_var                      = _Const(1.0);
        auto outputs                          = _Moments(input, {2, 3}, notused_var, true);
        const std::vector<float> expectedMean = {1.5, 0.5, -0.5, -1.5};
        const std::vector<float> expectedVar  = {1.25, 1.25, 1.25, 1.25};
        auto gotOutputMean                    = outputs[0]->readMap<float>();
        auto gotOutputVar                     = outputs[1]->readMap<float>();
        if (!checkVector<float>(gotOutputMean, expectedMean.data(), 4, 0.01)) {
            MNN_ERROR("MomentsTest test failed!\n");
            return false;
        }
        if (!checkVector<float>(gotOutputVar, expectedVar.data(), 4, 0.01)) {
            MNN_ERROR("MomentsTest test failed!\n");
            return false;
        }
        return true;
    }
};
MNNTestSuiteRegister(MomentsTest, "op/moments");
