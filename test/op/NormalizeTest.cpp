//
//  NormalizeTest.cpp
//  MNNTests
//
//  Created by MNN on 2021/10/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"

using namespace MNN::Express;
class NormalizeTest : public MNNTestCase {
public:
    virtual ~NormalizeTest() = default;
    virtual bool run(int precision) {
        auto input = _Input({1, 2, 2, 1}, NCHW);
        // set input data
        const float inpudata[] = {-1.0, -2.0, 3.0, 4.0};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 4 * sizeof(float));
        input = _Convert(input, NC4HW4);
        auto output = _Normalize(input, 0, 0, 0.00, {0.5, 0.5});
        output = _Convert(output, NCHW);
        const std::vector<float> expectedOutput = {-0.223607, -0.447214, 0.300000, 0.400000};
        auto gotOutput                        = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 4, 1e-5)) {
            MNN_ERROR("NormalizeTest test failed!\n");
            return false;
        }
        return true;
    }
};
MNNTestSuiteRegister(NormalizeTest, "op/normalize");
