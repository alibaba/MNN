//
//  TanHTest.cpp
//  MNNTests
//
//  Created by MNN on 2020/07/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"

using namespace MNN::Express;
class TanHTest : public MNNTestCase {
public:
    virtual ~TanHTest() = default;
    virtual bool run(int precision) {
        auto input = _Input({5}, NCHW);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {-7.8, -2.0, 0.0, 1.0, 999.0};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 5 * sizeof(float));
        input->unMap();
        auto output                             = _Tanh(input);
        const std::vector<float> expectedOutput = {-1.0, -0.96, 0.0, 0.76, 1.0};
        auto gotOutput                          = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 5, 0.01)) {
            MNN_ERROR("EluTest test failed!\n");
            return false;
        }
        return true;
    }
};
MNNTestSuiteRegister(TanHTest, "op/tanh");
