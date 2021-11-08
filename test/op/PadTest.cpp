//
//  PadTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/12/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"
using namespace MNN::Express;
class PadTest : public MNNTestCase {
public:
    virtual ~PadTest() = default;
    virtual bool run(int precision) {
        auto input = _Input({1, 2, 2, 1}, NCHW);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {-1.0, -2.0, 3.0, 4.0};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 4 * sizeof(float));
        input->unMap();
        const int paddings_data[]               = {0, 0, 1, 1, 1, 1, 0, 0};
        auto paddings                           = _Const(paddings_data, {4, 2}, NCHW, halide_type_of<int>());
        auto output                             = _Pad(input, paddings);
        const std::vector<float> expectedOutput = {0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -2.0, 0.0,
                                                   0.0, 3.0, 4.0, 0.0, 0.0, 0.0,  0.0,  0.0};
        auto gotOutput                          = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 16, 0.01)) {
            MNN_ERROR("PadTest test failed!\n");
            return false;
        }
        return true;
    }
};
MNNTestSuiteRegister(PadTest, "op/pad");
