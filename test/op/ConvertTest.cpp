//
//  ConvertTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/01/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"

using namespace MNN::Express;
class ConvertTest : public MNNTestCase {
public:
    virtual ~ConvertTest() = default;
    virtual bool run(int precision) {
        auto input = _Input({4, 1, 1, 3}, NHWC);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 12 * sizeof(float));
        input->unMap();
        auto output                             = _Convert(input, NC4HW4);
        const std::vector<float> expectedOutput = {1.0f, 2.0f, 3.0f, 0.0f, 4.0f,  5.0f,  6.0f,  0.0f,
                                                   7.0f, 8.0f, 9.0f, 0.0f, 10.0f, 11.0f, 12.0f, 0.0f};
        {
            auto gotOutput = output->readMap<float>();
            if (!checkVector<float>(gotOutput, expectedOutput.data(), 16, 0.01)) {
                MNN_ERROR("ConvertTest test failed!\n");
                for (int i = 0; i < 16; ++i) {
                    MNN_PRINT("Correct: %f - Compute: %f\n", expectedOutput[i], gotOutput[i]);
                }
                return false;
            }
        }
        output = _Convert(output, NHWC);
        {
            auto gotOutput = output->readMap<float>();
            if (!checkVector<float>(gotOutput, inpudata, 12, 0.01)) {
                MNN_ERROR("ConvertTest test failed!\n");
                for (int i = 0; i < 12; ++i) {
                    MNN_PRINT("Correct: %f - Compute: %f\n", inpudata[i], gotOutput[i]);
                }
                return false;
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(ConvertTest, "op/convert");
