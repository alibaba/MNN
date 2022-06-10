//
//  WhereTest.cpp
//  MNNTests
//
//  Created by MNN on 2021/11/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"

using namespace MNN::Express;


class WhereTest : public MNNTestCase {
public:
    virtual ~WhereTest() = default;
    virtual bool run(int precision) {

        return commonCase() &&
               zeroCase();

    }
    bool commonCase() {
        auto input = _Input({2, 3}, NCHW);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = { 1.0, 0.0, 2.0,
                                   3.0, 0.0, 4.0 };
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 6 * sizeof(float));
        input->unMap();
        auto output                           = _Where(input);
        const std::vector<int> expectedOutput = {0, 0, 0, 2, 1, 0, 1, 2};
        const std::vector<int> expectedShape = {4, 2};

        auto realShape = output->getInfo()->dim;
        if (!checkVector<int>(realShape.data(), expectedShape.data(), 2, 0)) {
            MNN_ERROR("WhereTest shape mismatch!\n");
            return false;
        }

        auto gotOutput                        = output->readMap<int>();
        if (!checkVector<int>(gotOutput, expectedOutput.data(), 8, 0)) {
            MNN_ERROR("WhereTest test failed!\n");
            return false;
        }
        return true;
    }

    bool zeroCase() {

        auto input = _Input({2, 4}, NCHW);
        input->setName("input_tensor");
        // set input data
        const float inpudata_zero[] = { 0.0, 0.0, 0.0, 0.0,
                                        0.0, 0.0, 0.0, 0.0 };
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata_zero, 8 * sizeof(float));
        input->unMap();
        auto output                           = _Where(input);
        const std::vector<int> expectedOutput = {};
        const std::vector<int> expectedShape = {0, 2};

        auto realShape = output->getInfo()->dim;
        if (!checkVector<int>(realShape.data(), expectedShape.data(), 2, 0)) {
            MNN_ERROR("WhereTest zero shape mismatch!\n");
            return false;
        }
        return true;
    }
};
MNNTestSuiteRegister(WhereTest, "op/where");
