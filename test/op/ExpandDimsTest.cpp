//
//  ExpandDimsTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/12/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"

using namespace MNN::Express;
class ExpandDimsTest : public MNNTestCase {
public:
    virtual ~ExpandDimsTest() = default;
    virtual bool run(int precision) {
        auto input = _Input({4}, NCHW);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {-1.0, -2.0, 3.0, 4.0};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 4 * sizeof(float));
        input->unMap();
        auto output                             = _ExpandDims(input, 0);
        const std::vector<float> expectedOutput = {-1.0, -2.0, 3.0, 4.0};
        auto gotOutput                          = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 4, 0.0001)) {
            MNN_ERROR("ExpandDimsTest test failed!\n");
            return false;
        }
        const std::vector<int> expectedDim = {1, 4};
        auto gotDim                        = output->getInfo()->dim;
        if (!checkVector<int>(gotDim.data(), expectedDim.data(), 2, 0)) {
            MNN_ERROR("ExpandDimsTest test failed!\n");
            return false;
        }
        return true;
    }
};
MNNTestSuiteRegister(ExpandDimsTest, "op/expand_dims");
