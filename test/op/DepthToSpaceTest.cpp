//
//  DepthToSpaceTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/12/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//


#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"
using namespace MNN::Express;
class DepthToSpaceTest : public MNNTestCase {
public:
    virtual ~DepthToSpaceTest() = default;
    virtual bool run() {
        auto input = _Input({4, 1, 1, 4}, NHWC);
        input->setName("input");
        // set input data
        const float input_data[] = {-1.0, 2.0, -3.0, 4.0,
                                  5.0, 6.0, 7.0, -8.0,
                                  -9.0, -10.0, 11.0, 12.0,
                                  13.0, 14.0, -15.0, -16.0};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, input_data, 16 * sizeof(float));
        input->unMap();
        auto output = _DepthToSpace(input, 2);
        const std::vector<float> expectedOutput = {-1.0, 2.0, -3.0, 4.0,
                                  5.0, 6.0, 7.0, -8.0,
                                  -9.0, -10.0, 11.0, 12.0,
                                  13.0, 14.0, -15.0, -16.0};
	const std::vector<int> expectedDim = {4, 2, 2, 1};
        auto gotOutput = output->readMap<float>();
        auto gotDim = output->getInfo()->dim;
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 16, 0)) {
            MNN_ERROR("DepthToSpaceTest test failed!\n");
            return false;
        } 
        if (!checkVector<int>(gotDim.data(), expectedDim.data(), 4, 0)) {
            MNN_ERROR("DepthToSpaceTest test failed!\n");
            return false;
        }
        return true;
    }
};
MNNTestSuiteRegister(DepthToSpaceTest, "op/depthtospace");

