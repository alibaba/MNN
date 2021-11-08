//
//  RankTest.cpp
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
class RankTest : public MNNTestCase {
public:
    virtual ~RankTest() = default;
    virtual bool run(int precision) {
        auto input = _Input({2, 2}, NCHW);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {-1.0, -2.0, 3.0, 4.0};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 4 * sizeof(float));
        input->unMap();
        auto output                           = _Rank(input);
        const std::vector<int> expectedOutput = {2};
        auto gotOutput                        = output->readMap<int>();
        if (!checkVector<int>(gotOutput, expectedOutput.data(), 1, 0)) {
            MNN_ERROR("RankTest test failed!\n");
            return false;
        }
        auto dims = output->getInfo()->dim;
        if (dims.size() != 0) {
            MNN_ERROR("RankTest test failed!\n");
            return false;
        }
        return true;
    }
};
MNNTestSuiteRegister(RankTest, "op/rank");
