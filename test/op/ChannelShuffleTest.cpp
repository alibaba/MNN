//
//  ChannelShuffleTest.cpp
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
class ChannelShuffleTest : public MNNTestCase {
public:
    virtual ~ChannelShuffleTest() = default;
    virtual bool run(int precision) {
        auto input = _Input({1, 1, 2, 4}, NHWC);
        // set input data
        const float inpudata[] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 8 * sizeof(float));
        auto output = _ChannelShuffle(input, 2);
        output = _Convert(output, NHWC);
        const std::vector<float> expectedOutput = {0.0, 2.0, 1.0, 3.0, 4.0, 6.0, 5.0, 7.0};
        auto gotOutput                        = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 8, 1e-5)) {
            MNN_ERROR("ChannelShuffleTest test failed!\n");
            return false;
        }
        return true;
    }
};
MNNTestSuiteRegister(ChannelShuffleTest, "op/channel_shuffle");
