//
//  SoftmaxTest.cpp
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
class SoftmaxTest : public MNNTestCase {
public:
    virtual ~SoftmaxTest() = default;
    virtual bool run(int precision) {
        // testcase 0
        {
            auto input = _Input({1, 4}, NCHW);
            input->setName("input_tensor");
            // set input data
            const float inpudata[] = {1.0, 2.0, 3.0, 4.0};
            auto inputPtr          = input->writeMap<float>();
            memcpy(inputPtr, inpudata, 4 * sizeof(float));
            input->unMap();
            auto output                             = _Softmax(input);
            const std::vector<float> expectedOutput = {0.0320586, 0.0871443, 0.236883, 0.643914};
            auto gotOutput                          = output->readMap<float>();
            if (!checkVector<float>(gotOutput, expectedOutput.data(), 4, 0.001)) {
                MNN_ERROR("SoftmaxTest0 test failed!\n");
                return false;
            }
        }
        // testcase 1
        {
            auto input = _Input({2, 4}, NCHW);
            input->setName("input_tensor");
            // set input data
            const float inpudata[] = {1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0};
            auto inputPtr          = input->writeMap<float>();
            memcpy(inputPtr, inpudata, 8 * sizeof(float));
            input->unMap();
            auto output                             = _Softmax(input);
            const std::vector<float> expectedOutput = {0.0320586, 0.0871443, 0.236883,  0.643914,
                                                       0.643914,  0.236883,  0.0871443, 0.0320586};
            auto gotOutput                          = output->readMap<float>();
            if (!checkVector<float>(gotOutput, expectedOutput.data(), 8, 0.001)) {
                MNN_ERROR("SoftmaxTest1 test failed!\n");
                return false;
            }
        }
        // testcase 2
        {
            auto input = _Input({2, 5}, NCHW);
            input->setName("input_tensor");
            // set input data
            const float inpudata[] = {1.0, 2.0, 3.0, 4.0, 5.0, -1.0, -2.0, -3.0, -4.0, -5.0};
            auto inputPtr          = input->writeMap<float>();
            memcpy(inputPtr, inpudata, 10 * sizeof(float));
            input->unMap();
            auto output                             = _Softmax(input);
            const std::vector<float> expectedOutput = {0.0116558, 0.0316853, 0.0861187, 0.234124,  0.636416,
                                                       0.636416,  0.234124,  0.0861187, 0.0316853, 0.0116558};
            auto gotOutput                          = output->readMap<float>();
            if (!checkVector<float>(gotOutput, expectedOutput.data(), 10, 0.001)) {
                MNN_ERROR("SoftmaxTest2 test failed!\n");
                return false;
            }
        }
        // testcase 3
        {
            auto input = _Input({2, 2}, NCHW);
            input->setName("input_tensor");
            // set input data
            const float inpudata[] = {-1.0, -2.0, 3.0, 4.0};
            auto inputPtr          = input->writeMap<float>();
            memcpy(inputPtr, inpudata, 4 * sizeof(float));
            input->unMap();
            auto output                             = _Softmax(input);
            const std::vector<float> expectedOutput = {0.7310586, 0.26894143, 0.26894143, 0.7310586};
            auto gotOutput                          = output->readMap<float>();
            if (!checkVector<float>(gotOutput, expectedOutput.data(), 4, 0.001)) {
                MNN_ERROR("SoftmaxTest test failed!\n");
                return false;
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(SoftmaxTest, "op/softmax");
