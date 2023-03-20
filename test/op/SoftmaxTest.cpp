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

static std::vector<float> naiveSoftmax(const float* input, const int outside, const int axis, const int inside) {
    std::vector<float> output(outside * axis * inside, 0);
    for(int y = 0; y < outside; y++) {
        for(int x = 0; x < inside; x++) {
            const float* src = input + y * axis * inside + x;
            float* dst = (float *)(output.data()) + y * axis * inside + x;
            float maxValue = (float)src[0];
            for (int z=1; z<axis; ++z) {
                maxValue = maxValue > src[z * inside] ? maxValue : src[z * inside];
            }
            float sumValue = 0.0;
            for (int z=0; z<axis; ++z) {
                sumValue = sumValue + exp((float)src[z * inside] - maxValue);
            }
            sumValue = 1.0 / sumValue;
            for (int z=0; z<axis; ++z) {
                dst[z*inside] = (exp((float)src[z * inside] - maxValue) * sumValue);
            }
        }
    }
    return output;
}

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
        // testcase
        if(0)
        {
            const std::vector<int> axis_vec    = {1, 4, 32, 128, 256, 576, 1024};
            const std::vector<int> outside_vec = {1, 32, 1024, 65536};
            const std::vector<int> inside_vec = {1, 4, 7};

            for(int k = 0; k < outside_vec.size(); k++) {
                for(int j = 0; j < axis_vec.size(); j++) {
                    for(int i = 0; i < inside_vec.size(); i++) {
                        int outside = outside_vec[k];
                        int axis = axis_vec[j];
                        int inside = inside_vec[i];
                        auto input = _Input({outside, axis, inside}, NCHW);
                        // set input data
                        auto total = outside * axis * inside;
                        std::vector<float> inpudata(total);
                        for(int i = 0; i < total; i++) {
                            inpudata[i] = 1.0 * i / total;
                        }
                        auto inputPtr          = input->writeMap<float>();
                        memcpy(inputPtr, inpudata.data(), total * sizeof(float));
                        input->unMap();
                        auto output                             = _Softmax(input, 1);

                        std::vector<float> expectedOutput = naiveSoftmax((float *)inpudata.data(), outside, axis, inside);
                        auto gotOutput                          = output->readMap<float>();
                        int count = 0;
                        for (int i = 0; i < expectedOutput.size(); ++i) {
                            if(gotOutput[i] - expectedOutput[i] > 0.0001 || gotOutput[i] - expectedOutput[i] < -0.0001) {
                                count++;
                            }
                        }
                        if (!checkVector<float>(gotOutput, expectedOutput.data(), total, 0.0001)) {
                            MNN_ERROR("SoftmaxTest test failed! error count:%d\n", count);
                            return false;
                        }
                    }
                }
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(SoftmaxTest, "op/softmax");
