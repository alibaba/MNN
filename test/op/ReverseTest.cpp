//
//  ReverseTest.cpp
//  MNNTests
//
//  Created by MNN on 2021/02/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"

using namespace MNN::Express;
class ReverseTest : public MNNTestCase {
public:
    virtual ~ReverseTest() = default;
    virtual bool run(int precision) {
        {
            auto input = _Input({3, 2, 3}, NCHW);
            input->setName("input_tensor");
            // set input data
            const float inpudata[] = { 1,  2,  3,  4,  5,  6,
                                    7,  8,  9,  10, 11, 12,
                                    13, 14, 15, 16, 17, 18 };
            auto inputPtr          = input->writeMap<float>();
            memcpy(inputPtr, inpudata, 18 * sizeof(float));
            auto output0                             = _Reverse(input, _Scalar<int32_t>(0));
            const std::vector<float> expectedOutput0 = { 13, 14, 15, 16, 17, 18,
                                                        7,  8,  9,  10, 11, 12,
                                                        1,  2,  3,  4,  5,  6 };
            auto gotOutput0                          = output0->readMap<float>();
            for (int i = 0; i < 18; ++i) {
                auto diff = ::fabsf(gotOutput0[i] - expectedOutput0[i]);
                if (diff > 0.01) {
                    MNN_ERROR("ReverseTest[axis=0] test failed: %f - %f!\n", expectedOutput0[i], gotOutput0[i]);
                    return false;
                }
            }
            auto output1                             = _Reverse(input, _Scalar<int32_t>(1));
            const std::vector<float> expectedOutput1 = { 4,  5,  6,  1,  2,  3,
                                                        10, 11, 12, 7,  8,  9,
                                                        16, 17, 18, 13, 14, 15 };
            auto gotOutput1                          = output1->readMap<float>();
            for (int i = 0; i < 18; ++i) {
                auto diff = ::fabsf(gotOutput1[i] - expectedOutput1[i]);
                if (diff > 0.01) {
                    MNN_ERROR("ReverseTest[axis=1] test failed: %f - %f!\n", expectedOutput1[i], gotOutput1[i]);
                    return false;
                }
            }
            auto output2                             = _Reverse(input, _Scalar<int32_t>(2));
            const std::vector<float> expectedOutput2 = { 3,  2,  1,  6,  5,  4,
                                                        9,  8,  7,  12, 11, 10,
                                                        15, 14, 13, 18, 17, 16 };
            auto gotOutput2                          = output2->readMap<float>();
            for (int i = 0; i < 18; ++i) {
                auto diff = ::fabsf(gotOutput2[i] - expectedOutput2[i]);
                if (diff > 0.01) {
                    MNN_ERROR("ReverseTest[axis=2] test failed: %f - %f!\n", expectedOutput2[i], gotOutput2[i]);
                    return false;
                }
            }
        }

        {
            auto input = _Input({2, 2, 2, 2}, NCHW, halide_type_of<uint8_t>());
            input->setName("input_tensor");
            // set input data
            const uint8_t inpudata[] = { 1,  2, 3, 4,  
                                       5,  6, 7, 8,
                                       9,  10, 11, 12,
                                       13, 14, 15, 16 };
            auto inputPtr          = input->writeMap<uint8_t>();
            memcpy(inputPtr, inpudata, 16 * sizeof(uint8_t));
            auto output0                             = _Reverse(input, _Scalar<int32_t>(0));
            const std::vector<uint8_t> expectedOutput0 = {
                                                        9,  10, 11, 12, 13, 14, 15, 16,
                                                        1,  2,  3,  4,  5,  6, 7,  8   };
            auto gotOutput0                          = output0->readMap<uint8_t>();
            for (int i = 0; i < 16; ++i) {
                auto diff = ::fabsf(gotOutput0[i] - expectedOutput0[i]);
                if (diff > 0.01) {
                    MNN_ERROR("ReverseTest4D[axis=0] test failed: %d - %d!\n", expectedOutput0[i], gotOutput0[i]);
                    return false;
                }
            }
            auto output1                             = _Reverse(input, _Scalar<int32_t>(1));
            const std::vector<uint8_t> expectedOutput1 = {5,  6, 7, 8, 1,  2,  3,  4,
                                                        13, 14, 15, 16,  9,  10, 11, 12,
                                                        };
            auto gotOutput1                          = output1->readMap<uint8_t>();
            for (int i = 0; i < 16; ++i) {
                auto diff = ::fabsf(gotOutput1[i] - expectedOutput1[i]);
                if (diff > 0.01) {
                    MNN_ERROR("ReverseTest4D[axis=1] test failed: %d - %d!\n", expectedOutput1[i], gotOutput1[i]);
                    return false;
                }
            }
            auto output2                             = _Reverse(input, _Scalar<int32_t>(2));
            const std::vector<uint8_t> expectedOutput2 = { 3,  4,  1,  2,  7,  8, 5, 6,
                                                        11, 12, 9, 10, 15, 16, 13, 14 };
            auto gotOutput2                          = output2->readMap<uint8_t>();
            for (int i = 0; i < 16; ++i) {
                auto diff = ::fabsf(gotOutput2[i] - expectedOutput2[i]);
                if (diff > 0.01) {
                    MNN_ERROR("ReverseTest4D[axis=2] test failed: %d - %d!\n", expectedOutput2[i], gotOutput2[i]);
                    return false;
                }
            }
            auto output3                             = _Reverse(input, _Scalar<int32_t>(3));
            const std::vector<uint8_t> expectedOutput3 = { 2,  1,  4,  3,  6, 5, 8, 7,
                                                        10, 9, 12, 11, 14, 13, 16, 15 };
            auto gotOutput3                          = output3->readMap<uint8_t>();
            for (int i = 0; i < 16; ++i) {
                auto diff = ::fabsf(gotOutput3[i] - expectedOutput3[i]);
                if (diff > 0.01) {
                    MNN_ERROR("ReverseTest4D[axis=3] test failed: %d - %d!\n", expectedOutput3[i], gotOutput3[i]);
                    return false;
                }
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(ReverseTest, "op/reverse");
