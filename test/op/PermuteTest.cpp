//
//  PermuteTest.cpp
//  MNNTests
//
//  Created by MNN on 2020/05/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"

using namespace MNN::Express;

class PermuteTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        {
            // NC4HW4
            auto input  = _Input({10, 8, 37, 37}, NCHW, halide_type_of<float>());
            auto output = _Permute(input, {0, 3, 2, 1});
            std::vector<float> targetOutputs(10 * 8 * 37 * 37, 0.f);
            float maxValue = 1.0f;
            {
                auto func = FP32Converter[precision];
                auto ptr = input->writeMap<float>();
                ::memset(ptr, 0, input->getInfo()->size * sizeof(float));
                float index = 0.0f;
                for (int b = 0; b < 10; ++b) {
                    auto ptrB    = ptr + b * 8 * 37 * 37;
                    auto targetB = targetOutputs.data() + b * 8 * 37 * 37;
                    for (int c = 0; c < 8; ++c) {
                        auto targetC = targetB + c;
                        auto ptrC    = ptrB + c * 37 * 37;
                        for (int y = 0; y < 37; ++y) {
                            auto ptrY    = ptrC + y * 37;
                            auto targetY = targetC + y * 8;
                            for (int x = 0; x < 37; ++x) {
                                auto ptrX    = ptrY + x;
                                auto targetX = targetY + x * 8 * 37;
                                *ptrX        = func(index);
                                *targetX     = func(index);
                                index += 0.01f;
                            }
                        }
                    }
                }
                maxValue = index;
            }
            float errorScale = precision <= MNN::BackendConfig::Precision_High ? 1 : 50;
            {

                auto ptr  = output->readMap<float>();
                auto size = output->getInfo()->size;
                for (int i = 0; i < size; ++i) {
                    auto v = ptr[i] - targetOutputs[i];
                    if (v < 0) {
                        v = -v;
                    }
                    if (v / maxValue > 0.001f * errorScale) {
                        MNN_ERROR("%d, NCHW %f - %f error \n", i, ptr[i], targetOutputs[i]);
                        return false;
                    }
                }
            }
            // NC4HW4
            auto input2    = _Convert(input, NC4HW4);
            auto output2   = _Convert(_Permute(input2, {0, 3, 2, 1}), NCHW);
            auto ptr  = output2->readMap<float>();
            auto size = output2->getInfo()->size;
            for (int i = 0; i < size; ++i) {
                auto v = ptr[i] - targetOutputs[i];
                if (v < 0) {
                    v = -v;
                }
                if (v / maxValue > 0.001f * errorScale) {
                    MNN_ERROR("%d, NC4HW4 %f - %f error \n", i, ptr[i], targetOutputs[i]);
                    return false;
                }
            }
        }
        return true;
    }
};

MNNTestSuiteRegister(PermuteTest, "op/PermuteTest");
