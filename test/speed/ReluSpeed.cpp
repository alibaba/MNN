//
//  ReluSpeed.cpp
//  MNNTests
//
//  Created by MNN on 2019/09/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <math.h>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <random>
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include "MNNTestSuite.h"
using namespace MNN::Express;
#define WIDTH 5001
#define HEIGHT 1001
#define TIME 100
class ReluSpeed : public MNNTestCase {
public:
    void ReluTest() {
        auto x      = _Input({WIDTH, HEIGHT});
        auto output = _Relu(x);
        {
            AUTOTIME;
            for (int i = 0; i < TIME; ++i) {
                x->writeMap<float>();
                output->readMap<float>();
            }
        }
    }
    void Relu6Test() {
        auto x      = _Input({WIDTH, HEIGHT});
        auto output = _Relu6(x);
        {
            AUTOTIME;
            for (int i = 0; i < TIME; ++i) {
                x->writeMap<float>();
                output->readMap<float>();
            }
        }
    }
    virtual bool run(int precision) {
        MNN_PRINT("Test Relu for %d, %d x %d\n", WIDTH, HEIGHT, TIME);
        auto input0      = _Input({WIDTH, HEIGHT}, NHWC);
        auto input1      = _Input({WIDTH, HEIGHT}, NHWC);
        auto reluOutput  = _Relu(input0);
        auto relu6Output = _Relu6(input1);
        // Check Result
        {
            for (int i = 0; i < 2; ++i) {
                auto s0 = input0->writeMap<float>();
                auto s1 = input1->writeMap<float>();
                for (int y = 0; y < HEIGHT; ++y) {
                    for (int x = 0; x < WIDTH; ++x) {
                        float r0          = (((x * x + y * y * y) % 10000) - 5000) / 5000.0f;
                        float r1          = (((x * (WIDTH - x) + y * (HEIGHT - y)) % 10000) - 500) / 500.0f;
                        s0[y * WIDTH + x] = r0;
                        s1[y * WIDTH + x] = r1;
                    }
                }
                auto reluPtr  = reluOutput->readMap<float>();
                auto relu6Ptr = relu6Output->readMap<float>();
                for (int y = 0; y < HEIGHT; ++y) {
                    for (int x = 0; x < WIDTH; ++x) {
                        auto destRelu  = s0[y * WIDTH + x] > 0 ? s0[y * WIDTH + x] : 0.0f;
                        auto destRelu6 = s1[y * WIDTH + x];
                        destRelu6      = destRelu6 < 0 ? 0 : destRelu6;
                        destRelu6      = destRelu6 > 6 ? 6 : destRelu6;
                        auto absSub    = fabsf(reluPtr[y * WIDTH + x] - destRelu);
                        auto absAdd    = fabsf(relu6Ptr[y * WIDTH + x] - destRelu6);
                        if (absSub > 1e-6 || absAdd > 1e-6) {
                            return false;
                        }
                    }
                }
            }
        }
        ReluTest();
        Relu6Test();
        return true;
    }
};
MNNTestSuiteRegister(ReluSpeed, "speed/Relu");
