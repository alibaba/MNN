//
//  TransposeSpeed.cpp
//  MNNTests
//
//  Created by MNN on 2020/09/27.
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
#define WIDTH 501
#define HEIGHT 243
#define TIME 100
class TransposeSpeed : public MNNTestCase {
public:
    void SpeedTest() {
        auto x      = _Input({1, 1, HEIGHT, WIDTH}, NCHW, halide_type_of<float>());
        auto output = _Transpose(x, {0, 3, 2, 1});
        {
            AUTOTIME;
            for (int i = 0; i < TIME; ++i) {
                x->writeMap<float>();
                output->readMap<float>();
            }
        }
    }
    bool CorrectTest() {
        auto x      = _Input({1, 1, HEIGHT, WIDTH}, NCHW, halide_type_of<int>());
        std::vector<int> input(WIDTH * HEIGHT);
        for (int u=0; u<HEIGHT; ++u) {
            for (int v=0; v<WIDTH; ++v) {
                input[u * WIDTH + v] = u * u - v * 2;
            }
        }
        auto xPtr = x->writeMap<int>();
        ::memcpy(xPtr, input.data(), HEIGHT * WIDTH * sizeof(int));
        auto output = _Transpose(x, {0, 3, 2, 1});
        auto yPtr = output->readMap<int>();
        auto outputInfo = output->getInfo();
        MNN_PRINT("Output: %d, %d, %d, %d\n", outputInfo->dim[0], outputInfo->dim[1], outputInfo->dim[2], outputInfo->dim[3]);
        for (int u=0; u<HEIGHT; ++u) {
            for (int v=0; v<WIDTH; ++v) {
                auto correct = input[u * WIDTH + v];
                auto result = yPtr[u + v * HEIGHT];
                if (correct != result) {
                    MNN_ERROR("Error for %d - %d, correct: %d, error: %d\n", u, v, correct, result);
                    return false;
                }
            }
        }
        return true;
    }
    virtual bool run(int precision) {
        MNN_PRINT("Test Convert for %d, %d, x %d\n", WIDTH, HEIGHT, TIME);
        SpeedTest();
        return CorrectTest();
    }
};
MNNTestSuiteRegister(TransposeSpeed, "speed/Transpose");
