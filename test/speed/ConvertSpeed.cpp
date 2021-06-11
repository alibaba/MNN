//
//  ConvertSpeed.cpp
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
#define WIDTH 500
#define HEIGHT 200
#define CHANNEL 161
#define TIME 100
class ConvertSpeed : public MNNTestCase {
public:
    void PackTest() {
        auto x      = _Input({1, CHANNEL, WIDTH, HEIGHT}, NCHW, halide_type_of<float>());
        auto output = _Convert(x, NC4HW4);
        {
            AUTOTIME;
            for (int i = 0; i < TIME; ++i) {
                x->writeMap<float>();
                output->readMap<float>();
            }
        }
    }
    void UnpackTest() {
        auto x      = _Input({1, CHANNEL, WIDTH, HEIGHT}, NC4HW4, halide_type_of<float>());
        auto output = _Convert(x, NCHW);
        {
            AUTOTIME;
            for (int i = 0; i < TIME; ++i) {
                x->writeMap<float>();
                output->readMap<float>();
            }
        }
    }
    void TransposePackTest() {
        auto x      = _Input({1, HEIGHT, WIDTH, CHANNEL}, NHWC, halide_type_of<float>());
        auto output = _Convert(x, NC4HW4);
        {
            AUTOTIME;
            for (int i = 0; i < TIME; ++i) {
                x->writeMap<float>();
                output->readMap<float>();
            }
        }
    }
    void TransposeUnPackTest() {
        auto x      = _Input({1, CHANNEL, WIDTH, HEIGHT}, NC4HW4, halide_type_of<float>());
        auto output = _Convert(x, NHWC);
        {
            AUTOTIME;
            for (int i = 0; i < TIME; ++i) {
                x->writeMap<float>();
                output->readMap<float>();
            }
        }
    }
    virtual bool run(int precision) {
        MNN_PRINT("Test Convert for %d, %d, %d x %d\n", WIDTH, HEIGHT, CHANNEL, TIME);
        PackTest();
        UnpackTest();
        TransposePackTest();
        TransposeUnPackTest();

        return true;
    }
};
MNNTestSuiteRegister(ConvertSpeed, "speed/Convert");
