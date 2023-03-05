
//
//  ExecutorResetTest.cpp
//  MNNTests
//
//  Created by MNN on 2023/01/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include "MNNTestSuite.h"

using namespace MNN::Express;

class ExecutorResetTest : public MNNTestCase {
public:
    static VARP convBlock(VARP x, INTS channels, int stride) {
        int inputChannel = channels[0], outputChannel = channels[1];
        int group = inputChannel;
        x         = _Conv(0.0f, 0.0f, x, {inputChannel, inputChannel}, {3, 3}, SAME, {stride, stride}, {1, 1}, group);
        x         = _Conv(0.0f, 0.0f, x, {inputChannel, outputChannel}, {1, 1}, SAME, {1, 1}, {1, 1}, 1);
        return x;
    }
    static VARP convBlocTemp(VARP x, INTS channels, int stride) {
        int inputChannel = channels[0], outputChannel = channels[1];
        int group = inputChannel;
        x         = _Conv(0.0f, 0.0f, x, {inputChannel, inputChannel}, {3, 3}, SAME, {stride, stride}, {1, 1});
        x         = _Conv(0.0f, 0.0f, x, {inputChannel, outputChannel}, {1, 1}, SAME, {1, 1}, {1, 1}, 1);
        return x;
    }
    static VARP _mobileNetV1Expr(VARP x) {
        int inputSize = 224, poolSize; // MobileNet_224, MobileNet_192, MobileNet_160, MobileNet_128
        {
            inputSize = 224;
            poolSize  = inputSize / 32;
        }

        int channels[6]; // MobileNet_100, MobileNet_075, MobileNet_050, MobileNet_025
        { channels[0] = 32; }

        for (int i = 1; i < 6; ++i) {
            channels[i] = channels[0] * (1 << i);
        }

        x->setName("Input");
        x      = _Conv(0.0f, 0.0f, x, {3, channels[0]}, {3, 3}, SAME, {2, 2}, {1, 1}, 1);
        x      = convBlock(x, {channels[0], channels[1]}, 1);
        x      = convBlock(x, {channels[1], channels[2]}, 2);
        x      = convBlock(x, {channels[2], channels[2]}, 1);
        x      = convBlock(x, {channels[2], channels[3]}, 2);
        x      = convBlock(x, {channels[3], channels[3]}, 1);
        x      = convBlock(x, {channels[3], channels[4]}, 2);
        x      = convBlock(x, {channels[4], channels[4]}, 1);
        x      = convBlocTemp(x, {channels[4], channels[4]}, 1);
        x      = convBlock(x, {channels[4], channels[4]}, 1);
        x      = convBlock(x, {channels[4], channels[4]}, 1);
        x      = convBlock(x, {channels[4], channels[4]}, 1);
        x      = convBlock(x, {channels[4], channels[5]}, 2);
        x      = convBlock(x, {channels[5], channels[5]}, 1);
        x      = _AvePool(x, {poolSize, poolSize}, {1, 1}, VALID);
        x      = _Conv(0.0f, 0.0f, x, {channels[5], 1001}, {1, 1}, VALID, {1, 1}, {1, 1}, 1); // reshape FC with Conv1x1
        x      = _Softmax(x, -1);
        x      = _Convert(x, NCHW);
        x->setName("Prob");
        return x;
    }

    virtual bool run(int precision) {
        MNN::BackendConfig bnConfig;
        auto exe = Executor::newExecutor(MNN_FORWARD_CPU, bnConfig, 1);
        ExecutorScope scope(exe);
        exe->setGlobalExecutorConfig(MNN_FORWARD_CPU, bnConfig, 4);

        auto x = _Input({1, 3, 224, 224}, NC4HW4);
        auto y = _ReduceSum(_Multiply(x, x), {});
        ::memset(x->writeMap<float>(), 0, x->getInfo()->size * sizeof(float));
        y->readMap<float>();

        exe->setGlobalExecutorConfig(MNN_FORWARD_CPU, bnConfig, 4);
        ::memset(x->writeMap<float>(), 0, x->getInfo()->size * sizeof(float));
        y->readMap<float>();
        return true;
    }
};
MNNTestSuiteRegister(ExecutorResetTest, "expr/ExecutorReset");
