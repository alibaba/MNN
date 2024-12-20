//
//  StftSpeed.cpp
//  MNNTests
//
//  Created by MNN on 2024/11/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef MNN_BUILD_AUDIO

#include <math.h>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <random>
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include "MNNTestSuite.h"
using namespace MNN::Express;
#define SAMPLE 10240
#define NFFT 256
#define HOP 128
#define TIME 100
class StftSpeed : public MNNTestCase {
public:
    virtual bool run(int precision) {
        auto x = _Input({SAMPLE}, NHWC);
        auto w = _Input({NFFT}, NHWC);
        auto y = _Stft(x, w, NFFT, HOP);
        {
            AUTOTIME;
            for (int i = 0; i < TIME; ++i) {
                x->writeMap<float>();
                w->writeMap<float>();
                y->readMap<float>();
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(StftSpeed, "speed/stft");
#endif // MNN_BUILD_AUDIO