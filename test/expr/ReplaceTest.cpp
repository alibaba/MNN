//
//  ReplaceTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/09/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include <math.h>

using namespace MNN::Express;
class PrecomputeTest : public MNNTestCase {
public:
    virtual bool run() {
        auto x = _Input({100}, NCHW);
        auto xPtr = x->writeMap<float>();
        for (int i=0; i<100; ++i) {
            xPtr[i] = (float)i - 50.0f;
        }
        auto y = _Abs(x);
        
        auto z = _Square(y);
        auto u = _Sin(z);
        auto v = _Cos(z);
        Variable::prepareCompute({y, u, v});
        auto check = [&](int number) {
            {
                auto yPtr = y->readMap<float>();
                if (nullptr == yPtr) {
                    return false;
                }
                for (int i=0; i<number; ++i) {
                    if (yPtr[i] != fabs((float)i - 50.0f)) {
                        MNN_PRINT("PrecomputeTest Error: %f, %f\n", yPtr[i], fabs((float)i - 50.0f));
                        return false;
                    }
                }
                auto uPtr = u->readMap<float>();
                for (int i=0; i<number; ++i) {
                    auto target = sinf(yPtr[i] * yPtr[i]);
                    auto diff = fabsf(uPtr[i] - target);
                    if (diff > 0.00001f) {
                        MNN_PRINT("PrecomputeTest Error: %f, %f\n",uPtr[i], target);
                        return false;
                    }
                }
                auto vPtr = v->readMap<float>();
                for (int i=0; i<number; ++i) {
                    auto target = cosf(yPtr[i] * yPtr[i]);
                    auto diff = fabsf(vPtr[i] - target);
                    if (diff > 0.00001f) {
                        MNN_PRINT("PrecomputeTest Error: %f, %f\n",vPtr[i], target);
                        return false;
                    }
                }
            }
            return true;
        };
        if (!check(100)) {
            return false;
        }
        {
            x->resize({1, 101});
            auto xPtr = x->writeMap<float>();
            for (int i=0; i<101; ++i) {
                xPtr[i] = (float)i - 50.0f;
            }
        }
        if (!check(101)) {
            return false;
        }
        // Delete end var, check if the cache can work
        u = nullptr;
        {
            x->writeMap<float>();
            auto xPtr = x->writeMap<float>();
            auto number = 101;
            for (int i=0; i<number; ++i) {
                xPtr[i] = (float)i - 50.0f;
            }
            auto yPtr = y->readMap<float>();
            if (nullptr == yPtr) {
                return false;
            }
            for (int i=0; i<number; ++i) {
                if (yPtr[i] != fabs((float)i - 50.0f)) {
                    MNN_PRINT("PrecomputeTest Error: %f, %f\n", yPtr[i], fabs((float)i - 50.0f));
                    return false;
                }
            }
            auto vPtr = v->readMap<float>();
            for (int i=0; i<number; ++i) {
                auto target = cosf(yPtr[i] * yPtr[i]);
                auto diff = fabsf(vPtr[i] - target);
                if (diff > 0.00001f) {
                    MNN_PRINT("PrecomputeTest Error: %f, %f\n",vPtr[i], target);
                    return false;
                }
            }

            number = 102;
            x->resize({number, 1});
            xPtr = x->writeMap<float>();
            for (int i=0; i<number; ++i) {
                xPtr[i] = (float)i - 50.0f;
            }
            yPtr = y->readMap<float>();
            if (nullptr == yPtr) {
                return false;
            }
            for (int i=0; i<number; ++i) {
                if (yPtr[i] != fabs((float)i - 50.0f)) {
                    MNN_PRINT("PrecomputeTest Error: %f, %f\n", yPtr[i], fabs((float)i - 50.0f));
                    return false;
                }
            }
            vPtr = v->readMap<float>();
            for (int i=0; i<number; ++i) {
                auto target = cosf(yPtr[i] * yPtr[i]);
                auto diff = fabsf(vPtr[i] - target);
                if (diff > 0.00001f) {
                    MNN_PRINT("PrecomputeTest Error: %f, %f\n",vPtr[i], target);
                    return false;
                }
            }
        }
        return true;
    }
};

class ReplaceTest : public MNNTestCase {
public:
    virtual bool run() {
        auto c1 = MNN::Express::_Const(1.f, {1, 1, 1, 1}, MNN::Express::NHWC);
        auto c2 = MNN::Express::_Const(2.f, {1, 1, 1, 1}, MNN::Express::NHWC);
        auto c3 = MNN::Express::_Const(3.f, {1, 1, 1, 1}, MNN::Express::NHWC);
        auto c4 = MNN::Express::_Const(4.f, {1, 1, 1, 1}, MNN::Express::NHWC);
        auto c5 = MNN::Express::_Const(5.f, {1, 1, 1, 1}, MNN::Express::NHWC);
        auto b1 = MNN::Express::_Add(c1, c2);
        auto b2 = MNN::Express::_Multiply(c3, c4);

        auto r1 = b1->readMap<float>();
        if (3.0f != r1[0]) {
            MNN_PRINT("1 + 2 = %f\n", r1[0]);
            return false;
        }

        MNN::Express::Variable::replace(c2, b2);
        auto r2 = b1->readMap<float>();
        if (13.0f != r2[0]) {
            MNN_PRINT("1 + 3 x 4 = %f\n", r2[0]);
            return false;
        }
        MNN::Express::Variable::replace(c3, c5);
        auto r3 = b1->readMap<float>();
        if (21.0f != r3[0]) {
            MNN_PRINT("1 + 5 x 4 = %f\n", r3[0]);
            return false;
        }
        auto d0 = _Const(7.f, {1, 3, 1, 1}, NHWC);
        auto d = _Split(d0, {1, 1, 1}, 1)[0];
        Variable::replace(c3, d);
        r3 = b1->readMap<float>();
        if (29.0f != r3[0]) {
            MNN_PRINT("1 + 7 x 4 = %f\n", r3[0]);
            return false;
        }
        return true;
    }
};
MNNTestSuiteRegister(ReplaceTest, "expr/Replace");
MNNTestSuiteRegister(PrecomputeTest, "expr/Precompute");
