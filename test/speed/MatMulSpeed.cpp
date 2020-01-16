//
//  MatMulSpeedTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/09/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <math.h>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Optimizer.hpp>
#include "MNNTestSuite.h"
#include "MNN_generated.h"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
using namespace MNN::Express;

static void fillFloat(float* dst, int h, int w, float offset = 0.0f) {
    for (int y=0; y<h; ++y) {
        auto dstY = dst + w*y;
        for (int x=0; x<w; ++x) {
            dstY[x] = ((float)x * 0.1f + (float)y + offset) / 10000.0f;
        }
    }
}

static bool checkMatMul(const float* C, const float* A, const float* B, int e, int l, int h) {
    for (int y=0; y<h; ++y) {
        auto AY = A + l*y;
        auto CY = C + e*y;
        for (int x=0; x<e; ++x) {
            auto BX = B + x;
            float expected = 0.0f;
            auto computed = CY[x];
            for (int k=0; k<l; ++k) {
                expected += AY[k] * BX[k*e];
            }
            auto diff = fabsf(expected-computed);
            if (diff > 0.000001f) {
                printf("expected:%f,  computed:%f\n", expected, computed);
                return false;
            }
        }
    }
    return true;
}

class MatMulSpeedTest : public MNNTestCase {
public:
    virtual bool run() {
        int e=540, h=540, l=320;
        {
            //Test MatMul
            std::unique_ptr<MNN::OpT> op(new MNN::OpT);
            op->type = MNN::OpType_MatMul;
            op->main.type = MNN::OpParameter_MatMul;
            op->main.value = new MNN::MatMulT;
            auto matmulParam = op->main.AsMatMul();
            matmulParam->transposeA = false;
            matmulParam->transposeB = false;

            auto x0 = _Input({}, NHWC, halide_type_of<float>());
            auto x1 = _Input({}, NHWC, halide_type_of<float>());
            auto y = Variable::create(Expr::create(op.get(), {x0, x1}));
            x0->resize({h, l});
            x1->resize({l, e});
            fillFloat(x0->writeMap<float>(), h, l);
            fillFloat(x1->writeMap<float>(), l, e);

            auto res = checkMatMul(y->readMap<float>(), x0->readMap<float>(), x1->readMap<float>(), e, l, h);
            if (!res) {
                FUNC_PRINT(1);
                return false;
            }
            const auto time = 100;
            MNN_PRINT("MatMut: [%d, %d, %d], run %d\n", h, l, e, time);
            AUTOTIME;
            for (int t=0; t<time; ++t) {
                x0->writeMap<float>();
                x1->writeMap<float>();
                y->readMap<float>();
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(MatMulSpeedTest, "speed/MatMulTest");
