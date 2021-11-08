//
//  MatMulSpeed.cpp
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
    for (int y = 0; y < h; ++y) {
        auto dstY = dst + w * y;
        for (int x = 0; x < w; ++x) {
            dstY[x] = ((float)x * 0.1f + (float)y + offset) / 10000.0f;
        }
    }
}

static void _originMatMul(float* C, const float* A, const float* B, int e, int l, int h) {
    for (int y = 0; y < e; ++y) {
        auto AY = A + l * y;
        auto CY = C + h * y;
        for (int x = 0; x < h; ++x) {
            auto BX        = B + x;
            float expected = 0.0f;
            for (int k = 0; k < l; ++k) {
                expected += AY[k] * BX[k * h];
            }
            CY[x] = expected;
        }
    }
}

static std::vector<std::vector<int>> _getComputeSize() {
    std::vector<std::vector<int>> elh {
        {540, 540, 320},
        {1024, 1024, 1024},
        {5, 1024, 1024},
        {1024, 1024, 5},
        {1024, 5, 1024},
        {8, 1024, 1024},
        {10, 1024, 1024},
        {1024, 1024, 10},
        {1024, 10, 1024},
        {1, 1024, 1024},
        {1024, 1024, 1},
        {1024, 1, 1024},
        {128, 128, 3072},
        {128, 3072, 128},
    };
    return elh;
}
class BatchMatMulSpeedTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        std::vector<std::vector<int>> elh = _getComputeSize();
        for (auto& iter : elh) {
            int e = iter[0];
            int h = iter[2];
            int l = iter[1];
            auto res = _run(e, h, l);
            if (!res) {
                return false;
            }
        }
        return true;
    }

    bool _run(int e, int h, int l) {
        {
            int batch = 20;
            // Test MatMul
            std::unique_ptr<MNN::OpT> op(new MNN::OpT);
            op->type                = MNN::OpType_MatMul;
            op->main.type           = MNN::OpParameter_MatMul;
            op->main.value          = new MNN::MatMulT;
            auto matmulParam        = op->main.AsMatMul();
            matmulParam->transposeA = false;
            matmulParam->transposeB = false;

            auto x0 = _Input({}, NHWC, halide_type_of<float>());
            auto x1 = _Input({}, NHWC, halide_type_of<float>());
            x0->resize({batch, e, l});
            x1->resize({batch, l, h});
            auto y = Variable::create(Expr::create(op.get(), {x0, x1}));
            Variable::prepareCompute({y});
            ::memset(x0->writeMap<float>(), 0, x0->getInfo()->size * sizeof(float));
            ::memset(x1->writeMap<float>(), 0, x1->getInfo()->size * sizeof(float));
            y->readMap<float>();

            const auto time = 5;
            MNN::Timer _t;
            for (int t = 0; t < time; ++t) {
                x0->writeMap<float>();
                x1->writeMap<float>();
                y->readMap<float>();
            }
            float timeCost = _t.durationInUs() / 1000.0f / (float)time;
            float flops = (float)batch * (float)e * (float)l * (float)h / timeCost / 1000.0f / 1000.0f;
            MNN_PRINT("[%d, %d, %d], Avg time: %f ms , flops: %f G\n", e, l, h, timeCost, flops);
        }
        return true;
    }
};

class MatMulSpeedTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        std::vector<std::vector<int>> elh = _getComputeSize();
        for (auto& iter : elh) {
            int e = iter[0];
            int h = iter[2];
            int l = iter[1];
            auto res = _run(e, h, l);
            if (!res) {
                return false;
            }
        }
        return true;
    }

    bool _run(int e, int h, int l) {
        {
            // Test MatMul
            std::unique_ptr<MNN::OpT> op(new MNN::OpT);
            op->type                = MNN::OpType_MatMul;
            op->main.type           = MNN::OpParameter_MatMul;
            op->main.value          = new MNN::MatMulT;
            auto matmulParam        = op->main.AsMatMul();
            matmulParam->transposeA = false;
            matmulParam->transposeB = false;

            auto x0 = _Input({}, NHWC, halide_type_of<float>());
            auto x1 = _Input({}, NHWC, halide_type_of<float>());
            x0->resize({e, l});
            x1->resize({l, h});
            auto y = Variable::create(Expr::create(op.get(), {x0, x1}));
            Variable::prepareCompute({y});
            auto dstY = _Input({e, h}, NHWC, halide_type_of<float>());
            fillFloat(x0->writeMap<float>(), e, l);
            fillFloat(x1->writeMap<float>(), l, h);
            _originMatMul(dstY->writeMap<float>(), x0->readMap<float>(), x1->readMap<float>(), e, l, h);

            auto absMaxV = _ReduceMax(_Abs(dstY));
            auto diffV   = _ReduceMax(_Abs(dstY - y));
            Variable::prepareCompute({absMaxV, diffV}, true);

            auto absMax = absMaxV->readMap<float>()[0];
            MNN_ASSERT(absMax != 0.0f);
            auto diff = diffV->readMap<float>()[0];

            bool res  = false;
            if (diff < 0.01f * absMax) {
                res = true;
            }
            if (!res) {
                MNN_PRINT("%f error larger than %f * 0.001f\n", diff, absMax);
                //                return false;
            }
            const auto time = 100;
            MNN::Timer _t;
            for (int t = 0; t < time; ++t) {
                x0->writeMap<float>();
                x1->writeMap<float>();
                y->readMap<float>();
            }
            float timeCost = _t.durationInUs() / 1000.0f / (float)time;
            float flops = (float)e * (float)l * (float)h / timeCost / 1000.0f / 1000.0f;
            MNN_PRINT("[%d, %d, %d], Avg time: %f ms , flops: %f G\n", e, l, h, timeCost, flops);
        }
        return true;
    }
};

class MatMulSpeedConstTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        std::vector<std::vector<int>> elh = _getComputeSize();
        for (auto& iter : elh) {
            int e = iter[0];
            int h = iter[2];
            int l = iter[1];
            auto res = _runConst(e, h, l);
            if (!res) {
                return false;
            }
        }
        return true;
    }

    bool _runConst(int e, int h, int l) {
        {
            // Use Conv1x1 instead of MatMul
            auto x0 = _Input({1, l, e, 1}, NC4HW4, halide_type_of<float>());
            auto y = _Conv(0.0f, 0.0f, x0, {l, h}, {1, 1});
            Variable::prepareCompute({y});
            int time = 100;
            if (e < 100 || l < 100 || h < 100) {
                time = 1000;
            }
            MNN_PRINT("MatMul B Const (Conv1x1): [%d, %d, %d], run %d\n", e, l, h, time);
            {
                AUTOTIME;
                //Prepare
                x0->writeMap<float>();
                y->readMap<float>();
            }
            MNN::Timer _t;
            for (int t = 0; t < time; ++t) {
                x0->writeMap<float>();
                y->readMap<float>();
            }
            float timeCost = _t.durationInUs() / 1000.0f / (float)time;
            float flops = (float)e * (float)l * (float)h / timeCost / 1000.0f / 1000.0f;
            MNN_PRINT("[%d, %d, %d], Avg time: %f ms , flops: %f G\n", e, l, h, timeCost, flops);
        }
        return true;
    }
};
MNNTestSuiteRegister(MatMulSpeedTest, "speed/MatMulTest");
MNNTestSuiteRegister(BatchMatMulSpeedTest, "speed/MatMulBatchTest");
MNNTestSuiteRegister(MatMulSpeedConstTest, "speed/MatMulBConstTest");
