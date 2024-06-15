//
//  MatMulTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/09/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <math.h>
#include <MNN/expr/ExprCreator.hpp>
#include <random>
#include "MNNTestSuite.h"
#include "MNN_generated.h"
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Module.hpp>

using namespace MNN::Express;
using namespace MNN;

static void fillFloat(float* dst, int h, int w, float offset = 0.0f) {
    for (int y = 0; y < h; ++y) {
        auto dstY = dst + w * y;
        for (int x = 0; x < w; ++x) {
            dstY[x] = (float)x * 0.1f + (float)y + offset;
        }
    }
}

static bool checkMatMul(const float* C, const float* A, const float* B, int e, int l, int h) {
    bool res = true;
    for (int y = 0; y < h; ++y) {
        auto AY = A + l * y;
        auto CY = C + e * y;
        for (int x = 0; x < e; ++x) {
            auto BX        = B + x;
            float expected = 0.0f;
            auto computed  = CY[x];
            for (int k = 0; k < l; ++k) {
                expected += AY[k] * BX[k * e];
            }
            auto diff = fabsf(expected - computed);
            if (diff > 0.003f * fabsf(expected)) {
                MNN_PRINT("%f -> %f\n", expected, computed);
                res = false;
            }
        }
    }
    return res;
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
class MatMulTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        int e = 5, h = 4, l = 6;
        if (true) {
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
            auto y  = Variable::create(Expr::create(op.get(), {x0, x1}));
            x0->resize({h, l});
            x1->resize({l, e});
            fillFloat(x0->writeMap<float>(), h, l);
            fillFloat(x1->writeMap<float>(), l, e);

            auto res = checkMatMul(y->readMap<float>(), x0->readMap<float>(), x1->readMap<float>(), e, l, h);
            if (!res) {
                FUNC_PRINT(1);
                return false;
            }
            auto tranposeA          = _Transpose(x0, {1, 0});
            matmulParam->transposeA = true;
            matmulParam->transposeB = false;
            y                       = Variable::create(Expr::create(op.get(), {tranposeA, x1}));
            res = checkMatMul(y->readMap<float>(), x0->readMap<float>(), x1->readMap<float>(), e, l, h);
            if (!res) {
                FUNC_PRINT(1);
                return false;
            }
            auto tranposeB          = _Transpose(x1, {1, 0});
            matmulParam->transposeA = true;
            matmulParam->transposeB = true;
            y                       = Variable::create(Expr::create(op.get(), {tranposeA, tranposeB}));
            res = checkMatMul(y->readMap<float>(), x0->readMap<float>(), x1->readMap<float>(), e, l, h);
            if (!res) {
                FUNC_PRINT(1);
                return false;
            }
            matmulParam->transposeA = false;
            matmulParam->transposeB = true;
            y                       = Variable::create(Expr::create(op.get(), {x0, tranposeB}));
            res = checkMatMul(y->readMap<float>(), x0->readMap<float>(), x1->readMap<float>(), e, l, h);
            if (!res) {
                FUNC_PRINT(1);
                return false;
            }
        }
        if (true) {
            std::unique_ptr<MNN::OpT> op(new MNN::OpT);
            op->type       = MNN::OpType_BatchMatMul;
            op->main.type  = MNN::OpParameter_BatchMatMulParam;
            op->main.value = new MNN::BatchMatMulParamT;
            auto param     = op->main.AsBatchMatMulParam();
            param->adjX    = false;
            param->adjY    = false;

            int batch = 5;
            auto x0   = _Input({}, NHWC, halide_type_of<float>());
            auto x1   = _Input({}, NHWC, halide_type_of<float>());
            x0->resize({5, h, l});
            x1->resize({5, l, e});
            auto x0Ptr = x0->writeMap<float>();
            auto x1Ptr = x1->writeMap<float>();
            for (int b = 0; b < batch; ++b) {
                fillFloat(x0Ptr + b * h * l, h, l, (float)b * 10);
                fillFloat(x1Ptr + b * e * l, l, e, (float)b * 10);
            }
            auto y    = Variable::create(Expr::create(op.get(), {x0, x1}));
            auto yPtr = y->readMap<float>();
            for (int b = 0; b < batch; ++b) {
                auto res = checkMatMul(yPtr + b * e * h, x0Ptr + b * h * l, x1Ptr + b * e * l, e, l, h);
                if (!res) {
                    FUNC_PRINT(1);
                    return false;
                }
            }
        }

        {
            std::unique_ptr<MNN::OpT> op(new MNN::OpT);
            op->type       = MNN::OpType_BatchMatMul;
            op->main.type  = MNN::OpParameter_BatchMatMulParam;
            op->main.value = new MNN::BatchMatMulParamT;
            auto param     = op->main.AsBatchMatMulParam();
            param->adjX    = true;
            param->adjY    = false;

            int batch = 5;
            auto x0   = _Input({}, NHWC, halide_type_of<float>());
            auto x1   = _Input({}, NHWC, halide_type_of<float>());
            x0->resize({batch, h, l});
            x1->resize({batch, l, e});
            auto x0Ptr = x0->writeMap<float>();
            auto x1Ptr = x1->writeMap<float>();
            for (int b = 0; b < batch; ++b) {
                fillFloat(x0Ptr + b * h * l, h, l, (float)b * 10);
                fillFloat(x1Ptr + b * e * l, l, e, (float)b * 10);
            }
            auto tranposeA = _Transpose(x0, {0, 2, 1});
            auto y         = Variable::create(Expr::create(op.get(), {tranposeA, x1}));

            auto yPtr = y->readMap<float>();
            for (int b = 0; b < batch; ++b) {
                auto res = checkMatMul(yPtr + b * e * h, x0Ptr + b * h * l, x1Ptr + b * e * l, e, l, h);
                if (!res) {
                    FUNC_PRINT(1);
                    return false;
                }
            }
        }

        {
            std::unique_ptr<MNN::OpT> op(new MNN::OpT);
            op->type       = MNN::OpType_BatchMatMul;
            op->main.type  = MNN::OpParameter_BatchMatMulParam;
            op->main.value = new MNN::BatchMatMulParamT;
            auto param     = op->main.AsBatchMatMulParam();
            param->adjX    = false;
            param->adjY    = true;

            int batch = 5;
            auto x0   = _Input({}, NHWC, halide_type_of<float>());
            auto x1   = _Input({}, NHWC, halide_type_of<float>());
            x0->resize({5, h, l});
            x1->resize({5, l, e});
            auto x0Ptr = x0->writeMap<float>();
            auto x1Ptr = x1->writeMap<float>();
            for (int b = 0; b < batch; ++b) {
                fillFloat(x0Ptr + b * h * l, h, l, (float)b * 10);
                fillFloat(x1Ptr + b * e * l, l, e, (float)b * 10);
            }
            auto tranposeB = _Transpose(x1, {0, 2, 1});
            auto y         = Variable::create(Expr::create(op.get(), {x0, tranposeB}));

            auto yPtr = y->readMap<float>();
            for (int b = 0; b < batch; ++b) {
                auto res = checkMatMul(yPtr + b * e * h, x0Ptr + b * h * l, x1Ptr + b * e * l, e, l, h);
                if (!res) {
                    FUNC_PRINT(1);
                    return false;
                }
            }
        }

        {
            std::unique_ptr<MNN::OpT> op(new MNN::OpT);
            op->type       = MNN::OpType_BatchMatMul;
            op->main.type  = MNN::OpParameter_BatchMatMulParam;
            op->main.value = new MNN::BatchMatMulParamT;
            auto param     = op->main.AsBatchMatMulParam();
            param->adjX    = true;
            param->adjY    = true;

            int batch = 5;
            auto x0   = _Input({}, NHWC, halide_type_of<float>());
            auto x1   = _Input({}, NHWC, halide_type_of<float>());
            x0->resize({5, h, l});
            x1->resize({5, l, e});
            auto x0Ptr = x0->writeMap<float>();
            auto x1Ptr = x1->writeMap<float>();
            for (int b = 0; b < batch; ++b) {
                fillFloat(x0Ptr + b * h * l, h, l, (float)b * 10);
                fillFloat(x1Ptr + b * e * l, l, e, (float)b * 10);
            }
            auto tranposeA = _Transpose(x0, {0, 2, 1});
            auto tranposeB = _Transpose(x1, {0, 2, 1});

            auto y = Variable::create(Expr::create(op.get(), {tranposeA, tranposeB}));

            auto yPtr = y->readMap<float>();
            for (int b = 0; b < batch; ++b) {
                auto res = checkMatMul(yPtr + b * e * h, x0Ptr + b * h * l, x1Ptr + b * e * l, e, l, h);
                if (!res) {
                    FUNC_PRINT(1);
                    return false;
                }
            }
        }
        // Broadcast
        {
            std::unique_ptr<MNN::OpT> op(new MNN::OpT);
            op->type       = MNN::OpType_BatchMatMul;
            op->main.type  = MNN::OpParameter_BatchMatMulParam;
            op->main.value = new MNN::BatchMatMulParamT;
            auto param     = op->main.AsBatchMatMulParam();
            param->adjX    = true;
            param->adjY    = true;

            int b0 = 5;
            int b1 = 1;
            auto x0   = _Input({}, NHWC, halide_type_of<float>());
            x0->setName("x0");
            auto x1   = _Input({}, NHWC, halide_type_of<float>());
            x1->setName("x1");
            // Run as Module
            flatbuffers::FlatBufferBuilder builderOutput(1024);
            {
                auto tranposeA = _Transpose(x0, {0, 2, 1});
                auto tranposeB = _Transpose(x1, {0, 2, 1});
                auto y = Variable::create(Expr::create(op.get(), {tranposeA, tranposeB}));
                y->setName("y");
                std::unique_ptr<MNN::NetT> net(new NetT);
                Variable::save({y}, net.get());
                auto len = MNN::Net::Pack(builderOutput, net.get());
                builderOutput.Finish(len);
            }
            int sizeOutput    = builderOutput.GetSize();
            auto bufferOutput = builderOutput.GetBufferPointer();
            std::shared_ptr<MNN::Express::Module> module(Module::load(std::vector<std::string>{"x0", "x1"}, std::vector<std::string>{"y"}, bufferOutput, sizeOutput));
            for (int k=2; k<5; ++k) {
                b0 = k;
                x0->resize({b0, h, l});
                x1->resize({b1, l, e});
                auto x0Ptr = x0->writeMap<float>();
                auto x1Ptr = x1->writeMap<float>();
                for (int b = 0; b < b0; ++b) {
                    fillFloat(x0Ptr + b * h * l, h, l, (float)b * 10);
                }
                for (int b = 0; b < b1; ++b) {
                    fillFloat(x1Ptr + b * e * l, l, e, (float)b * 10);
                }
                auto y = module->onForward({x0, x1})[0];
                auto yPtr = y->readMap<float>();
                for (int b = 0; b < b0; ++b) {
                    auto res = checkMatMul(yPtr + b * e * h, x0Ptr + b * h * l, x1Ptr, e, l, h);
                    if (!res) {
                        FUNC_PRINT(1);
                        return false;
                    }
                }
            }
        }
        {
            int e = 23;
            int l = 33;
            int h = 9;
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
                
                if (diff > 0.01f * absMax) {
                    MNN_PRINT("%f error larger than %f * 0.001f\n", diff, absMax);
                    return false;
                }
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(MatMulTest, "expr/MatMul");
