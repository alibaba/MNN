//
//  BatchMatMulTest.cpp
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
#include "TestUtils.h"

using namespace MNN::Express;

static void fillFloat(float* dst, int h, int w, ConvertFP32 functor, float offset = 0.0f) {
    for (int y = 0; y < h; ++y) {
        auto dstY = dst + w * y;
        for (int x = 0; x < w; ++x) {
            int temp = (x + y) % 31;
            dstY[x] = functor(((float)temp + offset) * 0.01f);
        }
    }
}

static bool checkMatMul(const float* C, const float* A, const float* B, int e, int l, int h, ConvertFP32 functor) {
    bool res = true;
    for (int y = 0; y < h; ++y) {
        auto AY = A + l * y;
        auto CY = C + e * y;
        for (int x = 0; x < e; ++x) {
            auto BX        = B + x;
            float expected = 0.0f;
            auto computed  = CY[x];
            for (int k = 0; k < l; ++k) {
                expected += functor(AY[k]) * functor(BX[k * e]);
            }
            expected = functor(expected);
            auto diff = fabsf(expected - computed);
            if (diff / fabsf(expected) > 0.005f) {
                MNN_PRINT("%f -> %f\n", expected, computed);
                res = false;
            }
        }
    }
    return res;
}

class BatchMatMulTest : public MNNTestCase {
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
            fillFloat(x0->writeMap<float>(), h, l, FP32Converter[precision]);
            fillFloat(x1->writeMap<float>(), l, e, FP32Converter[precision]);

            auto res = checkMatMul(y->readMap<float>(), x0->readMap<float>(), x1->readMap<float>(), e, l, h, FP32Converter[precision]);
            if (!res) {
                FUNC_PRINT(1);
                return false;
            }
            auto tranposeA          = _Transpose(x0, {1, 0});
            matmulParam->transposeA = true;
            matmulParam->transposeB = false;
            y                       = Variable::create(Expr::create(op.get(), {tranposeA, x1}));
            res = checkMatMul(y->readMap<float>(), x0->readMap<float>(), x1->readMap<float>(), e, l, h, FP32Converter[precision]);
            if (!res) {
                FUNC_PRINT(1);
                return false;
            }
            auto tranposeB          = _Transpose(x1, {1, 0});
            matmulParam->transposeA = true;
            matmulParam->transposeB = true;
            y                       = Variable::create(Expr::create(op.get(), {tranposeA, tranposeB}));
            res = checkMatMul(y->readMap<float>(), x0->readMap<float>(), x1->readMap<float>(), e, l, h, FP32Converter[precision]);
            if (!res) {
                FUNC_PRINT(1);
                return false;
            }
            matmulParam->transposeA = false;
            matmulParam->transposeB = true;
            y                       = Variable::create(Expr::create(op.get(), {x0, tranposeB}));
            res = checkMatMul(y->readMap<float>(), x0->readMap<float>(), x1->readMap<float>(), e, l, h, FP32Converter[precision]);
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
                fillFloat(x0Ptr + b * h * l, h, l, FP32Converter[precision], (float)b * 10);
                fillFloat(x1Ptr + b * e * l, l, e, FP32Converter[precision], (float)b * 10);
            }
            auto y    = Variable::create(Expr::create(op.get(), {x0, x1}));
            auto yPtr = y->readMap<float>();
            for (int b = 0; b < batch; ++b) {
                auto res = checkMatMul(yPtr + b * e * h, x0Ptr + b * h * l, x1Ptr + b * e * l, e, l, h, FP32Converter[precision]);
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
                fillFloat(x0Ptr + b * h * l, h, l, FP32Converter[precision], (float)b * 10);
                fillFloat(x1Ptr + b * e * l, l, e, FP32Converter[precision], (float)b * 10);
            }
            auto tranposeA = _Transpose(x0, {0, 2, 1});
            auto y         = Variable::create(Expr::create(op.get(), {tranposeA, x1}));

            auto yPtr = y->readMap<float>();
            for (int b = 0; b < batch; ++b) {
                auto res = checkMatMul(yPtr + b * e * h, x0Ptr + b * h * l, x1Ptr + b * e * l, e, l, h, FP32Converter[precision]);
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
                fillFloat(x0Ptr + b * h * l, h, l, FP32Converter[precision], (float)b * 10);
                fillFloat(x1Ptr + b * e * l, l, e, FP32Converter[precision], (float)b * 10);
            }
            auto tranposeB = _Transpose(x1, {0, 2, 1});
            auto y         = Variable::create(Expr::create(op.get(), {x0, tranposeB}));

            auto yPtr = y->readMap<float>();
            for (int b = 0; b < batch; ++b) {
                auto res = checkMatMul(yPtr + b * e * h, x0Ptr + b * h * l, x1Ptr + b * e * l, e, l, h, FP32Converter[precision]);
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
                fillFloat(x0Ptr + b * h * l, h, l, FP32Converter[precision], (float)b * 10);
                fillFloat(x1Ptr + b * e * l, l, e, FP32Converter[precision], (float)b * 10);
            }
            auto tranposeA = _Transpose(x0, {0, 2, 1});
            auto tranposeB = _Transpose(x1, {0, 2, 1});

            auto y = Variable::create(Expr::create(op.get(), {tranposeA, tranposeB}));

            auto yPtr = y->readMap<float>();
            for (int b = 0; b < batch; ++b) {
                auto res = checkMatMul(yPtr + b * e * h, x0Ptr + b * h * l, x1Ptr + b * e * l, e, l, h, FP32Converter[precision]);
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
            auto x1   = _Input({}, NHWC, halide_type_of<float>());
            x0->resize({b0, h, l});
            x1->resize({b1, l, e});
            auto x0Ptr = x0->writeMap<float>();
            auto x1Ptr = x1->writeMap<float>();
            for (int b = 0; b < b0; ++b) {
                fillFloat(x0Ptr + b * h * l, h, l, FP32Converter[precision], (float)b * 10);
            }
            for (int b = 0; b < b1; ++b) {
                fillFloat(x1Ptr + b * e * l, l, e, FP32Converter[precision], (float)b * 10);
            }
            auto tranposeA = _Transpose(x0, {0, 2, 1});
            auto tranposeB = _Transpose(x1, {0, 2, 1});

            auto y = Variable::create(Expr::create(op.get(), {tranposeA, tranposeB}));

            auto yPtr = y->readMap<float>();
            for (int b = 0; b < b0; ++b) {
                auto res = checkMatMul(yPtr + b * e * h, x0Ptr + b * h * l, x1Ptr, e, l, h, FP32Converter[precision]);
                if (!res) {
                    FUNC_PRINT(1);
                    return false;
                }
            }
        }
        // BatchMatMul batch = 1 with large K
        {
            std::vector<std::vector<int>> values = {
                {16, 262144, 15},
                {3, 262144, 16}
            };
            for(auto value : values) {
                e = value[0];
                l = value[1];
                h = value[2];
                
                std::unique_ptr<MNN::OpT> op(new MNN::OpT);
                op->type       = MNN::OpType_BatchMatMul;
                op->main.type  = MNN::OpParameter_BatchMatMulParam;
                op->main.value = new MNN::BatchMatMulParamT;
                auto param     = op->main.AsBatchMatMulParam();
                param->adjX    = false;
                param->adjY    = true;

                int batch = 1;
                auto x0   = _Input({}, NHWC, halide_type_of<float>());
                auto x1   = _Input({}, NHWC, halide_type_of<float>());
                x0->resize({batch, h, l});
                x1->resize({batch, l, e});
                auto x0Ptr = x0->writeMap<float>();
                auto x1Ptr = x1->writeMap<float>();
                for (int b = 0; b < batch; ++b) {
                    fillFloat(x0Ptr + b * h * l, h, l, FP32Converter[precision], (float)b * 10);
                    fillFloat(x1Ptr + b * e * l, l, e, FP32Converter[precision], (float)b * 10);
                }
                auto tranposeB = _Transpose(x1, {0, 2, 1});
                auto y         = Variable::create(Expr::create(op.get(), {x0, tranposeB}));

                auto yPtr = y->readMap<float>();
                for (int b = 0; b < batch; ++b) {
                    auto res = checkMatMul(yPtr + b * e * h, x0Ptr + b * h * l, x1Ptr + b * e * l, e, l, h, FP32Converter[precision]);
                    if (!res) {
                        FUNC_PRINT(1);
                        return false;
                    }
                }
            }

        }

        // BatchMatMul Large batch with small exlxh shape
        {
            std::unique_ptr<MNN::OpT> op(new MNN::OpT);
            op->type       = MNN::OpType_BatchMatMul;
            op->main.type  = MNN::OpParameter_BatchMatMulParam;
            op->main.value = new MNN::BatchMatMulParamT;
            auto param     = op->main.AsBatchMatMulParam();
            param->adjX    = false;
            param->adjY    = false;

            int batch = 532480;
            e = 1;
            l = 2;
            h = 2;
            auto x0   = _Input({}, NHWC, halide_type_of<float>());
            auto x1   = _Input({}, NHWC, halide_type_of<float>());
            x0->resize({batch, h, l});
            x1->resize({batch, l, e});
            auto x0Ptr = x0->writeMap<float>();
            auto x1Ptr = x1->writeMap<float>();
            for (int b = 0; b < batch; ++b) {
                fillFloat(x0Ptr + b * h * l, h, l, FP32Converter[precision], (float)((b * 10) % 5));
                fillFloat(x1Ptr + b * e * l, l, e, FP32Converter[precision], (float)((b * 10) % 5));
            }
            auto y    = Variable::create(Expr::create(op.get(), {x0, x1}));
            auto yPtr = y->readMap<float>();
            for (int b = 0; b < batch; ++b) {
                auto res = checkMatMul(yPtr + b * e * h, x0Ptr + b * h * l, x1Ptr + b * e * l, e, l, h, FP32Converter[precision]);
                if (!res) {
                    FUNC_PRINT(1);
                    return false;
                }
            }
        }
        // Broadcast matmul Large batch with 1d left shape
        {
            std::unique_ptr<MNN::OpT> op(new MNN::OpT);
            op->type       = MNN::OpType_BatchMatMul;
            op->main.type  = MNN::OpParameter_BatchMatMulParam;
            op->main.value = new MNN::BatchMatMulParamT;
            auto param     = op->main.AsBatchMatMulParam();
            param->adjX    = false;
            param->adjY    = false;

            int batch = 10;
            e = 2;
            l = 2;
            h = 1;
            auto x0   = _Input({}, NHWC, halide_type_of<float>());
            auto x1   = _Input({}, NHWC, halide_type_of<float>());
            x0->resize({l});
            x1->resize({batch, l, e});
            auto x0Ptr = x0->writeMap<float>();
            auto x1Ptr = x1->writeMap<float>();
            fillFloat(x0Ptr, h, l, FP32Converter[precision], 0.03f);
            for (int b = 0; b < batch; ++b) {
                fillFloat(x1Ptr + b * e * l, l, e, FP32Converter[precision], (float)((b * 10) % 5));
            }
            auto y    = Variable::create(Expr::create(op.get(), {x0, x1}));
            auto yPtr = y->readMap<float>();
            for (int b = 0; b < batch; ++b) {
                auto res = checkMatMul(yPtr + b * e * h, x0Ptr, x1Ptr + b * e * l, e, l, h, FP32Converter[precision]);
                if (!res) {
                    FUNC_PRINT(1);
                    return false;
                }
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(BatchMatMulTest, "op/BatchMatMul");
