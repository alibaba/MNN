//
//  MatMulGrad.cpp
//  MNN
//
//  Created by MNN on 2019/05/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MatMulGrad.hpp"
using namespace std;
using namespace MNN;
using namespace MNN::Express;
class BatchMatMulGrad : public OpGrad {
public:
    BatchMatMulGrad() {
        mType = LINEAR;
    }
    virtual std::vector<Express::VARP> onGrad(Express::EXPRP expr,
                                              const std::vector<Express::VARP>& backwardOutput) override {
        std::vector<Express::VARP> res;
        auto inputs = expr->inputs();
        res.resize(inputs.size());
        auto outputDiff = backwardOutput[0];

        const bool transA = expr->get()->main_as_BatchMatMulParam()->adjX();
        const bool transB = expr->get()->main_as_BatchMatMulParam()->adjY();

        if (!transA && !transB) {
            {
                // A' = C' * BT
                res[0] = _BatchMatMul(outputDiff, inputs[1], false, true);
                // B' = AT * C'
                res[1] = _BatchMatMul(inputs[0], outputDiff, true, false);
            }
        }

        if (transA && !transB) {
            {
                // AT' = C' * BT ==> A' = B * CT'
                res[0] = _BatchMatMul(inputs[1], outputDiff, false, true);
            }

            {
                // B' = ATT * C' = A * C'
                res[1] = _BatchMatMul(inputs[0], outputDiff, false, false);
            }
        }

        if (!transA && transB) {
            {
                // A' = C' * BTT = C' * B
                res[0] = _BatchMatMul(outputDiff, inputs[1], false, false);
            }

            {
                // BT' = AT * C' ==> B' = CT' * A
                res[1] = _BatchMatMul(outputDiff, inputs[0], true, false);
            }
        }

        if (transA && transB) {
            {
                // AT' = C' * BTT  ==> A' = BT * CT'
                res[0] = _BatchMatMul(inputs[1], outputDiff, true, true);
            }

            {
                // BT' = ATT * C'  ==>  B' = CT' * AT
                res[1] = _BatchMatMul(outputDiff, inputs[0], true, true);
            }
        }

        return res;
    }
};
class MatMulGrad : public OpGrad {
public:
    MatMulGrad() {
        mType = LINEAR;
    }
    virtual std::vector<Express::VARP> onGrad(Express::EXPRP expr,
                                              const std::vector<Express::VARP>& backwardOutput) override {
        std::vector<Express::VARP> res;
        auto inputs = expr->inputs();
        res.resize(inputs.size());
        auto outputDiff = backwardOutput[0];

        const bool transA = expr->get()->main_as_MatMul()->transposeA();
        const bool transB = expr->get()->main_as_MatMul()->transposeB();

        if (!transA && !transB) {
            {
                // A' = C' * BT
                unique_ptr<OpT> newOp(new OpT);
                newOp->type                        = OpType_MatMul;
                newOp->main.type                   = OpParameter_MatMul;
                newOp->main.value                  = new MatMulT;
                newOp->main.AsMatMul()->transposeB = true;
                auto expr                          = Expr::create(std::move(newOp), {outputDiff, inputs[1]});
                res[0]                             = Variable::create(expr);
            }

            {
                // B' = AT * C'
                unique_ptr<OpT> newOp(new OpT);
                newOp->type                        = OpType_MatMul;
                newOp->main.type                   = OpParameter_MatMul;
                newOp->main.value                  = new MatMulT;
                newOp->main.AsMatMul()->transposeA = true;
                auto expr                          = Expr::create(std::move(newOp), {inputs[0], outputDiff});
                res[1]                             = Variable::create(expr);
            }
        }

        if (transA && !transB) {
            {
                // AT' = C' * BT ==> A' = B * CT'
                unique_ptr<OpT> newOp(new OpT);
                newOp->type                        = OpType_MatMul;
                newOp->main.type                   = OpParameter_MatMul;
                newOp->main.value                  = new MatMulT;
                newOp->main.AsMatMul()->transposeA = false;
                newOp->main.AsMatMul()->transposeB = true;
                auto expr                          = Expr::create(std::move(newOp), {inputs[1], outputDiff});
                res[0]                             = Variable::create(expr);
            }

            {
                // B' = ATT * C' = A * C'
                unique_ptr<OpT> newOp(new OpT);
                newOp->type                        = OpType_MatMul;
                newOp->main.type                   = OpParameter_MatMul;
                newOp->main.value                  = new MatMulT;
                newOp->main.AsMatMul()->transposeA = false;
                newOp->main.AsMatMul()->transposeB = false;
                auto expr                          = Expr::create(std::move(newOp), {inputs[0], outputDiff});
                res[1]                             = Variable::create(expr);
            }
        }

        if (!transA && transB) {
            {
                // A' = C' * BTT = C' * B
                unique_ptr<OpT> newOp(new OpT);
                newOp->type                        = OpType_MatMul;
                newOp->main.type                   = OpParameter_MatMul;
                newOp->main.value                  = new MatMulT;
                newOp->main.AsMatMul()->transposeA = false;
                newOp->main.AsMatMul()->transposeB = false;
                auto expr                          = Expr::create(std::move(newOp), {outputDiff, inputs[1]});
                res[0]                             = Variable::create(expr);
            }

            {
                // BT' = AT * C' ==> B' = CT' * A
                unique_ptr<OpT> newOp(new OpT);
                newOp->type                        = OpType_MatMul;
                newOp->main.type                   = OpParameter_MatMul;
                newOp->main.value                  = new MatMulT;
                newOp->main.AsMatMul()->transposeA = true;
                newOp->main.AsMatMul()->transposeB = false;
                auto expr                          = Expr::create(std::move(newOp), {outputDiff, inputs[0]});
                res[1]                             = Variable::create(expr);
            }
        }

        if (transA && transB) {
            {
                // AT' = C' * BTT  ==> A' = BT * CT'
                unique_ptr<OpT> newOp(new OpT);
                newOp->type                        = OpType_MatMul;
                newOp->main.type                   = OpParameter_MatMul;
                newOp->main.value                  = new MatMulT;
                newOp->main.AsMatMul()->transposeA = true;
                newOp->main.AsMatMul()->transposeB = true;
                auto expr                          = Expr::create(std::move(newOp), {inputs[1], outputDiff});
                res[0]                             = Variable::create(expr);
            }

            {
                // BT' = ATT * C'  ==>  B' = CT' * AT
                unique_ptr<OpT> newOp(new OpT);
                newOp->type                        = OpType_MatMul;
                newOp->main.type                   = OpParameter_MatMul;
                newOp->main.value                  = new MatMulT;
                newOp->main.AsMatMul()->transposeA = true;
                newOp->main.AsMatMul()->transposeB = true;
                auto expr                          = Expr::create(std::move(newOp), {outputDiff, inputs[0]});
                res[1]                             = Variable::create(expr);
            }
        }

        return res;
    }
};
static const auto gRegister = []() {
    static MatMulGrad _c;
    OpGrad::insert(OpType_MatMul, &_c);
    static BatchMatMulGrad _d;
    OpGrad::insert(OpType_BatchMatMul, &_d);
    return true;
}();
