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

class MatMulGrad : public OpGrad {
public:
    MatMulGrad() {
        mType = LINEAR;
    }
    virtual std::vector<Express::VARP> onGrad(Express::EXPRP expr, const std::vector<Express::VARP>& output, const std::vector<Express::VARP>& backwardOutput) override {
        std::vector<Express::VARP> res;
        auto inputs = expr->inputs();
        res.resize(inputs.size());
        auto outputDiff = backwardOutput[0];
        
        {
            // A' = C' * BT
            unique_ptr<OpT> newOp(new OpT);
//            newOp->inputIndexes                = {outputDiff, forwardOp->inputIndexes[1]};
            newOp->type                        = OpType_MatMul;
            newOp->main.type                   = OpParameter_MatMul;
            newOp->main.value                  = new MatMulT;
            newOp->main.AsMatMul()->transposeB = true;
            auto expr = Expr::create(std::move(newOp), {outputDiff, inputs[1]});
            res[0] = Variable::create(expr);
        }
        {
            // B' = AT * C'
            unique_ptr<OpT> newOp(new OpT);
//            newOp->inputIndexes                = {forwardOp->inputIndexes[0], outputDiff};
//            newOp->outputIndexes               = {gradTensors[1]};
            newOp->type                        = OpType_MatMul;
            newOp->main.type                   = OpParameter_MatMul;
            newOp->main.value                  = new MatMulT;
            newOp->main.AsMatMul()->transposeA = true;
            auto expr = Expr::create(std::move(newOp), {inputs[0], outputDiff});
            res[1] = Variable::create(expr);
        }
        return res;
    }
};
static const auto gRegister = []() {
    static MatMulGrad _c;
    OpGrad::insert(OpType_MatMul, &_c);
    return true;
}();
