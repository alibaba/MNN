//
//  ReluGrad.cpp
//  MNN
//
//  Created by MNN on 2019/04/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ReluGrad.hpp"
#include "core/Macro.h"
using namespace std;
using namespace MNN;

class ReluGrad : public OpGrad {
public:
    ReluGrad() {
        mType = SEMI_LINEAR;
    }
    virtual std::vector<Express::VARP> onGrad(Express::EXPRP expr,
                                              const std::vector<Express::VARP>& backwardOutput) override {
        std::vector<Express::VARP> result(1, nullptr);

        unique_ptr<OpT> newOp(new OpT);
        newOp->type       = OpType_ReluGrad;
        newOp->main.type  = OpParameter_Relu;
        newOp->main.value = new ReluT;

        result[0] =
            Express::Variable::create(Express::Expr::create(std::move(newOp), {expr->inputs()[0], backwardOutput[0]}));
        result[0]->setName(expr->name() + "_Grad");

        return result;
    }
};
class Relu6Grad : public OpGrad {
public:
    Relu6Grad() {
        mType = SEMI_LINEAR;
    }
    virtual std::vector<Express::VARP> onGrad(Express::EXPRP expr,
                                              const std::vector<Express::VARP>& backwardOutput) override {
        std::vector<Express::VARP> result{nullptr};

        unique_ptr<OpT> newOp(new OpT);
        newOp->type      = OpType_Relu6Grad;
        newOp->main.type = OpParameter_NONE;
        result[0] =
            Express::Variable::create(Express::Expr::create(std::move(newOp), {expr->inputs()[0], backwardOutput[0]}));
        result[0]->setName(expr->name() + "_Grad");
        return result;
    }
};
static const auto gRegister = []() {
    static ReluGrad _c;
    OpGrad::insert(OpType_ReLU, &_c);
    static Relu6Grad _d;
    OpGrad::insert(OpType_ReLU6, &_d);
    return true;
}();
