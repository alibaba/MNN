//
//  SoftmaxGrad.cpp
//  MNN
//
//  Created by MNN on 2019/04/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "SoftmaxGrad.hpp"
#include "core/Macro.h"
using namespace std;
using namespace MNN;

class SoftmaxGrad : public OpGrad {
public:
    SoftmaxGrad() {
        mType = NO_LINEAR;
    }
    virtual std::vector<Express::VARP> onGrad(Express::EXPRP expr,
                                              const std::vector<Express::VARP>& backwardOutput) override {
        std::vector<Express::VARP> result(1, nullptr);
        unique_ptr<OpT> newOp(new OpT);
        newOp->type                = OpType_SoftmaxGrad;
        newOp->main.type           = OpParameter_Axis;
        newOp->main.value          = new AxisT;
        newOp->main.AsAxis()->axis = expr->get()->main_as_Axis()->axis();
        result[0]                  = Express::Variable::create(
            Express::Expr::create(std::move(newOp), {Express::Variable::create(expr, 0), backwardOutput[0]}));
        result[0]->setName(expr->name() + "_Grad");
        return result;
    }
};
static const auto gRegister = []() {
    static SoftmaxGrad _c;
    OpGrad::insert(OpType_Softmax, &_c);
    return true;
}();
