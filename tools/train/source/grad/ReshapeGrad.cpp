//
//  ReshapeGrad.cpp
//  MNN
//
//  Created by MNN on 2019/04/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ReshapeGrad.hpp"
#include "core/Macro.h"
using namespace std;
using namespace MNN;
using namespace MNN::Express;

class ReshapeGrad : public OpGrad {
public:
    virtual std::vector<Express::VARP> onGrad(Express::EXPRP expr,
                                              const std::vector<Express::VARP>& backwardOutput) override {
        auto inputs = expr->inputs();
        std::vector<VARP> result(inputs.size(), nullptr);
        // Create Shape Op and Tensor
        unique_ptr<OpT> newOp(new OpT);
        newOp->type = OpType_Shape;
        auto shape  = Variable::create(Expr::create(std::move(newOp), {inputs[0]}));

        // Create Reshape Op
        result[0] = _Reshape(backwardOutput[0], shape);
        result[0]->setName(expr->name() + "_Grad");
        return result;
    }
};

static const auto gRegister = []() {
    static ReshapeGrad _c;
    OpGrad::insert(OpType_Reshape, &_c);
    OpGrad::insert(OpType_Squeeze, &_c);
    OpGrad::insert(OpType_Unsqueeze, &_c);
    return true;
}();
