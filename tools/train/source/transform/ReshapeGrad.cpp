//
//  ReshapeGrad.cpp
//  MNN
//
//  Created by MNN on 2019/04/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ReshapeGrad.hpp"
#include "Macro.h"
using namespace std;
using namespace MNN;
using namespace MNN::Express;

class ReshapeGrad : public OpGrad {
public:
    virtual std::vector<Express::VARP> onGrad(Express::EXPRP expr, const std::vector<Express::VARP>& output, const std::vector<Express::VARP>& backwardOutput) override {
        std::vector<VARP> result;
        auto inputs = expr->inputs();
        result.resize(inputs.size());
        // Create Shape Op and Tensor
        unique_ptr<OpT> newOp(new OpT);
        newOp->type          = OpType_Shape;
        auto shape = Variable::create(Expr::create(std::move(newOp), {inputs[0]}));

        // Create Reshape Op
        result[0] = _Reshape(backwardOutput[0], shape);
        return result;
    }
};

static const auto gRegister = []() {
    static ReshapeGrad _c;
    OpGrad::insert(OpType_Reshape, &_c);
    OpGrad::insert(OpType_Squeeze, &_c);
    return true;
}();
