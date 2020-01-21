//
//  PoolGrad.cpp
//  MNN
//
//  Created by MNN on 2019/04/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "PoolGrad.hpp"
#include "core/Macro.h"
using namespace std;
using namespace MNN;
using namespace MNN::Express;

class PoolGrad : public OpGrad {
public:
    PoolGrad() {
        mType = SEMI_LINEAR;
    }

    virtual std::vector<Express::VARP> onGrad(Express::EXPRP expr,
                                              const std::vector<Express::VARP>& backwardOutput) override {
        std::vector<Express::VARP> result(1, nullptr);
        auto outputDiff = backwardOutput[0];
        std::unique_ptr<OpT> forwardOp(expr->get()->UnPack());
        unique_ptr<OpT> newOp(new OpT);
        newOp->type       = OpType_PoolGrad;
        auto copyP        = new PoolT(*forwardOp->main.AsPool());
        newOp->main.type  = OpParameter_Pool;
        newOp->main.value = copyP;

        result[0] = Variable::create(
            Expr::create(std::move(newOp), {expr->inputs()[0], Variable::create(expr, 0), outputDiff}));
        result[0]->setName(expr->name() + "_Grad");
        return result;
    }
};

static const auto gRegister = []() {
    static PoolGrad _c;
    OpGrad::insert(OpType_Pooling, &_c);
    return true;
}();
