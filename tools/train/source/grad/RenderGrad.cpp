//
//  RenderGrad.cpp
//  MNN
//
//  Created by MNN on 2023/07/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpGrad.hpp"
#include "core/Macro.h"
using namespace std;
using namespace MNN;
using namespace MNN::Express;

class RasterDiffGrad : public OpGrad {
public:
    virtual std::vector<Express::VARP> onGrad(Express::EXPRP expr,
                                              const std::vector<Express::VARP>& backwardOutput) override {
        std::vector<Express::VARP> res{nullptr};
        std::unique_ptr<OpT> forwardOp(expr->get()->UnPack());
        forwardOp->main.value = new ExtraT;
        forwardOp->main.type = OpParameter_Extra;
        auto diffExpr = Expr::create(forwardOp.get(), backwardOutput, 1);
        res[0] = Variable::create(diffExpr, 0);
        return res;
    }
};

static const auto gRegister = []() {
    static RasterDiffGrad _c;
    OpGrad::insert(OpType_RasterDiff, &_c);
    return true;
}();
