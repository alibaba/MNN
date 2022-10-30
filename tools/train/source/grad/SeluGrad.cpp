//
//  SeluGrad.cpp
//  MNN
//
//  Created by MNN on 2022/08/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpGrad.hpp"
#include "core/Macro.h"
using namespace std;
using namespace MNN;
using namespace MNN::Express;

class SeluGrad : public OpGrad {
public:
    virtual std::vector<Express::VARP> onGrad(Express::EXPRP expr,
                                              const std::vector<Express::VARP>& backwardOutput) override {
        std::vector<Express::VARP> res{nullptr};
        auto input = expr->inputs()[0];
        std::unique_ptr<OpT> forwardOp(expr->get()->UnPack());
        auto param = forwardOp->main.AsSelu();
        float scale = param->scale;
        float alpha = param->alpha;

        // selu(x) = scale * alpha * (exp(x) - 1), if x <= 0
        //           scale * x, if x > 0

        // d selu(x) = scale * alpha * exp(x), if x <= 0
        //             scale, if x > 0

        auto mask0 = _Cast<float>(_Greater(input, _Scalar(0.0f)));
        auto mask1 = _Cast<float>(_LessEqual(input, _Scalar(0.0f)));
        auto factor = mask0 * _Scalar<float>(scale) + mask1 * _Scalar<float>(scale * alpha) * _Exp(input);

        res[0] = factor * backwardOutput[0];
        return res;
    }
};

static const auto gRegister = []() {
    static SeluGrad _c;
    OpGrad::insert(OpType_Selu, &_c);
    return true;
}();
