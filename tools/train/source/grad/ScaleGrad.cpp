//
//  ScaleGrad.cpp
//  MNN
//
//  Created by MNN on 2022/08/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpGrad.hpp"
#include "core/Macro.h"
using namespace std;
using namespace MNN;
using namespace MNN::Express;

class ScaleGrad : public OpGrad {
public:
    virtual std::vector<Express::VARP> onGrad(Express::EXPRP expr,
                                              const std::vector<Express::VARP>& backwardOutput) override {
        auto inputs = expr->inputs();
        std::vector<VARP> res(inputs.size(), nullptr);
        auto outputDiff = backwardOutput[0];

        std::unique_ptr<OpT> forwardOp(expr->get()->UnPack());
        auto param = forwardOp->main.AsScale();
        auto channels = param->channels;
        auto scale = param->scaleData;
        auto zeros = std::vector<float>(scale.size(), 0.0f);

        res[0] = _Scale(outputDiff, channels, std::move(scale), std::move(zeros));

        return res;
    }
};

static const auto gRegister = []() {
    static ScaleGrad _c;
    OpGrad::insert(OpType_Scale, &_c);
    return true;
}();
