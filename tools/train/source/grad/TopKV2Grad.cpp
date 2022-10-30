//
//  TopKV2Grad.cpp
//  MNN
//
//  Created by MNN on 2022/08/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpGrad.hpp"
#include "core/Macro.h"
using namespace std;
using namespace MNN;
using namespace MNN::Express;

class TopKV2Grad : public OpGrad {
public:
    virtual std::vector<Express::VARP> onGrad(Express::EXPRP expr,
                                              const std::vector<Express::VARP>& backwardOutput) override {
        auto inputs = expr->inputs();
        std::vector<VARP> res(inputs.size(), nullptr);
        auto outputDiff = backwardOutput[0];
        auto values = Variable::create(expr, 0);
        auto indices = Variable::create(expr, 1);

        auto zeros = _ZerosLike(inputs[0]);
        auto axis = _Scalar<int>(-1);
        res[0] = _ScatterElements(zeros, indices, outputDiff, axis);

        return res;
    }
};

static const auto gRegister = []() {
    static TopKV2Grad _c;
    OpGrad::insert(OpType_TopKV2, &_c);
    return true;
}();
