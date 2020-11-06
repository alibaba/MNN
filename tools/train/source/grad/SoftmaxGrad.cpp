//
//  SoftmaxGrad.cpp
//  MNN
//
//  Created by MNN on 2019/04/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "SoftmaxGrad.hpp"
#include "core/Macro.h"
#include <MNN/expr/ExprCreator.hpp>
using namespace std;
using namespace MNN;
using namespace MNN::Express;

class SoftmaxGrad : public OpGrad {
public:
    SoftmaxGrad() {
        mType = NO_LINEAR;
    }
    virtual std::vector<Express::VARP> onGrad(Express::EXPRP expr,
                                              const std::vector<Express::VARP>& backwardOutput) override {
        MNN_ASSERT(expr->inputs().size() == 1 && backwardOutput.size() == 1);
        auto input = expr->inputs()[0];
        auto info = input->getInfo();
        auto gradSoftmax = backwardOutput[0];
        if (nullptr == info) {
            return {};
        }
        auto axis = expr->get()->main_as_Axis()->axis();
        if (axis < 0) {
            axis = axis + info->dim.size();
        }
        auto softmax = Express::Variable::create(expr, 0);
        auto originOrder = info->order;
        if (originOrder == NC4HW4) {
            gradSoftmax = _Convert(gradSoftmax, NCHW);
            softmax = _Convert(softmax, NCHW);
        }
        auto sumAxis = _ReduceSum(softmax * gradSoftmax, {axis}, true);
        auto inputGrad = (gradSoftmax - sumAxis) * softmax;
        if (originOrder == NC4HW4) {
            inputGrad = _Convert(inputGrad, NC4HW4);
        }
        return {inputGrad};
    }
};
static const auto gRegister = []() {
    static SoftmaxGrad _c;
    OpGrad::insert(OpType_Softmax, &_c);
    return true;
}();
