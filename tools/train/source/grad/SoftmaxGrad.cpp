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
        auto outputDiff = backwardOutput[0];
        if (nullptr == info) {
            return {};
        }
        auto axis = expr->get()->main_as_Axis()->axis();
        if (axis < 0) {
            axis = axis + info->dim.size();
        }
        unique_ptr<OpT> newOp(new OpT);
        newOp->type                = OpType_SoftmaxGrad;
        newOp->main.type           = OpParameter_Axis;
        newOp->main.value          = new AxisT;
        newOp->main.AsAxis()->axis = 1;
        auto originOrder = info->order;
        auto output = Express::Variable::create(expr, 0);
        if (axis != 1) {
            if (originOrder == NC4HW4) {
                outputDiff = _Convert(outputDiff, NCHW);
                output = _Convert(output, NCHW);
            }
            std::vector<int> permuteDims(info->dim.size());
            for (int i=0; i<info->dim.size(); ++i) {
                permuteDims[i] = i;
            }
            permuteDims[1] = axis;
            permuteDims[axis] = 1;
            auto res = Express::Variable::create(Express::Expr::create(std::move(newOp), {_Transpose(output, permuteDims), _Transpose(outputDiff, permuteDims)}));
            res = _Transpose(res, permuteDims);
            if (originOrder == NC4HW4) {
                res = _Convert(res, originOrder);
            }
            return {res};
        }
        std::vector<Express::VARP> result(1, nullptr);
        result[0]                  = Express::Variable::create(
            Express::Expr::create(std::move(newOp), {Express::Variable::create(expr, 0), backwardOutput[0]}));
        return result;
    }
};
static const auto gRegister = []() {
    static SoftmaxGrad _c;
    OpGrad::insert(OpType_Softmax, &_c);
    return true;
}();
