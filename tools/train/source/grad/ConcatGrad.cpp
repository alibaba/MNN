//
//  ConcatGrad.cpp
//  MNN
//
//  Created by MNN on 2019/12/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpGrad.hpp"
#include "core/Macro.h"
using namespace std;
using namespace MNN;
using namespace MNN::Express;

class ConcatGrad : public OpGrad {
public:
    virtual std::vector<Express::VARP> onGrad(Express::EXPRP expr,
                                              const std::vector<Express::VARP>& backwardOutput) override {
        std::vector<VARP> res(expr->inputs().size());
        if (!expr->requireInfo()) {
            return res;
        }
        auto axis = expr->get()->main_as_Axis()->axis();
        if (axis < 0) {
            axis = expr->outputInfo(0)->dim.size() + axis;
        }
        std::vector<int> points(res.size());
        for (int i = 0; i < res.size(); ++i) {
            auto input = expr->inputs()[i];
            points[i]  = input->getInfo()->dim[axis];
        }
        res = _Split(backwardOutput[0], points, axis);
        return res;
    }
};

static const auto gRegister = []() {
    static ConcatGrad _c;
    OpGrad::insert((int)OpType_Concat, &_c);
    return true;
}();
