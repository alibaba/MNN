//
//  SliceGrad.cpp
//  MNN
//
//  Created by MNN on 2019/12/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpGrad.hpp"
using namespace std;
using namespace MNN;
using namespace MNN::Express;

class SliceGrad : public OpGrad {
public:
    virtual std::vector<Express::VARP> onGrad(Express::EXPRP expr,
                                              const std::vector<Express::VARP>& backwardOutput) override {
        MNN_ASSERT(expr->inputs().size() == 1);
        auto slice = expr->get()->main_as_Slice();
        auto axis  = slice->axis();
        std::vector<VARP> res{nullptr};
        std::vector<VARP> validBackward(backwardOutput.size());
        for (int i = 0; i < backwardOutput.size(); ++i) {
            auto origin = Variable::create(expr, i);
            if (nullptr != backwardOutput[i]) {
                validBackward[i] = backwardOutput[i];
                continue;
            }
            auto info = origin->getInfo();
            if (nullptr == info) {
                MNN_ERROR("Error for sliceGrad's %d output\n", i);
                return res;
            }
            validBackward[i] = _Const(0.0f, info->dim, info->order);
        }
        res[0] = _Concat(validBackward, axis);
        // FUNC_PRINT_ALL(_Sum(res[0], {})->readMap<float>()[0], f);
        return res;
    }
};

static const auto gRegister = []() {
    static SliceGrad _c;
    OpGrad::insert((int)OpType_Slice, &_c);
    return true;
}();
