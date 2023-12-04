//
//  SelectGrad.cpp
//  MNN
//
//  Created by MNN on 2019/05/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "SelectGrad.hpp"
#include "core/Macro.h"
using namespace std;
using namespace MNN;
using namespace MNN::Express;

class SelectGrad : public OpGrad {
public:
    SelectGrad() {
        mType = SEMI_LINEAR;
    }
    virtual std::vector<Express::VARP> onGrad(Express::EXPRP expr,
                                              const std::vector<Express::VARP>& backwardOutput) override {
        auto inputs = expr->inputs();
        auto type = inputs[1]->getInfo()->type;
        std::vector<VARP> result(inputs.size(), nullptr);
        if (type.code != halide_type_float) {
            return result;
        }
        auto outputDiff = backwardOutput[0];
        // d (select(x, a, b)) = da * (x>0) + db * (x < 0)
        {
            auto zero = MNN::Express::_ZerosLike(expr->inputs()[1]);
            result[1] = MNN::Express::_Select(inputs[0], outputDiff, zero);
            result[2] = MNN::Express::_Select(inputs[0], zero, outputDiff);
        }
        return result;
    }
};

static const auto gRegister = []() {
    static SelectGrad _c;
    OpGrad::insert(OpType_Select, &_c);
    return true;
}();
