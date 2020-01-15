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
        std::vector<VARP> result(inputs.size(), nullptr);
        auto outputDiff = backwardOutput[0];
        // d (select(x, a, b)) = da * (x>0) + db * (x < 0)
        {
            // Cast x>0 -> float
            unique_ptr<OpT> mask(new OpT);
            mask->type                     = OpType_Cast;
            mask->main.type                = OpParameter_CastParam;
            mask->main.value               = new CastParamT;
            mask->main.AsCastParam()->dstT = DataType_DT_FLOAT;
            mask->main.AsCastParam()->srcT = DataType_DT_BOOL;

            auto maskVar = Variable::create(Expr::create(std::move(mask), {inputs[0]}));

            // da * (x>0)
            result[1] = _Multiply(outputDiff, maskVar);

            // db * -((x>0)-1)
            auto one  = _Const(1.0f);
            auto sub  = _Subtract(maskVar, one);
            auto neg  = _Negative(sub);
            result[2] = _Multiply(outputDiff, neg);
        }

        return result;
    }
};

static const auto gRegister = []() {
    static SelectGrad _c;
    OpGrad::insert(OpType_Select, &_c);
    return true;
}();
