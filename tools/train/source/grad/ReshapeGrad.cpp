//
//  ReshapeGrad.cpp
//  MNN
//
//  Created by MNN on 2019/04/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ReshapeGrad.hpp"
#include "core/Macro.h"
using namespace std;
using namespace MNN;
using namespace MNN::Express;

class ReshapeGrad : public OpGrad {
public:
    virtual std::vector<Express::VARP> onGrad(Express::EXPRP expr,
                                              const std::vector<Express::VARP>& backwardOutput) override {
        auto inputs = expr->inputs();
        std::vector<VARP> result(inputs.size(), nullptr);
        auto info = inputs[0]->getInfo();
        if (nullptr == info) {
            return {};
        }
        // Create Reshape Op
        result[0] = _Reshape(backwardOutput[0], _Const(info->dim.data(), {(int)info->dim.size()}, NCHW, halide_type_of<int32_t>()));
        result[0]->setName(expr->name() + "_Grad");
        return result;
    }
};

static const auto gRegister = []() {
    static ReshapeGrad _c;
    OpGrad::insert(OpType_Reshape, &_c);
    OpGrad::insert(OpType_Squeeze, &_c);
    OpGrad::insert(OpType_Unsqueeze, &_c);
    return true;
}();
