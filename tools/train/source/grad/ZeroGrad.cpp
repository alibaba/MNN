//
//  ZeroGrad.cpp
//  MNN
//
//  Created by MNN on 2019/04/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ReluGrad.hpp"
#include "core/Macro.h"
using namespace std;
using namespace MNN;

class ZeroGrad : public OpGrad {
public:
    ZeroGrad() {
        mType = LINEAR;
    }
    virtual std::vector<Express::VARP> onGrad(Express::EXPRP expr,
                                              const std::vector<Express::VARP>& backwardOutput) override {
        std::vector<Express::VARP> result(1, nullptr);
        result[0] = backwardOutput[0];
        return result;
    }
};
static const auto gRegister = []() {
    static ZeroGrad _c;
    OpGrad::insert(OpType_ZeroGrad, &_c);
    return true;
}();
