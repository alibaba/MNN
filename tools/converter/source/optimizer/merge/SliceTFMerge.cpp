//
//  SliceTFMerge.cpp
//  MNNConverter
//
//  Created by MNN on 2019/09/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "../TemplateMerge.hpp"
#include "MNN_generated.h"

namespace MNN {
namespace Express {
static auto gRegister = []() {
    auto compare = [](EXPRP expr) {
        if (nullptr == expr->get()) {
            return false;
        }
        if (expr->get()->type() != OpType_SliceTf) {
            return false;
        }
        auto inputs    = expr->inputs();
        auto input     = inputs[0];
        auto begin     = inputs[1];
        auto size      = inputs[2];
        auto inputInfo = input->getInfo();
        if (nullptr == inputInfo) {
            return false;
        }
        auto beginPtr = begin->readMap<int>();
        auto endPtr   = size->readMap<int>();
        if (nullptr == beginPtr || nullptr == endPtr) {
            return false;
        }
        for (int i = 0; i < inputInfo->dim.size(); ++i) {
            if (beginPtr[i] > 0 || endPtr[i] != inputInfo->dim[i]) {
                return false;
            }
        }
        return true;
    };
    auto modify = [](EXPRP expr) {
        auto inputs = expr->inputs();
        Expr::replace(expr, inputs[0]->expr().first);
        return true;
    };
    //    TemplateMerge::getInstance("Merge").insertTemplate("SliceTFMerge", compare, modify);
    return true;
}();
}
} // namespace MNN
