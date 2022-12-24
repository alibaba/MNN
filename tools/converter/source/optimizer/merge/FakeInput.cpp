//
//  FakeInput.cpp
//  MNNConverter
//
//  Created by MNN on 2022/12/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "../TemplateMerge.hpp"
#include "MNN/expr/MathOp.hpp"
#include "MNN/expr/NeuralNetWorkOp.hpp"
#include "MNN_generated.h"

namespace MNN {
namespace Express {

static auto gRegister = []() {
    auto match = [](EXPRP expr) {
        if (nullptr == expr->get()) {
            return false;
        }
        switch (expr->get()->type()) {
            case OpType_RandomNormal:
            case OpType_RandomUniform:
                return expr->inputs().size() == 1;
            default:
                return false;
        }
    };

    auto transform = [](EXPRP expr) {
        const_cast<std::vector<VARP>&>(expr->inputs()).push_back(_Input({1}));
        return true;
    };

    TemplateMerge::getInstance("Merge").insertTemplate("AddFakeInput", match, transform, PASS_PRIORITY_FRONT);
    return true;
}();

static auto gRegister2 = []() {
    auto match = [](EXPRP expr) {
        if (nullptr == expr->get()) {
            return false;
        }
        switch (expr->get()->type()) {
            case OpType_RandomNormal:
            case OpType_RandomUniform:
                return expr->inputs().size() == 2;
            default:
                return false;
        }
    };

    auto transform = [](EXPRP expr) {
        const_cast<std::vector<VARP>&>(expr->inputs()).pop_back();
        return true;
    };

    TemplateMerge::getInstance("Merge").insertTemplate("RemoveFakeInput", match, transform, PASS_PRIORITY_FINAL);
    return true;
}();
}
} // namespace MNN
