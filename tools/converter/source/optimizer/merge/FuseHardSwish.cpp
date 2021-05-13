//
//  FuseHardSwish.cpp
//  MNNConverter
//
//  Created by MNN on 2021/04/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "../TemplateMerge.hpp"
#include "MNN/expr/MathOp.hpp"
#include "MNN/expr/NeuralNetWorkOp.hpp"
#include "MNN_generated.h"
#include "../../common/Global.hpp"
#include "config.hpp"
namespace MNN {
namespace Express {
// left is pattern, right is dest
static bool isSameOp(const MNN::Op* op0, const MNN::Op* op1) {
    if (op0->type() != op1->type()) {
        return false;
    }
    if (op0->main_type() != op1->main_type()) {
        return false;
    }
    if (op0->main_type() == OpParameter_NONE) {
        return true;
    }
    if (op0->type() == OpType_ReLU) {
        return op0->main_as_Relu()->slope() == op1->main_as_Relu()->slope();
    }
    if (op0->type() == OpType_ReLU6) {
        return op0->main_as_Relu6()->maxValue() == op1->main_as_Relu6()->maxValue() && op0->main_as_Relu6()->minValue() == op1->main_as_Relu6()->minValue();
    }
    if (op0->main_type() == OpParameter_UnaryOp) {
        return op0->main_as_UnaryOp()->opType() == op1->main_as_UnaryOp()->opType();
    }
    if (op0->main_type() == OpParameter_BinaryOp) {
        return op0->main_as_BinaryOp()->opType() == op1->main_as_BinaryOp()->opType();
    }
    return false;
}
static bool isTheSame(EXPRP left, EXPRP right) {
    auto lop = left->get();
    auto rop = right->get();
    if (nullptr == lop) {
        if (left->inputType() == VARP::INPUT) {
            return true;
        }
        if (nullptr != rop) {
            return false;
        }
    }
    if (left->inputs().size() != right->inputs().size()) {
        return false;
    }
    if (left->outputSize() != right->outputSize()) {
        return false;
    }
    if (nullptr != lop && nullptr == rop) {
        return false;
    }
    if (nullptr == lop && nullptr == rop) {
        // Constant
        if (left->inputType() != right->inputType()) {
            return false;
        }
        auto leftV = Variable::create(left);
        auto rightV = Variable::create(right);
        if (leftV->getInfo() == nullptr || rightV->getInfo() == nullptr) {
            return false;
        }
        auto lInfo = leftV->getInfo();
        auto rInfo = rightV->getInfo();
        if (lInfo->type != rInfo->type) {
            return false;
        }
        if (lInfo->dim != rInfo->dim) {
            return false;
        }
        if (lInfo->size != rInfo->size) {
            return false;
        }
        auto lPtr = leftV->readMap<void>();
        auto rPtr = rightV->readMap<void>();
        if (nullptr == lPtr || nullptr == rPtr) {
            return false;
        }
        if (0 != ::memcmp(lPtr, rPtr, lInfo->size * lInfo->type.bytes())) {
            return false;
        }
        return true;
    }
    // Check Op
    if (!isSameOp(lop, rop)) {
        return false;
    }
    for (int i=0; i<left->inputs().size(); ++i) {
        if (!isTheSame(left->inputs()[i]->expr().first, right->inputs()[i]->expr().first)) {
            return false;
        }
    }
    return true;
}

static auto gRegister = []() {
    auto input = _Input({}, NCHW);
    auto left = _Relu6(_Add(input, _Scalar<float>(3)));
    auto res = _Divide(_Multiply(input, left), _Scalar<float>(6));
    auto match = [res](EXPRP expr) {
        auto config = Global<modelConfig>::Get();
        auto version = config->targetVersion;
        if (version < 1.2f) {
            // For target version < 1.2 , don't support hardswish
            return false;
        }
        return isTheSame(res->expr().first, expr);
    };

    auto transform = [](EXPRP expr) {
        auto inputVar = expr->inputs()[0]->expr().first->inputs()[0];
        std::unique_ptr<MNN::OpT> newOp(new OpT);
        newOp->type = OpType_UnaryOp;
        newOp->main.value = new UnaryOpT;
        newOp->main.type = OpParameter_UnaryOp;
        newOp->main.AsUnaryOp()->opType = UnaryOpOperation_HARDSWISH;
        auto newVar = Variable::create(Expr::create(newOp.get(), {inputVar}, 1));
        newVar->setName(expr->outputName(0));
        Expr::replace(expr, newVar->expr().first);
        return true;
    };

    TemplateMerge::getInstance("Merge").insertTemplate("FuseHardSwish", match, transform);
    return true;
}();

}
} // namespace MNN
