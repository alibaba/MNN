//
//  TFArgMaxMerge.cpp
//  MNNConverter
//
//  Created by MNN on 2019/09/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNN_generated.h"
#include "TFExtraManager.hpp"

namespace MNN {
namespace Express {
class ArgMaxTransform : public TFExtraManager::Transform {
public:
    enum ArgMinOrMax { ARGMIN, ARGMAX };
    ArgMaxTransform(ArgMinOrMax mode) : mMode(mode) {
    }
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs                 = expr->inputs();
        auto op                     = expr->get();
        std::vector<VARP> subInputs = {inputs[0]};
        std::unique_ptr<MNN::OpT> ArgMaxOp(new OpT);
        if (mMode == ARGMIN) {
            ArgMaxOp->type = OpType_ArgMin;
        } else {
            ArgMaxOp->type = OpType_ArgMax;
        }
        ArgMaxOp->name       = op->name()->str();
        ArgMaxOp->main.type  = OpParameter_ArgMax;
        ArgMaxOp->main.value = new ArgMaxT;
        auto ArgMaxParameter = ArgMaxOp->main.AsArgMax();
        {
            auto ArgMaxPoint     = inputs[1];
            auto ArgMaxPointInfo = ArgMaxPoint->getInfo();
            auto ArgMaxPointPtr  = ArgMaxPoint->readMap<int32_t>();
            if (nullptr == ArgMaxPointInfo || nullptr == ArgMaxPointPtr) {
                MNN_ERROR("Don't support not const ArgMax point\n");
                return nullptr;
            }
            ArgMaxParameter->axis = ArgMaxPointPtr[0];
        }
        auto newExpr = Expr::create(ArgMaxOp.get(), subInputs, expr->outputSize());
        return newExpr;
    }

private:
    ArgMinOrMax mMode;
};
static auto gRegister = []() {
    TFExtraManager::get()->insert("ArgMin", std::shared_ptr<TFExtraManager::Transform>(
                                                new ArgMaxTransform(ArgMaxTransform::ArgMinOrMax::ARGMIN)));
    TFExtraManager::get()->insert("ArgMax", std::shared_ptr<TFExtraManager::Transform>(
                                                new ArgMaxTransform(ArgMaxTransform::ArgMinOrMax::ARGMAX)));
    return true;
}();
} // namespace Express
} // namespace MNN
