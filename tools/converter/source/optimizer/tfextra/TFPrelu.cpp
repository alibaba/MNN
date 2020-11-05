//
//  TFPrelu.cpp
//  MNNConverter
//
//  Created by MNN on 2019/09/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNN_generated.h"
#include "TFExtraManager.hpp"

namespace MNN {
namespace Express {
class PreluTransform : public TFExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs                 = expr->inputs();
        auto op                     = expr->get();
        std::vector<VARP> subInputs = {inputs[0]};
        std::unique_ptr<MNN::OpT> PreluOp(new OpT);
        PreluOp->type       = OpType_PReLU;
        PreluOp->name       = op->name()->str();
        PreluOp->main.type  = OpParameter_PRelu;
        PreluOp->main.value = new PReluT;
        auto PreluParameter = PreluOp->main.AsPRelu();
        {
            auto PreluPoint     = inputs[1];
            auto PreluPointInfo = PreluPoint->getInfo();
            auto PreluPointPtr  = PreluPoint->readMap<float>();
            if (nullptr == PreluPointInfo || nullptr == PreluPointPtr) {
                MNN_ERROR("Don't support not const Prelu point\n");
                return nullptr;
            }
            PreluParameter->slope.resize(PreluPointInfo->size);
            ::memcpy(PreluParameter->slope.data(), PreluPointPtr, PreluPointInfo->size * sizeof(float));
            PreluParameter->slopeCount = PreluPointInfo->size;
        }
        auto newExpr = Expr::create(PreluOp.get(), subInputs, expr->outputSize());
        return newExpr;
    }
};
static auto gRegister = []() {
    TFExtraManager::get()->insert("Prelu", std::shared_ptr<TFExtraManager::Transform>(new PreluTransform));
    return true;
}();
} // namespace Express
} // namespace MNN
