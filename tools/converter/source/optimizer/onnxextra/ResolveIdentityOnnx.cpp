//
//  ResolveIdentityOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2019/10/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNN_generated.h"
#include "OnnxExtraManager.hpp"

namespace MNN {
namespace Express {

class ResolveIdentityOnnx : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs = expr->inputs();
        MNN_CHECK(inputs.size() == 1, "Identity Should have one input");

        auto outputs = expr->outputs();
        MNN_CHECK(outputs.size() == 1, "Identity Should have one output");
        auto outputVaribale = outputs.front();

        auto outputExpr   = outputVaribale.lock();
        auto outputExprOp = outputExpr->get();

        std::unique_ptr<OpT> newOp(new OpT);
        newOp->name = outputExprOp->name()->str();
        newOp->type       = outputExprOp->type();
        newOp->main.type  = outputExprOp->main_type();
        newOp->main.value = const_cast<void*>(outputExprOp->main());

        auto outputExprInputs = outputExpr->inputs();

        // find the matched input, then replace it
        const int size = outputExprInputs.size();
        for (int i = 0; i < size; ++i) {
            if (outputExprInputs[i]->expr().first.get() == outputExpr.get()) {
                outputExprInputs[i] = inputs[0];
                break;
            }
        }
        return Expr::create(newOp.get(), outputExprInputs);
    }
};

static auto gRegister = []() {
    OnnxExtraManager::get()->insert("Dropout", std::shared_ptr<OnnxExtraManager::Transform>(new ResolveIdentityOnnx));

    OnnxExtraManager::get()->insert("Identity", std::shared_ptr<OnnxExtraManager::Transform>(new ResolveIdentityOnnx));
    return true;
}();

} // namespace Express
} // namespace MNN
