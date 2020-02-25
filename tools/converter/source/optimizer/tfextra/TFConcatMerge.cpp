//
//  TFConcatMerge.cpp
//  MNNConverter
//
//  Created by MNN on 2019/09/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNN_generated.h"
#include "TFExtraManager.hpp"

namespace MNN {
namespace Express {
class ConcatTransform : public TFExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        MNN_ASSERT(op->type() == OpType_Extra);
        auto type   = op->main_as_Extra()->type()->str();
        auto inputs = expr->inputs();
        MNN_ASSERT(inputs.size() > 1);
        auto axisNode = inputs[0];
        std::vector<VARP> subInputs;
        if (type == "ConcatV2") {
            axisNode = inputs[inputs.size() - 1];
            for (int i = 0; i < inputs.size() - 1; ++i) {
                subInputs.emplace_back(inputs[i]);
            }
        } else if (type == "Concat") {
            for (int i = 0; i < inputs.size() - 1; ++i) {
                subInputs.emplace_back(inputs[i]);
            }
        } else {
            for (int i = 0; i < inputs.size(); ++i) {
                subInputs.emplace_back(inputs[i]);
            }
        }

        const int* axisPtr = nullptr;
        if (type != "ParallelConcat") {
            axisPtr = axisNode->readMap<int32_t>();
            if (nullptr == axisPtr) {
                MNN_ERROR("Don't Support Axis not const for concat\n");
                return nullptr;
            }
        }

        std::unique_ptr<OpT> newOp(new OpT);
        newOp->name       = op->name()->str();
        newOp->type       = OpType_Concat;
        newOp->main.type  = OpParameter_Axis;
        newOp->main.value = new AxisT;
        if (type == "ParallelConcat") {
            newOp->main.AsAxis()->axis = 0;
        } else {
            newOp->main.AsAxis()->axis = axisPtr[0];
        }
        auto newExpr = Expr::create(newOp.get(), subInputs, 1);
        return newExpr;
    }
};
static auto gRegister = []() {
    TFExtraManager::get()->insert("ConcatV2", std::shared_ptr<TFExtraManager::Transform>(new ConcatTransform));
    TFExtraManager::get()->insert("Concat", std::shared_ptr<TFExtraManager::Transform>(new ConcatTransform));
    TFExtraManager::get()->insert("ParallelConcat", std::shared_ptr<TFExtraManager::Transform>(new ConcatTransform));
    return true;
}();
} // namespace Express
} // namespace MNN
