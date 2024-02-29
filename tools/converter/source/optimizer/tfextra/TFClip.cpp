//
//  TFClip.cpp
//  MNNConverter
//
//  Created by MNN on 2020/07/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNN_generated.h"
#include "TFExtraManager.hpp"

namespace MNN {
namespace Express {
class ClipTransform : public TFExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs = expr->inputs();
        if (inputs.size() != 3) {
            MNN_ERROR("ClipByValue should has 3 inputs\n");
            return nullptr;
        }
        float minValue = 0.0f;
        float maxValue = 0.0f;
        {
            auto minInfo = inputs[1]->getInfo();
            auto maxInfo = inputs[2]->getInfo();
            auto minP    = inputs[1]->readMap<float>();
            auto maxP    = inputs[2]->readMap<float>();
            if (nullptr == minP || nullptr == maxP || 1 != minInfo->size || 1 != maxInfo->size) {
                // Not const clip op, use max(min) instead
                auto sameVar = _Minimum(_Maximum(inputs[0], inputs[1]), inputs[2]);
                sameVar->setName(expr->name());
                return sameVar->expr().first;
            }
            minValue = *minP;
            maxValue = *maxP;
        }
        if(maxValue > std::numeric_limits<float>::max()) {
            maxValue = std::numeric_limits<float>().max();
        }
        if(minValue < std::numeric_limits<float>::lowest()) {
            minValue = std::numeric_limits<float>().lowest();
        }
        std::vector<VARP> subInputs = {inputs[0]};
        std::unique_ptr<MNN::OpT> clipOp(new OpT);
        clipOp->type       = OpType_ReLU6;
        clipOp->main.type  = OpParameter_Relu6;
        clipOp->main.value = new Relu6T;
        auto param         = clipOp->main.AsRelu6();
        {
            param->maxValue = maxValue;
            param->minValue = minValue;
        }
        auto newExpr = Expr::create(clipOp.get(), subInputs, 1);
        newExpr->setName(expr->name());
        return newExpr;
    }
};
static auto gRegister = []() {
    TFExtraManager::get()->insert("ClipByValue", std::shared_ptr<TFExtraManager::Transform>(new ClipTransform));
    return true;
}();
} // namespace Express
} // namespace MNN
