//
//  OnnxClip.cpp
//  MNNConverter
//
//  Created by MNN on 2020/06/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <limits>
#include "MNN_generated.h"
#include "OnnxExtraManager.hpp"
namespace MNN {
namespace Express {

class OnnxClipTransform : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs     = expr->inputs();
        auto op         = expr->get();
        auto extraParam = op->main_as_Extra();
        float maxValue  = std::numeric_limits<float>().max();
        float minValue  = -std::numeric_limits<float>().max();
        bool setReady   = false;
        if (nullptr != extraParam->attr()) {
            const int attrSize = extraParam->attr()->size();
            for (int i = 0; i < attrSize; ++i) {
                auto attr       = extraParam->attr()->GetAs<Attribute>(i);
                const auto& key = attr->key()->str();
                if (key == "max") {
                    maxValue = attr->f();
                    setReady = true;
                } else if (key == "min") {
                    minValue = attr->f();
                    setReady = true;
                }
            }
        }
        if (inputs.size() == 2 && (!setReady)) {
            auto minPtr = inputs[1]->readMap<float>();
            if (nullptr == minPtr) {
                return nullptr;
            }
            minValue    = minPtr[0];
        }
        if (inputs.size() >= 3 && (!setReady)) {
            auto minPtr = inputs[1]->readMap<float>();
            if (nullptr == minPtr) {
                return nullptr;
            }
            minValue    = minPtr[0];
            auto maxPtr = inputs[2]->readMap<float>();
            if (nullptr == maxPtr) {
                return nullptr;
            }
            maxValue = maxPtr[0];
        }
        std::unique_ptr<OpT> newOp(new OpT);
        newOp->type                     = OpType_ReLU6;
        newOp->main.type                = OpParameter_Relu6;
        newOp->main.value               = new Relu6T;
        newOp->main.AsRelu6()->maxValue = maxValue;
        newOp->main.AsRelu6()->minValue = minValue;
        auto res = Expr::create(newOp.get(), {inputs[0]});
        res->setName(expr->name());
        return res;
    }
};

static auto gRegister = []() {
    OnnxExtraManager::get()->insert("Clip", std::shared_ptr<OnnxExtraManager::Transform>(new OnnxClipTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
