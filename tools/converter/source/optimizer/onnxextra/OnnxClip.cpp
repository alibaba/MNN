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

template<typename T> 
static EXPRP clipConvert(EXPRP expr) {
    auto inputs     = expr->inputs();
    auto op         = expr->get();
    auto extraParam = op->main_as_Extra();
    // auto dataType = expr->outputInfo(0)->type.code;
    auto maxValue  = std::numeric_limits<T>().max();
    auto minValue  = std::numeric_limits<T>().min();
    if (nullptr != extraParam->attr()) {
        const int attrSize = extraParam->attr()->size();
        for (int i = 0; i < attrSize; ++i) {
            auto attr       = extraParam->attr()->GetAs<Attribute>(i);
            const auto& key = attr->key()->str();
            if (key == "max") {
                maxValue = attr->f();
            } else if (key == "min") {
                minValue = attr->f();
            }
        }
    }
    bool unknown_min_max = false;
    if (inputs.size() == 2 || (inputs.size() == 3 && inputs[1].get() != nullptr)) {
        auto minPtr = inputs[1]->readMap<T>();
        if (nullptr == minPtr) {
            unknown_min_max = true;
        } else {
            minValue = minPtr[0];
        }
    }
    if (inputs.size() == 3 && !unknown_min_max) {
        auto maxPtr = inputs[2]->readMap<T>();
        if (nullptr == maxPtr) {
            unknown_min_max = true;
        } else {
            maxValue = maxPtr[0];
        }
    }
    if (unknown_min_max) {
        auto minVar = _Scalar<T>(minValue);
        auto maxVar = _Scalar<T>(maxValue);
        if (inputs.size() >= 2 && inputs[1].get() != nullptr) {
            minVar = inputs[1];
        }
        if (inputs.size() >= 3) {
            maxVar = inputs[2];
        }
        auto res = _Minimum(_Maximum(inputs[0], minVar), maxVar);
        auto newExpr = res->expr().first;
        newExpr->setName(expr->name());
        return newExpr;
    }
    if(maxValue > std::numeric_limits<T>::max()) {
        maxValue = std::numeric_limits<T>().max();
    }
    if(minValue < std::numeric_limits<T>::lowest()) {
        minValue = std::numeric_limits<T>().lowest();
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

class OnnxClipTransform : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs = expr->inputs();
        halide_type_code_t type;
        for (int i = 0; i < inputs.size(); ++i) {
            if (nullptr != inputs[i] && nullptr != inputs[i]->getInfo()) {
                type = static_cast<halide_type_code_t>(inputs[i]->getInfo()->type.code);
                break;
            }
        }
        if (type == halide_type_float) {
            return clipConvert<float>(expr);
        } else {
            return clipConvert<int32_t>(expr);
        }
    }
};

static auto gRegister = []() {
    OnnxExtraManager::get()->insert("Clip", std::shared_ptr<OnnxExtraManager::Transform>(new OnnxClipTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
