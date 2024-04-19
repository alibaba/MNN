//
//  OnnxOneHot.cpp
//  MNNConverter
//
//  Created by MNN on 2021/04/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <limits>
#include "MNN_generated.h"
#include "OnnxExtraManager.hpp"
namespace MNN {
namespace Express {

class OnnxOneHotTransform : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs     = expr->inputs();
        auto op         = expr->get();
        auto extraParam = op->main_as_Extra();
        int axis = 0;
        if (nullptr != extraParam->attr()) {
            const int attrSize = extraParam->attr()->size();
            for (int i = 0; i < attrSize; ++i) {
                auto attr       = extraParam->attr()->GetAs<Attribute>(i);
                const auto& key = attr->key()->str();
                if (key == "axis") {
                    axis = attr->i();
                }
            }
        }
        if (inputs.size() != 3) {
            MNN_ERROR("Don't support onehot for inputs != 3\n");
            return nullptr;
        }
        auto onOff = _Split(inputs[2], std::vector<int>{2}, 0);
        auto res = _OneHot(inputs[0], inputs[1], onOff[1], onOff[0], axis);
        res->setName(expr->name());
        return res->expr().first;
    }
};

static auto gRegister = []() {
    OnnxExtraManager::get()->insert("OneHot", std::shared_ptr<OnnxExtraManager::Transform>(new OnnxOneHotTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
