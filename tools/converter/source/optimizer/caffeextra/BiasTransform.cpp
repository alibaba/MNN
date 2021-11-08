//
//  BiasTransform.cpp
//  MNNConverter
//
//  Created by MNN on 2019/12/13.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <vector>
#include "CaffeExtraManager.hpp"
#include "MNN_generated.h"

namespace MNN {
namespace Express {

class BiasTransform : public CaffeExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op      = expr->get();
        auto inputs  = expr->inputs();
        auto axis    = op->main_as_Extra()->attr()->GetAs<Attribute>(0)->i();
        auto numAxes = op->main_as_Extra()->attr()->GetAs<Attribute>(1)->i();

        if (inputs.size() == 1) {
            std::vector<int> biasShape{1, 1, 1, 1};
            auto shape = op->main_as_Extra()->attr()->GetAs<Attribute>(2)->tensor()->dims();
            for (int i = 0; i < shape->size(); i++) {
                biasShape[axis + i] = shape->data()[i];
            }
            auto biasData = op->main_as_Extra()->attr()->GetAs<Attribute>(2)->tensor()->float32s()->data();
            auto newVar   = _Add(inputs[0], _Const(biasData, biasShape, NCHW));
            return newVar->expr().first;
        } else {
            MNN_ASSERT(inputs.size() == 2);

            std::vector<int> biasShape{1, 1, 1, 1};
            auto shape = inputs[1]->getInfo()->dim;
            for (int i = 0; i < shape.size(); i++) {
                biasShape[axis + i] = shape[i];
            }
            auto newVar = _Add(inputs[0], _Const(inputs[1]->readMap<void>(), biasShape, NCHW));
            return newVar->expr().first;
        }
    }
};

static auto gRegister = []() {
    CaffeExtraManager::get()->insert("Bias", std::shared_ptr<CaffeExtraManager::Transform>(new BiasTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
