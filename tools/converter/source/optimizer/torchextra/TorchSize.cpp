//
//  TorchSize.cpp
//  MNNConverter
//
//  Created by MNN on 2021/08/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNN_generated.h"
#include "TorchExtraManager.hpp"
#include "logkit.h"

namespace MNN {
namespace Express {

class TorchSizeTransform : public TorchExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs = expr->inputs();
        auto op     = expr->get();
        auto opName = op->name()->str();
        MNN_ASSERT(inputs.size() == 1 || inputs.size() == 2);
        auto shape = _Shape(inputs[0], true);
        // shape
        if (inputs.size() == 1) {
            shape->setName(opName);
            return shape->expr().first;
        }
        // size
        auto index =  _Add(inputs[1], _Multiply(_Rank(inputs[0]), _Less(inputs[1], _Scalar(0))));
        auto gather = _GatherV2(shape, index, _Scalar(0));
        gather->setName(opName);
        return gather->expr().first;
    }
};

static auto gRegister = []() {
    TorchExtraManager::get()->insert("size", std::shared_ptr<TorchExtraManager::Transform>(new TorchSizeTransform));
    return true;
}();
} // namespace Express
} // namespace MNN
