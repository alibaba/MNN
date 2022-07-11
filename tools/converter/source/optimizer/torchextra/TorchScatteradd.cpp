//
//  TorchScatteradd.cpp
//  MNNConverter
//
//  Created by MNN on 2022/06/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNN_generated.h"
#include "TorchExtraManager.hpp"
#include "logkit.h"

namespace MNN {
namespace Express {

class TorchScatteraddTransform : public TorchExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs = expr->inputs();
        auto op     = expr->get();
        auto opName = op->name()->str();
        auto input = inputs[0];
        auto index = inputs[1];
        auto src  = inputs[2];
        int axis = 0;
        auto info = op->main_as_Extra();
        if (nullptr != info->attr() && info->attr()->size() == 1) {
            const auto attr = info->attr()->GetAs<Attribute>(0);
            axis = attr->i();
        }
        auto res = input + _ScatterElements(_ZerosLike(input), index, src, axis);
        res->setName(opName);
        return res->expr().first;
    }
};

static auto gRegister = []() {
    TorchExtraManager::get()->insert("scatter_add", std::shared_ptr<TorchExtraManager::Transform>(new TorchScatteraddTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
