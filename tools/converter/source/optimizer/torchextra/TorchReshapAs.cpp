//
//  TorchReshapeAs.cpp
//  MNNConverter
//
//  Created by MNN on 2022/04/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNN_generated.h"
#include "TorchExtraManager.hpp"
#include "logkit.h"

namespace MNN {
namespace Express {

class TorchReshapeAsTransform : public TorchExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs = expr->inputs();
        auto op     = expr->get();
        auto opName = op->name()->str();
        std::unique_ptr<OpT> reshape(new OpT);
        reshape->type       = OpType_Reshape;
        reshape->main.type  = OpParameter_Reshape;
        reshape->main.value = new ReshapeT;
        auto input = inputs[0];
        auto other = inputs[1];
        auto shape = _Shape(other);
        std::vector<VARP> newInputs{input, shape};
        auto res = Expr::create(reshape.get(), newInputs);
        res->setName(opName);
        return res;
    }
};

class TorchBroadcaseAsTransform : public TorchExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs = expr->inputs();
        auto op     = expr->get();
        auto opName = op->name()->str();
        std::unique_ptr<OpT> broadcast(new OpT);
        broadcast->type       = OpType_BroadcastTo;
        broadcast->main.type  = OpParameter_NONE;
        broadcast->main.value = nullptr;
        auto input = inputs[0];
        auto other = inputs[1];
        auto shape = _Shape(other);
        std::vector<VARP> newInputs{input, shape};
        auto res = Expr::create(broadcast.get(), newInputs);
        res->setName(opName);
        return res;
    }
};

static auto gRegister = []() {
    TorchExtraManager::get()->insert("reshape_as", std::shared_ptr<TorchExtraManager::Transform>(new TorchReshapeAsTransform));
    TorchExtraManager::get()->insert("broadcast_as", std::shared_ptr<TorchExtraManager::Transform>(new TorchBroadcaseAsTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
