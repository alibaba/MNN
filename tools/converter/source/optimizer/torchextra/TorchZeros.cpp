//
//  TorchZeros.cpp
//  MNNConverter
//
//  Created by MNN on 2022/04/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNN_generated.h"
#include "TorchExtraManager.hpp"
#include "logkit.h"

namespace MNN {
namespace Express {

class TorchZerosTransform : public TorchExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs = expr->inputs();
        auto op     = expr->get();
        auto opName = op->name()->str();
        MNN_ASSERT(inputs.size() == 1);
        auto zeros = _Fill(inputs[0], _Const(0.));
        zeros->setName(opName);
        return zeros->expr().first;
    }
};

class TorchOnesTransform : public TorchExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs = expr->inputs();
        auto op     = expr->get();
        auto opName = op->name()->str();
        MNN_ASSERT(inputs.size() == 1);
        auto zeros = _Fill(inputs[0], _Const(1.));
        zeros->setName(opName);
        return zeros->expr().first;
    }
};

class TorchOnesLikeTransform : public TorchExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs = expr->inputs();
        auto op     = expr->get();
        auto opName = op->name()->str();
        MNN_ASSERT(inputs.size() == 1);
        auto zeros = _Fill(_Shape(inputs[0]), _Const(1.));
        zeros->setName(opName);
        return zeros->expr().first;
    }
};

class TorchFullLikeTransform : public TorchExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs = expr->inputs();
        auto op     = expr->get();
        auto opName = op->name()->str();
        MNN_ASSERT(inputs.size() == 2);
        auto full = _Fill(_Shape(inputs[0]), inputs[1]);
        full->setName(opName);
        return full->expr().first;
    }
};

static auto gRegister = []() {
    TorchExtraManager::get()->insert("zeros", std::shared_ptr<TorchExtraManager::Transform>(new TorchZerosTransform));
    TorchExtraManager::get()->insert("ones", std::shared_ptr<TorchExtraManager::Transform>(new TorchOnesTransform));
    TorchExtraManager::get()->insert("ones_like", std::shared_ptr<TorchExtraManager::Transform>(new TorchOnesLikeTransform));
    TorchExtraManager::get()->insert("full_like", std::shared_ptr<TorchExtraManager::Transform>(new TorchFullLikeTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
