//
//  TorchSilu.cpp
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

class TorchSiluTransform : public TorchExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs = expr->inputs();
        auto op     = expr->get();
        auto opName = op->name()->str();
        MNN_ASSERT(inputs.size() == 1);
        auto silu = inputs[0] * _Sigmoid(inputs[0]);
        silu->setName(opName);
        return silu->expr().first;
    }
};

static auto gRegister = []() {
    TorchExtraManager::get()->insert("silu", std::shared_ptr<TorchExtraManager::Transform>(new TorchSiluTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
