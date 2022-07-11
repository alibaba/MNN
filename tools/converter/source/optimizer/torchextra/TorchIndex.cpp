//
//  TorchIndex.cpp
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

class TorchSelectTransform : public TorchExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs = expr->inputs();
        auto op     = expr->get();
        auto opName = op->name()->str();
        MNN_ASSERT(inputs.size() == 3);
        auto input = inputs[0];
        auto axis = inputs[1];
        auto index = inputs[2];
        auto output = _GatherV2(input, _Squeeze(index), axis);
        output->setName(opName);
        return output->expr().first;
    }
};

class TorchIndexTransform : public TorchExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs = expr->inputs();
        auto op     = expr->get();
        auto opName = op->name()->str();
        MNN_ASSERT(inputs.size() == 2);
        auto input = inputs[0];
        auto mask = inputs[1];
        auto index = _GatherND(input, _Where(mask));
        index->setName(opName);
        return index->expr().first;
    }
};

class TorchIndexStridedSliceTransform : public TorchExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        // TODO
        auto inputs = expr->inputs();
        auto op     = expr->get();
        auto opName = op->name()->str();
        MNN_ASSERT(inputs.size() == 2);
        auto input = inputs[0];
        auto mask = inputs[1];
        auto index = _GatherND(input, _Where(mask));
        index->setName(opName);
        return index->expr().first;
    }
};

class TorchIndexPutTransform : public TorchExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs = expr->inputs();
        auto op     = expr->get();
        auto opName = op->name()->str();
        MNN_ASSERT(inputs.size() >= 3);
        auto input = inputs[0];
        auto mask = inputs[1];
        auto value = inputs[2];
        auto idx = _Where(mask);
        auto index_put = _ScatterNd(idx, value, _Shape(input), input);
        index_put->setName(opName);
        return index_put->expr().first;
    }
};

static auto gRegister = []() {
    TorchExtraManager::get()->insert("select", std::shared_ptr<TorchExtraManager::Transform>(new TorchSelectTransform));
    TorchExtraManager::get()->insert("index_select", std::shared_ptr<TorchExtraManager::Transform>(new TorchSelectTransform));
    TorchExtraManager::get()->insert("index", std::shared_ptr<TorchExtraManager::Transform>(new TorchIndexTransform));
    TorchExtraManager::get()->insert("index_stridedslice", std::shared_ptr<TorchExtraManager::Transform>(new TorchIndexStridedSliceTransform));
    TorchExtraManager::get()->insert("index_put", std::shared_ptr<TorchExtraManager::Transform>(new TorchIndexPutTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
