//
//  TFExtraManager.cpp
//  MNNConverter
//
//  Created by MNN on 2019/09/21.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TFExtraManager.hpp"
#include "OpCount.hpp"
#include "MNN_generated.h"
namespace MNN {
namespace Express {
std::shared_ptr<TFExtraManager> TFExtraManager::gInstance;
std::shared_ptr<TFExtraManager> TFExtraManager::get() {
    if (nullptr == gInstance) {
        gInstance.reset(new TFExtraManager);
    }
    return gInstance;
}

void TFExtraManager::insert(const std::string& name, std::shared_ptr<Transform> transform) {
    OpCount::get()->insertOp("TF", name);
    mTransform.insert(std::make_pair(name, transform));
}
std::shared_ptr<TFExtraManager::Transform> TFExtraManager::find(const std::string& name) const {
    auto iter = mTransform.find(name);
    if (iter == mTransform.end()) {
        return nullptr;
    }
    return iter->second;
}

static auto gRegister = []() {
    auto extra = TFExtraManager::get();
    auto judge = [extra](EXPRP expr) {
        auto op = expr->get();
        if (nullptr == op || op->type() != OpType_Extra) {
            return false;
        }
        auto engine = op->main_as_Extra()->engine()->str();
        if (engine != "Tensorflow") {
            return false;
        }
        auto type = op->main_as_Extra()->type()->str();
        if (extra->find(type) == nullptr) {
            return false;
        }
        return true;
    };
    auto modify = [extra](EXPRP expr) {
        auto op = expr->get();
        MNN_ASSERT(op->type() == OpType_Extra);
        auto type        = op->main_as_Extra()->type()->str();
        auto transformer = extra->find(type);
        MNN_ASSERT(nullptr != transformer);
        auto newExpr = transformer->onExecute(expr);
        if (nullptr == newExpr) {
            MNN_ERROR("Converte Tensorflow's Op %s , type = %s, failed, may be some node is not const\n",
                      expr->name().c_str(), type.c_str());
            return false;
        }
        if (newExpr->name().empty()) {
            newExpr->setName(expr->name());
        }
        Expr::replace(expr, newExpr);
        return true;
    };
    TemplateMerge::getInstance("TFExtra").insertTemplate("TFExtraManager", judge, modify);
    return true;
}();
} // namespace Express
} // namespace MNN
