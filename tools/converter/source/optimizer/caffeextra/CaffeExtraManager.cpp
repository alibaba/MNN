//
//  CaffeExtraManager.cpp
//  MNNConverter
//
//  Created by MNN on 2019/12/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CaffeExtraManager.hpp"
#include "MNN_generated.h"
#include "OpCount.hpp"
namespace MNN {
namespace Express {
CaffeExtraManager* CaffeExtraManager::get() {
    static std::shared_ptr<CaffeExtraManager> gInstance(new CaffeExtraManager);
    return gInstance.get();
}

void CaffeExtraManager::insert(const std::string& name, std::shared_ptr<Transform> transform) {
    OpCount::get()->insertOp("CAFFE", name);
    mTransform.insert(std::make_pair(name, transform));
}
std::shared_ptr<CaffeExtraManager::Transform> CaffeExtraManager::find(const std::string& name) const {
    auto iter = mTransform.find(name);
    if (iter == mTransform.end()) {
        return nullptr;
    }
    return iter->second;
}

static auto gRegister = []() {
    auto extra = CaffeExtraManager::get();
    auto judge = [extra](EXPRP expr) {
        auto op = expr->get();
        if (op->type() != OpType_Extra) {
            return false;
        }
        auto engine = op->main_as_Extra()->engine()->str();
        if (engine != "Caffe") {
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
            MNN_ERROR("Converte Caffe's Op %s , type = %s, failed, may be some node is not const\n",
                      expr->name().c_str(), type.c_str());
            return false;
        }
        newExpr->setName(expr->name());
        Expr::replace(expr, newExpr);
        return true;
    };
    TemplateMerge::getInstance("CaffeExtra").insertTemplate("CaffeExtraManager", judge, modify);
    return true;
}();
} // namespace Express
} // namespace MNN
