//
//  TFliteExtraManager.cpp
//  MNNConverter
//
//  Created by MNN on 2020/03/21.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TFliteExtraManager.hpp"
#include "OpCount.hpp"
#include "MNN_generated.h"
namespace MNN {
namespace Express {
std::shared_ptr<TFliteExtraManager> TFliteExtraManager::gInstance;
std::shared_ptr<TFliteExtraManager> TFliteExtraManager::get() {
    if (nullptr == gInstance) {
        gInstance.reset(new TFliteExtraManager);
    }
    return gInstance;
}

void TFliteExtraManager::insert(const std::string& name, std::shared_ptr<Transform> transform) {
    mTransform.insert(std::make_pair(name, transform));
}
std::shared_ptr<TFliteExtraManager::Transform> TFliteExtraManager::find(const std::string& name) const {
    auto iter = mTransform.find(name);
    if (iter == mTransform.end()) {
        return nullptr;
    }
    OpCount::get()->insertOp("TFLITE", name);
    return iter->second;
}

static auto gRegister = []() {
    auto extra = TFliteExtraManager::get();
    auto judge = [extra](EXPRP expr) {
        auto op = expr->get();
        if (nullptr == op || op->type() != OpType_Extra) {
            return false;
        }
        auto engine = op->main_as_Extra()->engine()->str();
        if (engine != "Tflite") {
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
            MNN_ERROR("Converte Tflite's Op %s , type = %s, failed, may be some node is not const\n",
                      expr->name().c_str(), type.c_str());
            return false;
        }
        newExpr->setName(expr->name());
        Expr::replace(expr, newExpr);
        return true;
    };
    TemplateMerge::getInstance("TFliteExtra").insertTemplate("TFliteExtraManager", judge, modify);
    return true;
}();
} // namespace Express
} // namespace MNN
