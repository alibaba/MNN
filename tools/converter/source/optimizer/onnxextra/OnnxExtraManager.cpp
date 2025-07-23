//
//  OnnxExtraManager.cpp
//  MNNConverter
//
//  Created by MNN on 2019/09/29.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OnnxExtraManager.hpp"
#include "OpCount.hpp"
#include "MNN_generated.h"
namespace MNN {
namespace Express {
VARP OnnxExtraManager::_ReshapeF(VARP x, VARP shape, int format) {
    MNN_ASSERT(nullptr != x);
    std::unique_ptr<OpT> reshape(new OpT);
    reshape->type                      = OpType_Reshape;
    reshape->main.type                 = OpParameter_Reshape;
    reshape->main.value                = new ReshapeT;
    reshape->main.AsReshape()->dimType = (MNN_DATA_FORMAT)format;
    return (Variable::create(Expr::create(reshape.get(), {x, shape})));
}
std::shared_ptr<OnnxExtraManager> OnnxExtraManager::get() {
    static std::shared_ptr<OnnxExtraManager> gInstance;
    if (nullptr == gInstance) {
        gInstance.reset(new OnnxExtraManager);
    }
    return gInstance;
}

void OnnxExtraManager::insert(const std::string& name, std::shared_ptr<Transform> transform) {
    mTransform.insert(std::make_pair(name, transform));
    OpCount::get()->insertOp("ONNX", name);
}
std::shared_ptr<OnnxExtraManager::Transform> OnnxExtraManager::find(const std::string& name) const {
    auto iter = mTransform.find(name);
    if (iter == mTransform.end()) {
        return nullptr;
    }
    return iter->second;
}

static auto gRegister = []() {
    auto extra = OnnxExtraManager::get();
    auto judge = [extra](EXPRP expr) {
        auto op = expr->get();
        if (nullptr == op) {
            return false;
        }
        if (op->type() != OpType_Extra) {
            return false;
        }
        auto engine = op->main_as_Extra()->engine()->str();
        if (engine != "ONNX") {
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
            MNN_ERROR("Convert Onnx's Op %s , type = %s, failed, may be some node is not const\n", expr->name().c_str(),
                      type.c_str());
            return false;
        }
        if (expr->outputSize() == newExpr->outputSize()) {
            for (int i=0; i<expr->outputSize(); ++i) {
                Variable::create(newExpr, i)->setName(expr->outputName(i));
            }
        }
        Expr::replace(expr, newExpr);
        return true;
    };
    TemplateMerge::getInstance("OnnxExtra").insertTemplate("OnnxExtraManager", judge, modify);
    return true;
}();
} // namespace Express
} // namespace MNN
