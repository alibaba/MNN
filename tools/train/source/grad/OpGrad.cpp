//
//  OpGrad.cpp
//  MNN
//
//  Created by MNN on 2019/05/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpGrad.hpp"
using namespace std;
using namespace MNN::Express;
namespace MNN {
static std::map<int, OpGrad*>& getConverter() {
    static std::map<int, OpGrad*> gConverterMap;
    return gConverterMap;
}

OpGrad* OpGrad::get(int type) {
    auto& converterMap = getConverter();
    auto iter          = converterMap.find(type);
    if (iter != converterMap.end()) {
        return iter->second;
    }
    return nullptr;
}

void OpGrad::insert(int type, OpGrad* converter) {
    auto& converterMap = getConverter();
    converterMap.insert(std::make_pair(type, converter));
}

std::map<Express::VARP, Express::VARP> OpGrad::grad(VARP loss, const std::set<Express::VARP>& parameters) {
    std::map<EXPRP, std::vector<VARP>> backwardMap;
    {
        auto shape = loss->getInfo();
        MNN_ASSERT(shape->size == 1);
        auto init                       = _Const(1.0f, shape->dim, shape->order);
        backwardMap[loss->expr().first] = std::vector<VARP>{init};
    }
    auto executeOrder = Variable::getExecuteOrder({loss});
    for (auto iter = executeOrder.rbegin(); iter != executeOrder.rend(); iter++) {
        auto expr    = *iter;
        auto& inputs = expr->inputs();
        if (backwardMap.find(expr) == backwardMap.end()) {
            continue;
        }
        if (nullptr == expr->get()) {
            continue;
        }
        auto grad = OpGrad::get(expr->get()->type());
        if (nullptr == grad) {
            // MNN_PRINT("Can't grad for %s, %d\n", expr->name().c_str(), expr->get()->type());
            continue;
        }
        auto inputGrad = grad->onGrad(expr, backwardMap[expr]);
        auto empty     = true;
        for (auto grad : inputGrad) {
            if (nullptr != grad) {
                empty = false;
                break;
            }
        }
        if (empty) {
            // MNN_PRINT("Can't grad for %s, %d\n", expr->name().c_str(), expr->get()->type());
            continue;
        }
        MNN_ASSERT(inputGrad.size() <= inputs.size());
        for (int i = 0; i < inputGrad.size(); ++i) {
            auto inputExpr = inputs[i]->expr().first;
            auto index     = inputs[i]->expr().second;
            auto backward  = inputGrad[i];
            if (nullptr == backward) {
                continue;
            }
            if (backwardMap.find(inputExpr) == backwardMap.end()) {
                backwardMap.insert(std::make_pair(inputExpr, std::vector<VARP>(inputExpr->outputSize())));
            }
            auto& inputVarMap = backwardMap[inputExpr];
            if (nullptr == inputVarMap[index]) {
                inputVarMap[index] = backward;
            } else {
                inputVarMap[index] = _Add(inputVarMap[index], backward);
            }
        }
    }
    std::map<Express::VARP, Express::VARP> grads;
    std::map<Expr*, VARP> parametersExpr;
    for (auto p : parameters) {
        parametersExpr.insert(std::make_pair(p->expr().first.get(), p));
    }
    for (auto iter : backwardMap) {
        auto expr = iter.first.get();
        if (parametersExpr.find(expr) != parametersExpr.end()) {
            auto parameter   = parametersExpr[expr];
            grads[parameter] = iter.second[parameter->expr().second];
        }
    }
    // MNN_PRINT("Grad: %d <- %d\n", grads.size(), parameters.size());
    return grads;
}

} // namespace MNN
