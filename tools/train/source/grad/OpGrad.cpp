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

std::vector<Express::VARP> OpGrad::gradLinear(Express::VARP loss, const std::vector<Express::VARP>& parameters, const std::vector<Express::VARP>& outputDiff, const std::vector<std::string> blockExpr) {
    std::map<EXPRP, std::vector<VARP>> backwardMap;
    auto outputSize = loss->expr().first->outputSize();
    if (outputSize != outputDiff.size()) {
        MNN_ERROR("The expr output %d, but diff size is %d\n", outputSize, (int)outputDiff.size());
        return {};
    }
    backwardMap[loss->expr().first] = outputDiff;
    std::set<VARP> parameterSet;
    for (auto p : parameters) {
        parameterSet.insert(p);
    }
    auto res = gradCommon(loss, parameterSet, backwardMap, blockExpr);
    std::vector<VARP> linearRes(parameters.size(), nullptr);
    for (int i=0; i<parameters.size(); ++i) {
        auto iter = res.find(parameters[i]);
        if (iter != res.end()) {
            linearRes[i] = iter->second;
        }
    }
    return linearRes;
}

std::map<Express::VARP, Express::VARP> OpGrad::grad(VARP loss, const std::set<Express::VARP>& parameters, const std::vector<std::string> blockName) {
    std::map<EXPRP, std::vector<VARP>> backwardMap;
    {
        auto shape = loss->getInfo();
        MNN_ASSERT(shape->size == 1);
        auto init                       = _Const(1.0f, shape->dim, shape->order);
        backwardMap[loss->expr().first] = std::vector<VARP>{init};
    }
    return gradCommon(loss, parameters, backwardMap, blockName);
}
std::map<Express::VARP, Express::VARP> OpGrad::gradCommon(Express::VARP loss, const std::set<Express::VARP>& parameters, std::map<EXPRP, std::vector<VARP>>& backwardMap, const std::vector<std::string> blockName) {
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
            // MNN_PRINT("Can't grad for %s, %s\n", expr->name().c_str(), MNN::EnumNameOpType(expr->get()->type()));
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
        if (!blockName.empty()) {
            if (std::find(blockName.begin(), blockName.end(), expr->name()) != blockName.end()) {
                for (int ii = 0; ii <inputGrad.size(); ii++) {
                    inputGrad[ii] = nullptr;
                }
                continue;
            }
        }
#ifdef MNN_TRAIN_DEBUG
        for (int i = 0; i < inputGrad.size(); ++i) {
            if (nullptr == inputGrad[i]) {
                continue;
            }
            auto info = inputGrad[i]->getInfo();
            if (nullptr == info) {
                MNN_ERROR("Grad error for %s, %d\n", expr->name().c_str(), expr->get()->type());
                break;
            }
        }
#endif
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
