//
//  TemplateMerge.cpp
//  MNNConverter
//
//  Created by MNN on 2019/09/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TemplateMerge.hpp"
#include <MNN_generated.h>
#include <set>
#include <unordered_set>
#include <MNN/expr/ExecutorScope.hpp>
namespace MNN {
namespace Express {

static std::vector<EXPRP> splitInBoundary(const std::vector<EXPRP>& execute, const std::vector<VARP>& boundary) {
    std::unordered_set<EXPRP> blacklist;
    for (const auto expr : Variable::getExecuteOrder(boundary)) {
        blacklist.insert(expr);
    }
    std::vector<EXPRP> executeValid;
    for (const auto expr : execute) {
        if (blacklist.count(expr) == 0) {
            executeValid.push_back(expr);
        }
    }
    return executeValid;
}

static bool crossBoundary(EXPRP origin, EXPRP opt, const std::unordered_set<EXPRP>& boundary) {
    if (boundary.empty()) {
        return false;
    }
    int flag[] = {1, 2, 4, 8}; // four state: origin visited, opt visited, origin contained, opt contained
    std::unordered_map<EXPRP, int> exprState;
    std::unordered_set<EXPRP> edge[] = {{origin}, {opt}};
    auto step = [&](int index) {
        if (edge[index].empty()) {
            return;
        }
        std::unordered_set<EXPRP> nextEdge;
        while (!edge[index].empty()) {
            auto now = *edge[index].begin();
            edge[index].erase(now);
            if (exprState[now] & flag[index]) {
                continue;
            }
            exprState[now] &= flag[index];
            // opposite side contain the expr too, remove itself and parent nodes
            if (exprState[now] & flag[1 - index + 2]) {
                Expr::visit(now, [&](EXPRP expr) {
                    if (!(exprState[expr] & flag[1 - index + 2])) {
                        return false;
                    }
                    exprState[expr] ^= flag[1 - index + 2];
                    edge[1 - index].erase(expr);
                    return true;
                }, [](EXPRP expr) { return true; });
                continue;
            }
            exprState[now] &= flag[index + 2];
            for (auto input : now->inputs()) {
                if (input.get() == nullptr) {
                    continue;
                }
                auto next = input->expr().first;
                if (!(exprState[next] & flag[index])) {
                    if (edge[index].count(next) == 0) {
                        nextEdge.insert(next);
                    }
                }
            }
        }
        edge[index] = std::move(nextEdge);
    };
    bool optDone = false;
    while (!(edge[0].empty() && edge[1].empty())) {
        // alternate iterate origin pass and opt pass, which control time complexity
        step(0);
        step(1);
        if (edge[1].empty()) {
            // opt pass step done, origin pass expr won't be remove. check whether cross boundary
            if (!optDone) {
                optDone = true;
                for (auto expr : boundary) {
                    if (exprState[expr] & flag[2]) {
                        return true;
                    }
                }
            }
            for (auto expr : edge[0]) {
                // check whether new step edge of origin pass cross boundary
                if (boundary.count(expr) && !(exprState[expr] & flag[3])) {
                    return true;
                }
            }
        }
    }
    // check whether origin pass (replaced by opt pass) cross boundary
    for (auto expr : boundary) {
        if (exprState[expr] & flag[2]) {
            return true;
        }
    }
    return false;
}

static std::map<std::string, VARP> updateInputVarOfExpr(EXPRP expr) {
    std::map<std::string, VARP> res;
    auto inputs = expr->inputs();
    for (int i = 0; i < inputs.size(); ++i) {
        VARP input = inputs.at(i);
        res[input->name()] = input;
    }
    return res;
}

bool TemplateMerge::onExecute(const std::vector<VARP>& outputs, PassPriority priority, std::map<std::string, VARP>& updateVars, const std::vector<VARP>& boundary) {
    if (mPriorities.size() <= priority) {
        return false;
    }
    bool hasChange = false;
    std::unordered_set<EXPRP> boundaryExpr;
    for (auto it : boundary) {
        boundaryExpr.insert(it->expr().first);
    }
    do {
        hasChange = false;
        for (const auto& pass_name : mPriorities.at(priority)) {
            auto& pass = mTemplates.at(pass_name);
            std::set<EXPRP> invalidVARP;
            auto execute = splitInBoundary(Variable::getExecuteOrder(outputs), boundary);
            for (int i=0; i<execute.size(); ++i) {
                auto var = execute[i];
                execute[i] = nullptr;
                if (var->get() == nullptr) {
                    continue;
                }
                if (invalidVARP.find(var) != invalidVARP.end()) {
                    continue;
                }
                // track arguments need by Expr::create, not create backup expr to avoid influence optimize
                auto originArgs = make_tuple(var->extra(), var->inputs(), var->outputSize());
                if (pass(var)) {
                    auto originVar = Expr::create(std::get<0>(originArgs), std::move(std::get<1>(originArgs)), std::get<2>(originArgs));
                    if (crossBoundary(originVar, var, boundaryExpr)) {
                        Expr::replace(var, originVar);
                        invalidVARP.insert(var);
                        continue;
                    }
                    hasChange = true;
#ifdef MNN_OPTIMIZE_DEBUG
                    MNN_ERROR("%s changed by %s\n", var->name().c_str(), pass_name.c_str());
#endif
                } else {
                    invalidVARP.insert(var);
                }
            }
        }
        MNN::Express::ExecutorScope::Current()->gc();
    } while (hasChange);
    return true;
}

TemplateMerge& TemplateMerge::getInstance(const std::string& pass) {
    static std::map<std::string, TemplateMerge> gMerge;
    if (gMerge.find(pass) == gMerge.end()) {
        gMerge.insert(std::make_pair(pass, TemplateMerge()));
    }
    auto iter = gMerge.find(pass);
    return iter->second;
}

void TemplateMerge::insertTemplateV2(std::string key,
                                   std::function<bool(EXPRP)> transform, PassPriority priority) {
    if (mPriorities.size() <= priority) {
        mPriorities.resize(priority + 1);
    }
    mPriorities[priority].push_back(key);
    mTemplates.insert(std::make_pair(key, transform));
}
void TemplateMerge::insertTemplate(std::string key, std::function<bool(EXPRP)> compare,
                                   std::function<bool(EXPRP)> transform, PassPriority priority) {
    auto wrap = [compare, transform](EXPRP expr) {
        if (!compare(expr)) {
            return false;
        }
        return transform(expr);
    };
    insertTemplateV2(key, wrap, priority);
}
} // namespace Express
} // namespace MNN
