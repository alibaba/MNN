//
//  TemplateMerge.cpp
//  MNNConverter
//
//  Created by MNN on 2019/09/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TemplateMerge.hpp"
#include <set>
namespace MNN {
namespace Express {
bool TemplateMerge::onExecute(const std::vector<VARP>& outputs, PassPriority priority) {
    if (mPriorities.size() <= priority) {
        return false;
    }
    bool hasChange = false;
    do {
        hasChange = false;
        for (const auto& pass_name : mPriorities.at(priority)) {
            auto& pass = mTemplates.at(pass_name);
            std::set<EXPRP> invalidVARP;
            auto execute = Variable::getExecuteOrder(outputs);
            for (int i=0; i<execute.size(); ++i) {
                auto var = execute[i];
                execute[i] = nullptr;
                if (var->get() == nullptr) {
                    continue;
                }
                if (invalidVARP.find(var) != invalidVARP.end()) {
                    continue;
                }
                if (pass(var)) {
                    hasChange = true;
#ifdef MNN_OPTIMIZE_DEBUG
                    MNN_ERROR("%s changed by %s\n", var->name().c_str(), pass_name.c_str());
#endif
                } else {
                    invalidVARP.insert(var);
                }
            }
        }
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
