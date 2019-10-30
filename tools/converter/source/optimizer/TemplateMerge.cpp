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
bool TemplateMerge::onExecute(const std::vector<VARP>& outputs, std::shared_ptr<Parameters> parameters) {
    bool hasChange = false;
    do {
        hasChange = false;
        for (auto& iter : mTemplates) {
            std::set<VARP> invalidVARP;
            auto execute = Variable::getExecuteOrder(outputs);
            for (auto var : execute) {
                if (invalidVARP.find(var) != invalidVARP.end()) {
                    continue;
                }
                if (iter.second.first(var)) {
                    hasChange = hasChange || iter.second.second(var);
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

void TemplateMerge::insertTemplate(std::string key, std::function<bool(VARP)> compare,
                                   std::function<bool(VARP)> transform) {
    mTemplates.insert(std::make_pair(key, std::make_pair(compare, transform)));
}
} // namespace Express
} // namespace MNN
