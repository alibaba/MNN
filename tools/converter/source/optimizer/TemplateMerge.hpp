//
//  TemplateMerge.hpp
//  MNNConverter
//
//  Created by MNN on 2019/09/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Optimizer.hpp>
#include "Global.hpp"
#include "config.hpp"

#define MNN_THROW_CHECK(success, log) \
if(!(success)){ \
MNN_ERROR("Check failed: %s ==> %s\n", #success, #log); \
}

namespace MNN {
namespace Express {

enum PassPriority : int {
   PASS_PRIORITY_FRONT = 0,
   PASS_PRIORITY_HIGH = 1,
   PASS_PRIORITY_MIDDLE = 2,
   PASS_PRIORITY_LOW = 3,
   PASS_PRIORITY_FINAL = 4,
};

class TemplateMerge : public Optimizer {
public:
    virtual Cost onMeasure(const std::vector<VARP>& outputs,
                           std::shared_ptr<Parameters> parameters = nullptr) override {
        return Cost();
    }
    bool onExecute(const std::vector<VARP>& outputs,
                   std::shared_ptr<Parameters> parameters = nullptr) override {
        std::map<std::string, VARP> map;
        return onExecute(outputs, PASS_PRIORITY_HIGH, map);
    }
    bool onExecute(const std::vector<VARP>& outputs, PassPriority priority, std::map<std::string, VARP>& updateVars, const std::vector<VARP>& boundary = {});

    static TemplateMerge& getInstance(const std::string& pass);

    void insertTemplate(std::string key, std::function<bool(EXPRP)> compare, std::function<bool(EXPRP)> transform,
                        PassPriority priority = PASS_PRIORITY_HIGH);
    void insertTemplateV2(std::string key, std::function<bool(EXPRP)> transform, PassPriority priority = PASS_PRIORITY_HIGH);

private:
    TemplateMerge() {
    }
    std::vector<std::vector<std::string>> mPriorities;
    std::map<std::string, std::function<bool(EXPRP)>> mTemplates;
};
class TemplateMergeRegister {
public:
    TemplateMergeRegister(const std::string& pass, std::string key, std::function<bool(EXPRP)> compare, std::function<bool(EXPRP)> transform,
                          PassPriority priority) {
        TemplateMerge::getInstance(pass).insertTemplate(key, compare, transform, priority);
    }
};
} // namespace Express
} // namespace MNN
