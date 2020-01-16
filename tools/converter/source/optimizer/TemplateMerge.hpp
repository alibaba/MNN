//
//  TemplateMerge.hpp
//  MNNConverter
//
//  Created by MNN on 2019/09/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Optimizer.hpp>
namespace MNN {
namespace Express {
class TemplateMerge : public Optimizer {
public:
    virtual Cost onMeasure(const std::vector<VARP>& outputs,
                           std::shared_ptr<Parameters> parameters = nullptr) override {
        return Cost();
    }

    virtual bool onExecute(const std::vector<VARP>& outputs, std::shared_ptr<Parameters> parameters = nullptr) override;

    static TemplateMerge& getInstance(const std::string& pass);

    void insertTemplate(std::string key, std::function<bool(EXPRP)> compare, std::function<bool(EXPRP)> transform);

private:
    TemplateMerge() {
    }
    std::map<std::string, std::pair<std::function<bool(EXPRP)>, std::function<bool(EXPRP)>>> mTemplates;
};
class TemplateMergeRegister {
public:
    TemplateMergeRegister(const std::string& pass, std::string key, std::function<bool(EXPRP)> compare, std::function<bool(EXPRP)> transform) {
        TemplateMerge::getInstance(pass).insertTemplate(key, compare, transform);
    }
};
} // namespace Express
} // namespace MNN
