//
//  TemplateMerge.hpp
//  MNNConverter
//
//  Created by MNN on 2019/09/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Optimizer.hpp"
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

    void insertTemplate(std::string key, std::function<bool(VARP)> compare, std::function<bool(VARP)> transform);

private:
    TemplateMerge() {
    }
    std::map<std::string, std::pair<std::function<bool(VARP)>, std::function<bool(VARP)>>> mTemplates;
};
class TemplateMergeRegister {
public:
    TemplateMergeRegister(const std::string& pass, std::string key, std::function<bool(VARP)> compare, std::function<bool(VARP)> transform) {
        TemplateMerge::getInstance(pass).insertTemplate(key, compare, transform);
    }
};
} // namespace Express
} // namespace MNN
