//
//  PipelineModule.hpp
//  MNN
//
//  Created by MNN on 2020/01/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef PipelineModule_hpp
#define PipelineModule_hpp
#include "Module.hpp"
namespace MNN {
namespace Train {

class MNN_PUBLIC PipelineModule : public Module {
public:
    typedef std::function<std::pair<std::vector<int>, std::shared_ptr<Module>>(Express::EXPRP)> Transformer;
    PipelineModule(std::vector<Express::VARP> inputs, std::vector<Express::VARP> outputs,
                   Transformer& transformFunction);
    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP>& inputs) override;
    virtual void onClearCache() override;

private:
    std::vector<std::tuple<std::shared_ptr<Module>, std::vector<int>, std::vector<int>>> mSubModules;
    std::vector<Express::VARP> mStack;
    std::vector<int> mInputIndexes;
    std::vector<int> mOutputIndexes;
};
} // namespace Train
} // namespace MNN

#endif
