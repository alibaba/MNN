//
//  FixModule.cpp
//  MNN
//
//  Created by MNN on 2019/12/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "FixModule.hpp"
#include <MNN/expr/ExprCreator.hpp>
using namespace MNN::Express;
namespace MNN {
namespace Express {
FixModule::FixModule(std::vector<Express::VARP> output, std::vector<Express::VARP> parameters,
                     std::vector<std::pair<Express::VARP, Express::Dimensionformat>> inputs) {
    for (auto p : parameters) {
        addParameter(p);
    }
    mInputs = std::move(inputs);
    mOutput = std::move(output);
}
void FixModule::onClearCache() {
    for (auto v : mInputs) {
        v.first.fix(VARP::INPUT);
    }
}

std::vector<Express::VARP> FixModule::onForward(const std::vector<Express::VARP>& inputs) {
    MNN_ASSERT(inputs.size() == mInputs.size());
    for (int i = 0; i < inputs.size(); ++i) {
        auto var = inputs[i];
        var      = _Convert(var, mInputs[i].second);
        Variable::replace(mInputs[i].first, var);
    }
    return mOutput;
}

Module* FixModule::clone(CloneContext* ctx) const {
    FixModule* module(new FixModule);
    for (auto& it : mInputs) {
        VARP v = ctx->getOrClone(it.first);
        module->mInputs.push_back(std::make_pair(v, it.second));
    }
    for (auto& it : mOutput) {
        VARP v = ctx->getOrClone(it);
        module->mOutput.push_back(v);
    }
    return this->cloneBaseTo(ctx, module);
}

} // namespace Express
} // namespace MNN
