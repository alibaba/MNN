//
//  ParameterOptimizer.cpp
//  MNN
//
//  Created by MNN on 2019/11/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ParameterOptimizer.hpp"

namespace MNN {
namespace Train {

bool ParameterOptimizer::step(Express::VARP loss) {
    mStep++;
    auto res = this->onGetNextParameter(loss);
    for (auto iter : res) {
        iter.second.fix(Express::VARP::CONST);
    }
    for (auto iter : res) {
        Express::Variable::replace(iter.first, iter.second);
    }
    return !res.empty();
}

int ParameterOptimizer::currentStep() {
    return mStep;
}

void ParameterOptimizer::setCurrentStep(int step) {
    mStep = step;
}

void ParameterOptimizer::append(const std::set<Express::VARP>& parameters) {
    for (auto p : parameters) {
        mParameters.insert(p);
    }
    this->onAppend(parameters);
}
void ParameterOptimizer::remove(const std::set<Express::VARP>& parameters) {
    for (auto p : parameters) {
        mParameters.erase(p);
    }
    this->onRemove(parameters);
}
const std::set<Express::VARP>& ParameterOptimizer::parameters() const {
    return mParameters;
}

} // namespace Train
} // namespace MNN
