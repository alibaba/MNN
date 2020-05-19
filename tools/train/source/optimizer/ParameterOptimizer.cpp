//
//  ParameterOptimizer.cpp
//  MNN
//
//  Created by MNN on 2019/11/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ParameterOptimizer.hpp"
#include "SGD.hpp"
#include "ADAM.hpp"
namespace MNN {
namespace Train {

ParameterOptimizer* ParameterOptimizer::createSGD(float lr, float momentum, float weightDecay, RegularizationMethod method) {
    auto sgd = new SGD;
    sgd->setLearningRate(lr);
    sgd->setMomentum(momentum);
    sgd->setWeightDecay(weightDecay);
    sgd->setRegularizationMethod(method);
    return sgd;
}

ParameterOptimizer* ParameterOptimizer::createADAM(float lr, float momentum, float momentum2, float weightDecay, float eps, RegularizationMethod method) {
    auto adam = new ADAM;
    adam->setLearningRate(lr);
    adam->setMomentum(momentum);
    adam->setMomentum2(momentum2);
    adam->setWeightDecay(weightDecay);
    adam->setEps(eps);
    adam->setRegularizationMethod(method);
    return adam;
}

bool ParameterOptimizer::step(Express::VARP loss) {
    mStep++;
    auto res = this->onGetNextParameter(loss);
    for (auto iter : res) {
        iter.second.fix(Express::VARP::TRAINABLE);
    }
    for (auto iter : res) {
        iter.first->input(iter.second);
    }
    return !res.empty();
}

int ParameterOptimizer::currentStep() {
    return mStep;
}

void ParameterOptimizer::setCurrentStep(int step) {
    mStep = step;
}
void ParameterOptimizer::append(const std::vector<Express::VARP>& parameters) {
    for (auto p : parameters) {
        if (p->expr().first->inputType() == Express::VARP::TRAINABLE) {
            mParameters.insert(p);
            this->onAppend(p);
        }
    }
}
void ParameterOptimizer::remove(const std::vector<Express::VARP>& parameters) {
    for (auto p : parameters) {
        mParameters.erase(p);
        this->onRemove(p);
    }
}
const std::set<Express::VARP>& ParameterOptimizer::parameters() const {
    return mParameters;
}

} // namespace Train
} // namespace MNN
