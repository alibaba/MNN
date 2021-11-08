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
using namespace MNN::Express;
namespace MNN {
namespace Train {
ParameterOptimizer::ParameterOptimizer(std::shared_ptr<Module> module) {
    auto parameters = module->parameters();
    for (auto p : parameters) {
        if (nullptr == p.get()) {
            continue;
        }
        if (p->expr().first->get() != nullptr) {
            continue;
        }
        if (p->expr().first->inputType() == Express::VARP::TRAINABLE) {
            mTrainable.insert(p);
        }
    }
    mModule = module;
}

ParameterOptimizer* ParameterOptimizer::createSGD(std::shared_ptr<Module> module, float lr, float momentum, float weightDecay, RegularizationMethod method) {
    auto sgd = new SGD(module);
    sgd->setLearningRate(lr);
    sgd->setMomentum(momentum);
    sgd->setWeightDecay(weightDecay);
    sgd->setRegularizationMethod(method);
    return sgd;
}

ParameterOptimizer* ParameterOptimizer::createADAM(std::shared_ptr<Module> module, float lr, float momentum, float momentum2, float weightDecay, float eps, RegularizationMethod method) {
    auto adam = new ADAM(module);
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

} // namespace Train
} // namespace MNN
