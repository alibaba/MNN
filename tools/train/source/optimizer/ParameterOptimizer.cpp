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

ParameterOptimizer* ParameterOptimizer::createSGD(float lr, float momentum) {
    auto sgd = new SGD;
    sgd->setLearningRate(lr);
    sgd->setMomentum(momentum);
    sgd->setWeightDecay(0.0005f);
    return sgd;
}
ParameterOptimizer* ParameterOptimizer::createADAM(float lr, float momentum, float momentum2) {
    auto opt = new ADAM;
    opt->setMomentum(momentum);
    opt->setLearningRate(lr);
    opt->setMomentum2(momentum2);
    return opt;
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
