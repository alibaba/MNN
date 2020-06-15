//
//  SGD.cpp
//  MNN
//
//  Created by MNN on 2019/11/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "SGD.hpp"
#include "OpGrad.hpp"

using namespace MNN::Express;

namespace MNN {
namespace Train {

void SGD::setLearningRate(float rate) {
    mLearningRate = rate;
}

void SGD::setMomentum(float momentum) {
    mMomentum = momentum;
}

void SGD::setWeightDecay(float decay) {
    mWeightDecay = decay;
}

void SGD::setRegularizationMethod(RegularizationMethod method) {
    mRegularizationMethod = method;
}

float SGD::currentLearningRate() {
    return mLearningRate;
}

float SGD::getMomentum() {
    return mMomentum;
}

float SGD::getWeightDecay() {
    return mWeightDecay;
}

SGD::RegularizationMethod SGD::getRegularizationMethod() {
    return mRegularizationMethod;
}

void SGD::onAppend(Express::VARP p) {
    mHistory[p] = _Const(0.0f, p->getInfo()->dim, p->getInfo()->order);
}

void SGD::onRemove(Express::VARP p) {
    mHistory.erase(p);
}

Express::VARP SGD::regularizeParameters(Express::VARP param, Express::VARP grad) {
    VARP addWeightDecayGrad;
    if (mRegularizationMethod == L1) {
        auto temp          = _Sign(param);
        addWeightDecayGrad = _Const(mWeightDecay, {}, NCHW) * temp + grad;
    } else if (mRegularizationMethod == L2) {
        addWeightDecayGrad = _Const(mWeightDecay, {}, NCHW) * param + grad;
    } else if (mRegularizationMethod == L1L2) {
        auto temp          = _Sign(param);
        auto L1 = _Const(mWeightDecay, {}, NCHW) * temp;
        auto L2 = _Const(mWeightDecay, {}, NCHW) * param;
        addWeightDecayGrad = L1 + L2 + grad;
    }

    return addWeightDecayGrad;
}

Express::VARP SGD::onComputeUpdateValue(Express::VARP param, Express::VARP grad) {
    auto lr         = _Const(mLearningRate, {}, NCHW);
    mHistory[param] = lr * grad + _Const(mMomentum, {}, NCHW) * mHistory[param];
    mHistory[param].fix(Express::VARP::CONSTANT);
    //FUNC_PRINT_ALL(_ReduceMax(grad)->readMap<float>()[0], f);
    return mHistory[param];
}

std::map<Express::VARP, Express::VARP> SGD::onGetNextParameter(Express::VARP loss) {
    auto grad = OpGrad::grad(loss, parameters(), mGradBlockExprName);
    std::vector<VARP> prepareCompute;
    for (auto& iter : grad) {
        prepareCompute.emplace_back(iter.second);
    }
    Variable::prepareCompute(prepareCompute);

    for (auto& iter : grad) {
        // apply regularization
        auto addWeightDecayGrad = regularizeParameters(iter.first, iter.second);
        addWeightDecayGrad.fix(Express::VARP::CONSTANT);
        // apply momentum, etc.
        auto updateValue = this->onComputeUpdateValue(iter.first, addWeightDecayGrad);
        // apply update
        auto newParameter = iter.first - updateValue;
        iter.second       = newParameter;
    }
    return grad;
}

} // namespace Train
} // namespace MNN
