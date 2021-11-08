//
//  ADAM.cpp
//  MNN
//
//  Created by MNN on 2019/12/03.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ADAM.hpp"
#include "OpGrad.hpp"

using namespace MNN::Express;

namespace MNN {
namespace Train {
ADAM::ADAM(std::shared_ptr<Module> module) : SGD(module) {
    auto train = ParameterOptimizer::trainable();
    for (auto p : train) {
        mHistory2[p] = _Const(0.0f, p->getInfo()->dim, p->getInfo()->order);
    }
}

void ADAM::setMomentum2(float momentum2) {
    mMomentum2 = momentum2;
}

void ADAM::setEps(float eps) {
    mEps = eps;
}

float ADAM::getMomentum2() {
    return mMomentum2;
}

float ADAM::getEps() {
    return mEps;
}

Express::VARP ADAM::onComputeUpdateValue(Express::VARP param, Express::VARP grad) {
    auto lr    = _Const(mLearningRate, {}, NCHW);
    auto step  = _Const(currentStep(), {}, NCHW);
    auto beta1 = _Const(mMomentum, {}, NCHW);
    auto beta2 = _Const(mMomentum2, {}, NCHW);
    auto eps   = _Const(mEps, {}, NCHW);
    // auto m = mHistory[param];
    // auto v = mHistory2[param];

    auto correction = _Sqrt(_Const(1.0f, {}, NCHW) - _Pow(beta2, step)) / (_Const(1.0f, {}, NCHW) - _Pow(beta1, step));

    mHistory[param] = beta1 * mHistory[param] + (_Const(1.0f, {}, NCHW) - beta1) * grad;
    mHistory[param].fix(Express::VARP::CONSTANT);

    mHistory2[param] = beta2 * mHistory2[param] + (_Const(1.0f, {}, NCHW) - beta2) * _Square(grad);
    mHistory2[param].fix(Express::VARP::CONSTANT);

    auto updateValue = lr * correction * (mHistory[param] / (_Sqrt(mHistory2[param]) + eps));
    updateValue.fix(Express::VARP::CONSTANT);

    return updateValue;
}

} // namespace Train
} // namespace MNN
