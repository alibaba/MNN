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

void ADAM::setMomentum2(float momentum2) {
    mMomentum2 = momentum2;
}

void ADAM::setEps(float eps) {
    mEps = eps;
}

void ADAM::onAppend(const std::set<Express::VARP>& parameters) {
    for (auto p : parameters) {
        mHistory[p]  = _Const(0.0f, p->getInfo()->dim, p->getInfo()->order);
        mHistory2[p] = _Const(0.0f, p->getInfo()->dim, p->getInfo()->order);
    }
}

void ADAM::onRemove(const std::set<Express::VARP>& parameters) {
    for (auto p : parameters) {
        mHistory.erase(p);
        mHistory2.erase(p);
    }
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
    mHistory[param].fix(Express::VARP::CONST);

    mHistory2[param] = beta2 * mHistory2[param] + (_Const(1.0f, {}, NCHW) - beta2) * _Square(grad);
    mHistory2[param].fix(Express::VARP::CONST);

    auto updateValue = lr * correction * (mHistory[param] / (_Sqrt(mHistory2[param]) + eps));
    updateValue.fix(Express::VARP::CONST);

    return updateValue;
}

} // namespace Train
} // namespace MNN
