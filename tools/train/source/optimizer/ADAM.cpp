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

std::pair<std::vector<Express::VARP>, std::vector<Express::VARP>>  ADAM::onMakeParameterUpdateGraphByGrad(const std::vector<ParameterOptGrad>& parameterGrads) {
    auto step  = _Const(1.0f, {}, NCHW);
    auto beta1 = _Const(mMomentum, {}, NCHW);
    auto beta2 = _Const(mMomentum2, {}, NCHW);
    auto eps = _Const(mEps,{}, NCHW);
    beta1->setName("Beta1");
    beta2->setName("Beta2");
    eps->setName("Eps");

    std::map<MNN::Express::VARP, MNN::Express::VARP> varUpdateMap;
    step->setName("optimize_step");
    auto stepPlus1 = step + _Scalar<float>(1.0f);
    stepPlus1->setName("optimize_step+1");
    varUpdateMap[step] = stepPlus1;
    auto correction = _Sqrt(_Const(1.0f, {}, NCHW) - _Pow(beta2, step)) / (_Const(1.0f, {}, NCHW) - _Pow(beta1, step));
    correction->setName("correction");
    auto weightDecay = _Const(mWeightDecay, {}, NCHW);
    for (auto iter : parameterGrads) {
        auto p = iter.parameter;
        MNN_PRINT("optimize variable: %s\n", p->name().c_str());
        p.fix(VARP::TRAINABLE);
        auto grad = iter.parameterGrad;
        grad->setName(p->name()+"_grad");
#if 0
        if (p->name().find("_BN_RunningMean_Weight") != string::npos) {
            varUpdateMap[p] = trainInfo[p->name()];
            continue; // not update running stats
        }
        if (p->name().find("_BN_RunningVariance_Weight") != string::npos) {
            varUpdateMap[p] = trainInfo[p->name()];
            continue; // not update running stats
        }
        if (p->name().find("_BN_Eps_Weight") != string::npos) {
            continue; // not update eps
        }
#endif
        auto pInfo = p->getInfo();
        auto pDims = pInfo->dim;

        auto l2grad = weightDecay * p;
        l2grad->setName(p->name() + "_l2grad");

        VARP gradWithDecay = grad + l2grad;
    
        VARP history1 = _Const(0.0f, pDims, pInfo->order);
        history1->setName(p->name() + "_momentum1");
        history1.fix(VARP::TRAINABLE);
        auto newHistory1 = beta1 * history1 + (_Scalar(1.0f) - beta1) * gradWithDecay;
        newHistory1->setName("update_" + history1->name());

        VARP history2 = _Const(0.0f, pDims, pInfo->order);
        history2->setName(p->name() + "_momentum2");
        history2.fix(VARP::TRAINABLE);
        auto newHistory2 = beta2 * history2 + (_Scalar(1.0f) - beta2) * _Square(gradWithDecay);
        newHistory2->setName("update_" + history2->name());

        VARP localLearningRate = iter.learningRate;
        auto finalGrad = localLearningRate * correction * (history1 / (_Sqrt(history2 + _Scalar<float>(1e-8)) + eps));
        finalGrad->setName(p->name() + "_final_grad");

        auto updateValue = _Subtract(p, finalGrad);
        updateValue->setName("update_" + p->name());
        varUpdateMap[p] = updateValue;
        varUpdateMap[history1] = newHistory1;
        varUpdateMap[history2] = newHistory2;
    }
    std::vector<Express::VARP> res;
    std::vector<Express::VARP> resUpdate;
    for (auto& iter : varUpdateMap) {
        res.push_back(iter.first);
        resUpdate.push_back(iter.second);
    }
    return std::make_pair(res, resUpdate);
}
} // namespace Train
} // namespace MNN
