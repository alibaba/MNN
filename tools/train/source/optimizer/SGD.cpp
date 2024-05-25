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
SGD::SGD(std::shared_ptr<Module> module) : ParameterOptimizer(module) {
    auto train = ParameterOptimizer::trainable();
    for (auto p : train) {
        mHistory[p] = _Const(0.0f, p->getInfo()->dim, p->getInfo()->order);
    }
}

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
    mHistory[param] = lr * grad + _Const(mMomentum, {}, NCHW) * mHistory[param];;
    mHistory[param].fix(Express::VARP::CONSTANT);
    //FUNC_PRINT_ALL(_ReduceMax(grad)->readMap<float>()[0], f);
    return mHistory[param];
}

std::map<Express::VARP, Express::VARP> SGD::onGetNextParameter(Express::VARP loss) {
    auto grad = OpGrad::grad(loss, trainable(), mGradBlockExprName);
    auto parameters = module()->parameters();
    std::vector<VARP> prepareCompute;
    for (auto iter : parameters) {
        if (iter->expr().first->get() != nullptr) {
            prepareCompute.emplace_back(iter);
        }
    }
    for (auto& iter : grad) {
        prepareCompute.emplace_back(iter.second);
    }
    Variable::prepareCompute(prepareCompute);
    std::vector<VARP> replaceOp(prepareCompute.size());
    for (int i=0; i<prepareCompute.size(); ++i) {
        auto info = prepareCompute[i]->getInfo();
        auto ptr = prepareCompute[i]->readMap<void>();
        if (nullptr == ptr) {
            MNN_ERROR("Compute error in SGD\n");
            return {};
        }
        auto newVar = _Const(ptr, info->dim, info->order, info->type);
        replaceOp[i]= newVar;
    }
    for (int i=0; i<prepareCompute.size(); ++i) {
        Variable::replace(prepareCompute[i], replaceOp[i]);
    }

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

std::pair<std::vector<Express::VARP>, std::vector<Express::VARP>>  SGD::onMakeParameterUpdateGraphByGrad(const std::vector<ParameterOptGrad>& parameterGrads) {
    std::map<MNN::Express::VARP, MNN::Express::VARP> varUpdateMap;
    auto momentum = _Const(mMomentum, {}, NCHW);
    auto weightDecay = _Const(mWeightDecay, {}, NCHW);
    for (int pIndex=0; pIndex<parameterGrads.size(); ++pIndex) {
        auto p = parameterGrads[pIndex].parameter;
        auto grad = parameterGrads[pIndex].parameterGrad;

        // FIXME
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

        VARP history = _Const(0.0f, pDims, pInfo->order);
        history->setName(p->name() + "_momentum");
        history.fix(VARP::TRAINABLE);

        auto newHistory = gradWithDecay + momentum * history;
        newHistory->setName("update_" + history->name());

        VARP localLearningRate = parameterGrads[pIndex].learningRate;
        VARP finalGrad = localLearningRate * history;
        finalGrad->setName(p->name() + "_final_grad");

        auto updateValue = _Subtract(p, finalGrad);
        updateValue->setName("update_" + p->name());
        varUpdateMap[p] = updateValue;
        varUpdateMap[history] = newHistory;
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
