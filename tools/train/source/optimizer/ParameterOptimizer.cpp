//
//  ParameterOptimizer.cpp
//  MNN
//
//  Created by MNN on 2019/11/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <sstream>
#include "MNN_generated.h"
#include "ParameterOptimizer.hpp"
#include "SGD.hpp"
#include "ADAM.hpp"
using namespace MNN::Express;
// TODO: need Refract
static bool _ReNameTensor(std::unique_ptr<MNN::NetT>& net) {
    auto& mNet = net;
    // Check dup name and modify
    std::set<std::string> opnames;
    for (int i = 0; i < mNet->oplists.size(); ++i) {
        auto& op    = mNet->oplists[i];
        auto opName = op->name;
        if (opName.empty()) {
            std::ostringstream defaultName;
            defaultName << EnumNameOpType(op->type);
            defaultName << i;
            op->name = defaultName.str();
            MNN_PRINT("%d op name is empty, set to %s\n", i, op->name.c_str());
        }
        bool rename = false;
        do {
            if (opnames.find(op->name) == opnames.end()) {
                break;
            }
            op->name = op->name + "_";
            rename = true;
        } while (true);
        opName = op->name;
        if (rename) {
            MNN_PRINT("%d op name is dup, set to %s\n", i, op->name.c_str());
        }
        opnames.insert(opName);
    }
    std::set<std::string> tensorNames;
    for (int i = 0; i < mNet->tensorName.size(); ++i) {
        auto tensorName = mNet->tensorName[i];
        if (tensorName.empty()) {
            tensorName = std::to_string(i);
        }
        bool rename = false;
        do {
            if (tensorNames.find(tensorName) == tensorNames.end()) {
                break;
            }
            tensorName = tensorName + "_";
            rename = true;
        } while (true);
        if (rename) {
            MNN_PRINT("%d tensor name is dup, set to %s\n", i, tensorName.c_str());
        }
        mNet->tensorName[i] = tensorName;
        tensorNames.insert(tensorName);
    }
    return true;
}

namespace MNN {
namespace Train {
ParameterOptimizer::ParameterOptimizer(std::shared_ptr<Module> module) {
    mModule = module;
    if (nullptr == mModule) {
        mModule.reset(Module::createEmpty(std::vector<MNN::Express::VARP>{}));
    }
    auto parameters = mModule->parameters();
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
}

ParameterOptimizer* ParameterOptimizer::createSGD(std::shared_ptr<Module> module, float lr, float momentum, float weightDecay, RegularizationMethod method) {
    auto sgd = new SGD(module);
    sgd->setLearningRate(lr);
    sgd->setMomentum(momentum);
    sgd->setWeightDecay(weightDecay);
    sgd->setRegularizationMethod(method);
    return sgd;
}

std::pair<std::vector<Express::VARP>, std::vector<Express::VARP>> ParameterOptimizer::makeParameterUpdateGraphByGrad(const std::vector<Express::VARP>& p, const std::vector<Express::VARP>& pd, const std::vector<Express::VARP>& lr) {
    if (p.size() != pd.size() || lr.size() != pd.size()) {
        MNN_ERROR("[ParameterOptimizer] makeParameterUpdateGraphByGrad: Size not match\n");
        std::pair<std::vector<Express::VARP>, std::vector<Express::VARP>> temp;
        return temp;
    }
    std::vector<ParameterOptGrad> grads;
    for (int i=0; i<p.size(); ++i) {
        ParameterOptGrad g;
        g.parameter = p[i];
        g.parameterGrad = pd[i];
        g.learningRate = lr[i];
        grads.emplace_back(g);
    }
    return this->onMakeParameterUpdateGraphByGrad(grads);
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

std::pair<std::vector<Express::VARP>, std::vector<Express::VARP>>  ParameterOptimizer::onMakeParameterUpdateGraphByGrad(const std::vector<ParameterOptGrad>& parameterGrads) {
    MNN_ERROR("[ParameterOptimizer]: Don't support make static graph for update parameters\n");
    return std::make_pair(std::vector<Express::VARP>{}, std::vector<Express::VARP>{});
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
static void _saveMNN(MNN::NetT* netStruct, const char* mnnFileName) {
    flatbuffers::FlatBufferBuilder builder(1024);
    auto offset = Net::Pack(builder, netStruct);
    builder.Finish(offset);
    // TODO, use FileWriter instead
    FILE* f = fopen(mnnFileName, "wb");
    fwrite(builder.GetBufferPointer(), 1, builder.GetSize(), f);
    fclose(f);
}
void ParameterOptimizer::makeLoopModel(const char* mnnFileName, std::vector<VARP> outputs, const std::pair<std::vector<Express::VARP>, std::vector<Express::VARP>>& parameters) {
    if (parameters.first.size() != parameters.second.size()) {
        MNN_ERROR("[ParameterOptimizer] makeLoopModel Size not match\n");
        return;
    }
    auto parameterSize = parameters.first.size();
    for (int i=0; i<parameterSize; ++i) {
        outputs.emplace_back(parameters.second[i]);
    }
    std::unique_ptr<MNN::NetT> netStruct(new MNN::NetT);
    Variable::save(outputs, netStruct.get());
    _ReNameTensor(netStruct);
    if (parameterSize == 0) {
        _saveMNN(netStruct.get(), mnnFileName);
        return;
    }
    for (int i = 0; i < netStruct->oplists.size(); ++i) {
        auto& op = netStruct->oplists[i];
        for (int v=0; v<parameterSize; ++v) {
            auto pu = parameters.second[v];
            auto pi = parameters.first[v];
            if (pu->name() == op->name) {
                for (int j = 0; j < netStruct->oplists.size(); ++j) {
                    auto& opSub = netStruct->oplists[j];
                    if (opSub->name == pi->name()) {
                        auto indexOri = op->outputIndexes;
                        op->outputIndexes = opSub->outputIndexes;

                        if ((opSub->name.find("_BN_RunningMean_Weight") != std::string::npos) || (opSub->name.find("_BN_RunningVariance_Weight") != std::string::npos)) {
                            for (int k = 0; k < netStruct->oplists.size(); ++k) {
                                auto& opSubSub = netStruct->oplists[k];
                                if (opSubSub->inputIndexes.size() > 0) {
                                    for (int kk = 0; kk < opSubSub->inputIndexes.size(); kk++) {
                                        if (opSubSub->inputIndexes[kk] == indexOri[0]) {
                                            opSubSub->inputIndexes[kk] = opSub->outputIndexes[0];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    _saveMNN(netStruct.get(), mnnFileName);
}

} // namespace Train
} // namespace MNN
