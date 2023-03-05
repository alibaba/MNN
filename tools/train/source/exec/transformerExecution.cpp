//
//  transformerExecution.cpp
//  MNN
//
//  Created by MNN on 2019/05/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/ExprCreator.hpp>
#include <fstream>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include <MNN/expr/Module.hpp>
#include "OpGrad.hpp"
#include "Transformer.hpp"
#include "core/Macro.h"
#define USE_ELU
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include "rapidjson/document.h"
#include <algorithm>

using namespace MNN;
using namespace MNN::Express;
using namespace MNN::Train;
using namespace std;

int main(int argc, const char* argv[]) {
    if (argc < 4) {
        MNN_PRINT("Usage: ./transformer.out temp.bin dst.bin config.json\n");
        return 0;
    }
    rapidjson::Document document;
    {
        std::ifstream fileNames(argv[3]);
        std::ostringstream output;
        output << fileNames.rdbuf();
        auto outputStr = output.str();
        document.Parse(outputStr.c_str());
        if (document.HasParseError()) {
            MNN_ERROR("Invalid json\n");
            return 0;
        }
        FUNC_PRINT(document.HasParseError());
        FUNC_PRINT(document.IsArray());
        FUNC_PRINT(document.IsObject());
    }
    auto configObject = document.GetObject();
    std::vector<std::string> noUpdateOps;
    std::vector<std::string> onlyUpdateOps;
    std::vector<std::string> stopBackPropOps;
    std::string optimizerType = "SGD";
    if (configObject.HasMember("Optimizor")) {
        auto optimizor = configObject["Optimizor"].GetObject();
        if (optimizor.HasMember("OnlyUpdateOps")) {
            auto limitArray = optimizor["OnlyUpdateOps"].GetArray();
            for (auto vIter = limitArray.begin(); vIter != limitArray.end(); vIter++) {
                onlyUpdateOps.emplace_back(vIter->GetString());
                MNN_PRINT("will only update: %s \n", vIter->GetString());
            }
        }
        if (optimizor.HasMember("NoUpdateOps")) {
            auto limitArray = optimizor["NoUpdateOps"].GetArray();
            for (auto vIter = limitArray.begin(); vIter != limitArray.end(); vIter++) {
                noUpdateOps.emplace_back(vIter->GetString());
                if (onlyUpdateOps.empty())
                    MNN_PRINT("will not update: %s \n", vIter->GetString());
            }
        }
        if (optimizor.HasMember("StopBackPropOps")) {
            auto limitArray = optimizor["StopBackPropOps"].GetArray();
            for (auto vIter = limitArray.begin(); vIter != limitArray.end(); vIter++) {
                stopBackPropOps.emplace_back(vIter->GetString());
                MNN_PRINT("will stop back prop from (also not update this op): %s \n", vIter->GetString());
            }
        }
        if (optimizor.HasMember("type")) {
            optimizerType = std::string(optimizor["type"].GetString());
            MNN_PRINT("optimizer type: %s\n", optimizerType.c_str());
        }
    }
    auto bnMomentum = new MNN::AttributeT;
    bnMomentum->f = 0.99;
    if (configObject.HasMember("BatchNorm")) {
        auto bnConfig = configObject["BatchNorm"].GetObject();
        if (bnConfig.HasMember("momentum")) {
            bnMomentum->f = bnConfig["momentum"].GetFloat();
        }
    }
    const char* inputModeFileName = argv[1];
    FUNC_PRINT_ALL(inputModeFileName, s);
    std::map<std::string, VARP> inputVars;
    std::map<std::string, VARP> outputVars;
    {
        auto inputsOutputs = Variable::getInputAndOutput(Variable::loadMap(argv[1]));
        inputVars = inputsOutputs.first;
        outputVars = inputsOutputs.second;
    }
    Transformer::TrainConfig trainConfig;
    trainConfig.noUpdateOps = std::move(noUpdateOps);
    trainConfig.onlyUpdateOps = std::move(onlyUpdateOps);
    trainConfig.extraParams["BatchNorm"]["momentum"] = bnMomentum;
    auto turnTrainable = Train::TurnTrainable(trainConfig);
    turnTrainable.onExecute(Variable::mapToSequence(outputVars));
    auto trainInfo = turnTrainable.trainInfo;
    if (configObject.HasMember("Shape")) {
        auto shapeArray = configObject["Shape"].GetObject();
        for (auto shapeIter = shapeArray.begin(); shapeIter != shapeArray.end(); shapeIter++) {
            auto dimArray = shapeIter->value.GetArray();
            std::vector<int> dims;
            for (auto dimIter = dimArray.begin(); dimIter != dimArray.end(); dimIter++) {
                dims.emplace_back(dimIter->GetInt());
            }
            FUNC_PRINT_ALL(shapeIter->name.GetString(), s);
            std::string key = shapeIter->name.GetString();
            for (auto& varIter : inputVars) {
                auto var = varIter.second;
                if (var->name() == key) {
                    var->resize(dims);
                    break;
                }
            }
        }
    }
    auto exprs = Variable::getExecuteOrder(Variable::mapToSequence(outputVars));

    // Collect Const Variable
    std::set<VARP> parameters;
    for (auto v : exprs) {
        if (v->get() == nullptr && VARP::TRAINABLE == v->inputType()) {
            auto va = Variable::create(v, 0);
            parameters.insert(va);
        }
    }
    for (auto p : parameters) {
        p.fix(VARP::CONSTANT);
    }

    VARP loss;
    bool hasLoss = configObject.HasMember("Loss");
    if (!hasLoss) {
        auto output      = outputVars.begin()->second;
        auto outputShape = output->getInfo();
        if (outputShape->order == NC4HW4) {
            auto outputName = output->name();
            output->setName(outputName + "Origin");
            output      = _Convert(output, NHWC);
            outputShape = output->getInfo();
            output->setName(outputName);
        }
        auto outputReal = _Input(outputShape->dim, outputShape->order);
        outputReal->setName(output->name() + "_Compare");
#ifdef USE_ELU
        auto sub = _Subtract(output, outputReal);
        sub->setName(output->name() + "_Sub");
        loss = (_ReduceSum(_Multiply(sub, sub), {}));
#else
        auto mul = _Multiply(_Log(output), outputReal);
        mul->setName(output->name() + "_Mul");
        loss = _Negative(_ReduceSum(mul, {}));
#endif
        auto l2 = _Const(0.0f);
        for (auto var : parameters) {
            l2 = l2 + (var * var).sum({});
        }
        loss = loss + _Multiply(l2, _Const(0.0005f));
        loss->setName("Loss");
        exprs = Variable::getExecuteOrder({loss});
    } else {
        std::string lossName = configObject["Loss"].GetObject()["op"].GetString();
        for (auto expr : exprs) {
            if (expr->name() == lossName) {
                loss = Variable::create(expr);
                break;
            }
        }
        for (auto iter : outputVars) {
            if (iter.first == lossName) {
                outputVars.erase(iter.first);
                break;
            }
        }
    }
    MNN_ASSERT(nullptr != loss);
    auto gradMap = OpGrad::grad(loss, parameters, stopBackPropOps);
    // Make Update
    std::map<VARP, VARP> varUpdateMap;
    auto learningRate = _Input();
    learningRate->setName("LearningRate");
    auto weightDecay = _Input();
    weightDecay->setName("WeightDecay");

    auto step = _Scalar<float>(1.0f);
    step->setName("optimize_step");
    step.fix(VARP::TRAINABLE);
    auto stepPlus1 = step + _Scalar<float>(1.0f);
    stepPlus1->setName("optimize_step+1");
    varUpdateMap[step] = stepPlus1;

    std::map<std::string, std::string> extraInputs;
    extraInputs["LearningRate"] = "float";
    extraInputs["WeightDecay"] = "float";
    if (trainInfo["is_training_float"]->linkNumber() > 0) {
        extraInputs["is_training"] = "int, 0 or 1";
    }

    if (optimizerType == "SGD") {
        auto momentum = _Input();
        momentum->setName("Momentum");
        extraInputs["Momentum"] = "float";

        for (auto iter : gradMap) {
            auto p = iter.first;
            p.fix(VARP::TRAINABLE);
            auto grad = iter.second;
            grad->setName(p->name()+"_grad");

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

            auto pInfo = p->getInfo();
            auto pDims = pInfo->dim;

            auto l2grad = weightDecay * p;
            l2grad->setName(p->name() + "_l2grad");

            VARP gradWithDecay = nullptr;
            if (p->name().find("Weight") != string::npos) {
                gradWithDecay = grad + l2grad;
            } else {
                gradWithDecay = grad;
            }

            VARP history = _Const(0.0f, pDims, pInfo->order);
            history->setName(p->name() + "_momentum");
            history.fix(VARP::TRAINABLE);

            auto newHistory = gradWithDecay + momentum * history;
            newHistory->setName("update_" + history->name());

            auto finalGrad = learningRate * history;
            finalGrad->setName(p->name() + "_final_grad");

            auto updateValue = _Subtract(p, finalGrad);
            updateValue->setName("update_" + p->name());
            varUpdateMap[p] = updateValue;
            varUpdateMap[history] = newHistory;
        }
    } else if (optimizerType == "ADAM") {
        auto beta1 = _Input();
        beta1->setName("Beta1");
        auto beta2 = _Input();
        beta2->setName("Beta2");
        auto eps = _Input();
        eps->setName("Eps");

        extraInputs["Beta1"] = "float";
        extraInputs["Beta2"] = "float";
        extraInputs["Eps"] = "float";

        auto correction = _Sqrt(_Const(1.0f, {}, NCHW) - _Pow(beta2, step)) / (_Const(1.0f, {}, NCHW) - _Pow(beta1, step));
        correction->setName("correction");
        
        for (auto iter : gradMap) {
            auto p = iter.first;
            p.fix(VARP::TRAINABLE);
            auto grad = iter.second;
            grad->setName(p->name()+"_grad");

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

            auto pInfo = p->getInfo();
            auto pDims = pInfo->dim;

            auto l2grad = weightDecay * p;
            l2grad->setName(p->name() + "_l2grad");

            VARP gradWithDecay = nullptr;
            if (p->name().find("Weight") != string::npos) {
                gradWithDecay = grad + l2grad;
            } else {
                gradWithDecay = grad;
            }
        
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

            auto finalGrad = learningRate * correction * (history1 / (_Sqrt(history2 + _Scalar<float>(1e-8)) + eps));
            finalGrad->setName(p->name() + "_final_grad");

            auto updateValue = _Subtract(p, finalGrad);
            updateValue->setName("update_" + p->name());
            varUpdateMap[p] = updateValue;
            varUpdateMap[history1] = newHistory1;
            varUpdateMap[history2] = newHistory2;
        }
    } else {
        MNN_ERROR("error: don't support optimizer type: %s\n", optimizerType.c_str());
    }

    MNN_PRINT(">>>\nextra input tensors for %s:\n\n", optimizerType.c_str());
    for (auto& input : extraInputs) {
        MNN_PRINT("name: %s, \ttype: %s\n", input.first.c_str(), input.second.c_str());
    }
    MNN_PRINT("<<<\n");

    std::unique_ptr<MNN::NetT> netStruct(new MNN::NetT);
    std::vector<VARP> resultOutputs;
    for (auto output : outputVars) {
        resultOutputs.emplace_back(output.second);
    }
    resultOutputs.emplace_back(loss);
    for (auto iter : varUpdateMap) {
        resultOutputs.emplace_back(iter.second);
    }
    Variable::save(resultOutputs, ".grad.mnn");
    Variable::save(resultOutputs, netStruct.get());
    for (int i = 0; i < netStruct->oplists.size(); ++i) {
        auto& op = netStruct->oplists[i];
        for (auto iter : varUpdateMap) {
            if (iter.second->name() == op->name) {
                for (int j = 0; j < netStruct->oplists.size(); ++j) {
                    auto& opSub = netStruct->oplists[j];
                    if (opSub->name == iter.first->name()) {
                        auto indexOri = op->outputIndexes;
                        op->outputIndexes = opSub->outputIndexes;

                        if ((opSub->name.find("_BN_RunningMean_Weight") != string::npos) || (opSub->name.find("_BN_RunningVariance_Weight") != string::npos)) {
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
    netStruct->usage = MNN::Usage_TRAIN;

    {
        flatbuffers::FlatBufferBuilder builder(1024);
        auto offset = Net::Pack(builder, netStruct.get());
        builder.Finish(offset);
        // TODO, use FileWriter instead
        FILE* f = fopen(argv[2], "wb");
        fwrite(builder.GetBufferPointer(), 1, builder.GetSize(), f);
        fclose(f);
    }

    return 0;
}
