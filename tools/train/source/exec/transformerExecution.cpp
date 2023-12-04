//
//  transformerExecution.cpp
//  MNN
//
//  Created by MNN on 2019/05/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/ExprCreator.hpp>
#include "ParameterOptimizer.hpp"
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
#include "flatbuffers/idl.h"
#include "flatbuffers/minireflect.h"
#include "flatbuffers/util.h"
#include "TrainInfo_generated.h"
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
        MNN_PRINT("Usage: ./transformer.out temp.bin dst.bin config.json [revertInfo.json]\n");
        return 0;
    }
    std::string revertConfigFile = "revert.json";
    if (argc >= 5) {
        revertConfigFile = argv[4];
    }
    FUNC_PRINT_ALL(revertConfigFile.c_str(), s);
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
    std::vector<std::string> fixAsConstOps;
    std::vector<std::vector<std::string>> weightNameGroups;
    std::vector<MNN::Express::VARP> lrNames;
    if (configObject.HasMember("Optimizer")) {
        auto optimizer = configObject["Optimizer"].GetObject();
        if (optimizer.HasMember("OnlyUpdateOps")) {
            auto limitArray = optimizer["OnlyUpdateOps"].GetArray();
            for (auto vIter = limitArray.begin(); vIter != limitArray.end(); vIter++) {
                onlyUpdateOps.emplace_back(vIter->GetString());
                MNN_PRINT("will only update: %s \n", vIter->GetString());
            }
        }
        if (optimizer.HasMember("NoUpdateOps")) {
            auto limitArray = optimizer["NoUpdateOps"].GetArray();
            for (auto vIter = limitArray.begin(); vIter != limitArray.end(); vIter++) {
                noUpdateOps.emplace_back(vIter->GetString());
                if (onlyUpdateOps.empty())
                    MNN_PRINT("will not update: %s \n", vIter->GetString());
            }
        }
        if (optimizer.HasMember("StopBackPropOps")) {
            auto limitArray = optimizer["StopBackPropOps"].GetArray();
            for (auto vIter = limitArray.begin(); vIter != limitArray.end(); vIter++) {
                stopBackPropOps.emplace_back(vIter->GetString());
                MNN_PRINT("will stop back prop from (also not update this op): %s \n", vIter->GetString());
            }
        }
        if (optimizer.HasMember("type")) {
            optimizerType = std::string(optimizer["type"].GetString());
            MNN_PRINT("optimizer type: %s\n", optimizerType.c_str());
        }
        if (optimizer.HasMember("FixAsConstOps")) {
            auto limitArray = optimizer["FixAsConstOps"].GetArray();
            for (auto vIter = limitArray.begin(); vIter != limitArray.end(); vIter++) {
                fixAsConstOps.emplace_back(vIter->GetString());
                MNN_PRINT("this op will be fixed as Const, and maybe turn to Trainable later: %s \n", vIter->GetString());
            }
        }
        if (optimizer.HasMember("ParameterOptConfig")) {
            auto pConf = optimizer["ParameterOptConfig"].GetArray();
            for (auto vIter = pConf.begin(); vIter != pConf.end(); vIter++) {
                auto conf = vIter->GetObject();
                if (conf.HasMember("WeightNames") && conf.HasMember("LrName")) {
                    auto wn = conf["WeightNames"].GetArray();
                    std::vector<std::string> wNames;
                    for (auto wIter = wn.begin(); wIter != wn.end(); wIter++) {
                        wNames.push_back(wIter->GetString());
                    }
                    weightNameGroups.push_back(wNames);
                    auto lr = _Input({}, NCHW);
                    lr->setName(conf["LrName"].GetString());
                    lrNames.push_back(lr);
                }
            }
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
    MNN::Usage netUsage;
    {
        // Load usage
        std::shared_ptr<MNN::Interpreter> net(MNN::Interpreter::createFromFile(argv[1]));
        auto buffer = net->getModelBuffer();
        auto netStruct = flatbuffers::GetRoot<MNN::Net>(buffer.first);
        netUsage = netStruct->usage();
    }
    if (Usage_INFERENCE_STATIC == netUsage) {
        Executor::getGlobalExecutor()->setLazyComputeMode(MNN::Express::Executor::LAZY_CONTENT);
    }
    {
        auto inputsOutputs = Variable::getInputAndOutput(Variable::loadMap(argv[1]));
        inputVars = inputsOutputs.first;
        outputVars = inputsOutputs.second;
    }
    for (auto& varIter : inputVars) {
        auto var = varIter.second;
        auto varInfo = var->getInfo();
        auto vDims = varInfo->dim;
        
        if (!fixAsConstOps.empty()) {
            if (std::find(fixAsConstOps.begin(), fixAsConstOps.end(), var->name()) != fixAsConstOps.end()) {
                var.fix(VARP::CONSTANT);
            }
        }
    }
    Transformer::TrainConfig trainConfig;
    trainConfig.noUpdateOps = std::move(noUpdateOps);
    trainConfig.onlyUpdateOps = std::move(onlyUpdateOps);
    trainConfig.extraParams["BatchNorm"]["momentum"] = bnMomentum;
    auto turnTrainable = Train::TurnTrainable(trainConfig);
    turnTrainable.onExecute(Variable::mapToSequence(outputVars));
    {
        // Save Train Revert Info
        std::unique_ptr<MNNTrain::TrainInfoT> trainInfo(new MNNTrain::TrainInfoT);
        for (auto& bnIter : turnTrainable.mTrainInfo.bnVariables) {
            std::unique_ptr<MNNTrain::KVT> kv(new MNNTrain::KVT);
            kv->key = bnIter.first;
            kv->value = bnIter.second->name();
            trainInfo->batchnormal.emplace_back(std::move(kv));
        }
        for (auto& iter : turnTrainable.mTrainInfo.trainables) {
            std::unique_ptr<MNNTrain::KVT> kv(new MNNTrain::KVT);
            kv->key = iter.first;
            kv->value = iter.second;
            trainInfo->trainables.emplace_back(std::move(kv));
        }
        for (auto& iter : turnTrainable.mTrainInfo.convolutionVariables) {
            std::unique_ptr<MNNTrain::OpInfoT> kv(new MNNTrain::OpInfoT);
            kv->op = iter.first;
            kv->weight = iter.second.first;
            kv->bias = iter.second.second;
            trainInfo->convolutions.emplace_back(std::move(kv));
        }
        flatbuffers::FlatBufferBuilder builder;
        builder.Finish(MNNTrain::TrainInfo::Pack(builder, trainInfo.get()));
        std::ofstream _t(revertConfigFile.c_str());
        auto s = flatbuffers::FlatBufferToString((const uint8_t*)builder.GetBufferPointer(), MNNTrain::TrainInfoTypeTable());
        _t << s;
    }
    auto trainInfo = turnTrainable.mTrainInfo.bnVariables;
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
    bool train = configObject.HasMember("Train");
    if (!train) {
        MNN_PRINT("Don't has member Train, generate grad model\n");
    }
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
        if (nullptr == loss.get()) {
            MNN_ERROR("Can't find loss op\n");
            return 0;
        }
    }
    auto lossInfo = loss->getInfo();
    MNN_ASSERT(nullptr != loss);
    auto gradMap = OpGrad::grad(loss, parameters, stopBackPropOps);
    if (gradMap.empty()) {
        MNN_ERROR("Grad error, don't has grad\n");
        return 0;
    }
    for (auto iter : gradMap) {
        if (!iter.first->name().empty()) {
            iter.second->setName(iter.first->name() + "::grad");
        }
    }
    if (!train) {
        std::vector<MNN::Express::VARP> gradVars = {loss};
        for (auto iter : gradMap) {
            iter.first.fix(VARP::INPUT);
            gradVars.emplace_back(iter.second);
        }
        ParameterOptimizer::makeLoopModel(argv[2], gradVars, std::make_pair(std::vector<MNN::Express::VARP>{}, std::vector<MNN::Express::VARP>{}));
        return 0;
    }
    // Make Update
    std::shared_ptr<MNN::Train::ParameterOptimizer> optimizer;
    if (optimizerType == "SGD") {
        optimizer.reset(MNN::Train::ParameterOptimizer::createSGD(nullptr, 0.01f, 0.90f, 0.00f, MNN::Train::ParameterOptimizer::L1));
    } else if (optimizerType == "ADAM") {
        optimizer.reset(MNN::Train::ParameterOptimizer::createADAM(nullptr, 0.01f, 0.90f, 0.999f, 0.00f, 0.00005f, MNN::Train::ParameterOptimizer::L1));
    }

    auto learningRate = _Input({}, NCHW);
    learningRate->setName("LearningRate");
    std::vector<ParameterOptimizer::ParameterOptGrad> gradVars;
    for (auto iter : gradMap) {
        ParameterOptimizer::ParameterOptGrad gradVar;
        gradVar.parameter = iter.first;
        gradVar.parameterGrad = iter.second;
        gradVar.learningRate = learningRate;
        if (!lrNames.empty()) {
            // Find lr Index
            auto pName = iter.first->name();
            for (int ii = 0; ii < weightNameGroups.size(); ii++) {
                if (std::find(weightNameGroups[ii].begin(), weightNameGroups[ii].end(), pName) != weightNameGroups[ii].end()) {
                    gradVar.learningRate = lrNames[ii];
                    break;
                }
            }
        }
        gradVars.emplace_back(gradVar);
    }
    auto loopPair = optimizer->onMakeParameterUpdateGraphByGrad(gradVars);

    std::unique_ptr<MNN::NetT> netStruct(new MNN::NetT);
    std::vector<VARP> resultOutputs = {loss};
    ParameterOptimizer::makeLoopModel(argv[2], resultOutputs, loopPair);
    return 0;
}
