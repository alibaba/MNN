//
//  testTrain.cpp
//  MNN
//
//  Created by MNN on 2021/06/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#define MNN_OPEN_TIME_TRACE

#include <MNN/MNNDefine.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <MNN/AutoTime.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <MNN/expr/Expr.hpp>
#include <fstream>
#include <map>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <set>
#include <algorithm>
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/prettywriter.h"

#define NONE "\e[0m"
#define RED "\e[0;31m"
#define GREEN "\e[0;32m"
#define L_GREEN "\e[1;32m"
#define BLUE "\e[0;34m"
#define L_BLUE "\e[1;34m"
#define BOLD "\e[1m"

template<typename T>
inline T stringConvert(const char* number) {
    std::istringstream os(number);
    T v;
    os >> v;
    return v;
}

MNN::Tensor* createTensor(const MNN::Tensor* shape, const char* path) {
    std::ifstream stream(path);
    if (stream.fail()) {
        return NULL;
    }

    auto result = new MNN::Tensor(shape, shape->getDimensionType());
    auto data   = result->host<float>();
    for (int i = 0; i < result->elementSize(); ++i) {
        double temp = 0.0f;
        stream >> temp;
        data[i] = temp;
    }
    stream.close();
    return result;
}

int main(int argc, const char* argv[]) {
    // check given & expect
    if (argc < 3) {
        return 0;
    }
    const char* jsonPath    = argv[1];
    const char* dirPath     = argv[2];
    rapidjson::Document document;
    {
        std::ifstream fileNames(jsonPath);
        std::ostringstream output;
        output << fileNames.rdbuf();
        auto outputStr = output.str();
        document.Parse(outputStr.c_str());
        if (document.HasParseError()) {
            MNN_ERROR("Invalid json\n");
            return 0;
        }
    }
    auto picObj = document.GetObject();
    float learnRate;
    if (document.HasMember("LearningRate")) {
        learnRate = document["LearningRate"].GetFloat();
    }
    auto modelPath = std::string(dirPath) + "/" + picObj["Model"].GetString();
    auto lossName = picObj["Loss"].GetString();
    auto inputName = picObj["Input"].GetString();
    auto targetName = picObj["Target"].GetString();
    auto dataArray = picObj["Data"].GetArray();
    auto lR = picObj["LR"].GetString();
    auto decay = picObj["Decay"].GetFloat();
    
    // create net
    auto type = MNN_FORWARD_CPU;
    MNN::BackendConfig::PrecisionMode precision = MNN::BackendConfig::Precision_Low;
    std::shared_ptr<MNN::Interpreter> net =
        std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(modelPath.c_str()));

    // create session
    MNN::ScheduleConfig config;
    config.type = type;
    config.saveTensors.emplace_back(lossName);
    MNN::BackendConfig backendConfig;
    backendConfig.precision = precision;
    config.backendConfig = &backendConfig;
    auto session         = net->createSession(config);
    if (nullptr == net->getSessionInput(session, inputName) || nullptr == net->getSessionInput(session, targetName) || nullptr == net->getSessionInput(session, lR) || nullptr == net->getSessionOutput(session, lossName)) {
        MNN_ERROR("Invalid model for train\n");
        return 0;
    }
    static bool gDebug = false;
    bool onlyInfer = false;
    auto lossTensor = net->getSessionOutput(session, lossName);
    std::vector<float> loss;
    MNN::TensorCallBack beforeCallBack = [&](const std::vector<MNN::Tensor*>& ntensors, const std::string& opName) {
        return true;
    };
    MNN::TensorCallBack callBack = [&](const std::vector<MNN::Tensor*>& ntensors, const std::string& opName) {
        for (int i = 0; i < ntensors.size(); ++i) {
            auto ntensor    = ntensors[i];
            if (onlyInfer && ntensor == lossTensor) {
                return false;
            }
            if (ntensor->getType().code != halide_type_float) {
                continue;
            }
            if (gDebug) {
                auto outDimType = ntensor->getDimensionType();
                auto expectTensor = new MNN::Tensor(ntensor, outDimType);
                ntensor->copyToHostTensor(expectTensor);
                auto size = expectTensor->elementSize();
                float summer = 0.0f;
                for (int i=0; i<size; ++i) {
                    summer += expectTensor->host<float>()[i];
                }
                delete expectTensor;
                MNN_PRINT("For op %s, summer=%f\n", opName.c_str(), summer);
            }
        }
        return true;
    };
    auto lrTensor = net->getSessionInput(session, lR);
    std::shared_ptr<MNN::Tensor> userLR(new MNN::Tensor(lrTensor, lrTensor->getDimensionType()));
    
    int runTime = 2;
    for (int i=0; i<runTime; ++i) {
        onlyInfer = i == (runTime-1);
        for (auto iter = dataArray.begin(); iter != dataArray.end(); iter++) {
            auto dataName = std::string(dirPath) + "/" + std::string(iter->GetString());
            auto varMap = MNN::Express::Variable::load(dataName.c_str());
            if (varMap.empty()) {
                continue;
            }
            userLR->host<float>()[0] = learnRate;
            lrTensor->copyFromHostTensor(userLR.get());
            for (auto v : varMap) {
                auto target = net->getSessionInput(session, v->name().c_str());
                if (nullptr == target) {
                    MNN_ERROR("Invalid data %s\n", v->name().c_str());
                    continue;
                }
                std::shared_ptr<MNN::Tensor> targetUser(new MNN::Tensor(target, target->getDimensionType()));
                ::memcpy(targetUser->host<void>(), v->readMap<void>(), targetUser->size());
                target->copyFromHostTensor(targetUser.get());
            }
            net->runSessionWithCallBack(session, beforeCallBack, callBack);
            std::shared_ptr<MNN::Tensor> lossTemp(new MNN::Tensor(lossTensor, lossTensor->getDimensionType()));
            lossTensor->copyToHostTensor(lossTemp.get());
            loss.emplace_back(lossTemp->host<float>()[0]);
        }
    }
    bool correct = false;
    if (loss.size() < 2) {
        printf("Test Failed, data invalid %s!\n", modelPath.c_str());
        return 0;
    }
    auto firstLoss = loss[0];
    auto lastLoss = loss[(int)loss.size() - 1];
    bool validFirst = firstLoss < 0.0f || firstLoss >= 0.0f;
    bool validLast = lastLoss < 0.0f || lastLoss >= 0.0f;
    MNN_PRINT("Loss from %f -> %f\n", firstLoss, lastLoss);
    bool lossValid = lastLoss < firstLoss * decay;
    if (!lossValid) {
        MNN_PRINT("Invalid loss decrease\n");
        return 0;
    }
    // Test Update
    net->updateSessionToModel(session);
    auto buffer = net->getModelBuffer();
    config.path.mode = MNN::ScheduleConfig::Path::Tensor;
    config.path.outputs.emplace_back(lossName);
    std::shared_ptr<MNN::Interpreter> newNet(MNN::Interpreter::createFromBuffer(buffer.first, buffer.second), MNN::Interpreter::destroy);
    net.reset();
    net = newNet;
    session = net->createSession(config);
    lossTensor = net->getSessionOutput(session, lossName);
    onlyInfer = true;
    lrTensor = net->getSessionInput(session, lR);
    for (auto iter = dataArray.begin(); iter != dataArray.end(); iter++) {
        auto dataName = std::string(dirPath) + "/" + std::string(iter->GetString());
        auto varMap = MNN::Express::Variable::load(dataName.c_str());
        if (varMap.empty()) {
            continue;
        }
        userLR->host<float>()[0] = learnRate;
        lrTensor->copyFromHostTensor(userLR.get());
        for (auto v : varMap) {
            auto target = net->getSessionInput(session, v->name().c_str());
            if (nullptr == target) {
                MNN_ERROR("Invalid data %s\n", v->name().c_str());
                continue;
            }
            std::shared_ptr<MNN::Tensor> targetUser(new MNN::Tensor(target, target->getDimensionType()));
            ::memcpy(targetUser->host<void>(), v->readMap<void>(), targetUser->size());
            target->copyFromHostTensor(targetUser.get());
        }
        net->runSessionWithCallBack(session, beforeCallBack, callBack);
        {
            std::shared_ptr<MNN::Tensor> lossTemp(new MNN::Tensor(lossTensor, lossTensor->getDimensionType()));
            lossTensor->copyToHostTensor(lossTemp.get());
            auto newLoss = lossTemp->host<float>()[0];
            MNN_PRINT("Update and reload, loss from %f -> %f\n", lastLoss, newLoss);
            if (newLoss > lastLoss + 0.1f) {
                MNN_ERROR("newLoss not valid\n");
                return 0;
            }
        }
    }
    MNN_PRINT("Test %s Correct!\n", modelPath.c_str());
    return 0;
}
