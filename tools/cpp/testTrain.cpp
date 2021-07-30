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
    MNN::BackendConfig::PrecisionMode precision = MNN::BackendConfig::Precision_High;
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
    auto lossTensor = net->getSessionOutput(session, lossName);
    std::vector<float> loss;
    for (int i=0; i<2; ++i) {
        for (auto iter = dataArray.begin(); iter != dataArray.end(); iter++) {
            auto dataName = std::string(dirPath) + "/" + std::string(iter->GetString());
            auto varMap = MNN::Express::Variable::load(dataName.c_str());
            if (varMap.empty()) {
                continue;
            }
            net->getSessionInput(session, lR)->host<float>()[0] = learnRate;
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
            net->runSession(session);
            loss.emplace_back(lossTensor->host<float>()[0]);
        }
    }
    bool correct = false;
    if (loss.size() < 2) {
        printf("Test Failed, data invalid %s!\n", modelPath.c_str());
        return 0;
    }
    auto firstLoss = loss[0];
    auto lastLoss = loss[(int)loss.size() - 1];
    if (lastLoss < firstLoss * decay) {
        printf("From %f -> %f, Test %s Correct!\n", firstLoss, lastLoss, modelPath.c_str());
    } else {
        printf("From %f -> %f, Test Failed %s!\n", firstLoss, lastLoss, modelPath.c_str());
    }
    return 0;
}
