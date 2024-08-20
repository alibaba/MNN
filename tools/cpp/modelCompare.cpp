//
//  modelCompare.cpp
//  MNN
//
//  Created by MNN on 2019/01/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#define MNN_OPEN_TIME_TRACE

#include <math.h>
#include <stdlib.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <MNN/AutoTime.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include "core/TensorUtils.hpp"
#include "rapidjson/document.h"

template<typename T>
inline T stringConvert(const char* number) {
    std::istringstream os(number);
    T v;
    os >> v;
    return v;
}

using namespace MNN;

static void compareNet(Interpreter* net, Interpreter* net2, MNNForwardType expectType, float tolerance,
                              const std::map<std::string, std::shared_ptr<Tensor>>& inputs, const std::string& stopOp, BackendConfig::PrecisionMode precision, int modeNum) {
    std::vector<std::shared_ptr<MNN::Tensor>> correctResult;
    int index;
    MNN::ScheduleConfig expectConfig;
    BackendConfig backendConfig;
    backendConfig.precision = precision;
    expectConfig.type   = expectType;
    expectConfig.backendConfig = &backendConfig;
    expectConfig.mode = modeNum;
    auto expectSession  = net->createSession(expectConfig);
    auto compareSession = net2->createSession(expectConfig);

    bool allCorrect = true;

    MNN::TensorCallBackWithInfo beginCallBack = [&](const std::vector<MNN::Tensor*>& t, const OperatorInfo* op) {
        if (op->name() == stopOp) {
            return false;
        }
        return true;
    };
    MNN::TensorCallBackWithInfo saveExpect = [&](const std::vector<MNN::Tensor*>& t, const OperatorInfo* op) {
        if (op->name() == stopOp) {
            return false;
        }
        if (op->type() == "Raster") {
            return true;
        }
        for (int i=0; i<t.size(); ++i) {
            auto tensor = t[i];
            if (tensor->elementSize() <= 0) {
                return true;
            }
            if (tensor->buffer().device == 0 && tensor->buffer().host == nullptr) {
                return true;
            }
            std::shared_ptr<MNN::Tensor> copyTensor(MNN::Tensor::createHostTensorFromDevice(tensor, true));
            correctResult.emplace_back(copyTensor);
        }
        return true;
    };
    MNN::TensorCallBackWithInfo compareExpect = [&](const std::vector<MNN::Tensor*>& t, const OperatorInfo* op) {
        if (op->name() == stopOp) {
            return false;
        }
        if (op->type() == "Raster") {
            return true;
        }
        for (int i=0; i<t.size(); ++i) {
            auto tensor = t[i];
            if (tensor->elementSize() <= 0) {
                return true;
            }
            if (tensor->buffer().device == 0 && tensor->buffer().host == nullptr) {
                return true;
            }
            std::shared_ptr<MNN::Tensor> copyTensor(MNN::Tensor::createHostTensorFromDevice(tensor, true));
            auto expectTensor = correctResult[index++];
            auto correct      = TensorUtils::compareTensors(copyTensor.get(), expectTensor.get(), tolerance, true);
            if (!correct) {
                MNN_PRINT("%s - %d is error\n", op->name().c_str(), i);
                allCorrect = false;
            }
        }
        return allCorrect;
    };

    for (auto& iter : inputs) {
        Tensor* expectInput = net->getSessionInput(expectSession, iter.first.empty() ? NULL : iter.first.c_str());
        expectInput->copyFromHostTensor(iter.second.get());
        Tensor* compareInput = net->getSessionInput(compareSession, iter.first.empty() ? NULL : iter.first.c_str());
        compareInput->copyFromHostTensor(iter.second.get());
    }
    correctResult.clear();
    net->runSessionWithCallBackInfo(expectSession, beginCallBack, saveExpect);
    index = 0;
    net2->runSessionWithCallBackInfo(compareSession, beginCallBack, compareExpect);
    if (allCorrect) {
        MNN_PRINT("Correct ! Run second pass\n");
    } else {
        return;
    }
    index = 0;
    for (auto& iter : inputs) {
        Tensor* compareInput = net->getSessionInput(compareSession, iter.first.empty() ? NULL : iter.first.c_str());
        compareInput->copyFromHostTensor(iter.second.get());
    }
    net2->runSessionWithCallBackInfo(compareSession, beginCallBack, compareExpect);
    if (allCorrect) {
        MNN_PRINT("Correct !\n");
    }
}

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        MNN_PRINT("Usage: ./modelCompare.out origin.mnn origin_quant.mnn [0.05]");
    }
    // read args
    std::string cmd = argv[0];
    std::string pwd = "./";
    auto rslash     = cmd.rfind("/");
    if (rslash != std::string::npos) {
        pwd = cmd.substr(0, rslash + 1);
    }

    const char* fileName = argv[1];

    const char* compareFileName = argv[2];

    float tolerance = 0.05f;
    if (argc > 3) {
        tolerance = stringConvert<float>(argv[3]);
    }
    MNN_PRINT("Tolerance Rate: %f\n", tolerance);

    // create net
    MNN_PRINT("Open Model %s, %s\n", fileName, compareFileName);
    std::shared_ptr<MNN::Interpreter> net =
        std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(fileName));
    net->setSessionMode(Interpreter::Session_Debug);
    std::shared_ptr<MNN::Interpreter> net2 =
        std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(compareFileName));
    net2->setSessionMode(Interpreter::Session_Debug);

    // create session for get input info
    ScheduleConfig config;
    config.type  = MNN_FORWARD_CPU;
    auto session = net->createSession(config);

    std::map<std::string, std::shared_ptr<MNN::Tensor>> inputs;
    std::vector<std::string> inputNames;
    do {
        rapidjson::Document document;
        std::ostringstream jsonNameOs;
        jsonNameOs << pwd << "/input.json";
        std::ifstream fileNames(jsonNameOs.str().c_str());
        if (fileNames.fail()) {
            break;
        }
        std::ostringstream output;
        output << fileNames.rdbuf();
        auto outputStr = output.str();
        document.Parse(outputStr.c_str());
        if (document.HasParseError()) {
            MNN_ERROR("Invalid json\n");
            break;
        }
        if (document.HasMember("inputs")) {
            auto inputsInfo = document["inputs"].GetArray();
            for (auto iter = inputsInfo.begin(); iter !=inputsInfo.end(); iter++) {
                auto obj = iter->GetObject();
                std::string name = obj["name"].GetString();
                inputNames.emplace_back(name);
            }
        }
    } while (false);
    if (!inputNames.empty()) {
        MNN_PRINT("Find input.json, use inputs:");
        for (auto& n : inputNames) {
            MNN_PRINT(" %s, ", n.c_str());
        }
        MNN_PRINT("\n");
        for (auto name : inputNames) {
            auto inputTensor = net->getSessionInput(session, name.c_str());
            std::shared_ptr<MNN::Tensor> givenTensor(new Tensor(inputTensor, inputTensor->getDimensionType()));
            {
                std::ostringstream fileName;
                fileName << pwd << name << ".txt";
                std::ifstream input(fileName.str().c_str());
                MNN_ASSERT(!input.fail());

                int size_w = inputTensor->width();
                int size_h = inputTensor->height();
                int bpp    = inputTensor->channel();
                int batch  = inputTensor->batch();
                // auto backend = net->getBackend(session, inputTensor);
                // MNN_ASSERT(!input.fail());
                MNN_PRINT("Input: %d,%d,%d,%d\n", size_w, size_h, bpp, batch);
                auto inputData = givenTensor->host<float>();
                auto size      = givenTensor->size() / sizeof(float);
                for (int i = 0; i < size; ++i) {
                    input >> inputData[i];
                }
                inputs.insert(std::make_pair(name, givenTensor));
            }

        }
    } else {
        auto inputTensor = net->getSessionInput(session, NULL);
        std::shared_ptr<MNN::Tensor> givenTensor(new Tensor(inputTensor, inputTensor->getDimensionType()));
        {
            std::ostringstream fileName;
            fileName << pwd << "input_0"
                     << ".txt";
            std::ifstream input(fileName.str().c_str());

            int size_w = inputTensor->width();
            int size_h = inputTensor->height();
            int bpp    = inputTensor->channel();
            int batch  = inputTensor->batch();
            // auto backend = net->getBackend(session, inputTensor);
            // MNN_ASSERT(!input.fail());
            MNN_PRINT("Input: %d,%d,%d,%d\n", size_w, size_h, bpp, batch);
            auto inputData = givenTensor->host<float>();
            auto size      = givenTensor->size() / sizeof(float);
            for (int i = 0; i < size; ++i) {
                input >> inputData[i];
            }
            inputs.insert(std::make_pair("", givenTensor));
        }
    }
    net->releaseSession(session);
    BackendConfig::PrecisionMode precision = BackendConfig::Precision_Normal;
    if (argc > 4) {
        precision = (BackendConfig::PrecisionMode)atoi(argv[4]);
    }
    FUNC_PRINT(precision);
    int modeNum = 1;
    if(argc > 5) {
        modeNum = atoi(argv[5]);//set gpu mode
    }
    FUNC_PRINT(modeNum);
    std::string stopOp = "";
    if (argc > 6) {
        stopOp = argv[6];
    }
    FUNC_PRINT_ALL(stopOp.c_str(), s);
    compareNet(net.get(), net2.get(), MNN_FORWARD_CPU, tolerance, inputs, stopOp, precision, modeNum);

    return 0;
}
