//
//  backendTest.cpp
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
#include "AutoTime.hpp"
#include "Interpreter.hpp"
#include "Tensor.hpp"
#include "TensorUtils.hpp"

using namespace MNN;

static void compareForwadType(Interpreter* net, MNNForwardType expectType, MNNForwardType compareType, float tolerance,
                              const std::map<std::string, Tensor*>& inputs, const std::string& stopOp, BackendConfig::PrecisionMode precision) {
    std::map<std::string, std::shared_ptr<MNN::Tensor>> correctResult;

    MNN::ScheduleConfig expectConfig, compareConfig;
    BackendConfig backendConfig;
    backendConfig.precision = precision;
    expectConfig.type   = expectType;
    compareConfig.type  = compareType;
    compareConfig.backendConfig = &backendConfig;
    auto expectSession  = net->createSession(expectConfig);
    auto compareSession = net->createSession(compareConfig);

    bool allCorrect = true;

    MNN::TensorCallBack beginCallBack = [&](const std::vector<MNN::Tensor*>& t, const std::string& op) {
        if (op == stopOp) {
            return false;
        }
        return true;
    };
    MNN::TensorCallBack saveExpect = [&](const std::vector<MNN::Tensor*>& t, const std::string& op) {
        if (op == stopOp) {
            return false;
        }

        auto tensor = t[0];
        if (tensor->elementSize() <= 0) {
            return true;
        }
        std::shared_ptr<MNN::Tensor> copyTensor(new MNN::Tensor(tensor, tensor->getDimensionType()));
        tensor->copyToHostTensor(copyTensor.get());
        correctResult.insert(std::make_pair(op, copyTensor));
        return true;
    };
    MNN::TensorCallBack compareExpect = [&](const std::vector<MNN::Tensor*>& t, const std::string& op) {
        if (op == stopOp) {
            return false;
        }
        auto tensor = t[0];
        if (tensor->elementSize() <= 0) {
            return true;
        }
        std::shared_ptr<MNN::Tensor> copyTensor(new MNN::Tensor(tensor, tensor->getDimensionType()));
        tensor->copyToHostTensor(copyTensor.get());
        auto expectTensor = correctResult.find(op)->second;
        auto correct      = TensorUtils::compareTensors(copyTensor.get(), expectTensor.get(), tolerance, true);
        if (!correct) {
            MNN_PRINT("%s is error\n", op.c_str());
            allCorrect = false;
        }
        return correct;
    };

    for (auto& iter : inputs) {
        Tensor* expectInput = net->getSessionInput(expectSession, iter.first.empty() ? NULL : iter.first.c_str());
        expectInput->copyFromHostTensor(iter.second);
        Tensor* compareInput = net->getSessionInput(compareSession, iter.first.empty() ? NULL : iter.first.c_str());
        compareInput->copyFromHostTensor(iter.second);
    }
    net->runSessionWithCallBack(expectSession, beginCallBack, saveExpect);
    net->runSessionWithCallBack(compareSession, beginCallBack, compareExpect);
    net->releaseSession(expectSession);
    net->releaseSession(compareSession);
    if (allCorrect) {
        MNN_PRINT("Correct !\n");
    }
}

int main(int argc, const char* argv[]) {
    // read args
    std::string cmd = argv[0];
    std::string pwd = "./";
    auto rslash     = cmd.rfind("/");
    if (rslash != std::string::npos) {
        pwd = cmd.substr(0, rslash + 1);
    }

    const char* fileName = argv[1];

    auto type = MNN_FORWARD_CPU;
    if (argc > 2) {
        type = (MNNForwardType)atoi(argv[2]);
    }
    MNN_PRINT("Test forward type: %d\n", type);

    float tolerance = 0.05f;
    if (argc > 3) {
        tolerance = atof(argv[3]);
    }
    MNN_PRINT("Tolerance Rate: %f\n", tolerance);

    // create net
    MNN_PRINT("Open Model %s\n", fileName);
    std::shared_ptr<MNN::Interpreter> net =
        std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(fileName));

    // create session
    ScheduleConfig config;
    config.type  = MNN_FORWARD_CPU;
    auto session = net->createSession(config);

    std::map<std::string, MNN::Tensor*> inputs;

    auto inputTensor = net->getSessionInput(session, NULL);
    MNN::Tensor givenTensor(inputTensor, inputTensor->getDimensionType());
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
        auto inputData = givenTensor.host<float>();
        auto size      = givenTensor.size() / sizeof(float);
        for (int i = 0; i < size; ++i) {
            input >> inputData[i];
        }
        inputs.insert(std::make_pair("", &givenTensor));
    }
    BackendConfig::PrecisionMode precision = BackendConfig::Precision_Normal;
    if (argc > 4) {
        precision = (BackendConfig::PrecisionMode)atoi(argv[4]);
    }
    FUNC_PRINT(precision);
    std::string stopOp = "";
    if (argc > 5) {
        stopOp = argv[5];
    }
    FUNC_PRINT_ALL(stopOp.c_str(), s);
    compareForwadType(net.get(), MNN_FORWARD_CPU, type, tolerance, inputs, stopOp, precision);

    return 0;
}
