//
//  timeProfile.cpp
//  MNN
//
//  Created by MNN on 2019/01/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#define MNN_OPEN_TIME_TRACE
#include <stdlib.h>
#include <cstring>
#include <memory>
#include <string>
#include "AutoTime.hpp"
#include "Interpreter.hpp"
#include "MNNDefine.h"
#include "Macro.h"
#include "Profiler.hpp"
#include "Tensor.hpp"
#include "revertMNNModel.hpp"

using namespace MNN;

int main(int argc, const char* argv[]) {
    std::string cmd = argv[0];
    std::string pwd = "./";
    auto rslash     = cmd.rfind("/");
    if (rslash != std::string::npos) {
        pwd = cmd.substr(0, rslash + 1);
    }

    // read args
    const char* fileName = argv[1];
    int runTime          = 100;
    if (argc > 2) {
        runTime = ::atoi(argv[2]);
    }
    auto type = MNN_FORWARD_CPU;
    if (argc > 3) {
        type = (MNNForwardType)atoi(argv[3]);
        printf("Use extra forward type: %d\n", type);
    }

    // input dims
    std::vector<int> inputDims;
    if (argc > 4) {
        std::string inputShape(argv[4]);
        const char* delim = "x";
        std::ptrdiff_t p1 = 0, p2;
        while (1) {
            p2 = inputShape.find(delim, p1);
            if (p2 != std::string::npos) {
                inputDims.push_back(atoi(inputShape.substr(p1, p2 - p1).c_str()));
                p1 = p2 + 1;
            } else {
                inputDims.push_back(atoi(inputShape.substr(p1).c_str()));
                break;
            }
        }
    }
    for (auto dim : inputDims) {
        MNN_PRINT("%d ", dim);
    }
    MNN_PRINT("\n");

    // revert MNN model if necessary
    auto revertor = std::unique_ptr<Revert>(new Revert(fileName));
    revertor->initialize();
    auto modelBuffer = revertor->getBuffer();
    auto bufferSize  = revertor->getBufferSize();

    // create net
    MNN_PRINT("Open Model %s\n", fileName);
    auto net = std::shared_ptr<Interpreter>(Interpreter::createFromBuffer(modelBuffer, bufferSize));
    if (nullptr == net) {
        return 0;
    }
    revertor.reset();

    // create session
    MNN::ScheduleConfig config;
    config.type           = type;
    MNN::Session* session = NULL;
    session               = net->createSession(config);
    auto inputTensor      = net->getSessionInput(session, NULL);
    if (!inputDims.empty()) {
        net->resizeTensor(inputTensor, inputDims);
        net->resizeSession(session);
    }
    net->releaseModel();
    std::shared_ptr<MNN::Tensor> inputTensorUser(MNN::Tensor::createHostTensorFromDevice(inputTensor, false));
    auto outputTensor = net->getSessionOutput(session, NULL);
    if (outputTensor->size() <= 0) {
        MNN_ERROR("Output not available\n");
        return 0;
    }
    std::shared_ptr<MNN::Tensor> outputTensorUser(MNN::Tensor::createHostTensorFromDevice(outputTensor, false));

    auto profiler      = MNN::Profiler::getInstance();
    auto beginCallBack = [&](const std::vector<Tensor*>& inputs, const OperatorInfo* info) {
        profiler->start(info);
        return true;
    };
    auto afterCallBack = [&](const std::vector<Tensor*>& inputs, const OperatorInfo* info) {
        profiler->end(info);
        return true;
    };

    AUTOTIME;
    // just run
    for (int i = 0; i < runTime; ++i) {
        inputTensor->copyFromHostTensor(inputTensorUser.get());
        net->runSessionWithCallBackInfo(session, beginCallBack, afterCallBack);
        outputTensor->copyToHostTensor(outputTensorUser.get());
    }

    profiler->printTimeByType(runTime);
    return 0;
}
