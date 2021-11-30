//
//  aoa_nlu_encoder.cpp
//  MNN
//
//  Created by MNN on b'2021/09/06'.
//  Copyright © 2018 - 2021, Alibaba Group Holding Limited
//

#include <fstream>
#include <sstream>
#include <stdio.h>
#include<string.h>

#include <MNN/AutoTime.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/MNNDefine.h>
#include <MNN/Tensor.hpp>
#include "../../tools/cpp/revertMNNModel.hpp"

using namespace MNN;
using namespace std;


void RunNLU(std::string modelName, int loop, int warmup, int sequenceLength, float sparsity, int sparseBlockOC) {

    auto revertor = std::unique_ptr<Revert>(new Revert(modelName.c_str()));
    revertor->initialize(sparsity, sparseBlockOC, true);
    auto modelBuffer      = revertor->getBuffer();
    const auto bufferSize = revertor->getBufferSize();
    auto net = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromBuffer(modelBuffer, bufferSize));
    revertor.reset();
    net->setSessionMode(MNN::Interpreter::Session_Release);
    MNN::ScheduleConfig config;
    config.numThread = 1;
    config.type      = static_cast<MNNForwardType>(MNN_FORWARD_CPU);
    MNN::BackendConfig backendConfig;
    backendConfig.precision = MNN::BackendConfig::Precision_Normal;
    backendConfig.power = MNN::BackendConfig::Power_High;
    config.backendConfig = &backendConfig;

    MNN::Session* session = net->createSession(config);
    if (nullptr == session) {
        return;
    }

    Tensor* inputTensor = net->getSessionInput(session, NULL);
    MNN_PRINT("origin Input shape:\n");
    inputTensor->printShape();


    auto allInput = net->getSessionInputAll(session);
    MNN_PRINT("search all input tensors dims: %zu\n", allInput.size());
    for (auto& iter : allInput) {
        // MNN_PRINT("tensor name:%s\n", iter.first.c_str());
        // MNN_PRINT("origin shape:\t");
        // iter.second->printShape();
        auto oneTensor = iter.second;
        auto originShape = oneTensor->shape();
        for (auto& ilength :originShape) {
            ilength = ilength < 0 ? sequenceLength : ilength;
        }
        net->resizeTensor(oneTensor, originShape);
        // MNN_PRINT("resize new shape:");
        // oneTensor->printShape();

    }
    net->resizeSession(session);

    {
        auto allOutputs = net->getSessionOutputAll(session);
        for (auto& iter : allOutputs) {
            // MNN_PRINT("output name: %s\n", iter.first.c_str());
        }
    }

    float memoryUsage = 0.0f;
    net->getSessionInfo(session, MNN::Interpreter::MEMORY, &memoryUsage);
    float flops = 0.0f;
    net->getSessionInfo(session, MNN::Interpreter::FLOPS, &flops);
    int backendType[2];
    net->getSessionInfo(session, MNN::Interpreter::BACKENDS, backendType);
    MNN_PRINT("Session Info: memory use %f MB, flops is %f M, backendType is %d\n", memoryUsage, flops, backendType[0]);

    allInput = net->getSessionInputAll(session);
    for (auto& iter : allInput) {
        auto inputTensor = iter.second;
        auto size = inputTensor->size();
        if (size <= 0) {
            MNN_PRINT("skip memset tensor:%s",  iter.first.c_str());
            continue;
        }
        MNN::Tensor tempTensor(inputTensor, inputTensor->getDimensionType());
        ::memset(tempTensor.host<void>(), 0, tempTensor.size());
        inputTensor->copyFromHostTensor(&tempTensor);
    }
    net->releaseModel();


    for (int i = 0; i < warmup; ++i) {
        net->runSession(session);
    }

    Timer timer;
    timer.reset();
    for (int round = 0; round < loop; round++) {
        net->runSession(session);
    }
    float averageTime =  (timer.durationInUs() / 1000.0f) / loop;
    MNN_PRINT("benchmark sparsed aoa v3 sparsity:%f, block:1x%d warmup loops:%d, run loops:%d\nname = %s, avg = %.3f ms\n",
        sparsity, sparseBlockOC, warmup, loop, modelName.c_str(), averageTime);

    return;
}

int main(int argc, const char* argv[]) {
    if (argc < 2) {
        MNN_ERROR("Don't has model name\n");
        return 0;
    }
    int loop = 1;
    int warmup = 1;
    int sequenceLength = 128;
    float sparsity = 0.0f;
    int sparseBlockOC = 1;
    if (argc >= 3) {
        loop = atoi(argv[2]);
    }
    if (argc >= 4) {
        warmup = atoi(argv[3]);
    }
    if (argc >= 5) {
        sequenceLength = atoi(argv[4]);
    }

    if(argc >= 6) {
        sparsity = atof(argv[5]);
    }

    if(argc >= 7) {
        sparseBlockOC = atoi(argv[6]);
    }

    BackendConfig config;
    // Executor::getGlobalExecutor()->setGlobalExecutorConfig(MNN_FORWARD_CPU, config, 1);
    std::string modelName = argv[1];

    RunNLU(modelName, loop, warmup, sequenceLength, sparsity, sparseBlockOC);

    return 0;
}
