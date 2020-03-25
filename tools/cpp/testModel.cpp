//
//  testModel.cpp
//  MNN
//
//  Created by MNN on 2019/01/22.
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
#include <fstream>
#include <map>
#include <sstream>
#include "core/Backend.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

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
    const char* modelPath  = argv[1];
    const char* givenName  = argv[2];
    const char* expectName = argv[3];
    MNN_PRINT("Testing model %s, input: %s, output: %s\n", modelPath, givenName, expectName);

    // create net
    auto type = MNN_FORWARD_CPU;
    if (argc > 4) {
        type = (MNNForwardType)stringConvert<int>(argv[4]);
    }
    auto tolerance = 0.1f;
    if (argc > 5) {
        tolerance = stringConvert<float>(argv[5]);
    }
    std::shared_ptr<MNN::Interpreter> net =
        std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(modelPath));

    // create session
    MNN::ScheduleConfig config;
    config.type = type;
    MNN::BackendConfig backendConfig;
    if (type != MNN_FORWARD_CPU) {
        // Use Precision_High for other backend
        // Test CPU ARM v8.2 and other approciate method
        backendConfig.precision = MNN::BackendConfig::Precision_High;
    }
    config.backendConfig = &backendConfig;
    auto session         = net->createSession(config);

    auto allInput = net->getSessionInputAll(session);
    for (auto& iter : allInput) {
        auto size = iter.second->size();

        auto bnType   = MNN_FORWARD_CPU;
        auto tensorBn = MNN::TensorUtils::getDescribe(iter.second)->backend;
        if (tensorBn) {
            bnType = tensorBn->type();
        }
        // memory is fp16, but size == element * sizeof(float)
        if (bnType == MNN_FORWARD_CPU_EXTENSION) {
            size /= 2;
        }

        auto ptr = iter.second->host<void>();
        std::shared_ptr<MNN::Tensor> tempTensor;
        if (nullptr == ptr) {
            tempTensor = std::shared_ptr<MNN::Tensor>(MNN::Tensor::createHostTensorFromDevice(iter.second, false),
                                                      [&iter](void* t) {
                                                          auto hostTensor = (MNN::Tensor*)t;
                                                          iter.second->copyFromHostTensor(hostTensor);
                                                          delete hostTensor;
                                                      });
            ptr        = tempTensor->host<float>();
        }
        ::memset(ptr, 0, size);
    }

    // write input tensor
    auto inputTensor = net->getSessionInput(session, NULL);
    auto givenTensor = createTensor(inputTensor, givenName);
    if (!givenTensor) {
#if defined(_MSC_VER)
        printf("Failed to open input file %s.\n", givenName);
#else
        printf(RED "Failed to open input file %s.\n" NONE, givenName);
#endif
        return -1;
    }
    inputTensor->copyFromHostTensor(givenTensor);
    delete givenTensor;

    // infer
    net->runSession(session);

    // read expect tensor
    auto outputTensor = net->getSessionOutput(session, NULL);
    std::shared_ptr<MNN::Tensor> expectTensor(createTensor(outputTensor, expectName));
    if (!expectTensor.get()) {
#if defined(_MSC_VER)
        printf("Failed to open expect file %s.\n", expectName);
#else
        printf(RED "Failed to open expect file %s.\n" NONE, expectName);
#endif
        return -1;
    }

    // compare output with expect
    bool correct = MNN::TensorUtils::compareTensors(outputTensor, expectTensor.get(), tolerance, true);
    if (correct) {
#if defined(_MSC_VER)
        printf("Test %s Correct!\n", modelPath);
#else
        printf(GREEN BOLD "Test %s Correct!\n" NONE, modelPath);
#endif
    } else {
#if defined(_MSC_VER)
        printf("Test Failed %s!\n", modelPath);
#else
        printf(RED "Test Failed %s!\n" NONE, modelPath);
#endif
    }
    return 0;
}
