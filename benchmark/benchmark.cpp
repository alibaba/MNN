//
//  benchmark.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <dirent.h>
#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include "Backend.hpp"
#include "Interpreter.hpp"
#include "MNNDefine.h"
#include "Tensor.hpp"
#include "revertMNNModel.hpp"
/**
 TODOs:
 1. dynamically get CPU related info.
 2. iOS support
 */
struct Model {
    std::string name;
    std::string model_file;
};

inline bool file_exist(const char* file) {
    struct stat buffer;
    return stat(file, &buffer) == 0;
}

std::vector<Model> findModelFiles(const char* dir) {
    std::vector<Model> models;

    DIR* root;
    if ((root = opendir(dir)) == NULL) {
        std::cout << "open " << dir << " failed: " << strerror(errno) << std::endl;
        return models;
    }

    struct dirent* ent;
    while ((ent = readdir(root)) != NULL) {
        Model m;
        if (ent->d_name[0] != '.') {
            m.name       = ent->d_name;
            m.model_file = std::string(dir) + "/" + m.name;
            if (file_exist(m.model_file.c_str())) {
                models.push_back(std::move(m));
            }
        }
    }
    closedir(root);
    return models;
}

void setInputData(MNN::Tensor* tensor) {
    float* data = tensor->host<float>();
    for (int i = 0; i < tensor->elementSize(); i++) {
        data[i] = Revert::getRandValue();
    }
}

std::vector<float> doBench(Model& model, int loop, int forward = MNN_FORWARD_CPU, bool only_inference = true,
                           int numberThread = 4) {
    auto revertor = std::unique_ptr<Revert>(new Revert(model.model_file.c_str()));
    revertor->initialize();
    auto modelBuffer      = revertor->getBuffer();
    const auto bufferSize = revertor->getBufferSize();
    auto net = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromBuffer(modelBuffer, bufferSize));
    MNN::ScheduleConfig config;
    config.numThread = numberThread;
    config.type      = static_cast<MNNForwardType>(forward);

    std::vector<float> costs;
    MNN::Session* session = net->createSession(config);
    MNN::Tensor* input    = net->getSessionInput(session, NULL);

    // if the model has not the input dimension, umcomment the below code to set the input dims
    // std::vector<int> dims{1, 3, 224, 224};
    // net->resizeTensor(input, dims);
    // net->resizeSession(session);

    const MNN::Backend* inBackend = net->getBackend(session, input);

    std::shared_ptr<MNN::Tensor> givenTensor(new MNN::Tensor(input, input->getDimensionType()));

    auto outputTensor = net->getSessionOutput(session, NULL);
    MNN::Tensor expectTensor(outputTensor, outputTensor->getDimensionType());
    // Warming up...
    for (int i = 0; i < 3; ++i) {
        net->runSession(session);
    }

    for (int round = 0; round < loop; round++) {
        struct timeval time_begin, time_end;
        gettimeofday(&time_begin, NULL);

        inBackend->onCopyBuffer(givenTensor.get(), input);
        net->runSession(session);
        outputTensor->copyToHostTensor(&expectTensor);

        gettimeofday(&time_end, NULL);
        costs.push_back((time_end.tv_sec - time_begin.tv_sec) * 1000.0 +
                        (time_end.tv_usec - time_begin.tv_usec) / 1000.0);
    }
    return costs;
}

void displayStats(const std::string& name, const std::vector<float>& costs) {
    float max = 0, min = FLT_MAX, sum = 0, avg;
    for (auto v : costs) {
        max = fmax(max, v);
        min = fmin(min, v);
        sum += v;
    }
    avg = costs.size() > 0 ? sum / costs.size() : 0;
    printf("[ - ] %-24s    max = %8.3fms  min = %8.3fms  avg = %8.3fms\n", name.c_str(), max, avg == 0 ? 0 : min, avg);
}
static inline std::string forwardType(MNNForwardType type) {
    switch (type) {
        case MNN_FORWARD_CPU:
            return "CPU";
        case MNN_FORWARD_VULKAN:
            return "Vulkan";
        case MNN_FORWARD_OPENCL:
            return "OpenCL";
        case MNN_FORWARD_METAL:
            return "Metal";
        default:
            break;
    }
    return "N/A";
}
int main(int argc, const char* argv[]) {
    std::cout << "MNN benchmark" << std::endl;
    int loop               = 10;
    MNNForwardType forward = MNN_FORWARD_CPU;
    int numberThread       = 4;
    if (argc <= 2) {
        std::cout << "Usage: " << argv[0] << " models_folder [loop_count] [forwardtype]" << std::endl;
        return 1;
    }
    if (argc >= 3) {
        loop = atoi(argv[2]);
    }
    if (argc >= 4) {
        forward = static_cast<MNNForwardType>(atoi(argv[3]));
    }
    if (argc >= 5) {
        numberThread = atoi(argv[4]);
    }
    std::cout << "Forward type: **" << forwardType(forward) << "** thread=" << numberThread << std::endl;
    std::vector<Model> models = findModelFiles(argv[1]);

    std::cout << "--------> Benchmarking... loop = " << argv[2] << std::endl;
    for (auto& m : models) {
        std::vector<float> costs = doBench(m, loop, forward, false, numberThread);
        displayStats(m.name, costs);
    }
}
