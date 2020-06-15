//
//  benchmark.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>
#if defined(_MSC_VER)
#include <Windows.h>
#undef min
#undef max
#else
#include <sys/time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#endif

#include "core/Backend.hpp"
#include <MNN/Interpreter.hpp>
#include <MNN/MNNDefine.h>
#include <MNN/Tensor.hpp>
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

#if !defined(_MSC_VER)
inline bool file_exist(const char* file) {
    struct stat buffer;
    return stat(file, &buffer) == 0;
}
#endif

std::vector<Model> findModelFiles(const char* dir) {
    std::vector<Model> models;
#if defined(_MSC_VER)
    WIN32_FIND_DATA ffd;
    HANDLE hFind = INVALID_HANDLE_VALUE;
    std::string mnn_model_pattern = std::string(dir) + "\\*.mnn"; 
    hFind = FindFirstFile(mnn_model_pattern.c_str(), &ffd);
    if (INVALID_HANDLE_VALUE == hFind) {
        std::cout << "open " << dir << " failed: " << strerror(errno) << std::endl;
        return models;
    }
    do {
        Model m;
        m.name       = ffd.cFileName;
        m.model_file = std::string(dir) + "\\" + m.name;
        if(INVALID_FILE_ATTRIBUTES != GetFileAttributes(m.model_file.c_str()) && GetLastError() != ERROR_FILE_NOT_FOUND) {
            models.push_back(std::move(m));
        }
    } while (FindNextFile(hFind, &ffd) != 0);
    FindClose(hFind);
#else
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
#endif
    return models;
}

void setInputData(MNN::Tensor* tensor) {
    float* data = tensor->host<float>();
    for (int i = 0; i < tensor->elementSize(); i++) {
        data[i] = Revert::getRandValue();
    }
}

static inline uint64_t getTimeInUs() {
    uint64_t time;
#if defined(_MSC_VER)
    LARGE_INTEGER now, freq;
    QueryPerformanceCounter(&now);
    QueryPerformanceFrequency(&freq);
    uint64_t sec = now.QuadPart / freq.QuadPart;
    uint64_t usec = (now.QuadPart % freq.QuadPart) * 1000000 / freq.QuadPart;
    time = sec * 1000000 + usec;
#else
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    time = static_cast<uint64_t>(tv.tv_sec) * 1000000 + tv.tv_usec;
#endif
    return time;
}

std::vector<float> doBench(Model& model, int loop, int warmup = 10, int forward = MNN_FORWARD_CPU, bool only_inference = true,
                           int numberThread = 4, int precision = 2) {
    auto revertor = std::unique_ptr<Revert>(new Revert(model.model_file.c_str()));
    revertor->initialize();
    auto modelBuffer      = revertor->getBuffer();
    const auto bufferSize = revertor->getBufferSize();
    auto net = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromBuffer(modelBuffer, bufferSize));
    revertor.reset();
    MNN::ScheduleConfig config;
    config.numThread = numberThread;
    config.type      = static_cast<MNNForwardType>(forward);
    MNN::BackendConfig backendConfig;
    backendConfig.precision = (MNN::BackendConfig::PrecisionMode)precision;
    backendConfig.power = MNN::BackendConfig::Power_High;
    config.backendConfig = &backendConfig;

    std::vector<float> costs;
    MNN::Session* session = net->createSession(config);
    net->releaseModel();
    MNN::Tensor* input    = net->getSessionInput(session, NULL);

    // if the model has not the input dimension, umcomment the below code to set the input dims
    // std::vector<int> dims{1, 3, 224, 224};
    // net->resizeTensor(input, dims);
    // net->resizeSession(session);

    const MNN::Backend* inBackend = net->getBackend(session, input);

    std::shared_ptr<MNN::Tensor> givenTensor(MNN::Tensor::createHostTensorFromDevice(input, false));

    auto outputTensor = net->getSessionOutput(session, NULL);
    std::shared_ptr<MNN::Tensor> expectTensor(MNN::Tensor::createHostTensorFromDevice(outputTensor, false));
    // Warming up...
    for (int i = 0; i < warmup; ++i) {
        input->copyFromHostTensor(givenTensor.get());
        net->runSession(session);
        outputTensor->copyToHostTensor(expectTensor.get());
    }

    for (int round = 0; round < loop; round++) {
        auto timeBegin = getTimeInUs();

        input->copyFromHostTensor(givenTensor.get());
        net->runSession(session);
        outputTensor->copyToHostTensor(expectTensor.get());

        auto timeEnd = getTimeInUs();
        costs.push_back((timeEnd - timeBegin) / 1000.0);
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
    int warmup             = 10;
    MNNForwardType forward = MNN_FORWARD_CPU;
    int numberThread       = 4;
    if (argc <= 2) {
        std::cout << "Usage: " << argv[0] << " models_folder [loop_count] [warmup] [forwardtype] [numberThread] [precision]" << std::endl;
        return 1;
    }
    if (argc >= 3) {
        loop = atoi(argv[2]);
    }
    if (argc >= 4) {
        warmup = atoi(argv[3]);
    }
    if (argc >= 5) {
        forward = static_cast<MNNForwardType>(atoi(argv[4]));
    }
    if (argc >= 6) {
        numberThread = atoi(argv[5]);
    }
    int precision = 2;
    if (argc >= 7) {
        precision = atoi(argv[6]);
    }
    std::cout << "Forward type: **" << forwardType(forward) << "** thread=" << numberThread << "** precision=" <<precision << std::endl;
    std::vector<Model> models = findModelFiles(argv[1]);

    std::cout << "--------> Benchmarking... loop = " << argv[2] << ", warmup = " << warmup << std::endl;
    for (auto& m : models) {
        std::vector<float> costs = doBench(m, loop, warmup, forward, false, numberThread, precision);
        displayStats(m.name, costs);
    }
}
