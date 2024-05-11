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
#include <MNN/AutoTime.hpp>
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
    Revert::fillRandValue(data, tensor->elementSize());
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
                           int numberThread = 4, int precision = 2, float sparsity = 0.0f, int sparseBlockOC = 1, bool testQuantModel=false) {
    auto revertor = std::unique_ptr<Revert>(new Revert(model.model_file.c_str()));
    if (testQuantModel) {
        revertor->initialize(0, sparseBlockOC, false, true);
    } else {
        revertor->initialize(sparsity, sparseBlockOC);
    }

    auto modelBuffer      = revertor->getBuffer();
    const auto bufferSize = revertor->getBufferSize();
    auto net = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromBuffer(modelBuffer, bufferSize), MNN::Interpreter::destroy);
    revertor.reset();
    net->setSessionMode(MNN::Interpreter::Session_Release);
    MNN::ScheduleConfig config;
    config.numThread = numberThread;
    config.type      = static_cast<MNNForwardType>(forward);
    MNN::BackendConfig backendConfig;
    backendConfig.precision = (MNN::BackendConfig::PrecisionMode)precision;
    backendConfig.power = MNN::BackendConfig::Power_High;
    config.backendConfig = &backendConfig;

    std::vector<float> costs;
    MNN::Session* session = net->createSession(config);

    MNN::Tensor* input    = net->getSessionInput(session, NULL);

    // if the model has not the input dimension, umcomment the below code to set the input dims
    // std::vector<int> dims{1, 3, 224, 224};
    // net->resizeTensor(input, dims);
    // net->resizeSession(session);

    net->releaseModel();

    const MNN::Backend* inBackend = net->getBackend(session, input);

    std::shared_ptr<MNN::Tensor> givenTensor(MNN::Tensor::createHostTensorFromDevice(input, false));

    auto outputTensor = net->getSessionOutput(session, NULL);
    std::shared_ptr<MNN::Tensor> expectTensor(MNN::Tensor::createHostTensorFromDevice(outputTensor, false));
    // Warming up...
    for (int i = 0; i < warmup; ++i) {
        void* host = input->map(MNN::Tensor::MAP_TENSOR_WRITE,  input->getDimensionType());
        input->unmap(MNN::Tensor::MAP_TENSOR_WRITE,  input->getDimensionType(), host);

        net->runSession(session);

        host = outputTensor->map(MNN::Tensor::MAP_TENSOR_READ,  outputTensor->getDimensionType());
        outputTensor->unmap(MNN::Tensor::MAP_TENSOR_READ,  outputTensor->getDimensionType(), host);
    }

    for (int round = 0; round < loop; round++) {
        MNN::Timer _t;
        void* host = input->map(MNN::Tensor::MAP_TENSOR_WRITE,  input->getDimensionType());
        input->unmap(MNN::Tensor::MAP_TENSOR_WRITE,  input->getDimensionType(), host);
        net->runSession(session);
        host = outputTensor->map(MNN::Tensor::MAP_TENSOR_READ,  outputTensor->getDimensionType());
        outputTensor->unmap(MNN::Tensor::MAP_TENSOR_READ,  outputTensor->getDimensionType(), host);
        auto time = (float)_t.durationInUs() / 1000.0f;
        costs.push_back(time);
    }
    return costs;
}

void displayStats(const std::string& name, const std::vector<float>& costs, int quant = 0) {
    float max = 0, min = FLT_MAX, sum = 0, avg;
    for (auto v : costs) {
        max = fmax(max, v);
        min = fmin(min, v);
        sum += v;
        //printf("[ - ] cost：%f ms\n", v);
    }
    avg = costs.size() > 0 ? sum / costs.size() : 0;
    std::string model = name;
    if (quant == 1) {
        model = "quant-" + name;
    }
    printf("[ - ] %-24s    max = %8.3f ms  min = %8.3f ms  avg = %8.3f ms\n", model.c_str(), max, avg == 0 ? 0 : min, avg);
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



#ifdef __ANDROID__
#include <errno.h>
#include <unistd.h>
#include <sys/prctl.h>
#include <sys/syscall.h>

#define BUFFER_SIZE 1024

static uint32_t getNumberOfCPU() {
    FILE* fp = fopen("/proc/cpuinfo", "rb");
    if (!fp) {
        return 1;
    }
    uint32_t number = 0;
    char buffer[BUFFER_SIZE];
    while (!feof(fp)) {
        char* str = fgets(buffer, BUFFER_SIZE, fp);
        if (!str) {
            break;
        }
        if (memcmp(buffer, "processor", 9) == 0) {
            number++;
        }
    }
    fclose(fp);
    if (number < 1) {
        number = 1;
    }
    return number;
}

static int getCPUMaxFreqKHz(int cpuID) {
    char path[256];
    sprintf(path, "/sys/devices/system/cpu/cpufreq/stats/cpu%d/time_in_state", cpuID);
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        sprintf(path, "/sys/devices/system/cpu/cpu%d/cpufreq/stats/time_in_state", cpuID);
        fp = fopen(path, "rb");
        if (!fp) {
            sprintf(path, "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", cpuID);
            fp = fopen(path, "rb");
            if (!fp) {
                return -1;
            }
            int maxfrequency = -1;
            fscanf(fp, "%d", &maxfrequency);
            fclose(fp);
            return maxfrequency;
        }
    }
    int maxfrequency = 0;
    while (!feof(fp)) {
        int frequency = 0;
        int history   = fscanf(fp, "%d %*d", &frequency);
        if (history != 1) {
            break;
        }
        if (frequency > maxfrequency) {
            maxfrequency = frequency;
        }
    }
    fclose(fp);
    return maxfrequency;
}

static int sortCPUIDByMaxFrequency(std::vector<int>& cpuIDs, int* littleClusterOffset) {
    const int cpuNumbers = cpuIDs.size();
    *littleClusterOffset = 0;
    if (cpuNumbers == 0) {
        return 0;
    }
    std::vector<int> cpusFrequency;
    cpusFrequency.resize(cpuNumbers);
    for (int i = 0; i < cpuNumbers; ++i) {
        int frequency    = getCPUMaxFreqKHz(i);
        cpuIDs[i]        = i;
        cpusFrequency[i] = frequency;
        // MNN_PRINT("cpu fre: %d, %d\n", i, frequency);
    }
    for (int i = 0; i < cpuNumbers; ++i) {
        for (int j = i + 1; j < cpuNumbers; ++j) {
            if (cpusFrequency[i] < cpusFrequency[j]) {
                // id
                int temp  = cpuIDs[i];
                cpuIDs[i] = cpuIDs[j];
                cpuIDs[j] = temp;
                // frequency
                temp             = cpusFrequency[i];
                cpusFrequency[i] = cpusFrequency[j];
                cpusFrequency[j] = temp;
            }
        }
    }
    int midMaxFrequency = (cpusFrequency.front() + cpusFrequency.back()) / 2;
    if (midMaxFrequency == cpusFrequency.back()) {
        return 0;
    }
    for (int i = 0; i < cpuNumbers; ++i) {
        if (cpusFrequency[i] < midMaxFrequency) {
            *littleClusterOffset = i;
            break;
        }
    }
    return 0;
}


//#define CPU_SETSIZE 1024
#define __NCPUBITS  (8 * sizeof (unsigned long))

#endif

void set_cpu_affinity()
{
#ifdef __ANDROID__
    int cpu_core_num = sysconf(_SC_NPROCESSORS_CONF);
    //LOG_MCNN_CL_INF("cpu core num = %d\n", cpu_core_num);
    int cpu_id = 0;
    cpu_set_t mask;
    CPU_ZERO(&mask);

    auto numberOfCPUs = getNumberOfCPU();
    static std::vector<int> sortedCPUIDs;
    static int littleClusterOffset = 0;
    if (sortedCPUIDs.empty()) {
        sortedCPUIDs.resize(numberOfCPUs);
        for (int i = 0; i < numberOfCPUs; ++i) {
            sortedCPUIDs[i] = i;
        }
        sortCPUIDByMaxFrequency(sortedCPUIDs, &littleClusterOffset);
    }

    printf("max core:");
    for (cpu_id = 0; cpu_id < littleClusterOffset; cpu_id++)
    {
        printf("%d ", sortedCPUIDs[cpu_id]);
        CPU_SET(sortedCPUIDs[cpu_id], &mask);
    }
    printf("\n");


    int sys_call_res = syscall(__NR_sched_setaffinity, gettid(), sizeof(mask), &mask);
    //LOG_MCNN_CL_INF("sys call res = %d\n", sys_call_res);
    if (sys_call_res)
    {
        printf("set_cpu_affinity errno = %d\n", (int)errno);
    }
#endif
}

#if TARGET_OS_IPHONE
void iosBenchAll(const char* modelPath) {
    std::cout << "MNN benchmark" << std::endl;
    int loop               = 20;
    int warmup             = 10;
    MNNForwardType forward = MNN_FORWARD_CPU;
    forward = MNN_FORWARD_NN;
    int numberThread       = 4;
    int precision = 2;
    std::cout << "Forward type: **" << forwardType(forward) << "** thread=" << numberThread << "** precision=" <<precision << std::endl;
    std::vector<Model> models = findModelFiles(modelPath);
    std::cout << "--------> Benchmarking... loop = " << loop << ", warmup = " << warmup << std::endl;

    for (auto& m : models) {
        std::vector<float> costs = doBench(m, loop, warmup, forward, false, numberThread, precision);
        displayStats(m.name, costs);
    }
}
#else
int main(int argc, const char* argv[]) {
    std::cout << "MNN benchmark" << std::endl;
    int loop               = 10;
    int warmup             = 10;
    MNNForwardType forward = MNN_FORWARD_CPU;
    int testQuantizedModel = 0;
    int numberThread       = 4;
    int precision = 2;
    float sparsity = 0.0f;
    int sparseBlockOC = 1;
    if (argc <= 2) {
        std::cout << "Usage: " << argv[0] << " models_folder [loop_count] [warmup] [forwardtype] [numberThread] [precision] [weightSparsity] [testQuantizedModel]" << std::endl;
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
    if (argc >= 7) {
        precision = atoi(argv[6]);
    }
    if (argc >= 8) {
        sparsity = atof(argv[7]);
    }
    if(argc >= 9) {
        sparseBlockOC = atoi(argv[8]);
    }
    if(argc >= 10) {
        testQuantizedModel = atoi(argv[9]);
    }

    std::cout << "Forward type: " << forwardType(forward) << " thread=" << numberThread << " precision=" <<precision << " sparsity=" <<sparsity << " sparseBlockOC=" << sparseBlockOC << " testQuantizedModel=" << testQuantizedModel << std::endl;
    std::vector<Model> models = findModelFiles(argv[1]);

    std::cout << "--------> Benchmarking... loop = " << argv[2] << ", warmup = " << warmup << std::endl;
    std::string fpInfType = "precision!=2, use fp32 inference.";
    if (precision == 2) {
        fpInfType = "precision=2, use fp16 inference if your device supports and open MNN_ARM82=ON.";
    }
    MNN_PRINT("[-INFO-]: %s\n", fpInfType.c_str());
    if (testQuantizedModel) {
        MNN_PRINT("[-INFO-]: Auto set sparsity=0 when test quantized model in benchmark...\n");
    }

    /* not called yet */
    // set_cpu_affinity();
    if (testQuantizedModel) {
        printf("Auto set sparsity=0 when test quantized model in benchmark...\n");
    }

    for (auto& m : models) {
        std::vector<float> costs = doBench(m, loop, warmup, forward, false, numberThread, precision, sparsity, sparseBlockOC, false);
        displayStats(m.name.c_str(), costs, false);
        if (testQuantizedModel) {
            costs = doBench(m, loop, warmup, forward, false, numberThread, precision, sparsity, sparseBlockOC, true);
            displayStats(m.name, costs, 1);
        }
    }
}
#endif
