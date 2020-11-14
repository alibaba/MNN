#include <iostream>
#include <cstdio>
#include <string>
#include <vector>
#include <cfloat>
#include <map>
#include <cstring>
#include <cstdlib>
#if defined(_MSC_VER)
#include <Windows.h>
#undef min
#undef max
#else
#include <sys/time.h>
#include <sys/stat.h>
#include <sys/types.h>
#endif

#include "MNN_generated.h"
#include <MNN/MNNForwardType.h>
#include <MNN/Interpreter.hpp>
#include <MNN/expr/Expr.hpp>
#include "ExprModels.hpp"

using namespace MNN;
using namespace MNN::Express;

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

static std::vector<std::string> splitArgs(const std::string& args, const std::string& delimiter) {
    std::vector<std::string> result;
    size_t pos = 0, nextPos = args.find(delimiter, 0);
    while (nextPos != std::string::npos) {
        result.push_back(args.substr(pos, nextPos - pos));
        pos = nextPos + delimiter.length();
        nextPos = args.find(delimiter, pos);
    }
    result.push_back(args.substr(pos, args.length() - pos));
    return result;
}

static void displayStats(const std::string& name, const std::vector<float>& costs) {
    float max = 0, min = FLT_MAX, sum = 0, avg;
    for (auto v : costs) {
        max = max < v ? v : max;
        min = min > v ? v : min;
        sum += v;
    }
    avg = costs.size() > 0 ? sum / costs.size() : 0;
    printf("[ - ] %-24s    max = %8.3fms  min = %8.3fms  avg = %8.3fms\n", name.c_str(), max, avg == 0 ? 0 : min, avg);
}

static std::vector<float> runNet(VARP netOutput, const ScheduleConfig& config, int loop) {
    std::unique_ptr<NetT> netTable(new NetT);
    Variable::save({netOutput}, netTable.get());
    flatbuffers::FlatBufferBuilder builder(1024);
    auto offset = CreateNet(builder, netTable.get());
    builder.Finish(offset);
    const void* buf = builder.GetBufferPointer();
    size_t size = builder.GetSize();
    std::unique_ptr<Interpreter> net(Interpreter::createFromBuffer(buf, size));
    net->setSessionMode(MNN::Interpreter::Session_Release);
    auto session = net->createSession(config);
    net->releaseModel();
    auto inputTensor = net->getSessionInput(session, NULL);
    std::shared_ptr<Tensor> inputTensorHost(Tensor::createHostTensorFromDevice(inputTensor, false));
    int eleSize = inputTensorHost->elementSize();
    for (int i = 0; i < eleSize; ++i) {
        inputTensorHost->host<float>()[i] = 0.0f;
    }
    auto outputTensor = net->getSessionOutput(session, NULL);
    std::shared_ptr<Tensor> outputTensorHost(Tensor::createHostTensorFromDevice(outputTensor, false));

    // Warming up...
    for (int i = 0; i < 3; ++i) {
        inputTensor->copyFromHostTensor(inputTensorHost.get());
        net->runSession(session);
        outputTensor->copyToHostTensor(outputTensorHost.get());
    }

    std::vector<float> costs;

    // start run
    for (int i = 0; i < loop; ++i) {
        auto timeBegin = getTimeInUs();

        inputTensor->copyFromHostTensor(inputTensorHost.get());
        net->runSession(session);
        outputTensor->copyToHostTensor(outputTensorHost.get());

        auto timeEnd = getTimeInUs();
        costs.push_back((timeEnd - timeBegin) / 1000.0);
    }
    return costs;
}

static void _printHelp() {
    std::cout << "Usage: " << " model_to_benchmark [loop_count] [forwardtype] [numberThread]" << std::endl;
    std::cout << "model_to_benchmark: " << std::endl;
    std::cout << "\t default: run standard models" << std::endl;
    std::cout << "\t MobileNetV1_{numClass}_{width}_{resolution}, width: {1.0, 0.75, 0.5, 0.25}, resolution: {224, 192, 160, 128}, e.g: MobileNetV1_100_1.0_224" << std::endl;
    std::cout << "\t MobileNetV2_{numClass}, e.g: MobileNetV2_100" << std::endl;
    std::cout << "\t ResNet_{numClass}_{layer}, layer: {18, 34, 50, 101, 152}, e.g: ResNet_100_18" << std::endl;
    std::cout << "\t GoogLeNet_{numClass}, e.g: GoogLeNet_100" << std::endl;
    std::cout << "\t SqueezeNet_{numClass}, e.g: SqueezeNet_100" << std::endl;
    std::cout << "\t ShuffleNet_{numClass}_{group}, group: [1, 2, 3, 4, 8], e.g: ShuffleNet_100_4" << std::endl;
}
static std::vector<std::string> gDefaultModels = {
    "MobileNetV1_1000_1.0_224",
    "MobileNetV2_1000",
    "GoogLeNet_1000",
    "ShuffleNet_1000_4",
    "SqueezeNet_1000",
    "ResNet_1000_18",
    "ResNet_1000_50",
};

int main(int argc, const char* argv[]) {
    std::cout << "MNN Expr Models benchmark" << std::endl;
    size_t loop = 10;
    MNNForwardType forward = MNN_FORWARD_CPU;
    size_t numThread = 4;
    if (argc <= 1) {
        _printHelp();
        return 0;
    }
    if (((argc > 1) && (strcmp(argv[1], "help") == 0)) || argc > 5) {
        _printHelp();
        return 0;
    }
    std::vector<std::string> models;
    if (((argc > 1) && (strcmp(argv[1], "default") == 0)) || argc > 5) {
        models = gDefaultModels;
    } else {
        models = {argv[1]};
    }

    if (argc >= 3) {
        loop = atoi(argv[2]);
    }
    if (argc >= 4) {
        forward = static_cast<MNNForwardType>(atoi(argv[3]));
    }
    if (argc >= 5) {
        numThread = atoi(argv[4]);
    }
    std::cout << "Forward type: **" << forwardType(forward) << "** thread=" << numThread << std::endl;
    ScheduleConfig config;
    config.type = forward;
    config.numThread = numThread;
    BackendConfig bnConfig;
    bnConfig.precision = BackendConfig::Precision_Low;
    bnConfig.power = BackendConfig::Power_High;
    config.backendConfig = &bnConfig;

    std::vector<float> costs;

    // ResNet18 benchmark
    for (auto model : models) {
        auto modelArgs = splitArgs(model.c_str(), "_");
        auto modelType = modelArgs[0];
        int numClass   = atoi(modelArgs[1].c_str());
        if (modelType == "MobileNetV1") {
            auto mobileNetWidthType = EnumMobileNetWidthTypeByString(modelArgs[2]);
            if (mobileNetWidthType < 0) {
                std::cout << "Not support MobileNetWidthType " << modelArgs[2] << std::endl;
                std::cout << "Only [1.0, 0.75, 0.5, 0.25] be support" << std::endl;
                return 1;
            }
            auto mobileNetResolutionType = EnumMobileNetResolutionTypeByString(modelArgs[3]);
            if (mobileNetResolutionType < 0) {
                std::cout << "Not support MobileNetResolutionType " << modelArgs[3] << std::endl;
                std::cout << "Only [224, 192, 160, 128] be support" << std::endl;
                return 1;
            }
            costs = runNet(mobileNetV1Expr(mobileNetWidthType, mobileNetResolutionType, numClass), config, loop);
        } else if (modelType == "MobileNetV2") {
            costs = runNet(mobileNetV2Expr(numClass), config, loop);
        } else if (modelType == "ResNet") {
            auto resNetType = EnumResNetTypeByString(modelArgs[2]);
            if (resNetType < 0) {
                std::cout << "Not support ResNet layer " << modelArgs[2] << std::endl;
                std::cout << "Only [18, 34, 50, 101, 152] be support" << std::endl;
                return 1;
            }
            costs = runNet(resNetExpr(resNetType, numClass), config, loop);
        } else if (modelType == "GoogLeNet") {
            costs = runNet(googLeNetExpr(numClass), config, loop);
        } else if (modelType == "SqueezeNet") {
            costs = runNet(squeezeNetExpr(numClass), config, loop);
        } else if (modelType == "ShuffleNet") {
            int group = atoi(modelArgs[2].c_str());
            costs = runNet(shuffleNetExpr(group, numClass), config, loop);
        } else {
            std::cout << "Not support Model Type " << modelType << std::endl;
            continue;
        }
        displayStats(model.c_str(), costs);
    }
    return 0;
}
