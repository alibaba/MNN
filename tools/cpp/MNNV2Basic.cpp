//
//  MNNV2Basic.cpp
//  MNN
//
//  Created by MNN on 2019/01/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#define MNN_OPEN_TIME_TRACE

#include <stdlib.h>
#include <sys/time.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include "AutoTime.hpp"
#include "Interpreter.hpp"
#include "MNNDefine.h"
#include "Tensor.hpp"
#include "revertMNNModel.hpp"

//#define FEED_INPUT_NAME_VALUE

using namespace MNN;

static void dumpTensor2File(const Tensor* tensor, const char* file) {
    std::ofstream outputOs(file);
    auto type = tensor->getType();

    int dimension = tensor->buffer().dimensions;
    int width     = 1;
    if (dimension > 1) {
        width = tensor->length(dimension - 1);
    }

    int outside = tensor->elementSize() / width;

    if (type.code == halide_type_float) {
        auto data = tensor->host<float>();
        for (int z = 0; z < outside; ++z) {
            for (int x = 0; x < width; ++x) {
                outputOs << data[x + z * width] << "\t";
            }
            outputOs << "\n";
        }
    }
    if (type.code == halide_type_int && type.bytes() == 4) {
        auto data = tensor->host<int32_t>();
        for (int z = 0; z < outside; ++z) {
            for (int x = 0; x < width; ++x) {
                outputOs << data[x + z * width] << "\t";
            }
            outputOs << "\n";
        }
    }
    if (type.code == halide_type_uint && type.bytes() == 1) {
        auto data = tensor->host<uint8_t>();
        for (int z = 0; z < outside; ++z) {
            for (int x = 0; x < width; ++x) {
                outputOs << (int)data[x + z * width] << "\t";
            }
            outputOs << "\n";
        }
    }
}

static int test_main(int argc, const char* argv[]) {
    if (argc < 2) {
        MNN_PRINT("========================================================================\n");
        MNN_PRINT("Arguments: model.MNN numThread runTimes saveAllTensors forwardType\n");
        MNN_PRINT("========================================================================\n");
        return -1;
    }

    std::string cmd = argv[0];
    std::string pwd = "./";
    auto rslash     = cmd.rfind("/");
    if (rslash != std::string::npos) {
        pwd = cmd.substr(0, rslash + 1);
    }

    // read args
    const char* fileName = argv[1];

    int runTime = 1;
    if (argc > 2) {
        runTime = ::atoi(argv[2]);
    }

    int saveAllTensors = 0;
    if (argc > 3) {
        saveAllTensors = atoi(argv[3]);
        if (saveAllTensors) {
            MNN_PRINT("Save AllTensors to output/*.txt\n");
        }
    }

    int saveInput = 0;
    if (saveAllTensors > 1) {
        saveInput = 1;
    }

    auto type = MNN_FORWARD_CPU;
    if (argc > 4) {
        type = (MNNForwardType)atoi(argv[4]);
        MNN_PRINT("Use extra forward type: %d\n", type);
    }

    // input dims
    std::vector<int> inputDims;
    if (argc > 5) {
        std::string inputShape(argv[5]);
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

    int numThread = 4;
    if (argc > 6) {
        numThread = ::atoi(argv[6]);
    }

    auto revertor = std::unique_ptr<Revert>(new Revert(fileName));
    revertor->initialize();
    auto modelBuffer = revertor->getBuffer();
    auto bufferSize  = revertor->getBufferSize();

    // create net
    MNN_PRINT("Open Model %s\n", fileName);
    std::shared_ptr<MNN::Interpreter> net =
        std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromBuffer(modelBuffer, bufferSize));
    if (nullptr == net) {
        return 0;
    }
    revertor.reset();

    // create session
    MNN::ScheduleConfig config;
    config.type      = type;
    config.numThread = numThread;
    // If type not fount, let it failed
    config.backupType = type;
    BackendConfig backendConfig;
    // config.path.outputs.push_back("ResizeBilinear_2");
    // backendConfig.power = BackendConfig::Power_High;
    backendConfig.precision = BackendConfig::Precision_Low;
    // backendConfig.memory = BackendConfig::Memory_High;
    config.backendConfig     = &backendConfig;
    MNN::Session* session    = NULL;
    MNN::Tensor* inputTensor = nullptr;
    {
        AUTOTIME;
        session = net->createSession(config);
        if (nullptr == session) {
            return 0;
        }
        inputTensor = net->getSessionInput(session, NULL);
        if (!inputDims.empty()) {
            MNN_PRINT("===========> Resize Again...\n");
            net->resizeTensor(inputTensor, inputDims);
            net->resizeSession(session);
        }
    }
    auto allInput = net->getSessionInputAll(session);
    for (auto& iter : allInput) {
        auto size = iter.second->size();
        auto ptr  = iter.second->host<void>();
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
#ifdef FEED_INPUT_NAME_VALUE
    auto feedInput = [&net, session](const std::string input_name, int value) {
        auto inputTensor = net->getSessionInput(session, input_name.c_str());
        MNN::Tensor givenTensor(inputTensor, inputTensor->getDimensionType());
        auto value_type_code = givenTensor.getType().code;
        const int size       = givenTensor.elementSize();
        switch (value_type_code) {
            case halide_type_int: {
                if (4 == givenTensor.getType().bytes()) {
                    auto inputData = givenTensor.host<int32_t>();
                    for (int i = 0; i < size; ++i) {
                        inputData[i] = value;
                    }
                } else if (8 == givenTensor.getType().bytes()) {
                    auto inputData = givenTensor.host<int64_t>();
                    for (int i = 0; i < size; ++i) {
                        inputData[i] = static_cast<int64_t>(value);
                    }
                }

            } break;
            case halide_type_float: {
                auto inputData = givenTensor.host<float>();
                for (int i = 0; i < size; ++i) {
                    inputData[i] = static_cast<float>(value);
                }
            } break;
            default:
                MNN_ASSERT(false);
                break;
        }
        inputTensor->copyFromHostTensor(&givenTensor);
    };
#endif
    MNN_PRINT("===========> Session Resize Done.\n");
    MNN_PRINT("===========> Session Start running...\n");
    net->releaseModel();

    // input
    auto dimType = inputTensor->getDimensionType();
    if (inputTensor->getType().code == halide_type_uint || inputTensor->getType().code == halide_type_int) {
        dimType = Tensor::TENSORFLOW;
    }
    MNN::Tensor givenTensor(inputTensor, dimType);
    {
        int size_w = inputTensor->width();
        int size_h = inputTensor->height();
        int bpp    = inputTensor->channel();
        int batch  = inputTensor->batch();
        MNN_PRINT("Input: %d, %d, %d, %d\n", batch, size_h, size_w, bpp);

        std::ostringstream fileName;
        fileName << pwd << "input_0"
                 << ".txt";
        std::ifstream input(fileName.str().c_str());

        if (givenTensor.getType().code == halide_type_int) {
            auto size = givenTensor.elementSize();
            if (givenTensor.getType().bytes() == 4) {
                auto inputData = givenTensor.host<int32_t>();
                for (int i = 0; i < size; ++i) {
                    input >> inputData[i];
                }
            }
        } else if (givenTensor.getType().code == halide_type_uint) {
            auto size = givenTensor.elementSize();
            {
                FUNC_PRINT(givenTensor.getType().bytes());
                auto inputData = givenTensor.host<uint8_t>();
                for (int i = 0; i < size; ++i) {
                    int p;
                    input >> p;
                    inputData[i] = (uint8_t)p;
                }
            }
        } else if (givenTensor.getType().code == halide_type_float) {
            auto inputData = givenTensor.host<float>();
            auto size      = givenTensor.elementSize();
            for (int i = 0; i < size; ++i) {
                input >> inputData[i];
            }
        }
        inputTensor->copyFromHostTensor(&givenTensor);
    }

    if (saveAllTensors) {
        MNN::TensorCallBack beforeCallBack = [&](const std::vector<MNN::Tensor*>& ntensors, const std::string& opName) {
            if (!saveInput) {
                return true;
            }
            for (int i = 0; i < ntensors.size(); ++i) {
                auto ntensor      = ntensors[i];
                auto expectTensor = new MNN::Tensor(ntensor, MNN::Tensor::TENSORFLOW);
                ntensor->copyToHostTensor(expectTensor);

                auto tensor = ntensor;

                std::ostringstream outputFileName;
                auto opCopyName = opName;
                for (int j = 0; j < opCopyName.size(); ++j) {
                    if (opCopyName[j] == '/') {
                        opCopyName[j] = '_';
                    }
                }
                MNN_PRINT("Dump %s Input, %d, %d X %d X %d\n", opName.c_str(), i, tensor->width(), tensor->height(),
                          tensor->channel());
                outputFileName << "output/Input_" << opCopyName << "_" << i;
                dumpTensor2File(expectTensor, outputFileName.str().c_str());
                delete expectTensor;
            }
            return true;
        };
        MNN::TensorCallBack callBack = [&](const std::vector<MNN::Tensor*>& ntensors, const std::string& opName) {
            for (int i = 0; i < ntensors.size(); ++i) {
                auto ntensor      = ntensors[i];
                auto outDimType = ntensor->getDimensionType();
                if (inputTensor->getType().code == halide_type_uint || inputTensor->getType().code == halide_type_int) {
                    outDimType = Tensor::TENSORFLOW;
                }

                auto expectTensor = new MNN::Tensor(ntensor, outDimType);
                ntensor->copyToHostTensor(expectTensor);

                auto tensor = expectTensor;

                std::ostringstream outputFileName;
                auto opCopyName = opName;
                for (int j = 0; j < opCopyName.size(); ++j) {
                    if (opCopyName[j] == '/') {
                        opCopyName[j] = '_';
                    }
                }
                MNN_PRINT("W,H,C,B: %d X %d X %d X %d, %s : %d\n", tensor->width(), tensor->height(), tensor->channel(),
                          tensor->batch(), opName.c_str(), i);
                outputFileName << "output/" << opCopyName << "_" << i;
                dumpTensor2File(expectTensor, outputFileName.str().c_str());
                delete expectTensor;
            }
            return true;
        };
        net->runSessionWithCallBack(session, beforeCallBack, callBack);
    } else {
        net->runSession(session);
    }

    // save output
    auto outputTensor = net->getSessionOutput(session, NULL);
    if (outputTensor->size() <= 0) {
        MNN_ERROR("Output not available\n");
        return 0;
    }
    MNN::Tensor expectTensor(outputTensor, outputTensor->getDimensionType());
    outputTensor->copyToHostTensor(&expectTensor);
    auto outputFile = pwd + "output.txt";
    dumpTensor2File(&expectTensor, outputFile.c_str());

    // benchmark. for CPU, op time means calc duration; for others, op time means schedule duration.
    {
        int t = runTime;
        MNN_PRINT("Run %d time:\n", t);
        std::map<std::string, std::pair<float, float>> opTimes;
        uint64_t opBegin = 0;

        MNN::TensorCallBackWithInfo beforeCallBack = [&](const std::vector<MNN::Tensor*>& ntensors,
                                                         const OperatorInfo* info) {
            struct timeval now;
            gettimeofday(&now, NULL);
            opBegin = now.tv_sec * 1000000 + now.tv_usec;
            if (opTimes.find(info->name()) == opTimes.end()) {
                opTimes.insert(std::make_pair(info->name(), std::make_pair(0.0f, info->flops())));
            }
            return true;
        };
        MNN::TensorCallBackWithInfo afterCallBack = [&](const std::vector<MNN::Tensor*>& ntensors,
                                                        const OperatorInfo* info) {
            struct timeval now;
            gettimeofday(&now, NULL);
            auto opEnd = now.tv_sec * 1000000 + now.tv_usec;
            float cost = (float)(opEnd - opBegin) / 1000.0f;

            opTimes[info->name()].first += cost;
            return true;
        };

        if (t > 0) {
            struct timeval now;
            std::vector<float> times(t, 0.0f);
            for (int i = 0; i < t; ++i) {
                gettimeofday(&now, nullptr);
                auto begin = now.tv_sec * 1000000 + now.tv_usec;

                inputTensor->copyFromHostTensor(&givenTensor);
                net->runSessionWithCallBackInfo(session, beforeCallBack, afterCallBack, false);
                outputTensor->copyToHostTensor(&expectTensor);

                gettimeofday(&now, nullptr);
                auto end = now.tv_sec * 1000000 + now.tv_usec;
                times[i] = (end - begin) / 1000.0f;
            }

            auto minTime = std::min_element(times.begin(), times.end());
            auto maxTime = std::max_element(times.begin(), times.end());
            float sum    = 0.0f;
            for (auto time : times) {
                sum += time;
            }
            std::vector<std::pair<float, std::pair<std::string, float>>> allOpsTimes;
            float sumFlops = 0.0f;
            for (auto& iter : opTimes) {
                allOpsTimes.push_back(
                    std::make_pair(iter.second.first, std::make_pair(iter.first, iter.second.second)));
                sumFlops += iter.second.second;
            }

            std::sort(allOpsTimes.begin(), allOpsTimes.end());
            for (auto iter : allOpsTimes) {
                MNN_PRINT("%*s run %d average cost %f ms, %.3f %%, FlopsRate: %.3f %%\n", 50, iter.second.first.c_str(), runTime,
                          iter.first / (float)runTime, iter.first / sum * 100.0f, iter.second.second / sumFlops * 100.0f);
            }
            MNN_PRINT("Avg= %f ms, min= %f ms, max= %f ms\n", sum / (float)t, *minTime, *maxTime);
        }
    }
    return 0;
}

int main(int argc, const char* argv[]) {
    // For Detect Memory Leak, set circle as true
    bool circle = false;
    do {
        test_main(argc, argv);
    } while (circle);
    return 0;
}
