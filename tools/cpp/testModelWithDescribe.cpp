//
//  testModelWithDescribe.cpp
//  MNN
//
//  Created by MNN on 2019/01/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#define MNN_OPEN_TIME_TRACE

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <MNN/AutoTime.hpp>
#include "core/Backend.hpp"
#include "ConfigFile.hpp"
#include <MNN/Interpreter.hpp>
#include <MNN/MNNDefine.h>
#include "core/Macro.h"
#include <MNN/Tensor.hpp>
#include "core/TensorUtils.hpp"
#include <MNN/expr/Module.hpp>
#include <MNN/expr/MathOp.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>

#define NONE "\e[0m"
#define RED "\e[0;31m"
#define GREEN "\e[0;32m"
#define L_GREEN "\e[1;32m"
#define BLUE "\e[0;34m"
#define L_BLUE "\e[1;34m"
#define BOLD "\e[1m"

using namespace MNN::Express;

template<typename T>
inline T stringConvert(const char* number) {
    std::istringstream os(number);
    T v;
    os >> v;
    return v;
}

int loadData(const std::string name, void* ptr, int size, halide_type_t dtype) {
    std::ifstream stream(name.c_str());
    if (stream.fail()) {
        return -1;
    }
    switch (dtype.code) {
        case halide_type_float: {
            auto data = static_cast<float*>(ptr);
            for (int i = 0; i < size; ++i) {
                double temp = 0.0f;
                stream >> temp;
                data[i] = temp;
            }
        } break;
        case halide_type_int: {
            MNN_ASSERT(dtype.bits == 32);
            auto data = static_cast<int32_t*>(ptr);
            for (int i = 0; i < size; ++i) {
                int temp = 0;
                stream >> temp;
                data[i] = temp;
            }
        } break;
        case halide_type_uint: {
            MNN_ASSERT(dtype.bits == 8);
            auto data = static_cast<uint8_t*>(ptr);
            for (int i = 0; i < size; ++i) {
                int temp = 0;
                stream >> temp;
                data[i] = temp;
            }
        } break;
        default: {
            stream.close();
            return -1;
        }
    }

    stream.close();
    return 0;
}

MNN::Tensor* createTensor(const MNN::Tensor* shape, const std::string name) {
    auto result = new MNN::Tensor(shape, shape->getDimensionType());
    result->buffer().type = shape->buffer().type;
    if (!loadData(name, result->host<void>(), result->elementSize(), result->getType())) {
        return result;
    }
    delete result;
    return NULL;
}

VARP createVar(const std::string name, INTS shape, halide_type_t dtype) {
    int size = 1;
    for (int dim : shape) {
        size *= dim;
    }
    std::unique_ptr<char[]> data(new char[size * dtype.bytes()]);
    loadData(name, data.get(), size, dtype);
    return _Const(data.get(), shape, NHWC, dtype);
}

template <typename T>
bool compareVar(VARP var, std::string name) {
    auto targetValue = createVar(name, var->getInfo()->dim, var->getInfo()->type);
    auto absMax = _ReduceMax(_Abs(targetValue), {});
    absMax = _Maximum(absMax, _Scalar<T>(0));
    auto diff = _Abs(targetValue - var);
    auto diffAbsMax = _ReduceMax(diff);
    auto absMaxV = absMax->readMap<T>()[0];
    auto diffAbsMaxV = diffAbsMax->readMap<T>()[0];
    // The implemention of isnan in VS2017 isn't accept integer type, so cast all type to double
#ifdef _MSC_VER
#define ALI_ISNAN(x) std::isnan(static_cast<long double>(x))
#else
#define ALI_ISNAN(x) std::isnan(x)
#endif
    if (absMaxV * 0.01f < diffAbsMaxV || ALI_ISNAN(absMaxV)) {
        std::cout << "TESTERROR " << name << " value error : absMaxV:" << absMaxV << " - DiffMax:" << diffAbsMaxV << std::endl;
        return false;
    }
    return true;
}

void log_result(bool correct) {
    if (correct) {
#if defined(_MSC_VER)
        std::cout << "Correct!" << std::endl;
#else
        std::cout << GREEN << BOLD << "Correct!" << NONE << std::endl;
#endif
    }
}

int main(int argc, const char* argv[]) {
    // modelName is xxx/xxx/temp.bin ===> xxx/xxx is the root path
    const char* modelName = argv[1];
    std::string modelDir  = argv[2];
    modelDir              = modelDir.substr(0, modelDir.find("config.txt"));
    std::cout << "model dir: " << modelDir << std::endl;

    // read args
    auto type = MNN_FORWARD_CPU;
    if (argc > 3) {
        type = (MNNForwardType)stringConvert<int>(argv[3]);
    }
    auto tolerance = 0.1f;
    if (argc > 4) {
        tolerance = stringConvert<float>(argv[4]);
    }
    auto precision = MNN::BackendConfig::Precision_High;
    if (argc > 5) {
        precision = (MNN::BackendConfig::PrecisionMode)(stringConvert<int>(argv[5]));
    }

    // input config
    ConfigFile config(argv[2]);
    auto numOfInputs = config.Read<int>("input_size");
    auto numOfOuputs = config.Read<int>("output_size");
    auto inputNames  = splitNames(numOfInputs, config.Read<std::string>("input_names"));
    auto inputDims   = splitDims(numOfInputs, config.Read<std::string>("input_dims"));
    auto expectNames = splitNames(numOfOuputs, config.Read<std::string>("output_names"));
    bool controlFlow = config.KeyExists("control_flow") && config.Read<bool>("control_flow");
    auto dataType = halide_type_of<float>();
    if (config.KeyExists("data_type")) {
        auto dtype = config.Read<std::string>("data_type");
        if (dtype == "float") {
            dataType = halide_type_of<float>();
        } else if (dtype == "int32_t") {
            dataType = halide_type_of<int32_t>();
        } else if (dtype == "uint8_t") {
            dataType = halide_type_of<uint8_t>();
        }
    }
    // create net & session
#if defined(_MSC_VER)
    MNN_PRINT("Testing Model ====> %s\n", modelName);
#else
    MNN_PRINT(GREEN "Testing Model ====> %s\n" NONE, modelName);
#endif
    if (controlFlow) {
        std::shared_ptr<Module> model(Module::load(inputNames, expectNames, modelName));
        std::vector<VARP> inputs;
        for (int i = 0; i < numOfInputs; i++) {
            auto inputName = modelDir + inputNames[i] + ".txt";
            inputs.push_back(createVar(inputName, inputDims[i], dataType));
        }
        auto outputs = model->onForward(inputs);
        bool correct = true;
        for (int i = 0; i < numOfOuputs; i++) {
            auto dtype = outputs[i]->getInfo()->type;
            auto outputName = modelDir + expectNames[i] + ".txt";
            if (dtype == halide_type_of<int32_t>()) {
                correct = compareVar<int32_t>(outputs[i], outputName);
            } else if (dtype == halide_type_of<uint8_t>()) {
                correct = compareVar<uint8_t>(outputs[i], outputName);
            } else {
                correct = compareVar<float>(outputs[i], outputName);
            }
            if (!correct) {
                break;
            }
        }
        log_result(correct);
    } else {
        auto net = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(modelName));
        MNN::ScheduleConfig schedule;
        schedule.type = type;
        MNN::BackendConfig backendConfig;
        backendConfig.precision = precision;

        schedule.backendConfig = &backendConfig;

        auto session  = net->createSession(schedule);

        // resize
        for (int i = 0; i < numOfInputs; ++i) {
            auto inputTensor = net->getSessionInput(session, inputNames[i].c_str());
            net->resizeTensor(inputTensor, inputDims[i]);
        }
        net->resizeSession(session);
        auto checkFunction = [&]() {
            // [second] set input-tensor data
            for (int i = 0; i < numOfInputs; ++i) {
                auto inputTensor = net->getSessionInput(session, inputNames[i].c_str());
                auto inputName   = modelDir + inputNames[i] + ".txt";
                std::cout << "The " << i << " input: " << inputName << std::endl;

                auto givenTensor = createTensor(inputTensor, inputName);
                if (!givenTensor) {
#if defined(_MSC_VER)
                    std::cout << "Failed to open " << inputName << std::endl;
#else
                    std::cout << RED << "Failed to open " << inputName << NONE << std::endl;
#endif
                    break;
                }
                inputTensor->copyFromHostTensor(givenTensor);
                delete givenTensor;
            }

            // inference
            net->runSession(session);

            // get ouput-tensor and compare data
            bool correct = true;
            for (int i = 0; i < numOfOuputs; ++i) {
                auto outputTensor = net->getSessionOutput(session, expectNames[i].c_str());
                MNN::Tensor* expectTensor = nullptr;
                std::string expectName;
                // First Check outputname.txt
                {
                    std::ostringstream iStrOs;
                    iStrOs << expectNames[i];
                    expectName   = modelDir + iStrOs.str() + ".txt";
                    expectTensor = createTensor(outputTensor, expectName);
                }
                if (!expectTensor) {
                    // Second check number outputs
                    std::ostringstream iStrOs;
                    iStrOs << i;
                    expectName   = modelDir + iStrOs.str() + ".txt";
                    expectTensor = createTensor(outputTensor, expectName);
                }
                if (!expectTensor) {
#if defined(_MSC_VER)
                    std::cout << "Failed to open " << expectName << std::endl;
#else
                    std::cout << RED << "Failed to open " << expectName << NONE << std::endl;
#endif
                    break;
                }
                if (!MNN::TensorUtils::compareTensors(outputTensor, expectTensor, tolerance, true)) {
                    correct = false;
                    break;
                }
                delete expectTensor;
            }
            return correct;
        };
        auto correct = checkFunction();
        if (!correct) {
            return 0;
        } else {
            std::cout << "First Time Pass"<<std::endl;
        }
        // Second time
        correct =  checkFunction();
        log_result(correct);
    }
    return 0;
}
