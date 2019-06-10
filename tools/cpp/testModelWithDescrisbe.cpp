//
//  testModelWithDescrisbe.cpp
//  MNN
//
//  Created by MNN on 2019/01/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#define MNN_OPEN_TIME_TRACE

#include <math.h>
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include "AutoTime.hpp"
#include "Backend.hpp"
#include "Config.hpp"
#include "Interpreter.hpp"
#include "MNNDefine.h"
#include "Macro.h"
#include "Tensor.hpp"
#include "TensorUtils.hpp"

#define NONE "\e[0m"
#define RED "\e[0;31m"
#define GREEN "\e[0;32m"
#define L_GREEN "\e[1;32m"
#define BLUE "\e[0;34m"
#define L_BLUE "\e[1;34m"
#define BOLD "\e[1m"

std::vector<std::string> splitNames(const int size, const std::string names) {
    std::vector<std::string> split(size);
    std::string rest = names;
    for (int i = 0; i < size; ++i) {
        auto position = rest.find(",");
        split[i]      = rest.substr(0, position);
        rest          = rest.substr(position + 1, rest.size());
    }
    return split;
}

std::vector<int> stringToDims(const std::string& str, const std::string& pattern) {
    std::vector<int> res;
    if (str == "") {
        return res;
    }
    std::string strs = str + pattern;
    size_t pos       = strs.find(pattern);

    while (pos != strs.npos) {
        std::string temp = strs.substr(0, pos);
        std::istringstream tempIs(temp);
        int p;
        tempIs >> p;
        res.push_back(p);
        strs = strs.substr(pos + 1, strs.size());
        pos  = strs.find(pattern);
    }

    return res;
}

std::vector<std::vector<int>> splitDims(const int size, const std::string string) {
    std::vector<std::vector<int>> dims(size);
    std::string rest = string;
    for (int i = 0; i < size; ++i) {
        auto position = rest.find(",");
        dims[i]       = stringToDims(rest.substr(0, position), "x");
        rest          = rest.substr(position + 1, rest.size());
    }
    return dims;
}

MNN::Tensor* createTensor(const MNN::Tensor* shape, const std::string path) {
    std::ifstream stream(path.c_str());
    if (stream.fail()) {
        return NULL;
    }

    auto result           = new MNN::Tensor(shape, shape->getDimensionType());
    result->buffer().type = shape->buffer().type;
    switch (result->getType().code) {
        case halide_type_float: {
            auto data = result->host<float>();
            for (int i = 0; i < result->elementSize(); ++i) {
                double temp = 0.0f;
                stream >> temp;
                data[i] = temp;
            }
        } break;
        case halide_type_int: {
            MNN_ASSERT(result->getType().bits == 32);
            auto data = result->host<int>();
            for (int i = 0; i < result->elementSize(); ++i) {
                int temp = 0;
                stream >> temp;
                data[i] = temp;
            }
        } break;
        case halide_type_uint: {
            MNN_ASSERT(result->getType().bits == 8);
            auto data = result->host<uint8_t>();
            for (int i = 0; i < result->elementSize(); ++i) {
                int temp = 0;
                stream >> temp;
                data[i] = temp;
            }
        } break;
        default: {
            stream.close();
            return NULL;
        }
    }

    stream.close();
    return result;
}

int main(int argc, const char* argv[]) {
    // modelName is xxx/xxx/temp.bin ===> xxx/xxx is the root path
    const char* modelName = argv[1];
    std::string modelDir  = modelName;
    modelDir              = modelDir.substr(0, modelDir.find("temp.bin"));
    std::cout << "model dir: " << modelDir << std::endl;

    // read args
    auto type = MNN_FORWARD_CPU;
    if (argc > 3) {
        type = (MNNForwardType)::atoi(argv[3]);
    }
    auto tolerance = 0.1f;
    if (argc > 4) {
        tolerance = ::atof(argv[4]);
    }

    // input config
    Config config(argv[2]);
    auto numOfInputs = config.Read<int>("input_size");
    auto numOfOuputs = config.Read<int>("output_size");
    auto inputNames  = splitNames(numOfInputs, config.Read<std::string>("input_names"));
    auto inputDims   = splitDims(numOfInputs, config.Read<std::string>("input_dims"));
    auto expectNames = splitNames(numOfOuputs, config.Read<std::string>("output_names"));

    // create net & session
#if defined(_MSC_VER)
    MNN_PRINT("Testing Model ====> %s\n", modelName);
#else
    MNN_PRINT(GREEN "Testing Model ====> %s\n" NONE, modelName);
#endif
    auto net = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(modelName));
    MNN::ScheduleConfig schedule;
    schedule.type = type;
    auto session  = net->createSession(schedule);

    // resize
    for (int i = 0; i < numOfInputs; ++i) {
        auto inputTensor = net->getSessionInput(session, inputNames[i].c_str());
        net->resizeTensor(inputTensor, inputDims[i]);
    }
    net->resizeSession(session);

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
        for (int j = 0; j < inputDims.size(); j++) {
            MNN_ASSERT(inputDims[i][j] == givenTensor->length(j));
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
        std::ostringstream iStrOs;
        iStrOs << i;
        auto expectName   = modelDir + iStrOs.str() + ".txt";
        auto expectTensor = createTensor(outputTensor, expectName);
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

    if (correct) {
#if defined(_MSC_VER)
        std::cout << "Correct!" << std::endl;
#else
        std::cout << GREEN << BOLD << "Correct!" << NONE << std::endl;
#endif
    }

    return 0;
}
