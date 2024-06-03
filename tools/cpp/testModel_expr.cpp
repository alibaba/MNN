//
//  testModel_expr.cpp
//  MNN
//
//  Created by MNN on 2021/08/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#define MNN_OPEN_TIME_TRACE

#include <MNN/MNNDefine.h>
#include <math.h>
#include <algorithm>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <MNN/AutoTime.hpp>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <fstream>
#include <map>
#include <iostream>
#include <sstream>
#include "ExprDebug.hpp"
#define NONE "\e[0m"
#define RED "\e[0;31m"
#define GREEN "\e[0;32m"
#define L_GREEN "\e[1;32m"
#define BLUE "\e[0;34m"
#define L_BLUE "\e[1;34m"
#define BOLD "\e[1m"

void log_result(bool correct) {
    if (correct) {
#if defined(_MSC_VER)
        std::cout << "Correct!" << std::endl;
#else
        std::cout << GREEN << BOLD << "Correct!" << NONE << std::endl;
#endif
    }
}

template<typename T>
inline T stringConvert(const char* number) {
    std::istringstream os(number);
    T v;
    os >> v;
    return v;
}

template <typename T>
static bool compareImpl(MNN::Express::VARP x, MNN::Express::VARP y, int size, double tolerance) {
#define _ABS(a) ((a) < 0 ? -(a) : (a))
#define _MAX(a, b) ((a) > (b) ? (a) : (b))
    auto px = x->readMap<T>();
    auto py = y->readMap<T>();
    // get max if using overall torelance
    T maxValue = _ABS(py[0]);
    for (int i = 1; i < size; i++) {
        maxValue = _MAX(maxValue, _ABS(py[i]));
    }
    // compare
    for (int i = 0; i < size; i++) {
        T vx = px[i], vy = py[i];
        if (_ABS(vx - vy) < tolerance * maxValue) {
            continue;
        }
        std::cout << i << ": " << vx << " != " << vy << std::endl;
        return false;
    }
    return true;
#undef _ABS
#undef _MAX
}

static bool compareVar(MNN::Express::VARP x, MNN::Express::VARP y, double tolerance) {
    auto info = y->getInfo();
    auto dtype = info->type;
    auto size = info->size;
    if (dtype == halide_type_of<int32_t>()) {
        return compareImpl<int32_t>(x, y, size, tolerance);
    }
    if (dtype == halide_type_of<uint8_t>()) {
        return compareImpl<uint8_t>(x, y, size, tolerance);
    }
    return compareImpl<float>(x, y, size, tolerance);
}

using namespace MNN::Express;
int main(int argc, const char* argv[]) {
    if (argc < 4) {
        MNN_PRINT("Usage: ./testModel_expr.out model.mnn input.mnn output.mnn [type] [tolerance] [precision]\n");
        return 0;
    }
    MNN::ScheduleConfig sdConfig;
    auto rtMgr = std::shared_ptr<MNN::Express::Executor::RuntimeManager>(MNN::Express::Executor::RuntimeManager::createRuntimeManager(sdConfig), MNN::Express::Executor::RuntimeManager::destroy);
//#define TEST_DEBUG
#ifdef TEST_DEBUG
    _initTensorStatic();
    //_initDebug();
    rtMgr->setMode(MNN::Interpreter::Session_Debug);
#endif
    // check given & expect
    const char* modelPath  = argv[1];
    const char* inputName  = argv[2];
    const char* outputName = argv[3];
    MNN_PRINT("Testing model %s, input: %s, output: %s\n", modelPath, inputName, outputName);
    
    // create net
    auto type = MNN_FORWARD_CPU;
    if (argc > 4) {
        type = (MNNForwardType)stringConvert<int>(argv[4]);
    }
    auto tolerance = 0.1f;
    if (argc > 5) {
        tolerance = stringConvert<float>(argv[5]);
    }
    MNN::BackendConfig::PrecisionMode precision = MNN::BackendConfig::Precision_High;
    if (argc > 6) {
        precision = (MNN::BackendConfig::PrecisionMode)stringConvert<int>(argv[6]);
    }
    auto inputVars = Variable::load(inputName);
    auto outputVars = Variable::load(outputName);
    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;
    for (auto v : inputVars) {
        inputNames.emplace_back(v->name());
    }
    for (auto v : outputVars) {
        outputNames.emplace_back(v->name());
    }
    if (inputVars.empty()) {
        MNN_ERROR("Input is Error\n");
        return 0;
    }
    if (outputVars.empty()) {
        MNN_ERROR("Output is Error\n");
        return 0;
    }
    Module::Config config;
    config.rearrange = true;
    std::shared_ptr<Module> m(Module::load(inputNames, outputNames, modelPath, rtMgr, &config), [](void* net) {
        MNN::Express::Module::destroy((MNN::Express::Module*)net);
    });
    if (nullptr == m) {
        MNN_ERROR("Model is Error\n");
        return 0;
    }
    // First
    auto outputs = m->onForward(inputVars);
    if (outputs.size() != outputVars.size()) {
        MNN_ERROR("Number not match\n");
        return 0;
    }
    bool success = true;
    for (int i=0; i<outputVars.size(); ++i) {
        success = compareVar(outputs[i], outputVars[i], tolerance);
        if (!success) {
            MNN_ERROR("Error for %s\n", outputVars[i]->name().c_str());
            break;
        }
    }
    if (!success) {
        return 0;
    }
    outputs = m->onForward(inputVars);
    for (int i=0; i<outputVars.size(); ++i) {
        success = compareVar(outputs[i], outputVars[i], tolerance);
        if (!success) {
            MNN_ERROR("Error for %s\n", outputVars[i]->name().c_str());
            break;
        }
    }
    if (!success) {
        MNN_ERROR("Error for test second\n");
        return 0;
    }
    log_result(success);
    return 0;
}
