//
//  ModuleBasic.cpp
//  MNN
//
//  Created by MNN on 2021/10/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNN_generated.h"
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/ExprCreator.hpp>
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include "rapidjson/document.h"
#include "core/MemoryFormater.h"
#include <numeric>
#include <chrono>
#include <iostream>
#include <thread>
#include "ExprDebug.hpp"

using namespace MNN::Express;
using namespace MNN;

static bool compareOutput(VARP output, const std::string& directName, const std::string& name, Dimensionformat dataFormat, int order) {

    auto info = output->getInfo();
    auto ptr = output->readMap<float>();
    if (info && info->size <= 0) {
        MNN_PRINT("skip checking value for zero content tensor %s\n", name.c_str());
        return true;
    }

    if (nullptr == info || nullptr == ptr) {
        MNN_ERROR("TESTERROR name:%s, info:%p, ptr:%p. size:%d\n", name.c_str(), info, ptr, info->size);
        return false;
    }

    std::ifstream outputOrigin;
    // First find key
    {
        std::ostringstream outputFileOs;
        outputFileOs << directName << "/" << name <<".txt";
        outputOrigin.open(outputFileOs.str().c_str());
    }
    // Second find order
    if (outputOrigin.fail()) {
        std::ostringstream outputFileOs;
        outputFileOs << directName << "/" << order <<".txt";
        outputOrigin.open(outputFileOs.str().c_str());
    }
    if (outputOrigin.fail()) {
        MNN_PRINT("Skip check %s\n", name.c_str());
        return true;
    }
    MNN_PRINT("before compare %s: (", name.c_str());
    for (int i=0; i<info->dim.size(); ++i) {
        MNN_PRINT("%d, ", info->dim[i]);
    }
    MNN_PRINT(")\n");
    auto targetValue = _Input(info->dim, info->order, info->type);
    auto targetPtr = targetValue->writeMap<float>();
    auto outputPtr = output->readMap<float>();
#define MNN_IS_INF(x) (fabs(x) == INFINITY)
#define MNN_IS_NAN(x) ((x) != (x))
    for (int i=0; i<info->size; ++i) {
        double targetValue;
        outputOrigin >> targetValue;
        targetPtr[i] = targetValue;
    }

    for (int i=0; i<info->size; ++i) {
        if (MNN_IS_INF(outputPtr[i]) || MNN_IS_NAN(outputPtr[i])) {
            MNN_ERROR("TESTERROR %s value error:%f\n", name.c_str(), outputPtr[i]);
            return false;
        }
    }
    auto absMax = _ReduceMax(_Abs(targetValue), {});
    absMax = _Maximum(absMax, _Scalar<float>(0.0001f));
    auto diff = _Abs(targetValue - output);
    auto diffAbsMax = _ReduceMax(diff);
    auto absMaxV = absMax->readMap<float>()[0];
    auto diffAbsMaxV = diffAbsMax->readMap<float>()[0];
    if (absMaxV * 0.01f < diffAbsMaxV || MNN_IS_NAN(absMaxV)) {
        MNN_ERROR("TESTERROR %s value error : absMaxV:%f - DiffMax %f\n", name.c_str(), absMaxV, diffAbsMaxV);
        return false;
    }
    return true;
}
int main(int argc, char *argv[]) {
    if (argc < 3) {
        MNN_ERROR("Usage: ./ModuleBasic.out ${test.mnn} ${Dir} [runMask] [forwardType] [runLoops] [numberThread] [precision | memory] [cacheFile]\n");
        return 0;
    }
    std::string modelName = argv[1];
    std::string directName = argv[2];
    MNN_PRINT("Test %s from input info: %s\n", modelName.c_str(), directName.c_str());
    std::map<std::string, float> inputInfo;
    std::map<std::string, std::vector<int>> inputShape;
    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;
    bool checkOutput = false;
    int runMask = 0;
    if (argc > 3) {
        runMask = atoi(argv[3]);
        if (runMask & 1) {
            _initDebug();
        }
        if (runMask & 2) {
            _initTensorStatic();
        }
    }
    int repeatNumber = 1;
    bool shapeMutable = true;
    std::vector<VARP> inputs;
    std::vector<VARP> outputs;
    if (runMask & 128) {
        MNN_PRINT("Use input.mnn and output.mnn for test\n");
        inputs = MNN::Express::Variable::load((directName + "/input.mnn").c_str());
        outputs = MNN::Express::Variable::load((directName + "/output.mnn").c_str());
        if (inputs.size() > 0 && outputs.size() > 0) {
            MNN_PRINT("Has input.mnn, use input.mnn and output.mnn instead of json\n");
        }
        for (auto v : inputs) {
            inputNames.emplace_back(v->name());
        }
        for (auto v : outputs) {
            outputNames.emplace_back(v->name());
        }
        checkOutput = outputs.size() > 0;
    }
    // Call Time / Per Second
    float freq = 0.0f;
    int cpuDecreaseRate = -1;
    if (inputNames.empty()) {
        rapidjson::Document document;
        std::ostringstream jsonNameOs;
        jsonNameOs << directName << "/input.json";
        std::ifstream fileNames(jsonNameOs.str().c_str());
        std::ostringstream output;
        output << fileNames.rdbuf();
        auto outputStr = output.str();
        document.Parse(outputStr.c_str());
        if (document.HasParseError()) {
            MNN_ERROR("Invalid json\n");
            return 0;
        }
        if (document.HasMember("inputs")) {
            auto inputsInfo = document["inputs"].GetArray();
            for (auto iter = inputsInfo.begin(); iter !=inputsInfo.end(); iter++) {
                auto obj = iter->GetObject();
                std::string name = obj["name"].GetString();
                inputNames.emplace_back(name);
                MNN_PRINT("%s\n", name.c_str());
                if (obj.HasMember("value")) {
                    float value = obj["value"].GetFloat();
                    inputInfo.insert(std::make_pair(name, value));
                }
                if (obj.HasMember("shape")) {
                    auto dims = obj["shape"].GetArray();
                    std::vector<int> shapes;
                    for (auto iter = dims.begin(); iter != dims.end(); iter++) {
                        shapes.emplace_back(iter->GetInt());
                    }
                    inputShape.insert(std::make_pair(name, shapes));
                }
            }
        }
        if (document.HasMember("outputs")) {
            checkOutput = true;
            auto array = document["outputs"].GetArray();
            for (auto iter = array.begin(); iter !=array.end(); iter++) {
                std::string name = iter->GetString();
                MNN_PRINT("output: %s\n", name.c_str());
                outputNames.emplace_back(name);
            }
        }
        if (document.HasMember("shapeMutable")) {
            shapeMutable = document["shapeMutable"].GetBool();
        }
        if (document.HasMember("repeat")) {
            repeatNumber = document["repeat"].GetInt();
        }
        if (document.HasMember("freq")) {
            freq = document["freq"].GetFloat();
        }
        if (document.HasMember("cpu_decrease_rate")) {
            cpuDecreaseRate = document["cpu_decrease_rate"].GetInt();
        }
    }
    auto type = MNN_FORWARD_CPU;
    if (argc > 4) {
        type = (MNNForwardType)atoi(argv[4]);
        MNN_PRINT("Use extra forward type: %d\n", type);
    }

    // Default single thread
    int modeNum = 1;
    if (argc > 6) {
        modeNum = ::atoi(argv[6]);
    }

    int power = BackendConfig::Power_Normal;
    int precision = BackendConfig::Precision_Normal;
    int memory = BackendConfig::Memory_Normal;
    if (argc > 7) {
        int mask = atoi(argv[7]);
        precision = mask % 4;
        memory = (mask / 4) % 4;
        power = (mask / 16) % 4;
    }
    const char* cacheFileName = ".tempcache";
    if (argc > 8) {
        cacheFileName = argv[8];
    }
    FUNC_PRINT(precision);
    FUNC_PRINT(memory);
    FUNC_PRINT(power);
    FUNC_PRINT_ALL(cacheFileName, s);
    // create session
    MNN::ScheduleConfig config;
    config.type      = type;
    /*modeNum means gpuMode for GPU usage, Or means numThread for CPU usage.*/
    config.numThread = modeNum;
    // If type not fount, let it failed
    config.backupType = type;
    BackendConfig backendConfig;
    // config.path.outputs.push_back("ResizeBilinear_2");
    backendConfig.power = (BackendConfig::PowerMode)power;
    backendConfig.precision = static_cast<MNN::BackendConfig::PrecisionMode>(precision);
    backendConfig.memory = static_cast<MNN::BackendConfig::MemoryMode>(memory);
    config.backendConfig     = &backendConfig;

    MNN::Express::Module::Config mConfig;
    if (runMask & 256) {
        mConfig.dynamic = true;
    }
    mConfig.shapeMutable = shapeMutable;
    std::shared_ptr<Executor::RuntimeManager> rtmgr(Executor::RuntimeManager::createRuntimeManager(config));
    rtmgr->setCache(cacheFileName);
    if (cpuDecreaseRate > 0 && cpuDecreaseRate <= 100) {
        rtmgr->setHint(Interpreter::CPU_LITTLECORE_DECREASE_RATE, cpuDecreaseRate);
    }
    if (runMask & 1) {
        // Need dump tensor, open debug
        rtmgr->setMode(Interpreter::Session_Debug);
    }
    if (runMask & 2) {
        // Need tensor static for each op, open debug
        rtmgr->setMode(Interpreter::Session_Debug);
    }
    // For Debug
    if (false) {
        int geometryMask = Interpreter::GeometryComputeMask::GEOMETRCOMPUTEMASK_ALL;
        geometryMask -= Interpreter::GeometryComputeMask::GEOMETRCOMPUTEMASK_FUSEREGION;
        geometryMask -= Interpreter::GeometryComputeMask::GEOMETRCOMPUTEMASK_OPENCACHE;
        rtmgr->setHint(Interpreter::GEOMETRY_COMPUTE_MASK, geometryMask);
    }
    if (runMask & 4) {
        // Need time trace for each op, open debug
        rtmgr->setMode(Interpreter::Session_Debug);
    }
    if (runMask & 8) {
        rtmgr->setMode(Interpreter::Session_Input_Inside);
    }
    if (runMask & 16) {
        rtmgr->setMode(Interpreter::Session_Backend_Auto);
        rtmgr->setHint(Interpreter::MAX_TUNING_NUMBER, 50);
    }
    if (runMask & 32) {
        mConfig.rearrange = true;
    }
    if (runMask & 512) {
        rtmgr->setHint(Interpreter::WINOGRAD_MEMORY_LEVEL, 0);
    }
    if (runMask & 1024) {
        rtmgr->setHint(Interpreter::DYNAMIC_QUANT_OPTIONS, 1);
    }
    std::shared_ptr<Module> net;
    {
        AUTOTIME;
        net.reset(Module::load(inputNames, outputNames, modelName.c_str(), rtmgr, &mConfig));
        if (net == nullptr) {
            MNN_PRINT("Error: can't load module\n");
            return 0;
        }
        if (runMask & 64) {
            net.reset(Module::clone(net.get()));
        }
    }
    auto mInfo = net->getInfo();

#define LOAD_DATA(TYPE)\
    if (inputInfo.find(inputName) != inputInfo.end()) {\
        auto value = inputInfo[inputName];\
        for (int i=0; i<info->size; ++i) {\
            ptr[i] = value;\
        }\
    } else {\
        std::ostringstream fileNameOs;\
        fileNameOs << directName << "/" << inputName << ".txt";\
        auto fileName = fileNameOs.str();\
        std::ifstream inputOs(fileName.c_str());\
        if (inputOs.fail()) {\
            MNN_ERROR("TESTERROR Can't open %s\n", fileName.c_str());\
            continue;\
        }\
        for (int i=0; i<info->size; ++i) {\
            double tempValue;\
            inputOs >> tempValue;\
            ptr[i] = tempValue;\
        }\
    }

    if (inputs.empty()) {
        inputs.resize(mInfo->inputs.size());
        for (int i=0; i<inputs.size(); ++i) {
            inputs[i] = _Input(mInfo->inputs[i].dim, mInfo->inputs[i].order, mInfo->inputs[i].type);
        }
        // Load inputs
        for (int i=0; i<inputs.size(); ++i) {
            auto inputName = inputNames[i];
            // Resize
            auto shapeIter = inputShape.find(inputName);
            if (shapeIter != inputShape.end()) {
                auto s = shapeIter->second;
                inputs[i] = _Input(s, mInfo->defaultFormat, mInfo->inputs[i].type);
            }
            auto info = inputs[i]->getInfo();
            if (info->type == halide_type_of<float>()){
                auto ptr = inputs[i]->writeMap<float>();
                LOAD_DATA(float)
            } else {
                auto floatVar = _Input(info->dim, info->order, halide_type_of<float>());
                auto ptr = floatVar->writeMap<float>();
                LOAD_DATA(float)
                auto temp = _Cast(floatVar, info->type);
                inputs[i]->input(temp);
            }
            inputs[i] = _Convert(inputs[i], mInfo->inputs[i].order);
        }
    }
#undef LOAD_DATA

    bool modelError = false;
    for (int repeat = 0; repeat < repeatNumber; ++repeat) {
        MNN_PRINT("Run for %d time\n", repeat);
        std::vector<VARP> subInputs = inputs;
        if (repeat % 2 == 1) {
            for (int i=0; i<inputs.size(); ++i) {
                subInputs[i] = _Clone(inputs[i], true);
            }
        }
        auto outputs = net->onForward(inputs);
        if (outputs.empty()) {
            MNN_ERROR("Error in forward\n");
            return 0;
        }
        for (int i=0; i<outputNames.size(); ++i) {
            auto name = outputNames[i];
            auto v = outputs[i];
            auto info = v->getInfo();
            if (nullptr == info) {
                continue;
            }
            if (info->order == NC4HW4 && info->dim.size() > 1) {
                v = _Convert(v, mInfo->defaultFormat);
            }
            if (info->type.code != halide_type_float) {
                v = _Cast<float>(v);
            }
            v.fix(VARP::CONSTANT);
            outputs[i] = v;
        }

        if (checkOutput) {
            for (int i=0; i<outputNames.size(); ++i) {
                auto output = outputs[i];
                bool success = compareOutput(output, directName, outputNames[i], mInfo->defaultFormat, i);
                if (!success) {
                    modelError = true;
                    MNN_ERROR("%d run Error for output %s\n", repeat, outputNames[i].c_str());
                }
            }
        }
        for (int i=0; i<outputNames.size(); ++i) {
            auto name = outputNames[i];
            auto v = outputs[i];
            auto info = v->getInfo();
            std::ostringstream fileNameOs;
            fileNameOs << "output/" << repeat <<"_"<< i << ".txt";
            auto fileName = fileNameOs.str();
            MNN_PRINT("Write %s output to %s\n", name.c_str(), fileName.c_str());
            std::ofstream _output(fileName.c_str());
            auto ptr = v->readMap<float>();
            for (int v=0; v<info->size; ++v) {
                _output << ptr[v] << "\n";
            }
        }
        // Print module's memory
        float memoryInMB = 0.0f;
        rtmgr->getInfo(Interpreter::MEMORY, &memoryInMB);
        FUNC_PRINT_ALL(memoryInMB, f);
    }

    // benchmark. for CPU, op time means calc duration; for others, op time means schedule duration.
    int runTime = 0;
    if (argc > 5) {
        runTime = ::atoi(argv[5]);
    }

    if (runTime > 0) {
        int t = runTime;
        std::vector<float> times(t, 0.0f);
        if (runMask & 4) {
            _initTimeTrace();
        }
        for (int i = 0; i < t; ++i) {
            Timer _l;
            auto out = net->onForward(inputs);
            for (auto o : out) {
                ((MNN::Tensor*)o->getTensor())->wait(MNN::Tensor::MAP_TENSOR_READ, true);
            }
            times[i] = _l.durationInUs() / 1000.0f;
            if (freq > 0.0f) {
                float remainMs = (1000.0f / freq) - times[i];
                if (remainMs > 0.0f) {
                    std::this_thread::sleep_for(std::chrono::milliseconds((int)remainMs));
                }
            }
        }
        if (nullptr != gTimeTraceInfo) {
            float opSummer = 0.0f;
            float opFlopsSummber = 0.0f;
            for (auto& iter : gTimeTraceInfo->mTypes) {
                float summer = 0.0f;
                float summerflops = 0.0f;
                for (auto& t : iter.second) {
                    for (auto& t0 : t.second) {
                        summer += t0.first;
                        summerflops += t0.second;
                    }
                }
                summer = summer / (float)t;
                summerflops = summerflops / (float)t;
                MNN_PRINT("%s : %.7f, FLOP: %.7f, Speed: %.7f GFlops\n", iter.first.c_str(), summer, summerflops, summerflops / summer);
                opSummer += summer;
                opFlopsSummber+= summerflops;
            }
            MNN_PRINT("OP Summer: %.7f, Flops: %.7f, Speed: %.7f GFlops\n", opSummer, opFlopsSummber, opFlopsSummber/opSummer);
        }
        auto minTime = std::min_element(times.begin(), times.end());
        auto maxTime = std::max_element(times.begin(), times.end());
        float sum    = 0.0f;
        for (auto time : times) {
            sum += time;
        }
        MNN_PRINT("Avg= %f ms, min= %f ms, max= %f ms\n", sum / (float)t, *minTime, *maxTime);
    }
    rtmgr->updateCache();
    return 0;
}

