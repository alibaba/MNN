//
//  cli.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "cli.hpp"
#if defined(_MSC_VER)
#include <Windows.h>
#undef min
#undef max
#else
#include <unistd.h>
#endif
#include "OpCount.hpp"
#include "cxxopts.hpp"
#include "config.hpp"
#include "logkit.h"
#include <MNN/MNNDefine.h>
#include "MNN_generated.h"
#include "PostConverter.hpp"
#include "addBizCode.hpp"
#include "caffeConverter.hpp"
#include "liteConverter.hpp"
#include "onnxConverter.hpp"
#include "tensorflowConverter.hpp"
#include "torchConverter.hpp"
#include "writeFb.hpp"
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "PostConverter.hpp"
#include "rapidjson/document.h"
#include <fstream>
#include <sstream>
#include <cmath>
#include "common/MemoryFormater.h"
namespace MNN {
static float gMNNVersion = 1.2f;

bool Cli::initializeMNNConvertArgs(modelConfig &modelPath, int argc, char **argv) {
    cxxopts::Options options("MNNConvert");

    options.positional_help("[optional args]").show_positional_help();

    options.allow_unrecognised_options().add_options()(std::make_pair("h", "help"), "Convert Other Model Format To MNN Model\n")(
                                                                                                                                 std::make_pair("v", "version"), "show current version")
        (std::make_pair("f", "framework"),
#ifdef MNN_BUILD_TORCH
        "model type, ex: [TF,CAFFE,ONNX,TFLITE,MNN,TORCH]",
#else
        "model type, ex: [TF,CAFFE,ONNX,TFLITE,MNN]",
#endif
            cxxopts::value<std::string>())
        (
            "modelFile",
            "tensorflow Pb or caffeModel, ex: *.pb,*caffemodel",
            cxxopts::value<std::string>()
        )
        (
            "batch",
            "if model input's batch is not set, set as the batch size you set",
            cxxopts::value<int>()
        )
        (
            "keepInputFormat",
            "keep input dimension format or not, default: false",
            cxxopts::value<bool>()
        )
        (
            "optimizeLevel",
            "graph optimize option, 0: don't run optimize(only support for MNN source), 1: use graph optimize only for every input case is right, 2: normally right but some case may be wrong, default 1",
            cxxopts::value<int>()
        )
        (
            "optimizePrefer",
            "graph optimize option, 0 for normal, 1 for smalleset, 2 for fastest",
            cxxopts::value<int>()
            )
        (
            "prototxt",
            "only used for caffe, ex: *.prototxt",
            cxxopts::value<std::string>())
        (
            "MNNModel",
            "MNN model, ex: *.mnn",
            cxxopts::value<std::string>())
        (
            "fp16",
            "save Conv's weight/bias in half_float data type")
        (
            "benchmarkModel",
            "Do NOT save big size data, such as Conv's weight,BN's gamma,beta,mean and variance etc. Only used to test the cost of the model")
        (
            "bizCode",
            "MNN Model Flag, ex: MNN",
            cxxopts::value<std::string>())
        (
            "debug",
            "Enable debugging mode."
        )
        (
            "forTraining",
            "whether or not to save training ops BN and Dropout, default: false",
            cxxopts::value<bool>()
        )
        (
            "weightQuantBits",
            "save conv/matmul/LSTM float weights to int8 type, only optimize for model size, 2-8 bits, default: 0, which means no weight quant",
            cxxopts::value<int>()
        )
        (
            "weightQuantAsymmetric",
            "the default weight-quant uses SYMMETRIC quant method, which is compatible with old MNN versions. "
                                "you can try set --weightQuantAsymmetric to use asymmetric quant method to improve accuracy of the weight-quant model in some cases, "
                                "but asymmetric quant model cannot run on old MNN versions. You will need to upgrade MNN to new version to solve this problem. default: false",
            cxxopts::value<bool>()
        )
        (
            "compressionParamsFile",
            "The path of the compression parameters that stores activation, "
            "weight scales and zero points for quantization or information "
            "for sparsity.",
            cxxopts::value<std::string>()
            )
        (
            "OP",
             "print framework supported op",
             cxxopts::value<bool>()
        )
        (
            "saveStaticModel",
             "save static model with fix shape, default: false",
             cxxopts::value<bool>()
        )
        (
            "targetVersion",
             "compability for old mnn engine, default: 1.2f",
             cxxopts::value<float>()
        )
        (
            "customOpLibs",
             "custom op libs ex: libmy_add.so;libmy_sub.so",
             cxxopts::value<std::string>()
        )
        (
            "authCode",
             "code for model authentication.",
             cxxopts::value<std::string>()
        )
        (
            "inputConfigFile",
             "set input config file for static model, ex: ~/config.txt",
             cxxopts::value<std::string>()
        )
        (
            "testdir",
            "set test dir, mnn will convert model and then check the result",
            cxxopts::value<std::string>()
        )
        (
            "thredhold",
            "if set test dir, thredhold mean the max rate permit for run MNN model and origin error",
            cxxopts::value<float>()
        )
        (
            "alignDenormalizedValue",
             "if 1, converter would align denormalized float(|x| < 1.18e-38) as zero, because of in ubuntu/protobuf or android/flatbuf, system behaviors are different. default: 1, range: {0, 1}",
             cxxopts::value<int>()
        );

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help({""}) << std::endl;
        return false;
    }

    if (result.count("version")) {
        std::cout << gMNNVersion << std::endl;
        return false;
    }

    modelPath.model = modelPath.MAX_SOURCE;
    // model source
    std::string frameWork;
    if (result.count("framework")) {
        frameWork = result["framework"].as<std::string>();
        if ("TF" == frameWork) {
            modelPath.model = modelConfig::TENSORFLOW;
        } else if ("CAFFE" == frameWork) {
            modelPath.model = modelConfig::CAFFE;
        } else if ("ONNX" == frameWork) {
            modelPath.model = modelConfig::ONNX;
        } else if ("MNN" == frameWork) {
            modelPath.model = modelConfig::MNN;
        } else if ("TFLITE" == frameWork) {
            modelPath.model = modelConfig::TFLITE;
#ifdef MNN_BUILD_TORCH
        } else if ("TORCH" == frameWork) {
            modelPath.model = modelConfig::TORCH;
#endif
        } else {
            std::cout << "Framework Input ERROR or Not Support This Model Type Now!" << std::endl;
            return false;
        }
    } else {
        std::cout << options.help({""}) << std::endl;
        DLOG(INFO) << "framework Invalid, use -f CAFFE/MNN/ONNX/TFLITE/TORCH !";
        return false;
    }
    if (result.count("OP")) {
        MNN_PRINT("Dump %s support Ops\n", frameWork.c_str());
        const auto& res = OpCount::get()->getMap().find(frameWork);
        if (res == OpCount::get()->getMap().end()) {
            return false;
        }
        for (const auto& iter : res->second) {
            MNN_PRINT("%s\n", iter.c_str());
        }
        MNN_PRINT("Total: %d\n", (int)res->second.size());
        return false;
    }

    // model file path
    if (result.count("modelFile")) {
        const std::string modelFile = result["modelFile"].as<std::string>();
        if (CommonKit::FileIsExist(modelFile)) {
            modelPath.modelFile = modelFile;
        } else {
            DLOG(INFO) << "Model File Does Not Exist! ==> " << modelFile;
            return false;
        }
    } else {
        DLOG(INFO) << "modelFile Not set Invalid, use --modelFile to set!";
        return false;
    }
    // Optimize Level
    if (result.count("optimizeLevel")) {
        modelPath.optimizeLevel = result["optimizeLevel"].as<int>();
        if (modelPath.optimizeLevel > 1) {
            DLOG(INFO) << "\n optimizeLevel > 1, some case may be wrong";
        }
    }

    // prototxt file path
    if (result.count("prototxt")) {
        const std::string prototxt = result["prototxt"].as<std::string>();
        if (CommonKit::FileIsExist(prototxt)) {
            modelPath.prototxtFile = prototxt;
        } else {
            DLOG(INFO) << "Proto File Does Not Exist!";
            return false;
        }
    } else {
        // caffe model must have this option
        if (modelPath.model == modelPath.CAFFE) {
            DLOG(INFO) << "Proto File Not Set, use --prototxt XXX.prototxt to set it!";
            return false;
        }
    }

    // MNN model output path
    if (result.count("MNNModel")) {
        const std::string MNNModelPath = result["MNNModel"].as<std::string>();
        modelPath.MNNModel             = MNNModelPath;
    } else {
        DLOG(INFO) << "MNNModel File Not Set, use --MNNModel XXX.prototxt to set it!";
        return false;
    }
    if (result.count("targetVersion")) {
        auto version = result["targetVersion"].as<float>();
        std::cout << "TargetVersion is " << version << std::endl;
        modelPath.targetVersion = version;
    }
    // add MNN bizCode
    if (result.count("bizCode")) {
        const std::string bizCode = result["bizCode"].as<std::string>();
        modelPath.bizCode         = bizCode;
    } else {
        MNN_ERROR("Don't has bizCode, use MNNTest for default\n");
        modelPath.bizCode = "MNNTest";
    }

    // input config file path
    if (result.count("inputConfigFile")) {
        const std::string inputConfigFile = result["inputConfigFile"].as<std::string>();
        modelPath.inputConfigFile         = inputConfigFile;
    }

    // benchmarkModel
    if (result.count("benchmarkModel")) {
        modelPath.benchmarkModel = true;
        modelPath.bizCode        = "benchmark";
    }
    // half float
    if (result.count("fp16")) {
        modelPath.saveHalfFloat = true;
    }
    if (result.count("forTraining")) {
        modelPath.forTraining = true;
    }
    if (result.count("batch")) {
        modelPath.defaultBatchSize = result["batch"].as<int>();
    }
    if (result.count("keepInputFormat")) {
        modelPath.keepInputFormat = true;
    }
    if (result.count("weightQuantBits")) {
        modelPath.weightQuantBits = result["weightQuantBits"].as<int>();
    }
    if (result.count("weightQuantAsymmetric")) {
        modelPath.weightQuantAsymmetric = true;
    }
    if (result.count("saveStaticModel")) {
        modelPath.saveStaticModel = true;
    }
    if (result.count("optimizePrefer")) {
        modelPath.optimizePrefer = result["optimizePrefer"].as<int>();
    }
    // Int8 calibration table path.
    if (result.count("compressionParamsFile")) {
        modelPath.compressionParamsFile =
            result["compressionParamsFile"].as<std::string>();
    }
    if (result.count("customOpLibs")) {
        modelPath.customOpLibs = result["customOpLibs"].as<std::string>();
    }
    if (result.count("authCode")) {
        modelPath.authCode = result["authCode"].as<std::string>();
    }
    if (result.count("alignDenormalizedValue")) {
        modelPath.alignDenormalizedValue = result["alignDenormalizedValue"].as<int>();
    }
    if (result.count("testdir")) {
        modelPath.testDir = result["testdir"].as<std::string>();
    }
    if (result.count("thredhold")) {
        modelPath.testThredhold = result["thredhold"].as<float>();
    }

    return true;
}

bool Cli::convertModel(modelConfig& modelPath) {
    std::cout << "Start to Convert Other Model Format To MNN Model..." << std::endl;
    std::unique_ptr<MNN::NetT> netT = std::unique_ptr<MNN::NetT>(new MNN::NetT());
    if (modelPath.model == modelConfig::CAFFE) {
        caffe2MNNNet(modelPath.prototxtFile, modelPath.modelFile, modelPath.bizCode, netT);
    } else if (modelPath.model == modelConfig::TENSORFLOW) {
        tensorflow2MNNNet(modelPath.modelFile, modelPath.bizCode, netT);
    } else if (modelPath.model == modelConfig::MNN) {
        addBizCode(modelPath.modelFile, modelPath.bizCode, netT);
    } else if (modelPath.model == modelConfig::ONNX) {
        onnx2MNNNet(modelPath.modelFile, modelPath.bizCode, netT);
    } else if (modelPath.model == modelConfig::TFLITE) {
        tflite2MNNNet(modelPath.modelFile, modelPath.bizCode, netT);
#ifdef MNN_BUILD_TORCH
    } else if (modelPath.model == modelConfig::TORCH) {
        torch2MNNNet(modelPath.modelFile, modelPath.bizCode, netT, modelPath.customOpLibs);
#endif
    } else {
        std::cout << "Not Support Model Type" << std::endl;
    }
    if (netT.get() == nullptr) {
        MNN_ERROR("Convert error\n");
        return false;
    }
    int error = 0;
    if (modelPath.defaultBatchSize > 0) {
        for (const auto& op : netT->oplists) {
            if (op->type != OpType_Input || nullptr == op->main.AsInput()) {
                continue;
            }
            auto inputP = op->main.AsInput();
            if (inputP->dims.size() >= 1 && inputP->dims[0] <= 0) {
                std::cout << "Set " << op->name << " batch = " << modelPath.defaultBatchSize << std::endl;
                inputP->dims[0] = modelPath.defaultBatchSize;
            }
        }
    }
    if (modelPath.model != modelConfig::MNN || modelPath.optimizeLevel >= 2) {
        std::cout << "Start to Optimize the MNN Net..." << std::endl;
        std::unique_ptr<MNN::NetT> newNet = optimizeNet(netT, modelPath.forTraining, modelPath);
        error = writeFb(newNet, modelPath.MNNModel, modelPath);
    } else {
        error = writeFb(netT, modelPath.MNNModel, modelPath);
    }
    if (0 == error) {
        std::cout << "Converted Success!" << std::endl;
    } else {
        std::cout << "Converted Failed!" << std::endl;
    }
    if (modelPath.testDir.size() > 0) {
        std::cout << "Check convert result by " << modelPath.testDir << ", thredhold is " << modelPath.testThredhold << std::endl;
        Cli::testconvert(modelPath.MNNModel, modelPath.testDir, modelPath.testThredhold);
    }
    return true;
}

static bool compareOutput(MNN::Express::VARP output, const std::string& directName, const std::string& name, MNN::Express::Dimensionformat dataFormat, int order, float maxError) {
    auto info = output->getInfo();
    auto ptr = output->readMap<float>();
    if (info && info->size <= 0) {
        MNN_PRINT("skip checking value for zero content tensor %s\n", name.c_str());
        return true;
    }

    if (nullptr == info || nullptr == ptr) {
        MNN_ERROR("TESTERROR name:%s, info:%p, ptr:%p.\n", name.c_str(), info, ptr);
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
    if (info->order == MNN::Express::NC4HW4 && info->dim.size() > 1) {
        output = _Convert(output, dataFormat);
        info = output->getInfo();
    }
    if (info->type.code != halide_type_float) {
        output = MNN::Express::_Cast<float>(output);
        info = output->getInfo();
    }
    MNN_PRINT("%s: (", name.c_str());
    for (int i=0; i<info->dim.size(); ++i) {
        MNN_PRINT("%d, ", info->dim[i]);
    }
    MNN_PRINT(")\n");
    auto targetValue = _Input({info->dim}, info->order, info->type);
    auto targetPtr = targetValue->writeMap<float>();
    for (int i=0; i<info->size; ++i) {
        outputOrigin >> targetPtr[i];
    }

    auto absMax = MNN::Express::_ReduceMax(MNN::Express::_Abs(targetValue), {});
    absMax = MNN::Express::_Maximum(absMax, MNN::Express::_Scalar<float>(0.0001f));
    auto diff = MNN::Express::_Abs(targetValue - output);
    auto outputPtr = output->readMap<float>();
    auto diffAbsMax = MNN::Express::_ReduceMax(diff);
    auto absMaxV = absMax->readMap<float>()[0];
    auto diffAbsMaxV = diffAbsMax->readMap<float>()[0];
    if (absMaxV * maxError < diffAbsMaxV || std::isnan(absMaxV)) {
        MNN_ERROR("TESTERROR %s value error : absMaxV:%f - DiffMax %f\n", name.c_str(), absMaxV, diffAbsMaxV);

        MNN_PRINT("expected value\n");
        formatMatrix(targetValue->readMap<float>(), targetValue->getInfo()->dim);
        MNN_PRINT("real value\n");
        formatMatrix(output->readMap<float>(), output->getInfo()->dim);

        return false;
    }
    return true;
}

int Cli::testconvert(const std::string& defaultCacheFile, const std::string& directName, float maxErrorRate) {
    rapidjson::Document document;
    std::map<std::string, float> inputInfo;
    std::map<std::string, std::vector<int>> inputShape;
    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;
    {
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
            auto array = document["outputs"].GetArray();
            for (auto iter = array.begin(); iter !=array.end(); iter++) {
                std::string name = iter->GetString();
                MNN_PRINT("output: %s\n", name.c_str());
                outputNames.emplace_back(name);
            }
        }
    }

    // create session
    MNN::ScheduleConfig config;
    config.type      = MNN_FORWARD_CPU;
    /*modeNum means gpuMode for GPU usage, Or means numThread for CPU usage.*/
    config.numThread = 2;
    // If type not fount, let it failed
    config.backupType = MNN_FORWARD_CPU;
    BackendConfig backendConfig;
    // config.path.outputs.push_back("ResizeBilinear_2");
    // backendConfig.power = BackendConfig::Power_High;
    backendConfig.precision = static_cast<MNN::BackendConfig::PrecisionMode>(1);
    // backendConfig.memory = BackendConfig::Memory_High;
    config.backendConfig     = &backendConfig;

    MNN::Express::Module::Config mConfig;
    mConfig.shapeMutable = true;
    std::shared_ptr<MNN::Express::Executor::RuntimeManager> rtmgr(MNN::Express::Executor::RuntimeManager::createRuntimeManager(config));
    std::shared_ptr<MNN::Express::Module> net(MNN::Express::Module::load(inputNames, outputNames, defaultCacheFile.c_str(), rtmgr, &mConfig));
    auto mInfo = net->getInfo();
    std::vector<MNN::Express::VARP> inputs(mInfo->inputs.size());
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
            inputOs >> ptr[i];\
        }\
    }
    // Load inputs
    for (int i=0; i<inputs.size(); ++i) {
        auto inputName = inputNames[i];
        // Resize
        auto shapeIter = inputShape.find(inputName);
        if (shapeIter != inputShape.end()) {
            auto s = shapeIter->second;
            inputs[i] = _Input(s, mInfo->defaultFormat, mInfo->inputs[i].type);
        } else {
            inputs[i] = _Input(mInfo->inputs[i].dim, mInfo->inputs[i].order, mInfo->inputs[i].type);
        }
        auto info = inputs[i]->getInfo();
        auto iter = inputInfo.find(inputNames[i]);
        if (iter != inputInfo.end()) {
            auto ptr = inputs[i]->writeMap<float>();
            for (int v=0; v<mInfo->inputs[i].size; ++v) {
                ptr[v] = iter->second;
            }
            continue;
        }
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
#undef LOAD_DATA
    bool modelError = false;
    // Module Branch
    auto outputs = net->onForward(inputs);
    for (int i=0; i<outputNames.size(); ++i) {
        auto name = outputNames[i];
        auto v = outputs[i];
        auto info = v->getInfo();
        if (nullptr == info) {
            continue;
        }
        if (info->order == MNN::Express::NC4HW4 && info->dim.size() > 1) {
            v = MNN::Express::_Convert(v, mInfo->defaultFormat);
        }
        if (info->type.code != halide_type_float) {
            v = MNN::Express::_Cast<float>(v);
        }
        v.fix(MNN::Express::VARP::CONSTANT);
        outputs[i] = v;
    }

    for (int i=0; i<outputNames.size(); ++i) {
        auto output = outputs[i];
        bool success = compareOutput(output, directName, outputNames[i], mInfo->defaultFormat, i, maxErrorRate);
        if (!success) {
            modelError = true;
            MNN_ERROR("Error for output %s\n", outputNames[i].c_str());
        }
    }

    if (modelError) {
        MNN_ERROR("Save mnn result to  .error director\n");
        for (int i=0; i<outputNames.size(); ++i) {
            auto v = outputs[i];
            auto name = outputNames[i];
            auto info = v->getInfo();
            if (nullptr == info) {
                continue;
            }
            if (info->order == MNN::Express::NC4HW4 && info->dim.size() > 1) {
                v = MNN::Express::_Convert(v, mInfo->defaultFormat);
            }
            if (info->type.code != halide_type_float) {
                v = MNN::Express::_Cast<float>(v);
            }
            v.fix(MNN::Express::VARP::CONSTANT);
            info = v->getInfo();
            std::ofstream _output((".error/" + name + ".txt").c_str());
            auto ptr = v->readMap<float>();
            for (int v=0; v<info->size; ++v) {
                _output << ptr[v] << "\n";
            }
            v->setName(name);
            outputs.emplace_back(v);
        }
        MNN::Express::Variable::save(outputs, ".Error.mnn");
        return 0;
    }
    MNN_PRINT("TEST_SUCCESS\n");
    return 0;
}


};


bool CommonKit::FileIsExist(std::string path) {
#if defined(_MSC_VER)
    if (INVALID_FILE_ATTRIBUTES != GetFileAttributes(path.c_str()) && GetLastError() != ERROR_FILE_NOT_FOUND) {
        return true;
    }
#else
    if ((access(path.c_str(), F_OK)) != -1) {
        return true;
    }
#endif
    return false;
}

