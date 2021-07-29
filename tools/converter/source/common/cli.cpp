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
                                              cxxopts::value<std::string>())(
        "modelFile", "tensorflow Pb or caffeModel, ex: *.pb,*caffemodel", cxxopts::value<std::string>())(
        "batch", "if model input's batch is not set, set as the batch size you set", cxxopts::value<int>())(
        "keepInputFormat", "keep input dimension format or not, default: false", cxxopts::value<bool>())(
        "optimizeLevel", "graph optimize option, 1: use graph optimize only for every input case is right, 2: normally right but some case may be wrong, default 1", cxxopts::value<int>())(
        "optimizePrefer", "graph optimize option, 0 for normal, 1 for smalleset, 2 for fastest", cxxopts::value<int>())(
        "prototxt", "only used for caffe, ex: *.prototxt", cxxopts::value<std::string>())(
        "MNNModel", "MNN model, ex: *.mnn", cxxopts::value<std::string>())(
        "fp16", "save Conv's weight/bias in half_float data type")(
        "benchmarkModel",
        "Do NOT save big size data, such as Conv's weight,BN's gamma,beta,mean and variance etc. Only used to test "
        "the cost of the model")("bizCode", "MNN Model Flag, ex: MNN", cxxopts::value<std::string>())(
        "debug", "Enable debugging mode.")(
        "forTraining", "whether or not to save training ops BN and Dropout, default: false", cxxopts::value<bool>())(
        "weightQuantBits", "save conv/matmul/LSTM float weights to int8 type, only optimize for model size, 2-8 bits, default: 0, which means no weight quant", cxxopts::value<int>())(
        "weightQuantAsymmetric", "the default weight-quant uses SYMMETRIC quant method, which is compatible with old MNN versions. "
                                "you can try set --weightQuantAsymmetric to use asymmetric quant method to improve accuracy of the weight-quant model in some cases, "
                                "but asymmetric quant model cannot run on old MNN versions. You will need to upgrade MNN to new version to solve this problem. default: false", cxxopts::value<bool>())(
        "compressionParamsFile",
            "The path of the compression parameters that stores activation, "
            "weight scales and zero points for quantization or information "
            "for sparsity.", cxxopts::value<std::string>())(
        "OP", "print framework supported op", cxxopts::value<bool>())(
        "saveStaticModel", "save static model with fix shape, default: false", cxxopts::value<bool>())(
        "targetVersion", "compability for old mnn engine, default: 1.2f", cxxopts::value<float>())(
        "inputConfigFile", "set input config file for static model, ex: ~/config.txt", cxxopts::value<std::string>());

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
        modelPath.keepInputFormat = result["keepInputFormat"].as<bool>();
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
        torch2MNNNet(modelPath.modelFile, modelPath.bizCode, netT);
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
    if (modelPath.model != modelConfig::MNN) {
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
    return true;
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

