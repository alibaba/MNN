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
#include "config.hpp"
#include "logkit.h"
#include <MNN/MNNDefine.h>

/**
 *  Print Command Line Banner
 */
void Cli::printProjectBanner() {
    // print project detail
    // auto config = ProjectConfig::obtainSingletonInstance();

    std::cout << "\nMNNConverter Version: " << ProjectConfig::version << " - MNN @ 2018\n\n" << std::endl;
}

cxxopts::Options Cli::initializeMNNConvertArgs(modelConfig &modelPath, int argc, char **argv) {
    cxxopts::Options options("MNNConvert");

    options.positional_help("[optional args]").show_positional_help();

    options.allow_unrecognised_options().add_options()("h, help", "Convert Other Model Format To MNN Model\n")(
        "v, version", "show current version")("f, framework",
#ifdef MNN_BUILD_TORCHSCRIPT
        "model type, ex: [TF,CAFFE,ONNX,TFLITE,MNN,TS]",
#else
        "model type, ex: [TF,CAFFE,ONNX,TFLITE,MNN]",
#endif
                                              cxxopts::value<std::string>())(
        "modelFile", "tensorflow Pb or caffeModel, ex: *.pb,*caffemodel", cxxopts::value<std::string>())(
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
        "saveStaticModel", "save static model with fix shape, default: false", cxxopts::value<bool>())(
        "targetVersion", "compability for old mnn engine, default: 1.2f", cxxopts::value<float>())(
        "inputConfigFile", "set input config file for static model, ex: ~/config.txt", cxxopts::value<std::string>());

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help({""}) << std::endl;
        exit(EXIT_SUCCESS);
    }

    if (result.count("version")) {
        std::cout << ProjectConfig::version << std::endl;
        exit(EXIT_SUCCESS);
    }

    modelPath.model = modelPath.MAX_SOURCE;
    // model source
    if (result.count("framework")) {
        const std::string frameWork = result["framework"].as<std::string>();
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
#ifdef MNN_BUILD_TORCHSCRIPT
        } else if ("TS" == frameWork) {
            modelPath.model = modelConfig::TORCHSCRIPT;
#endif
        } else {
            std::cout << "Framework Input ERROR or Not Support This Model Type Now!" << std::endl;
            std::cout << options.help({""}) << std::endl;
            exit(EXIT_FAILURE);
        }
    } else {
        std::cout << options.help({""}) << std::endl;
        exit(EXIT_FAILURE);
    }

    // model file path
    if (result.count("modelFile")) {
        const std::string modelFile = result["modelFile"].as<std::string>();
        if (CommonKit::FileIsExist(modelFile)) {
            modelPath.modelFile = modelFile;
        } else {
            DLOG(INFO) << "Model File Does Not Exist! ==> " << modelFile;
            exit(EXIT_FAILURE);
        }
    } else {
        std::cout << options.help({""}) << std::endl;
        exit(EXIT_FAILURE);
    }

    // prototxt file path
    if (result.count("prototxt")) {
        const std::string prototxt = result["prototxt"].as<std::string>();
        if (CommonKit::FileIsExist(prototxt)) {
            modelPath.prototxtFile = prototxt;
        } else {
            DLOG(INFO) << "Model File Does Not Exist!";
            exit(EXIT_FAILURE);
        }
    } else {
        // caffe model must have this option
        if (modelPath.model == modelPath.CAFFE) {
            std::cout << options.help({""}) << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    // MNN model output path
    if (result.count("MNNModel")) {
        const std::string MNNModelPath = result["MNNModel"].as<std::string>();
        modelPath.MNNModel             = MNNModelPath;
    } else {
        std::cout << options.help({""}) << std::endl;
        exit(EXIT_FAILURE);
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
        std::cout << options.help({""}) << std::endl;
        exit(EXIT_FAILURE);
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

    return options;
}

bool CommonKit::FileIsExist(string path) {
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
