//
//  cli.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "cli.hpp"
#include <unistd.h>
#include "config.hpp"
#include "logkit.h"

/**
 *  Print Command Line Banner
 */
void Cli::printProjectBanner() {
    // print project detail
    auto config = ProjectConfig::obtainSingletonInstance();

    std::cout << "\nMNNConverter Version: " << ProjectConfig::version << " - MNN @ 2018\n\n" << std::endl;
}

cxxopts::Options Cli::initializeMNNConvertArgs(modelConfig &modelPath, int argc, char **argv) {
    cxxopts::Options options("MNNConvert");

    options.positional_help("[optional args]").show_positional_help();

    try {
        options.allow_unrecognised_options().add_options()("h, help", "Convert Other Model Format To MNN Model\n")(
            "v, version", "show current version")("f, framework", "model type, ex: [TF,CAFFE,ONNX,TFLITE,MNN]",
                                                  cxxopts::value<std::string>())(
            "modelFile", "tensorflow Pb or caffeModel, ex: *.pb,*caffemodel", cxxopts::value<std::string>())(
            "prototxt", "only used for caffe, ex: *.prototxt", cxxopts::value<std::string>())(
            "MNNModel", "MNN model, ex: *.mnn", cxxopts::value<std::string>())(
            "benchmarkModel",
            "Do NOT save big size data, such as Conv's weight,BN's gamma,beta,mean and variance etc. Only used to test "
            "the cost of the model")
            ("bizCode", "MNN Model Flag, ex: MNN", cxxopts::value<std::string>())(
            "debug", "Enable debugging mode.");

        auto result = options.parse(argc, argv);

        if (result.count("help")) {
            std::cout << options.help({""}) << std::endl;
            exit(EXIT_SUCCESS);
        }

        if (result.count("version")) {
            std::cout << PROJECT_VERSION << std::endl;
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

        // add MNN bizCode
        if (result.count("bizCode")) {
            const std::string bizCode = result["bizCode"].as<std::string>();
            modelPath.bizCode         = bizCode;
        } else {
            std::cout << options.help({""}) << std::endl;
            exit(EXIT_FAILURE);
        }

        // benchmarkModel
        if (result.count("benchmarkModel")) {
            modelPath.benchmarkModel = true;
            modelPath.bizCode        = "benchmark";
        }
    } catch (const cxxopts::OptionException &e) {
        std::cerr << "Error while parsing options! " << std::endl;
        std::cerr << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    return options;
}

bool CommonKit::FileIsExist(string path) {
    if ((access(path.c_str(), F_OK)) != -1) {
        return true;
    }
    return false;
}
