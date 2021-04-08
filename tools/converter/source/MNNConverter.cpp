//
//  MNNConverter.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "cli.hpp"

#include "MNN_generated.h"
#include "PostConverter.hpp"
#include "addBizCode.hpp"
#include "caffeConverter.hpp"
#include "liteConverter.hpp"
#include "onnxConverter.hpp"
#include "tensorflowConverter.hpp"
#include "torchscriptConverter.hpp"
#include "writeFb.hpp"
#include "common/Global.hpp"

int main(int argc, char *argv[]) {
    modelConfig modelPath;

    // parser command line arg
    try {
        Cli::initializeMNNConvertArgs(modelPath, argc, argv);
        Cli::printProjectBanner();

        Global<modelConfig>::Reset(&modelPath);

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
#ifdef MNN_BUILD_TORCHSCRIPT
        } else if (modelPath.model == modelConfig::TORCHSCRIPT) {
            torchscript2MNNNet(modelPath.modelFile, modelPath.bizCode, netT);
#endif
        } else {
            std::cout << "Not Support Model Type" << std::endl;
        }

        if (modelPath.model != modelConfig::MNN) {
            std::cout << "Start to Optimize the MNN Net..." << std::endl;
            std::unique_ptr<MNN::NetT> newNet = optimizeNet(netT, modelPath.forTraining);
            writeFb(newNet, modelPath.MNNModel, modelPath);
        } else {
            writeFb(netT, modelPath.MNNModel, modelPath);
        }
    } catch (const cxxopts::OptionException &e) {
        std::cerr << "Error while parsing options! " << std::endl;
        std::cerr << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }
    catch (const std::runtime_error &e) {
      std::cerr << "Error while converting the model! " << std::endl;
      std::cerr << e.what() << std::endl;
      exit(EXIT_FAILURE);
    }
    std::cout << "Converted Done!" << std::endl;

    return 0;
}
