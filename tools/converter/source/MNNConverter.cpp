//
//  MNNConverter.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "cli.hpp"

#include "MNN_generated.h"
#include "addBizCode.hpp"
#include "caffeConverter.hpp"
#include "liteConverter.hpp"
#include "onnxConverter.hpp"
#include "optimizer.hpp"
#include "tensorflowConverter.hpp"
#include "writeFb.hpp"

int main(int argc, char *argv[]) {
    modelConfig modelPath;

    // parser command line arg
    Cli::initializeMNNConvertArgs(modelPath, argc, argv);
    Cli::printProjectBanner();

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
    } else {
        std::cout << "Not Support Model Type" << std::endl;
    }

    if (modelPath.model != modelConfig::MNN) {
        std::cout << "Start to Optimize the MNN Net..." << std::endl;
        std::unique_ptr<MNN::NetT> newNet = optimizeNet(netT);
        writeFb(newNet, modelPath.MNNModel, modelPath.benchmarkModel);
    } else {
        writeFb(netT, modelPath.MNNModel, modelPath.benchmarkModel);
    }

    std::cout << "Converted Done!" << std::endl;

    return 0;
}
