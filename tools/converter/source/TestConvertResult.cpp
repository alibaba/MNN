//
//  TestConvertResult.cpp
//  MNNConverter
//
//  Created by MNN on 2020/01/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <sstream>
#include "cli.hpp"
int main(int argc, char *argv[]) {
    if (argc < 3) {
        MNN_ERROR("Usage: ./TestConvertResult [Onnx, Tf, Tflite, Torch] ${Dir} [config.json]\n");
        return 0;
    }
    std::string inputType = argv[1];
    std::string directName = argv[2];
    std::string configFile;
    if (argc >= 4) {
        configFile = argv[3];
    }
    auto inputModel = modelConfig::ONNX;
    auto suffix = ".onnx";
    if (inputType == "Tf") {
        inputModel = modelConfig::TENSORFLOW;
        suffix = ".pb";
    } else if (inputType == "Tflite") {
        inputModel = modelConfig::TFLITE;
        suffix = ".tflite";
    } else if (inputType == "Torch") {
        inputModel = modelConfig::TORCH;
        suffix = ".pt";
    }
    MNN_PRINT("Test %s\n", directName.c_str());
    std::string defaultCacheFile = "convert_cache.mnn";
    {
        modelConfig modelPath;
        modelPath.model = inputModel;
        std::ostringstream modelNameOs;
        modelNameOs << directName << "/test" << suffix;
        modelPath.modelFile = modelNameOs.str();
        modelPath.MNNModel = defaultCacheFile;
        modelPath.keepInputFormat = true;
        modelPath.saveExternalData = true;
        MNN::Cli::convertModel(modelPath);
    }
    return MNN::Cli::testconvert(defaultCacheFile, directName, 0.01f, configFile);
}
